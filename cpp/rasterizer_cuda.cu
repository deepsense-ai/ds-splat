#include "rasterizer_kernels.cuh"
#include <cassert>

#include <cstdint>
#include <functional>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

void sort_by_keys(int32_t* indices, int64_t* keys, const int num_of_gaussians)
{
    auto device_indices = thrust::device_pointer_cast(indices);
    auto device_keys = thrust::device_pointer_cast(keys);

    thrust::sort_by_key(device_keys, device_keys + num_of_gaussians, device_indices);
}

thrust::device_vector<int32_t> identify_tile_ranges(const int64_t* keys, const int image_width, const int image_height,
                                                    const int num_of_gaussians)
{
    const int threads = 768;
    const int blocks = (num_of_gaussians + threads - 1) / threads;
    const int tiles_width = (image_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_height = (image_height + TILE_SIZE - 1) / TILE_SIZE;
    const int size = tiles_width * tiles_height + 1;

    thrust::device_vector<int32_t> indices(size, 0);
    indices[size - 1] = num_of_gaussians;

    auto indices_device_ptr = thrust::raw_pointer_cast(indices.data());

    identify_tile_ranges_kernel<<<blocks, threads>>>(keys, indices_device_ptr, num_of_gaussians);

    thrust::inclusive_scan(indices.begin(), indices.end(), indices.begin(), thrust::maximum{});

    return indices;
}

void rasterizer_cuda_forward(Gaussian* gaussian_tensor, const int32_t* indices, const int32_t* tile_indices,
                             int image_width, int image_height, float* output_image)
{
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    const dim3 grid_dim((image_width + block_dim.x - 1) / block_dim.x, (image_height + block_dim.y - 1) / block_dim.y);
    render_image_kernel<<<grid_dim, block_dim>>>(gaussian_tensor, indices, tile_indices, output_image, image_width,
                                                 image_height);
}

std::tuple<thrust::device_vector<int64_t>, thrust::device_vector<int32_t>>
duplicate_keys(int32_t* block_sums, Gaussian* gaussian_tensor, const int32_t* radii, int numBlocks_in_preprocess,
               int threads_in_preprocess, int num_of_gaussians, int image_width, int image_height)
{
    // reuse numBlocks and threads count for calculate_keys_and_indices_kernel, so we are sure
    // restoring global exclusive sum will work as intended

    auto block_sums_device_ptr = thrust::device_pointer_cast(block_sums);
    // calculate exclusive sum per each block
    thrust::exclusive_scan(block_sums_device_ptr, block_sums_device_ptr + numBlocks_in_preprocess,
                           block_sums_device_ptr);

    int32_t last_element = 0;
    if(numBlocks_in_preprocess > 0)
        cudaMemcpy(&last_element, block_sums + numBlocks_in_preprocess - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);

    const size_t offset_to_last_size_occupied_tiles =
        (num_of_gaussians - 1) * sizeof(Gaussian) + offsetof(Gaussian, size_occupied_tiles);
    int32_t total_size = 0;
    if(num_of_gaussians > 0)
        cudaMemcpy(&total_size, reinterpret_cast<uint8_t*>(gaussian_tensor) + offset_to_last_size_occupied_tiles,
                   sizeof(int32_t), cudaMemcpyDeviceToHost);
    total_size += (numBlocks_in_preprocess > 0) ? last_element : 0;

    thrust::device_vector<int64_t> keys(total_size);
    thrust::device_vector<int32_t> indices(total_size);
    calculate_keys_and_indices_kernel<<<numBlocks_in_preprocess, threads_in_preprocess>>>(
        gaussian_tensor, block_sums, radii, thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(indices.data()), num_of_gaussians, total_size, image_width, image_height);

    return std::make_tuple(keys, indices);
}

std::tuple<thrust::device_vector<int32_t>, thrust::device_vector<Gaussian>, thrust::device_vector<int32_t>, int, int>
rasterizer_forward_preprocessing(const float* means_3d, const float* shs, const float* colors_precomp,
                                 const float* opacities, const float* scales, const float* proj_matrix,
                                 const float* view_matrix, const float* camera_position, const float* rotations,
                                 const float* cov3D_precomp, int image_width, int image_height, float tan_fovx,
                                 float tan_fovy, const float scale_modifier, const int max_sh_degree,
                                 const int sh_degree, int num_of_gaussians)
{
    const int max_coeff = (max_sh_degree + 1) * (max_sh_degree + 1);
    const int coeff = (sh_degree + 1) * (sh_degree + 1);

    if(shs != nullptr)
    {
        assert(sh_degree >= 0 && sh_degree <= 4);
        // assert(shs.size(1) >= coeff); // todo pass proper size to the method
        assert(max_sh_degree == 3); // @TODO other max degrees not supported for now (they need separate cuda kenrels)
    }

    const int threads = 768;
    const int numBlocks = (num_of_gaussians + threads - 1) / threads;

    const int size_of_gaussian = sizeof(Gaussian);

    thrust::device_vector<int32_t> radii(num_of_gaussians);
    thrust::device_vector<Gaussian> gaussian_tensor(num_of_gaussians);
    Gaussian* gaussian_data_ptr = thrust::raw_pointer_cast(gaussian_tensor.data());

    const auto focal_y = image_height / (2 * tan_fovy);
    const auto focal_x = image_width / (2 * tan_fovx);
    const auto c_x = image_width / 2.0f;
    const auto c_y = image_height / 2.0f;

    // here we will keep all sizes sum per each block
    thrust::device_vector<int32_t> block_sums(numBlocks);

    rasterizer_preprocessing_kernel<<<numBlocks, threads, threads * sizeof(int)>>>(
        means_3d, shs, scales, rotations, proj_matrix, view_matrix, camera_position, opacities, cov3D_precomp,
        colors_precomp, scale_modifier, max_coeff, sh_degree, num_of_gaussians, focal_x, focal_y, tan_fovx, tan_fovy,
        c_x, c_y, image_width, image_height, thrust::raw_pointer_cast(radii.data()),
        thrust::raw_pointer_cast(block_sums.data()), gaussian_data_ptr);

    return std::make_tuple(radii, gaussian_tensor, block_sums, numBlocks, threads);
}

void rasterizer_forward_core_deepsense(const float* means_3d, const float* shs, const float* opacities,
                                       const float* scales, const float* rotations, int num_of_gaussians,
                                       const float* view_matrix, const float* proj_matrix, const float* camera_position,
                                       int image_width, int image_height, float tan_fovx, float tan_fovy,
                                       float scale_modifier, const int max_sh_degree, const int sh_degree,
                                       float* output_image)
{

    const float* colors_precomp = nullptr;
    const float* cov3D_precomp = nullptr;

    auto [radii, gaussian_tensor, block_sums, num_blocks_preprocessing, num_threads_preprocessing] =
        rasterizer_forward_preprocessing(means_3d, shs, colors_precomp, opacities, scales, proj_matrix, view_matrix,
                                         camera_position, rotations, cov3D_precomp, image_width, image_height, tan_fovx,
                                         tan_fovy, scale_modifier, max_sh_degree, sh_degree, num_of_gaussians);

    auto [keys, indices] =
        duplicate_keys(thrust::raw_pointer_cast(block_sums.data()), thrust::raw_pointer_cast(gaussian_tensor.data()),
                       thrust::raw_pointer_cast(radii.data()), num_blocks_preprocessing, num_threads_preprocessing,
                       num_of_gaussians, image_width, image_height);

    auto indices_ptr = thrust::raw_pointer_cast(indices.data());
    auto keys_ptr = thrust::raw_pointer_cast(keys.data());
    sort_by_keys(indices_ptr, keys_ptr, num_of_gaussians);

    auto tile_indices = identify_tile_ranges(keys_ptr, image_width, image_height, num_of_gaussians);
    auto tile_indices_raw_ptr = thrust::raw_pointer_cast(tile_indices.data());

    rasterizer_cuda_forward(thrust::raw_pointer_cast(gaussian_tensor.data()), indices_ptr, tile_indices_raw_ptr,
                            image_width, image_height, output_image);
}
