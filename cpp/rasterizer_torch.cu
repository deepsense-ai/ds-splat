#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <torch/cuda.h>

#include "ds_cuda_rasterizer/rasterizer_torch.hpp"
#include "gsplat/bindings.h"
#include "rasterizer_kernels.cuh"

#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_FLOAT_INPUT(x)                                                                                           \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_FLOAT32(x);

__global__ void tensors_from_gaussians_kernel(const Gaussian* __restrict__ gaussian_data_ptr, float2* __restrict__ xy,
                                              float3* __restrict__ colors, float3* __restrict__ conic,
                                              const int num_of_gaussians);

torch::Tensor identify_tile_ranges(torch::Tensor keys, const TorchRasterizationSettings& settings)
{
    const int threads = 768;
    const int blocks = (keys.sizes()[0] + threads - 1) / threads;
    const int tiles_width = (settings.image_width + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_height = (settings.image_height + TILE_SIZE - 1) / TILE_SIZE;
    const int size = tiles_width * tiles_height + 1;

    torch::Tensor indices = torch::zeros({size}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    indices[size - 1] = keys.numel();

    identify_tile_ranges_kernel<<<blocks, threads>>>(keys.data_ptr<int64_t>(), indices.data_ptr<int32_t>(),
                                                     keys.numel());

    thrust::device_ptr<int32_t> begin_it = thrust::device_pointer_cast<int32_t>(indices.data_ptr<int32_t>());
    auto end_it = begin_it + indices.numel();
    thrust::inclusive_scan(begin_it, end_it, begin_it, thrust::maximum{});

    return indices;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rasterizer_cuda_forward(torch::Tensor gaussian_tensor, torch::Tensor indices, torch::Tensor tile_indices,
                        const TorchRasterizationSettings& settings)
{
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((settings.image_width + block_dim.x - 1) / block_dim.x,
                  (settings.image_height + block_dim.y - 1) / block_dim.y);

    torch::Tensor output_image = torch::empty({settings.image_height, settings.image_width, 3},
                                              torch::device(torch::kCUDA).dtype(torch::kFloat32));

    torch::Tensor final_Ts = torch::empty({settings.image_height, settings.image_width}, output_image.options());

    torch::Tensor final_idx =
        torch::empty({settings.image_height, settings.image_width}, output_image.options().dtype(torch::kInt32));

    Gaussian* gaussian_data_ptr = reinterpret_cast<Gaussian*>(gaussian_tensor.data_ptr<uint8_t>());

    render_image_kernel<<<grid_dim, block_dim>>>(gaussian_data_ptr, indices.data_ptr<int32_t>(),
                                                 tile_indices.data_ptr<int32_t>(), output_image.data_ptr<float>(),
                                                 settings.image_width, settings.image_height,
                                                 final_Ts.data_ptr<float>(), final_idx.data_ptr<int32_t>());

    return std::make_tuple(output_image, final_Ts, final_idx);
}

void sort_by_keys(torch::Tensor indices, torch::Tensor keys)
{
    auto device_indices = thrust::device_pointer_cast(indices.data_ptr<int32_t>());
    auto device_keys = thrust::device_pointer_cast(keys.data_ptr<int64_t>());

    thrust::sort_by_key(device_keys, device_keys + keys.numel(), device_indices);
}

float fov_to_focal(float fov, float pixels)
{
    return pixels / (2.0f * tan(fov / 2.0f));
}

std::tuple<torch::Tensor, torch::Tensor> duplicate_keys(torch::Tensor block_sums, torch::Tensor gaussian_tensor,
                                                        torch::Tensor radii, int numBlocks_in_preprocess,
                                                        int threads_in_preprocess,
                                                        const TorchRasterizationSettings& settings)
{
    // reuse numBlocks and threads count for calculate_keys_and_indices_kernel, so we are sure
    // restoring global exclusive sum will work as intended
    const auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    int num_of_gaussians = radii.numel();

    auto block_sums_device_ptr = thrust::device_pointer_cast(block_sums.data_ptr<int32_t>());

    thrust::exclusive_scan(block_sums_device_ptr, block_sums_device_ptr + block_sums.numel(), block_sums_device_ptr);

    Gaussian* gaussian_data_ptr = reinterpret_cast<Gaussian*>(gaussian_tensor.data_ptr<uint8_t>());
    const uint8_t* data_ptr = gaussian_tensor.data_ptr<uint8_t>();
    const size_t offset_to_last_size_occupied_tiles =
        (num_of_gaussians - 1) * sizeof(Gaussian) + offsetof(Gaussian, size_occupied_tiles);
    int32_t total_size = 0;
    if(num_of_gaussians > 0)
        cudaMemcpy(&total_size, data_ptr + offset_to_last_size_occupied_tiles, sizeof(int32_t), cudaMemcpyDeviceToHost);
    total_size += (block_sums.numel() > 0) ? block_sums[block_sums.numel() - 1].item<int32_t>() : 0;

    torch::Tensor keys = torch::empty({total_size}, options_int32.dtype(torch::kInt64));
    torch::Tensor indices = torch::empty({total_size}, options_int32.dtype(torch::kInt32));
    calculate_keys_and_indices_kernel<<<numBlocks_in_preprocess, threads_in_preprocess>>>(
        gaussian_data_ptr, block_sums.data_ptr<int32_t>(), radii.data_ptr<int32_t>(), keys.data_ptr<int64_t>(),
        indices.data_ptr<int>(), radii.sizes()[0], total_size, settings.image_width, settings.image_height);

    return std::make_tuple(keys, indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int>
rasterizer_forward_preprocessing(torch::Tensor means_3d, torch::Tensor shs, torch::Tensor colors_precomp,
                                 torch::Tensor opacities, torch::Tensor scales, torch::Tensor rotations,
                                 torch::Tensor cov3D_precomp, const TorchRasterizationSettings& settings)
{
    const float scale_modifier = float(settings.scale_modifier);
    const int max_coeff = (settings.max_sh_degree + 1) * (settings.max_sh_degree + 1);
    const int coeff = (settings.sh_degree + 1) * (settings.sh_degree + 1);

    if(shs.numel())
    {
        assert(settings.sh_degree >= 0 && settings.sh_degree <= 4);
        assert(shs.size(1) >= coeff);
        assert(settings.max_sh_degree == 3);
    }

    const int threads = 768;
    const int num_of_gaussians = means_3d.numel() / 3;
    const int numBlocks = (num_of_gaussians + threads - 1) / threads;

    const auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    const int size_of_gaussian = sizeof(Gaussian);

    const auto options_byte = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor radii = torch::empty({num_of_gaussians}, options_int32);
    torch::Tensor gaussian_tensor = torch::empty({num_of_gaussians * size_of_gaussian}, options_byte);

    Gaussian* gaussian_data_ptr = reinterpret_cast<Gaussian*>(gaussian_tensor.data_ptr<uint8_t>());

    const auto focal_y = settings.image_height / (2 * settings.tanfovy);
    const auto focal_x = settings.image_width / (2 * settings.tanfovx);
    const auto c_x = settings.image_width / 2.0f;
    const auto c_y = settings.image_height / 2.0f;

    // here we will keep all sizes sum per each block
    torch::Tensor block_sums = torch::empty({numBlocks}, options_int32.dtype(torch::kInt32));

    const float* cov3D_precomp_data_ptr = cov3D_precomp.numel() > 0 ? cov3D_precomp.data_ptr<float>() : nullptr;
    const float* colors_precomp_data_ptr = colors_precomp.numel() > 0 ? colors_precomp.data_ptr<float>() : nullptr;

    torch::Tensor cov_3d = cov3D_precomp.numel()
                               ? cov3D_precomp
                               : torch::empty({num_of_gaussians, 6}, torch::dtype(torch::kFloat).device(torch::kCUDA));

    rasterizer_preprocessing_kernel<<<numBlocks, threads, threads * sizeof(int)>>>(
        means_3d.data_ptr<float>(), shs.data_ptr<float>(), scales.data_ptr<float>(), rotations.data_ptr<float>(),
        settings.proj_matrix.data_ptr<float>(), settings.view_matrix.data_ptr<float>(),
        settings.campos.data_ptr<float>(), opacities.data_ptr<float>(), cov3D_precomp_data_ptr, colors_precomp_data_ptr,
        scale_modifier, max_coeff, settings.sh_degree, num_of_gaussians, focal_x, focal_y, settings.tanfovx,
        settings.tanfovy, c_x, c_y, settings.image_width, settings.image_height, radii.data_ptr<int32_t>(),
        block_sums.data_ptr<int32_t>(), gaussian_data_ptr, cov_3d.data_ptr<float>());

    return std::make_tuple(radii, gaussian_tensor, cov_3d, block_sums, numBlocks, threads);
}

std::vector<torch::Tensor> rasterizer_forward_deepsense(torch::Tensor means_3d, torch::Tensor means_2d,
                                                        torch::Tensor shs, torch::Tensor colors_precomp,
                                                        torch::Tensor opacities, torch::Tensor scales,
                                                        torch::Tensor rotations, torch::Tensor cov3D_precomp,
                                                        const TorchRasterizationSettings& settings)
{
    CHECK_INPUT(means_3d);
    CHECK_INPUT(means_2d);
    CHECK_INPUT(opacities);

    TORCH_CHECK((shs.numel() > 0 && shs.is_contiguous() && shs.is_cuda()) || shs.numel() == 0, "shs error");
    TORCH_CHECK((scales.numel() > 0 && scales.is_contiguous() && scales.is_cuda()) || scales.numel() == 0,
                "scales error");
    TORCH_CHECK((rotations.numel() > 0 && rotations.is_contiguous() && rotations.is_cuda()) || rotations.numel() == 0,
                "rotations error");
    TORCH_CHECK((cov3D_precomp.numel() > 0 && cov3D_precomp.is_contiguous() && cov3D_precomp.is_cuda()) ||
                    (cov3D_precomp.numel() == 0 && scales.numel() > 0 && scales.sizes()[0] == rotations.sizes()[0]),
                "cov3D error");
    TORCH_CHECK((colors_precomp.numel() > 0 && colors_precomp.is_contiguous() && colors_precomp.is_cuda()) ||
                    (colors_precomp.numel() == 0 && shs.numel() > 0),
                "SHS error");

    auto [radii, gaussian_tensor, cov_3d, block_sums, num_blocks_preprocessing, num_threads_preprocessing] =
        rasterizer_forward_preprocessing(means_3d, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp,
                                         settings);

    auto [keys, indices] = duplicate_keys(block_sums, gaussian_tensor, radii, num_blocks_preprocessing,
                                          num_threads_preprocessing, settings);

    sort_by_keys(indices, keys);

    torch::Tensor tile_indices = identify_tile_ranges(keys, settings);
    auto [rendered_image, final_Ts, final_idx] =
        rasterizer_cuda_forward(gaussian_tensor, indices, tile_indices, settings);

    torch::Tensor clamped_colors = torch::ones({means_3d.size(0), 3}, means_3d.options());

    return {rendered_image, radii, gaussian_tensor, final_Ts, final_idx, clamped_colors, cov_3d, indices, tile_indices};
}

torch::Tensor calculate_compensation(torch::Tensor a)
{
    auto b = a.clone();
    b.select(1, 0) -= 0.3f;
    b.select(1, 2) -= 0.3f;

    auto det_a = a.select(1, 0) * a.select(1, 2) - torch::pow(a.select(1, 1), 2);
    auto det_b = b.select(1, 0) * b.select(1, 2) - torch::pow(b.select(1, 1), 2);

    auto result = det_a / det_b;

    return torch::sqrt(torch::max(result, torch::zeros(result.sizes(), result.options())));
}

int sh_degree(int num_bases)
{
    switch(num_bases)
    {
        case 1:
            return 0;
        case 4:
            return 1;
        case 9:
            return 2;
        case 16:
            return 3;
        default:
            return 3;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tensors_from_gaussians(torch::Tensor gaussians)
{
    const Gaussian* gaussian_data_ptr = reinterpret_cast<const Gaussian*>(gaussians.data_ptr<uint8_t>());
    const int num_of_gaussians = gaussians.numel() / sizeof(Gaussian);

    const auto options = torch::dtype(torch::kFloat).device(torch::kCUDA);

    torch::Tensor xy = torch::empty({num_of_gaussians, 2}, options);
    torch::Tensor colors = torch::empty({num_of_gaussians, 3}, options);
    torch::Tensor conic = torch::empty({num_of_gaussians, 3}, options);

    const auto threads = 768;
    const int blocks = (num_of_gaussians + threads - 1) / threads;

    tensors_from_gaussians_kernel<<<blocks, threads>>>(
        gaussian_data_ptr, reinterpret_cast<float2*>(xy.data_ptr<float>()),
        reinterpret_cast<float3*>(colors.data_ptr<float>()), reinterpret_cast<float3*>(conic.data_ptr<float>()),
        num_of_gaussians);

    return std::make_tuple(xy, colors, conic);
}

std::vector<torch::Tensor> rasterizer_backward_deepsense(
    torch::Tensor means_3d, torch::Tensor means2d, torch::Tensor shs, torch::Tensor colors_precomp,
    torch::Tensor opacities, torch::Tensor scales, torch::Tensor rotations, torch::Tensor cov3D_precomp,
    torch::Tensor gaussians, torch::Tensor radii, torch::Tensor colors_clamped, torch::Tensor final_Ts,
    torch::Tensor final_index, torch::Tensor cov3D, torch::Tensor indices, torch::Tensor tile_indices,
    torch::Tensor v_image, const TorchRasterizationSettings& settings)
{
    const auto num_of_gaussians = means_3d.size(0);

    const dim3 block_dim(TILE_SIZE, TILE_SIZE, 1);
    const dim3 grid_dim((settings.image_width + block_dim.x - 1) / block_dim.x,
                        (settings.image_height + block_dim.y - 1) / block_dim.y, 1);

    const auto num_of_tiles = grid_dim.x * grid_dim.y;
    torch::Tensor tile_bins = torch::empty({num_of_tiles, 2}, means_3d.options().dtype(torch::kInt32));

    using torch::indexing::None;
    using torch::indexing::Slice;

    tile_bins.index_put_({Slice(0, num_of_tiles), 0}, tile_indices.index({Slice(0, num_of_tiles)}));
    tile_bins.index_put_({Slice(0, num_of_tiles), 1}, tile_indices.index({Slice(1, num_of_tiles + 1)}));

    torch::Tensor v_alpha = torch::zeros_like(v_image.index({"...", 0}));

    auto [v_xy, v_conic, v_colors, v_opacity] =
        rasterize_backward_tensor(settings.image_height, settings.image_width, TILE_SIZE, indices, tile_bins, gaussians,
                                  opacities, settings.bg, final_Ts, final_index, v_image, v_alpha);

    torch::Tensor viewdirs = means_3d.detach() - settings.campos;
    v_colors = v_colors * colors_clamped;

    auto v_shs =
        compute_sh_backward_tensor(num_of_gaussians, sh_degree(shs.size(-2)), settings.sh_degree, viewdirs, v_colors);

    const auto focal_x = fov_to_focal(2 * atan(settings.tanfovx), settings.image_width);
    const auto focal_y = fov_to_focal(2 * atan(settings.tanfovy), settings.image_height);
    const auto c_x = settings.image_width / 2.0f;
    const auto c_y = settings.image_height / 2.0f;

    // TODO: remove compensation
    // TODO: compensation in
    auto [xy, colors, conics] = tensors_from_gaussians(gaussians);
    auto compensation = calculate_compensation(conics.detach());
    torch::Tensor v_depth = torch::zeros({num_of_gaussians, 1}, v_xy.options());
    torch::Tensor view_matrix = settings.view_matrix;

    auto [v_cov2d, v_cov3d, v_mean3d, v_scale, v_rotation] = project_gaussians_backward_tensor(
        num_of_gaussians, means_3d, scales, settings.scale_modifier, rotations, view_matrix, focal_x, focal_y, c_x, c_y,
        settings.image_height, settings.image_width, gaussians, cov3D, radii, compensation, v_xy, v_depth, v_conic);

    v_xy = torch::cat({v_xy, torch::zeros({v_xy.size(0), 1}, v_xy.options())}, 1);

    return {v_mean3d, v_xy, v_shs, v_colors, v_opacity, v_scale, v_rotation, v_cov3d};
}
