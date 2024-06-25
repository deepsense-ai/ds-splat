#include "rasterizer_kernels.cuh"
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__device__ int32_t is_gaussian_in_frustum(const float3& means_3d, const float* projection_matrix)
{
    const auto inv_w = 1.f / (projection_matrix[12] * means_3d.x + projection_matrix[13] * means_3d.y +
                              projection_matrix[14] * means_3d.z + projection_matrix[15]);

    const auto x = inv_w * (projection_matrix[0] * means_3d.x + projection_matrix[1] * means_3d.y +
                            projection_matrix[2] * means_3d.z + projection_matrix[3]);

    const auto y = inv_w * (projection_matrix[4] * means_3d.x + projection_matrix[5] * means_3d.y +
                            projection_matrix[6] * means_3d.z + projection_matrix[7]);

    const auto z = inv_w * (projection_matrix[8] * means_3d.x + projection_matrix[9] * means_3d.y +
                            projection_matrix[10] * means_3d.z + projection_matrix[11]);

    const auto limits = 1.f;

    return x > -limits && x < limits && y > -limits && y < limits && z > 0.1f;
}

__device__ float3 switch_shs_func(const float* features, const float3& means3d, int deg, const float3& camera_center,
                                  int index_features, int active_sh_degree)
{
    if(active_sh_degree == 0)
        return calc_shs_deg_0(features, means3d, index_features);

    if(active_sh_degree == 1)
        return calc_shs_deg_1(features, means3d, camera_center, index_features);

    if(active_sh_degree == 2)
        return calc_shs_deg_2(features, means3d, camera_center, index_features);

    if(active_sh_degree == 3)
        return calc_shs_deg_3(features, means3d, camera_center, index_features);

    return {0.0f, 0.0f, 0.0f};
}

__device__ float6 make_float6(float x, float y, float z, float w, float v, float u)
{
    return {x, y, z, w, v, u};
}

__device__ float3 calculate_conic(const float3& cov_2d)
{
    const auto det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
    const auto det_inv = 1.0f / (det + 1e-6f);

    return {cov_2d.z * det_inv, -cov_2d.y * det_inv, cov_2d.x * det_inv};
}

__global__ void rasterizer_preprocessing_kernel(
    const float* __restrict__ means3d, const float* features, const float* __restrict__ scales,
    const float* __restrict__ rotations, const float* __restrict__ projection_matrix,
    const float* __restrict__ view_matrix, const float* __restrict__ camera_center, const float* __restrict__ opacities,
    const float* __restrict__ cov3D_precomp, const float* __restrict__ colors_precomp, float scale_modifier,
    int max_coeff, int active_sh_degree, int num_of_gaussians, float focal_x, float focal_y, float tan_fovx,
    float tan_fovy, float c_x, float c_y, unsigned int img_width, int img_height, int32_t* __restrict__ radii,
    int* __restrict__ block_sums, Gaussian* __restrict__ gaussians, float* __restrict__ cov_3d_output)
{
    __shared__ float _projection_matrix[16];
    __shared__ float _view_matrix[12];
    __shared__ float _camera_center[3];
    extern __shared__ int shared_inclusive_sum[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int bdim = blockDim.x;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_4 = index * 4;
    const int index_3 = index_4 - index; // index*3
    const int index_6 = index * 6;
    int index_features = max_coeff * index_3;

    if(threadIdx.x < 16)
        _projection_matrix[threadIdx.x] = projection_matrix[threadIdx.x];
    if(threadIdx.x < 12)
        _view_matrix[threadIdx.x] = view_matrix[threadIdx.x];
    if(threadIdx.x < 3)
        _camera_center[threadIdx.x] = camera_center[threadIdx.x];

    __syncthreads();

    if(index < num_of_gaussians)
    {
        const float6 _cov3D =
            cov3D_precomp == nullptr
                ? calc_cov3d(&scales[index_3], &rotations[index_4], scale_modifier)
                : make_float6(cov3D_precomp[index_6], cov3D_precomp[index_6 + 1], cov3D_precomp[index_6 + 2],
                              cov3D_precomp[index_6 + 3], cov3D_precomp[index_6 + 4], cov3D_precomp[index_6 + 5]);

        if(cov_3d_output && cov3D_precomp == nullptr)
        {
            cov_3d_output[index_6] = _cov3D.x;
            cov_3d_output[index_6 + 1] = _cov3D.y;
            cov_3d_output[index_6 + 2] = _cov3D.z;
            cov_3d_output[index_6 + 3] = _cov3D.w;
            cov_3d_output[index_6 + 4] = _cov3D.v;
            cov_3d_output[index_6 + 5] = _cov3D.u;
        }

        const float3 _means_3d = float3{means3d[index_3], means3d[index_3 + 1], means3d[index_3 + 2]};
        const float3 _campos = float3{_camera_center[0], _camera_center[1], _camera_center[2]};

        const float3 colors =
            (colors_precomp == nullptr)
                ? switch_shs_func(features, _means_3d, active_sh_degree, _campos, index_features, active_sh_degree)
                : make_float3(colors_precomp[index_3], colors_precomp[index_3 + 1], colors_precomp[index_3 + 2]);

        int32_t mask_radii = is_gaussian_in_frustum(_means_3d, _projection_matrix);
        float3 _means_2d = project_point(_means_3d, _view_matrix);
        float3 _cov_2d =
            calculate_cov_2d(focal_x, focal_y, tan_fovx, tan_fovy, _means_2d, _view_matrix, _cov3D, num_of_gaussians);
        const float2 _means_2d_update = project_to_image(_means_2d, focal_x, focal_y, c_x, c_y, num_of_gaussians);

        _means_2d.x = _means_2d_update.x;
        _means_2d.y = _means_2d_update.y;

        mask_radii = calculate_radii(_cov_2d, mask_radii);

        const int _size = calculate_sizes(_means_2d, mask_radii, img_width, img_height);
        radii[index] = mask_radii; // global mem write needed for keys preparation next kernel

        const float _opacity = opacities[index];

        shared_inclusive_sum[tid] = (tid < num_of_gaussians) ? _size : 0; // todo simplify
        __syncthreads();

        // Intra-block inclusive scan using shared memory
        for(int d = 1; d < bdim; d *= 2)
        {
            int t = (tid >= d) ? shared_inclusive_sum[tid - d] : 0;
            __syncthreads();
            if(tid >= d)
                shared_inclusive_sum[tid] += t;
            __syncthreads();
        }

        const float3 conic = calculate_conic(_cov_2d);

        gaussians[index].means_2d = _means_2d;
        gaussians[index].conic = conic;
        gaussians[index].color = colors;
        gaussians[index].alpha = _opacity;
        gaussians[index].size_occupied_tiles = (tid > 0) ? shared_inclusive_sum[tid - 1] : 0; // local exclusive sum
        gaussians[index].size_occupied_tiles_next =
            shared_inclusive_sum[tid]; // local inclusive sum (or next thread local exlcusive sum)
    }

    if(tid == bdim - 1)
    {
        // write entire block inclusive sum; we will use it to calculate exclusive sum between blocks
        // and later restore global exclusive sum
        block_sums[bid] = shared_inclusive_sum[tid];
    }
}

__device__ int32_t calculate_radii(const float3& cov2d, int32_t _radii)
{
    const float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    const float mid = 0.5f * (cov2d.x + cov2d.z);
    const float diff = mid * mid - det;
    const float lambda1 = mid + sqrt(max(0.1f, diff));
    const float lambda2 = mid - sqrt(max(0.1f, diff));

    return ceilf(3 * sqrtf(max(lambda1, lambda2))) * (det > 1e-6f && _radii > 0);
}

__device__ float3 calculate_cov_2d(float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float3& means_2d,
                                   const float* __restrict__ view_matrix, const float6& cov3D, int num_points)
{
    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = means_2d.x / means_2d.z;
    const float tytz = means_2d.y / means_2d.z;
    const float t[3] = {min(limx, max(-limx, txtz)) * means_2d.z, min(limy, max(-limy, tytz)) * means_2d.z, means_2d.z};

    const float J[3][3] = {{focal_x / t[2], 0, -(focal_x * t[0]) / (t[2] * t[2])},
                           {0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])},
                           {0, 0, 0}};

    const float W[3][3] = {{view_matrix[0], view_matrix[4], view_matrix[8]},
                           {view_matrix[1], view_matrix[5], view_matrix[9]},
                           {view_matrix[2], view_matrix[6], view_matrix[10]}};

    float T[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            for(int k = 0; k < 3; ++k)
            {
                T[i][j] += W[i][k] * J[j][k];
            }
        }
    }

    const float Vrk[3][3] = {{cov3D.x, cov3D.y, cov3D.z}, {cov3D.y, cov3D.w, cov3D.v}, {cov3D.z, cov3D.v, cov3D.u}};

    float cov[2][2] = {{0, 0}, {0, 0}};
    float temp[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            for(int k = 0; k < 3; ++k)
            {
                temp[i][j] += T[k][i] * Vrk[k][j];
            }
        }
    }

    for(int i = 0; i < 2; ++i)
    {
        for(int j = 0; j < 2; ++j)
        {
            for(int k = 0; k < 3; ++k)
            {
                cov[i][j] += temp[i][k] * T[k][j];
            }
        }
    }

    return {cov[0][0] + 0.3f, cov[0][1], cov[1][1] + 0.3f};
}

__device__ int calculate_sizes(const float3& xyd, int radii, unsigned int img_width, int img_height)
{
    const auto x = xyd.x;
    const auto y = xyd.y;
    const auto radius = radii;

    const float2 center = {x / TILE_SIZE, y / TILE_SIZE};
    const float2 dims = {radius / (float)TILE_SIZE, radius / (float)TILE_SIZE};
    const dim3 img_size = {static_cast<unsigned int>((img_width + TILE_SIZE - 1) / TILE_SIZE),
                           static_cast<unsigned int>((img_height + TILE_SIZE - 1) / TILE_SIZE)};

    const int start_x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    const int end_x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    const int start_y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    const int end_y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);

    return (end_x - start_x) * (end_y - start_y) * (radius > 0);
}

__global__ void calculate_keys_and_indices_kernel(const Gaussian* __restrict__ gaussians,
                                                  const int32_t* __restrict__ block_sums,
                                                  const int32_t* __restrict__ radii, int64_t* __restrict__ keys,
                                                  int32_t* __restrict__ indices, int num_ndc, int num_indices,
                                                  unsigned int img_width, unsigned int img_height)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid = blockIdx.x;

    // store exclusive sums for this thread and next thread.
    // Next thread block sums might (and probably is) different if current thread is last thread in the block
    __shared__ int blocks_prefix[1];

    if(threadIdx.x == 0)
    {
        blocks_prefix[0] = block_sums[bid];
    }
    __syncthreads();

    if(index < num_ndc)
    {
        Gaussian gaussian = gaussians[index];
        // restore global exclusive sum for start and end indices
        auto start_index = gaussian.size_occupied_tiles + blocks_prefix[0];
        auto end_index = (index + 1) < num_ndc ? gaussian.size_occupied_tiles_next + blocks_prefix[0] : num_indices;

        if(end_index - start_index > 0)
        {
            const auto x = gaussian.means_2d.x;
            const auto y = gaussian.means_2d.y;
            const auto radius = radii[index];
            const int64_t depth = (int64_t) * (int32_t*)&(gaussian.means_2d.z);
            const int tiles_per_width = (img_width + TILE_SIZE - 1) / TILE_SIZE;

            const float2 center = {x / TILE_SIZE, y / TILE_SIZE};
            const float2 dims = {radius / (float)TILE_SIZE, radius / (float)TILE_SIZE};
            const dim3 img_size = {(img_width + TILE_SIZE - 1) / TILE_SIZE, (img_height + TILE_SIZE - 1) / TILE_SIZE};
            const int start_x = min(max(0, (int)(center.x - dims.x)), img_size.x);
            const int end_x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
            const int start_y = min(max(0, (int)(center.y - dims.y)), img_size.y);
            const int end_y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);

            int local_index = start_index;
            for(int tile_y = start_y; tile_y < end_y; ++tile_y)
            {
                for(int tile_x = start_x; tile_x < end_x; ++tile_x, ++local_index)
                {
                    const int64_t tile_index = tile_y * tiles_per_width + tile_x;
                    keys[local_index] = (tile_index << 32) | depth;
                    indices[local_index] = index;
                }
            }
        }
    }
}

__global__ void identify_tile_ranges_kernel(const int64_t* __restrict__ keys, int32_t* __restrict__ tile_indices,
                                            int size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size - 1)
    {
        const int64_t current_key = keys[index];
        const int64_t next_key = keys[index + 1];
        const int32_t current_tile = current_key >> 32;
        const int32_t next_tile = next_key >> 32;

        if(current_tile != next_tile)
            tile_indices[current_tile + 1] = index + 1;
    }
}

__device__ float3 project_point(const float3& points, const float* __restrict__ projection)
{
    return {projection[0] * points.x + projection[1] * points.y + projection[2] * points.z + projection[3],
            projection[4] * points.x + projection[5] * points.y + projection[6] * points.z + projection[7],
            projection[8] * points.x + projection[9] * points.y + projection[10] * points.z + projection[11]};
}

__device__ float2 project_to_image(float3& points, float fx, float fy, float cx, float cy, int numPoints)
{
    const auto z_inv = 1.f / (points.z + 1e-6f);

    return {fx * points.x * z_inv + cx, fy * points.y * z_inv + cy};
}

static constexpr float alpha_threshold = 1.0f / 255.0f;

__global__ void render_image_kernel(const Gaussian* __restrict__ gaussians_global, const int32_t* __restrict__ indices,
                                    const int32_t* __restrict__ tile_indices, float* __restrict__ output_image,
                                    int image_width, int image_height, float* __restrict__ final_Ts,
                                    int32_t* __restrict__ final_idx)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ Gaussian gaussians[num_of_gaussians_in_shared_memory];

    const auto tile_x = blockIdx.x;
    const auto tile_y = blockIdx.y;
    const auto tile_index = tile_y * gridDim.x + tile_x;
    const auto start_index = tile_indices[tile_index];
    const auto end_index = tile_indices[tile_index + 1];
    const auto num_of_gaussians = end_index - start_index;

    float4 blendedColor = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    int32_t curr_idx = 0;

    const int num_of_syncs = (num_of_gaussians + num_of_threads_in_block - 1) / num_of_threads_in_block;

    bool is_ok = true;
    for(int s = 0; s < num_of_syncs; ++s)
    {
        if(__syncthreads_count(!is_ok) >= num_of_threads_in_block)
        {
            break;
        }

        const int thread_idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;
        Gaussian& gaussian_to_init = gaussians[thread_idx_in_block];
        const int current_index_for_thread = start_index + s * num_of_gaussians_in_shared_memory + thread_idx_in_block;

        if(current_index_for_thread < end_index)
        {
            const auto gaussian_index = indices[current_index_for_thread];
            gaussian_to_init = gaussians_global[gaussian_index];
        }

        __syncthreads();

        const int gaussians_in_current_sync =
            min(num_of_gaussians_in_shared_memory, num_of_gaussians - s * num_of_gaussians_in_shared_memory);

        for(int i = 0; i < gaussians_in_current_sync && is_ok; ++i)
        {
            const Gaussian& gaussian = gaussians[i];
            const float3& conic = gaussian.conic;
            const float2 d = {x + 0.5f - gaussian.means_2d.x, y + 0.5f - gaussian.means_2d.y};
            const float power = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
            const float alpha = min(0.999f, gaussian.alpha * expf(power));

            if(power >= 0.f || alpha < alpha_threshold)
            {
                continue;
            }

            const float T = (1.0f - alpha) * blendedColor.w;

            is_ok = T >= 0.0001f;

            if(is_ok)
            {
                const float vis = alpha * blendedColor.w;

                blendedColor.x += vis * gaussian.color.x;
                blendedColor.y += vis * gaussian.color.y;
                blendedColor.z += vis * gaussian.color.z;
                blendedColor.w = T;

                curr_idx = start_index + s * num_of_gaussians_in_shared_memory + i;
            }
        }
    }

    if(x < image_width && y < image_height)
    {
        const auto pixel_idx = 3 * (y * image_width + x);
        output_image[pixel_idx] = blendedColor.x;
        output_image[pixel_idx + 1] = blendedColor.y;
        output_image[pixel_idx + 2] = blendedColor.z;

        if(final_Ts && final_idx)
        {
            const auto final_index = y * image_width + x;
            final_Ts[final_index] = blendedColor.w;
            final_idx[final_index] = curr_idx;
        }
    }
}

__device__ void add_shs_deg_0(const float* __restrict__ features, float3& result)
{
    const float sh0_R = features[0];
    const float sh0_G = features[1];
    const float sh0_B = features[2];

    const float C0 = 0.28209479177387814f;
    result.x += C0 * sh0_R;
    result.y += C0 * sh0_G;
    result.z += C0 * sh0_B;
}

__device__ void add_shs_deg_1(const float* __restrict__ features, const float3& viewdir, float3& result)
{
    const float sh1_R = features[3];
    const float sh1_G = features[4];
    const float sh1_B = features[5];
    const float sh2_R = features[6];
    const float sh2_G = features[7];
    const float sh2_B = features[8];
    const float sh3_R = features[9];
    const float sh3_G = features[10];
    const float sh3_B = features[11];

    const float C1 = 0.4886025119029199f;
    result.x += C1 * (-viewdir.y * sh1_R + viewdir.z * sh2_R - viewdir.x * sh3_R);
    result.y += C1 * (-viewdir.y * sh1_G + viewdir.z * sh2_G - viewdir.x * sh3_G);
    result.z += C1 * (-viewdir.y * sh1_B + viewdir.z * sh2_B - viewdir.x * sh3_B);
}

__device__ void add_shs_deg_2(const float* __restrict__ features, const float3& viewdir, float3& result)
{
    const float sh4_R = features[12];
    const float sh4_G = features[13];
    const float sh4_B = features[14];
    const float sh5_R = features[15];
    const float sh5_G = features[16];
    const float sh5_B = features[17];
    const float sh6_R = features[18];
    const float sh6_G = features[19];
    const float sh6_B = features[20];
    const float sh7_R = features[21];
    const float sh7_G = features[22];
    const float sh7_B = features[23];
    const float sh8_R = features[24];
    const float sh8_G = features[25];
    const float sh8_B = features[26];

    const float xx = viewdir.x * viewdir.x;
    const float yy = viewdir.y * viewdir.y;
    const float zz = viewdir.z * viewdir.z;
    const float xy = viewdir.x * viewdir.y;
    const float xz = viewdir.x * viewdir.z;
    const float yz = viewdir.y * viewdir.z;

    const float C2_0 = 1.0925484305920792f;
    const float C2_1 = -1.0925484305920792f;
    const float C2_2 = 0.31539156525252005f;
    const float C2_3 = -1.0925484305920792f;
    const float C2_4 = 0.5462742152960396f;

    result.x += C2_0 * xy * sh4_R + C2_1 * yz * sh5_R + C2_2 * (2.0f * zz - xx - yy) * sh6_R + C2_3 * xz * sh7_R +
                C2_4 * (xx - yy) * sh8_R;
    result.y += C2_0 * xy * sh4_G + C2_1 * yz * sh5_G + C2_2 * (2.0f * zz - xx - yy) * sh6_G + C2_3 * xz * sh7_G +
                C2_4 * (xx - yy) * sh8_G;
    result.z += C2_0 * xy * sh4_B + C2_1 * yz * sh5_B + C2_2 * (2.0f * zz - xx - yy) * sh6_B + C2_3 * xz * sh7_B +
                C2_4 * (xx - yy) * sh8_B;
}

__device__ void add_shs_deg_3(const float* __restrict__ features, const float3& viewdir, float3& result)
{
    const float sh9_R = features[27];
    const float sh9_G = features[28];
    const float sh9_B = features[29];
    const float sh10_R = features[30];
    const float sh10_G = features[31];
    const float sh10_B = features[32];
    const float sh11_R = features[33];
    const float sh11_G = features[34];
    const float sh11_B = features[35];
    const float sh12_R = features[36];
    const float sh12_G = features[37];
    const float sh12_B = features[38];
    const float sh13_R = features[39];
    const float sh13_G = features[40];
    const float sh13_B = features[41];
    const float sh14_R = features[42];
    const float sh14_G = features[43];
    const float sh14_B = features[44];
    const float sh15_R = features[45];
    const float sh15_G = features[46];
    const float sh15_B = features[47];

    const float xx = viewdir.x * viewdir.x;
    const float yy = viewdir.y * viewdir.y;
    const float zz = viewdir.z * viewdir.z;
    const float xy = viewdir.x * viewdir.y;

    const float C3_0 = -0.5900435899266435f;
    const float C3_1 = 2.890611442640554f;
    const float C3_2 = -0.4570457994644658f;
    const float C3_3 = 0.3731763325901154f;
    const float C3_4 = -0.4570457994644658f;
    const float C3_5 = 1.445305721320277f;
    const float C3_6 = -0.5900435899266435f;

    result.x += C3_0 * viewdir.y * (3 * xx - yy) * sh9_R + C3_1 * xy * viewdir.z * sh10_R +
                C3_2 * viewdir.y * (4 * zz - xx - yy) * sh11_R +
                C3_3 * viewdir.z * (2 * zz - 3 * xx - 3 * yy) * sh12_R +
                C3_4 * viewdir.x * (4 * zz - xx - yy) * sh13_R + C3_5 * viewdir.z * (xx - yy) * sh14_R +
                C3_6 * viewdir.x * (xx - 3 * yy) * sh15_R;
    result.y += C3_0 * viewdir.y * (3 * xx - yy) * sh9_G + C3_1 * xy * viewdir.z * sh10_G +
                C3_2 * viewdir.y * (4 * zz - xx - yy) * sh11_G +
                C3_3 * viewdir.z * (2 * zz - 3 * xx - 3 * yy) * sh12_G +
                C3_4 * viewdir.x * (4 * zz - xx - yy) * sh13_G + C3_5 * viewdir.z * (xx - yy) * sh14_G +
                C3_6 * viewdir.x * (xx - 3 * yy) * sh15_G;
    result.z += C3_0 * viewdir.y * (3 * xx - yy) * sh9_B + C3_1 * xy * viewdir.z * sh10_B +
                C3_2 * viewdir.y * (4 * zz - xx - yy) * sh11_B +
                C3_3 * viewdir.z * (2 * zz - 3 * xx - 3 * yy) * sh12_B +
                C3_4 * viewdir.x * (4 * zz - xx - yy) * sh13_B + C3_5 * viewdir.z * (xx - yy) * sh14_B +
                C3_6 * viewdir.x * (xx - 3 * yy) * sh15_B;
}

__device__ float3 calc_shs_deg_0(const float* __restrict__ features, const float3& means3d, int index_features)
{
    float3 result = {0.0f, 0.0f, 0.0f};

    add_shs_deg_0(&features[index_features], result);

    return {fmaxf(result.x + 0.5f, 0.0f), fmaxf(result.y + 0.5f, 0.0f), fmaxf(result.z + 0.5f, 0.0f)};
}

__device__ float3 calculate_viewdir(const float3& means_3d, const float3& camera_center)
{
    const float3 viewdir = {means_3d.x - camera_center.x, means_3d.y - camera_center.y, means_3d.z - camera_center.z};
    const float inv_norm = rsqrtf(viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z);

    return {viewdir.x * inv_norm, viewdir.y * inv_norm, viewdir.z * inv_norm};
}

__device__ float3 calc_shs_deg_1(const float* __restrict__ features, const float3& means_3d,
                                 const float3& camera_center, int index_features)
{
    const auto viewdir = calculate_viewdir(means_3d, camera_center);

    float3 result = {0.0f, 0.0f, 0.0f};

    add_shs_deg_0(&features[index_features], result);
    add_shs_deg_1(&features[index_features], viewdir, result);

    return {fmaxf(result.x + 0.5f, 0.0f), fmaxf(result.y + 0.5f, 0.0f), fmaxf(result.z + 0.5f, 0.0f)};
}

__device__ float3 calc_shs_deg_2(const float* __restrict__ features, const float3& means_3d,
                                 const float3& camera_center, int index_features)
{
    const auto viewdir = calculate_viewdir(means_3d, camera_center);

    float3 result = {0.0f, 0.0f, 0.0f};

    add_shs_deg_0(&features[index_features], result);
    add_shs_deg_1(&features[index_features], viewdir, result);
    add_shs_deg_2(&features[index_features], viewdir, result);

    return {fmaxf(result.x + 0.5f, 0.0f), fmaxf(result.y + 0.5f, 0.0f), fmaxf(result.z + 0.5f, 0.0f)};
}

__device__ float3 calc_shs_deg_3(const float* __restrict__ features, const float3& means_3d,
                                 const float3& camera_center, int index_features)
{
    const auto viewdir = calculate_viewdir(means_3d, camera_center);

    float3 result = {0.0f, 0.0f, 0.0f};

    add_shs_deg_0(&features[index_features], result);
    add_shs_deg_1(&features[index_features], viewdir, result);
    add_shs_deg_2(&features[index_features], viewdir, result);
    add_shs_deg_3(&features[index_features], viewdir, result);

    return {fmaxf(result.x + 0.5f, 0.0f), fmaxf(result.y + 0.5f, 0.0f), fmaxf(result.z + 0.5f, 0.0f)};
}

__device__ float6 calc_cov3d(const float* __restrict__ scales, const float* rotations, float scale_modifier)
{
    float r = rotations[0];
    float x = rotations[1];
    float y = rotations[2];
    float z = rotations[3];

    // normalize
    const float inv_norm = rsqrtf(r * r + x * x + y * y + z * z);
    r = r * inv_norm;
    x = x * inv_norm;
    y = y * inv_norm;
    z = z * inv_norm;

    const float3 scale = {scales[0] * scale_modifier, scales[1] * scale_modifier, scales[2] * scale_modifier};

    const float r00 = 1.0f - 2.0f * (y * y + z * z);
    const float r01 = 2.0f * (x * y - r * z);
    const float r02 = 2.0f * (x * z + r * y);
    const float r10 = 2.0f * (x * y + r * z);
    const float r11 = 1.0f - 2.0f * (x * x + z * z);
    const float r12 = 2.0f * (y * z - r * x);
    const float r20 = 2.0f * (x * z - r * y);
    const float r21 = 2.0f * (y * z + r * x);
    const float r22 = 1.0f - 2.0f * (x * x + y * y);

    const float m00 = scale.x * r00;
    const float m01 = scale.y * r01;
    const float m02 = scale.z * r02;
    const float m10 = scale.x * r10;
    const float m11 = scale.y * r11;
    const float m12 = scale.z * r12;
    const float m20 = scale.x * r20;
    const float m21 = scale.y * r21;
    const float m22 = scale.z * r22;

    const float cov00 = m00 * m00 + m01 * m01 + m02 * m02;
    const float cov01 = m00 * m10 + m01 * m11 + m02 * m12;
    const float cov02 = m00 * m20 + m01 * m21 + m02 * m22;
    const float cov11 = m10 * m10 + m11 * m11 + m12 * m12;
    const float cov12 = m10 * m20 + m11 * m21 + m12 * m22;
    const float cov22 = m20 * m20 + m21 * m21 + m22 * m22;

    return {cov00, cov01, cov02, cov11, cov12, cov22};
}

__global__ void tensors_from_gaussians_kernel(const Gaussian* __restrict__ gaussian_data_ptr, float2* __restrict__ xy,
                                              float3* __restrict__ colors, float3* __restrict__ conic,
                                              const int num_of_gaussians)
{
    const auto index = cg::this_grid().thread_rank();

    if(index < num_of_gaussians)
    {
        const auto& gaussian = gaussian_data_ptr[index];

        xy[index] = {gaussian.means_2d.x, gaussian.means_2d.y};
        conic[index] = gaussian.conic;
        colors[index] = gaussian.color;
    }
}
