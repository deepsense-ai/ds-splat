#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../rasterizer_kernels.cuh"

// TODO
/* clang-format off */

__global__ void project_gaussians_backward_kernel(
    const int num_points,
    const float3* __restrict__ means3d,
    const float3* __restrict__ scales,
    const float glob_scale,
    const float4* __restrict__ quats,
    const float* __restrict__ viewmat,
    const float4 intrins,
    const dim3 img_size,
    const Gaussian* __restrict__ gaussians,
    const float* __restrict__ cov3d,
    const int* __restrict__ radii,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cov2d,
    float* __restrict__ v_cov3d,
    float3* __restrict__ v_mean3d,
    float3* __restrict__ v_scale,
    float4* __restrict__ v_quat
);


__global__ void rasterize_backward_kernel(
    const dim3 tile_bounds,
    const dim3 img_size,
    const int32_t* __restrict__ gaussian_ids_sorted,
    const int2* __restrict__ tile_bins,
    const Gaussian* __restrict__ gaussians,
    const float* __restrict__ opacities,
    const float3& __restrict__ background,
    const float* __restrict__ final_Ts,
    const int* __restrict__ final_index,
    const float3* __restrict__ v_output,
    const float* __restrict__ v_output_alpha,
    float2* __restrict__ v_xy,
    float3* __restrict__ v_conic,
    float3* __restrict__ v_rgb,
    float* __restrict__ v_opacity
);

__device__ void project_cov3d_ewa_vjp(
    const float3 &mean3d,
    const float *cov3d,
    const float *viewmat,
    const float fx,
    const float fy,
    const float3 &v_cov2d,
    float3 &v_mean3d,
    float *v_cov3d
);

__device__ void scale_rot_to_cov3d_vjp(
    const float3 scale,
    const float glob_scale,
    const float4 quat,
    const float *v_cov3d,
    float3 &v_scale,
    float4 &v_quat
);
