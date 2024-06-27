#ifndef RASTERIZER_KERNELS_CUH
#define RASTERIZER_KERNELS_CUH

#include <cstdlib>

constexpr auto TILE_SIZE = 16;
constexpr auto num_of_threads_in_block = TILE_SIZE * TILE_SIZE;
constexpr auto num_of_gaussians_in_shared_memory = num_of_threads_in_block;

__global__ void check_gaussians_in_frustum(const float* means_3d, const float* scales_3d, int32_t* results,
                                           const float* projection, int num_gaussians);

struct __align__(16) float6
{
    float x, y, z, w, v, u;
};

struct __align__(16) Gaussian
{
    float3 means_2d;              // 0-12
    float3 conic;                 // 12-24
    float3 color;                 // 24-36
    float alpha;                  // 36-40
    int size_occupied_tiles;      // 40-44
    int size_occupied_tiles_next; // 4 padding to 48
};

__device__ float6 calc_cov3d(const float* __restrict__ scales, const float* __restrict__ rotations,
                             float scale_modifier);
__device__ int32_t calculate_radii(const float3& cov2d, int32_t _radii);
__device__ float3 calc_shs_deg_0(const float* features, const float3& means3d, int index_features);
__device__ float3 calc_shs_deg_1(const float* features, const float3& means3d, const float3& camera_center,
                                 int index_features);
__device__ float3 calc_shs_deg_2(const float* features, const float3& means3d, const float3& camera_center,
                                 int index_features);
__device__ float3 calc_shs_deg_3(const float* features, const float3& means3d, const float3& camera_center,
                                 int index_features);
__device__ float3 project_point(const float3& points, const float* __restrict__ projection);
__device__ int32_t is_gaussian_in_frustum(const float3& means_3d, const float* projection_matrix);
__device__ float3 calculate_cov_2d(float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float3& means_2d,
                                   const float* __restrict__ view_matrix, const float6& cov3D, int num_points);
__device__ float2 project_to_image(float3& points, float fx, float fy, float cx, float cy, int numPoints);
__device__ int calculate_sizes(const float3& xyd, int radii, unsigned int img_width, int img_height);

__device__ float3 switch_shs_func(const float* features, const float3& means3d, int deg, const float3& camera_center,
                                  int index_features, int active_sh_degree);
__device__ float6 make_float6(float x, float y, float z, float w, float v, float u);

__global__ void rasterizer_preprocessing_kernel(
    const float* __restrict__ means3d, const float* features, const float* __restrict__ scales,
    const float* __restrict__ rotations, const float* __restrict__ projection_matrix,
    const float* __restrict__ view_matrix, const float* __restrict__ camera_center, const float* __restrict__ opacities,
    const float* __restrict__ cov3D_precomp, const float* __restrict__ colors_precomp, float scale_modifier,
    int max_coeff, int active_sh_degree, int num_of_gaussians, float focal_x, float focal_y, float tan_fovx,
    float tan_fovy, float c_x, float c_y, unsigned int img_width, int img_height, int32_t* __restrict__ radii,
    int* __restrict__ block_sums, Gaussian* __restrict__ gaussians, float* __restrict__ cov_3d_output = nullptr);

__global__ void calculate_sizes_kernel(const float* __restrict__ ndc, const int32_t* __restrict__ radii,
                                       int* __restrict__ sizes, unsigned int num_ndc, unsigned int img_width,
                                       int img_height);

__global__ void calculate_keys_and_indices_kernel(const Gaussian* __restrict__ gaussians,
                                                  const int32_t* __restrict__ block_sums,
                                                  const int32_t* __restrict__ radii, int64_t* __restrict__ keys,
                                                  int32_t* __restrict__ indices, int num_ndc, int num_indices,
                                                  unsigned int img_width, unsigned int img_height);

__global__ void identify_tile_ranges_kernel(const int64_t* __restrict__ keys, int32_t* __restrict__ tile_indices,
                                            int size);
__global__ void render_image_kernel(const Gaussian* __restrict__ gaussians_global, const int32_t* __restrict__ indices,
                                    const int32_t* __restrict__ tile_indices, float* __restrict__ output_image,
                                    int image_width, int image_height, float* __restrict__ final_Ts = nullptr,
                                    int32_t* __restrict__ final_idx = nullptr);

#endif
