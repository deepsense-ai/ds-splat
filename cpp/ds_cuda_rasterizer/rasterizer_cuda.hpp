#ifndef RASTERIZER_CUDA_HPP_
#define RASTERIZER_CUDA_HPP_

void rasterizer_forward_core_deepsense(const float* means_3d, const float* shs, const float* opacities,
                                       const float* scales, const float* rotations, int num_of_gaussians,
                                       const float* view_matrix, const float* proj_matrix, const float* camera_position,
                                       int image_width, int image_height, float tan_fovx, float tan_fovy,
                                       float scale_modifier, const int max_sh_degree, const int sh_degree,
                                       float* output_image);

#endif // RASTERIZER_CUDA_HPP_
