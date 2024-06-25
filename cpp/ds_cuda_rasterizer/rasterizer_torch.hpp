#ifndef DS_RASTERIZER_HPP_
#define DS_RASTERIZER_HPP_

#include <torch/extension.h>
#include <vector>

struct TorchRasterizationSettings
{
    int image_height;
    int image_width;
    float tanfovx;
    float tanfovy;
    torch::Tensor bg;
    float scale_modifier;
    torch::Tensor view_matrix;
    torch::Tensor proj_matrix;
    int sh_degree;
    int max_sh_degree;
    torch::Tensor campos;
    bool prefiltered;
    bool debug;
};

std::vector<torch::Tensor> rasterizer_forward_deepsense(torch::Tensor means3D, torch::Tensor means2D, torch::Tensor shs,
                                                        torch::Tensor colors_precomp, torch::Tensor opacities,
                                                        torch::Tensor scales, torch::Tensor rotations,
                                                        torch::Tensor cov3D_precomp,
                                                        const TorchRasterizationSettings& settings);

std::vector<torch::Tensor> rasterizer_backward_deepsense(
    torch::Tensor means_3d, torch::Tensor means2d, torch::Tensor shs, torch::Tensor colors_precomp,
    torch::Tensor opacities, torch::Tensor scales, torch::Tensor rotations, torch::Tensor cov3D_precomp,
    torch::Tensor radii, torch::Tensor gaussians, torch::Tensor colors_clamped, torch::Tensor final_Ts,
    torch::Tensor final_index, torch::Tensor cov3D, torch::Tensor indices, torch::Tensor tile_indices,
    torch::Tensor grad_image, const TorchRasterizationSettings& settings);

#endif // DS_RASTERIZER_HPP_
