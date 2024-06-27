/* clang-format off */

#include "cuda_runtime.h"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

torch::Tensor compute_sh_backward_tensor(
    unsigned num_points,
    unsigned degree,
    unsigned degrees_to_use,
    torch::Tensor &viewdirs,
    torch::Tensor &v_colors
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor &means3d,
    torch::Tensor &scales,
    const float glob_scale,
    torch::Tensor &quats,
    torch::Tensor &viewmat,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const unsigned img_height,
    const unsigned img_width,
    torch::Tensor &cov3d,
    torch::Tensor &radii,
    torch::Tensor &conics,
    torch::Tensor &compensation,
    torch::Tensor &v_xy,
    torch::Tensor &v_depth,
    torch::Tensor &v_conic
);

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const unsigned block_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &gaussians,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output, // dL_dout_color
        const torch::Tensor &v_output_alpha
    );
