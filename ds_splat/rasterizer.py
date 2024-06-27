#!/usr/bin/env python3
from dataclasses import dataclass
import torch
import ds_splat_cuda as _cuda_impl


class DsRasterizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means_3d,
        means_2d,
        shs,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov_3d_precomp,
        raster_settings,
    ):

        outputs = _cuda_impl.forward_deepsense(
            means_3d,
            means_2d,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov_3d_precomp,
            raster_settings,
        )

        (
            rendered_image,
            radii,
            gaussian_tensor,
            final_Ts,
            final_idx,
            clamped_colors,
            cov_3d,
            indices,
            tile_indices,
        ) = outputs

        ctx.raster_settings = raster_settings
        ctx.save_for_backward(
            means_3d,
            means_2d,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov_3d_precomp,
            radii,
            final_Ts,
            final_idx,
            clamped_colors,
            gaussian_tensor,
            cov_3d,
            indices,
            tile_indices,
        )

        return rendered_image, radii

    @staticmethod
    def backward(ctx, rendered_img_grad, _):
        (
            means_3d,
            means_2d,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov_3d_precomp,
            radii,
            final_Ts,
            final_idx,
            clamped_colors,
            gaussian_tensor,
            cov_3d,
            indices,
            tile_indices,
        ) = ctx.saved_tensors

        raster_settings = ctx.raster_settings

        output = _cuda_impl.backward_deepsense(
            means_3d,
            means_2d,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov_3d_precomp,
            gaussian_tensor,
            radii,
            clamped_colors,
            final_Ts,
            final_idx,
            cov_3d,
            indices,
            tile_indices,
            rendered_img_grad,
            raster_settings,
        )

        return (*output, None)


@dataclass
class GaussianRasterizationSettings:
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    max_sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(torch.nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super(GaussianRasterizer, self).__init__()
        self._settings = raster_settings

    @property
    def settings_for_cpp_code(self):
        settings = _cuda_impl.RasterizationSettings()

        settings.image_height = self._settings.image_height
        settings.image_width = self._settings.image_width
        settings.tanfovx = self._settings.tanfovx
        settings.tanfovy = self._settings.tanfovy
        settings.bg = self._settings.bg
        settings.scale_modifier = self._settings.scale_modifier
        settings.view_matrix = self._settings.viewmatrix.t().contiguous()
        settings.proj_matrix = self._settings.projmatrix.t().contiguous()
        settings.sh_degree = self._settings.sh_degree
        settings.max_sh_degree = self._settings.max_sh_degree
        settings.campos = self._settings.campos
        settings.prefiltered = self._settings.prefiltered
        settings.debug = self._settings.debug

        return settings

    def forward(
        self,
        means_3d: torch.Tensor,
        means_2d: torch.Tensor,
        shs: torch.Tensor,
        colors_precomp: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov_3d_precomp: torch.Tensor,
    ):

        if cov_3d_precomp is None:
            cov_3d_precomp = torch.Tensor().cuda()

        if colors_precomp is None:
            colors_precomp = torch.Tensor().cuda()

        if scales is None:
            scales = torch.Tensor().cuda()

        if rotations is None:
            rotations = torch.Tensor().cuda()

        if shs is None:
            shs = torch.Tensor().cuda()

        return DsRasterizerFunction.apply(
            means_3d,
            means_2d,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov_3d_precomp,
            self.settings_for_cpp_code,
        )
