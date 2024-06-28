#!/usr/bin/env python3

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_extension():
    extra_compile_args = {"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math"]}

    return CUDAExtension(
        "ds_splat_cuda",
        sources=[
            "cpp/rasterizer_python.cpp",
            "cpp/rasterizer_torch.cu",
            "cpp/rasterizer_cuda.cu",
            "cpp/rasterizer_kernels.cu",
            "cpp/gsplat/backward.cu",
            "cpp/gsplat/bindings.cu",
        ],
        extra_compile_args=extra_compile_args,
    )


setup(
    name="ds-splat",
    ext_modules=[get_cuda_extension()],
       cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
)
