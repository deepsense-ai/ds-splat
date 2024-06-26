#!/usr/bin/env python3

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from __version__ import __version__


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
    name="ds_splat",
    author="deepsense.ai",
    author_email="contact@deepsense.ai",
    version=__version__,
    description="A CUDA-based gaussian splatting rasterizer extension for PyTorch.",
    readme="README.md",
    python_requires=">=3.7",
    keywords=["pytorch", "cuda", "rasterizer", "deep learning"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=[get_cuda_extension()],
    install_requires=[
        "torch",
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=find_packages(),
    include_package_data=True,
)
