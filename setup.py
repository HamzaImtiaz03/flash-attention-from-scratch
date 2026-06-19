"""OPTIONAL ahead-of-time build of the CUDA extension.

The PRIMARY build path is JIT via torch.utils.cpp_extension.load() (see
flash_attn/cuda_attn.py), which is what the Colab notebook uses -- no separate
build step required. This setup.py is provided only as an alternative for users
who want a pre-compiled wheel / in-place build:

    pip install -e .            # or:  python setup.py build_ext --inplace

It targets the T4 (Turing, sm_75) and supports fp16 + fp32 only.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flash_attn_cuda",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_cuda",
            sources=[
                "csrc/flash_attn.cpp",
                "csrc/flash_fwd_kernel.cu",
            ],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", "arch=compute_75,code=sm_75",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
