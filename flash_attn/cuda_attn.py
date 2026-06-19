"""Python wrapper for the hand-written CUDA FlashAttention (Stage 4).

JIT-compiles csrc/ via torch.utils.cpp_extension.load() (the PRIMARY build path
so it works inside Colab with no separate build step) and exposes it through a
``CudaAttention`` autograd.Function.

Forward  -> hand-written CUDA kernel (csrc/flash_fwd_kernel.cu).
Backward -> the already-proven Triton backward (or the PyTorch FA-2 mirror if
            Triton is unavailable). The CUDA backward is intentionally not
            implemented; see csrc/flash_bwd_kernel.cu and the README.
"""

import math
import os

import torch

_EXT = None  # cached compiled extension


def _csrc_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")


def get_extension():
    """JIT-compile (once) and return the CUDA extension module."""
    global _EXT
    if _EXT is not None:
        return _EXT

    from torch.utils.cpp_extension import load

    csrc = _csrc_dir()
    _EXT = load(
        name="flash_attn_cuda",
        sources=[
            os.path.join(csrc, "flash_attn.cpp"),
            os.path.join(csrc, "flash_fwd_kernel.cu"),
        ],
        extra_include_paths=[csrc],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            # Target the T4 explicitly (Turing, sm_75).
            "-gencode", "arch=compute_75,code=sm_75",
            # Enable __half arithmetic/conversions in device code.
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _EXT


class CudaAttention(torch.autograd.Function):
    """FlashAttention with a hand-written CUDA forward and (reused) FA-2 backward.

    ``CudaAttention.apply(q, k, v, causal, sm_scale)`` with q/k/v of shape
    (B, H, N, d), fp16 or fp32, on a CUDA device.
    """

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        ext = get_extension()
        q, k, v = (x.contiguous() for x in (q, k, v))
        o, L = ext.fwd(q, k, v, causal, float(sm_scale))
        ctx.save_for_backward(q, k, v, o, L)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        from .triton_attn import HAS_TRITON, _triton_backward, _fa2_backward_torch

        if HAS_TRITON and q.is_cuda and q.dtype in (torch.float16, torch.float32):
            dq, dk, dv = _triton_backward(q, k, v, o, L, do, ctx.causal, ctx.sm_scale)
        else:
            dq, dk, dv = _fa2_backward_torch(q, k, v, o, L, do, ctx.causal, ctx.sm_scale)
        return dq, dk, dv, None, None


def cuda_attention(q, k, v, causal=False, sm_scale=None):
    """Convenience functional wrapper. Defaults sm_scale to 1/sqrt(d)."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return CudaAttention.apply(q, k, v, causal, sm_scale)
