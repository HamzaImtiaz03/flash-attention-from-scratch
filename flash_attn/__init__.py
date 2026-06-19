"""flash_attn: a from-scratch FlashAttention-2 (forward + backward).

Public API:
    reference_attention(q, k, v, causal, sm_scale) -> O(N^2) ground truth
    triton_attention(q, k, v, causal, sm_scale)    -> Triton FA-2 (autograd)
    cuda_attention(q, k, v, causal, sm_scale)      -> hand-written CUDA forward

The Triton and CUDA entry points require a CUDA GPU (and, for CUDA, a JIT
compile of csrc/). They are imported lazily so that `import flash_attn` works in
a CPU-only environment for the reference path and the test harness — the GPU
paths simply raise a clear error if you call them without the dependencies.
"""

from .reference import attention as reference_attention

__all__ = ["reference_attention", "triton_attention", "cuda_attention"]


def triton_attention(q, k, v, causal, sm_scale):
    """Lazy wrapper around the Triton implementation (Stage 2)."""
    from .triton_attn import TritonAttention

    return TritonAttention.apply(q, k, v, causal, sm_scale)


def cuda_attention(q, k, v, causal, sm_scale):
    """Lazy wrapper around the hand-written CUDA implementation (Stage 4)."""
    from .cuda_attn import CudaAttention

    return CudaAttention.apply(q, k, v, causal, sm_scale)
