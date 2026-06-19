"""Stage 3: forward correctness vs the O(N^2) reference.

fp32: atol=1e-2, rtol=0   |   fp16: atol=2e-2   (tolerances from the spec).
On CUDA these exercise the Triton kernels; on CPU the autograd.Function falls
back to the PyTorch FA-2 mirror (fp16 is CUDA-only and is skipped on CPU).
"""

import pytest
import torch

from flash_attn.reference import attention as reference_attention
from flash_attn.triton_attn import TritonAttention

from _common import DEVICE, HAS_CUDA, make_qkv, sm_scale

SHAPES = [
    (1, 2, 128, 64),
    (2, 4, 256, 64),
    (1, 8, 512, 32),
]


@pytest.mark.parametrize("B,H,N,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_forward_fp32(B, H, N, d, causal):
    q, k, v = make_qkv(B, H, N, d, torch.float32)
    scale = sm_scale(d)
    out = TritonAttention.apply(q, k, v, causal, scale)
    ref = reference_attention(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)


@pytest.mark.skipif(not HAS_CUDA, reason="fp16 path requires a CUDA GPU (T4)")
@pytest.mark.parametrize("B,H,N,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_forward_fp16(B, H, N, d, causal):
    q, k, v = make_qkv(B, H, N, d, torch.float16)
    scale = sm_scale(d)
    out = TritonAttention.apply(q, k, v, causal, scale)
    ref = reference_attention(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)
