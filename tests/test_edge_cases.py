"""Stage 3: edge cases — non-power-of-2 seqlen, head_dim in {32,64,128}, B,H>1.

All compared (forward + gradients) against the O(N^2) reference. fp32 runs
everywhere; on CUDA these drive the Triton kernels through their boundary
masking, on CPU they drive the PyTorch FA-2 mirror.
"""

import itertools

import pytest
import torch

from flash_attn.reference import attention as reference_attention
from flash_attn.triton_attn import TritonAttention

from _common import HAS_CUDA, make_qkv, sm_scale

NON_POW2_N = [200, 257, 100]
HEAD_DIMS = [32, 64, 128]


@pytest.mark.parametrize("N", NON_POW2_N)
@pytest.mark.parametrize("d", HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True])
def test_edge_forward(N, d, causal):
    B, H = 2, 3  # B, H > 1
    scale = sm_scale(d)
    q, k, v = make_qkv(B, H, N, d, torch.float32)
    out = TritonAttention.apply(q, k, v, causal, scale)
    ref = reference_attention(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)


@pytest.mark.parametrize("N", [200, 257])
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_edge_backward(N, d, causal):
    B, H = 2, 2
    scale = sm_scale(d)
    q, k, v = make_qkv(B, H, N, d, torch.float32, requires_grad=True)
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)

    out = TritonAttention.apply(q, k, v, causal, scale)
    ref = reference_attention(qr, kr, vr, causal, scale)

    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)

    torch.testing.assert_close(q.grad, qr.grad, atol=2e-2, rtol=0)
    torch.testing.assert_close(k.grad, kr.grad, atol=2e-2, rtol=0)
    torch.testing.assert_close(v.grad, vr.grad, atol=2e-2, rtol=0)
