"""Stage 3: backward correctness — the headline proof.

(a) torch.autograd.gradcheck in FLOAT64 on the TritonAttention function with
    small tensors. This is the primary proof of the backward pass. float64 has
    no tensor-core MMA, so the autograd.Function runs its PyTorch FA-2 mirror,
    which implements the EXACT formulas the Triton kernels use. Runs on CPU too.

(b) allclose of dQ, dK, dV against autograd of the O(N^2) reference, in fp32.
    On CUDA this checks the Triton backward kernels; on CPU it checks the mirror.
"""

import pytest
import torch

from flash_attn.reference import attention as reference_attention
from flash_attn.triton_attn import TritonAttention

from _common import DEVICE, HAS_CUDA, make_qkv, sm_scale


@pytest.mark.parametrize("causal", [False, True])
def test_gradcheck_float64(causal):
    # Small tensors as specified: B=1, H=2, N=16, d=16.
    B, H, N, d = 1, 2, 16, 16
    q, k, v = make_qkv(B, H, N, d, torch.float64, requires_grad=True)
    scale = sm_scale(d)
    func = lambda q, k, v: TritonAttention.apply(q, k, v, causal, scale)
    assert torch.autograd.gradcheck(func, (q, k, v), eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("B,H,N,d", [(1, 2, 128, 64), (2, 3, 256, 32)])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_allclose_fp32(B, H, N, d, causal):
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
