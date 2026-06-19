"""Stage 4 validation: the hand-written CUDA path against the same reference.

CUDA-only -- skipped entirely on a machine without a GPU. On Colab's T4 these
compile csrc/ via JIT and check forward (fp32 + fp16) and gradients (fp32)
against the O(N^2) reference, exactly like the Triton tests.
"""

import pytest
import torch

from _common import HAS_CUDA, make_qkv, sm_scale

pytestmark = pytest.mark.skipif(not HAS_CUDA, reason="CUDA path requires a GPU (T4)")

if HAS_CUDA:
    from flash_attn.reference import attention as reference_attention
    from flash_attn.cuda_attn import CudaAttention

SHAPES = [(1, 2, 128, 64), (2, 4, 256, 32), (2, 3, 200, 128)]  # incl. non-pow2 N, d=128


@pytest.mark.parametrize("B,H,N,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_forward_fp32(B, H, N, d, causal):
    q, k, v = make_qkv(B, H, N, d, torch.float32, device="cuda")
    scale = sm_scale(d)
    out = CudaAttention.apply(q, k, v, causal, scale)
    ref = reference_attention(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)


@pytest.mark.parametrize("B,H,N,d", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_forward_fp16(B, H, N, d, causal):
    q, k, v = make_qkv(B, H, N, d, torch.float16, device="cuda")
    scale = sm_scale(d)
    out = CudaAttention.apply(q, k, v, causal, scale)
    ref = reference_attention(q, k, v, causal, scale)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@pytest.mark.parametrize("B,H,N,d", [(1, 2, 128, 64), (2, 2, 257, 32)])
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_backward_fp32(B, H, N, d, causal):
    scale = sm_scale(d)
    q, k, v = make_qkv(B, H, N, d, torch.float32, device="cuda", requires_grad=True)
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)

    out = CudaAttention.apply(q, k, v, causal, scale)
    ref = reference_attention(qr, kr, vr, causal, scale)
    g = torch.randn_like(out)
    out.backward(g)
    ref.backward(g)

    torch.testing.assert_close(q.grad, qr.grad, atol=2e-2, rtol=0)
    torch.testing.assert_close(k.grad, kr.grad, atol=2e-2, rtol=0)
    torch.testing.assert_close(v.grad, vr.grad, atol=2e-2, rtol=0)
