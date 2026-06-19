"""Shared helpers for the test suite (not collected by pytest)."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# On the GPU path fp16/fp32 exercise the Triton kernels; on a CPU box the
# autograd.Function transparently falls back to the PyTorch FA-2 mirror, so the
# fp32 / float64 tests still run and still prove the backward math.
HAS_CUDA = torch.cuda.is_available()


def sm_scale(d):
    return 1.0 / math.sqrt(d)


def make_qkv(B, H, N, d, dtype, device=DEVICE, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    kw = dict(dtype=dtype, device=device, requires_grad=requires_grad)
    q = torch.randn(B, H, N, d, **kw)
    k = torch.randn(B, H, N, d, **kw)
    v = torch.randn(B, H, N, d, **kw)
    return q, k, v
