"""Stage 1 local sanity check (CPU-friendly).

Validates flash_attn.reference.attention against torch's own
F.scaled_dot_product_attention, which is an independent oracle. This lets us
prove the reference is correct on a machine with no GPU before we build the
GPU kernels that will be compared against it.

Run: python tests/_sanity_reference.py
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from flash_attn.reference import attention as reference_attention


def run_case(B, H, N, d, causal, dtype):
    torch.manual_seed(0)
    q = torch.randn(B, H, N, d, dtype=dtype)
    k = torch.randn(B, H, N, d, dtype=dtype)
    v = torch.randn(B, H, N, d, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(d)

    got = reference_attention(q, k, v, causal, sm_scale)
    # torch SDPA uses scale=sm_scale and is_causal for the mask.
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)

    atol = 1e-5 if dtype == torch.float32 else 2e-2
    ok = torch.allclose(got, ref, atol=atol, rtol=0)
    max_err = (got - ref).abs().max().item()
    tag = f"B{B} H{H} N{N} d{d} causal={causal!s:<5} {str(dtype).split('.')[-1]:>7}"
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}  max_abs_err={max_err:.3e}  (atol={atol})")
    return ok


def main():
    cases = [
        (1, 2, 16, 16, False, torch.float32),
        (1, 2, 16, 16, True, torch.float32),
        (2, 3, 128, 64, False, torch.float32),
        (2, 3, 128, 64, True, torch.float32),
        (1, 4, 200, 32, True, torch.float32),  # non-power-of-2
        (1, 4, 257, 128, False, torch.float32),  # non-power-of-2, d=128
        (1, 2, 64, 64, False, torch.float16),
        (1, 2, 64, 64, True, torch.float16),
    ]
    all_ok = all(run_case(*c) for c in cases)
    print("\nRESULT:", "ALL PASSED" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
