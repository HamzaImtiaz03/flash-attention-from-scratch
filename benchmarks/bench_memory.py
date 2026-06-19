"""Stage 5: peak VRAM vs sequence length -- our O(N) memory vs reference O(N^2).

Measures torch.cuda.max_memory_allocated() for a forward pass of each impl across
N = [512, 1024, 2048, 4096, 8192]. The PyTorch reference materializes the full
N x N score matrix and WILL OOM at large N on the T4's 16 GB -- we catch that and
record it as OOM rather than pretending. Results -> results_memory.csv.

Run on Colab:  python benchmarks/bench_memory.py
"""

import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.reference import attention as reference_attention
from flash_attn.triton_attn import TritonAttention
from flash_attn.utils import bytes_to_mb, peak_vram, set_seed

SEQLENS = [512, 1024, 2048, 4096, 8192]
B, H, D = 1, 8, 64
DTYPE = torch.float16
CAUSAL = False
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_memory.csv")


def _try_cuda_impl():
    """Return CudaAttention.apply or None if the extension can't be built."""
    try:
        from flash_attn.cuda_attn import CudaAttention

        # Force a JIT compile up front so build cost isn't charged to the probe.
        q = torch.randn(1, 1, 16, D, device="cuda", dtype=DTYPE)
        CudaAttention.apply(q, q, q, False, 1.0)
        return CudaAttention.apply
    except Exception as e:  # pragma: no cover
        print(f"[warn] CUDA impl unavailable, skipping: {e}")
        return None


def measure(fn, N):
    """Peak MB for one forward of fn, or 'OOM' string on out-of-memory."""
    sm_scale = 1.0 / (D ** 0.5)
    try:
        q = torch.randn(B, H, N, D, device="cuda", dtype=DTYPE)
        k = torch.randn(B, H, N, D, device="cuda", dtype=DTYPE)
        v = torch.randn(B, H, N, D, device="cuda", dtype=DTYPE)
        with peak_vram() as peak:
            out = fn(q, k, v, CAUSAL, sm_scale)
        del q, k, v, out
        return round(bytes_to_mb(peak[0]), 2)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return "OOM"
        raise
    finally:
        torch.cuda.empty_cache()


def main():
    assert torch.cuda.is_available(), "benchmarks require a CUDA GPU (T4)"
    set_seed(0)
    cuda_fn = _try_cuda_impl()

    impls = [("reference", reference_attention), ("triton", TritonAttention.apply)]
    if cuda_fn is not None:
        impls.append(("cuda", cuda_fn))

    rows = []
    for N in SEQLENS:
        row = {"N": N}
        for name, fn in impls:
            mb = measure(fn, N)
            row[name] = mb
            print(f"N={N:>5}  {name:>9}: {mb} MB")
        rows.append(row)

    fields = ["N"] + [name for name, _ in impls]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
