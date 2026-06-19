"""Stage 5: forward latency (ms) & tokens/s vs an official/SDPA baseline.

For each N we time a forward pass with CUDA events (warmup + median of 30) for:
  - reference  (PyTorch O(N^2); OOM at large N -> recorded as NaN)
  - triton     (ours)
  - cuda       (ours, if the extension builds)
  - baseline   (official flash-attn if importable & it runs on this GPU,
                otherwise torch.nn.functional.scaled_dot_product_attention)

We WILL lose to the official/SDPA fused kernel. That is expected and the point
of honest reporting. Results -> results_latency.csv (long format).

Note: official flash-attn (v2) does not support Turing/sm_75, so on the T4 the
baseline almost always resolves to torch SDPA. The script reports which one it
actually used.

Run on Colab:  python benchmarks/bench_latency.py
"""

import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_attn.reference import attention as reference_attention
from flash_attn.triton_attn import TritonAttention
from flash_attn.utils import attention_tokens_per_s, cuda_time_ms, set_seed

SEQLENS = [512, 1024, 2048, 4096, 8192]
B, H, D = 1, 8, 64
DTYPE = torch.float16
CAUSAL = False
WARMUP, ITERS = 10, 30
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_latency.csv")


def _baseline():
    """Return (name, fn(q,k,v,causal,scale)) for the strongest available baseline."""
    try:
        import flash_attn as official  # noqa: F401
        from flash_attn import flash_attn_func

        def run(q, k, v, causal, scale):
            # official expects (B, N, H, d); we store (B, H, N, d).
            qt, kt, vt = (x.transpose(1, 2).contiguous() for x in (q, k, v))
            return flash_attn_func(qt, kt, vt, causal=causal, softmax_scale=scale)

        # Probe once; Turing will raise here and we fall through to SDPA.
        probe = torch.randn(1, 2, 16, D, device="cuda", dtype=DTYPE)
        run(probe, probe, probe, False, 1.0)
        return "official_flash_attn", run
    except Exception as e:
        print(f"[info] official flash-attn unavailable on this GPU ({e}); using torch SDPA.")

    def sdpa(q, k, v, causal, scale):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    return "torch_sdpa", sdpa


def _cuda_impl():
    try:
        from flash_attn.cuda_attn import CudaAttention

        q = torch.randn(1, 1, 16, D, device="cuda", dtype=DTYPE)
        CudaAttention.apply(q, q, q, False, 1.0)
        return CudaAttention.apply
    except Exception as e:  # pragma: no cover
        print(f"[warn] CUDA impl unavailable, skipping: {e}")
        return None


def time_impl(fn, N):
    sm_scale = 1.0 / (D ** 0.5)
    try:
        q = torch.randn(B, H, N, D, device="cuda", dtype=DTYPE)
        k = torch.randn(B, H, N, D, device="cuda", dtype=DTYPE)
        v = torch.randn(B, H, N, D, device="cuda", dtype=DTYPE)
        ms = cuda_time_ms(lambda: fn(q, k, v, CAUSAL, sm_scale), WARMUP, ITERS)
        del q, k, v
        return ms
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return float("nan")
        raise
    finally:
        torch.cuda.empty_cache()


def main():
    assert torch.cuda.is_available(), "benchmarks require a CUDA GPU (T4)"
    set_seed(0)

    base_name, base_fn = _baseline()
    cuda_fn = _cuda_impl()

    impls = [("reference", reference_attention), ("triton", TritonAttention.apply)]
    if cuda_fn is not None:
        impls.append(("cuda", cuda_fn))
    impls.append((base_name, base_fn))

    rows = []
    for N in SEQLENS:
        for name, fn in impls:
            ms = time_impl(fn, N)
            tps = attention_tokens_per_s(ms, B, H, N) if ms == ms else float("nan")
            rows.append({"N": N, "impl": name, "ms": round(ms, 4) if ms == ms else "nan",
                         "tokens_per_s": round(tps, 1) if tps == tps else "nan"})
            print(f"N={N:>5}  {name:>20}: {ms if ms==ms else 'OOM':>10} ms")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "impl", "ms", "tokens_per_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nBaseline used: {base_name}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
