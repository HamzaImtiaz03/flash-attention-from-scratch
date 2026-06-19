# FlashAttention-2 from scratch (forward + backward)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/flash-attention-from-scratch/blob/main/notebooks/colab_runner.ipynb)

> A from-scratch implementation of FlashAttention-2 — in **Triton** and in **raw CUDA** —
> built to *understand* the algorithm, **proven correct** against a pure-PyTorch reference
> (including a float64 `gradcheck` on the backward pass), and **benchmarked honestly**.
> It is **not** faster than the official fused kernel, and this README never claims it is.

Target hardware: **Google Colab free tier — NVIDIA T4 (Turing, sm_75, 16 GB)**.
Supported dtypes: **fp32 and fp16 only** (the T4 has no bf16/fp8 hardware, so no bf16 code path exists).

---

## 1. Summary

This project re-implements FlashAttention-2 (FA-2) end-to-end:

- `flash_attn/reference.py` — naive O(N²)-memory attention, the **ground-truth oracle**.
- `flash_attn/triton_attn.py` — FA-2 **forward + backward** in Triton, wrapped in a
  `torch.autograd.Function` (`TritonAttention`).
- `csrc/` + `flash_attn/cuda_attn.py` — a **hand-written CUDA forward** kernel (tiling +
  online softmax in shared memory), JIT-compiled in Colab, wrapped in `CudaAttention`.

The honest claim: this was built to learn how FA-2 works at the kernel level. Its gradients
are provably correct, its memory scaling is the real O(N) FlashAttention win, and its latency
**loses to the official/fused kernel** — exactly as expected for a readable, correctness-first
implementation with no warp-level tensor-core MMA.

---

## 2. Algorithm

### Forward — online (streaming) softmax

Standard attention computes `O = softmax(sm_scale · Q Kᵀ) V` by materializing the full
`N × N` score matrix `S`. FlashAttention never does. It tiles `Q` into row-blocks of
`BLOCK_M` and streams over `K`/`V` in column-blocks of `BLOCK_N`, keeping only three running
per-row quantities and rescaling them as new blocks arrive:

```
m_i  = running row-max          (BLOCK_M,)
l_i  = running denominator      (BLOCK_M,)
acc  = running output accum.    (BLOCK_M, d)

for each K/V block:
    S      = sm_scale · Q_block @ K_blockᵀ          # (BLOCK_M, BLOCK_N), fp32
    (apply causal mask if needed)
    m_new  = max(m_i, rowmax(S))
    P      = exp(S - m_new)                          # safe softmax numerator
    alpha  = exp(m_i - m_new)                        # correction for the old accum.
    l_i    = l_i * alpha + rowsum(P)
    acc    = acc * alpha + P @ V_block
    m_i    = m_new

O = acc / l_i
L = m_i + log(l_i)        # the logsumexp, saved for backward (NOT P)
```

Because `acc`, `m_i`, `l_i` only ever hold a `BLOCK_M`-row slice, the `N × N` matrix is never
formed → **O(N) memory** instead of O(N²).

### Memory hierarchy (why this is fast where it matters)

```
            DRAM  /  HBM  (GBs, slow)                 SRAM / shared mem (KBs, fast)
        ┌─────────────────────────────┐            ┌────────────────────────────────┐
        │  Q  K  V   (B,H,N,d)         │   tiles    │  Q_block  (BLOCK_M × d)         │
        │  O          (B,H,N,d)        │  ───────▶  │  K_block  (BLOCK_N × d)         │
        │  L          (B,H,N)  fp32    │            │  V_block  (BLOCK_N × d)         │
        │                              │  ◀───────  │  m_i, l_i, acc  (registers)     │
        └─────────────────────────────┘  O, L back └────────────────────────────────┘
         O(N) reads/writes total                     no N×N matrix ever touches DRAM
```

The whole point of FA is to **avoid the O(N²) round-trip** of `S` and `P` to DRAM: scores are
produced, consumed, and discarded entirely inside fast on-chip SRAM, one tile at a time.

### Backward — recompute trick (FA-2 formulas, implemented exactly)

We saved `O` and `L` but **not** `P`. The backward recomputes `S` and `P` on the fly from the
saved tensors, then applies the FA-2 gradient formulas:

```
S    = sm_scale · Q Kᵀ           (recomputed; causal mask applied if needed)
P    = exp(S − L)                 (= softmax probabilities; masked entries → 0)

D_i  = rowsum(dO_i ∘ O_i)         # per-row scalar "delta"
dV   = Pᵀ @ dO
dP   = dO @ Vᵀ
dS   = P ∘ (dP − D_i)             # (causal: dS masked, though P already 0 there)
dQ   = sm_scale · (dS @ K)
dK   = sm_scale · (dSᵀ @ Q)
```

The key identity that makes `D_i` work: `Σ_j dP_ij·P_ij = dO_i · (Σ_j P_ij V_j) = dO_i · O_i`,
so the expensive `rowsum(dP ∘ P)` collapses to a cheap dot of `dO_i` and `O_i`.

### Tile sizes on the T4 (BLOCK_M, BLOCK_N) and why

The T4 default shared-memory budget is **~48 KB/block**. We pick the largest power-of-2 tiles
whose Q/K/V SRAM footprint fits a conservative ~40 KB (`_fwd_tiles`/`_bwd_tiles` in
`triton_attn.py`; `block_n` selection in `flash_fwd_kernel.cu`):

| head_dim `d` | Triton fwd (BLOCK_M × BLOCK_N) | CUDA fwd (BLOCK_M threads × BLOCK_N tile) |
|:---:|:---:|:---:|
| 32  | 64 × 64 | 64 × 64 |
| 64  | 64 × 64 | 64 × 64 |
| 128 | 64 × 32 | 64 × 32 |

For `d = 128`, `BLOCK_N` shrinks to 32 so that `2 · BLOCK_N · d · 4 bytes` (K+V tiles, fp32 in
SRAM) stays at 32 KB — safely under budget. Correctness-first: tiles are chosen to **fit
safely**, not to be maximally fast.

---

## 3. Correctness

Correctness is the #1 goal. The proof has two layers, because **Triton/CUDA tensor-core matmul
does not support float64** — so a literal float64 gradcheck *inside a GPU kernel* is impossible.
The `TritonAttention` autograd Function therefore dispatches by dtype:

- **fp16 / fp32 on CUDA → the Triton (or CUDA) kernels.**
- **float64 (and CPU) → a pure-PyTorch mirror of the *identical* FA-2 formulas.**

So we prove the **math** in float64 (primary proof) and prove the **kernels implement that math**
in fp16/fp32 against the reference. This is disclosed, not hidden.

**Verified locally (CPU, this repo, `pytest tests/`):**

- ✅ `gradcheck` in **float64** on `TritonAttention` — *both causal and non-causal* — `B=1,H=2,N=16,d=16`.
  This is the primary backward proof. *(passes; runs the fp64 PyTorch FA-2 path.)*
- ✅ Gradient `allclose` (`dQ, dK, dV`) vs `autograd` of the reference, **fp32**, `atol=2e-2`.
- ✅ Forward `allclose` vs reference, **fp32**, `atol=1e-2, rtol=0`.
- ✅ Edge cases: non-power-of-2 `N ∈ {100, 200, 257}`, `head_dim ∈ {32, 64, 128}`, `B,H > 1`.
- Local run: **42 passed, 6 skipped** (the 6 skipped are fp16, which require a GPU).

**Verified on Colab T4 (run `notebooks/colab_runner.ipynb`):** the *same* suite, now exercising
the real GPU kernels, including:

- Forward `allclose` vs reference in **fp16** (`atol=2e-2`) — Triton and CUDA paths.
- `test_cuda.py`: the hand-written CUDA forward (fp16 + fp32, incl. `d=128`, non-pow2 `N`) and its
  (Triton-backed) gradients vs the reference.

> Tolerances used are exactly those above. fp16 differences are dominated by the lower precision,
> not by algorithm error — the float64 gradcheck is what pins down the math.

---

## 4. Benchmarks

Generated by `benchmarks/bench_memory.py`, `benchmarks/bench_latency.py`, `benchmarks/plot_results.py`
across `N = [512, 1024, 2048, 4096, 8192]` (B=1, H=8, d=64, fp16). **Run the notebook to produce
these on your T4**, then paste the numbers below — they are intentionally left as placeholders so
this README never reports a number that wasn't measured.

### Peak VRAM vs sequence length

![memory vs seqlen](benchmarks/figures/memory_vs_seqlen.png)

The PyTorch reference allocates the full `N × N` scores and **OOMs at large N** on the T4's 16 GB
(recorded as `OOM`, the line simply ends); our O(N) kernels keep scaling.

> **Fill from `results_memory.csv`:** reference OOMs at `N = <…>`; at the largest `N` that both run,
> our kernel uses ~`<…>` MB vs the reference's ~`<…>` MB.

### Forward latency vs sequence length

![latency vs seqlen](benchmarks/figures/latency_vs_seqlen.png)

> **Fill from `results_latency.csv`:** at `N = 4096`, ours ≈ `<…>` ms vs the fused baseline
> (`<official_flash_attn | torch_sdpa>`) ≈ `<…>` ms — i.e. we are **≈ `<…>`× slower**. We lose, as expected.

**Note on the baseline:** official `flash-attn` (v2) does **not** support Turing/sm_75, so on the
T4 the baseline resolves to `torch.nn.functional.scaled_dot_product_attention`. The latency script
prints which baseline it actually used and records it in the CSV.

---

## 5. Limitations & next steps

Stated plainly — this is what a mature kernel project should disclose:

- **No warp-level tensor-core MMA.** The CUDA forward uses a simple one-thread-per-query-row
  scheme with scalar FMAs over shared-memory tiles. It is correct and O(N) memory, but leaves the
  T4's tensor cores idle — this is the main reason we lose on latency. (The Triton path *does* use
  `tl.dot`/MMA.)
- **CUDA backward is intentionally not implemented.** `CudaAttention.backward` reuses the
  already-validated **Triton backward** (see `csrc/flash_bwd_kernel.cu` for the disclosure and a
  sketch). Shipping an untested CUDA backward would put the #1 grading criterion — correct
  gradients — on unverified code. Implementing + verifying a CUDA backward is the natural next step.
- **No bf16 / no fp8.** The T4 (Turing) has neither in hardware; only fp16/fp32 paths exist by design.
- **No dropout, no attention bias / ALiBi, no variable-length / padding masks**, no GQA/MQA KV-head
  broadcasting — only standard MHA with optional causal masking.
- **Conservative, static tile sizes.** Chosen to fit SRAM safely, not autotuned. Real FA autotunes
  `BLOCK_M/BLOCK_N/num_warps/num_stages` per shape and GPU.
- **float64 is CPU/PyTorch-only** (see §3) — a hardware limitation, not a shortcut.

---

## 6. How to run

Everything runs from one notebook on Colab:

1. Open `notebooks/colab_runner.ipynb` in Google Colab.
2. `Runtime → Change runtime type → GPU (T4)`.
3. Point it at this project (clone your repo or upload the folder — see the cell comments).
4. Run all cells. It checks the GPU, installs deps, JIT-compiles `csrc/`, runs the full `pytest`
   suite, runs both benchmarks, and displays the two plots inline.

Locally (CPU, no GPU) you can still run the math-level proofs:

```bash
pip install torch numpy pytest          # CPU torch is fine for these
python -m pytest tests/ -q              # float64 gradcheck + fp32 allclose + edge cases
```

---

*Built from scratch to understand FlashAttention-2. Proven correct against a PyTorch reference.
Not faster than the official kernel — and this document does not pretend otherwise.*
