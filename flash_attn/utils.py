"""Shared utilities: timing, VRAM probes, seeding, plotting.

Kept dependency-light so the package still imports on a CPU-only box; matplotlib
is imported lazily inside the plotting helpers.
"""

import contextlib
import statistics

import torch


def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_time_ms(fn, warmup=10, iters=30):
    """Median wall time (ms) of ``fn`` using CUDA events.

    Warms up, then records ``iters`` runs and returns the median to be robust to
    occasional scheduling jitter. Requires CUDA.
    """
    assert torch.cuda.is_available(), "cuda_time_ms requires a GPU"
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times)


@contextlib.contextmanager
def peak_vram():
    """Context manager yielding a 1-element list that holds peak bytes on exit."""
    assert torch.cuda.is_available(), "peak_vram requires a GPU"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    holder = [0]
    try:
        yield holder
    finally:
        torch.cuda.synchronize()
        holder[0] = torch.cuda.max_memory_allocated()


def bytes_to_mb(n):
    return n / (1024 ** 2)


def attention_tokens_per_s(ms, B, H, N):
    """tokens/s = (batch * seqlen) processed per second (heads share the seq)."""
    if ms <= 0:
        return float("nan")
    tokens = B * N
    return tokens / (ms / 1000.0)


def ensure_dir(path):
    import os

    os.makedirs(path, exist_ok=True)
    return path
