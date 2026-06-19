"""Stage 5: render the two README figures from the saved CSVs.

  figures/memory_vs_seqlen.png   -- peak VRAM vs N (log-y); reference OOM points
                                    are dropped (the line simply ends).
  figures/latency_vs_seqlen.png  -- forward latency (ms) vs N (log-log).

Run after the benchmarks:  python benchmarks/plot_results.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
MEM_CSV = os.path.join(HERE, "results_memory.csv")
LAT_CSV = os.path.join(HERE, "results_latency.csv")


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None  # "OOM" / "nan" / empty


def plot_memory(plt):
    if not os.path.exists(MEM_CSV):
        print(f"[skip] {MEM_CSV} not found")
        return
    with open(MEM_CSV) as f:
        rows = list(csv.DictReader(f))
    impls = [c for c in rows[0].keys() if c != "N"]

    plt.figure(figsize=(7, 5))
    for impl in impls:
        xs, ys = [], []
        for r in rows:
            y = _to_float(r[impl])
            if y is not None:
                xs.append(int(r["N"]))
                ys.append(y)
        if xs:
            plt.plot(xs, ys, marker="o", label=impl)
    plt.yscale("log")
    plt.xlabel("sequence length N")
    plt.ylabel("peak VRAM (MB, log scale)")
    plt.title("Peak memory vs sequence length\n(reference OOMs at large N -> line ends)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    out = os.path.join(FIG_DIR, "memory_vs_seqlen.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved {out}")


def plot_latency(plt):
    if not os.path.exists(LAT_CSV):
        print(f"[skip] {LAT_CSV} not found")
        return
    with open(LAT_CSV) as f:
        rows = list(csv.DictReader(f))

    by_impl = {}
    for r in rows:
        ms = _to_float(r["ms"])
        by_impl.setdefault(r["impl"], {"x": [], "y": []})
        if ms is not None:
            by_impl[r["impl"]]["x"].append(int(r["N"]))
            by_impl[r["impl"]]["y"].append(ms)

    plt.figure(figsize=(7, 5))
    for impl, d in by_impl.items():
        if d["x"]:
            plt.plot(d["x"], d["y"], marker="o", label=impl)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length N (log2)")
    plt.ylabel("forward latency (ms, log scale)")
    plt.title("Forward latency vs sequence length")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    out = os.path.join(FIG_DIR, "latency_vs_seqlen.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved {out}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_memory(plt)
    plot_latency(plt)


if __name__ == "__main__":
    main()
