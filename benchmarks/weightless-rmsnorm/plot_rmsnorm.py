"""Visualize weightless RMSNorm benchmark results.

Usage:
    python benchmarks/plot_rmsnorm.py benchmarks/weightless-rmsnorm-results.txt --out results.png
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── parser ────────────────────────────────────────────────────────────────────

_ROW_RE = re.compile(
    r"bs=\s*(\d+)\s+h=\s*(\d+)\s+"
    r"with_w=\s*([\d.]+).*?weightless=\s*([\d.]+).*?speedup=\s*([+-][\d.]+)%"
)
_HEADER_RE = re.compile(r"──\s+(\w+)\s+\[(\w+)\]")
_MODEL_RE = re.compile(r"\[(.+),\s*h=(\d+)\]")


def parse(path: str) -> list[dict]:
    rows = []
    op = dtype = model = None
    with open(path) as f:
        for line in f:
            m = _HEADER_RE.search(line)
            if m:
                op, dtype = m.group(1), m.group(2)
                continue
            m = _MODEL_RE.search(line)
            if m:
                model = m.group(1)
                continue
            m = _ROW_RE.search(line)
            if m:
                rows.append(
                    dict(
                        op=op,
                        dtype=dtype,
                        model=model,
                        bs=int(m.group(1)),
                        hidden=int(m.group(2)),
                        t_w=float(m.group(3)),
                        t_wl=float(m.group(4)),
                        speedup_pct=float(m.group(5)),
                    )
                )
    return rows


# ── plot ──────────────────────────────────────────────────────────────────────

COLORS = {
    "Llama-3.1-8B": "#4C72B0",
    "Qwen3-14B": "#DD8452",
    "DeepSeek-V3": "#55A868",
    "Llama-3.1-70B": "#C44E52",
}
MARKERS = ["o", "s", "^", "D"]


def plot(rows: list[dict], out: str) -> None:
    # group by (op, dtype)
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["op"], r["dtype"])].append(r)

    n_panels = len(groups)
    ncols = min(n_panels, 2)
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.5 * ncols, 5.0 * nrows), sharey=False
    )
    axes_flat = np.array(axes).flatten().tolist()
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    gpu_name = "NVIDIA L20"  # fallback; could be parsed from file

    for ax, (key, panel_rows) in zip(axes_flat, sorted(groups.items())):
        op, dtype = key

        # gather per-model series
        by_model: dict[str, dict] = defaultdict(
            lambda: {"bs": [], "speedup": [], "t_w": [], "t_wl": []}
        )
        for r in panel_rows:
            by_model[r["model"]]["bs"].append(r["bs"])
            by_model[r["model"]]["speedup"].append(r["speedup_pct"])
            by_model[r["model"]]["t_w"].append(r["t_w"])
            by_model[r["model"]]["t_wl"].append(r["t_wl"])

        for i, (model, data) in enumerate(by_model.items()):
            bs = np.array(data["bs"])
            sp = np.array(data["speedup"])
            ax.plot(
                bs,
                sp,
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS.get(model, f"C{i}"),
                label=model,
                linewidth=1.5,
                markersize=5,
            )

        # zero line
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(sorted({r["bs"] for r in panel_rows}))

        # y-axis: show percentage with explicit ticks to make small values readable
        all_sp = [r["speedup_pct"] for r in panel_rows]
        ymin, ymax = min(all_sp), max(all_sp)
        pad = max(1.0, (ymax - ymin) * 0.15)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))

        ax.set_xlabel("Batch size", fontsize=11)
        ax.set_ylabel("Speedup (weightless vs with-weight)", fontsize=10)
        ax.set_title(f"{op} [{dtype}]", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Weightless RMSNorm speedup — {gpu_name}\n"
        f"(FLASHINFER_USE_CUDA_NORM=1, same CUDA JIT kernel, only weight=nullptr differs)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Benchmark result file (bench_rmsnorm.py output)")
    parser.add_argument(
        "--out", default="rmsnorm_speedup.png", help="Output image path"
    )
    args = parser.parse_args()

    rows = parse(args.input)
    if not rows:
        print("No data parsed — check file format.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(rows)} rows from {args.input}")
    plot(rows, args.out)


if __name__ == "__main__":
    main()
