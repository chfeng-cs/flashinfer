"""Benchmark weightless vs standard RMSNorm and FusedAddRMSNorm.

Usage:
    python benchmarks/bench_rmsnorm.py
    python benchmarks/bench_rmsnorm.py --csv results.csv
    python benchmarks/bench_rmsnorm.py --dtype float16
"""

import argparse
import csv
import os
import pathlib
import statistics
import sys

_repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")

import torch

import flashinfer

DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# (model_name, hidden_size)
HIDDEN_CONFIGS = [
    ("DeepSeek-V3", 7168),
]

# decode (1–64) + prefill (256–4096)
BATCH_SIZES = [16, 64, 256, 1024, 4096]

_DRY_RUN = 50
_REPEAT = 1000
_INNER = 100  # inner loop keeps weight tensor hot in L2


def _bench(fn):
    """Return (median_ms, std_ms) via CUDA events with inner loop."""
    for _ in range(_DRY_RUN):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(_REPEAT):
        start.record()
        for _ in range(_INNER):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / _INNER)

    return statistics.median(times), statistics.stdev(times)


def benchmark_rmsnorm(batch_size: int, hidden_size: int, dtype: torch.dtype) -> dict:
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")
    out = torch.empty_like(x)

    t_wl, std_wl = _bench(lambda: flashinfer.rmsnorm(x, None, out=out))
    t_w, std_w = _bench(lambda: flashinfer.rmsnorm(x, w, out=out))

    elem = batch_size * hidden_size
    bw_w = (2 * elem + hidden_size) * x.element_size() / (t_w * 1e-3) / 1e9
    bw_wl = 2 * elem * x.element_size() / (t_wl * 1e-3) / 1e9

    return {
        "op": "rmsnorm",
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "dtype": str(dtype).split(".")[-1],
        "time_with_weight_us": t_w * 1e3,
        "std_with_weight_us": std_w * 1e3,
        "time_weightless_us": t_wl * 1e3,
        "std_weightless_us": std_wl * 1e3,
        "speedup_pct": (t_w / t_wl - 1) * 100,
        "bw_with_weight_GBs": bw_w,
        "bw_weightless_GBs": bw_wl,
    }


def benchmark_fused_add_rmsnorm(
    batch_size: int, hidden_size: int, dtype: torch.dtype
) -> dict:
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    r = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")

    t_wl, std_wl = _bench(lambda: flashinfer.fused_add_rmsnorm(x, r))
    t_w, std_w = _bench(lambda: flashinfer.fused_add_rmsnorm(x, r, w))

    elem = batch_size * hidden_size
    bw_w = (4 * elem + hidden_size) * x.element_size() / (t_w * 1e-3) / 1e9
    bw_wl = 4 * elem * x.element_size() / (t_wl * 1e-3) / 1e9

    return {
        "op": "fused_add_rmsnorm",
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "dtype": str(dtype).split(".")[-1],
        "time_with_weight_us": t_w * 1e3,
        "std_with_weight_us": std_w * 1e3,
        "time_weightless_us": t_wl * 1e3,
        "std_weightless_us": std_wl * 1e3,
        "speedup_pct": (t_w / t_wl - 1) * 100,
        "bw_with_weight_GBs": bw_w,
        "bw_weightless_GBs": bw_wl,
    }


def print_row(r: dict) -> None:
    print(
        f"  bs={r['batch_size']:>4}  h={r['hidden_size']:>5}  "
        f"with_w={r['time_with_weight_us']:>7.2f}±{r['std_with_weight_us']:>5.2f} µs  "
        f"weightless={r['time_weightless_us']:>7.2f}±{r['std_weightless_us']:>5.2f} µs  "
        f"speedup={r['speedup_pct']:>+6.1f}%"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="all", choices=list(DTYPE_MAP) + ["all"])
    parser.add_argument("--csv", default=None, help="Path to save CSV results")
    args = parser.parse_args()

    dtypes = (
        list(DTYPE_MAP.items())
        if args.dtype == "all"
        else [(args.dtype, DTYPE_MAP[args.dtype])]
    )
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    all_rows = []

    for dtype_name, dtype in dtypes:
        for op_fn, op_name in [
            (benchmark_rmsnorm, "rmsnorm"),
            (benchmark_fused_add_rmsnorm, "fused_add_rmsnorm"),
        ]:
            print(
                f"── {op_name} [{dtype_name}] ──────────────────────────────────────────────"
            )
            for model_name, hidden_size in HIDDEN_CONFIGS:
                print(f"  [{model_name}, h={hidden_size}]")
                for batch_size in BATCH_SIZES:
                    row = op_fn(batch_size, hidden_size, dtype)
                    row["gpu"] = gpu_name
                    row["model"] = model_name
                    print_row(row)
                    all_rows.append(row)
            print()

    if args.csv:
        fieldnames = [
            "gpu",
            "op",
            "model",
            "batch_size",
            "hidden_size",
            "dtype",
            "time_with_weight_us",
            "std_with_weight_us",
            "time_weightless_us",
            "std_weightless_us",
            "speedup_pct",
            "bw_with_weight_GBs",
            "bw_weightless_GBs",
        ]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
