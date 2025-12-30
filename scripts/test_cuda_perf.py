#!/usr/bin/env python3
# Generated using GPT 5.2 with the following prompt
# Write a Python program using pytorch to test the CUDA performance as follows:
# FLOPs
# Memory bandwidth
"""
CUDA microbenchmarks (PyTorch):
- GEMM throughput (approx TFLOPs): C = A @ B
- Memory bandwidth (approx GB/s): device-to-device copy and elementwise add

Notes:
- Results depend heavily on GPU model, clock, thermals, power limits, dtype, and sizes.
- GEMM TFLOPs is computed as 2*M*N*K / time.
- Bandwidth is computed from bytes moved / time (approximate).
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List, Tuple

import torch


def _to_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


@torch.no_grad()
def _time_cuda(fn: Callable[[], None], warmup: int, iters: int) -> List[float]:
    """Return per-iteration milliseconds using CUDA events."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms: List[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _stats(values: List[float]) -> Tuple[float, float, float]:
    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, median, stdev


@torch.no_grad()
def bench_flops(
    device: torch.device,
    dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
) -> None:
    # Heuristics: allow TF32 on Ampere+ for fp32 matmul if requested by user/environment.
    # Users can override via env vars or by editing here.
    if dtype == torch.float32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"\n== GEMM (FLOPs) on {device} ==")
    print(f"dtype={dtype}, M={m}, N={n}, K={k}")

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)

    # Ensure allocation of output doesn't dominate timing
    c = torch.empty((m, n), device=device, dtype=dtype)

    def fn() -> None:
        # Use out= to reduce allocator effects
        torch.matmul(a, b, out=c)

    times_ms = _time_cuda(fn, warmup=warmup, iters=iters)
    mean_ms, median_ms, stdev_ms = _stats(times_ms)

    # FLOPs for GEMM: 2*M*N*K (multiply+add)
    flops = 2.0 * m * n * k
    t_s = mean_ms / 1e3
    tflops = (flops / t_s) / 1e12

    print(
        f"time: mean={mean_ms:.3f} ms, median={median_ms:.3f} ms, stdev={stdev_ms:.3f} ms ({iters} iters)"
    )
    print(f"throughput: {tflops:.3f} TFLOPs (approx, using 2*M*N*K)")


@torch.no_grad()
def bench_bandwidth(
    device_from: torch.device,
    device_to: torch.device,
    dtype: torch.dtype,
    tensor_mb: int,
    warmup: int,
    iters: int,
    pinned_src: bool = False,  # Only applicable for CPU source
) -> None:
    # Allocate ~tensor_mb MiB per tensor
    bytes_target = int(tensor_mb) * 1024 * 1024
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = max(1, bytes_target // elem_size)
    actual_mb = (numel * elem_size) / (1024 * 1024)
    print(f"\n== Memory Bandwidth {actual_mb:.1f} MiB {device_from} to {device_to} Pinned? {pinned_src} ==")

    src = torch.empty((numel,), device=device_from, dtype=dtype)
    if pinned_src and device_from.type == "cpu":
        src = src.pin_memory()
    dst = torch.empty((numel,), device=device_to, dtype=dtype)

    # 1) Copy (read+write) ~= 2 * bytes
    def copy_fn() -> None:
        dst.copy_(src)

    copy_times_ms = _time_cuda(copy_fn, warmup=warmup, iters=iters)
    copy_mean_ms, copy_median_ms, copy_stdev_ms = _stats(copy_times_ms)

    bytes_moved_copy = 2.0 * (numel * elem_size)  # read src + write dst
    gbps_copy = (bytes_moved_copy / (copy_mean_ms / 1e3)) / 1e9

    # 2) Elementwise add into preallocated output (read+write) ~= 2 * bytes
    out = torch.empty_like(src)

    def add_fn() -> None:
        torch.add(src, 1.0, out=out)

    # add_times_ms = _time_cuda(add_fn, warmup=warmup, iters=iters)
    # add_mean_ms, add_median_ms, add_stdev_ms = _stats(add_times_ms)

    # bytes_moved_add = 2.0 * (numel * elem_size)  # read src + write out
    # gbps_add = (bytes_moved_add / (add_mean_ms / 1e3)) / 1e9

    print(
        f"dtype={dtype}, tensor_size≈{actual_mb:.1f} MiB (numel={numel}, elem_size={elem_size} bytes)"
    )
    print(
        f"Copy from {device_from} to {device_to}: time mean={copy_mean_ms:.3f} ms, median={copy_median_ms:.3f} ms, stdev={copy_stdev_ms:.3f} ms -> {gbps_copy:.2f} GB/s"
    )
    # print(f"add out for {device_from}: time mean={add_mean_ms:.3f} ms, median={add_median_ms:.3f} ms, stdev={add_stdev_ms:.3f} ms -> {gbps_add:.2f} GB/s")
    # print("Bandwidth math assumes ~2x tensor bytes moved (read+write).")


def main() -> None:
    p = argparse.ArgumentParser(
        description="PyTorch CUDA FLOPs and memory bandwidth microbenchmarks"
    )
    p.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
        help="Computation dtype",
    )
    p.add_argument("--m", type=int, default=8192, help="GEMM M")
    p.add_argument("--n", type=int, default=8192, help="GEMM N")
    p.add_argument("--k", type=int, default=8192, help="GEMM K")
    p.add_argument(
        "--tensor-mb",
        type=int,
        default=512,
        help="Tensor size in MiB for bandwidth tests",
    )
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p.add_argument("--iters", type=int, default=50, help="Measured iterations")
    p.add_argument(
        "--device_from",
        type=str,
        default="cuda",
        help="Source device for bandwidth/gemm tests",
    )
    p.add_argument(
        "--device_to",
        type=str,
        default="cpu",
        help="Destination device for bandwidth tests",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. Install a CUDA-enabled PyTorch and run on a CUDA-capable GPU."
        )

    device_from = torch.device(args.device_from)
    device_to = torch.device(args.device_to)
    if device_to == device_from:
        raise SystemExit(
            "device_to and device_from must be different for bandwidth tests."
        )
    dtype = _to_dtype(args.dtype)

    torch.cuda.init()
    torch.cuda.synchronize()

    prop = torch.cuda.get_device_properties(device_from)
    print("== Device ==")
    print(f"name: {prop.name}")
    print(f"compute capability: {prop.major}.{prop.minor}")
    print(f"total memory: {prop.total_memory / (1024**3):.2f} GiB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Reduce variability from CPU scheduling and lazy init
    torch.manual_seed(0)
    torch.cuda.synchronize()
    time.sleep(0.05)


    print("-----------------BANDWIDTH TEST (non-pinned) ----------------")
    for mb in [1, 2, 3, 4, 8, 16, 64, 256, args.tensor_mb]:
        bench_bandwidth(
            device_from=device_to,
            device_to=device_from,
            dtype=dtype,
            tensor_mb=mb,
            warmup=args.warmup,
            iters=args.iters,
            pinned_src=False
        )


    print("-----------------BANDWIDTH TEST (pinned) ----------------")
    for mb in [1, 2, 3, 4, 8, 16, 64, 256, args.tensor_mb]:
        bench_bandwidth(
            device_from=device_to,
            device_to=device_from,
            dtype=dtype,
            tensor_mb=mb,
            warmup=args.warmup,
            iters=args.iters,
            pinned_src=True
        )

    print("-----------------Now testing FLOPS ----------------")
    # CPU version will take a very long time so reduce m/n/k
    bench_flops(
        device=device_to,
        dtype=dtype,
        m=int(args.m / 4),
        n=int(args.n / 4),
        k=int(args.k / 4),
        warmup=args.warmup,
        iters=args.iters,
    )



    bench_flops(
        device=device_from,
        dtype=dtype,
        m=args.m,
        n=args.n,
        k=args.k,
        warmup=args.warmup,
        iters=args.iters,
    )
    bench_bandwidth(
        device_from=device_from,
        device_to=device_to,
        dtype=dtype,
        tensor_mb=args.tensor_mb,
        warmup=args.warmup,
        iters=args.iters,
    )

    bench_bandwidth(
        device_from=device_from,
        device_to=device_from,
        dtype=dtype,
        tensor_mb=args.tensor_mb,
        warmup=args.warmup,
        iters=args.iters,
    )

    bench_bandwidth(
        device_from=device_to,
        device_to=device_to,
        dtype=dtype,
        tensor_mb=args.tensor_mb,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
