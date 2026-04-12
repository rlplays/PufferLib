#!/usr/bin/env python3
# I used the help of GPT 5.2 to generate the basic structure with the following prompt

# --
# Write a Python program using pytorch to test the CUDA performance as follows:
# FLOPs
# Memory bandwidth
# --

# Heavily modified to add profiling, cuda<->cpu etc
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
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension
import torch.profiler
import torch.cuda._memory_viz

import os
from datetime import datetime

from collections import defaultdict, deque
from datetime import datetime

global cuda_trace_enabled
cuda_trace_enabled: bool = False

global _DUMMY_LAUNCH_EXT
_DUMMY_LAUNCH_EXT = None

def _load_dummy_launch_ext():
    """
    Builds/loads a tiny CUDA extension that launches an empty kernel in a C++ loop.

    Why: If you launch kernels in a Python loop, Python overhead dominates.
    This extension launches many kernels per call, so timing approximates cudaLaunchKernel overhead.
    """
    global _DUMMY_LAUNCH_EXT
    if _DUMMY_LAUNCH_EXT is not None:
        return _DUMMY_LAUNCH_EXT

    # NOTE: This requires a working C++ toolchain + NVCC (or compatible CUDA build tooling).
    name = "dummy_cuda_launch_ext"

    cpp_src = r"""
#include <torch/extension.h>

void launch_dummy(int64_t iters, int64_t blocks, int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_dummy", &launch_dummy, "Launch a dummy CUDA kernel in a loop (cudaLaunchKernel overhead proxy)");
}
"""

    cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void dummy_kernel() {
  // Intentionally empty (measure launch + minimal scheduling).
}

void launch_dummy(int64_t iters, int64_t blocks, int64_t threads) {
  if (iters <= 0) return;
  if (blocks <= 0) blocks = 1;
  if (threads <= 0) threads = 1;

  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  for (int64_t i = 0; i < iters; i++) {
    dummy_kernel<<<(int)blocks, (int)threads, 0, stream>>>();
  }
}
"""

    import sys
    extra_cuda_cflags = ["-O3"]
    extra_cflags = ["/O2"] if sys.platform == "win32" else ["-O3"]

    _DUMMY_LAUNCH_EXT = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=None,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=False,
    )
    return _DUMMY_LAUNCH_EXT


@torch.no_grad()
def bench_cuda_launch_kernel(
    device: torch.device,
    warmup: int,
    iters: int,
    inner_launches: int,
    blocks: int = 1,
    threads: int = 1,
) -> None:
    """
    Measures approximate cudaLaunchKernel overhead by launching an empty kernel many times from C++.

    Timing includes:
      - kernel enqueue (launch) overhead
      - minimal kernel execution/scheduling
    """
    if device.type != "cuda":
        raise ValueError("bench_cuda_launch_kernel requires a CUDA device")

    ext = _load_dummy_launch_ext()

    print(f"\n== Kernel Launch (cudaLaunchKernel proxy) on {device} ==")
    print(
        f"launch config: blocks={blocks}, threads={threads}, inner_launches_per_timed_iter={inner_launches}"
    )

    # Warmup (also triggers compilation on first run)
    for _ in range(warmup):
        ext.launch_dummy(int(inner_launches), int(blocks), int(threads))
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times_ms: List[float] = []
    for _ in range(iters):
        start.record()
        ext.launch_dummy(int(inner_launches), int(blocks), int(threads))
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    mean_ms, median_ms, stdev_ms = _stats(times_ms)

    per_launch_us_mean = (mean_ms * 1e3) / max(1, inner_launches)
    per_launch_us_median = (median_ms * 1e3) / max(1, inner_launches)

    launches_per_s = (max(1, inner_launches) / (mean_ms / 1e3)) if mean_ms > 0 else float("inf")

    print(
        f"time (batch): mean={mean_ms:.3f} ms, median={median_ms:.3f} ms, stdev={stdev_ms:.3f} ms ({iters} iters)"
    )
    print(
        f"per-launch: mean≈{per_launch_us_mean:.3f} µs, median≈{per_launch_us_median:.3f} µs"
    )
    print(f"launch rate: ≈{launches_per_s:,.0f} launches/s")


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
def _time_cuda(
    fn: Callable[[], None], warmup: int, iters: int, profile_name: str
) -> List[float]:
    """Return per-iteration milliseconds using CUDA events."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if cuda_trace_enabled == 1:
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        print("Now capturing CUDA trace. This may take a while...")
        trace_file = f"experiments/torchtrace_{ts}_{profile_name}.json"
        from torch.profiler import profile, record_function, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("trace"):
                fn()
        print(f"Profiling completed. Exporting to trace file {trace_file}...")
        perf_results = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=50
        )
        print(perf_results)
        prof.export_chrome_trace(trace_file)
        print(f"Exported trace to {trace_file}")

    # Warmup
    for i in range(warmup):
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

    times_ms = _time_cuda(
        fn, warmup=warmup, iters=iters, profile_name=f"flops_{dtype}_M{m}_N{n}_K{k}"
    )
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
    print(
        f"\n== Memory Bandwidth {actual_mb:.1f} MiB {device_from} to {device_to} Pinned? {pinned_src} =="
    )

    src = torch.empty((numel,), device=device_from, dtype=dtype)
    if pinned_src and device_from.type == "cpu":
        src = src.pin_memory()
    dst = torch.empty((numel,), device=device_to, dtype=dtype)

    # 1) Copy (read+write) ~= 2 * bytes
    def copy_fn() -> None:
        dst.copy_(src)

    copy_times_ms = _time_cuda(
        copy_fn,
        warmup=warmup,
        iters=iters,
        profile_name=f"bandwidth_copy_{dtype}_{device_from}_to_{device_to}_pinned{pinned_src}_size{actual_mb:.1f}MB",
    )
    copy_mean_ms, copy_median_ms, copy_stdev_ms = _stats(copy_times_ms)

    bytes_moved_copy = (numel * elem_size)
    gbps_copy = (bytes_moved_copy / (copy_mean_ms / 1e3)) / 1e9

    # 2) Elementwise add into preallocated output (read+write) ~= 2 * bytes
    out = torch.empty_like(src)

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
    p.add_argument(
        "--trace",
        type=bool,
        default=False,
        help="Enable CUDA tracing via torch.profiler",
    )
    p.add_argument(
        "--kernel-launch-inner",
        type=int,
        default=10000,
        help="Dummy kernel launches per timed iteration (reduces Python overhead)",
    )
    p.add_argument(
        "--kernel-launch-blocks",
        type=int,
        default=1,
        help="Blocks for dummy kernel launch",
    )
    p.add_argument(
        "--kernel-launch-threads",
        type=int,
        default=1,
        help="Threads per block for dummy kernel launch",
    )    

    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. Install a CUDA-enabled PyTorch and run on a CUDA-capable GPU."
        )

    if args.trace:
        global cuda_trace_enabled
        cuda_trace_enabled = True
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

    print("-----------------CUDA launch kernel test ----------------")
    try:
        bench_cuda_launch_kernel(
            device=device_from if device_from.type == "cuda" else torch.device("cuda"),
            warmup=max(1, int(args.warmup // 2)),
            iters=args.iters,
            inner_launches=args.kernel_launch_inner,
            blocks=args.kernel_launch_blocks,
            threads=args.kernel_launch_threads,
        )
    except Exception as e:
        print(f"\n[warn] kernel-launch benchmark failed: {e}")
        print("[warn] likely missing NVCC / build toolchain for torch extensions")

    print("-----------------BANDWIDTH TEST (non-pinned) ----------------")
    for mb in [1, 2, 3, 4, 8, 16, 64, 256, args.tensor_mb]:
        bench_bandwidth(
            device_from=device_to,
            device_to=device_from,
            dtype=dtype,
            tensor_mb=mb,
            warmup=args.warmup,
            iters=args.iters,
            pinned_src=False,
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
            pinned_src=True,
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
