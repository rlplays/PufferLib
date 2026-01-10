
## Multithreaded Libtorch fork of PufferLib 
 Main repo: https://github.com/pufferai/pufferlib

> 
>**DO NOT USE THIS YET** 
>
> Still in-progress. Go to https://puffer.ai  or the main repo https://github.com/pufferai/pufferlib
>
 

This repo contains a C++-native version of `evaluate` that uses libtorch + CUDA streams + threads to sub-linearly scale the core `eval<->train` loop.


**`Evaluate loop` optimization notes**

- **Independent Multithreading for GPU batches and envs**
  - GPU Batches: ~8 batches each with its own CUDA stream (depends on the GPU / GPU bandwidth).
    * HostToDevice copy (obs/rewards/terminals) and DeviceToHost copy (actions/logprobs). 
    * GPU copies across different batches proceed in parallel to GPU ops, both of which are in parallel to the envs.
    * This is different from the Multiprocessing backend: __Each batch/segment in a horizon proceeds sequentially but in parallel to other batches/segments.__
  - Env batches: ~12-16 (depending on the CPU)
    * The Env and GPU threads are separate (with different priorities).
    * 'Fat envs' such as go (or my pixel platformer) benefit a lot just from these two batching.
    * May need an autotune for the GPU/env batch sizes but  8 GPU batches/threads + 12-16 env batches/threads is a good pareto frontier number.
 - Both thread groups/batches have exactly one lock per batch. 
   * Per-env/gpu ops cost (inside a batch) is order of magnitude lower as there it's completely lock/atomics free (tight loop).
- **Fused kernels with out params**
  - Uses 3 tuned kernels for LSTM network (only discrete actions so far)
  - Preallocated tensors filled via out params
    * Obviate the need for cuda graphs (see below for why)
    * No tensor allocs during `evaluate` loop
    * Avoids bad CUDA caching allocator problems especially with multiple streams/threads (see below)
  - Reduced from ~26 cuda launch kernels down to total 6 (VERIFY)
    * For example, the `sample logits` did a bunch of tensor manipulation, sampling etc with many ops.
    * The new version is a single CUDA kernel that outputs logprobs + actions to two (prealloc'ed) output tensors
  - I tried different versions (tried the internal libtorch `_out` functions, their own CUDA kernels) before settling on these three cuda kernels.
    * One nice side-effect is that the cuda kernels are closer to the real puffernet one (e.g. gelu approximation) rather than the full lstm kernel in `models.py`.

- **Preallocated tensors**
  - Entire horizon is preallocated with the correct output tensors
  - Very minimal mallocs (CPU-side) and zero tensor allocs during the core evaluate loop
  - A preallocated random tensor for sampling discrete actions as `uniform_` calls are very expensive to run as part of CUDA. (GPUs have minimal PRNG capability esp. with SIMT)
  - Setup/teardown cost for a full horizon run is very minimal (~1ms on 2080RTX)

- **Maintain parity with existing PufferLib**
  - Supports incremental turning on/off of different features:
    - Maintains full parity existing `Multiprocessing` backend via `evaluate_python`
    - Add just the new `Multithreading` backend with existing PyTorch `evaluate_python` (no native C++ libtorch code) `enable_native_libtorch=0`
    - Add native libtorch with `enable_native_libtorch=0` with configurable GPU batches `num_gpu_batches` /env batches `max_num_threads` 
    - Experimental (not recommended) CUDA graph mode using `use_cuda_graphs = 1`

This new approach has been verified with full eval->train on breakout, go, pacman and my pixel platformer env with stable perf (i.e. final RL perf/score). 


****

**Misc/Tools**
 - Added `scripts/test_cuda_perf.py` to test out bandwidth/FLOPs/launch kernel costs. Quick-n-dirty benchmarks when testing out on a vast.ai/runpod.io machine for comparison purposes.
 - Added/fixed the profiler to have a development loop of `measure, analyze, optimize`
   - Outputs the CUDA profile (.json-> ui.perfetto.dev) along with useful stats into a text file.
   - Use the `start_profile_env.sh` script to profile a bunch of envs in one go, and open their profiles in `ui.perfetto.dev`.
 - Misc stuff: 
   - CUDA graph mode (not used/not recommended) `use_cuda_graphs = 1` in your .ini 
   - Single threaded mode (non-multi-threaded version for debugging via `#define PUFFER_SINGLE_THREADED 1`)
   - 'cuda memcheck' mode in C++ that outputs which of 'our' tensors are being cached by the CUDA caching allocator `#define PUFFER_CUDA_MEMCHECK 1`
   - Micro benchmarks (see `PerfTimer`) + tensor comparisons inside the core C++ code to test stability and performance with realistic data/harness.
   - Timing etc wired up to the main python-side so the dashboard/profile all work seamlessly. 
*Appendix*

**Why not CUDA graphs?**
 - CUDA graphs help eliminate multiple `launch kernel` costs.
 - However, based on experimentation, I found that it doesn't meet our needs:
   - Requires copying data (even if it's DeviceToDevice for obs, it's a lot).
   - `cudaMemcpyAsync` has a similar cost to `launch kernel`
 - The cost + complicated nature to get the same perf as a few cuda kernel launches obviates their need.

**CUDA Caching allocator problems**
  - With multiple streams+threads, the caching allocator maintains large `Tensor` allocs for a very long time even past a horizon. 
  - We run out of GPU memory or worse fragmented sections resulting in bad perf
  - Even with combinations of hacky flags proposed by pytorch docs such as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (and others) it OOMs pretty fast
