
## Multithreaded Libtorch fork of PufferLib 
 Main repo: https://github.com/pufferai/pufferlib

> 
>**DO NOT USE THIS YET** 
>
> Still in-progress. Go to https://puffer.ai  or the main repo https://github.com/pufferai/pufferlib
>
 

This repo contains a C++-native version of `evaluate` that uses libtorch + CUDA streams + threads to sub-linearly scale the core `eval<->train` loop.

TODO: Add profiling data for
 - Multi-proc
 - Multi-threaded
 - Multiple batches

## Native Multithreading + Libtorch `evaluate`

Key improvements:
- Multi-threaded `copy obs to GPU` in batches (chunk size configurable based on GPU<-> bandwidth)
  - Use one CUDA stream per batch
  - Batches accrue segments across an horizon independently from each other. 
    - Segments proceed linearly but parts of it are run async
- Each batch then runs the forward pass to obtain actions/logprobs/values etc.
- Multithreaded batched env `step` (utilizes all cores)
- Each batch independently transfers actions/logprobs/values from CPU to GPU per-segment using a per-segment / per-batch stream.
  

## Speed ups

TODO: Add data here from our experiments.


TODO: Similar improvements to training loop as well: parts of processing each minibatch can be parallelized.


