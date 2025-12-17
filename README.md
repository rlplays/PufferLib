
## Multithreaded Libtorch fork of PufferLib 
 Main repo: https://github.com/pufferai/pufferlib

> 
>**DO NOT USE THIS YET** 
>
> Still in-progress. Go to https://puffer.ai  or the main repo https://github.com/pufferai/pufferlib
>
 

This repo contains a C++-native version of `evaluate` that uses libtorch + CUDA streams + threads to sub-linearly scale the core `eval<->train` loop.


## Native Multithreading + Libtorch approach

Key improvements:
- Multi-threaded `copy obs to GPU` in batches (based on GPU<-> bandwidth/configurable)
  - Use one CUDA stream per batch
  - Batches accrue segments across an horizon independently
- Each batch then runs the forward pass
- Multithreaded batched env `step` (utilizes all cores)



