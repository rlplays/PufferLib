# Notes on 4.0 perf investigation

2080 RTX
| env     | 3.0 | 4.0-lstm-peru | 4.0-gru-joseph |
|---------|-----|----------|-----|
| breakout|  1.2M SPS   | 4.3M SPS         |    1.3M SPS |
| go      |   630K SPS  |   1.6M SPS       |   840K SPS  |
| g2048   |   745K SPS  |   745K SPS       |   ? - doesn't run   |


CUDA profiles in [the same directory](./):

- Breakout eval:  GRU:  55ms per epoch default 2threads? (comparable to 4.0-lstm 58ms 8t/8b)
- Breakout train: GRU: 346ms per epoch (vs 4.0-lstm-peru: 90ms)

- go eval:  GRU:  49ms per epoch (4.0-lstm-peru: 186ms - BPTT horizon/lstm config is different, so more SPS per epoch)
- go train: GRU: 267ms per epoch (vs 4.0-lstm-peru: 320ms)

Quick analysis from CUDA profiling:

- Eval multithreading is not the problem. 
- Train (single-threaded, CUDA-graphed/fused kernels) is the main problem.
- Breakout/Go: train: sgemm operators are not block/thread/grid optimized for specific devices.
  -> Torch has some internal magic to set the right params?

**g2048** : unsigned char -> float conversion is a big problem.

`        .mb_obs = torch::zeros({mb_segments, horizon, input_size}, opts),`

- This will convert the `char` obs to `float` (f32 or bf16) which is 4x the bw.

Both your native code and mine end up converting char to float. This is where libtorch shines - their defaults work with whatever dtypes you throw at them... :(

- My native eval+train impl for g2048 is about the same SPS as the Python version simply because whatever ill-gotten speed gains via multi-threading is all lost on the bandwidth.

**Notes on why train is much slower on some GPUs+envs with latest 4.0**

I think the CUDA magic that Pytorch uses for their autograd/gemm is optimized for various architectures that unless we use the core functions they have (skip the high-level torch stuff) we won't be able to 'beat 'em' :

- libtorch code: RNN/LSTM/GRU etc including cudnn [here](https://github.com/pytorch/pytorch/blob/65842c7d4f616a61c2efa794bf9835eb34ce04db/aten/src/ATen/native/cudnn/RNN.cpp#L4) and [here](https://github.com/pytorch/pytorch/blob/65842c7d4f616a61c2efa794bf9835eb34ce04db/aten/src/ATen/native/RNN.cpp#L31) and their cuda kernels [here](https://github.com/pytorch/pytorch/blob/65842c7d4f616a61c2efa794bf9835eb34ce04db/aten/src/ATen/native/cuda/RNN.cu#L4) -> it's quite sophisticated especially how they figure out the correct [grid/block/thread sizes](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-blocks-and-grids) which is quite important. 

  -  **spit-balling** As much I don't like the higher level PyTorch stuff, their low-level libtorch stuff is highly optimized were we to use these functions directly. I end up doing that for `eval` but not for native `train` yet (as I don't think I fully grok autograd yet to mess with it - still learning - I still doubt whether I can surpass their performance just yet as it handles a wide variety of GPUs).

* Also their [`LinearAlgebra`](https://github.com/pytorch/pytorch/blob/65842c7d4f616a61c2efa794bf9835eb34ce04db/aten/src/ATen/native/LinearAlgebra.cpp#L4)  / `GEMM` stuff is full of ..ahem.. gems. Using their `_out` is probably a better idea than using full custom fused kernels. I ended up using fused kernels only at the sample_logits level, but the rest uses their gemm stuff as it's pretty well optimized.

 




