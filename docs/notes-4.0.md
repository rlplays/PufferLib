# Notes on 4.0 perf investigation

2080 RTX
| env     | 3.0 | 4.0-lstm-peru | 4.0-gru-joseph |
|---------|-----|----------|-----|
| breakout|  1.2M SPS   | 1.3M SPS         |    4.3M SPS |
| go      |   630K SPS  |   1.6M SPS       |   840K SPS  |


CUDA profile:

- Breakout eval:  GRU:  55ms per epoch default 2threads? (comparable to 4.0-lstm 58ms 8t/8b)
- Breakout train: GRU: 346ms per epoch (vs 4.0-lstm-peru: 90ms)

- go eval:  GRU:  49ms per epoch (4.0-lstm-peru: 186ms - BPTT horizon/lstm config is different, so more SPS per epoch)
- go train: GRU: 267ms per epoch (vs 4.0-lstm-peru: 320ms)

Quick analysis from CUDA profiling:

- Eval multithreading is not the problem. 
- Train (single-threaded, CUDA-graphed/fused kernels) is the main problem.
- Breakout/Go: train: sgemm operators are not block/thread/grid optimized for specific devices.
  -> Torch has some internal magic to set the right params?

g2048: unsigned char -> float conversion is a big problem?

(still investigating)

