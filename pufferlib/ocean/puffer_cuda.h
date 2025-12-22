#pragma once
#ifdef PUFFER_CUDA
#include <cuda_runtime.h>
#include <torch/torch.h>


void launch_linear_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    cudaStream_t stream);

#endif // PUFFER_CUDA

