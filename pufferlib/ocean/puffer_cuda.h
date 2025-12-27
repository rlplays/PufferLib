#pragma once
#ifdef PUFFER_CUDA
#include <cuda_runtime.h>
#include <torch/torch.h>

using torch::Tensor;
void launch_linear_forward(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
  at::Tensor& output, cudaStream_t stream);

void launch_lineargelu_forward(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
  at::Tensor& output, cudaStream_t stream);

void lstm_forward_impl(const Tensor& input_gates, const Tensor& hidden_gates,
  const Tensor& input_bias, const Tensor& hidden_bias, const Tensor& cx,
  const Tensor& hy, const Tensor& cy, const Tensor& workspace);

#endif // PUFFER_CUDA
