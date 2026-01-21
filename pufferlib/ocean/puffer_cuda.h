#pragma once
#include <cuda_runtime.h>
#include <torch/torch.h>

using torch::Tensor;

// The following fused kernels were generated using Opus 4.5 using the initial unfused ones.

void launch_dual_linear_forward(const Tensor& input, // [B, In]
  const Tensor& weight1,                             // [Out1, In] - decoder
  const Tensor& bias1,                               // [Out1]
  Tensor& output1,                                   // [B, Out1]
  const Tensor& weight2,                             // [Out2, In] - value
  const Tensor& bias2,                               // [Out2]
  Tensor& output2);                                  // [B, Out2]

void launch_fused_lstm_cell(const Tensor& input, // [B, input_size]
  const Tensor& hidden,                          // [B, hidden_size]
  const Tensor& weight_ih,                       // [4*hidden_size, input_size]
  const Tensor& weight_hh,                       // [4*hidden_size, hidden_size]
  const Tensor& bias_ih,                         // [4*hidden_size]
  const Tensor& bias_hh,                         // [4*hidden_size]
  const Tensor& cx,                              // [B, hidden_size]
  Tensor& hy,                                    // [B, hidden_size]
  Tensor& cy);                                   // [B, hidden_size]

void lstm_forward_impl(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& input_bias,
  const Tensor& hidden_bias, const Tensor& cx, const Tensor& hy, const Tensor& cy,
  const Tensor& workspace);

void launch_sample_logits_kernel(const Tensor& random_vals, // [B, num_actions]
  const Tensor& sizes_gpu,                                  // [num_actions]
  const Tensor& offsets_gpu,                                // [num_actions]
  const Tensor& logits,                                     // [B, total_logits]
  int64_t num_actions,
  Tensor& actions,   // [B, num_actions] or [B]
  Tensor& logprobs); // [B]

//! @brief CPU kernel: Calculates output advantages and importance weights.
void compute_puff_advantage_cpu(Tensor values, Tensor rewards, Tensor dones, Tensor importance,
  Tensor& advantages_out, double gamma, double lambda, double rho_clip, double c_clip);

//! @brief CUDA kernel: Calculates output advantages and importance weights.
void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
  torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
  double gamma, double lambda, double rho_clip, double c_clip);
