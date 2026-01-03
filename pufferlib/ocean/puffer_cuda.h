#pragma once
#include <cuda_runtime.h>
#include <torch/torch.h>

using torch::Tensor;

// =============================================================================
// Single Linear Forward
// =============================================================================
void launch_linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias,
    Tensor& output);

// The following fused kernels were generated using Opus 4.5 using the initial unfused ones.
// =============================================================================
// Dual Linear Forward - computes two independent linear layers on same input
// Useful for decoder + value head which both take h2 as input
// =============================================================================
void launch_dual_linear_forward(
    const Tensor& input,    // [B, In]
    const Tensor& weight1,  // [Out1, In] - decoder
    const Tensor& bias1,    // [Out1]
    Tensor& output1,        // [B, Out1]
    const Tensor& weight2,  // [Out2, In] - value
    const Tensor& bias2,    // [Out2]
    Tensor& output2);       // [B, Out2]

// =============================================================================
// Fused LSTM Cell with integrated matmuls
// Combines: igates = input @ W_ih.T, hgates = hidden @ W_hh.T, lstm_cell
// =============================================================================
void launch_fused_lstm_cell(
    const Tensor& input,      // [B, input_size]
    const Tensor& hidden,     // [B, hidden_size]
    const Tensor& weight_ih,  // [4*hidden_size, input_size]
    const Tensor& weight_hh,  // [4*hidden_size, hidden_size]
    const Tensor& bias_ih,    // [4*hidden_size]
    const Tensor& bias_hh,    // [4*hidden_size]
    const Tensor& cx,         // [B, hidden_size]
    Tensor& hy,               // [B, hidden_size]
    Tensor& cy);              // [B, hidden_size]

// =============================================================================
// Original LSTM forward (separate igates/hgates already computed)
// =============================================================================
void lstm_forward_impl(const Tensor& input_gates, const Tensor& hidden_gates,
    const Tensor& input_bias, const Tensor& hidden_bias, const Tensor& cx,
    const Tensor& hy, const Tensor& cy, const Tensor& workspace);

// =============================================================================
// Linear + Categorical Sampling (decoder + sample fused)
// Supports variable action sizes per action dimension
// =============================================================================
void launch_linear_sample(
    const Tensor& input,        // [B, In]
    const Tensor& weight,       // [total_logits, In]
    const Tensor& bias,         // [total_logits]
    const Tensor& random_vals,  // [B, num_actions] - pre-generated uniform [0,1)
    const Tensor& action_sizes, // [num_actions] - int64 tensor with size of each action (CPU)
    Tensor& actions,            // [B, num_actions] output
    Tensor& logprobs,           // [B] output - sum of log probs
    Tensor* logits_out = nullptr); // optional: [B, total_logits] output for logits

// Overload for uniform action sizes (simpler interface)
void launch_linear_sample_uniform(
    const Tensor& input,        // [B, In]
    const Tensor& weight,       // [num_actions * action_size, In]
    const Tensor& bias,         // [num_actions * action_size]
    const Tensor& random_vals,  // [B, num_actions]
    int64_t num_actions,
    int64_t action_size,
    Tensor& actions,            // [B, num_actions] output
    Tensor& logprobs,           // [B] output
    Tensor* logits_out = nullptr); // optional: [B, total_logits]

// =============================================================================
// Sample-only kernel (when logits are already computed)
// =============================================================================
void launch_sample_logits(
    const Tensor& logits,       // [B, total_logits]
    const Tensor& random_vals,  // [B, num_actions]
    int64_t num_actions,
    const int64_t* logit_sizes, // array of sizes (CPU pointer)
    Tensor& actions,            // [B, num_actions]
    Tensor& logprobs);          // [B]

