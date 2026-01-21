#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/macros/Macros.h>

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <cuda_runtime.h>
#include <torch/torch.h>

using at::Tensor;
using at::cuda::detail::TensorInfo;


// Code copied from libtorch. See LICENSE file in pytorch root directory; also included in the main PufferLib LICENSE
// file.

using at::cuda::detail::canUse32BitIndexMath;
using at::cuda::detail::getTensorInfo;
using at::cuda::detail::IndexToOffset;
using at::cuda::detail::TensorInfo;


// TODO(perumaal): Some of the fused kernels are slower when hidden_size/input_size > 128.


/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b)
{
  return (a + b - 1) / b;
}
// Threads per block for our apply kernel
// FIXME: use occupancy calculator instead
constexpr uint32_t AT_APPLY_THREADS_PER_BLOCK = 512;
// constexpr uint32_t AT_APPLY_BLOCKS_PER_SM = 4; // WARNING: unused

bool allContiguous(at::TensorList tensors)
{
  return std::all_of(tensors.begin(), tensors.end(),
                     [](const at::Tensor& t) { return !t.defined() || t.is_contiguous(); });
}

template <int step = 1>
inline bool getApplyGrid(uint64_t totalElements, dim3& grid, c10::DeviceIndex curDevice,
                         int max_threads_per_block = AT_APPLY_THREADS_PER_BLOCK)
{
  if (curDevice == -1)
    return false;
  uint64_t numel_per_thread = static_cast<uint64_t>(max_threads_per_block) * static_cast<uint64_t>(step);
  uint64_t numBlocks = ATenCeilDiv(totalElements, numel_per_thread);
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
    numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

inline dim3 getApplyBlock(int max_threads_per_block = AT_APPLY_THREADS_PER_BLOCK)
{
  return dim3(max_threads_per_block);
}

void getLaunchConfig(dim3* block, dim3* grid, int64_t numel)
{
  c10::DeviceIndex curDevice = -1;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&curDevice));
  *block = getApplyBlock();
  TORCH_CHECK(getApplyGrid(numel, *grid, curDevice), "Could not get grid size for pointwise apply.");
}

template <typename T, typename T2>
TensorInfo<T, T2> tryGetTensorInfo(const at::Tensor& t)
{
  return t.defined() ? getTensorInfo<T, T2>(t) : TensorInfo<T, T2>{};
}

void collapseDims() {};
template <typename T, typename T2, typename... Args>
void collapseDims(TensorInfo<T, T2>& info, Args&... infos)
{
  info.collapseDims();
  collapseDims(infos...);
}

#define DEVICE_LINEAR_GET(D_TENSOR, INDEX)                                                                             \
  D_TENSOR.data[IndexToOffset<scalar_t, index_type, indexing_kind>::get(INDEX, D_TENSOR)]

// Biases are always 1D
#define DEVICE_BIAS_GET(D_TENSOR, INDEX) D_TENSOR.data[IndexToOffset<scalar_t, index_type, 1>::get(INDEX, D_TENSOR)]

#define H2F(input) static_cast<accscalar_t>(input)
#define F2H(input) static_cast<scalar_t>(input)


template <typename T>
__device__ __forceinline__ T sigmoid(T in)
{
  T one = static_cast<T>(1.0);
  return one / (one + ::exp(-in));
}
namespace kernel
{

template <typename scalar_t, typename accscalar_t, typename index_type, int indexing_kind>
C10_LAUNCH_BOUNDS_2(512, 4)
__global__ void lstm_cell_forward(TensorInfo<scalar_t, index_type> input, TensorInfo<scalar_t, index_type> hidden,
                                  TensorInfo<scalar_t, index_type> bias1, TensorInfo<scalar_t, index_type> bias2,
                                  TensorInfo<scalar_t, index_type> _cx, TensorInfo<scalar_t, index_type> _hy,
                                  TensorInfo<scalar_t, index_type> _cy, TensorInfo<scalar_t, index_type> workspace,
                                  index_type hsz, index_type totalElements)
{
  bool has_bias = bias1.data != nullptr;
  for (index_type linearIndex = blockIdx.x * blockDim.x + threadIdx.x; linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x)
  {
    index_type offset = (linearIndex / hsz) * 4 * hsz + linearIndex % hsz;

    scalar_t iig = DEVICE_LINEAR_GET(input, offset + 0 * hsz);
    scalar_t ifg = DEVICE_LINEAR_GET(input, offset + 1 * hsz);
    scalar_t icg = DEVICE_LINEAR_GET(input, offset + 2 * hsz);
    scalar_t iog = DEVICE_LINEAR_GET(input, offset + 3 * hsz);

    scalar_t hig = DEVICE_LINEAR_GET(hidden, offset + 0 * hsz);
    scalar_t hfg = DEVICE_LINEAR_GET(hidden, offset + 1 * hsz);
    scalar_t hcg = DEVICE_LINEAR_GET(hidden, offset + 2 * hsz);
    scalar_t hog = DEVICE_LINEAR_GET(hidden, offset + 3 * hsz);

    scalar_t* wig = &DEVICE_LINEAR_GET(workspace, offset + 0 * hsz);
    scalar_t* wfg = &DEVICE_LINEAR_GET(workspace, offset + 1 * hsz);
    scalar_t* wcg = &DEVICE_LINEAR_GET(workspace, offset + 2 * hsz);
    scalar_t* wog = &DEVICE_LINEAR_GET(workspace, offset + 3 * hsz);

    scalar_t cx = DEVICE_LINEAR_GET(_cx, linearIndex);

    scalar_t* hy = &DEVICE_LINEAR_GET(_hy, linearIndex);
    scalar_t* cy = &DEVICE_LINEAR_GET(_cy, linearIndex);

    scalar_t b1i, b1f, b1c, b1o;
    scalar_t b2i, b2f, b2c, b2o;

    if (has_bias)
    {
      b1i = DEVICE_BIAS_GET(bias1, linearIndex % hsz + 0 * hsz);
      b1f = DEVICE_BIAS_GET(bias1, linearIndex % hsz + 1 * hsz);
      b1c = DEVICE_BIAS_GET(bias1, linearIndex % hsz + 2 * hsz);
      b1o = DEVICE_BIAS_GET(bias1, linearIndex % hsz + 3 * hsz);

      b2i = DEVICE_BIAS_GET(bias2, linearIndex % hsz + 0 * hsz);
      b2f = DEVICE_BIAS_GET(bias2, linearIndex % hsz + 1 * hsz);
      b2c = DEVICE_BIAS_GET(bias2, linearIndex % hsz + 2 * hsz);
      b2o = DEVICE_BIAS_GET(bias2, linearIndex % hsz + 3 * hsz);
    }
    else
    {
#ifndef THC_REAL_IS_HALF
      b1i = 0.0;
      b1f = 0.0;
      b1c = 0.0;
      b1o = 0.0;
      b2i = 0.0;
      b2f = 0.0;
      b2c = 0.0;
      b2o = 0.0;
#else
      b1i = F2H(0.0);
      b1f = F2H(0.0);
      b1c = F2H(0.0);
      b1o = F2H(0.0);
      b2i = F2H(0.0);
      b2f = F2H(0.0);
      b2c = F2H(0.0);
      b2o = F2H(0.0);
#endif
    }

    accscalar_t ig, fg, cg, og;
    accscalar_t f_hy, f_cy;

    ig = sigmoid(H2F(iig) + H2F(hig) + H2F(b1i) + H2F(b2i));
    fg = sigmoid(H2F(ifg) + H2F(hfg) + H2F(b1f) + H2F(b2f));
    cg = ::tanh(H2F(icg) + H2F(hcg) + H2F(b1c) + H2F(b2c));
    og = sigmoid(H2F(iog) + H2F(hog) + H2F(b1o) + H2F(b2o));

    f_cy = (fg * H2F(cx)) + (ig * cg);
    f_hy = og * ::tanh(f_cy);

    *hy = F2H(f_hy);
    *cy = F2H(f_cy);

    // SAVE FOR BACKWARDS
    // Also need cy and cx but can be saved easily in python
    *wig = F2H(ig);
    *wfg = F2H(fg);
    *wcg = F2H(cg);
    *wog = F2H(og);
  }
}
} // namespace kernel

void lstm_forward_impl(const Tensor& input_gates, const Tensor& hidden_gates, const Tensor& input_bias,
                       const Tensor& hidden_bias, const Tensor& cx, const Tensor& hy, const Tensor& cy,
                       const Tensor& workspace)
{

  dim3 block, grid;
  int64_t numel = cx.numel();
  if (numel == 0)
    return;
  getLaunchConfig(&block, &grid, numel);

  auto input_gatesI = getTensorInfo<float, size_t>(input_gates);
  auto hidden_gatesI = getTensorInfo<float, size_t>(hidden_gates);
  auto input_biasI = tryGetTensorInfo<float, size_t>(input_bias);
  auto hidden_biasI = tryGetTensorInfo<float, size_t>(hidden_bias);
  auto cxI = getTensorInfo<float, size_t>(cx);
  auto hyI = getTensorInfo<float, size_t>(hy);
  auto cyI = getTensorInfo<float, size_t>(cy);
  auto workspaceI = getTensorInfo<float, size_t>(workspace);
  size_t hidden_size = cxI.sizes[cxI.dims - 1];

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (allContiguous({input_gates, hidden_gates, input_bias, hidden_bias, cx, hy, cy, workspace}))
  {
    collapseDims(input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI);
    kernel::lstm_cell_forward<float, float, size_t, 1><<<grid, block, 0, stream>>>(
      input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI, hidden_size, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  else
  {
    kernel::lstm_cell_forward<float, float, size_t, 2><<<grid, block, 0, stream>>>(
      input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI, hidden_size, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

/**
 *
 * Used Opus 4.5 for help - mostly handwritten as Opus 4.5 gets most of this wrong.
 */

__global__ void dual_linear_forward_kernel(
  const float* __restrict__ h2_in, int64_t h2_input_stride0, int64_t h2_input_stride1, // h2
  const float* __restrict__ decoder_weights, int64_t decoder_weight_stride0,
  int64_t decoder_weight_stride1,                                                            // decoder_weights
  const float* __restrict__ decoder_bias,                                                    // decoder_bias
  float* __restrict__ decoder_out, int64_t decoder_out_stride0, int64_t decoder_out_stride1, // decoder_out
  const float* __restrict__ value_weights, int64_t value_weight_stride0, int64_t value_weight_stride1, // values weights
  const float* __restrict__ value_bias,                                                                // values bias
  float* __restrict__ values_out, int64_t values_out_stride0, int64_t values_out_stride1,              // values out
  int64_t batch_size, int64_t hidden_size, int64_t decoder_weight_size, int64_t value_weights_size)
{
  // The idea here is to generate both the decoder output (for logits computation) and the value output for
  // training later. The logits will be used by another kernel to compute final action logits/logprobs.
  // This kernel is agnostic to the number of actions/logits-per-action. The flattened output tensor
  // decoder_out will be split into logits for logprobs correctly later. This would also match the simple
  // eval-time puffernet decoder (sans the value computation).
  for (int64_t batch_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; batch_idx < batch_size;
       batch_idx += static_cast<int64_t>(blockDim.x) * gridDim.x)
  {
    // For breakout: Input is of shape (say) [2048 (envs), 128 (hidden_size)] which is the output from the lstm (h2).
    // batch indexing is over the envs.
    // h2_input_base = env# in the given batch (for this CUDA block)
    const int64_t h2_input_base = batch_idx * h2_input_stride0;

    // decoder_out = [batch_size, logits]
    for (int64_t logit_idx = 0; logit_idx < decoder_weight_size; logit_idx++)
    {
      // logit_idx traverses over the logits dimension (decoder_weights[0])
      float sum = 0.0f;
      const int64_t weight_base = logit_idx * decoder_weight_stride0;

      // decoder_weight = [logits, hidden_size]
      for (int64_t i = 0; i < hidden_size; ++i)
      {
        // output = sum(h2[batch][i]*w[i+out_j]) + b[out_j]
        sum += h2_in[h2_input_base + i * h2_input_stride1] * decoder_weights[weight_base + i * decoder_weight_stride1];
      }
      sum += decoder_bias[logit_idx];
      // Because the decoder output (1) is used by logits, better to clamp/clean it right here once
      // rather than as part of the logits computation later to save on extra ops.
      // (compute 1 h2_in used by multiple outputs).
      decoder_out[batch_idx * decoder_out_stride0 + logit_idx * decoder_out_stride1] =
        (isnan(sum) || isinf(sum)) ? -1e10f : sum;
    }

    // Compute values_out (value) elements - (assumes value_weights_size = 1)
    // for (int64_t out_idx = 0; out_idx < value_weights_size; out_idx++)
    int64_t out_idx = 0;
    {
      float sum = 0.0f;
      const int64_t weight_base = out_idx * value_weight_stride0;

      // hidden_size = num_envs * total_logits. For breakout, it's discrete, 1 action 3 total logits (left/right/stay).
      // for things like go/g2048, it's much larger and may have disparate logits (1 action with 2 logits, another one
      // with 4 etc).
      for (int64_t i = 0; i < hidden_size; ++i)
      {
        // output = sum(h2[batch][i]*w[i+out_j]) + b[out_j]
        sum += h2_in[h2_input_base + i * h2_input_stride1] * value_weights[weight_base + i * value_weight_stride1];
      }
      sum += value_bias[out_idx]; // Only one value bias (shape [1])
      values_out[batch_idx * values_out_stride0 + out_idx * values_out_stride1] = sum;
    }
    // printf("batch %d (%d size) / block x %d block y %d block dim x %d block dim y %d\n", int(batch_idx),
    // int(batch_size), int(blockIdx.x), int(blockIdx.y), int(blockDim.x), int(blockDim.y));
  }
}

void launch_dual_linear_forward(const Tensor& h2_in,           // [B, In]
                                const Tensor& decoder_weights, // [Out1, In] - decoder
                                const Tensor& decoder_bias,    // [Out1]
                                Tensor& decoder_out,           // [B, Out1]
                                const Tensor& value_weights,   // [Out2, In] - value
                                const Tensor& value_bias,      // [Out2]
                                Tensor& values_out)            // [B, Out2]
{
  const auto batch_size = h2_in.size(0);                    // num_envs (num_envs here always means per CUDA batch)
  const auto hidden_size = h2_in.size(1);                   // hidden size
  const auto decoder_weight_size = decoder_weights.size(0); // (num logits)
  const auto value_weights_size = value_weights.size(0);    // (usually 1)

  // Validation
  TORCH_CHECK(h2_in.is_cuda() && decoder_weights.is_cuda() && value_weights.is_cuda(),
              "All input tensors must be CUDA");
  TORCH_CHECK(decoder_bias.is_cuda() && value_bias.is_cuda(), "All bias tensors must be CUDA");
  TORCH_CHECK(decoder_weights.is_contiguous() && value_weights.is_contiguous(),
              "All weight tensors must be contiguous CUDA tensors");
  TORCH_CHECK(decoder_bias.is_contiguous() && value_bias.is_contiguous(),
              "All bias tensors must be contiguous CUDA tensors");
  TORCH_CHECK(decoder_out.is_cuda() && values_out.is_cuda(), "All output tensors must be CUDA");

  TORCH_CHECK(h2_in.dtype() == torch::kFloat32, "h2_in must be float32");
  TORCH_CHECK(decoder_weights.sizes() == at::IntArrayRef({decoder_weight_size, hidden_size}),
              "decoder_weight must have shape [logits, hidden_size]");
  TORCH_CHECK(decoder_bias.sizes() == at::IntArrayRef({decoder_weight_size}), "decoder_bias must have shape [logits]");
  TORCH_CHECK(decoder_out.sizes() == at::IntArrayRef({batch_size, decoder_weight_size}),
              "decoder_out must have shape [logits]");
  TORCH_CHECK(value_weights.sizes() == at::IntArrayRef({1, hidden_size}),
              "value_weights must have shape [1, hidden_size]");
  // If we change this, we should update the out_idx loop in the kernel too as it assumes size 1 (unrolled).
  TORCH_CHECK(value_bias.sizes() == at::IntArrayRef({1}), "value_bias must have shape [1]");
  TORCH_CHECK(values_out.sizes() == at::IntArrayRef({batch_size, 1}), "values_out must have shape [batch_size, 1]");


  // Grid covers the larger output dimension
  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dual_linear_forward_kernel<<<blocks, threads, 0, stream>>>(
    h2_in.data_ptr<float>(), h2_in.stride(0), h2_in.stride(1),                               // h2
    decoder_weights.data_ptr<float>(), decoder_weights.stride(0), decoder_weights.stride(1), // decoder_weights
    decoder_bias.data_ptr<float>(), decoder_out.data_ptr<float>(), decoder_out.stride(0),
    decoder_out.stride(1), // decoder bias/out
    value_weights.data_ptr<float>(), value_weights.stride(0), value_weights.stride(1),
    value_bias.data_ptr<float>(),                                             // values weights/bias
    values_out.data_ptr<float>(), values_out.stride(0), values_out.stride(1), // values out
    batch_size, hidden_size, decoder_weight_size, value_weights_size);

  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dual_linear_forward_kernel failed");
}

// Templatized version that hard-codes based on num_actions <= 5 for better performance.
template <int num_actions_t>
__global__ void
sample_logits_kernel(const float* __restrict__ logits, // [B, total_logits]
                     int64_t logits_stride0,
                     const float* __restrict__ random_vals,      // [B, num_actions] or [B] if num_actions==1
                     int64_t random_vals_stride,                 // num_actions if 2D, 1 if 1D
                     const int64_t* __restrict__ action_sizes,   // [num_actions] - size of each action dim
                     const int64_t* __restrict__ action_offsets, // [num_actions] - cumulative offset for each action
                     int32_t* __restrict__ actions,              // [B, num_actions] or [B] if num_actions==1
                     int64_t actions_stride0,                    // stride 0 for actions
                     int64_t actions_stride1,                    // stride 1 for actions (multidiscrete only)
                     float* __restrict__ logprobs,               // [B] output - sum of log probs
                     int64_t logprobs_stride,                    // stride for logprobs
                     int64_t batch_size, int num_actions_override)
{
  const auto num_actions = num_actions_t == 0 ? num_actions_override : num_actions_t;
  for (int64_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x; batch_idx < batch_size;
       batch_idx += static_cast<int64_t>(blockDim.x) * gridDim.x)
  {
    const float* my_logits = logits + batch_idx * logits_stride0;
    float total_logprob = 0.0f;

    for (int64_t a = 0; a < num_actions; ++a)
    {
      int64_t action_size = action_sizes[a];
      int64_t offset = action_offsets[a];
      const float* action_logits = my_logits + offset;

      // Find max for numerical stability
      float max_val = action_logits[0];
      for (int64_t i = 1; i < action_size; ++i)
      {
        max_val = fmaxf(max_val, action_logits[i]);
      }

      // Compute softmax denominator
      float sum_exp = 0.0f;
      for (int64_t i = 0; i < action_size; ++i)
      {
        sum_exp += expf(action_logits[i] - max_val);
      }

      // Sample from categorical
      float rand_val = random_vals[batch_idx * random_vals_stride];
      float total_sum = 0.0f;
      int32_t sampled_action = action_size - 1;

      for (int64_t i = 0; i < action_size; ++i)
      {
        float prob = expf(action_logits[i] - max_val) / sum_exp;
        total_sum += prob;
        if (rand_val < total_sum)
        {
          sampled_action = i;
          break; // WARP'ed: break is syntactic sugar for 'keep running this kernel SIMT, but noop'.
        }
      }

      // Store action using stride
      actions[batch_idx * actions_stride0 + a * actions_stride1] = sampled_action;

      // Accumulate log prob
      float log_prob = (action_logits[sampled_action] - max_val) - logf(sum_exp);
      total_logprob += log_prob;
    }
    logprobs[batch_idx * logprobs_stride] = total_logprob;
  }
}

void launch_sample_logits_kernel(const Tensor& random_vals, // [B, num_actions] or [B]
                                 const Tensor& sizes_gpu,   // [num_actions]
                                 const Tensor& offsets_gpu, // [num_actions]
                                 const Tensor& logits,      // [B, total_logits]
                                 int64_t num_actions,
                                 Tensor& actions,  // [B, num_actions] or [B]
                                 Tensor& logprobs) // [B]
{
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  TORCH_CHECK(random_vals.is_contiguous(), "random_vals must be contiguous");

  const auto batch_size = logits.size(0);
  if (num_actions > 1)
  {
    TORCH_CHECK(actions.sizes() == at::IntArrayRef({batch_size, num_actions}),
                "Multidiscrete actions must have shape [batch_size, num_actions]");
  }
  else
  {
    TORCH_CHECK(actions.sizes() == at::IntArrayRef({batch_size}), "Discrete actions must have shape [batch_size]");
  }
  TORCH_CHECK(random_vals.sizes() == at::IntArrayRef({batch_size}),
              "(Multi)Discrete random sampler must have shape [batch_size]");

  TORCH_CHECK(logprobs.sizes() == at::IntArrayRef({batch_size}),
              "logprobs (for discrete/multidiscrete) must have shape [batch_size]");
  TORCH_CHECK(logprobs.dtype() == torch::kFloat, "logprobs must be float32");
  TORCH_CHECK(actions.dtype() == torch::kInt32, "actions must be int32");
  TORCH_CHECK(random_vals.dtype() == torch::kFloat, "random_vals must be float32");
  TORCH_CHECK(logits.dtype() == torch::kFloat, "logits must be float32");

  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;

  const int64_t random_vals_stride = random_vals.stride(0);

  const int64_t actions_stride0 = actions.stride(0);
  const int64_t actions_stride1 = (actions.dim() == 1) ? 1 : actions.stride(1);
  const int64_t logprobs_stride = logprobs.stride(0);

  // printf("DEBUG launch_sample_logits_kernel: batch_size=%ld, blocks=%ld, logprobs.size(0)=%ld,
  // logprobs_stride=%ld\n",
  //       (long)batch_size, (long)blocks, (long)logprobs.size(0), (long)logprobs_stride);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (num_actions == 1) // discrete / breakout
  {
    sample_logits_kernel<1><<<blocks, threads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), random_vals.data_ptr<float>(), random_vals_stride,
      sizes_gpu.data_ptr<int64_t>(), offsets_gpu.data_ptr<int64_t>(), actions.data_ptr<int32_t>(), actions_stride0,
      actions_stride1, logprobs.data_ptr<float>(), logprobs_stride, batch_size, num_actions);
  }
  else if (num_actions == 2)
  {
    sample_logits_kernel<2><<<blocks, threads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), random_vals.data_ptr<float>(), random_vals_stride,
      sizes_gpu.data_ptr<int64_t>(), offsets_gpu.data_ptr<int64_t>(), actions.data_ptr<int32_t>(), actions_stride0,
      actions_stride1, logprobs.data_ptr<float>(), logprobs_stride, batch_size, num_actions);
  }
  else if (num_actions == 3) // multidiscrete / rlplays
  {
    sample_logits_kernel<3><<<blocks, threads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), random_vals.data_ptr<float>(), random_vals_stride,
      sizes_gpu.data_ptr<int64_t>(), offsets_gpu.data_ptr<int64_t>(), actions.data_ptr<int32_t>(), actions_stride0,
      actions_stride1, logprobs.data_ptr<float>(), logprobs_stride, batch_size, num_actions);
  }
  else if (num_actions == 4)
  {
    sample_logits_kernel<4><<<blocks, threads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), random_vals.data_ptr<float>(), random_vals_stride,
      sizes_gpu.data_ptr<int64_t>(), offsets_gpu.data_ptr<int64_t>(), actions.data_ptr<int32_t>(), actions_stride0,
      actions_stride1, logprobs.data_ptr<float>(), logprobs_stride, batch_size, num_actions);
  }
  else if (num_actions == 5)
  {
    sample_logits_kernel<5><<<blocks, threads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), random_vals.data_ptr<float>(), random_vals_stride,
      sizes_gpu.data_ptr<int64_t>(), offsets_gpu.data_ptr<int64_t>(), actions.data_ptr<int32_t>(), actions_stride0,
      actions_stride1, logprobs.data_ptr<float>(), logprobs_stride, batch_size, num_actions);
  }
  else
  {
    sample_logits_kernel<0><<<blocks, threads, 0, stream>>>(
      logits.data_ptr<float>(), logits.stride(0), random_vals.data_ptr<float>(), random_vals_stride,
      sizes_gpu.data_ptr<int64_t>(), offsets_gpu.data_ptr<int64_t>(), actions.data_ptr<int32_t>(), actions_stride0,
      actions_stride1, logprobs.data_ptr<float>(), logprobs_stride, batch_size, num_actions);
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "sample_logits_kernel failed");
}


// Copied from pufferlib.cpp


// CPU version:
void puff_advantage_row_cpu(float* values, float* rewards, float* dones, float* importance, float* advantages, float gamma,
                        float lambda, float rho_clip, float c_clip, int horizon)
{
  float lastpufferlam = 0;
  for (int t = horizon - 2; t >= 0; t--)
  {
    int t_next = t + 1;
    float nextnonterminal = 1.0 - dones[t_next];
    float rho_t = fminf(importance[t], rho_clip);
    float c_t = fminf(importance[t], c_clip);
    float delta = rho_t * (rewards[t_next] + gamma * values[t_next] * nextnonterminal - values[t]);
    lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
    advantages[t] = lastpufferlam;
  }
}

void vtrace_check_cpu(torch::Tensor values, torch::Tensor rewards, torch::Tensor dones, torch::Tensor importance,
                  torch::Tensor advantages, int num_steps, int horizon)
{

  // Validate input tensors
  torch::Device device = torch::kCPU;
  for (const torch::Tensor& t : {values, rewards, dones, importance, advantages})
  {
    TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
    TORCH_CHECK(t.device() == device, "All tensors must be on the same device: CPU");
    TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
    TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
    TORCH_CHECK(t.is_contiguous(), "All tensors must be contiguous");
  }
}


// [num_steps, horizon]
void puff_advantage_cpu(float* values, float* rewards, float* dones, float* importance, float* advantages, float gamma,
                    float lambda, float rho_clip, float c_clip, int num_steps, const int horizon)
{
  for (int offset = 0; offset < num_steps * horizon; offset += horizon)
  {
    puff_advantage_row_cpu(values + offset, rewards + offset, dones + offset, importance + offset, advantages + offset,
                       gamma, lambda, rho_clip, c_clip, horizon);
  }
}


void compute_puff_advantage_cpu(torch::Tensor values, torch::Tensor rewards, torch::Tensor dones,
                                torch::Tensor importance, torch::Tensor& advantages_out, double gamma, double lambda,
                                double rho_clip, double c_clip)
{
  int num_steps = values.size(0);
  int horizon = values.size(1);
  // TODO: optimize next. Prevent trampolining cpu <-> gpu here.
  importance = importance.to(torch::kCPU);
  advantages_out = advantages_out.to(torch::kCPU);

  vtrace_check_cpu(values, rewards, dones, importance, advantages_out, num_steps, horizon);
  puff_advantage_cpu(values.data_ptr<float>(), rewards.data_ptr<float>(), dones.data_ptr<float>(),
                 importance.data_ptr<float>(), advantages_out.data_ptr<float>(), gamma, lambda, rho_clip, c_clip, num_steps,
                 horizon);
  // Move back to original device.
  advantages_out = advantages_out.to(torch::kCUDA);
}



__host__ __device__ void puff_advantage_row_cuda(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

void vtrace_check_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}

 // [num_steps, horizon]
__global__ void puff_advantage_kernel(float* values, float* rewards,
        float* dones, float* importance, float* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) {
        return;
    }
    int offset = row*horizon;
    puff_advantage_row_cuda(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda(values, rewards, dones, importance, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    puff_advantage_kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        importance.data_ptr<float>(),
        advantages.data_ptr<float>(),
        gamma,
        lambda,
        rho_clip,
        c_clip,
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
