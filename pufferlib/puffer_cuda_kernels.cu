#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/macros/Macros.h>

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <ATen/cuda/detail/TensorInfo.cuh>

using at::Tensor;
using at::cuda::detail::TensorInfo;

// Kernel: each thread computes one batch of output elements @ (batch_idx, out_idx)
// Supports strided (non-contiguous) 2D tensors via explicit strides (in element counts).
__global__ void linear_forward_kernel_strided(const float* __restrict__ input,
                                              int64_t input_stride0,
                                              int64_t input_stride1,
                                              const float* __restrict__ weight,
                                              int64_t weight_stride0,
                                              int64_t weight_stride1,
                                              const float* __restrict__ bias,
                                              float* __restrict__ output,
                                              int64_t output_stride0,
                                              int64_t output_stride1,
                                              int64_t batch_size,
                                              int64_t in_features,
                                              int64_t out_features)
{
  for (int64_t batch_idx = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
       batch_idx < batch_size;
       batch_idx += static_cast<int64_t>(blockDim.y) * gridDim.y)
  {
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < out_features;
         out_idx += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
      float sum = 0.0f;

      const int64_t input_base = batch_idx * input_stride0;
      const int64_t weight_base = out_idx * weight_stride0;

      for (int64_t i = 0; i < in_features; ++i)
      {
        const float x = input[input_base + i * input_stride1];
        const float w = weight[weight_base + i * weight_stride1];
        sum += x * w;
      }

      sum += bias[out_idx];

      output[batch_idx * output_stride0 + out_idx * output_stride1] = sum;
    }
  }
}

static void CHECK_PARAMS(const Tensor& input,  // [B, In]
                         const Tensor& weight, // [Out, In]
                         const Tensor& bias,   // [Out] or empty
                         Tensor& output)       // [B, Out], preallocated
{
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
  TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");

  TORCH_CHECK(input.dim() == 2, "input must be 2D [B, In]");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [Out, In]");
  TORCH_CHECK(output.dim() == 2, "output must be 2D [B, Out]");
  TORCH_CHECK(weight.size(1) == input.size(1), "weight.shape[1] (in_features) must match input.shape[1]");
  TORCH_CHECK(output.size(0) == input.size(0), "output.shape[0] must match input.shape[0]");
  TORCH_CHECK(output.size(1) == weight.size(0), "output.shape[1] must match weight.shape[0]");

  TORCH_CHECK(bias.defined(), "bias must be defined");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
  TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D [Out]");
  TORCH_CHECK(bias.size(0) == weight.size(0), "bias.shape[0] must match weight.shape[0]");
}

void launch_linear_forward(const Tensor& input,  // [B, In]
                           const Tensor& weight, // [Out, In]
                           const Tensor& bias,   // [Out]
                           Tensor& output)       // [B, Out], preallocated
{
  CHECK_PARAMS(input, weight, bias, output);

  const auto batch_size = input.size(0);
  const auto in_features = input.size(1);
  const auto out_features = weight.size(0);

  const float* input_ptr = input.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();

  const int64_t input_stride0 = input.stride(0);
  const int64_t input_stride1 = input.stride(1);
  const int64_t weight_stride0 = weight.stride(0);
  const int64_t weight_stride1 = weight.stride(1);
  const int64_t output_stride0 = output.stride(0);
  const int64_t output_stride1 = output.stride(1);

  // 2D grid: (out_features, batch_size).
  const dim3 block_dim(16, 16);
  const dim3 grid_dim(static_cast<unsigned int>((out_features + block_dim.x - 1) / block_dim.x),
                      static_cast<unsigned int>((batch_size + block_dim.y - 1) / block_dim.y));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  linear_forward_kernel_strided<<<grid_dim, block_dim, 0, stream>>>(
    input_ptr, input_stride0, input_stride1,
    weight_ptr, weight_stride0, weight_stride1,
    bias_ptr,
    output_ptr, output_stride0, output_stride1,
    batch_size, in_features, out_features);

  const auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "linear_forward_kernel launch failed: ", cudaGetErrorString(err));
}


// Code copied from libtorch. See LICENSE file in pytorch root directory; also included in the main PufferLib LICENSE file.

using at::cuda::detail::TensorInfo;
using at::cuda::detail::getTensorInfo;
using at::cuda::detail::IndexToOffset;
using at::cuda::detail::canUse32BitIndexMath;

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}
// Threads per block for our apply kernel
// FIXME: use occupancy calculator instead
constexpr uint32_t AT_APPLY_THREADS_PER_BLOCK = 512;
//constexpr uint32_t AT_APPLY_BLOCKS_PER_SM = 4; // WARNING: unused 

bool allContiguous(at::TensorList tensors) {
  return std::all_of(tensors.begin(), tensors.end(),
                     [](const at::Tensor& t) { return !t.defined() || t.is_contiguous(); });
}

template <int step = 1>
inline bool getApplyGrid(uint64_t totalElements, dim3& grid, c10::DeviceIndex curDevice, int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK) {
  if (curDevice == -1) return false;
  uint64_t numel_per_thread = static_cast<uint64_t>(max_threads_per_block) * static_cast<uint64_t>(step);
  uint64_t numBlocks = ATenCeilDiv(totalElements, numel_per_thread);
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
    numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

inline dim3 getApplyBlock(int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK) {
  return dim3(max_threads_per_block);
}

void getLaunchConfig(dim3* block, dim3* grid, int64_t numel) {
  c10::DeviceIndex curDevice = -1;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&curDevice));
  *block = getApplyBlock();
  TORCH_CHECK(getApplyGrid(numel, *grid, curDevice), "Could not get grid size for pointwise apply.");
}

template<typename T, typename T2>
TensorInfo<T, T2> tryGetTensorInfo(const at::Tensor& t) {
  return t.defined() ? getTensorInfo<T, T2>(t) : TensorInfo<T, T2>{};
}

void collapseDims() {};
template<typename T, typename T2, typename... Args>
void collapseDims(TensorInfo<T, T2>& info, Args&... infos) {
  info.collapseDims();
  collapseDims(infos...);
}

#define DEVICE_LINEAR_GET(D_TENSOR, INDEX)                              \
  D_TENSOR.data[IndexToOffset<scalar_t, index_type, indexing_kind>::get(INDEX, D_TENSOR)]

// Biases are always 1D
#define DEVICE_BIAS_GET(D_TENSOR, INDEX)                              \
  D_TENSOR.data[IndexToOffset<scalar_t, index_type, 1>::get(INDEX, D_TENSOR)]

#define H2F(input) static_cast<accscalar_t>(input)
#define F2H(input) static_cast<scalar_t>(input)


template<typename T>
__device__ __forceinline__
T sigmoid(T in)  {
  T one = static_cast<T>(1.0);
  return one / (one + ::exp(-in));
}
namespace kernel {

template <typename scalar_t, typename accscalar_t, typename index_type, int indexing_kind>
C10_LAUNCH_BOUNDS_2(512, 4)
__global__ void lstm_cell_forward(
            TensorInfo<scalar_t, index_type> input,
            TensorInfo<scalar_t, index_type> hidden,
            TensorInfo<scalar_t, index_type> bias1,
            TensorInfo<scalar_t, index_type> bias2,
            TensorInfo<scalar_t, index_type> _cx,
            TensorInfo<scalar_t, index_type> _hy,
            TensorInfo<scalar_t, index_type> _cy,
            TensorInfo<scalar_t, index_type> workspace,
            index_type hsz,
            index_type totalElements) {
    bool has_bias = bias1.data != nullptr;
    for (index_type linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
      index_type offset = (linearIndex/hsz)*4*hsz+linearIndex%hsz;

      scalar_t iig = DEVICE_LINEAR_GET(input, offset+0*hsz);
      scalar_t ifg = DEVICE_LINEAR_GET(input, offset+1*hsz);
      scalar_t icg = DEVICE_LINEAR_GET(input, offset+2*hsz);
      scalar_t iog = DEVICE_LINEAR_GET(input, offset+3*hsz);

      scalar_t hig = DEVICE_LINEAR_GET(hidden, offset+0*hsz);
      scalar_t hfg = DEVICE_LINEAR_GET(hidden, offset+1*hsz);
      scalar_t hcg = DEVICE_LINEAR_GET(hidden,  offset+2*hsz);
      scalar_t hog = DEVICE_LINEAR_GET(hidden,  offset+3*hsz);

      scalar_t* wig = &DEVICE_LINEAR_GET(workspace, offset+0*hsz);
      scalar_t* wfg = &DEVICE_LINEAR_GET(workspace, offset+1*hsz);
      scalar_t* wcg = &DEVICE_LINEAR_GET(workspace, offset+2*hsz);
      scalar_t* wog = &DEVICE_LINEAR_GET(workspace, offset+3*hsz);

      scalar_t cx = DEVICE_LINEAR_GET(_cx, linearIndex);

      scalar_t* hy = &DEVICE_LINEAR_GET(_hy, linearIndex);
      scalar_t* cy = &DEVICE_LINEAR_GET(_cy, linearIndex);

      scalar_t b1i, b1f, b1c, b1o;
      scalar_t b2i, b2f, b2c, b2o;

      if (has_bias) {
        b1i = DEVICE_BIAS_GET(bias1, linearIndex % hsz + 0 * hsz);
        b1f = DEVICE_BIAS_GET(bias1, linearIndex % hsz + 1 * hsz);
        b1c = DEVICE_BIAS_GET(bias1, linearIndex % hsz + 2 * hsz);
        b1o = DEVICE_BIAS_GET(bias1, linearIndex % hsz + 3 * hsz);

        b2i = DEVICE_BIAS_GET(bias2, linearIndex % hsz + 0 * hsz);
        b2f = DEVICE_BIAS_GET(bias2, linearIndex % hsz + 1 * hsz);
        b2c = DEVICE_BIAS_GET(bias2, linearIndex % hsz + 2 * hsz);
        b2o = DEVICE_BIAS_GET(bias2, linearIndex % hsz + 3 * hsz);
      } else {
#ifndef THC_REAL_IS_HALF
        b1i = 0.0; b1f = 0.0; b1c = 0.0; b1o = 0.0;
        b2i = 0.0; b2f = 0.0; b2c = 0.0; b2o = 0.0;
#else
        b1i = F2H(0.0); b1f = F2H(0.0); b1c = F2H(0.0); b1o = F2H(0.0);
        b2i = F2H(0.0); b2f = F2H(0.0); b2c = F2H(0.0); b2o = F2H(0.0);
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

      //SAVE FOR BACKWARDS
      //Also need cy and cx but can be saved easily in python
      *wig = F2H(ig);
      *wfg = F2H(fg);
      *wcg = F2H(cg);
      *wog = F2H(og);
    }
}
}

void lstm_forward_impl(const Tensor& input_gates, const Tensor& hidden_gates,
                       const Tensor& input_bias, const Tensor& hidden_bias,
                       const Tensor& cx,
                       const Tensor& hy, const Tensor& cy, const Tensor& workspace) {

  dim3 block, grid;
  int64_t numel = cx.numel();
  if (numel == 0) return;
  getLaunchConfig(&block, &grid, numel);

  auto input_gatesI = getTensorInfo<float, size_t>(input_gates);
  auto hidden_gatesI = getTensorInfo<float, size_t>(hidden_gates);
  auto input_biasI = tryGetTensorInfo<float, size_t>(input_bias);
  auto hidden_biasI = tryGetTensorInfo<float, size_t>(hidden_bias);
  auto cxI = getTensorInfo<float, size_t>(cx);
  auto hyI = getTensorInfo<float, size_t>(hy);
  auto cyI = getTensorInfo<float, size_t>(cy);
  auto workspaceI = getTensorInfo<float, size_t>(workspace);
  size_t hidden_size = cxI.sizes[cxI.dims-1];

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (allContiguous({input_gates, hidden_gates, input_bias, hidden_bias, cx, hy, cy, workspace})) {
    collapseDims(input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI);
    kernel::lstm_cell_forward<float, float, size_t, 1>
      <<<grid, block, 0, stream>>>
        (input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI, hidden_size, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    kernel::lstm_cell_forward<float, float, size_t, 2><<<grid, block, 0, stream>>>
        (input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI, hidden_size, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

/**
 * 
 * Used Opus 4.5 to generate the fused kernels using the prompt below:
 * I want to fuse as many of these kernels as possible to minimize cuda kernel launch time. 
 * I already have puffer_cuda_kernels with launch_linear_forward. 
 * Propose a plan to fuse these kernels and implement them.
 */

// =============================================================================
// Fusion 1: Dual Linear Forward - computes two independent linear layers on same input
// Useful for decoder + value head which both take h2 as input
// =============================================================================
__global__ void dual_linear_forward_kernel(
    const float* __restrict__ input,       // [B, In]
    int64_t input_stride0,
    int64_t input_stride1,
    const float* __restrict__ weight1,     // [Out1, In] - decoder
    int64_t weight1_stride0,
    int64_t weight1_stride1,
    const float* __restrict__ bias1,       // [Out1]
    float* __restrict__ output1,           // [B, Out1]
    int64_t output1_stride0,
    int64_t output1_stride1,
    const float* __restrict__ weight2,     // [Out2, In] - value
    int64_t weight2_stride0,
    int64_t weight2_stride1,
    const float* __restrict__ bias2,       // [Out2]
    float* __restrict__ output2,           // [B, Out2]
    int64_t output2_stride0,
    int64_t output2_stride1,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features1,
    int64_t out_features2)
{
  // Grid-stride loop over batch dimension
  for (int64_t batch_idx = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
       batch_idx < batch_size;
       batch_idx += static_cast<int64_t>(blockDim.y) * gridDim.y)
  {
    const int64_t input_base = batch_idx * input_stride0;
    
    // Compute output1 (decoder) elements
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < out_features1;
         out_idx += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
      float sum = 0.0f;
      const int64_t weight_base = out_idx * weight1_stride0;
      
      for (int64_t i = 0; i < in_features; ++i)
      {
        sum += input[input_base + i * input_stride1] * 
               weight1[weight_base + i * weight1_stride1];
      }
      sum += bias1[out_idx];
      output1[batch_idx * output1_stride0 + out_idx * output1_stride1] = sum;
    }
    
    // Compute output2 (value) elements - typically much smaller (out_features2 = 1)
    for (int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         out_idx < out_features2;
         out_idx += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
      float sum = 0.0f;
      const int64_t weight_base = out_idx * weight2_stride0;
      
      for (int64_t i = 0; i < in_features; ++i)
      {
        sum += input[input_base + i * input_stride1] * 
               weight2[weight_base + i * weight2_stride1];
      }
      sum += bias2[out_idx];
      output2[batch_idx * output2_stride0 + out_idx * output2_stride1] = sum;
    }
  }
}

void launch_dual_linear_forward(
    const Tensor& input,    // [B, In]
    const Tensor& weight1,  // [Out1, In] - decoder
    const Tensor& bias1,    // [Out1]
    Tensor& output1,        // [B, Out1]
    const Tensor& weight2,  // [Out2, In] - value  
    const Tensor& bias2,    // [Out2]
    Tensor& output2)        // [B, Out2]
{
  // Validation
  TORCH_CHECK(input.is_cuda() && weight1.is_cuda() && weight2.is_cuda(), "All tensors must be CUDA");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  
  const auto batch_size = input.size(0);
  const auto in_features = input.size(1);
  const auto out_features1 = weight1.size(0);
  const auto out_features2 = weight2.size(0);
  
  // Grid covers the larger output dimension
  const int64_t max_out = std::max(out_features1, out_features2);
  const dim3 block_dim(16, 16);
  const dim3 grid_dim(
      static_cast<unsigned int>((max_out + block_dim.x - 1) / block_dim.x),
      static_cast<unsigned int>((batch_size + block_dim.y - 1) / block_dim.y));
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  dual_linear_forward_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input.data_ptr<float>(), input.stride(0), input.stride(1),
      weight1.data_ptr<float>(), weight1.stride(0), weight1.stride(1),
      bias1.data_ptr<float>(),
      output1.data_ptr<float>(), output1.stride(0), output1.stride(1),
      weight2.data_ptr<float>(), weight2.stride(0), weight2.stride(1),
      bias2.data_ptr<float>(),
      output2.data_ptr<float>(), output2.stride(0), output2.stride(1),
      batch_size, in_features, out_features1, out_features2);
  
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dual_linear_forward_kernel failed");
}

// =============================================================================
// Fusion 2: Fused LSTM with integrated matmuls
// Combines: igates = input @ W_ih.T, hgates = hidden @ W_hh.T, lstm_cell
// =============================================================================
__global__ void fused_lstm_cell_kernel(
    const float* __restrict__ input,       // [B, input_size] - encoder output
    int64_t input_stride0,
    int64_t input_stride1,
    const float* __restrict__ hidden,      // [B, hidden_size] - previous h
    int64_t hidden_stride0,
    int64_t hidden_stride1,
    const float* __restrict__ weight_ih,   // [4*hidden_size, input_size]
    int64_t weight_ih_stride0,
    int64_t weight_ih_stride1,
    const float* __restrict__ weight_hh,   // [4*hidden_size, hidden_size]
    int64_t weight_hh_stride0,
    int64_t weight_hh_stride1,
    const float* __restrict__ bias_ih,     // [4*hidden_size]
    const float* __restrict__ bias_hh,     // [4*hidden_size]
    const float* __restrict__ cx,          // [B, hidden_size]
    int64_t cx_stride0,
    int64_t cx_stride1,
    float* __restrict__ hy,                // [B, hidden_size]
    int64_t hy_stride0,
    int64_t hy_stride1,
    float* __restrict__ cy,                // [B, hidden_size]
    int64_t cy_stride0,
    int64_t cy_stride1,
    int64_t batch_size,
    int64_t input_size,
    int64_t hidden_size)
{
  // Each thread handles one (batch, hidden_idx) pair
  for (int64_t batch_idx = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
       batch_idx < batch_size;
       batch_idx += static_cast<int64_t>(blockDim.y) * gridDim.y)
  {
    for (int64_t h_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         h_idx < hidden_size;
         h_idx += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
      // Compute all 4 gates for this hidden index
      float gates[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      
      const int64_t input_base = batch_idx * input_stride0;
      const int64_t hidden_base = batch_idx * hidden_stride0;
      
      // Compute input contribution to gates (input @ W_ih.T)
      for (int64_t i = 0; i < input_size; ++i)
      {
        float x = input[input_base + i * input_stride1];
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
          int64_t gate_idx = g * hidden_size + h_idx;
          gates[g] += x * weight_ih[gate_idx * weight_ih_stride0 + i * weight_ih_stride1];
        }
      }
      
      // Compute hidden contribution to gates (hidden @ W_hh.T)
      for (int64_t i = 0; i < hidden_size; ++i)
      {
        float h = hidden[hidden_base + i * hidden_stride1];
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
          int64_t gate_idx = g * hidden_size + h_idx;
          gates[g] += h * weight_hh[gate_idx * weight_hh_stride0 + i * weight_hh_stride1];
        }
      }
      
      // Add biases
      #pragma unroll
      for (int g = 0; g < 4; ++g)
      {
        int64_t gate_idx = g * hidden_size + h_idx;
        gates[g] += bias_ih[gate_idx] + bias_hh[gate_idx];
      }
      
      // Apply activations: i, f, o = sigmoid; g = tanh
      float i_gate = 1.0f / (1.0f + expf(-gates[0]));  // input gate
      float f_gate = 1.0f / (1.0f + expf(-gates[1]));  // forget gate
      float g_gate = tanhf(gates[2]);                   // cell gate
      float o_gate = 1.0f / (1.0f + expf(-gates[3]));  // output gate
      
      // Compute new cell state
      float c_prev = cx[batch_idx * cx_stride0 + h_idx * cx_stride1];
      float c_new = f_gate * c_prev + i_gate * g_gate;
      
      // Compute new hidden state
      float h_new = o_gate * tanhf(c_new);
      
      // Write outputs
      cy[batch_idx * cy_stride0 + h_idx * cy_stride1] = c_new;
      hy[batch_idx * hy_stride0 + h_idx * hy_stride1] = h_new;
    }
  }
}

void launch_fused_lstm_cell(
    const Tensor& input,      // [B, input_size]   [ hidden out = [N envs, input_size = 128]]
    const Tensor& hidden,     // [B, hidden_size]  [ h1 = previous hidden = [N envs, hidden_size = 128] ]
    const Tensor& weight_ih,  // [4*hidden_size, input_size] [512, 128]
    const Tensor& weight_hh,  // [4*hidden_size, hidden_size] [512, 128]
    const Tensor& bias_ih,    // [4*hidden_size] [512, 128]
    const Tensor& bias_hh,    // [4*hidden_size] [512, 128]
    const Tensor& cx,         // [B, hidden_size] [ c1 = previous hidden = [N envs, hidden_size = 128] ]
    Tensor& hy,               // [B, hidden_size] [ h2 = new hidden = [N envs, hidden_size = 128] ]
    Tensor& cy)               // [B, hidden_size] [ c2 = new hidden = [N envs, hidden_size = 128] ]
{
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  
  const auto num_envs = input.size(0);
  const auto input_size = input.size(1);
  const auto hidden_size = hidden.size(1);
  const auto min_dim_x = std::min(32, int(hidden_size));
  const auto min_dim_y = std::min(16, int(num_envs));
  const dim3 block_dim(min_dim_x, min_dim_y);
  const dim3 grid_dim(
      static_cast<unsigned int>((hidden_size + block_dim.x - 1) / block_dim.x),
      static_cast<unsigned int>((num_envs + block_dim.y - 1) / block_dim.y));
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  fused_lstm_cell_kernel<<<grid_dim, block_dim, 0, stream>>>(
      input.data_ptr<float>(), input.stride(0), input.stride(1),
      hidden.data_ptr<float>(), hidden.stride(0), hidden.stride(1),
      weight_ih.data_ptr<float>(), weight_ih.stride(0), weight_ih.stride(1),
      weight_hh.data_ptr<float>(), weight_hh.stride(0), weight_hh.stride(1),
      bias_ih.data_ptr<float>(),
      bias_hh.data_ptr<float>(),
      cx.data_ptr<float>(), cx.stride(0), cx.stride(1),
      hy.data_ptr<float>(), hy.stride(0), hy.stride(1),
      cy.data_ptr<float>(), cy.stride(0), cy.stride(1),
      num_envs, input_size, hidden_size);
  
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fused_lstm_cell_kernel failed");
}

// =============================================================================
// Fusion 3: Linear + Categorical Sampling (decoder + sample in one kernel)
// =============================================================================
__global__ void linear_sample_kernel(
    const float* __restrict__ input,       // [B, In]
    int64_t input_stride0,
    int64_t input_stride1,
    const float* __restrict__ weight,      // [Out, In]
    int64_t weight_stride0,
    int64_t weight_stride1,
    const float* __restrict__ bias,        // [Out]
    const float* __restrict__ random_vals, // [B, num_actions] - uniform random [0,1)
    int64_t* __restrict__ actions,         // [B, num_actions] output
    float* __restrict__ logprobs,          // [B] output - sum of log probs
    int64_t batch_size,
    int64_t in_features,
    int64_t num_actions,
    int64_t action_size)                   // logits per action (assumes uniform)
{
  extern __shared__ float shared_logits[];
  
  // Each block handles one batch element
  int64_t batch_idx = blockIdx.x;
  if (batch_idx >= batch_size) return;
  
  const int64_t total_logits = num_actions * action_size;
  float* my_logits = shared_logits;
  
  // Step 1: Compute logits (linear layer) cooperatively
  for (int64_t out_idx = threadIdx.x; out_idx < total_logits; out_idx += blockDim.x)
  {
    float sum = bias[out_idx];
    const int64_t input_base = batch_idx * input_stride0;
    const int64_t weight_base = out_idx * weight_stride0;
    
    for (int64_t i = 0; i < in_features; ++i)
    {
      sum += input[input_base + i * input_stride1] * 
             weight[weight_base + i * weight_stride1];
    }
    my_logits[out_idx] = sum;
  }
  __syncthreads();
  
  // Step 2: For each action, compute softmax and sample
  // Thread 0 handles the sequential sampling (could parallelize with care)
  if (threadIdx.x == 0)
  {
    float total_logprob = 0.0f;
    
    for (int64_t a = 0; a < num_actions; ++a)
    {
      float* action_logits = my_logits + a * action_size;
      
      // Find max for numerical stability
      float max_val = action_logits[0];
      for (int64_t i = 1; i < action_size; ++i)
      {
        max_val = fmaxf(max_val, action_logits[i]);
      }
      
      // Compute exp and sum
      float sum_exp = 0.0f;
      for (int64_t i = 0; i < action_size; ++i)
      {
        action_logits[i] = expf(action_logits[i] - max_val);
        sum_exp += action_logits[i];
      }
      
      // Sample from categorical
      float rand_val = random_vals[batch_idx * num_actions + a];
      float cumsum = 0.0f;
      int64_t sampled_action = action_size - 1;  // default to last
      
      for (int64_t i = 0; i < action_size; ++i)
      {
        float prob = action_logits[i] / sum_exp;
        cumsum += prob;
        if (rand_val < cumsum)
        {
          sampled_action = i;
          break;
        }
      }
      
      // Store action and accumulate log prob
      if (num_actions == 1)
      {
        actions[batch_idx] = sampled_action;
      }
      else
      {
        actions[batch_idx * num_actions + a] = sampled_action;
      }
      
      float log_prob = logf(action_logits[sampled_action] / sum_exp);
      total_logprob += log_prob;
    }
    
    logprobs[batch_idx] = total_logprob;
  }
}

void launch_linear_sample(
    const Tensor& input,       // [B, In]
    const Tensor& weight,      // [Out, In]
    const Tensor& bias,        // [Out]
    const Tensor& random_vals, // [B, num_actions]
    Tensor& actions,           // [B] or [B, num_actions]
    Tensor& logprobs,          // [B]
    int64_t num_actions,
    int64_t action_size)
{
  const auto batch_size = input.size(0);
  const auto in_features = input.size(1);
  const auto total_logits = num_actions * action_size;
  
  // One block per batch element, shared memory for logits
  const int threads = std::min(256L, (long)total_logits);
  const size_t shared_mem = total_logits * sizeof(float);
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  linear_sample_kernel<<<batch_size, threads, shared_mem, stream>>>(
      input.data_ptr<float>(), input.stride(0), input.stride(1),
      weight.data_ptr<float>(), weight.stride(0), weight.stride(1),
      bias.data_ptr<float>(),
      random_vals.data_ptr<float>(),
      actions.data_ptr<int64_t>(),
      logprobs.data_ptr<float>(),
      batch_size, in_features, num_actions, action_size);
  
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "linear_sample_kernel failed");
}