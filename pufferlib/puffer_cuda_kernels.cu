#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/macros/Macros.h>

#include <cuda_runtime.h>
#include <torch/torch.h>
// #include <puffer_cuda.h>

// Some of this was written with Claude Sonnet 4.5 help.



// Kernel: each thread computes one output element (batch_idx, out_idx)
__global__ void linear_forward_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                                      const float* __restrict__ bias, float* __restrict__ output, int64_t batch_size,
                                      int64_t in_features, int64_t out_features)
{
  const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;   // column in output
  const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y; // row in output
  // Branches are expensive, so we better ensure batch_size/out_features are (approximate) multiples of blockDim.x/y
  // This ensures that if partial blocks are launched, the threads outside the valid range exit early.
  if (batch_idx >= batch_size || out_idx >= out_features)
  {
    return;
  }

  // Row-major: input[b, i] = input[b * in_features + i]
  // weight[o, i] = weight[o * in_features + i]
  float sum = 0.0f;
  const int64_t input_row_offset = batch_idx * in_features;
  const int64_t weight_row_offset = out_idx * in_features;

  for (int64_t i = 0; i < in_features; ++i)
  {
    sum += input[input_row_offset + i] * weight[weight_row_offset + i];
  }

  sum += bias[out_idx];

  // output[b, o] = sum
  output[batch_idx * out_features + out_idx] = sum;
}

void CHECK_PARAMS(const at::Tensor& input,  // [B, In]
                  const at::Tensor& weight, // [Out, In]
                  const at::Tensor& bias,   // [Out] or empty
                  at::Tensor& output)       // [B, Out], preallocated

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
}

void launch_linear_forward(const at::Tensor& input,  // [B, In]
                           const at::Tensor& weight, // [Out, In]
                           const at::Tensor& bias,   // [Out] or empty
                           at::Tensor& output,       // [B, Out], preallocated
                           cudaStream_t stream)
{
  CHECK_PARAMS(input, weight, bias, output);
  const auto batch_size = input.size(0);
  const auto in_features = input.size(1);
  const auto out_features = weight.size(0);

  const float* input_ptr = input.data_ptr<float>();
  const float* weight_ptr = weight.data_ptr<float>();
  const float* bias_ptr = bias.data_ptr<float>();
  float* output_ptr = output.data_ptr<float>();

  // 2D grid: (out_features, batch_size). This works reasonably well for things like encoder, value layers
  // because out_features and batch_size are often in the hundreds. For the decoder though, because of
  // num_atns_heads * head_dim, out_features can be small (e.g., 64), so performance may be suboptimal.
  // Use addmm_out instead.
  const dim3 block_dim(16, 16);
  const dim3 grid_dim(static_cast<unsigned int>((out_features + block_dim.x - 1) / block_dim.x),
                      static_cast<unsigned int>((batch_size + block_dim.y - 1) / block_dim.y));

  linear_forward_kernel<<<grid_dim, block_dim, 0, stream>>>(input_ptr, weight_ptr, bias_ptr, output_ptr, batch_size,
                                                            in_features, out_features);
  const auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "linear_forward_kernel launch failed: ", cudaGetErrorString(err));
}



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

// Code copied from libtorch
// TODO: Update LICENSE.txt


template<typename scalar_t, typename index_type>
void lstm_forward_impl(const Tensor& input_gates, const Tensor& hidden_gates,
                       const Tensor& input_bias, const Tensor& hidden_bias,
                       const Tensor& cx,
                       const Tensor& hy, const Tensor& cy, const Tensor& workspace) {
  using accscalar_t = acc_type<scalar_t, /*is_cuda=*/true>;

  dim3 block, grid;
  int64_t numel = cx.numel();
  if (numel == 0) return;
  getLaunchConfig(&block, &grid, numel);

  auto input_gatesI = getTensorInfo<scalar_t, index_type>(input_gates);
  auto hidden_gatesI = getTensorInfo<scalar_t, index_type>(hidden_gates);
  auto input_biasI = tryGetTensorInfo<scalar_t, index_type>(input_bias);
  auto hidden_biasI = tryGetTensorInfo<scalar_t, index_type>(hidden_bias);
  auto cxI = getTensorInfo<scalar_t, index_type>(cx);
  auto hyI = getTensorInfo<scalar_t, index_type>(hy);
  auto cyI = getTensorInfo<scalar_t, index_type>(cy);
  auto workspaceI = getTensorInfo<scalar_t, index_type>(workspace);
  index_type hidden_size = cxI.sizes[cxI.dims-1];

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (allContiguous({input_gates, hidden_gates, input_bias, hidden_bias, cx, hy, cy, workspace})) {
    collapseDims(input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI);
    kernel::lstm_cell_forward<scalar_t, accscalar_t, index_type, 1>
      <<<grid, block, 0, stream>>>
        (input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI, hidden_size, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    kernel::lstm_cell_forward<scalar_t, accscalar_t, index_type, 2>
      <<<grid, block, 0, stream>>>
        (input_gatesI, hidden_gatesI, input_biasI, hidden_biasI, cxI, hyI, cyI, workspaceI, hidden_size, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}