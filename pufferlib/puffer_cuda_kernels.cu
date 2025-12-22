#include <cuda_runtime.h>
#include <torch/torch.h>
#include <puffer_cuda.h>

// Kernel: each thread computes one output element (batch_idx, out_idx)
__global__ void linear_forward_kernel(
    const float* __restrict__ input,   // [B, In]
    const float* __restrict__ weight,  // [Out, In]
    const float* __restrict__ bias,    // [Out]
    float* __restrict__ output,        // [B, Out]
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x; // column in output
    const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y; // row in output

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

    if (bias != nullptr)
    {
        sum += bias[out_idx];
    }

    // output[b, o] = sum
    output[batch_idx * out_features + out_idx] = sum;
}

// Host-side launcher: expects CUDA tensors and writes result into `output`.
void launch_linear_forward(
    const at::Tensor& input,    // [B, In]
    const at::Tensor& weight,   // [Out, In]
    const at::Tensor& bias,     // [Out] or empty
    at::Tensor& output,         // [B, Out], preallocated
    cudaStream_t stream)
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

    const auto batch_size = input.size(0);
    const auto in_features = input.size(1);
    const auto out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features,
        "weight.shape[1] (in_features) must match input.shape[1]");
    TORCH_CHECK(output.size(0) == batch_size,
        "output.shape[0] must match input.shape[0]");
    TORCH_CHECK(output.size(1) == out_features,
        "output.shape[1] must match weight.shape[0]");

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.defined() && bias.numel() > 0
        ? bias.data_ptr<float>()
        : nullptr;
    float* output_ptr = output.data_ptr<float>();

    // 2D grid: (out_features, batch_size)
    const dim3 block_dim(16, 16);
    const dim3 grid_dim(
        static_cast<unsigned int>((out_features + block_dim.x - 1) / block_dim.x),
        static_cast<unsigned int>((batch_size   + block_dim.y - 1) / block_dim.y));

    linear_forward_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_features,
        out_features);

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "linear_forward_kernel launch failed: ",
        cudaGetErrorString(err));
}

