// profile_kernels.cu
// Single-file profiling harness for PufferLib CUDA kernels and training pipeline
//
// Build without torch:
//   nvcc -O3 -arch=sm_89 -DPRECISION_FLOAT -I. profile_kernels.cu -o profile
//
// Build with torch (use setup.py):
//   python setup.py build_profiler --env=breakout
//
// Usage:
//   ./profile <profile>
//   ./profile kernels          # All individual kernel profiles
//   ./profile trainforward     # Training forward+backward breakdown
//   ./profile trainstep        # Full training step with Muon optimizer
//   ./profile rolloutcopy      # Per-minibatch data prep: advantage+prio+copy
//   ./profile forwardcall      # Inference forward pass
//   ./profile envspeed         # Environment throughput
//   ./profile all              # Everything

// ============================================================================
// Section 1: Shared infrastructure
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <chrono>


#ifdef USE_TORCH
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAGraph.h>
#include "pufferlib/extensions/modules.cu"

// models.cpp provides Policy, MinGRU, DefaultEncoder, DefaultDecoder,
// reference impls, PRECISION_DTYPE, cuda_f32/f64/i32/i64, Tensor typedef
namespace pufferlib {
#include "pufferlib/extensions/models.cpp"
}
using namespace pufferlib;

// Muon optimizer (production uses Muon, not Adam)
#include "pufferlib/extensions/muon.h"

// Advantage computation (puff_advantage CUDA kernel)
// Redefine TORCH_LIBRARY_IMPL to skip dispatch registration —
// profiler calls compute_puff_advantage_cuda directly, no dispatch needed.
#undef TORCH_LIBRARY_IMPL
#define TORCH_LIBRARY_IMPL(ns, k, m) \
    static void _profiler_noop_advantage([[maybe_unused]] torch::Library& m)
#include "pufferlib/extensions/cuda/advantage.cu"

#endif

#ifndef USE_TORCH
#include "pufferlib/extensions/cuda/kernels.cu"
#endif

// ============================================================================
// Constants
// ============================================================================

const int WARMUP_ITERS = 100;
const int TIMING_ITERS = 1000;

const int BUF = 2;
const int BR = 4096;   // Rollout batch (no T dim)
const int BT = 512;    // Train batch (with T dim)
const int T = 64;
const int H = 128;
const int A = 4;
const int INPUT_SIZE = 96;

typedef void (*kernel_fn)(void*);

// ============================================================================
// Helpers
// ============================================================================

inline void print_timing(const char* name, float ms, int N) {
    printf("  %-28s %8.1f us  %8.2f M elem/s\n", name, ms * 1000, N / ms / 1e3);
}

inline void print_timing_pct(const char* name, float ms, int N, float total_ms) {
    float pct = (total_ms > 0) ? (ms / total_ms * 100.0f) : 0.0f;
    printf("  %-28s %8.1f us  %5.1f%%\n", name, ms * 1000, pct);
}

inline void warmup_gpu() {
    float* dummy;
    cudaMalloc(&dummy, 64 * 1024 * 1024);
    for (int i = 0; i < 100; i++) {
        cudaMemset(dummy, 0, 64 * 1024 * 1024);
    }
    cudaDeviceSynchronize();
    cudaFree(dummy);
}

inline float rand1() {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

// Safe host-to-device copy: converts float[] → precision_t[] before cudaMemcpy.
// Prevents buffer overflow when precision_t is bf16 (2 bytes) but source is float (4 bytes).
inline void float_to_device(precision_t* dst, const float* src, int count) {
    precision_t* tmp = (precision_t*)malloc(count * sizeof(precision_t));
    for (int i = 0; i < count; ++i) tmp[i] = (precision_t)src[i];
    cudaMemcpy(dst, tmp, count * sizeof(precision_t), cudaMemcpyHostToDevice);
    free(tmp);
}

// ============================================================================
// Profiling harness
// ============================================================================

float profile_kernel(kernel_fn fn, void* args, const char* name = nullptr) {
    for (int i = 0; i < WARMUP_ITERS; ++i) fn(args);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    for (int i = 0; i < TIMING_ITERS; ++i) {
        fn(args);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
#ifdef USE_TORCH
    c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    return ms / TIMING_ITERS;
}

#ifdef USE_TORCH
float profile_graph(kernel_fn fn, void* args, const char* name = nullptr) {
    cudaDeviceSynchronize();

    at::cuda::CUDAGraph cuda_graph;
    at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();

    // Warmup
    at::cuda::CUDAStream warmup_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(warmup_stream);
    for (int i = 0; i < WARMUP_ITERS; ++i) fn(args);
    warmup_stream.synchronize();

    // Capture graph
    at::cuda::CUDAStream cap_stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(cap_stream);
    cuda_graph.capture_begin();
    fn(args);
    cuda_graph.capture_end();
    cap_stream.synchronize();

    cudaDeviceSynchronize();
    at::cuda::setCurrentCUDAStream(current_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);

    for (int i = 0; i < TIMING_ITERS; ++i) {
        cuda_graph.replay();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    return ms / TIMING_ITERS;
}
#endif

// ============================================================================
// Section 2: MinGRU gate profiling
// ============================================================================

typedef struct {
    precision_t* state;       // (B, H)
    precision_t* combined;    // (B, 3*H) = [hidden, gate, proj]
    precision_t* out;         // (B, H)
    precision_t* next_state;  // (B, H)
    int B;
    int H;
} MingruGateArgs;

MingruGateArgs* create_mingrugateargs(int batch, int hidden) {
    MingruGateArgs* args = (MingruGateArgs*)calloc(1, sizeof(MingruGateArgs));
    args->B = batch;
    args->H = hidden;

    int N_state = batch * hidden;
    int N_combined = batch * 3 * hidden;

    cudaMalloc(&args->state, N_state * sizeof(precision_t));
    cudaMalloc(&args->combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->out, N_state * sizeof(precision_t));
    cudaMalloc(&args->next_state, N_state * sizeof(precision_t));

    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* combined_buf = (float*)malloc(N_combined * sizeof(float));

    for (int i = 0; i < N_state; ++i) {
        state_buf[i] = fabsf(rand1()) + 0.1f;
    }
    for (int b = 0; b < batch; ++b) {
        int base = b * 3 * hidden;
        for (int h = 0; h < hidden; ++h) {
            combined_buf[base + h] = rand1() * 5.0f;
            combined_buf[base + hidden + h] = rand1() * 5.0f;
            combined_buf[base + 2 * hidden + h] = rand1() * 2.0f;
        }
    }

    float_to_device(args->state, state_buf, N_state);
    float_to_device(args->combined, combined_buf, N_combined);

    free(state_buf);
    free(combined_buf);
    return args;
}

void free_mingrugateargs(MingruGateArgs* args) {
    cudaFree(args->state);
    cudaFree(args->combined);
    cudaFree(args->out);
    cudaFree(args->next_state);
    free(args);
}

void run_mingrugate_forward(MingruGateArgs* args) {
    mingru_gate_inference_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->next_state, args->combined, args->state,
        args->H, args->B);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor state;
    torch::Tensor combined;
    int B;
    int H;
} MingruGateArgsTorch;

MingruGateArgsTorch* create_mingrugateargs_torch(MingruGateArgs* raw) {
    MingruGateArgsTorch* args = new MingruGateArgsTorch();
    args->B = raw->B;
    args->H = raw->H;
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->state = torch::from_blob(raw->state, {raw->B, raw->H}, opts);
    args->combined = torch::from_blob(raw->combined, {raw->B, 3 * raw->H}, opts);
    return args;
}

void run_mingrugate_forward_torch(MingruGateArgsTorch* args) {
    torch::NoGradGuard no_grad;
    mingru_gate(args->state, args->combined);
}

void run_mingrugate_forward_cpp(MingruGateArgsTorch* args) {
    torch::NoGradGuard no_grad;
    mingru_gate_cpp(args->state, args->combined);
}

void test_mingrugate_correct(MingruGateArgsTorch* args) {
    // Kernel path: native precision (kernel internally computes in float32)
    auto kernel_outputs = mingru_gate(args->state, args->combined);
    auto kernel_out = kernel_outputs[0];
    auto kernel_next_state = kernel_outputs[1];

    // Cpp path: float32 for accurate reference
    auto cpp_outputs = mingru_gate_cpp(
        args->state.to(torch::kFloat32),
        args->combined.to(torch::kFloat32));
    auto cpp_out = cpp_outputs[0];
    auto cpp_next_state = cpp_outputs[1];

    float rtol = USE_BF16 ? 5e-2f : 1e-3f;
    float atol = USE_BF16 ? 1e-2f : 1e-4f;
    bool out_match = torch::allclose(kernel_out.to(torch::kFloat32), cpp_out, rtol, atol);
    float out_max_diff = (kernel_out.to(torch::kFloat32) - cpp_out).abs().max().item<float>();
    bool next_state_match = torch::allclose(kernel_next_state.to(torch::kFloat32), cpp_next_state, rtol, atol);
    float next_state_max_diff = (kernel_next_state.to(torch::kFloat32) - cpp_next_state).abs().max().item<float>();

    printf("  correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff,
           next_state_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", next_state_max_diff);
}

#endif

void profile_mingrugate(int batch, int hidden) {
    printf("mingru_gate (B=%d, H=%d, combined=%dx%d)\n", batch, hidden, batch, 3*hidden);

    MingruGateArgs* args = create_mingrugateargs(batch, hidden);
    float fwd_ms = profile_kernel((kernel_fn)run_mingrugate_forward, args);
    print_timing("forward", fwd_ms, batch);

#ifdef USE_TORCH
    MingruGateArgsTorch* args_torch = create_mingrugateargs_torch(args);
    test_mingrugate_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_mingrugate_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_mingrugate_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_mingrugate_forward_cpp, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");
    free_mingrugateargs(args);
}

// ============================================================================
// Section 3: LogCumsumExp profiling
// ============================================================================

typedef struct {
    precision_t* x;
    precision_t* out;
    double* s_buf;
    precision_t* grad_x;
    precision_t* grad_out;
    int B;
    int T;
    int H;
    int N;
} LogcumsumexpArgs;

LogcumsumexpArgs* create_logcumsumexpargs(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = (LogcumsumexpArgs*)calloc(1, sizeof(LogcumsumexpArgs));
    args->B = batch;
    args->T = seq;
    args->H = hidden;
    args->N = batch * seq * hidden;

    cudaMalloc(&args->x, args->N * sizeof(precision_t));
    cudaMalloc(&args->out, args->N * sizeof(precision_t));
    cudaMalloc(&args->s_buf, args->N * sizeof(double));
    cudaMalloc(&args->grad_x, args->N * sizeof(precision_t));
    cudaMalloc(&args->grad_out, args->N * sizeof(precision_t));

    float* buf = (float*)malloc(args->N * sizeof(float) * 2);
    float* x_buf = buf;
    float* grad_out_buf = buf + args->N;
    for (int i = 0; i < args->N; ++i) {
        x_buf[i] = rand1();
        grad_out_buf[i] = rand1();
    }

    float_to_device(args->x, x_buf, args->N);
    float_to_device(args->grad_out, grad_out_buf, args->N);

    free(buf);
    return args;
}

void free_logcumsumexpargs(LogcumsumexpArgs* args) {
    cudaFree(args->x);
    cudaFree(args->out);
    cudaFree(args->s_buf);
    cudaFree(args->grad_x);
    cudaFree(args->grad_out);
    free(args);
}

void run_logcumsumexp_forward(LogcumsumexpArgs* args) {
    logcumsumexp_forward_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->s_buf, args->x, args->T, args->H, args->B);
}

void run_logcumsumexp_backward(LogcumsumexpArgs* args) {
    logcumsumexp_backward_kernel<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->grad_x, args->grad_out, args->x, args->s_buf, args->T, args->H, args->B);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor x;
    torch::Tensor out;
    torch::Tensor grad_out;
    int N;
} LogcumsumexpArgsTorch;

LogcumsumexpArgsTorch* create_logcumsumexpargs_torch(LogcumsumexpArgs* raw) {
    LogcumsumexpArgsTorch* args = new LogcumsumexpArgsTorch();
    args->N = raw->N;
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    // Keep in native precision — kernel wrappers cast to precision_t*
    args->x = torch::from_blob(raw->x, {raw->B, raw->T, raw->H}, opts).clone().requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts).clone();
    return args;
}

void run_logcumsumexp_forward_torch(LogcumsumexpArgsTorch* args) {
    torch::NoGradGuard no_grad;
    logcumsumexp_cuda(args->x);
}

void run_logcumsumexp_backward_torch(LogcumsumexpArgsTorch* args) {
    args->x.mutable_grad() = torch::Tensor();
    args->out.backward(args->grad_out, /*retain_graph=*/true);
}

void run_logcumsumexp_forward_cpp(LogcumsumexpArgsTorch* args) {
    torch::NoGradGuard no_grad;
    logcumsumexp_cpp(args->x);
}

void test_logcumsumexp_correct(LogcumsumexpArgsTorch* args) {
    // Kernel path: native precision (bf16 or f32) — kernel casts to precision_t*
    auto x_k = args->x.detach().clone().requires_grad_(true);
    auto out_k = logcumsumexp_cuda(x_k);

    // Cpp path: float32 for higher-precision reference
    auto x_c = args->x.detach().to(torch::kFloat32).requires_grad_(true);
    auto out_c = logcumsumexp_cpp(x_c);

    float rtol = USE_BF16 ? 5e-2f : 1e-3f;
    float atol = USE_BF16 ? 1e-2f : 1e-4f;
    float out_max_diff = (out_k.to(torch::kFloat32) - out_c).abs().max().item<float>();
    bool out_match = torch::allclose(out_k.to(torch::kFloat32), out_c, rtol, atol);
    printf("  forward correctness: %s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff);

    // Backward: kernel in native precision
    auto grad_k = args->grad_out.to(x_k.dtype());
    out_k.backward(grad_k, /*retain_graph=*/false);

    // Backward: cpp in float32
    auto grad_c = args->grad_out.to(torch::kFloat32);
    out_c.backward(grad_c, /*retain_graph=*/false);

    float grad_max_diff = (x_k.grad().to(torch::kFloat32) - x_c.grad()).abs().max().item<float>();
    bool grad_match = torch::allclose(x_k.grad().to(torch::kFloat32), x_c.grad(), rtol, atol);
    printf("  backward correctness: %s(%.2e)\n",
           grad_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_max_diff);
}

#endif

void profile_logcumsumexp(int batch, int seq, int hidden) {
    LogcumsumexpArgs* args = create_logcumsumexpargs(batch, seq, hidden);
    printf("logcumsumexp (N=%d, %dx%dx%d)\n", args->N, batch, seq, hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward, args);
    print_timing("forward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward, args);
    print_timing("backward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    LogcumsumexpArgsTorch* args_torch = create_logcumsumexpargs_torch(args);
    test_logcumsumexp_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch*seq);

    args_torch->out = logcumsumexp_cuda(args_torch->x);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward_torch, args_torch);
    print_timing("backward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_logcumsumexp_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch*seq);

    args_torch->out = logcumsumexp_cpp(args_torch->x);

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_logcumsumexp_backward_torch, args_torch);
    print_timing("backward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_logcumsumexp_forward_cpp, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");
    free_logcumsumexpargs(args);
}

// ============================================================================
// Section 4: Fused scan (checkpointed) profiling
// ============================================================================

typedef struct {
    precision_t* combined;       // (B, T, 3*H)
    precision_t* state;          // (B, 1, H)
    precision_t* out;            // (B, T, H)
    precision_t* next_state;     // (B, 1, H)
    float* a_star;               // (B, T+1, H)
    float* s_vals;               // (B, T+1, H)
    float* log_values_buf;       // (B, T+1, H)
    precision_t* grad_combined;  // (B, T, 3*H)
    precision_t* grad_state;     // (B, 1, H)
    precision_t* grad_out;       // (B, T, H)
    precision_t* grad_next_state;// (B, 1, H)
    int B;
    int T;
    int H;
    int N;
} FusedScanArgs;

FusedScanArgs* create_fusedscanargs(int batch, int seq, int hidden) {
    FusedScanArgs* args = (FusedScanArgs*)calloc(1, sizeof(FusedScanArgs));
    args->B = batch;
    args->T = seq;
    args->H = hidden;
    args->N = batch * seq * hidden;

    int N_combined = batch * seq * 3 * hidden;
    int N_state = batch * hidden;
    int N_buf = batch * (seq + 1) * hidden;

    cudaMalloc(&args->combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->state, N_state * sizeof(precision_t));
    cudaMalloc(&args->out, args->N * sizeof(precision_t));
    cudaMalloc(&args->next_state, N_state * sizeof(precision_t));
    cudaMalloc(&args->a_star, N_buf * sizeof(float));
    cudaMalloc(&args->s_vals, N_buf * sizeof(float));
    cudaMalloc(&args->log_values_buf, N_buf * sizeof(float));
    cudaMalloc(&args->grad_combined, N_combined * sizeof(precision_t));
    cudaMalloc(&args->grad_state, N_state * sizeof(precision_t));
    cudaMalloc(&args->grad_out, args->N * sizeof(precision_t));
    cudaMalloc(&args->grad_next_state, N_state * sizeof(precision_t));

    float* combined_buf = (float*)malloc(N_combined * sizeof(float));
    float* state_buf = (float*)malloc(N_state * sizeof(float));
    float* grad_out_buf = (float*)malloc(args->N * sizeof(float));
    float* grad_next_state_buf = (float*)malloc(N_state * sizeof(float));

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seq; ++t) {
            for (int h = 0; h < hidden; ++h) {
                int base = b * seq * 3 * hidden + t * 3 * hidden;
                combined_buf[base + h] = rand1() * 5.0f;
                combined_buf[base + hidden + h] = rand1() * 5.0f;
                combined_buf[base + 2 * hidden + h] = rand1() * 2.0f;
            }
        }
    }
    for (int i = 0; i < N_state; ++i) state_buf[i] = fabsf(rand1()) + 0.1f;
    for (int i = 0; i < args->N; ++i) grad_out_buf[i] = rand1();
    for (int i = 0; i < N_state; ++i) grad_next_state_buf[i] = rand1();

    float_to_device(args->combined, combined_buf, N_combined);
    float_to_device(args->state, state_buf, N_state);
    float_to_device(args->grad_out, grad_out_buf, args->N);
    float_to_device(args->grad_next_state, grad_next_state_buf, N_state);

    free(combined_buf);
    free(state_buf);
    free(grad_out_buf);
    free(grad_next_state_buf);
    return args;
}

void free_fusedscanargs(FusedScanArgs* args) {
    cudaFree(args->combined);
    cudaFree(args->state);
    cudaFree(args->out);
    cudaFree(args->next_state);
    cudaFree(args->a_star);
    cudaFree(args->s_vals);
    cudaFree(args->log_values_buf);
    cudaFree(args->grad_combined);
    cudaFree(args->grad_state);
    cudaFree(args->grad_out);
    cudaFree(args->grad_next_state);
    free(args);
}

void run_fusedscan_forward(FusedScanArgs* args) {
    fused_scan_forward_kernel_checkpointed<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->out, args->next_state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->combined, args->state,
        args->T, args->H, args->B);
}

void run_fusedscan_backward(FusedScanArgs* args) {
    fused_scan_backward_kernel_checkpointed<<<grid_size(args->B * args->H), BLOCK_SIZE>>>(
        args->grad_combined, args->grad_state,
        args->grad_out, args->grad_next_state,
        args->combined, args->state,
        args->a_star, args->s_vals, args->log_values_buf,
        args->T, args->H, args->B);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor combined;
    torch::Tensor state;
    torch::Tensor out;
    torch::Tensor next_state;
    torch::Tensor grad_out;
    torch::Tensor grad_next_state;
    int B;
    int T;
    int H;
} FusedScanArgsTorch;

FusedScanArgsTorch* create_fusedscanargs_torch(FusedScanArgs* raw) {
    FusedScanArgsTorch* args = new FusedScanArgsTorch();
    args->B = raw->B;
    args->T = raw->T;
    args->H = raw->H;
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    // Keep in native precision — kernel wrappers cast to precision_t*
    args->combined = torch::from_blob(raw->combined, {raw->B, raw->T, 3 * raw->H}, opts).clone().requires_grad_(true);
    args->state = torch::from_blob(raw->state, {raw->B, 1, raw->H}, opts).clone().requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->T, raw->H}, opts).clone();
    args->grad_next_state = torch::from_blob(raw->grad_next_state, {raw->B, 1, raw->H}, opts).clone();
    return args;
}

void run_fusedscan_forward_torch(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan_checkpointed(args->combined, args->state);
}

void run_fusedscan_backward_torch(FusedScanArgsTorch* args) {
    args->combined.mutable_grad() = torch::Tensor();
    args->state.mutable_grad() = torch::Tensor();
    torch::autograd::backward(
        {args->out, args->next_state},
        {args->grad_out, args->grad_next_state},
        /*retain_graph=*/true);
}

void run_fusedscan_forward_cpp(FusedScanArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fused_scan_cpp(args->combined, args->state);
}

void test_fusedscan_correct(FusedScanArgsTorch* args) {
    // Kernel path: native precision (bf16 or f32) — kernel casts to precision_t*
    auto combined_k = args->combined.detach().clone().requires_grad_(true);
    auto state_k = args->state.detach().clone().requires_grad_(true);
    combined_k.retain_grad();
    state_k.retain_grad();
    auto k_outputs = fused_scan_checkpointed(combined_k, state_k);
    auto k_out = k_outputs[0];
    auto k_next_state = k_outputs[1];

    // Cpp path: float32 for higher-precision reference
    auto combined_c = args->combined.detach().to(torch::kFloat32);
    auto state_c = args->state.detach().to(torch::kFloat32);
    auto c_outputs = fused_scan_cpp(combined_c, state_c);
    auto c_out = c_outputs[0];
    auto c_next_state = c_outputs[1];

    // bf16 accumulates error over 64 sequential timesteps — needs wider tolerance
    float rtol = USE_BF16 ? 1e-1f : 1e-3f;
    float atol = USE_BF16 ? 5e-2f : 2e-4f;
    bool out_match = torch::allclose(k_out.to(torch::kFloat32), c_out, rtol, atol);
    float out_max_diff = (k_out.to(torch::kFloat32) - c_out).abs().max().item<float>();
    bool ns_match = torch::allclose(k_next_state.to(torch::kFloat32), c_next_state, rtol, atol);
    float ns_max_diff = (k_next_state.to(torch::kFloat32) - c_next_state).abs().max().item<float>();
    printf("  forward correctness: out=%s(%.2e) next_state=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff,
           ns_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", ns_max_diff);

    // Backward: kernel path (native precision)
    auto grad_out_k = args->grad_out.to(k_out.dtype());
    auto grad_ns_k = args->grad_next_state.to(k_next_state.dtype());
    torch::autograd::backward({k_out, k_next_state}, {grad_out_k, grad_ns_k});
    auto grad_combined_k = combined_k.grad().clone();
    auto grad_state_k = state_k.grad().clone();

    // Backward: cpp path in float32
    auto combined_c2 = args->combined.detach().to(torch::kFloat32).requires_grad_(true);
    auto state_c2 = args->state.detach().to(torch::kFloat32).requires_grad_(true);
    auto c_out2 = fused_scan_cpp(combined_c2, state_c2);
    torch::autograd::backward({c_out2[0], c_out2[1]},
        {args->grad_out.to(torch::kFloat32), args->grad_next_state.to(torch::kFloat32)});

    bool gc_match = torch::allclose(grad_combined_k.to(torch::kFloat32), combined_c2.grad(), rtol, atol);
    float gc_diff = (grad_combined_k.to(torch::kFloat32) - combined_c2.grad()).abs().max().item<float>();
    bool gs_match = torch::allclose(grad_state_k.to(torch::kFloat32), state_c2.grad(), rtol, atol);
    float gs_diff = (grad_state_k.to(torch::kFloat32) - state_c2.grad()).abs().max().item<float>();
    printf("  backward correctness: grad_combined=%s(%.2e) grad_state=%s(%.2e)\n",
           gc_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", gc_diff,
           gs_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", gs_diff);
}

#endif

void profile_fusedscan(int batch, int seq, int hidden) {
    FusedScanArgs* args = create_fusedscanargs(batch, seq, hidden);
    printf("fused_scan (N=%d, %dx%dx%d, combined=%dx%dx%d)\n",
           args->N, batch, seq, hidden, batch, seq, 3*hidden);

    float fwd_ms = profile_kernel((kernel_fn)run_fusedscan_forward, args);
    print_timing("forward", fwd_ms, batch*seq);

    float bwd_ms = profile_kernel((kernel_fn)run_fusedscan_backward, args);
    print_timing("backward", bwd_ms, batch*seq);

#ifdef USE_TORCH
    FusedScanArgsTorch* args_torch = create_fusedscanargs_torch(args);
    test_fusedscan_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch*seq);

    auto scan_out = fused_scan_checkpointed(args_torch->combined, args_torch->state);
    args_torch->out = scan_out[0];
    args_torch->next_state = scan_out[1];

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("backward (torch)", bwd_torch_ms, batch*seq);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch*seq);

    auto scan_out_cpp = fused_scan_cpp(args_torch->combined, args_torch->state);
    args_torch->out = scan_out_cpp[0];
    args_torch->next_state = scan_out_cpp[1];

    float bwd_cpp_ms = profile_kernel((kernel_fn)run_fusedscan_backward_torch, args_torch);
    print_timing("backward (cpp)", bwd_cpp_ms, batch*seq);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fusedscan_forward_cpp, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch*seq);

    delete args_torch;
#endif
    printf("\n");
    free_fusedscanargs(args);
}

// ============================================================================
// Section 5: FCMax profiling
// ============================================================================

typedef struct {
    precision_t* x;              // (B, N, D_in)
    float* W;              // (D_out, D_in) - always float32
    float* b;              // (D_out) - always float32
    precision_t* out;            // (B, D_out)
    int* argmax_indices;   // (B, D_out)
    float* grad_x;         // (B, N, D_in) - always float32 for atomicAdd
    float* grad_W;         // (D_out, D_in)
    float* grad_b;         // (D_out)
    precision_t* grad_out;       // (B, D_out)
    int B;
    int N;
    int D_in;
    int D_out;
} FCMaxArgs;

FCMaxArgs* create_fcmaxargs(int batch, int num_points, int d_in, int d_out) {
    FCMaxArgs* args = (FCMaxArgs*)calloc(1, sizeof(FCMaxArgs));
    args->B = batch;
    args->N = num_points;
    args->D_in = d_in;
    args->D_out = d_out;

    int N_x = batch * num_points * d_in;
    int N_W = d_out * d_in;
    int N_out = batch * d_out;

    cudaMalloc(&args->x, N_x * sizeof(precision_t));
    cudaMalloc(&args->W, N_W * sizeof(float));
    cudaMalloc(&args->b, d_out * sizeof(float));
    cudaMalloc(&args->out, N_out * sizeof(precision_t));
    cudaMalloc(&args->argmax_indices, N_out * sizeof(int));
    cudaMalloc(&args->grad_x, N_x * sizeof(float));
    cudaMalloc(&args->grad_W, N_W * sizeof(float));
    cudaMalloc(&args->grad_b, d_out * sizeof(float));
    cudaMalloc(&args->grad_out, N_out * sizeof(precision_t));

    float* x_buf = (float*)malloc(N_x * sizeof(float));
    float* W_buf = (float*)malloc(N_W * sizeof(float));
    float* b_buf = (float*)malloc(d_out * sizeof(float));
    float* grad_out_buf = (float*)malloc(N_out * sizeof(float));

    for (int i = 0; i < N_x; ++i) x_buf[i] = rand1();
    for (int i = 0; i < N_W; ++i) W_buf[i] = rand1() * 0.1f;
    for (int i = 0; i < d_out; ++i) b_buf[i] = 0.0f;
    for (int i = 0; i < N_out; ++i) grad_out_buf[i] = rand1();

    float_to_device(args->x, x_buf, N_x);
    cudaMemcpy(args->W, W_buf, N_W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->b, b_buf, d_out * sizeof(float), cudaMemcpyHostToDevice);
    float_to_device(args->grad_out, grad_out_buf, N_out);

    free(x_buf);
    free(W_buf);
    free(b_buf);
    free(grad_out_buf);
    return args;
}

void free_fcmaxargs(FCMaxArgs* args) {
    cudaFree(args->x);
    cudaFree(args->W);
    cudaFree(args->b);
    cudaFree(args->out);
    cudaFree(args->argmax_indices);
    cudaFree(args->grad_x);
    cudaFree(args->grad_W);
    cudaFree(args->grad_b);
    cudaFree(args->grad_out);
    free(args);
}

void run_fcmax_forward(FCMaxArgs* args) {
    fc_max_forward_kernel<<<grid_size(args->B * args->D_out), BLOCK_SIZE>>>(
        args->out, args->argmax_indices,
        args->x, args->W, args->b,
        args->B, args->N, args->D_in, args->D_out);
}

void run_fcmax_backward(FCMaxArgs* args) {
    cudaMemset(args->grad_x, 0, args->B * args->N * args->D_in * sizeof(float));
    cudaMemset(args->grad_W, 0, args->D_out * args->D_in * sizeof(float));
    cudaMemset(args->grad_b, 0, args->D_out * sizeof(float));

    fc_max_backward_kernel<<<grid_size(args->B * args->D_out), BLOCK_SIZE>>>(
        args->grad_x, args->grad_W, args->grad_b,
        args->grad_out, args->x, args->W,
        args->argmax_indices,
        args->B, args->N, args->D_in, args->D_out);
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor x;       // (B, N, D_in)
    torch::Tensor W;       // (D_out, D_in)
    torch::Tensor b;       // (D_out)
    torch::Tensor out;     // (B, D_out)
    torch::Tensor grad_out;// (B, D_out)
    int B, N, D_in, D_out;
} FCMaxArgsTorch;

FCMaxArgsTorch* create_fcmaxargs_torch(FCMaxArgs* raw) {
    FCMaxArgsTorch* args = new FCMaxArgsTorch();
    args->B = raw->B;
    args->N = raw->N;
    args->D_in = raw->D_in;
    args->D_out = raw->D_out;

    auto prec_opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    // Keep x and grad_out in native precision — kernel casts to precision_t*
    // W and b are always float32 (kernel reads them as float directly)
    args->x = torch::from_blob(raw->x, {raw->B, raw->N, raw->D_in}, prec_opts).clone().requires_grad_(true);
    args->W = torch::from_blob(raw->W, {raw->D_out, raw->D_in}, cuda_f32).clone().requires_grad_(true);
    args->b = torch::from_blob(raw->b, {raw->D_out}, cuda_f32).clone().requires_grad_(true);
    args->grad_out = torch::from_blob(raw->grad_out, {raw->B, raw->D_out}, prec_opts).clone();

    return args;
}

void run_fcmax_forward_torch(FCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    fc_max(args->x, args->W, args->b);
}

void run_fcmax_backward_torch(FCMaxArgsTorch* args) {
    if (args->x.grad().defined()) args->x.grad().zero_();
    if (args->W.grad().defined()) args->W.grad().zero_();
    if (args->b.grad().defined()) args->b.grad().zero_();
    args->out.backward(args->grad_out.to(args->out.dtype()), /*retain_graph=*/true);
}

void run_fcmax_forward_cpp(FCMaxArgsTorch* args) {
    torch::NoGradGuard no_grad;
    // fc_max_cpp uses addmm which requires matching dtypes — upcast x to float32
    fc_max_cpp(args->x.to(torch::kFloat32), args->W, args->b);
}

void test_fcmax_correct(FCMaxArgsTorch* args) {
    // Kernel path: x in native precision (kernel casts to precision_t*)
    auto x_fused = args->x.detach().clone().requires_grad_(true);
    auto W_fused = args->W.detach().clone().requires_grad_(true);
    auto b_fused = args->b.detach().clone().requires_grad_(true);
    auto fused_out = fc_max(x_fused, W_fused, b_fused);

    // Cpp path: x in float32 (addmm requires matching dtypes)
    auto x_ref = args->x.detach().to(torch::kFloat32).requires_grad_(true);
    auto W_ref = args->W.detach().clone().requires_grad_(true);
    auto b_ref = args->b.detach().clone().requires_grad_(true);
    auto ref_out = fc_max_cpp(x_ref, W_ref, b_ref);

    float rtol = USE_BF16 ? 5e-2f : 1e-3f;
    float atol = USE_BF16 ? 1e-2f : 1e-4f;
    bool out_match = torch::allclose(fused_out.to(torch::kFloat32), ref_out, rtol, atol);
    float out_max_diff = (fused_out.to(torch::kFloat32) - ref_out).abs().max().item<float>();

    printf("  forward correctness: out=%s(%.2e)\n",
           out_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", out_max_diff);

    auto grad_fused = args->grad_out.to(fused_out.dtype());
    auto grad_ref = args->grad_out.to(torch::kFloat32);
    fused_out.backward(grad_fused);
    ref_out.backward(grad_ref);

    bool grad_x_match = torch::allclose(x_fused.grad().to(torch::kFloat32), x_ref.grad(), rtol, atol);
    float grad_x_max_diff = (x_fused.grad().to(torch::kFloat32) - x_ref.grad()).abs().max().item<float>();
    bool grad_W_match = torch::allclose(W_fused.grad(), W_ref.grad(), rtol, atol);
    float grad_W_max_diff = (W_fused.grad() - W_ref.grad()).abs().max().item<float>();

    printf("  backward correctness: grad_x=%s(%.2e) grad_W=%s(%.2e)\n",
           grad_x_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_x_max_diff,
           grad_W_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_W_max_diff);
}

#endif

void profile_fcmax(int batch, int num_points, int d_in, int d_out) {
    FCMaxArgs* args = create_fcmaxargs(batch, num_points, d_in, d_out);

    printf("fc_max (B=%d, N=%d, D_in=%d, D_out=%d)\n", batch, num_points, d_in, d_out);

    float fwd_ms = profile_kernel((kernel_fn)run_fcmax_forward, args);
    print_timing("forward", fwd_ms, batch);

    float bwd_ms = profile_kernel((kernel_fn)run_fcmax_backward, args);
    print_timing("backward", bwd_ms, batch);

#ifdef USE_TORCH
    FCMaxArgsTorch* args_torch = create_fcmaxargs_torch(args);

    test_fcmax_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_fcmax_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch);

    args_torch->out = fc_max(args_torch->x, args_torch->W, args_torch->b);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_fcmax_backward_torch, args_torch);
    print_timing("backward (torch)", bwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_fcmax_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_fcmax_forward_cpp, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_fcmaxargs(args);
}

// ============================================================================
// Section 6: PPO loss profiling
// ============================================================================

typedef struct {
    precision_t* logits;
    precision_t* values_pred;
    double* actions;          // float64 for both continuous and discrete
    precision_t* old_logprobs;
    float* advantages;        // always fp32 for precision
    precision_t* prio;
    precision_t* values;
    precision_t* returns;
    float* adv_mean;
    float* adv_var;           // variance, kernel does sqrt
    float* loss;
    float* losses_acc;         // (LOSS_N,) accumulator for loss components
    double* saved_for_backward;
    precision_t* ratio_out;
    precision_t* newvalue_out;
    float* grad_logits;
    float* grad_values_pred;
    float* grad_loss;
    int* act_sizes;           // (num_atns,) action head sizes
    int num_atns;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int N;
    int T;
    int A;
    int logits_stride_n;
    int logits_stride_t;
    int logits_stride_a;
    int values_stride_n;
    int values_stride_t;
} PPOLossArgs;

PPOLossArgs* create_ppolossargs(int batch, int seq, int actions) {
    PPOLossArgs* args = (PPOLossArgs*)calloc(1, sizeof(PPOLossArgs));
    args->N = batch;
    args->T = seq;
    args->A = actions;
    args->num_atns = 1;  // single action head for profiling

    int NT = batch*seq;
    int NTA = batch*seq * actions;

    cudaMalloc(&args->logits, NTA * sizeof(precision_t));
    cudaMalloc(&args->values_pred, NT * sizeof(precision_t));
    cudaMalloc(&args->actions, NT * sizeof(double));
    cudaMalloc(&args->old_logprobs, NT * sizeof(precision_t));
    cudaMalloc(&args->advantages, NT * sizeof(float));
    cudaMalloc(&args->prio, batch * sizeof(precision_t));
    cudaMalloc(&args->values, NT * sizeof(precision_t));
    cudaMalloc(&args->returns, NT * sizeof(precision_t));
    cudaMalloc(&args->adv_mean, sizeof(float));
    cudaMalloc(&args->adv_var, sizeof(float));
    cudaMalloc(&args->loss, sizeof(float));
    cudaMalloc(&args->losses_acc, LOSS_N * sizeof(float));
    cudaMemset(args->losses_acc, 0, LOSS_N * sizeof(float));
    cudaMalloc(&args->saved_for_backward, NT * 5 * sizeof(double));
    cudaMalloc(&args->ratio_out, NT * sizeof(precision_t));
    cudaMalloc(&args->newvalue_out, NT * sizeof(precision_t));
    cudaMalloc(&args->grad_logits, NTA * sizeof(float));
    cudaMalloc(&args->grad_values_pred, NT * sizeof(float));
    cudaMalloc(&args->grad_loss, sizeof(float));
    cudaMalloc(&args->act_sizes, sizeof(int));

    cudaMemcpy(args->act_sizes, &actions, sizeof(int), cudaMemcpyHostToDevice);

    float* buf = (float*)malloc((NTA + NT * 5 + batch) * sizeof(float));
    float* logits_buf = buf;
    float* values_pred_buf = buf + NTA;
    float* old_logprobs_buf = buf + NTA + NT;
    float* advantages_buf = buf + NTA + NT * 2;
    float* values_buf = buf + NTA + NT * 3;
    float* returns_buf = buf + NTA + NT * 4;
    float* prio_buf = buf + NTA + NT * 5;

    double* actions_buf = (double*)malloc(NT * sizeof(double));

    float adv_sum = 0.0f, adv_sq_sum = 0.0f;
    for (int i = 0; i < NT; ++i) {
        advantages_buf[i] = rand1();
        adv_sum += advantages_buf[i];
        adv_sq_sum += advantages_buf[i] * advantages_buf[i];
    }
    float adv_mean = adv_sum / NT;
    float adv_var = adv_sq_sum / NT - adv_mean * adv_mean;

    for (int i = 0; i < NTA; ++i) {
        logits_buf[i] = rand1() * 2.0f;
    }
    for (int i = 0; i < NT; ++i) {
        values_pred_buf[i] = rand1();
        actions_buf[i] = (double)(rand() % actions);
        old_logprobs_buf[i] = rand1() * 2.0f;
        values_buf[i] = rand1();
        returns_buf[i] = rand1();
    }
    for (int i = 0; i < batch; ++i) {
        prio_buf[i] = (float)rand() / RAND_MAX;
    }

    float_to_device(args->logits, logits_buf, NTA);
    float_to_device(args->values_pred, values_pred_buf, NT);
    cudaMemcpy(args->actions, actions_buf, NT * sizeof(double), cudaMemcpyHostToDevice);
    float_to_device(args->old_logprobs, old_logprobs_buf, NT);
    cudaMemcpy(args->advantages, advantages_buf, NT * sizeof(float), cudaMemcpyHostToDevice);
    float_to_device(args->prio, prio_buf, batch);
    float_to_device(args->values, values_buf, NT);
    float_to_device(args->returns, returns_buf, NT);
    cudaMemcpy(args->adv_mean, &adv_mean, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(args->adv_var, &adv_var, sizeof(float), cudaMemcpyHostToDevice);

    float grad_loss_val = 1.0f;
    cudaMemcpy(args->grad_loss, &grad_loss_val, sizeof(float), cudaMemcpyHostToDevice);

    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;

    args->logits_stride_n = seq * actions;
    args->logits_stride_t = actions;
    args->logits_stride_a = 1;
    args->values_stride_n = seq;
    args->values_stride_t = 1;

    free(buf);
    free(actions_buf);
    return args;
}

void free_ppolossargs(PPOLossArgs* args) {
    cudaFree(args->logits);
    cudaFree(args->values_pred);
    cudaFree(args->actions);
    cudaFree(args->old_logprobs);
    cudaFree(args->advantages);
    cudaFree(args->prio);
    cudaFree(args->values);
    cudaFree(args->returns);
    cudaFree(args->adv_mean);
    cudaFree(args->adv_var);
    cudaFree(args->loss);
    cudaFree(args->losses_acc);
    cudaFree(args->saved_for_backward);
    cudaFree(args->ratio_out);
    cudaFree(args->newvalue_out);
    cudaFree(args->grad_logits);
    cudaFree(args->grad_values_pred);
    cudaFree(args->grad_loss);
    cudaFree(args->act_sizes);
    free(args);
}

void run_ppoloss_forward(PPOLossArgs* args) {
    int total = args->N * args->T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    cudaMemset(args->loss, 0, sizeof(float));
    ppo_loss_forward_kernel_optimized<<<ppo_grid, PPO_THREADS>>>(
        args->loss, args->losses_acc,
        args->saved_for_backward,
        args->ratio_out, args->newvalue_out,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->act_sizes, args->num_atns,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        false);  // is_continuous
}

void run_ppoloss_backward(PPOLossArgs* args) {
    int total = args->N * args->T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;
    ppo_loss_backward_kernel_optimized<<<ppo_grid, PPO_THREADS>>>(
        args->grad_logits,
        nullptr,  // grad_logstd (nullptr for discrete)
        args->grad_values_pred, args->grad_loss,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->act_sizes, args->num_atns,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        args->T, args->A, args->N,
        args->logits_stride_n, args->logits_stride_t, args->logits_stride_a,
        args->values_stride_n, args->values_stride_t,
        false);  // is_continuous
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor logits;
    torch::Tensor values_pred;
    torch::Tensor actions;
    torch::Tensor old_logprobs;
    torch::Tensor advantages;
    torch::Tensor prio;
    torch::Tensor values;
    torch::Tensor returns;
    torch::Tensor adv_mean;
    torch::Tensor adv_var;  // variance, kernel computes sqrt
    torch::Tensor ratio_out;
    torch::Tensor newvalue_out;
    torch::Tensor act_sizes;
    torch::Tensor losses_acc;
    torch::Tensor loss;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    int N;
    int T;
    int A;
} PPOLossArgsTorch;

PPOLossArgsTorch* create_ppolossargs_torch(PPOLossArgs* raw) {
    PPOLossArgsTorch* args = new PPOLossArgsTorch();
    args->N = raw->N;
    args->T = raw->T;
    args->A = raw->A;
    args->clip_coef = raw->clip_coef;
    args->vf_clip_coef = raw->vf_clip_coef;
    args->vf_coef = raw->vf_coef;
    args->ent_coef = raw->ent_coef;

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);

    args->logits = torch::from_blob(raw->logits, {raw->N, raw->T, raw->A}, opts).clone().requires_grad_(true);
    args->values_pred = torch::from_blob(raw->values_pred, {raw->N, raw->T, 1}, opts).clone().requires_grad_(true);
    args->actions = torch::from_blob(raw->actions, {raw->N, raw->T, 1}, cuda_f64).clone();
    args->old_logprobs = torch::from_blob(raw->old_logprobs, {raw->N, raw->T}, opts).clone();
    args->advantages = torch::from_blob(raw->advantages, {raw->N, raw->T}, cuda_f32).clone();
    args->prio = torch::from_blob(raw->prio, {raw->N, 1}, opts).clone();
    args->values = torch::from_blob(raw->values, {raw->N, raw->T}, opts).clone();
    args->returns = torch::from_blob(raw->returns, {raw->N, raw->T}, opts).clone();
    args->adv_mean = torch::from_blob(raw->adv_mean, {1}, cuda_f32).clone();
    args->adv_var = torch::from_blob(raw->adv_var, {1}, cuda_f32).clone();
    args->ratio_out = torch::zeros({raw->N, raw->T}, opts);
    args->newvalue_out = torch::zeros({raw->N, raw->T}, opts);
    args->act_sizes = torch::tensor({raw->A}, cuda_i32);
    args->losses_acc = torch::zeros({NUM_LOSSES}, cuda_f32);

    return args;
}

// Shared helpers

// Run fused PPO loss forward and return the loss tensor
torch::Tensor run_fused_ppo_forward(PPOLossArgsTorch* args) {
    auto logstd = torch::empty({0}, args->logits.options());
    return fused_ppo_loss_optimized(
        args->logits, logstd, args->values_pred, args->actions,
        args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->adv_mean, args->adv_var,
        args->ratio_out, args->newvalue_out, args->act_sizes,
        args->losses_acc,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef)[0];
}

// Compute train loss (kernel or cpp) with fresh clones for independent gradient tracking
struct TrainLossResult {
    Tensor loss;
    Tensor logits;
    Tensor values_pred;
};

TrainLossResult compute_test_loss(PPOLossArgsTorch* args, bool use_kernels) {
    int N = args->N, T = args->T, A = args->A;
    auto logits = args->logits.detach().clone().requires_grad_(true);
    auto values_pred = args->values_pred.detach().clone().requires_grad_(true);
    auto ratio_out = torch::zeros({N, T}, logits.options());
    auto newvalue_out = torch::zeros({N, T}, logits.options());
    Logits logits_struct = {.mean = logits};
    auto loss = compute_train_loss(
        logits_struct, values_pred,
        args->actions, args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, ratio_out, newvalue_out,
        args->act_sizes, torch::tensor({(int64_t)A}, torch::dtype(torch::kInt64)),
        N * T, T,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        /*is_continuous=*/false, use_kernels, args->losses_acc);
    return {loss, logits, values_pred};
}

// Run functions

void run_ppoloss_forward_torch(PPOLossArgsTorch* args) {
    torch::NoGradGuard no_grad;
    run_fused_ppo_forward(args);
}

void run_ppoloss_backward_torch(PPOLossArgsTorch* args) {
    args->logits.mutable_grad() = torch::Tensor();
    args->values_pred.mutable_grad() = torch::Tensor();
    args->loss.backward({}, /*retain_graph=*/true);
}

void test_ppoloss_correct(PPOLossArgsTorch* args) {
    auto [loss_k, logits_k, values_pred_k] = compute_test_loss(args, /*use_kernels=*/true);
    auto [loss_c, logits_c, values_pred_c] = compute_test_loss(args, /*use_kernels=*/false);

    float rtol = 1e-2f, atol = 1e-3f;
    float loss_diff = (loss_k - loss_c).abs().item<float>();
    bool loss_match = loss_diff < atol;
    printf("  forward correctness: loss=%s(%.2e)\n",
           loss_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", loss_diff);

    loss_k.backward();
    loss_c.backward();

    auto grad_logits_k = logits_k.grad();
    auto grad_logits_c = logits_c.grad();
    float grad_logits_diff = (grad_logits_k - grad_logits_c).abs().max().item<float>();
    bool grad_logits_match = torch::allclose(grad_logits_k, grad_logits_c, rtol, atol);
    printf("  backward correctness: grad_logits=%s(%.2e)\n",
           grad_logits_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_logits_diff);

    auto grad_values_k = values_pred_k.grad();
    auto grad_values_c = values_pred_c.grad();
    float grad_values_diff = (grad_values_k - grad_values_c).abs().max().item<float>();
    bool grad_values_match = torch::allclose(grad_values_k, grad_values_c, rtol, atol);
    printf("  backward correctness: grad_values=%s(%.2e)\n",
           grad_values_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", grad_values_diff);
}

#endif

void profile_ppoloss(int batch, int seq, int actions) {
    PPOLossArgs* args = create_ppolossargs(batch, seq, actions);

    int NT = batch*seq;
    printf("ppo_loss (NT=%d, %dx%d, A=%d)\n", NT, batch, seq, actions);

    float fwd_ms = profile_kernel((kernel_fn)run_ppoloss_forward, args);
    print_timing("forward", fwd_ms, NT);

    float bwd_ms = profile_kernel((kernel_fn)run_ppoloss_backward, args);
    print_timing("backward", bwd_ms, NT);

#ifdef USE_TORCH
    PPOLossArgsTorch* args_torch = create_ppolossargs_torch(args);

    test_ppoloss_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, NT);

    args_torch->loss = run_fused_ppo_forward(args_torch);

    float bwd_torch_ms = profile_kernel((kernel_fn)run_ppoloss_backward_torch, args_torch);
    print_timing("backward (torch)", bwd_torch_ms, NT);

    float fwd_graph_ms = profile_graph((kernel_fn)run_ppoloss_forward_torch, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, NT);

    delete args_torch;
#endif
    printf("\n");

    free_ppolossargs(args);
}

// ============================================================================
// Section 7: Sample logits profiling
// ============================================================================

typedef struct {
    precision_t* logits;        // (B, A)
    precision_t* value;         // (B, 1)
    double* actions;      // (B, 1) - float64 for discrete/continuous compatibility
    precision_t* logprobs;      // (B,)
    precision_t* value_out;     // (B,)
    int64_t* offset;      // RNG offset (on device for CUDA graph support)
    int* act_sizes;       // (1,) - single action head
    uint64_t seed;
    int B;
    int A;
} SampleLogitsArgs;

SampleLogitsArgs* create_samplelogitsargs(int batch, int num_actions) {
    SampleLogitsArgs* args = (SampleLogitsArgs*)calloc(1, sizeof(SampleLogitsArgs));
    args->B = batch;
    args->A = num_actions;
    args->seed = 42;

    int N_logits = batch * num_actions;
    int N_batch = batch;

    cudaMalloc(&args->logits, N_logits * sizeof(precision_t));
    cudaMalloc(&args->value, N_batch * sizeof(precision_t));
    cudaMalloc(&args->actions, N_batch * sizeof(double));
    cudaMalloc(&args->logprobs, N_batch * sizeof(precision_t));
    cudaMalloc(&args->value_out, N_batch * sizeof(precision_t));
    cudaMalloc(&args->offset, sizeof(int64_t));
    cudaMalloc(&args->act_sizes, sizeof(int));
    cudaMemset(args->offset, 0, sizeof(int64_t));
    cudaMemcpy(args->act_sizes, &num_actions, sizeof(int), cudaMemcpyHostToDevice);

    float* logits_buf = (float*)malloc(N_logits * sizeof(float));
    float* value_buf = (float*)malloc(N_batch * sizeof(float));

    for (int i = 0; i < N_logits; ++i) {
        logits_buf[i] = rand1() * 5.0f;
    }
    for (int i = 0; i < N_batch; ++i) {
        value_buf[i] = rand1();
    }

    float_to_device(args->logits, logits_buf, N_logits);
    float_to_device(args->value, value_buf, N_batch);

    free(logits_buf);
    free(value_buf);
    return args;
}

void free_samplelogitsargs(SampleLogitsArgs* args) {
    cudaFree(args->logits);
    cudaFree(args->value);
    cudaFree(args->actions);
    cudaFree(args->logprobs);
    cudaFree(args->value_out);
    cudaFree(args->offset);
    cudaFree(args->act_sizes);
    free(args);
}

void run_samplelogits_forward(SampleLogitsArgs* args) {
    sample_logits_kernel<<<grid_size(args->B), BLOCK_SIZE>>>(
        args->actions, args->logprobs, args->value_out,
        args->logits,
        nullptr,  // logstd (nullptr for discrete)
        args->value,
        args->act_sizes, args->seed, args->offset,
        1,  // num_atns
        args->B,
        args->A,  // logits_stride
        0,        // logstd_stride (unused for discrete)
        1,        // value_stride
        false);   // is_continuous
}

#ifdef USE_TORCH

typedef struct {
    torch::Tensor logits;       // (B, A)
    torch::Tensor value;        // (B, 1) - input
    torch::Tensor actions;      // (B, 1) float64 - output
    torch::Tensor logprobs;     // (B,) - output
    torch::Tensor value_out;    // (B,) - output
    torch::Tensor offset;       // (1,) int64 - RNG offset tensor
    torch::Tensor act_sizes;    // (1,) int32 - action sizes
    torch::Tensor act_sizes_cpu;// (1,) int64 - CPU version for cpp path
    uint64_t seed;
    int B;
    int A;
} SampleLogitsArgsTorch;

SampleLogitsArgsTorch* create_samplelogitsargs_torch(SampleLogitsArgs* raw) {
    SampleLogitsArgsTorch* args = new SampleLogitsArgsTorch();
    args->B = raw->B;
    args->A = raw->A;
    args->seed = 42;

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->logits = torch::from_blob(raw->logits, {raw->B, raw->A}, opts);
    args->value = torch::from_blob(raw->value, {raw->B, 1}, opts);
    args->actions = torch::empty({raw->B, 1}, opts.dtype(torch::kFloat64));
    args->logprobs = torch::empty({raw->B}, opts);
    args->value_out = torch::empty({raw->B}, opts);
    args->offset = torch::zeros({1}, opts.dtype(torch::kInt64));
    args->act_sizes = torch::tensor({raw->A}, cuda_i32);
    args->act_sizes_cpu = torch::tensor({(int64_t)raw->A}, torch::dtype(torch::kInt64));

    return args;
}

void run_samplelogits_forward_torch(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    auto logstd = torch::Tensor();  // empty/undefined for discrete
    sample_logits(args->logits, logstd, args->value, args->actions, args->logprobs,
        args->value_out, args->act_sizes, args->seed, args->offset);
}

void run_samplelogits_forward_cpp(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;
    sample_discrete_cpp(args->logits, args->act_sizes_cpu, 1);
}

void test_samplelogits_correct(SampleLogitsArgsTorch* args) {
    torch::NoGradGuard no_grad;

    auto logstd = torch::Tensor();
    sample_logits(args->logits, logstd, args->value, args->actions, args->logprobs,
        args->value_out, args->act_sizes, args->seed, args->offset);

    auto log_probs = torch::log_softmax(args->logits.to(torch::kFloat32), 1);
    auto actions_i64 = args->actions.to(torch::kInt64);
    auto expected_logprobs = log_probs.gather(1, actions_i64).squeeze(1);
    auto actual_logprobs = args->logprobs.to(torch::kFloat32);

    float rtol = 1e-2f, atol = 1e-3f;
    float logprob_max_diff = (actual_logprobs - expected_logprobs).abs().max().item<float>();
    bool logprob_match = torch::allclose(actual_logprobs, expected_logprobs, rtol, atol);
    printf("  logprob consistency: %s(%.2e)\n",
           logprob_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", logprob_max_diff);

    auto expected_values = args->value.squeeze(1).to(torch::kFloat32);
    auto actual_values = args->value_out.to(torch::kFloat32);
    float value_max_diff = (actual_values - expected_values).abs().max().item<float>();
    bool value_match = torch::allclose(actual_values, expected_values, rtol, atol);
    printf("  value passthrough: %s(%.2e)\n",
           value_match ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m", value_max_diff);

    auto actions_flat = args->actions.to(torch::kInt64).flatten();
    bool valid_actions = (actions_flat.ge(0) & actions_flat.lt(args->A)).all().item<bool>();
    printf("  action validity: %s\n",
           valid_actions ? "\033[32mok\033[0m" : "\033[31mFAIL\033[0m");
}

#endif

void profile_samplelogits(int batch, int num_actions) {
    SampleLogitsArgs* args = create_samplelogitsargs(batch, num_actions);

    printf("sample_logits (B=%d, A=%d)\n", batch, num_actions);

    float fwd_ms = profile_kernel((kernel_fn)run_samplelogits_forward, args);
    print_timing("forward", fwd_ms, batch);

#ifdef USE_TORCH
    SampleLogitsArgsTorch* args_torch = create_samplelogitsargs_torch(args);

    test_samplelogits_correct(args_torch);

    float fwd_torch_ms = profile_kernel((kernel_fn)run_samplelogits_forward_torch, args_torch);
    print_timing("forward (torch)", fwd_torch_ms, batch);

    float fwd_cpp_ms = profile_kernel((kernel_fn)run_samplelogits_forward_cpp, args_torch);
    print_timing("forward (cpp)", fwd_cpp_ms, batch);

    float fwd_graph_ms = profile_graph((kernel_fn)run_samplelogits_forward_torch, args_torch);
    print_timing("forward (graph)", fwd_graph_ms, batch);

    delete args_torch;
#endif
    printf("\n");

    free_samplelogitsargs(args);
}

// ============================================================================
// Section 8: Forward call profiling
// ============================================================================

#ifdef USE_TORCH

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor obs;
    Tensor state;
    Tensor actions;
    Tensor logprobs;
    Tensor values;
    Tensor rng_offset;
    Tensor act_sizes;
    Tensor act_sizes_cpu;
    uint64_t seed;
    bool use_kernels;
    int batch;
} ForwardCallArgs;

ForwardCallArgs* create_forwardcallargs(int batch, int input_size, int hidden_size,
                                        int act_n, int num_layers, bool use_kernels) {
    int num_action_heads = 1;

    ForwardCallArgs* args = new ForwardCallArgs();
    args->use_kernels = use_kernels;
    args->seed = 42;
    args->batch = batch;

    auto enc = std::make_shared<DefaultEncoder>(input_size, hidden_size);
    auto dec = std::make_shared<DefaultDecoder>(hidden_size, act_n);
    auto rnn = std::make_shared<MinGRU>(hidden_size, num_layers, use_kernels);
    args->policy = std::make_shared<Policy>(enc, dec, rnn, input_size, act_n, hidden_size);
    args->policy->to(torch::kCUDA);

    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->obs = torch::randn({batch, input_size}, opts);
    args->state = args->policy->initial_state(batch, torch::kCUDA);
    args->actions = torch::zeros({batch, num_action_heads}, cuda_f64);
    args->logprobs = torch::zeros({batch}, opts);
    args->values = torch::zeros({batch}, opts);
    args->rng_offset = torch::zeros({1}, cuda_i64);
    args->act_sizes = torch::tensor({act_n}, cuda_i32);
    args->act_sizes_cpu = torch::tensor({(int64_t)act_n}, torch::dtype(torch::kInt64));

    return args;
}

void free_forwardcallargs(ForwardCallArgs* args) {
    delete args;
}

void run_forward_call(ForwardCallArgs* args) {
    torch::NoGradGuard no_grad;

    auto [logits_out, value, state_out] = args->policy->forward(args->obs, args->state);

    sample_actions(logits_out, value, args->actions, args->logprobs, args->values,
        args->act_sizes, args->act_sizes_cpu,
        /*is_continuous=*/false, args->use_kernels, args->seed, args->rng_offset);

    args->state.copy_(state_out, false);
}

#endif

void profile_forwardcall(int batch, int input_size, int hidden_size, int num_atns, int num_layers) {
#ifdef USE_TORCH
    printf("forward_call (B=%d, in=%d, H=%d, A=%d, layers=%d)\n",
           batch, input_size, hidden_size, num_atns, num_layers);

    ForwardCallArgs* args_kernel = create_forwardcallargs(batch, input_size, hidden_size, num_atns, num_layers, true);
    float fwd_kernel_ms = profile_kernel((kernel_fn)run_forward_call, args_kernel, "forward_call_kernel");
    print_timing("kernel path", fwd_kernel_ms, batch);

    float fwd_kernel_graph_ms = profile_graph((kernel_fn)run_forward_call, args_kernel, "forward_call_kernel_graph");
    print_timing("kernel (graph)", fwd_kernel_graph_ms, batch);

    free_forwardcallargs(args_kernel);

    ForwardCallArgs* args_torch = create_forwardcallargs(batch, input_size, hidden_size, num_atns, num_layers, false);
    float fwd_torch_ms = profile_kernel((kernel_fn)run_forward_call, args_torch, "forward_call_torch");
    print_timing("torch path", fwd_torch_ms, batch);

    float fwd_torch_graph_ms = profile_graph((kernel_fn)run_forward_call, args_torch, "forward_call_torch_graph");
    print_timing("torch (graph)", fwd_torch_graph_ms, batch);

    free_forwardcallargs(args_torch);

    printf("\n");
#else
    printf("forward_call: requires USE_TORCH\n\n");
#endif
}

// ============================================================================
// Section 9: Environment speed profiling
// ============================================================================

#ifdef USE_STATIC_ENV

#include "pufferlib/extensions/env_binding.h"
#include "pufferlib/extensions/ini.h"

#ifndef ENV_NAME
#error "ENV_NAME must be defined at compile time (e.g. -DENV_NAME=breakout)"
#endif
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

static void empty_net_callback(void* ctx, int buf, int t) {
    (void)ctx; (void)buf; (void)t;
}

static void empty_thread_init(void* ctx, int buf) {
    (void)ctx; (void)buf;
}

typedef struct {
    StaticVec* vec;
    int num_envs;
    int num_buffers;
    int num_threads;
    int horizon;
    int obs_size;
    int num_atns;
} EnvSpeedArgs;

static int ini_handler_env(void* user, const char* section,
                           const char* name, const char* value) {
    Dict* env_kwargs = (Dict*)user;
    if (strcmp(section, "env") == 0) {
        dict_set(env_kwargs, strdup(name), atof(value));
    }
    return 1;
}

typedef struct { int total_agents; int num_buffers; } VecDefaults;
static int ini_handler_vec(void* user, const char* section,
                           const char* name, const char* value) {
    VecDefaults* defaults = (VecDefaults*)user;
    if (strcmp(section, "vec") == 0) {
        if (strcmp(name, "total_agents") == 0) defaults->total_agents = atoi(value);
        else if (strcmp(name, "num_buffers") == 0) defaults->num_buffers = atoi(value);
    }
    return 1;
}

EnvSpeedArgs* create_envspeedargs(int total_agents, int num_buffers, int num_threads, int horizon) {
    char ini_path[512];
    snprintf(ini_path, sizeof(ini_path), "pufferlib/config/ocean/%s.ini", TOSTRING(ENV_NAME));

    VecDefaults defaults = {0};
    if (ini_parse(ini_path, ini_handler_vec, &defaults) < 0) {
        fprintf(stderr, "Warning: Could not load config %s\n", ini_path);
    }

    if (total_agents == 0) total_agents = defaults.total_agents > 0 ? defaults.total_agents : 8192;
    if (num_buffers == 0) num_buffers = defaults.num_buffers > 0 ? defaults.num_buffers : 2;

    Dict* env_kwargs = create_dict(64);
    if (ini_parse(ini_path, ini_handler_env, env_kwargs) < 0) {
        fprintf(stderr, "Warning: Could not load [env] config from %s\n", ini_path);
    }

    Dict* vec_kwargs = create_dict(8);
    dict_set(vec_kwargs, "total_agents", (double)total_agents);
    dict_set(vec_kwargs, "num_buffers", (double)num_buffers);

    StaticVec* vec = create_static_vec(total_agents, num_buffers, vec_kwargs, env_kwargs);
    if (!vec) {
        fprintf(stderr, "Failed to create environments\n");
        return nullptr;
    }
    for (int i = 0; i < num_buffers; i++) {
        cudaStreamCreateWithFlags(&vec->streams[i], cudaStreamNonBlocking);
    }

    int num_envs = vec->size;
    printf("Created %d envs (%s) for %d total_agents\n", num_envs, TOSTRING(ENV_NAME), total_agents);

    create_static_threads(vec, num_threads, horizon, nullptr, empty_net_callback, empty_thread_init);

    static_vec_reset(vec);
    cudaDeviceSynchronize();

    EnvSpeedArgs* args = (EnvSpeedArgs*)calloc(1, sizeof(EnvSpeedArgs));
    args->vec = vec;
    args->num_envs = num_envs;
    args->num_buffers = num_buffers;
    args->num_threads = num_threads;
    args->horizon = horizon;
    args->obs_size = get_obs_size();
    args->num_atns = get_num_atns();

    return args;
}

void free_envspeedargs(EnvSpeedArgs* args) {
    free(args);
}

void run_env_rollout(EnvSpeedArgs* args) {
    static_vec_omp_step(args->vec);
}

float profile_env_rollout(EnvSpeedArgs* args, const char* name) {
    const float ENV_TIMEOUT_SEC = 3.0f;  // timeout for CPU-based env stepping

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
        run_env_rollout(args);
        cudaDeviceSynchronize();
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed > ENV_TIMEOUT_SEC) break;
    }

    start_time = get_time_sec();
    cudaProfilerStart();
    if (name) nvtxRangePushA(name);
    cudaEventRecord(start);
    float completed = 0;
    for (int i = 0; i < 1000; ++i) {
        run_env_rollout(args);
        completed += 1;
        auto now = std::chrono::steady_clock::now();
        float elapsed = std::chrono::duration<float>(now - start_time).count();
        if (elapsed > ENV_TIMEOUT_SEC) break;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (name) nvtxRangePop();
    cudaProfilerStop();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / completed;
}

void profile_envspeed(int total_agents, int num_buffers, int num_threads, int horizon) {
    printf("env_speed_static (total_agents=%d, buffers=%d, threads=%d, horizon=%d)\n",
           total_agents, num_buffers, num_threads, horizon);

    EnvSpeedArgs* args = create_envspeedargs(total_agents, num_buffers, num_threads, horizon);
    if (!args) {
        printf("  Failed to create env - skipping\n\n");
        return;
    }

    printf("  num_envs=%d, obs_size=%d, num_atns=%d\n", args->num_envs, args->obs_size, args->num_atns);

    float rollout_ms = profile_env_rollout(args, "env_rollout");
    int total_steps = total_agents * horizon;
    printf("  rollout time: %.2f ms (%d steps)\n", rollout_ms, total_steps);

    float sps = total_steps / rollout_ms * 1000.0f;
    printf("  throughput: %.2f M steps/s\n", sps / 1e6);

    free_envspeedargs(args);
    printf("\n");
}

#endif  // USE_STATIC_ENV

// ============================================================================
// Section 10: Rollout copy profiling
// ============================================================================

#ifdef USE_TORCH

// RolloutCopyArgs: synthetic rollout + TrainGraph destination buffers

typedef struct {
    // Rollout buffers (shaped after transpose: [segments, horizon, ...])
    Tensor values;        // (S, T) bf16
    Tensor rewards;       // (S, T) bf16
    Tensor terminals;     // (S, T) bf16
    Tensor ratio;         // (S, T) bf16
    Tensor observations;  // (S, T, input_size) bf16
    Tensor actions;       // (S, T, num_atns) f64
    Tensor logprobs;      // (S, T) bf16
    Tensor advantages;    // (S, T) f32

    // TrainGraph destination buffers (minibatch sized: [mb_segs, T, ...])
    Tensor mb_obs;
    Tensor mb_state;
    Tensor mb_actions;
    Tensor mb_logprobs;
    Tensor mb_advantages;
    Tensor mb_prio;
    Tensor mb_values;
    Tensor mb_returns;

    // Pre-computed prio results (for isolated select+copy profiling)
    Tensor cached_idx;
    Tensor cached_mb_prio;

    // Config
    int num_segments;     // S = total_agents (full rollout rows)
    int horizon;          // T = sequence length
    int minibatch_segs;   // how many segments per minibatch
    int input_size;
    int num_atns;
    int num_layers;
    int hidden_size;
    float gamma;
    float gae_lambda;
    float rho_clip;
    float c_clip;
    float prio_alpha;
    float anneal_beta;
    int total_agents;
} RolloutCopyArgs;

RolloutCopyArgs* create_rolloutcopyargs(int num_segments, int horizon, int minibatch_segs,
                                         int input_size, int num_atns, int num_layers, int hidden) {
    auto* args = new RolloutCopyArgs();
    args->num_segments = num_segments;
    args->horizon = horizon;
    args->minibatch_segs = minibatch_segs;
    args->input_size = input_size;
    args->num_atns = num_atns;
    args->num_layers = num_layers;
    args->hidden_size = hidden;
    args->gamma = 0.99f;
    args->gae_lambda = 0.95f;
    args->rho_clip = 1.0f;
    args->c_clip = 1.0f;
    args->prio_alpha = 0.6f;
    args->anneal_beta = 0.4f;
    args->total_agents = num_segments;

    // Create synthetic rollout data (post-transpose shape: [segments, horizon, ...])
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->values = torch::randn({num_segments, horizon}, opts);
    args->rewards = torch::randn({num_segments, horizon}, opts) * 0.1f;
    args->terminals = torch::zeros({num_segments, horizon}, opts);
    // Sprinkle some terminals
    for (int i = 0; i < num_segments / 10; i++) {
        int row = rand() % num_segments;
        int col = rand() % horizon;
        args->terminals[row][col] = 1.0f;
    }
    args->ratio = torch::ones({num_segments, horizon}, opts);
    args->observations = torch::randn({num_segments, horizon, input_size}, opts);
    args->actions = torch::randint(0, num_atns, {num_segments, horizon, num_atns}, cuda_f64);
    args->logprobs = torch::randn({num_segments, horizon}, opts) * 0.5f;
    args->advantages = torch::zeros({num_segments, horizon}, cuda_f32);

    // TrainGraph destination buffers (minibatch sized)
    args->mb_obs = torch::zeros({minibatch_segs, horizon, input_size}, opts);
    args->mb_state = torch::zeros({num_layers, minibatch_segs, 1, hidden}, opts);
    args->mb_actions = torch::zeros({minibatch_segs, horizon, num_atns}, cuda_f64);
    args->mb_logprobs = torch::zeros({minibatch_segs, horizon}, opts);
    args->mb_advantages = torch::zeros({minibatch_segs, horizon}, cuda_f32);
    args->mb_prio = torch::zeros({minibatch_segs, 1}, opts);
    args->mb_values = torch::zeros({minibatch_segs, horizon}, opts);
    args->mb_returns = torch::zeros({minibatch_segs, horizon}, opts);

    return args;
}

void free_rolloutcopyargs(RolloutCopyArgs* args) {
    delete args;
}

// Shared helpers (single source of truth for prio + select logic)

struct PrioResult {
    Tensor idx;
    Tensor mb_prio;
};

// Priority-weighted sampling: advantages -> probabilities -> multinomial -> importance weights
PrioResult compute_prio_impl(RolloutCopyArgs* args) {
    Tensor adv = args->advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(args->prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6);
    Tensor idx = at::multinomial(prio_probs, args->minibatch_segs, true);
    Tensor mb_prio = torch::pow(
        args->total_agents * prio_probs.index_select(0, idx).unsqueeze(1),
        -args->anneal_beta);
    return {idx, mb_prio};
}

// index_select sampled segments + copy_ into graph buffers
void select_and_copy_impl(RolloutCopyArgs* args, const Tensor& idx, const Tensor& mb_prio) {
    Tensor mb_obs = args->observations.index_select(0, idx);
    Tensor mb_actions = args->actions.index_select(0, idx);
    Tensor mb_logprobs = args->logprobs.index_select(0, idx);
    Tensor mb_values = args->values.index_select(0, idx);
    Tensor mb_advantages = args->advantages.index_select(0, idx);
    Tensor mb_returns = mb_advantages + mb_values;

    args->mb_state.zero_();
    args->mb_obs.copy_(mb_obs, false);
    args->mb_actions.copy_(mb_actions, false);
    args->mb_logprobs.copy_(mb_logprobs, false);
    args->mb_advantages.copy_(mb_advantages, false);
    args->mb_prio.copy_(mb_prio, false);
    args->mb_values.copy_(mb_values, false);
    args->mb_returns.copy_(mb_returns, false);
}

// Phase runners (for individual profile_kernel calls)

// Helper: run an advantage dispatch function with args unpacked
typedef void (*adv_dispatch_fn)(Tensor, Tensor, Tensor, Tensor, Tensor,
                                double, double, double, double);

void run_advantage_impl(RolloutCopyArgs* args, adv_dispatch_fn fn) {
    args->advantages.fill_(0.0);
    fn(args->values, args->rewards, args->terminals, args->ratio,
       args->advantages, args->gamma, args->gae_lambda,
       args->rho_clip, args->c_clip);
}

// Phase 1: compute_advantage — vectorized CUDA kernel (production)
void run_compute_advantage(RolloutCopyArgs* args) {
    run_advantage_impl(args, compute_puff_advantage_cuda);
}

// Phase 1 (scalar baseline): scalar-only CUDA kernel (for benchmarking)
void run_compute_advantage_scalar(RolloutCopyArgs* args) {
    run_advantage_impl(args, compute_puff_advantage_cuda_scalar);
}

// Phase 1 (PyTorch reference): GAE in pure PyTorch ops
// Sequential loop over time, but vectorized across all rows per timestep
void run_compute_advantage_torch(RolloutCopyArgs* args) {
    auto& vals = args->values;
    auto& rews = args->rewards;
    auto& terms = args->terminals;
    auto& ratio = args->ratio;
    auto& advantages = args->advantages;
    float gamma = args->gamma;
    float gae_lambda = args->gae_lambda;
    float rho_clip = args->rho_clip;
    float c_clip = args->c_clip;
    int T = args->horizon;

    advantages.zero_();
    auto lastgaelam = torch::zeros({args->num_segments}, cuda_f32);

    for (int t = T - 2; t >= 0; t--) {
        auto nextnonterminal = 1.0f - terms.select(1, t + 1).to(torch::kFloat32);
        auto imp = ratio.select(1, t).to(torch::kFloat32);
        auto rho_t = imp.clamp_max(rho_clip);
        auto c_t = imp.clamp_max(c_clip);
        auto next_val = vals.select(1, t + 1).to(torch::kFloat32);
        auto cur_val = vals.select(1, t).to(torch::kFloat32);
        auto next_rew = rews.select(1, t + 1).to(torch::kFloat32);

        auto delta = rho_t * (next_rew + gamma * next_val * nextnonterminal - cur_val);
        lastgaelam = delta + gamma * gae_lambda * c_t * lastgaelam * nextnonterminal;
        advantages.select(1, t).copy_(lastgaelam);
    }
}

// Phase 2 (PyTorch reference): compute_prio — priority-weighted sampling (all PyTorch ops)
void run_compute_prio_torch(RolloutCopyArgs* args) {
    compute_prio_impl(args);
}

// Phase 2 (CUDA kernel): fused 3-kernel compute_prio (production path)
void run_compute_prio_kernel(RolloutCopyArgs* args) {
    compute_prio_cuda(
        args->advantages, args->prio_alpha, args->minibatch_segs,
        args->total_agents, args->anneal_beta);
}

// Phase 3 (torch): train_select_and_copy — pure PyTorch select+copy using cached prio results
void run_select_and_copy_torch(RolloutCopyArgs* args) {
    select_and_copy_impl(args, args->cached_idx, args->cached_mb_prio);
}

// Phase 3 (kernel): train_select_and_copy — fused CUDA kernel using cached prio results
void run_select_and_copy_kernel(RolloutCopyArgs* args) {
    train_select_and_copy_cuda(
        args->observations, args->actions, args->logprobs,
        args->values, args->advantages,
        args->cached_idx, args->cached_mb_prio,
        args->mb_obs, args->mb_state, args->mb_actions,
        args->mb_logprobs, args->mb_advantages, args->mb_prio,
        args->mb_values, args->mb_returns);
}

// Full rollout copy: advantage + kernel prio + kernel select_and_copy (one minibatch iteration)
void run_full_rolloutcopy(RolloutCopyArgs* args) {
    nvtxRangePushA("compute_advantage");
    run_compute_advantage(args);
    nvtxRangePop();

    nvtxRangePushA("compute_prio");
    auto [idx, mb_prio] = compute_prio_cuda(
        args->advantages, args->prio_alpha, args->minibatch_segs,
        args->total_agents, args->anneal_beta);
    nvtxRangePop();

    nvtxRangePushA("train_select_and_copy");
    args->cached_idx = idx;
    args->cached_mb_prio = mb_prio;
    run_select_and_copy_kernel(args);
    nvtxRangePop();
}

// Correctness check: vectorized vs scalar (must match exactly)

void test_advantage_correct(RolloutCopyArgs* args) {
    int S = args->num_segments;
    int T = args->horizon;

    // Run vectorized kernel
    auto adv_vec = torch::zeros({S, T}, cuda_f32);
    compute_puff_advantage_cuda(
        args->values, args->rewards, args->terminals, args->ratio,
        adv_vec, args->gamma, args->gae_lambda, args->rho_clip, args->c_clip);

    // Run scalar kernel
    auto adv_scalar = torch::zeros({S, T}, cuda_f32);
    compute_puff_advantage_cuda_scalar(
        args->values, args->rewards, args->terminals, args->ratio,
        adv_scalar, args->gamma, args->gae_lambda, args->rho_clip, args->c_clip);

    cudaDeviceSynchronize();

    // Compare: should be bit-identical (same float arithmetic, just different load pattern)
    float max_diff = (adv_vec - adv_scalar).abs().max().item<float>();
    const char* status = (max_diff < 1e-6f) ? "PASS" : "FAIL";
    printf("  correctness (vec vs scalar): %s (max_diff=%.2e)\n", status, max_diff);

    // Also compare against PyTorch reference (may have small fp differences)
    auto saved_adv = args->advantages;
    args->advantages = torch::zeros({S, T}, cuda_f32);
    run_compute_advantage_torch(args);
    auto adv_torch = args->advantages;
    args->advantages = saved_adv;
    cudaDeviceSynchronize();

    float max_diff_torch = (adv_vec - adv_torch).abs().max().item<float>();
    float atol = USE_BF16 ? 5e-2f : 1e-5f;  // bf16 inputs lose precision in .to() conversions
    const char* torch_status = (max_diff_torch < atol) ? "PASS" : "FAIL";
    printf("  correctness (vec vs torch):  %s (max_diff=%.2e)\n", torch_status, max_diff_torch);
}

// Correctness check: kernel prio vs PyTorch prio

void test_prio_correct(RolloutCopyArgs* args) {
    // Run the full kernel pipeline
    auto [idx, mb_prio_kernel] = compute_prio_cuda(
        args->advantages, args->prio_alpha, args->minibatch_segs,
        args->total_agents, args->anneal_beta);

    // PyTorch reference probs
    Tensor adv = args->advantages.abs().sum(1);
    Tensor prio_weights = adv.pow(args->prio_alpha).nan_to_num_(0.0, 0.0, 0.0);
    Tensor probs_torch = (prio_weights + 1e-6f) / (prio_weights.sum() + 1e-6f);

    // Check full pipeline: kernel's mb_prio vs torch mb_prio (using kernel's idx)
    Tensor mb_prio_torch = torch::pow(
        (float)args->total_agents * probs_torch.index_select(0, idx).unsqueeze(1),
        -args->anneal_beta);

    cudaDeviceSynchronize();

    float prio_diff = (mb_prio_kernel - mb_prio_torch).abs().max().item<float>();
    const char* prio_status = (prio_diff < 1e-2f) ? "PASS" : "FAIL";
    printf("  correctness (mb_prio):  %s (max_diff=%.2e)\n", prio_status, prio_diff);
}

// Correctness check: kernel select+copy vs PyTorch select+copy

void test_select_copy_correct(RolloutCopyArgs* args) {
    // Pre-compute prio results for both paths
    auto [idx, mb_prio] = compute_prio_cuda(
        args->advantages, args->prio_alpha, args->minibatch_segs,
        args->total_agents, args->anneal_beta);
    args->cached_idx = idx;
    args->cached_mb_prio = mb_prio;

    // Run PyTorch reference path
    run_select_and_copy_torch(args);
    auto ref_obs = args->mb_obs.clone();
    auto ref_actions = args->mb_actions.clone();
    auto ref_logprobs = args->mb_logprobs.clone();
    auto ref_advantages = args->mb_advantages.clone();
    auto ref_values = args->mb_values.clone();
    auto ref_returns = args->mb_returns.clone();

    // Zero destination buffers and run kernel path
    args->mb_obs.zero_();
    args->mb_actions.zero_();
    args->mb_logprobs.zero_();
    args->mb_advantages.zero_();
    args->mb_values.zero_();
    args->mb_returns.zero_();
    run_select_and_copy_kernel(args);

    cudaDeviceSynchronize();

    // Compare all outputs
    auto check = [](const char* name, Tensor& kernel, Tensor& ref) {
        float diff = (kernel.to(torch::kFloat32) - ref.to(torch::kFloat32)).abs().max().item<float>();
        const char* status = (diff < 1e-5f) ? "PASS" : "FAIL";
        printf("  %-16s %s (max_diff=%.2e)\n", name, status, diff);
    };

    check("obs", args->mb_obs, ref_obs);
    check("actions", args->mb_actions, ref_actions);
    check("logprobs", args->mb_logprobs, ref_logprobs);
    check("advantages", args->mb_advantages, ref_advantages);
    check("values", args->mb_values, ref_values);
    check("returns", args->mb_returns, ref_returns);
}

// Profile function

void profile_rolloutcopy(int num_segments, int horizon, int minibatch_segs,
                          int input_size, int num_atns, int num_layers, int hidden) {
    printf("========================================\n");
    printf("rolloutcopy (S=%d, T=%d, mb_segs=%d, in=%d, A=%d, H=%d)\n",
           num_segments, horizon, minibatch_segs, input_size, num_atns, hidden);
    printf("  rollout_rows=%d, minibatch=%d, using %s\n",
           num_segments, minibatch_segs * horizon, USE_BF16 ? "bf16" : "fp32");
    printf("========================================\n\n");

    auto* args = create_rolloutcopyargs(num_segments, horizon, minibatch_segs,
                                         input_size, num_atns, num_layers, hidden);

    // === Advantage Kernel Comparison ===
    printf("--- Advantage Kernel Comparison (S=%d, T=%d) ---\n", num_segments, horizon);

    float adv_vec_ms = profile_kernel((kernel_fn)run_compute_advantage, args, "advantage_vectorized");
    print_timing("advantage (vectorized)", adv_vec_ms, num_segments);

    float adv_scalar_ms = profile_kernel((kernel_fn)run_compute_advantage_scalar, args, "advantage_scalar");
    print_timing("advantage (scalar)", adv_scalar_ms, num_segments);

    float adv_torch_ms = profile_kernel((kernel_fn)run_compute_advantage_torch, args, "advantage_torch");
    print_timing("advantage (torch)", adv_torch_ms, num_segments);

    printf("  vectorized vs scalar:  %.2fx\n", adv_scalar_ms / adv_vec_ms);
    printf("  vectorized vs torch:   %.2fx\n", adv_torch_ms / adv_vec_ms);

    test_advantage_correct(args);
    printf("\n");

    // === Prio Kernel Comparison ===
    printf("--- Prio Kernel Comparison (S=%d, T=%d, mb=%d) ---\n",
           num_segments, horizon, minibatch_segs);

    // Ensure advantages are populated for prio
    run_compute_advantage(args);

    float prio_torch_ms = profile_kernel((kernel_fn)run_compute_prio_torch, args, "prio_torch");
    print_timing("prio (torch)", prio_torch_ms, num_segments);

    float prio_kernel_ms = profile_kernel((kernel_fn)run_compute_prio_kernel, args, "prio_kernel");
    print_timing("prio (kernel)", prio_kernel_ms, num_segments);

    printf("  kernel vs torch:       %.2fx\n", prio_torch_ms / prio_kernel_ms);

    test_prio_correct(args);
    printf("\n");

    // === Select+Copy Kernel Comparison ===
    printf("--- Select+Copy Kernel Comparison (mb=%d) ---\n", minibatch_segs);

    // Ensure advantages + cached prio are populated
    run_compute_advantage(args);
    {
        auto [idx, mb_prio] = compute_prio_cuda(
            args->advantages, args->prio_alpha, args->minibatch_segs,
            args->total_agents, args->anneal_beta);
        args->cached_idx = idx;
        args->cached_mb_prio = mb_prio;
    }

    float copy_torch_ms = profile_kernel((kernel_fn)run_select_and_copy_torch, args, "select_copy_torch");
    print_timing("select+copy (torch)", copy_torch_ms, minibatch_segs);

    float copy_kernel_ms = profile_kernel((kernel_fn)run_select_and_copy_kernel, args, "select_copy_kernel");
    print_timing("select+copy (kernel)", copy_kernel_ms, minibatch_segs);

    printf("  kernel vs torch:       %.2fx\n", copy_torch_ms / copy_kernel_ms);

    test_select_copy_correct(args);
    printf("\n");

    // === Per-Phase Timing (rollout copy pipeline, using kernel paths) ===
    printf("--- Per-Phase Timing ---\n");

    float adv_ms = adv_vec_ms;
    print_timing("compute_advantage", adv_ms, num_segments);

    run_compute_advantage(args);

    float prio_ms = prio_kernel_ms;  // use kernel prio (production path)
    print_timing("compute_prio", prio_ms, num_segments);

    float copy_ms = copy_kernel_ms;  // use kernel select+copy (production path)
    print_timing("train_select_and_copy", copy_ms, minibatch_segs);
    printf("\n");

    // --- Full rollout copy (all 3 phases, fresh args) ---
    printf("--- Full Rollout Copy (one minibatch iteration) ---\n");

    auto* full_args = create_rolloutcopyargs(num_segments, horizon, minibatch_segs,
                                         input_size, num_atns, num_layers, hidden);
    float full_ms = profile_kernel((kernel_fn)run_full_rolloutcopy, full_args, "rolloutcopy_full");
    print_timing("rolloutcopy (full)", full_ms, num_segments);
    free_rolloutcopyargs(full_args);
    printf("\n");

    // --- Proportional breakdown ---
    printf("--- Proportional Breakdown ---\n");
    float total_phases = adv_ms + prio_ms + copy_ms;
    print_timing_pct("compute_advantage", adv_ms, num_segments, total_phases);
    print_timing_pct("compute_prio", prio_ms, num_segments, total_phases);
    print_timing_pct("train_select_and_copy", copy_ms, minibatch_segs, total_phases);
    printf("  %-28s %8.1f us  100.0%%\n", "total (sum of phases)", total_phases * 1000);
    printf("  %-28s %8.1f us  (measured)\n", "full rolloutcopy actual", full_ms * 1000);
    printf("\n");

    free_rolloutcopyargs(args);
}

#endif  // USE_TORCH (rolloutcopy)

// ============================================================================
// Section 11: Training profiling
// ============================================================================

#ifdef USE_TORCH

// TrainArgs: shared state for training profiles

typedef struct {
    // Model
    std::shared_ptr<Policy> policy_bf16;
    std::shared_ptr<Policy> policy_fp32;  // fp32 master weights for mixed precision

    // Training data (shaped for minibatch)
    Tensor obs;           // (N, T, input_size) - training observations
    Tensor state;         // (num_layers, N, 1, H) - initial RNN state for training
    Tensor actions;       // (N, T, 1) float64
    Tensor old_logprobs;  // (N, T)
    Tensor advantages;    // (N, T)
    Tensor prio;          // (N, 1)
    Tensor values;        // (N, T)
    Tensor returns;       // (N, T)
    Tensor ratio_out;     // (N, T) - side-effect output
    Tensor newvalue_out;  // (N, T) - side-effect output
    Tensor act_sizes;     // (1,) int32 cuda
    Tensor act_sizes_cpu; // (1,) int64 cpu
    Tensor losses;        // (NUM_LOSSES,) float32 accumulator

    // Muon optimizer (matches production — NOT Adam)
    std::shared_ptr<torch::optim::Muon> muon;

    // Config
    bool use_kernels;
    int N;                // number of segments (batch dim before T)
    int T_seq;            // sequence length
    int H;
    int A;
    int input_size;
    float clip_coef;
    float vf_clip_coef;
    float vf_coef;
    float ent_coef;
    float max_grad_norm;
} TrainArgs;

TrainArgs* create_trainargs(int N, int T_seq, int input_size, int hidden, int act_n,
                            int num_layers, bool use_kernels) {
    TrainArgs* args = new TrainArgs();
    args->use_kernels = use_kernels;
    args->N = N;
    args->T_seq = T_seq;
    args->H = hidden;
    args->A = act_n;
    args->input_size = input_size;
    args->clip_coef = 0.1f;
    args->vf_clip_coef = 0.1f;
    args->vf_coef = 0.5f;
    args->ent_coef = 0.01f;
    args->max_grad_norm = 0.5f;

    // Create primary policy (bf16 when mixed-precision, fp32 otherwise)
    auto enc = std::make_shared<DefaultEncoder>(input_size, hidden);
    auto dec = std::make_shared<DefaultDecoder>(hidden, act_n);
    auto rnn = std::make_shared<MinGRU>(hidden, num_layers, use_kernels);
    args->policy_bf16 = std::make_shared<Policy>(enc, dec, rnn, input_size, act_n, hidden);
    args->policy_bf16->to(torch::kCUDA);
    if (USE_BF16) {
        args->policy_bf16->to(torch::kBFloat16);
    }

    // Create fp32 master weights (for mixed-precision training)
    auto enc32 = std::make_shared<DefaultEncoder>(input_size, hidden);
    auto dec32 = std::make_shared<DefaultDecoder>(hidden, act_n);
    auto rnn32 = std::make_shared<MinGRU>(hidden, num_layers, use_kernels);
    args->policy_fp32 = std::make_shared<Policy>(enc32, dec32, rnn32, input_size, act_n, hidden);
    args->policy_fp32->to(torch::kCUDA);

    // Sync bf16 from fp32 (only needed in mixed precision)
    if (USE_BF16) {
        sync_policy_weights(args->policy_bf16.get(), args->policy_fp32.get());
    }

    // Create Muon optimizer over fp32 master weights (matches production)
    args->muon = std::make_shared<torch::optim::Muon>(
        args->policy_fp32->parameters(),
        torch::optim::MuonOptions(0.0025));
    args->muon->init_contiguous_weights();

    // Create synthetic training data
    auto opts = torch::dtype(PRECISION_DTYPE).device(torch::kCUDA);
    args->obs = torch::randn({N, T_seq, input_size}, opts);
    args->state = torch::zeros({num_layers, N, 1, hidden}, opts);
    args->actions = torch::randint(0, act_n, {N, T_seq, 1}, cuda_f64);
    args->old_logprobs = torch::randn({N, T_seq}, opts) * 0.5f;
    args->advantages = torch::randn({N, T_seq}, cuda_f32);
    args->prio = torch::ones({N, 1}, opts);
    args->values = torch::randn({N, T_seq}, opts);
    args->returns = torch::randn({N, T_seq}, opts);
    args->ratio_out = torch::zeros({N, T_seq}, opts);
    args->newvalue_out = torch::zeros({N, T_seq}, opts);
    args->act_sizes = torch::tensor({act_n}, cuda_i32);
    args->act_sizes_cpu = torch::tensor({(int64_t)act_n}, torch::dtype(torch::kInt64));
    args->losses = torch::zeros({NUM_LOSSES}, cuda_f32);

    return args;
}

void free_trainargs(TrainArgs* args) {
    delete args;
}

// Helper: compute loss from forward outputs (eliminates duplicated arg list)

Tensor compute_loss_impl(TrainArgs* args, Logits& raw_logits, Tensor& newvalue) {
    Logits ls = {.mean = raw_logits.mean};
    if (raw_logits.logstd.defined()) ls.logstd = raw_logits.logstd;
    int mb = args->N * args->T_seq;
    return compute_train_loss(
        ls, newvalue,
        args->actions, args->old_logprobs, args->advantages, args->prio,
        args->values, args->returns, args->ratio_out, args->newvalue_out,
        args->act_sizes, args->act_sizes_cpu, mb, args->T_seq,
        args->clip_coef, args->vf_clip_coef, args->vf_coef, args->ent_coef,
        /*is_continuous=*/false, args->use_kernels, args->losses);
}

// Run functions for individual phases

// Full training forward + loss (no backward)
// NoGradGuard: we only measure forward time, no backward will follow.
// CUDA kernels execute identically — autograd tracking is CPU-side only.
void run_train_forward(TrainArgs* args) {
    torch::NoGradGuard no_grad;
    auto [logits, newvalue] = args->policy_bf16->forward_train(args->obs, args->state);
    compute_loss_impl(args, logits, newvalue);
}

// Full training forward + loss + backward
void run_train_forward_backward(TrainArgs* args) {
    args->policy_bf16->zero_grad();

    nvtxRangePushA("forward_train");
    auto [logits, newvalue] = args->policy_bf16->forward_train(args->obs, args->state);
    nvtxRangePop();

    nvtxRangePushA("compute_loss");
    auto loss = compute_loss_impl(args, logits, newvalue);
    nvtxRangePop();

    nvtxRangePushA("backward");
    loss.backward();
    nvtxRangePop();
}

// Full training step: forward + loss + backward + Muon optimizer (matches production)
void run_train_step(TrainArgs* args) {
    args->muon->zero_grad();
    args->policy_bf16->zero_grad();

    nvtxRangePushA("forward_train");
    auto [logits, newvalue] = args->policy_bf16->forward_train(args->obs, args->state);
    nvtxRangePop();

    nvtxRangePushA("compute_loss");
    auto loss = compute_loss_impl(args, logits, newvalue);
    nvtxRangePop();

    nvtxRangePushA("backward");
    loss.backward();
    nvtxRangePop();

    nvtxRangePushA("grad_sync");
    if (USE_BF16) {
        copy_gradients_to_fp32(args->policy_bf16.get(), args->policy_fp32.get());
    }
    clip_grad_norm_(args->policy_fp32->parameters(), args->max_grad_norm);
    nvtxRangePop();

    nvtxRangePushA("muon_step");
    args->muon->step();
    nvtxRangePop();

    nvtxRangePushA("weight_sync");
    args->muon->zero_grad();
    args->policy_bf16->zero_grad();
    if (USE_BF16) {
        sync_policy_weights(args->policy_bf16.get(), args->policy_fp32.get());
    }
    nvtxRangePop();
}

// Per-section isolated profiling (for forward breakdown)

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor obs;       // (N, T, input)
    Tensor state;     // (layers, N, 1, H)
    int N, T_seq, H, input_size;
} EncoderIsolatedArgs;

void run_encoder_isolated(EncoderIsolatedArgs* args) {
    torch::NoGradGuard no_grad;
    int B = args->N;
    int TT = args->T_seq;
    auto x = args->obs.reshape({B * TT, args->input_size});
    args->policy->encoder->forward(x);
}

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor h_encoded;  // (N, T, H)
    Tensor state;      // (layers, N, 1, H)
} RNNIsolatedArgs;

void run_rnn_isolated(RNNIsolatedArgs* args) {
    torch::NoGradGuard no_grad;
    args->policy->rnn->forward_train(args->h_encoded, args->state);
}

typedef struct {
    std::shared_ptr<Policy> policy;
    Tensor flat_h;  // (N*T, H)
} DecoderIsolatedArgs;

void run_decoder_isolated(DecoderIsolatedArgs* args) {
    torch::NoGradGuard no_grad;
    args->policy->decoder->forward(args->flat_h);
}

// Instrumented breakdowns: CUDA events at phase boundaries within a single run

struct StepTimings {
    float forward_ms;
    float loss_ms;
    float backward_ms;
    float grad_sync_ms;
    float muon_ms;
    float weight_sync_ms;
};

StepTimings profile_step_instrumented(TrainArgs* args, int num_iters = 200) {
    // Warmup
    for (int i = 0; i < 10; i++) run_train_step(args);
    cudaDeviceSynchronize();

    // Pre-allocate all events — NO per-iteration sync keeps GPU pipeline full
    const int P = 7;  // 7 boundary markers per iteration
    std::vector<cudaEvent_t> events(num_iters * P);
    for (auto& e : events) cudaEventCreate(&e);

    for (int i = 0; i < num_iters; i++) {
        args->muon->zero_grad();
        args->policy_bf16->zero_grad();
        int b = i * P;

        cudaEventRecord(events[b + 0]);
        auto [logits, newvalue] = args->policy_bf16->forward_train(args->obs, args->state);
        cudaEventRecord(events[b + 1]);

        auto loss = compute_loss_impl(args, logits, newvalue);
        cudaEventRecord(events[b + 2]);

        loss.backward();
        cudaEventRecord(events[b + 3]);

        if (USE_BF16) copy_gradients_to_fp32(args->policy_bf16.get(), args->policy_fp32.get());
        clip_grad_norm_(args->policy_fp32->parameters(), args->max_grad_norm);
        cudaEventRecord(events[b + 4]);

        args->muon->step();
        cudaEventRecord(events[b + 5]);

        args->muon->zero_grad();
        args->policy_bf16->zero_grad();
        if (USE_BF16) sync_policy_weights(args->policy_bf16.get(), args->policy_fp32.get());
        cudaEventRecord(events[b + 6]);
    }

    cudaDeviceSynchronize();  // single sync at end

    StepTimings sum = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < num_iters; i++) {
        int b = i * P;
        float ms;
        cudaEventElapsedTime(&ms, events[b+0], events[b+1]); sum.forward_ms += ms;
        cudaEventElapsedTime(&ms, events[b+1], events[b+2]); sum.loss_ms += ms;
        cudaEventElapsedTime(&ms, events[b+2], events[b+3]); sum.backward_ms += ms;
        cudaEventElapsedTime(&ms, events[b+3], events[b+4]); sum.grad_sync_ms += ms;
        cudaEventElapsedTime(&ms, events[b+4], events[b+5]); sum.muon_ms += ms;
        cudaEventElapsedTime(&ms, events[b+5], events[b+6]); sum.weight_sync_ms += ms;
    }

    for (auto& e : events) cudaEventDestroy(e);

    float n = (float)num_iters;
    return { sum.forward_ms/n, sum.loss_ms/n, sum.backward_ms/n,
             sum.grad_sync_ms/n, sum.muon_ms/n, sum.weight_sync_ms/n };
}

// Main profile functions

void profile_trainforward(int N, int T_seq, int input_size, int hidden, int act_n, int num_layers) {
    printf("========================================\n");
    printf("trainforward (N=%d, T=%d, in=%d, H=%d, A=%d, layers=%d)\n",
           N, T_seq, input_size, hidden, act_n, num_layers);
    printf("  minibatch=%d, using %s\n", N * T_seq,
           USE_BF16 ? "bf16" : "fp32");
    printf("========================================\n\n");

    bool use_kernels = true;

    // ----- Full forward + loss (no backward) -----
    printf("--- Forward + Loss (no backward, no autograd) ---\n");
    TrainArgs* args_fwd = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
    float fwd_ms = profile_kernel((kernel_fn)run_train_forward, args_fwd, "trainforward");
    print_timing("forward+loss (kernel)", fwd_ms, N * T_seq);
    // args_fwd kept alive — reused by isolated phase breakdown below

    {
        TrainArgs* args_fwd_cpp = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, false);
        float fwd_cpp_ms = profile_kernel((kernel_fn)run_train_forward, args_fwd_cpp, "trainforward_cpp");
        print_timing("forward+loss (cpp)", fwd_cpp_ms, N * T_seq);
        free_trainargs(args_fwd_cpp);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    printf("\n");

    // ----- Full forward + loss + backward -----
    printf("--- Forward + Loss + Backward ---\n");
    float fb_ms, fb_cpp_ms;
    {
        TrainArgs* args_fb = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
        fb_ms = profile_kernel((kernel_fn)run_train_forward_backward, args_fb, "train_fwd_bwd");
        print_timing("fwd+loss+bwd (kernel)", fb_ms, N * T_seq);
        free_trainargs(args_fb);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    {
        TrainArgs* args_fb_cpp = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, false);
        fb_cpp_ms = profile_kernel((kernel_fn)run_train_forward_backward, args_fb_cpp, "train_fwd_bwd_cpp");
        print_timing("fwd+loss+bwd (cpp)", fb_cpp_ms, N * T_seq);
        free_trainargs(args_fb_cpp);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    printf("\n");

    // ----- Per-section breakdown (forward only, no autograd) -----
    printf("--- Phase Breakdown (forward only, no autograd) ---\n");

    // Encoder
    auto enc_args = new EncoderIsolatedArgs();
    enc_args->policy = args_fwd->policy_bf16;
    enc_args->obs = args_fwd->obs;
    enc_args->state = args_fwd->state;
    enc_args->N = N;
    enc_args->T_seq = T_seq;
    enc_args->H = hidden;
    enc_args->input_size = input_size;
    float enc_ms = profile_kernel((kernel_fn)run_encoder_isolated, enc_args, "encoder");
    delete enc_args;
    c10::cuda::CUDACachingAllocator::emptyCache();

    // RNN
    auto rnn_args = new RNNIsolatedArgs();
    rnn_args->policy = args_fwd->policy_bf16;
    {
        torch::NoGradGuard no_grad;
        auto x = args_fwd->obs.reshape({N * T_seq, input_size});
        auto h = args_fwd->policy_bf16->encoder->forward(x);
        rnn_args->h_encoded = h.reshape({N, T_seq, hidden});
    }
    rnn_args->state = args_fwd->state;
    float rnn_ms = profile_kernel((kernel_fn)run_rnn_isolated, rnn_args, "rnn_scan");
    delete rnn_args;
    c10::cuda::CUDACachingAllocator::emptyCache();

    // Decoder
    auto dec_args = new DecoderIsolatedArgs();
    dec_args->policy = args_fwd->policy_bf16;
    {
        torch::NoGradGuard no_grad;
        auto x = args_fwd->obs.reshape({N * T_seq, input_size});
        auto h = args_fwd->policy_bf16->encoder->forward(x);
        h = h.reshape({N, T_seq, hidden});
        h = args_fwd->policy_bf16->rnn->forward_train(h, args_fwd->state);
        dec_args->flat_h = h.reshape({-1, hidden});
    }
    float dec_ms = profile_kernel((kernel_fn)run_decoder_isolated, dec_args, "decoder");
    delete dec_args;

    float total_phases = enc_ms + rnn_ms + dec_ms;
    print_timing_pct("encoder (linear)", enc_ms, N * T_seq, total_phases);
    print_timing_pct("rnn (fused_scan)", rnn_ms, N * T_seq, total_phases);
    print_timing_pct("decoder (linear)", dec_ms, N * T_seq, total_phases);
    printf("  %-28s %8.1f us  100.0%%\n", "total (sum of phases)", total_phases * 1000);
    printf("  %-28s %8.1f us  (measured)\n", "forward+loss actual", fwd_ms * 1000);
    printf("\n");

    free_trainargs(args_fwd);
}

void profile_trainstep(int N, int T_seq, int input_size, int hidden, int act_n, int num_layers) {
    printf("========================================\n");
    printf("trainstep (N=%d, T=%d, in=%d, H=%d, A=%d, layers=%d)\n",
           N, T_seq, input_size, hidden, act_n, num_layers);
    printf("  minibatch=%d, using %s, optimizer=Muon\n", N * T_seq,
           USE_BF16 ? "bf16" : "fp32");
    printf("========================================\n\n");

    bool use_kernels = true;

    // ----- Full training step (eager) -----
    printf("--- Full Training Step: fwd + loss + bwd + clip + Muon + sync ---\n");
    float step_ms;
    {
        TrainArgs* args = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
        step_ms = profile_kernel((kernel_fn)run_train_step, args, "trainstep");
        print_timing("trainstep (kernel)", step_ms, N * T_seq);
        free_trainargs(args);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }
    {
        TrainArgs* args_cpp = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, false);
        float step_cpp_ms = profile_kernel((kernel_fn)run_train_step, args_cpp, "trainstep_cpp");
        print_timing("trainstep (cpp)", step_cpp_ms, N * T_seq);
        free_trainargs(args_cpp);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // ----- Instrumented breakdown (CUDA events at phase boundaries) -----
    printf("\n--- Training Step Breakdown (instrumented, includes autograd overhead) ---\n");
    {
        TrainArgs* args_bd = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
        auto t = profile_step_instrumented(args_bd);
        float total = t.forward_ms + t.loss_ms + t.backward_ms
                    + t.grad_sync_ms + t.muon_ms + t.weight_sync_ms;
        print_timing_pct("forward", t.forward_ms, N * T_seq, total);
        print_timing_pct("loss", t.loss_ms, N * T_seq, total);
        print_timing_pct("backward", t.backward_ms, N * T_seq, total);
        print_timing_pct("grad_sync+clip", t.grad_sync_ms, N * T_seq, total);
        print_timing_pct("Muon step", t.muon_ms, N * T_seq, total);
        print_timing_pct("weight_sync", t.weight_sync_ms, N * T_seq, total);
        printf("  %-28s %8.1f us  100.0%%\n", "total step", total * 1000);
        printf("  %-28s %8.1f us  (profile_kernel)\n", "trainstep measured", step_ms * 1000);
        printf("\n");
        free_trainargs(args_bd);
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // ----- CUDA Graph training step -----
    // Production captures fwd+loss+bwd+optim+sync as a single graph.
    // profile_graph does warmup → capture → timed replay, matching production.
    printf("--- CUDA Graph Training Step ---\n");
    {
        TrainArgs* args_graph = create_trainargs(N, T_seq, input_size, hidden, act_n, num_layers, use_kernels);
        float graph_ms = profile_graph((kernel_fn)run_train_step, args_graph, "trainstep_graph");
        print_timing("trainstep (graph)", graph_ms, N * T_seq);

        float graph_speedup = step_ms / graph_ms;
        printf("  graph speedup vs eager:    %.2fx\n", graph_speedup);
        printf("\n");
        free_trainargs(args_graph);
    }
}

#endif  // USE_TORCH (training)

// ============================================================================
// Section 12: main() dispatcher
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s <profile>\n", prog);
    printf("\nProfiles:\n");
    printf("  kernels        - All individual kernel microbenchmarks\n");
    printf("  mingrugate     - MinGRU gate kernel only\n");
    printf("  logcumsumexp   - Logcumsumexp kernel only\n");
    printf("  fusedscan      - Fused scan (checkpointed) kernel only\n");
    printf("  samplelogits   - Sample logits kernel only\n");
    printf("  ppoloss        - PPO loss kernel only\n");
    printf("  fcmax          - FC+Max kernel only\n");
#ifdef USE_TORCH
    printf("  forwardcall    - Inference forward pass (requires torch)\n");
    printf("  trainforward   - Training fwd+loss+bwd breakdown (requires torch)\n");
    printf("  trainstep      - Full training step with Muon optimizer (requires torch)\n");
    printf("  rolloutcopy    - Per-minibatch data prep: advantage+prio+copy (requires torch)\n");
#endif
#ifdef USE_STATIC_ENV
    printf("  envspeed       - Environment step throughput (static linked)\n");
    printf("    --buffers N  - Number of buffers (default: %d)\n", BUF);
    printf("    --threads N  - Number of threads (default: 16)\n");
    printf("    --horizon N  - Horizon length (default: %d)\n", T);
#endif
    printf("  all            - Run all available profiles\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* profile = argv[1];

    // Parse optional CLI args for envspeed
    int buffers = BUF;
    int threads = 16;
    int horizon = T;
    int total_agents = BR * buffers;
    for (int i = 2; i < argc - 1; i++) {
        if (strcmp(argv[i], "--buffers") == 0) buffers = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--horizon") == 0) horizon = atoi(argv[++i]);
        else if (strcmp(argv[i], "--total-agents") == 0) total_agents = atoi(argv[++i]);
    }

    warmup_gpu();

    // Using typical breakout settings: INPUT_SIZE=96, H=128, A=4
    bool run_all = strcmp(profile, "all") == 0;

    // === Individual kernel microbenchmarks ===
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "mingrugate") == 0 || run_all) {
        profile_mingrugate(BR, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "logcumsumexp") == 0 || run_all) {
        profile_logcumsumexp(BT, T, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "fusedscan") == 0 || run_all) {
        profile_fusedscan(BT, T, H);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "samplelogits") == 0 || run_all) {
        profile_samplelogits(BR, A);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "ppoloss") == 0 || run_all) {
        profile_ppoloss(BT, T, A);
    }
    if (strcmp(profile, "kernels") == 0 || strcmp(profile, "fcmax") == 0 || run_all) {
        profile_fcmax(BR, 63, 7, 128);    // partner encoder (drive)
        profile_fcmax(BR, 200, 13, 128);  // road encoder (drive)
    }

    // === Composite profiles (require torch) ===
#ifdef USE_TORCH
    if (strcmp(profile, "forwardcall") == 0 || run_all) {
        profile_forwardcall(BR, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "trainforward") == 0 || run_all) {
        profile_trainforward(BT, T, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "trainstep") == 0 || run_all) {
        profile_trainstep(BT, T, INPUT_SIZE, H, A, 1);
    }
    if (strcmp(profile, "rolloutcopy") == 0 || run_all) {
        // num_segments = BR*BUF (full rollout), minibatch_segs = BT
        profile_rolloutcopy(BR * BUF, T, BT, INPUT_SIZE, A, 1, H);
    }
#endif

    // === Environment speed (requires static env link) ===
#ifdef USE_STATIC_ENV
    if (strcmp(profile, "envspeed") == 0 || strcmp(profile, "all") == 0) {
        profile_envspeed(total_agents, buffers, threads, horizon);
    }
#endif

    if (!run_all
        && strcmp(profile, "kernels") != 0
        && strcmp(profile, "mingrugate") != 0
        && strcmp(profile, "logcumsumexp") != 0
        && strcmp(profile, "fusedscan") != 0
        && strcmp(profile, "samplelogits") != 0
        && strcmp(profile, "ppoloss") != 0
        && strcmp(profile, "fcmax") != 0
#ifdef USE_TORCH
        && strcmp(profile, "forwardcall") != 0
        && strcmp(profile, "trainforward") != 0
        && strcmp(profile, "trainstep") != 0
        && strcmp(profile, "rolloutcopy") != 0
#endif
#ifdef USE_STATIC_ENV
        && strcmp(profile, "envspeed") != 0
#endif
    ) {
        printf("Unknown profile: %s\n\n", profile);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
