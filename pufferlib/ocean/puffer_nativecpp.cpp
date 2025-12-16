#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include "puffer_nativecpp.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <torch/torch.h>


#if DEBUG
constexpr bool debug_mode = true;
#else
constexpr bool debug_mode = false;
#endif
#ifdef PUFFER_CUDA
constexpr bool cuda_async = true;
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
using namespace ::c10::cuda;
// Uncomment this to print memory info while debugging.
// #define PUFFER_CUDA_MEMCHECK 1
#else
constexpr bool cuda_async = false;
#endif

#ifdef PUFFER_CUDA_MEMCHECK
inline void print_cuda_mem_info(std::string name, bool print_detailed = false)
{
  if (!torch::cuda::is_available()) return;

  // Get memory info
  const c10::CachingDeviceAllocator::DeviceStats stats = CUDACachingAllocator::getDeviceStats(
    c10::cuda::current_device());

  for (int i = 0; i < stats.allocated_bytes.size(); ++i)
  {
    std::cout << "Cuda mem stats: " << name << "_" << i << ":\t\t\t"
        << " [Allocated : " << (stats.allocated_bytes[i].current / (1024.0 * 1024.0)) << " MB ]"
        << " [Reserved bytes: " << (stats.reserved_bytes[i].current / (1024.0 * 1024.0)) << " MB ]"
        << " [Active allocs: " << stats.allocation[i].current << "]\n";
  }
  if (print_detailed)
  {
    size_t largestBlock = 0;
    CUDACachingAllocator::cacheInfo(c10::cuda::current_device(), &largestBlock);
    std::cout << "Cuda mem stats: " << name << "_detailed:\t"
        << " [Largest free block: " << (largestBlock / (1024.0 * 1024.0)) << " MB ]\n";
    // Get and print snapshot
    try
    {
      auto snapshot = CUDACachingAllocator::snapshot();

      std::cout << "Memory Snapshot for " << name << ":\n";
      std::cout << "  Device traces: " << snapshot.device_traces.size() << "\n";
      std::cout << "  Segments: " << snapshot.segments.size() << "\n";

      // Print top memory consuming segments
      size_t total_allocated = 0;
      size_t total_reserved = 0;
      int segment_count = 0;

      for (const auto& seg : snapshot.segments)
      {
        total_allocated += seg.allocated_size;
        total_reserved += seg.total_size;
        if (seg.allocated_size > 1024 * 256)
        {
          std::cout << "    Segment " << segment_count++
              << ": allocated=" << (seg.allocated_size / (1024.0 * 1024.0)) << " MB"
              << ", total=" << (seg.total_size / (1024.0 * 1024.0)) << " MB"
              << ", stream=" << seg.stream << "\n";
        }
      }

      std::cout << "  Total allocated: " << (total_allocated / (1024.0 * 1024.0)) << " MB\n";
      std::cout << "  Total reserved: " << (total_reserved / (1024.0 * 1024.0)) << " MB\n";
    }
    catch (const std::exception& e)
    {
      std::cout << "Error getting snapshot: " << e.what() << "\n";
    }
  }
}
#else
// Completely eliminate any std::string ops etc for non-mem-check builds.
#define print_cuda_mem_info(_1, ...) ((void)0)
#endif

using torch::Tensor;
using namespace std;

// LibTorch throws exceptions on errors, log them correctly in debug mode only.
#if DEBUG
#define BEGIN_LIBTORCH_CATCH try
#else
#define BEGIN_LIBTORCH_CATCH
#endif

#if DEBUG
#define END_LIBTORCH_CATCH                                                                                             \
  catch (const c10::Error& e)                                                                                          \
  {                                                                                                                    \
    std::cerr << "Error from libtorch: " << e.what() << std::endl;                                                     \
    PUFFER_ASSERT_BREAK();                                                                                             \
    throw;                                                                                                             \
  }

#else
#define END_LIBTORCH_CATCH
#endif


#ifndef PUFFER_EXTERN
// Silliness as the header is included in both C and C++ files (and from binding.c from each env). Makes it very hard to
// separate it.
struct Env;
struct VecEnv;
#define PUFFER_EXTERN extern "C"
#endif

// Have to manually pass the actions_data so each env can choose to decipher actions (for e.g. breakout uses float* for discrete actions).
PUFFER_EXTERN void c_step_batch(void* arg, int env_index, void* actions_data, int num_actions, float* rewards,
  float* terminals);

// Optional completion function that will be called back after all the batch tasks are completed.
struct BatchCompletion
{
  std::function<void(void*)> batch_completion_cb;
  std::atomic_int done_tasks = 0;
  std::atomic_int batch_total_tasks = 0;

  BatchCompletion(std::function<void(void*)> batch_completion) : batch_completion_cb(batch_completion)
  {
    PUFFER_ASSERT(batch_completion != nullptr, "BatchGroup requires a non-empty callback.");
  }

  explicit BatchCompletion() = delete; // Do not allow passing in an empty callback.
};


//
// Threading support.
//
struct ThreadWork
{
  std::function<void(void*, int)> func;
  void* arg;
  int start_index;
  int end_index;
  // Using a shared_ptr here to avoid locks (so the last thread that goes out of scope automatically releases this).
  // Also prevents alloc'ing completion stuff when there is no need to. Tried using a raw ptr here first, but it's
  // tricky to get right with multi-threading, would have reinvented shared_ptr anyways.
  std::shared_ptr<BatchCompletion> batch_completion;
};

void c_thread_func(void* arg);

struct Threading
{
  std::vector<ThreadWork> work_items;
  std::vector<std::thread> threads;
  std::atomic_int num_threads;
  std::mutex work_mutex;
  std::condition_variable work_cv;
  std::condition_variable done_cv;
  std::atomic_int work_count{0};

  explicit Threading(const int num_threads, const int work_capacity) : num_threads(num_threads)
  {
    work_items.reserve(work_capacity);
    for (int i = 0; i < num_threads; i++)
    {
      threads.emplace_back(std::thread([this] { this->c_thread_func(); }));
    }
  }

  // Wait for signal to do work, do work, signal if there is no more work in the queue.
  inline void c_thread_func()
  {
    int last_count = 0;
    while (true)
    {
      ThreadWork work;
      {
        std::unique_lock lock(work_mutex);
        // This ensures that wait_all_done is guaranteed to not miss a done_cv notification.
        if (last_count == 1)
        {
          done_cv.notify_all();
        }
        while (!(num_threads.load() == 0 || !work_items.empty()))
        {
          work_cv.wait(lock);
        }
        // Shortcuts to exit or try again in case we got woken up but no work.
        if (num_threads.load() == 0)
        {
          break;
        }
        if (work_items.empty())
        {
          continue;
        }
        work = work_items.back();
        work_items.pop_back(); // We have reserved space, so this won't realloc.
        work_count.fetch_add(1);
      }

      for (int i = work.start_index; i <= work.end_index; i++)
      {
        // NOTE: work.func could end up adding more tasks, so we have to notify the producer
        // only within the lock above to prevent race conditions/incomplete done-ness.
        work.func(work.arg, i);
      }
      work.func = nullptr; // Release any captured data.

      check_call_done(work);
      work = {}; // Relinquish any captured closures.
      last_count = work_count.fetch_sub(1);
    }
  }

  inline void check_call_done(ThreadWork& work) const
  {
    auto completion = work.batch_completion;
    if (completion == nullptr) { return; }
    // Must store done locally (this avoids a lock).
    const auto completed_count = work.end_index - work.start_index + 1;
    const auto done = work.batch_completion->done_tasks.fetch_add(completed_count) + completed_count;
    if (done == completion->batch_total_tasks)
    {
      completion->batch_completion_cb(work.arg);
      work.batch_completion = nullptr;
    }
  }


  void wait_all_done()
  {
    std::unique_lock<std::mutex> lock(work_mutex);
    // This ensures that any in-progress work items finish fully before we return.
    while (work_count.load() != 0 || !work_items.empty()) { done_cv.wait(lock); }
  }

  void add_work(const ThreadWork& work)
  {
    if (num_threads.load() == 0)
    {
      PUFFER_ASSERT(false, "Must have created at least one thread to add work to.");
      return;
    }
    {
      std::lock_guard<std::mutex> lock(work_mutex);
      work_items.push_back(work); // We have reserved space, so this won't realloc.
    }
    work_cv.notify_one();
  }

  ~Threading()
  {
    num_threads.store(0);
    work_cv.notify_all();
    wait_all_done();
    for (auto& thread : threads)
    {
      if (thread.joinable())
      {
        thread.join();
      }
    }
    threads.clear();
  }

  void check_empty()
  {
    std::lock_guard<std::mutex> lock(work_mutex);
    PUFFER_ASSERT(work_items.empty() && work_count.load() == 0, "Work queue not empty at start of work.");
  }
};

void c_init_multithreading(VecEnv* vec_env)
{
  PufferOptions* options = &vec_env->opts;
  PUFFER_ASSERT(options != nullptr && options->num_threads > 0 && vec_env->threading == nullptr,
    "Invalid options/thread data.");
  vec_env->threading = new Threading(options->num_threads, vec_env->num_envs);
}

void c_shutdown_multithreading(VecEnv* vec_env)
{
  if (vec_env->threading != nullptr)
  {
    c_wait_all_done(vec_env);
    DELETE_PTR(vec_env->threading);
  }
}

void c_start_work(struct VecEnv* vec_env)
{
  PUFFER_ASSERT(vec_env->threading != nullptr, "Invalid threading state.");
  vec_env->threading->check_empty();
}

// To debug multi-threading issues, uncomment the following line to force single-threaded execution.
// Also helps when profiling via py/libtorch profiler as it shows only the main thread (the other threads are initialized way ahead).
//#define PUFFER_SINGLE_THREADED 1

//! @brief Multi-threading start point: Queues up a batch of work defined by [start_index, end_index].
//! {@ref func} will be called with the provided {@ref arg} and each index in the range.
//! When the entire batch is done, {@ref batch_completion_cb} will be called if provided.
void c_add_work_batched(VecEnv* vec_env, const std::function<void(void*, int)>& func, void* arg, int start_index,
  int end_index, const std::function<void(void*)>& batch_completion_cb)
{
#if defined(PUFFER_SINGLE_THREADED)
  for (int i = start_index; i <= end_index; i++) { func(arg, i); }
  if (batch_completion_cb != nullptr) { batch_completion_cb(arg); }
  return;
#endif

  PUFFER_ASSERT(vec_env->threading != nullptr && end_index >= start_index, "Invalid threading state.");
  const auto num_threads = vec_env->threading->num_threads.load();
  std::shared_ptr<BatchCompletion> batch_completion = {};
  if (batch_completion_cb != nullptr)
  {
    batch_completion = std::make_shared<BatchCompletion>(batch_completion_cb);
    batch_completion->batch_total_tasks.fetch_add(end_index - start_index + 1);
  }
  if (end_index == start_index)
  {
    vec_env->threading->add_work({
      .func = func,
      .arg = arg,
      .start_index = start_index,
      .end_index = end_index,
      .batch_completion = batch_completion
    });
    return;
  }
  const int batch_size = (end_index - start_index + 1 + num_threads) / num_threads;
  for (; start_index < end_index; start_index += batch_size)
  {
    int item_end = start_index + batch_size;
    if (item_end >= end_index)
    {
      item_end = end_index;
    }
    else
    {
      item_end--;
    }
    vec_env->threading->add_work({
      .func = func,
      .arg = arg,
      .start_index = start_index,
      .end_index = item_end,
      .batch_completion = batch_completion
    });
  }
}

// Overload without batch group.
void c_add_work_batched(VecEnv* vec_env, work_func func, void* arg, int start_index, int end_index)
{
  // `func` gets converted to std::function automatically a la `[func](args) { func(args); }`
  c_add_work_batched(vec_env, func, arg, start_index, end_index, nullptr);
}

void c_wait_all_done(VecEnv* vec_env)
{
  PUFFER_ASSERT(vec_env->threading != nullptr, "Invalid threading state.");
  vec_env->threading->wait_all_done();
}

//
// LibTorch core functions.
//

void c_libtorch_info()
{
  std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
  std::cout << "cuDNN available: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;
  if (torch::cuda::is_available())
  {
    std::cout << "Number of CUDA devices: " << torch::cuda::device_count() << std::endl;
  }
  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  Tensor test_tensor = torch::zeros({2, 2}, device);
  std::cout << "Test tensor device: " << test_tensor.device() << std::endl;
}

// Callable from Python to ensure Python<->C++ views are consistent and that no copies are needed.
void c_print_tensor_info(Tensor tensor, string name = "", bool print_values = false)
{
#if DEBUG
  const auto numel = tensor.numel();
  const auto elem_size = tensor.element_size();
  const double total_bytes = double(static_cast<std::uint64_t>(numel) * static_cast<std::uint64_t>(elem_size));
  const double total_mb = total_bytes / (1024.0 * 1024.0);

  std::ostringstream device_ss;
  device_ss << tensor.device();
  std::ostringstream dtype_ss;
  dtype_ss << tensor.dtype();
  std::ostringstream sizes_ss;
  sizes_ss << tensor.sizes();

  std::printf(
    "Tensor: %s  %s / %s / %s / %.3f MB ] [ptr 0x%p]\n", name.c_str(), device_ss.str().c_str(), dtype_ss.str().c_str(),
    sizes_ss.str().c_str(), total_mb, tensor.const_data_ptr());

  if (print_values)
  {
    // VERY Expensive to do this, so strictly for debugging.
    auto t = tensor.cpu();
    if (t.dim() >= 2)
    {
      const int64_t max0 = std::min<int64_t>(4, t.size(0));
      const int64_t max1 = std::min<int64_t>(8, t.size(1));

      t = t.narrow(0, 0, max0).narrow(1, 0, max1);
    }

    std::cout << name << " (slice):\n{" << t.cpu() << "}\n\n";
  }
#endif
}


void c_print_tensor_infos(Tensor tensor1, Tensor tensor2, string name)
{
  c_print_tensor_info(tensor1, "Tensor 1: " + name);
  c_print_tensor_info(tensor2, "Tensor 2: " + name);
}

struct LSTMWrapper;

// Simple performance timer (NOT thread-safe, must ensure it's per-thread or per-batch).
struct PerfTimer
{
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  std::chrono::duration<double, std::milli> duration;
  std::string name;

  void start() { start_time = std::chrono::high_resolution_clock::now(); }

  void stop()
  {
    end_time = std::chrono::high_resolution_clock::now();
    duration += end_time - start_time;
  }
};

// Per-batch env state that has `env_count` envs.
struct PufferEnvState
{
  // Batch index within the envs.
  int batch_index;
  // The envs within this batch.
  int env_start_index;
  int env_count;
#ifdef PUFFER_CUDA
  // Using shared_ptr since there isn't a default constructor; plus avoids having a lock for the stream itself.
  // Stream 1 for copying obs to device and forward eval.
  std::vector<std::shared_ptr<CUDAStream>> cuda_streams;
#endif
  // For the LSTM wrapper.
  Tensor h, c;
  // The following can be released after a segment is processed.
  Tensor obs_cpu, obs_device;
  Tensor rewards_cpu, terminals_cpu;
  Tensor logits_entropy_unused;

  // Stores the intermediate segments across a horizon for copying into the out tensors.
  // One set of threads write to the arr[bptt_segment] while the other thread reads/copies over the tensors.
  Tensor *obs_horizon, *values_horizon, *logprob_horizon, *actions_horizon, *rewards_horizon, *terminals_horizon;
  // Global params for quick referencing.
  LSTMWrapper* lstm_wrapper;
  atomic_int bptt_segment;
  VecEnv* vec_env;
  PerfTimer perf_env_cpu;
  PerfTimer perf_to_device_copy; // Copy obs to GPU.
  PerfTimer perf_lstm_forward;
  PerfTimer perf_post_batch_copy; // Copy all the results back to the passed in Tensors.
};

struct PufferEvalResult
{
  // Perf stats (in ms) across all batches for this run.
  std::vector<std::tuple<std::string, double>> stats_millis;
  int64_t step_count;
  int64_t total_steps;
};

struct LogitsResult
{
  Tensor actions;
  Tensor logprobs;
  Tensor entropy;
};

static inline Tensor log_prob(Tensor logits, Tensor value)
{
  value = value.to(torch::kLong).unsqueeze(-1);
  auto res = torch::broadcast_tensors({value, logits});
  value = res[0];
  value = value.index({at::indexing::Ellipsis, at::indexing::Slice(0, 1)});
  auto log_pmf = res[1];
  log_pmf = log_pmf.gather(-1, value).squeeze(-1);
  res[0] = Tensor{};
  res[1] = Tensor{};
  return log_pmf;
}

//! @brief Returns a tuple of (actions, logprobs, entropy) sampled from the given raw logits.
//! Matches the Python version with optional entropy calculation (entropy might not be needed during eval for instance).
//! TODO(perumaal): Calc entropy and accept input actions during training.
static inline LogitsResult sample_logits(Tensor logits, int num_actions, int64_t* logit_sizes, bool calc_entropy)
{
  PUFFER_ASSERT(logits.dim() == 2, "Logits must be 2D (batch_size, total_num_logits).");
  if (num_actions == 1) { logits = logits.unsqueeze(0); }
  else
  {
    auto split_logits = logits.split(at::IntArrayRef(logit_sizes, num_actions), /*dim=*/1);
    logits = torch::stack(split_logits, /*dim=*/0);
  }
  auto normalized_logits = logits - torch::logsumexp(logits, /*dim=*/-1, /*keepdim=*/true);
  auto probs = torch::exp(torch::log_softmax(logits, -1));

  probs = torch::nan_to_num(probs, 1e-8, 1e-8, 1e-8);
  auto action = torch::multinomial(probs.reshape({-1, probs.size(-1)}), 1, /*replacement=*/ true);
  action = action.to(torch::kInt32);
  action = action.reshape(probs.sizes().slice(0, probs.dim() - 1));
  auto logprob = log_prob(normalized_logits, action);
  if (num_actions == 1)
  {
    action = action.squeeze(0);
    logprob = logprob.squeeze(0);
  }
  else
  {
    logprob = logprob.sum(0);
    action = action.transpose(0, 1);
  }
  return {action, logprob, Tensor{}};
}

struct LSTMWrapper : torch::nn::Module
{
  // Per-eval batch size (# of envs / batch) and count (# of batches).
  int eval_batch_size;
  int eval_batch_count;
  PufferOptions* opt{nullptr};
  int num_envs;

  LSTMWrapper(PufferOptions* opt, int num_envs) : opt(opt), num_envs(num_envs)
  {
#if PUFFER_CUDA
    if (device.type() == torch::kCUDA)
    {
      std::cout << "Using CUDA device for LSTMWrapper.\n";
    }
    torch::manual_seed(42);
    torch::cuda::manual_seed(42);

    // TODO(perumaal): These don't seem to have a big effect on performance, but keep them for now.
    // Enable cuDNN benchmarking
    torch::globalContext().setBenchmarkCuDNN(true);
    torch::globalContext().setDeterministicCuDNN(false);
    torch::globalContext().setBenchmarkLimitCuDNN(32);

    // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090)
    torch::globalContext().setAllowTF32CuBLAS(true);
    torch::globalContext().setAllowTF32CuDNN(true);

    // Enable faster FP16 reductions
    torch::globalContext().setAllowFP16ReductionCuBLAS(true);

    // BF16 reduction (if using bfloat16)
    torch::globalContext().setAllowBF16ReductionCuBLAS(true);

    // Enable memory history recording for detailed snapshots
#if PUFFER_CUDA_MEMCHECK
    CUDACachingAllocator::recordHistory(true, nullptr, 1024 * 1024 * 100, CUDACachingAllocator::RecordContext::NEVER,
      true);
#endif
#endif
    torch::NoGradGuard no_grad;
    device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    encoder_linear = layer_init(torch::nn::Linear(opt->obs_size, opt->hidden_size));
    encoder_gelu = torch::nn::GELU();
    encoder = register_module("encoder", torch::nn::Sequential(encoder_linear, encoder_gelu));
    if (opt->is_continuous)
    {
      decoder_mean =
          register_module("decoder_mean", layer_init(torch::nn::Linear(opt->hidden_size, opt->num_actions), 0.01));
      decoder_logstd = register_parameter("decoder_logstd", torch::zeros({1, opt->num_actions}));
    }
    else
    {
      opt->num_atns = 0;
      for (int i = 0; i < opt->num_actions; i++)
      {
        opt->num_atns += opt->logit_sizes[i];
      }
      decoder = register_module("decoder", layer_init(torch::nn::Linear(opt->hidden_size, opt->num_atns), 0.01));
    }
    value = register_module("value", layer_init(torch::nn::Linear(opt->hidden_size, 1), 1.0));
    lstm_cell = register_module("lstmcell", torch::nn::LSTMCell(opt->input_size, opt->hidden_size));
    int batch_chunk_size = (opt->batch_chunk_size_kb * 1024) / (opt->obs_size * sizeof(float));
    if (batch_chunk_size < 1) { batch_chunk_size = 1; }
    eval_batch_size = batch_chunk_size;
    eval_batch_count = (num_envs + batch_chunk_size - 1) / batch_chunk_size;
  }

  ~LSTMWrapper() override {}


  void info() const
  {
    for (auto& np : this->named_parameters())
    {
      std::cout << np.key() << ": " << np.value().sizes() << std::endl;
    }
  }

  inline void assign_tensors(Tensor& to, Tensor& from, string name)
  {
    // c_print_tensor_infos(to, from, "to (1) <- from (2)");

#if DEBUG
    PUFFER_ASSERT(to.sizes() == to.sizes(), "Tensor size mismatch.");
    PUFFER_ASSERT(to.device() == to.device(), "Tensor device mismatch.");
    PUFFER_ASSERT(to.dim() == to.dim(), "Tensor dims mismatch.");
#endif
    to = from;
  }

  inline void assign_out_tensors(std::vector<Tensor>& segments, Tensor out_tensor, string name)
  {
    // c_print_tensor_infos(out_tensor, segments, "to (1) <- from (2)");
    segments = {};
    segments.reserve(opt->bptt_horizon);
  }

  //! @brief Given the input full (all envs) obs/rewards/terminals tensors on CPU (and referencing the correct data),
  //! this routine will setup the obs/actions/logprobs/rewards/terminals/values output tensors (on device) and
  //! use the input weights and biases as the starting point. Call forward_eval_batch to run the full BPTT horizon
  //! across all segments using multi-threadeded libtorch.
  void start_batch_eval_lstm(VecEnv* vec_env, Tensor full_obs_cpu, Tensor full_rewards_cpu, Tensor full_terminals_cpu,
    Tensor encoder_linear_w, Tensor encoder_linear_b, Tensor decoder_linear_w,
    Tensor decoder_linear_b, Tensor value_w, Tensor value_b, Tensor weight_ih,
    Tensor weight_hh, Tensor bias_ih, Tensor bias_hh, Tensor obs_out, Tensor actions_out,
    Tensor logprobs_out, Tensor rewards_out, Tensor terminals_out, Tensor values_out)
  {
    BEGIN_LIBTORCH_CATCH
    {
      torch::NoGradGuard no_grad;
      env_states = new PufferEnvState*[eval_batch_count];
      for (int i = 0; i < eval_batch_count; i++)
      {
        auto* state = (env_states[i] = new PufferEnvState());
        const int start_idx = i * eval_batch_size;
        int env_count = eval_batch_size;
        if (i == eval_batch_count - 1)
        {
          env_count = num_envs - start_idx;
        }
        state->batch_index = i;
        state->env_start_index = start_idx;
        state->env_count = env_count;
      }

      this->vec_env = vec_env;
      this->horizon_steps = 0;
      assign_tensors(encoder_linear->weight, encoder_linear_w, "encoder_linear_w");
      assign_tensors(encoder_linear->bias, encoder_linear_b, "encoder_linear_b");
      assign_tensors(decoder->weight, decoder_linear_w, "decoder_linear_w");
      assign_tensors(decoder->bias, decoder_linear_b, "decoder_linear_b");
      assign_tensors(value->weight, value_w, "value_w");
      assign_tensors(value->bias, value_b, "value_b");
      assign_tensors(lstm_cell->weight_ih, weight_ih, "weight_ih");
      assign_tensors(lstm_cell->weight_hh, weight_hh, "weight_hh");
      assign_tensors(lstm_cell->bias_ih, bias_ih, "biash_ih");
      assign_tensors(lstm_cell->bias_hh, bias_hh, "biash_hh");
      PUFFER_ASSERT(obs_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon, opt->obs_size}),
        "Obs tensor size mismatch.");
      if (opt->num_actions == 1)
      {
        PUFFER_ASSERT(actions_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}),
          "Actions (discrete) tensor size mismatch.");
      }
      else
      {
        PUFFER_ASSERT(actions_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon, opt->num_actions}),
          "Actions (multidiscrete) tensor size mismatch.");
      }
      PUFFER_ASSERT(logprobs_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}),
        "logprobs tensor size mismatch.");
      PUFFER_ASSERT(rewards_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}),
        "rewards tensor size mismatch.");
      PUFFER_ASSERT(terminals_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}),
        "terminals tensor size mismatch.");
      PUFFER_ASSERT(values_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}),
        "values tensor size mismatch.");
      final_obs = obs_out;
      final_actions = actions_out;
      final_logprobs = logprobs_out;
      final_rewards = rewards_out;
      final_terminals = terminals_out;
      final_values = values_out;
      // c_print_tensor_infos(final_obs, final_actions, "final tensor obs/actions");
      // c_print_tensor_infos(final_logprobs, final_rewards, "final tensors logprobs/rewards");
      // c_print_tensor_infos(final_terminals, final_values, "final tensors terminals/values");


      for (int i = 0; i < eval_batch_count; i++)
      {
        auto* state = env_states[i];
        state->bptt_segment = 0;


        // Per-batch/per-bptt-segment slices.
        state->obs_device = Tensor{};
        state->obs_cpu = full_obs_cpu.narrow(0, state->env_start_index, state->env_count);
        state->rewards_cpu = full_rewards_cpu.narrow(0, state->env_start_index, state->env_count);
        state->terminals_cpu = full_terminals_cpu.narrow(0, state->env_start_index, state->env_count);
        alloc_tensor_arr(&state->obs_horizon);
        alloc_tensor_arr(&state->values_horizon);
        alloc_tensor_arr(&state->logprob_horizon);
        alloc_tensor_arr(&state->rewards_horizon);
        alloc_tensor_arr(&state->actions_horizon);
        alloc_tensor_arr(&state->terminals_horizon);

        // H/C state is tracked per batch across segments for the current horizon.
        state->h = torch::zeros({state->env_count, opt->hidden_size}, device);
        state->c = torch::zeros({state->env_count, opt->hidden_size}, device);
        state->logits_entropy_unused = Tensor{};
        state->lstm_wrapper = this;
        state->vec_env = vec_env;

        state->perf_env_cpu = PerfTimer{.name = "env_cpu"};
        state->perf_to_device_copy = PerfTimer{.name = "to_device_copy"};
        state->perf_lstm_forward = PerfTimer{.name = "lstm_forward"};
        state->perf_post_batch_copy = PerfTimer{.name = "post_batch_copy"};
#ifdef PUFFER_CUDA
        if (device.type() == torch::kCUDA)
        {
          state->cuda_streams = {};
          for (int j = 0; j < opt->bptt_horizon; j++) { state->cuda_streams.push_back({}); }
        }
#endif
      }
      perf_total_forward_eval = {.name = "total_forward_eval"};
    }
    END_LIBTORCH_CATCH
  }


  // Batched env forward eval. This starts the process per segment in the horizon. Waits for all segments to finish and
  // then return the batched tensor set back.
  void forward_eval_batch(VecEnv* vec_env)
  {
    BEGIN_LIBTORCH_CATCH
    {
      torch::NoGradGuard no_grad;
      // This is effectively useless as all the work is done in other threads, but keep it for safety.
      perf_total_forward_eval.start();

      c_start_work(vec_env);
      // Kick off this batch of work.
      // TODO: Should we do each batch-segment part of this horizon independently? or all at once?
      // We can start off with putting this whole thing in a for loop (i.e. each iteration, wait for all done) to begin
      // with. I think ideally, some stuff should just start going forward.
      c_add_work_batched(vec_env, run_next_bptt_segment, this, 0, eval_batch_count - 1);
      // full_obs is [num_envs, obs_size] in CPU side.
      // Transfer each obs batch to device independently.
      // Add batch work: torch_batch_eval(this, index)
      // Get the action[]/etc tensors from each batch.
      // Enqueue the env steps.
      // cat all tensors and return.
      c_wait_all_done(vec_env);
      perf_total_forward_eval.stop();
    } END_LIBTORCH_CATCH
  }

//! @brief Returns all the tensors (on target device) plus stats across all batches.
  PufferEvalResult finish_batch_eval_lstm(VecEnv* env)
  {
    PufferEvalResult result;

    BEGIN_LIBTORCH_CATCH
    {
      RECORD_FUNCTION("finish_batch_eval_cpp", std::vector<c10::IValue>({}));

      for (int i = 0; i < eval_batch_count; i++)
      {
        auto* state = env_states[i];
        for (int seg = 0; seg < opt->bptt_horizon; seg++)
        {
          state->obs_horizon[seg] = Tensor{};
          state->values_horizon[seg] = Tensor{};
          state->logprob_horizon[seg] = Tensor{};
          state->rewards_horizon[seg] = Tensor{};
          state->terminals_horizon[seg] = Tensor{};
          state->actions_horizon[seg] = Tensor{};
        }
        calc_total_perf_duration(result, state->perf_env_cpu);
        calc_total_perf_duration(result, state->perf_to_device_copy);
        calc_total_perf_duration(result, state->perf_lstm_forward);
        calc_total_perf_duration(result, state->perf_post_batch_copy);
#if PUFFER_CUDA
        for (auto& stream : state->cuda_streams)
        {
          if (stream != nullptr) { stream->synchronize(); }
          stream = nullptr;
        }
        state->cuda_streams = {};
#endif

        state->obs_cpu = Tensor{};
        state->obs_device = Tensor{};
        state->rewards_cpu = Tensor{};
        state->terminals_cpu = Tensor{};
        state->logits_entropy_unused = Tensor{};
        state->h = Tensor{};
        state->c = Tensor{};

        DELETE_ARRAY(state->obs_horizon);
        DELETE_ARRAY(state->values_horizon);
        DELETE_ARRAY(state->logprob_horizon);
        DELETE_ARRAY(state->actions_horizon);
        DELETE_ARRAY(state->rewards_horizon);
        DELETE_ARRAY(state->terminals_horizon);

        // Prepare for next run.
        state->lstm_wrapper = nullptr;
      }
      result.stats_millis.push_back({perf_total_forward_eval.name, perf_total_forward_eval.duration.count()});
      result.step_count = this->horizon_steps;
      result.total_steps = this->total_steps;
      vec_env = nullptr;
      final_obs = Tensor{};
      final_actions = Tensor{};
      final_logprobs = Tensor{};
      final_rewards = Tensor{};
      final_terminals = Tensor{};
      final_values = Tensor{};
#if PUFFER_CUDA
      if (device.type() == torch::kCUDA)
      {
        // Make sure all queued work across streams is complete before attempting to release cached blocks.
        c10::cuda::CUDAGuard device_guard(device);
        c10::cuda::device_synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
      }
#endif
      for (int i = 0; i < eval_batch_count; i++)
      {
        DELETE_PTR(env_states[i]);
      }
      DELETE_ARRAY(env_states);
      env_states = nullptr;
    }
    END_LIBTORCH_CATCH
    return result;
  }

private:
  void alloc_tensor_arr(Tensor** arr) const
  {
    *arr = new Tensor[opt->bptt_horizon];
    for (int i = 0; i < opt->bptt_horizon; i++) { (*arr)[i] = Tensor{}; }
  }

  void calc_total_perf_duration(PufferEvalResult& result, PerfTimer& timer)
  {
    auto duration_ms = timer.duration.count();
    auto name = timer.name;
    for (auto& stat : result.stats_millis)
    {
      if (std::get<0>(stat) == name)
      {
        std::get<1>(stat) += duration_ms;
        return;
      }
    }
    result.stats_millis.push_back({name, duration_ms});
  }

  [[nodiscard]] torch::nn::Linear layer_init(torch::nn::Linear layer, const double std = std::sqrt(2.0),
    const double bias_const = 0.0) const
  {
    torch::nn::init::orthogonal_(layer->weight, std);
    torch::nn::init::constant_(layer->bias, bias_const);
    return layer;
  }

  static void run_next_bptt_segment(void* arg, int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      auto* this_ptr = static_cast<LSTMWrapper*>(arg);
      // We must do this per thread work as it's TLS guarded.
      torch::NoGradGuard no_grad;
      auto* state = this_ptr->env_states[batch_index];
      auto segment = state->bptt_segment.load();
      print_cuda_mem_info("bptt_segment_S" + std::to_string(segment) + "_B" + std::to_string(batch_index), true);
      if (segment >= this_ptr->opt->bptt_horizon) { return; }
      // printf(" Batch %d: Running BPTT segment %d / %d\n", batch_index, state->bptt_segment, opt->bptt_horizon);
      // Ok to perform synchronously as we need the obs tensor + forward eval before we can start env steps.
#ifdef PUFFER_CUDA
      if (this_ptr->device == torch::kCUDA)
      {
        state->cuda_streams[segment] = std::make_shared<CUDAStream>(getStreamFromPool(/*isHighPriority=*/true));
        CUDAStreamGuard guard(*state->cuda_streams[segment]);
        this_ptr->copy_obs_forward_eval_batch(batch_index);
      }
      else // fallthrough
#endif
      {
        this_ptr->copy_obs_forward_eval_batch(batch_index);
      }
    }
    END_LIBTORCH_CATCH
  }

  //! @brief Async multi-threaded copy + forward eval pass for an entire batch of obs (with a separate stream if needed).
  //! This can/should overlap with the next segment's copy+forward eval.
  void copy_to_final_buffers_async(PufferEnvState* state, int segment)
  {
    BEGIN_LIBTORCH_CATCH
    {
      torch::NoGradGuard no_grad;
      state->perf_post_batch_copy.start();

#ifdef PUFFER_CUDA
      if (device == torch::kCUDA)
      {
        CUDAStreamGuard guard(*state->cuda_streams[segment]);
        copy_to_final_buffers(state, segment);
      }
      else // fallthrough
#endif
      {
        copy_to_final_buffers(state, segment);
      }
      state->perf_post_batch_copy.stop();
    }
    END_LIBTORCH_CATCH
  }

  //! @brief Async multi-threaded copy + forward eval pass for an entire batch of obs.
  //! Assumed that run_next_bptt_segment sets the right CUDA stream before calling this function.
  void copy_to_final_buffers(PufferEnvState* state, const int segment)
  {
    BEGIN_LIBTORCH_CATCH
    {
      RECORD_FUNCTION("final_copy_buffers",
        std::vector<c10::IValue>({static_cast<uint64_t>(state->batch_index), static_cast<uint64_t>(segment)}));
      // This entire copy can proceed lock-free because the other thread produces a work in a new index we
      // possibly couldn't see (i.e. guarded by the atomic segment). And this function is the sole
      // owner of segment_start, so there's no race / conflicts here to necessitate a lock.
      const int64_t env_start = state->env_start_index;
      const int64_t n = state->env_count;
      auto non_blocking = true;
      // Do copies first, but only clear horizon tensors until after the stream finishes.
      // Obs already copied during forward eval as we need it the first thing.
      // final_obs.narrow(0, env_start, n).select(1, segment).copy_(state->obs_horizon[segment], /*non_blocking=*/
      //   non_blocking);
      final_values.narrow(0, env_start, n).select(1, segment).copy_(state->values_horizon[segment], non_blocking);
      final_logprobs.narrow(0, env_start, n).select(1, segment).copy_(state->logprob_horizon[segment], non_blocking);
      final_rewards.narrow(0, env_start, n).select(1, segment).copy_(state->rewards_horizon[segment], non_blocking);
      final_terminals.narrow(0, env_start, n).select(1, segment).copy_(state->terminals_horizon[segment], non_blocking);
      final_actions.narrow(0, env_start, n).select(1, segment).copy_(state->actions_horizon[segment], non_blocking);
      state->obs_horizon[segment] = Tensor{};
      state->values_horizon[segment] = Tensor{};
      state->logprob_horizon[segment] = Tensor{};
      state->rewards_horizon[segment] = Tensor{};
      state->terminals_horizon[segment] = Tensor{};
      state->actions_horizon[segment] = Tensor{};
    }
    END_LIBTORCH_CATCH
  }

  //! @brief Async multi-threaded copy + forward eval pass for an entire batch of obs.
  //! Assumed that run_next_bptt_segment sets the right CUDA stream before calling this function.
  void copy_obs_forward_eval_batch(int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      // We must do this per thread work as it's TLS guarded.
      torch::NoGradGuard no_grad;
      auto* state = env_states[batch_index];
      {
        RECORD_FUNCTION("batch_copy_to_device", std::vector<c10::IValue>({static_cast<uint64_t>(batch_index)}));
        state->perf_to_device_copy.start();
        const auto segment = state->bptt_segment.load();
        // printf("batch obs copy: B %d S %d \n", batch_index, state->bptt_segment.load());
        print_cuda_mem_info("copy_obs_pre_S" + std::to_string(segment) + "_B" + std::to_string(batch_index), true);

        // NOTE: Env observations are memory mapped to the full_obs_cpu tensor already.
        // Once it's on device, changes are no longer reflected unless we copy again.
        const int64_t env_start = state->env_start_index;
        const int64_t n = state->env_count;
        state->obs_device = final_obs.narrow(0, env_start, n).select(1, segment);
        state->obs_device.copy_(state->obs_cpu, false);
        // Must copy blocking as the obs will be overwritten by the envs next.
        state->obs_horizon[segment] = state->obs_device;
        state->perf_to_device_copy.stop();
      }
      torch_batch_forward_eval(batch_index);
    }
    END_LIBTORCH_CATCH
  }


  //! @brief Async multi-threaded forward eval pass for an entire batch of obs.
  void torch_batch_forward_eval(int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      RECORD_FUNCTION("batch_forward_eval", std::vector<c10::IValue>({static_cast<uint64_t>(batch_index)}));

      // We must do this per thread work as it's TLS guarded.
      torch::NoGradGuard no_grad;
      auto* state = env_states[batch_index];
      auto segment = state->bptt_segment.load();

      state->perf_lstm_forward.start();
      auto obs_tensor = state->obs_device;
      state->obs_device = Tensor{};
      state->obs_horizon[segment] = Tensor{};

      Tensor hidden = encoder->forward(obs_tensor);
      obs_tensor = Tensor{};
      auto hc = lstm_cell->forward(hidden, std::make_tuple(state->h, state->c));
      hidden = Tensor{};
      state->h = std::get<0>(hc);
      state->c = std::get<1>(hc);
      void* actions_data = nullptr;
      if (opt->is_continuous)
      {
        PUFFER_ASSERT(!opt->is_continuous, "Only supports (multi)discrete for now.");
        throw std::runtime_error("Continuous action space not implemented yet.");
        // TODO(perumaal): Need to update state->logits as well and verify this with the puffernet impl.
      }
      else
      {
        // TODO: Parallelize these two forwards? Probably not worth it as these are just linear layers.
        auto logits = decoder->forward(state->h);
        auto values = value->forward(state->h);
        values = values.flatten();

        state->values_horizon[segment] = values;

        auto [actions_batch, logprobs, entropy_unused] =
            sample_logits(logits, opt->num_actions, opt->logit_sizes, /*calc_entropy=*/false);

        state->logprob_horizon[segment] = logprobs;
        logprobs = Tensor{};
        entropy_unused = Tensor{};

        state->actions_horizon[segment] = actions_batch;
        const auto actions_int = actions_batch.to(
          torch::kCPU,
          /*non_blocking=*/false,
          /*copy=*/true,
          {c10::MemoryFormat::Contiguous});
        actions_batch = Tensor{};

        actions_data = actions_int.data_ptr<int>();
      }
      state->perf_lstm_forward.stop();

      state->perf_env_cpu.start();
      // Run the batch's env steps independently in different threads.
      // Once all envs from this batch have completed, continue on to run the next BPTT segment.
      auto num_actions = opt->num_actions;
      auto* rewards_arr = static_cast<float*>(state->rewards_cpu.data_ptr());
      auto* terminals_arr = static_cast<float*>(state->terminals_cpu.data_ptr());
      c_add_work_batched(vec_env,
        [actions_data, num_actions, rewards_arr, terminals_arr](void* arg, int index)
        {
          c_step_batch(arg, index, actions_data, num_actions, rewards_arr, terminals_arr);
        }, state->vec_env->envs, state->env_start_index,
        state->env_start_index + state->env_count - 1,
        [state](void* _) // Unused as it's per-env, we need the batch captured state.
        {
          BEGIN_LIBTORCH_CATCH
          {
            torch::NoGradGuard no_grad;
            RECORD_FUNCTION("finalize_bptt_segment",
              std::vector<c10::IValue>({static_cast<uint64_t>(state->batch_index)}));
            auto segment = state->bptt_segment.load();
            state->lstm_wrapper->total_steps += state->env_count;
            state->lstm_wrapper->horizon_steps += state->env_count;
            state->perf_env_cpu.stop();
            state->rewards_horizon[segment] = (state->rewards_cpu);
            state->terminals_horizon[segment] = (state->terminals_cpu);
          }
          END_LIBTORCH_CATCH

          // Schedule this work for the next segment. (We could reuse this thread, but let's let the OS
          // manage the priorities and let the cascade happen naturally).
          auto segment = atomic_fetch_add(&state->bptt_segment, 1);

          // Queue up two work items:
          // 1) Copy to final buffers (async) for the previous segment.
          // 2) Run next BPTT segment forward eval for the next segment.
          c_add_work_batched(state->vec_env,
            [segment](void* arg, int _2)
            {
              auto* state = static_cast<PufferEnvState*>(arg);
              state->lstm_wrapper->copy_to_final_buffers_async(state, segment);
            },
            state, segment, segment, nullptr);

          c_add_work_batched(state->vec_env, run_next_bptt_segment, state->lstm_wrapper,
            state->batch_index, state->batch_index);
        });
    }
    END_LIBTORCH_CATCH
  }

  int64_t total_steps = 0;
  int64_t horizon_steps = 0;
  // All of these are thread-safe within a single eval call (except for update_model_weights).
  // Inference only for now (i.e. evaluate()).
  torch::nn::Sequential encoder{nullptr};
  torch::nn::Linear encoder_linear{nullptr};
  torch::nn::GELU encoder_gelu{nullptr};
  torch::nn::Linear decoder{nullptr};
  torch::nn::Linear value{nullptr};
  // Continuous action space:
  // TODO(perumaal): Implement continuous action space support - currently partial impl.
  torch::nn::Linear decoder_mean{nullptr};
  at::Tensor decoder_logstd{nullptr};

  // LSTM Policy on top of the encoder/decoder above.
  torch::nn::LSTMCell lstm_cell{nullptr};
  torch::Device device = torch::kCPU;

  // These may be accessed from any thread during eval.
  PufferEnvState** env_states;
  VecEnv* vec_env;
  Tensor final_obs, final_actions, final_logprobs, final_rewards, final_terminals, final_values;
  PerfTimer perf_total_forward_eval;
};

struct PufferTorch
{
  // Could hold other models too, but for now, just one.
  LSTMWrapper* model;
};

void c_setup_pufferoptions(VecEnv* vec_env, const int num_actions, const int num_logits, const int input_size,
  const int hidden_size, const bool is_continuous, const int batch_chunk_size_kb)
{
  PufferOptions* options = &vec_env->opts;
  options->num_actions = num_actions;
  options->num_logits = num_logits;
  options->logit_sizes = new int64_t[num_actions];
  options->batch_chunk_size_kb = batch_chunk_size_kb;
  for (int i = 0; i < num_actions; i++)
  {
    options->logit_sizes[i] = num_logits;
  }
  options->input_size = input_size;
  options->hidden_size = hidden_size;
  options->is_continuous = is_continuous;
}

void c_cleanup_pufferoptions(VecEnv* vec_env)
{
  if (vec_env->opts.logit_sizes)
  {
    DELETE_ARRAY(vec_env->opts.logit_sizes);
  }
  vec_env->opts = {};
}


PufferTorch* c_torch_alloc(VecEnv* vec_env)
{
  BEGIN_LIBTORCH_CATCH
  {
    PufferOptions* opts = &vec_env->opts;
    PUFFER_ASSERT(opts != nullptr && opts->num_actions > 0 && opts->num_atns == 0 && opts->logit_sizes != nullptr &&
      opts->enable_native_libtorch,
      "Invalid options.");
    auto* ptorch = new PufferTorch();
    opts->batch_chunk_size_kb = std::max(1, opts->batch_chunk_size_kb);
    ptorch->model = new LSTMWrapper(opts, vec_env->num_envs);
    vec_env->puff_torch = ptorch;


    printf(
      "Native multithreading/libtorch: %d envs on %d threads (batch size = max %d envs/batch; total %d batches)%s%s.\n",
      vec_env->num_envs, opts->num_threads, ptorch->model->eval_batch_size, ptorch->model->eval_batch_count,
      (debug_mode ? " [Debug Mode]" : " [Release Mode]"),
      (cuda_async ? " [CUDA multi-threaded streams ON]" : " [CUDA multi-threaded streams OFF]"));

    return ptorch;
  }
  END_LIBTORCH_CATCH
}

void c_torch_free(PufferTorch* pt)
{
  BEGIN_LIBTORCH_CATCH
  {
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr, "Invalid state.");
    DELETE_PTR(pt->model);
    delete pt;
  }
  END_LIBTORCH_CATCH
}

void c_torch_start_eval_lstm(uintptr_t vec_env_ptr, Tensor full_obs_cpu, Tensor full_rewards_cpu,
  Tensor full_terminals_cpu, Tensor encoder_linear_w, Tensor encoder_linear_b,
  Tensor decoder_linear_w, Tensor decoder_linear_b, Tensor value_w, Tensor value_b,
  Tensor weight_ih, Tensor weight_hh, Tensor bias_ih, Tensor bias_hh, Tensor obs_out,
  Tensor actions_out, Tensor logprobs_out, Tensor rewards_out, Tensor terminals_out,
  Tensor values_out)
{
  VecEnv* vec_env = (VecEnv*)vec_env_ptr;
  PufferTorch* puff_torch = vec_env->puff_torch;
  PUFFER_ASSERT(puff_torch != nullptr && puff_torch->model != nullptr, "Invalid state.");

  puff_torch->model->start_batch_eval_lstm(vec_env, full_obs_cpu, full_rewards_cpu, full_terminals_cpu,
    encoder_linear_w, encoder_linear_b, decoder_linear_w, decoder_linear_b,
    value_w, value_b, weight_ih, weight_hh, bias_ih, bias_hh, obs_out,
    actions_out, logprobs_out, rewards_out, terminals_out, values_out);
}

//! @brief Performs action (inference) + step segmented across a BPTT horizon batched by envs.
//! Waits for the entire run to finish. TODO: Clarify - full bptt horizon ? or a single segment? TODO: log timing perf
//! metrics
void c_torch_run_fulleval(uintptr_t vec_env_ptr)
{
  BEGIN_LIBTORCH_CATCH
  {
    RECORD_FUNCTION("torch_run_fulleval_cpp", std::vector<c10::IValue>({}));

    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferTorch* pt = vec_env->puff_torch;
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr && vec_env->num_envs > 0 && vec_env->envs != nullptr &&
      vec_env->threading != nullptr,
      "Invalid state/inputs.");
    pt->model->forward_eval_batch(vec_env);
  }
  END_LIBTORCH_CATCH
}

PufferEvalResult c_torch_finish_eval_lstm(uintptr_t vec_env_ptr)
{
  BEGIN_LIBTORCH_CATCH
  {
    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferTorch* pt = vec_env->puff_torch;
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr, "Invalid state.");
    return pt->model->finish_batch_eval_lstm(vec_env);
  }
  END_LIBTORCH_CATCH
}


// Include the pybind layer if needed. Tests and other units can use this file without pulling in Pythin/pybind stuff.
#ifdef PUFFER_NATIVECPP_PYBINDINGS
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Suggested by Claude to avoid pybind/C++ using import_array/numpy here while env_binding uses just the PyAPI alone
// (using non pybind).
#define PY_ARRAY_UNIQUE_SYMBOL puffer_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// Forward declaration for env_glue.h stuff to avoid circular references. Especially as binding.c (C only)
// includes C code that wraps C++ code/objects underneath.
extern "C" PyMethodDef* get_c_env_binding_methods();

PYBIND11_MODULE(binding, m)
{
  m.doc() = "PufferLib Libtorch API";

  py::class_<PufferEvalResult>(m, "PufferEvalResult")
      .def(py::init<>())
      .def_readwrite("stats_millis", &PufferEvalResult::stats_millis)
      .def_readwrite("step_count", &PufferEvalResult::step_count)
      .def_readwrite("total_steps", &PufferEvalResult::total_steps);

  import_array();
  PyModule_AddFunctions(m.ptr(), get_c_env_binding_methods());
  m.def("libtorch_info", &c_libtorch_info, "Print libtorch info to stdout.");
  m.def("torch_start_eval_lstm", &c_torch_start_eval_lstm, py::arg("vec_env"), py::arg("full_obs_cpu"),
    // Full observation tensor on CPU across all horizons/envs with shape [envs, horizon, obs_count].
    py::arg("full_rewards_cpu"),   // Full rewards tensor on CPU across all horizons/envs [envs, horizon, 1].
    py::arg("full_terminals_cpu"), // Full terminals tensor on CPU across all horizons/envs [envs, horizon, 1].
    py::arg("encoder_linear_w"), py::arg("encoder_linear_b"), py::arg("decoder_linear_w"),
    py::arg("decoder_linear_b"), py::arg("value_w"), py::arg("value_b"), py::arg("weight_ih"), py::arg("weight_hh"),
    py::arg("bias_ih"), py::arg("bias_hh"), py::arg("observations_out"), py::arg("actions_out"),
    py::arg("logprobs_out"), py::arg("rewards_out"), py::arg("terminals_out"), py::arg("values_out"),
    "Start the initial torch eval (before starting the horizon segments).");

  m.def("torch_run_fulleval", &c_torch_run_fulleval, py::arg("vec_env"),
    "Runs the full forward eval pass using libtorch for all segments in the horizon.");

  m.def("torch_finish_eval_lstm", &c_torch_finish_eval_lstm, py::arg("vec_env"),
    "Finish the torch eval (after all segments in the horizon are done).");
}

#endif
