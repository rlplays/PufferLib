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

#ifdef PUFFER_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
using ::c10::cuda::CUDAStream;
using ::c10::cuda::CUDAStreamGuard;
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


// APIs to separate env_glue/env_binding stuff from libtorch cleanly.
PUFFER_EXTERN float* get_obs_ptr(Env* env);
PUFFER_EXTERN int* get_actions_ptr(Env* env);
PUFFER_EXTERN float* get_rewards_ptr(Env* env);
PUFFER_EXTERN unsigned char* get_terminals_ptr(Env* env);
PUFFER_EXTERN void c_step_batch(void* arg, int index);


// Optional batch group that takes a completion function and tracks pending tasks.
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

void c_add_work_batched(VecEnv* vec_env, work_func func, void* arg, int start_index, int end_index,
  std::function<void(void*)> batch_completion_cb);

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
  std::cout << "Tensor: " << name << "  " << tensor.device() << " / " << tensor.dtype() << " / " << tensor.sizes()
      << " ]" << std::endl;
  if (print_values && tensor.device().is_cpu())
  {
    std::cout << name << ": " << tensor << std::endl;
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
  std::shared_ptr<CUDAStream> cuda_stream_1;
  // Stream 2 for copying results entire BPTT horizon segments for this batch back to host/device buffers for training.
  std::shared_ptr<CUDAStream> cuda_stream_2;
#endif
  // For the LSTM wrapper.
  Tensor obs_cpu, obs_device, h, c;
  Tensor rewards_cpu, terminals_cpu;
  Tensor logits_entropy_unused;

  // Stores the intermediate segments across an horizon for copying into the out tensors.
  // One set of threads write to the arr[bptt_segment] while the other thread reads/copies over the tensors.
  Tensor *obs_horizon, *values_horizon, *logprob_horizon, *actions_horizon, *rewards_horizon, *terminals_horizon;
  // Global params for quick referencing.
  LSTMWrapper* lstm_wrapper;
  // BPTT segment_end-1 denotes the last segment that was added to the horizons.
  // BPTT segment_start denotes the first segment that is yet to be copied over to the out tensors.
  // [start, end) will be copied over to the out tensors.
  atomic_int bptt_segment_start;
  atomic_int bptt_segment_end;
  VecEnv* vec_env;
  PerfTimer perf_env_cpu;
  PerfTimer perf_to_device_copy; // Copy obs to GPU.
  PerfTimer perf_lstm_forward;
  PerfTimer perf_post_batch_copy; // Copy all the results back to the passed in Tensors.
  PerfTimer perf_lstm_forward_1;
  PerfTimer perf_lstm_forward_2;
  PerfTimer perf_lstm_forward_3;
  PerfTimer perf_lstm_forward_4;
  PerfTimer perf_lstm_forward_5;
  PerfTimer perf_lstm_forward_6;
  PerfTimer perf_lstm_forward_7;
  PerfTimer perf_lstm_forward_8;
  PerfTimer perf_lstm_forward_9;
  PerfTimer perf_lstm_forward_10;
  PerfTimer perf_lstm_forward_11;
  PerfTimer perf_lstm_forward_12;
  PerfTimer perf_lstm_forward_13;
  PerfTimer perf_lstm_forward_14;
  PerfTimer perf_lstm_forward_15;
};

struct PufferEvalResult
{
  // Shape of each tensor (on device): [bptt_segment, env_count, #]
  Tensor obs;
  Tensor values;
  Tensor logits;
  Tensor logprob;
  Tensor entropy;
  Tensor actions;
  Tensor rewards;
  Tensor terminals;
  // Perf stats (in ms) across all batches for this run.
  std::vector<std::tuple<std::string, double>> stats_millis;
};


struct LSTMWrapper : torch::nn::Module
{
  // Per-eval batch size (# of envs / batch) and count (# of batches).
  int eval_batch_size;
  int eval_batch_count;

  LSTMWrapper(PufferOptions* opt, int num_envs) : opt(opt)
  {
#if PUFFER_CUDA
    if (device.type() == torch::kCUDA)
    {
      std::cout << "Using CUDA device for LSTMWrapper.\n";
    }
    torch::manual_seed(42);
    torch::cuda::manual_seed(42);

    //// Enable cuDNN benchmarking
    // torch::globalContext().setBenchmarkCuDNN(true);
    // torch::globalContext().setDeterministicCuDNN(false);
    // torch::globalContext().setBenchmarkLimitCuDNN(32);

    //// Enable TF32 for faster FP32 math (uses Tensor Cores on 4090)
    // torch::globalContext().setAllowTF32CuBLAS(true);
    // torch::globalContext().setAllowTF32CuDNN(true);

    //// Enable faster FP16 reductions
    // torch::globalContext().setAllowFP16ReductionCuBLAS(true);

    //// BF16 reduction (if using bfloat16)
    // torch::globalContext().setAllowBF16ReductionCuBLAS(true);
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
    if (batch_chunk_size < 1)
    {
      batch_chunk_size = 1;
    }
    eval_batch_size = batch_chunk_size;
    eval_batch_count = (num_envs + batch_chunk_size - 1) / batch_chunk_size;
    // TODO(perumaal): Ensure at most 32 batches per device (to limit CUDA streams; see
    // https://docs.pytorch.org/cppdocs/api/program_listing_file_c10_cuda_CUDAStream.h.html ).
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
      state->h = torch::zeros({env_count, opt->hidden_size}, device);
      state->c = torch::zeros({env_count, opt->hidden_size}, device);
      state->batch_index = i;
      state->env_start_index = start_idx;
      state->env_count = env_count;
#ifdef PUFFER_CUDA
      if (device.type() == torch::kCUDA)
      {
        // Use high-priority stream for the main LSTM forward pass including copying obs to device (these ops are
        // blocking per-batch).
        state->cuda_stream_1 = std::make_shared<CUDAStream>(at::cuda::getStreamFromPool(/* highPriority*/ true));
        // At most 33 concurrent low-priority streams possible, so we use the high-priority pool for the first (main)
        // stream. It's okay if we run out of low-pri streams (& hence copy is a bit slower) because it can overlap with
        // the next segment's forward eval.
        state->cuda_stream_2 = std::make_shared<CUDAStream>(at::cuda::getStreamFromPool(/* highPriority*/ false));
      }
#endif
    }
  }

  ~LSTMWrapper() override
  {
    for (int i = 0; i < eval_batch_count; i++)
    {
      DELETE_PTR(env_states[i]);
    }
    DELETE_ARRAY(env_states);
  }


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
      this->vec_env = vec_env;
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
      for (int i = 0; i < eval_batch_count; i++)
      {
        auto* state = env_states[i];
        state->bptt_segment_start = 0;
        state->bptt_segment_end = 0;
        // Per-batch/per-bptt-segment slices.
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
        state->h = state->h.zero_();
        state->c = state->c.zero_();
        state->logits_entropy_unused = Tensor{};
        state->lstm_wrapper = this;
        state->vec_env = vec_env;

        state->perf_env_cpu = PerfTimer{.name = "env_cpu"};
        state->perf_to_device_copy = PerfTimer{.name = "to_device_copy"};
        state->perf_lstm_forward = PerfTimer{.name = "lstm_forward"};
        state->perf_lstm_forward_1 = PerfTimer{.name = "lstm_forward1"};
        state->perf_lstm_forward_2 = PerfTimer{.name = "lstm_forward2"};
        state->perf_lstm_forward_3 = PerfTimer{.name = "lstm_forward3"};
        state->perf_lstm_forward_4 = PerfTimer{.name = "lstm_forward4"};
        state->perf_lstm_forward_5 = PerfTimer{.name = "lstm_forward5"};
        state->perf_lstm_forward_6 = PerfTimer{.name = "lstm_forward6"};
        state->perf_lstm_forward_7 = PerfTimer{.name = "lstm_forward7"};
        state->perf_lstm_forward_8 = PerfTimer{.name = "lstm_forward8"};
        state->perf_lstm_forward_9 = PerfTimer{.name = "lstm_forward9"};
        state->perf_lstm_forward_10 = PerfTimer{.name = "lstm_forward10"};
        state->perf_lstm_forward_11 = PerfTimer{.name = "lstm_forward11"};
        state->perf_lstm_forward_12 = PerfTimer{.name = "lstm_forward12"};
        state->perf_lstm_forward_13 = PerfTimer{.name = "lstm_forward13"};
        state->perf_lstm_forward_14 = PerfTimer{.name = "lstm_forward14"};
        state->perf_lstm_forward_15 = PerfTimer{.name = "lstm_forward15"};
        state->perf_post_batch_copy = PerfTimer{.name = "post_batch_copy"};
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
        calc_total_perf_duration(result, state->perf_env_cpu);
        calc_total_perf_duration(result, state->perf_to_device_copy);
        calc_total_perf_duration(result, state->perf_lstm_forward);
        calc_total_perf_duration(result, state->perf_lstm_forward_1);
        calc_total_perf_duration(result, state->perf_lstm_forward_2);
        calc_total_perf_duration(result, state->perf_lstm_forward_3);
        calc_total_perf_duration(result, state->perf_lstm_forward_4);
        calc_total_perf_duration(result, state->perf_lstm_forward_5);
        calc_total_perf_duration(result, state->perf_lstm_forward_6);
        calc_total_perf_duration(result, state->perf_lstm_forward_7);
        calc_total_perf_duration(result, state->perf_lstm_forward_8);
        calc_total_perf_duration(result, state->perf_lstm_forward_9);
        calc_total_perf_duration(result, state->perf_lstm_forward_10);
        calc_total_perf_duration(result, state->perf_lstm_forward_11);
        calc_total_perf_duration(result, state->perf_lstm_forward_12);
        calc_total_perf_duration(result, state->perf_lstm_forward_13);
        calc_total_perf_duration(result, state->perf_lstm_forward_14);
        calc_total_perf_duration(result, state->perf_lstm_forward_15);
        calc_total_perf_duration(result, state->perf_post_batch_copy);

        DELETE_ARRAY(state->obs_horizon);
        DELETE_ARRAY(state->values_horizon);
        DELETE_ARRAY(state->logprob_horizon);
        DELETE_ARRAY(state->actions_horizon);
        DELETE_ARRAY(state->rewards_horizon);
        DELETE_ARRAY(state->terminals_horizon);

        // Prepare for next run.
        state->lstm_wrapper = nullptr;
      }

      // Concatenate all at once
      result.obs = final_obs;
      result.values = final_values;
      result.logprob = final_logprobs;
      result.actions = final_actions;
      result.rewards = final_rewards;
      result.terminals = final_terminals;
      result.stats_millis.push_back({perf_total_forward_eval.name, perf_total_forward_eval.duration.count()});
    }
    END_LIBTORCH_CATCH
    return result;
  }

private:
  void alloc_tensor_arr(Tensor** arr) const
  {
    *arr = new Tensor[opt->bptt_horizon];
    for (size_t i = 0; i < opt->bptt_horizon; i++) { (*arr)[i] = Tensor{}; }
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
      auto segment_end = state->bptt_segment_end.load();
      if (segment_end > 0)
      {
        c_add_work_batched(state->vec_env, copy_to_final_buffers_async, state->lstm_wrapper, state->batch_index,
          state->batch_index);
      }

      if (segment_end >= this_ptr->opt->bptt_horizon) { return; }
      // printf(" Batch %d: Running BPTT segment %d / %d\n", batch_index, state->bptt_segment, opt->bptt_horizon);
      // Ok to perform synchronously as we need the obs tensor + forward eval before we can start env steps.
#ifdef PUFFER_CUDA
      if (this_ptr->device == torch::kCUDA)
      {
        // Using stream 1 Copy obs to device and forward eval on the correct CUDA stream in this thread.
        CUDAStreamGuard guard(*state->cuda_stream_1);
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
  static void copy_to_final_buffers_async(void* arg, int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      auto* this_ptr = static_cast<LSTMWrapper*>(arg);
      auto* state = this_ptr->env_states[batch_index];
      torch::NoGradGuard no_grad;
      state->perf_post_batch_copy.start();

#ifdef PUFFER_CUDA
      if (this_ptr->device == torch::kCUDA)
      {
        { // Using stream 1 Copy obs to device and forward eval on the correct CUDA stream in this thread.
          CUDAStreamGuard guard(*state->cuda_stream_2);
          this_ptr->copy_to_final_buffers(state);
        }
      }
      else // fallthrough
#endif
      {
        this_ptr->copy_to_final_buffers(state);
      }
      state->perf_post_batch_copy.stop();
    }
    END_LIBTORCH_CATCH
  }

//! @brief Async multi-threaded copy + forward eval pass for an entire batch of obs.
//! Assumed that run_next_bptt_segment sets the right CUDA stream before calling this function.
  void copy_to_final_buffers(PufferEnvState* state)
  {
    BEGIN_LIBTORCH_CATCH
    {
      auto batch_index = state->batch_index;
      RECORD_FUNCTION("final_copy_buffers", std::vector<c10::IValue>({static_cast<uint64_t>(batch_index)}));
      // This entire copy can proceed lock-free because the other thread produces a work in a new index we
      // possibly couldn't see (i.e. guarded by the atomic segment_end). And this function is the sole
      // owner of segment_start, so there's no race / conflicts here to necessitate a lock.
      auto segment_start = state->bptt_segment_start.load();
      auto segment_end = state->bptt_segment_end.load();
      if (!state->bptt_segment_start.compare_exchange_strong(segment_start, segment_end))
      {
        // Some other thread got here before we did. Just return.
        return;
      }

      const int64_t env_start = state->env_start_index;
      const int64_t n = state->env_count;

      for (auto seg = segment_start; seg < segment_end; seg++)
      {
        // final_obs: [N, H, O]  -> narrow envs => [n, H, O] -> select seg => [n, O]
        final_obs.narrow(0, env_start, n).select(1, seg).copy_(state->obs_horizon[seg], true);
        // final_values/logprobs/rewards/terminals: [N, H] -> narrow => [n, H] -> select => [n]
        final_values.narrow(0, env_start, n).select(1, seg).copy_(state->values_horizon[seg], true);
        final_logprobs.narrow(0, env_start, n).select(1, seg).copy_(state->logprob_horizon[seg], true);
        final_rewards.narrow(0, env_start, n).select(1, seg).copy_(state->rewards_horizon[seg], true);
        final_terminals.narrow(0, env_start, n).select(1, seg).copy_(state->terminals_horizon[seg], true);

        // - discrete: final_actions [N, H], horizon [n]
        // - multi-discrete: final_actions [N, H, A], horizon [n, A]
        final_actions.narrow(0, env_start, n).select(1, seg).copy_(state->actions_horizon[seg], true);
        state->obs_horizon[seg] = Tensor{};
        state->values_horizon[seg] = Tensor{};
        state->logprob_horizon[seg] = Tensor{};
        state->rewards_horizon[seg] = Tensor{};
        state->terminals_horizon[seg] = Tensor{};
        state->actions_horizon[seg] = Tensor{};
      }
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

        // printf("batch obs copy: %d\n", batch_index);
        // NOTE: Env observations are memory mapped to the full_obs_cpu tensor already.
        // Once it's on device, changes are no longer reflected unless we copy again.
        state->obs_device = state->obs_cpu.to(device);
        state->obs_horizon[state->bptt_segment_end.load()] = (state->obs_device);
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
      state->perf_lstm_forward.start();
      auto obs_tensor = state->obs_device;
      state->perf_lstm_forward_1.start();
      auto hidden = encoder->forward(obs_tensor);
      state->perf_lstm_forward_1.stop();

      state->perf_lstm_forward_2.start();
      auto hc = lstm_cell->forward(hidden, std::make_tuple(state->h, state->c));
      state->perf_lstm_forward_2.stop();
      auto h = std::get<0>(hc);
      auto c = std::get<1>(hc);
      state->h = h;
      state->c = c;
      if (opt->is_continuous)
      {
        PUFFER_ASSERT(!opt->is_continuous, "Only supports (multi)discrete for now.");
        auto mean = decoder_mean->forward(h);
        auto logstd = decoder_logstd.expand_as(mean);
        auto std_dev = torch::exp(logstd);
        auto noise = torch::randn_like(mean);
        auto action_sample = mean + std_dev * noise;
        for (int i = 0; i < opt->num_actions; i++)
        {
          // actions[i] = static_cast<int>(action_sample[0][i].item<float>());
        }
        throw std::runtime_error("Continuous action space not implemented yet.");
        // TODO(perumaal): Need to update state->logits as well and verify this with the puffernet impl.
      }
      else
      {
        // TODO: Parallelize these two forwards? Probably not worth it as these are just linear layers.
        state->perf_lstm_forward_3.start();
        auto logits = decoder->forward(h);
        state->perf_lstm_forward_3.stop();
        state->perf_lstm_forward_4.start();
        auto values = value->forward(h);
        state->perf_lstm_forward_4.stop();
        state->perf_lstm_forward_5.start();
        values = values.flatten();
        state->perf_lstm_forward_5.stop();
        // Put into a tuple of num_actions tensors, each with N logits.
        // Shape after split and stack: [num_actions, num_envs, logit_size]
        state->perf_lstm_forward_6.start();
        auto split_logits = logits.split(at::IntArrayRef(opt->logit_sizes, opt->num_actions), /*dim=*/1);
        state->perf_lstm_forward_6.stop();
        // logits: [num_envs, sum(logit_sizes)]
        // stacked_logits: [A, N, K_a] to match Python multi-discrete layout
        state->perf_lstm_forward_7.start();
        logits = torch::stack(split_logits, /*dim=*/0); // [A, N, K_a]
        state->perf_lstm_forward_7.stop();

        state->perf_lstm_forward_8.start();
        auto normalized_logits = logits - logits.logsumexp(/*dim=*/-1, /*keepdim=*/true);
        state->perf_lstm_forward_8.stop();

        state->perf_lstm_forward_9.start();
        auto probs = torch::softmax(logits, /*dim=*/-1);
        state->perf_lstm_forward_9.stop();

        state->perf_lstm_forward_10.start();
        probs = torch::nan_to_num(
          probs,
          /*nan=*/1e-8,
          /*posinf=*/1e-8,
          /*neginf=*/1e-8);
        state->perf_lstm_forward_10.stop();

        state->perf_lstm_forward_11.start();
        // probs: [A, N, K] when A >= 1
        auto actions_flat = torch::multinomial(
          probs.reshape({-1, probs.size(-1)}),
          /*num_samples=*/1,
          /*replacement=*/true);                                                       // [A*N, 1]
        auto actions_heads_env = actions_flat.reshape({probs.size(0), probs.size(1)}); // [A, N]
        state->perf_lstm_forward_11.stop();

        auto segment = state->bptt_segment_end.load();
        state->values_horizon[segment] = values;

        state->perf_lstm_forward_12.start();

        // Discrete: A == 1, Python returns action.squeeze(0), logprob.squeeze(0)
        Tensor logprob_sampled;
        Tensor actions_for_env;

        if (opt->num_actions == 1)
        {
          // actions_heads_env: [1, N] -> env-major [N]
          auto actions_env = actions_heads_env.squeeze(0).to(torch::kLong); // [N]

          // normalized_logits: [1, N, K] -> [N, K] for discrete
          auto norm_logits_discrete = normalized_logits.squeeze(0); // [N, K]

          // logprob = log_prob(norm_logits_discrete, actions_env) -> [N]
          auto actions_env_exp = actions_env.unsqueeze(-1); // [N, 1]
          auto gathered = norm_logits_discrete.gather(
            /*dim=*/-1,
            actions_env_exp);                     // [N, 1]
          logprob_sampled = gathered.squeeze(-1); // [N]

          // Store actions as [N] to match Python discrete path
          actions_for_env = actions_env.to(torch::kInt32); // [N]
        }
        else
        {
          // Multi-discrete: A > 1
          auto actions_t = actions_heads_env.to(torch::kLong); // [A, N]
          auto actions_expanded = actions_t.unsqueeze(-1);     // [A, N, 1]
          auto gathered_logprob = normalized_logits.gather(
            /*dim=*/-1,
            actions_expanded).squeeze(-1);                   // [A, N]
          logprob_sampled = gathered_logprob.sum(/*dim=*/0); // [N]

          // Store actions as env-major [N, A] (transpose) to match Python return
          actions_for_env = actions_heads_env.transpose(0, 1).to(torch::kInt32); // [N, A]
        }

        state->logprob_horizon[segment] = logprob_sampled;
        state->actions_horizon[segment] = actions_for_env;

        state->perf_lstm_forward_12.stop();

        state->perf_lstm_forward_14.start();
        const auto actions_int = actions_for_env.to(
          torch::kCPU,
          /*non_blocking=*/true,
          /*copy=*/true,
          {c10::MemoryFormat::Contiguous});
        auto* actions_data = actions_int.data_ptr<int>();
        state->perf_lstm_forward_14.stop();

        state->perf_lstm_forward_15.start();
        for (int i = 0; i < state->env_count; i++)
        {
          const int env_index = state->env_start_index + i;
          Env* env = state->vec_env->envs[env_index];
          int* actions_ptr = get_actions_ptr(env);
          const int* src = actions_data + static_cast<int64_t>(i) * opt->num_actions;
          std::memcpy(
            actions_ptr,
            src,
            static_cast<size_t>(opt->num_actions) * sizeof(int));
        }
        state->perf_lstm_forward_15.stop();
      }

      state->perf_lstm_forward.stop();

      state->perf_env_cpu.start();
      // Run the batch's env steps independently in different threads.
      // Once all envs from this batch have completed, continue on to run the next BPTT segment.
      c_add_work_batched(vec_env, c_step_batch, state->vec_env->envs, state->env_start_index,
        state->env_start_index + state->env_count - 1,
        [state](void* _) // Unused as it's per-env, we need the batch captured state.
        {
          BEGIN_LIBTORCH_CATCH
          {
            RECORD_FUNCTION("finalize_bptt_segment",
              std::vector<c10::IValue>({static_cast<uint64_t>(state->batch_index)}));

            state->perf_env_cpu.stop();
            auto* rewards_arr = static_cast<float*>(state->rewards_cpu.data_ptr());
            auto* terminals_arr = static_cast<float*>(state->terminals_cpu.data_ptr());
            for (int i = 0; i < state->env_count; i++)
            {
              const int env_index = state->env_start_index + i;
              Env* env = state->vec_env->envs[env_index];
              float r = get_rewards_ptr(env)[0];
              r = std::max(-1.0f, std::min(1.0f, r));
              auto* terminals_ptr = get_terminals_ptr(env);
              rewards_arr[i] = r;
              terminals_arr[i] = (terminals_ptr[0] != 0 ? 1.0f : 0.0f);
            }
            auto segment = state->bptt_segment_end.load();
            state->rewards_horizon[segment] = (state->rewards_cpu);
            state->terminals_horizon[segment] = (state->terminals_cpu);
          }
          END_LIBTORCH_CATCH

          // Schedule this work for the next segment. (We could reuse this thread, but let's let the OS
          // manage the priorities and let the cascade happen naturally).
          atomic_fetch_add(&state->bptt_segment_end, 1);
          c_add_work_batched(state->vec_env, run_next_bptt_segment, state->lstm_wrapper,
            state->batch_index, state->batch_index);
        });
    }
    END_LIBTORCH_CATCH
  }

// All of these are thread-safe during a single eval call (except for update_model_weights).
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

  PufferOptions* opt{nullptr};

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
      "Native multithreading/libtorch: %d envs on %d threads (batch size = max %d envs/batch; total %d batches).\n",
      vec_env->num_envs, opts->num_threads, ptorch->model->eval_batch_size, ptorch->model->eval_batch_count);

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

//
// Threading support.
//
struct ThreadWork
{
  work_func func;
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
        work_items.pop_back();
        work_count.fetch_add(1);
      }

      for (int i = work.start_index; i <= work.end_index; i++)
      {
        // NOTE: work.func could end up adding more tasks, so we have to notify the producer
        // only within the lock above to prevent race conditions/incomplete done-ness.
        work.func(work.arg, i);
      }

      check_call_done(work);

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
    while (work_count.load() != 0 || !work_items.empty())
    {
      done_cv.wait(lock);
    }
  }

  void add_work(const ThreadWork& work)
  {
    if (num_threads.load() == 0)
    {
      return;
    } // TODO: Throw?
    {
      std::lock_guard<std::mutex> lock(work_mutex);
      work_items.push_back(work);
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

//! Internal function to add batched work with optional batch group (if provided, batch group will be first setup to
//! track total tasks). Use the optional batch group to queue up a completion routine on the full batch of work added.
void c_add_work_batched(VecEnv* vec_env, work_func func, void* arg, int start_index, int end_index,
  std::function<void(void*)> batch_completion_cb)
{
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
  c_add_work_batched(vec_env, func, arg, start_index, end_index, nullptr);
}

void c_wait_all_done(VecEnv* vec_env)
{
  PUFFER_ASSERT(vec_env->threading != nullptr, "Invalid threading state.");
  vec_env->threading->wait_all_done();
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
      .def_readwrite("obs", &PufferEvalResult::obs)
      .def_readwrite("values", &PufferEvalResult::values)
      .def_readwrite("logits", &PufferEvalResult::logits)
      .def_readwrite("logprob", &PufferEvalResult::logprob)
      .def_readwrite("entropy", &PufferEvalResult::entropy)
      .def_readwrite("actions", &PufferEvalResult::actions)
      .def_readwrite("rewards", &PufferEvalResult::rewards)
      .def_readwrite("terminals", &PufferEvalResult::terminals)
      .def_readwrite("stats_millis", &PufferEvalResult::stats_millis);


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
