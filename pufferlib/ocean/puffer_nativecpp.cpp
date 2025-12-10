#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include "puffer_nativecpp.h"

#include <torch/torch.h>
#include <cassert>
#include <iostream>
#include <atomic>
#include <condition_variable>
#include <thread>

#ifdef PUFFER_CUDA
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
using ::c10::cuda::CUDAStream;
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
#define END_LIBTORCH_CATCH        \
    catch (const c10::Error& e)   \
    {                             \
      std::cerr << "Error from libtorch: " << e.what() << std::endl;\
      PUFFER_ASSERT_BREAK();      \
      throw;                      \
    }

#else
#define END_LIBTORCH_CATCH
#endif


#ifndef PUFFER_EXTERN
// Silliness as the header is included in both C and C++ files (and from binding.c from each env). Makes it very hard to separate it.
struct Env;
struct VecEnv;
#define PUFFER_EXTERN extern "C"
#endif


// APIs to separate env_glue/env_binding stuff from libtorch cleanly.
PUFFER_EXTERN float* get_obs_ptr(Env* env);
PUFFER_EXTERN int* get_actions_ptr(Env* env);
PUFFER_EXTERN float* get_rewards_ptr(Env* env);
PUFFER_EXTERN unsigned char* get_terminals_ptr(Env* env);
PUFFER_EXTERN void c_step_glue(Env* env);


// Optional batch group that takes a completion function and tracks pending tasks.
struct BatchCompletion
{
  std::mutex mutex; // Mainly for the caller to hold on to while waiting on cv below.
  // Use this callback to do your thing after the batch completes naturally instead of waiting 
  // for the batch to complete. A la promises/futures that do not block the current threads (as we only have a few threads to service
  // many tasks).
  std::function<void(void*)> batch_completion_cb;
  std::atomic_int done_tasks = 0;
  std::atomic_int batch_total_tasks = 0;

  BatchCompletion(std::function<void(void*)> batch_completion) : batch_completion_cb(batch_completion)
  {
    PUFFER_ASSERT(batch_completion != nullptr, "BatchGroup requires a non-empty callback.");
  }

  explicit BatchCompletion() = delete; // Do not allow passing in an empty callback.

  inline void check_call_done(void* arg, const int completed_count)
  {
    std::unique_lock<std::mutex> lock(mutex);
    done_tasks.fetch_add(completed_count);
    // Must perform this under a lock because additional tasks may be added (also ensure we only call once per batch).
    if (done_tasks == batch_total_tasks)
    {
      // The callback can end up adding more tasks to the batch.
      batch_completion_cb(arg);
    }
  }
};

void c_add_work_batched(VecEnv* vec_env, work_func func, void* arg, int start_index, int end_index,
  std::shared_ptr<BatchCompletion> batch_completion);

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
  std::cout << "Tensor: " << name << "  " << tensor.device() << " / " << tensor.dtype() << " / " << tensor.sizes() <<
      " ]" << std::endl;
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
  // For the LSTM wrapper.
  Tensor obs_cpu, obs_device, h, c;
  Tensor rewards_cpu, terminals_cpu;
  Tensor logits_entropy_unused;

  // Across the entire BPTT horizon for training later on.
  std::vector<Tensor> obs_horizon, values_horizon, logprob_horizon, actions_horizon, rewards_horizon, terminals_horizon;
  // Global params for quick referencing.
  LSTMWrapper* lstm_wrapper;
  int bptt_segment;
  VecEnv* vec_env;
  PerfTimer perf_env_cpu;
  PerfTimer perf_to_device_copy; // Copy obs to GPU.
  PerfTimer perf_lstm_forward;
  PerfTimer perf_to_cpu_copy; // Copy actions back to CPU.
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
#endif
    torch::NoGradGuard no_grad;
    device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    encoder_linear = layer_init(torch::nn::Linear(opt->obs_size, opt->hidden_size));
    encoder_gelu = torch::nn::GELU();
    encoder = register_module("encoder", torch::nn::Sequential(encoder_linear, encoder_gelu));
    if (opt->is_continuous)
    {
      decoder_mean = register_module("decoder_mean",
        layer_init(torch::nn::Linear(opt->hidden_size, opt->num_actions), 0.01));
      decoder_logstd = register_parameter("decoder_logstd", torch::zeros({1, opt->num_actions}));
    }
    else
    {
      opt->num_atns = 0;
      for (int i = 0; i < opt->num_actions; i++) { opt->num_atns += opt->logit_sizes[i]; }
      decoder = register_module("decoder", layer_init(torch::nn::Linear(opt->hidden_size, opt->num_atns), 0.01));
    }
    value = register_module("value", layer_init(torch::nn::Linear(opt->hidden_size, 1), 1.0));
    lstm_cell = register_module("lstmcell", torch::nn::LSTMCell(opt->input_size, opt->hidden_size));
    int batch_chunk_size = (opt->batch_chunk_size_kb * 1024) / (opt->obs_size * sizeof(float));
    if (batch_chunk_size < 1) { batch_chunk_size = 1; }
    eval_batch_size = batch_chunk_size;
    eval_batch_count = (num_envs + batch_chunk_size - 1) / batch_chunk_size;
    env_states = new PufferEnvState*[eval_batch_count];
    for (int i = 0; i < eval_batch_count; i++)
    {
      auto* state = (env_states[i] = new PufferEnvState());
      const int start_idx = i * eval_batch_size;
      int env_count = eval_batch_size;
      if (i == eval_batch_count - 1) { env_count = num_envs - start_idx; }
      state->h = torch::zeros({env_count, opt->hidden_size}, device);
      state->c = torch::zeros({env_count, opt->hidden_size}, device);
      state->batch_index = i;
      state->env_start_index = start_idx;
      state->env_count = env_count;
    }
  }

  ~LSTMWrapper() override
  {
    for (int i = 0; i < eval_batch_count; i++)
    {
      delete env_states[i];
      env_states[i] = nullptr;
    }
    delete[] env_states;
    env_states = nullptr;
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

  void start_batch_eval_lstm(VecEnv* vec_env, Tensor full_obs_cpu, Tensor full_rewards_cpu, Tensor full_terminals_cpu,
    Tensor encoder_linear_w, Tensor encoder_linear_b,
    Tensor decoder_linear_w, Tensor decoder_linear_b,
    Tensor value_w, Tensor value_b,
    Tensor weight_ih, Tensor weight_hh,
    Tensor bias_ih, Tensor bias_hh)
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
      full_obs_cpu = full_obs_cpu;
      for (int i = 0; i < eval_batch_count; i++)
      {
        auto* state = env_states[i];
        state->bptt_segment = 0;
        // Per-batch/per-bptt-segment slices.
        state->obs_cpu = full_obs_cpu.narrow(0, state->env_start_index, state->env_count);
        state->rewards_cpu = full_rewards_cpu.narrow(0, state->env_start_index, state->env_count);
        state->terminals_cpu = full_terminals_cpu.narrow(0, state->env_start_index, state->env_count);

        state->h = state->h.zero_();
        state->c = state->c.zero_();
        state->logits_entropy_unused = Tensor{};
        state->lstm_wrapper = this;
        state->vec_env = vec_env;

        state->perf_env_cpu = PerfTimer{.name = "env_cpu"};
        state->perf_to_device_copy = PerfTimer{.name = "to_device_copy"};
        state->perf_lstm_forward = PerfTimer{.name = "lstm_forward"};
        state->perf_to_cpu_copy = PerfTimer{.name = "to_cpu_copy"};
      }
    }
    END_LIBTORCH_CATCH
  }

  // Batched env forward eval. This starts the process per segment in the horizon. Waits for all segments to finish and then return
  // the batched tensor set back.
  void forward_eval_batch(VecEnv* vec_env)
  {
    BEGIN_LIBTORCH_CATCH
    {
      torch::NoGradGuard no_grad;

      c_start_work(vec_env);
      // Kick off this batch of work.
      // TODO: Should we do each batch-segment part of this horizon independently? or all at once?
      // We can start off with putting this whole thing in a for loop (i.e. each iteration, wait for all done) to begin with.
      // I think ideally, some stuff should just start going forward.
      c_add_work_batched(vec_env, run_next_bptt_segment, this, 0, eval_batch_count - 1);
      // full_obs is [num_envs, obs_size] in CPU side.
      // Transfer each obs batch to device independently.
      // Add batch work: torch_batch_eval(this, index)
      // Get the action[]/etc tensors from each batch.
      // Enqueue the env steps.
      // cat all tensors and return.
      c_wait_all_done(vec_env);
    }
    END_LIBTORCH_CATCH
  }

  //! @brief Returns all the tensors (on target device) plus stats across all batches.
  PufferEvalResult finish_batch_eval_lstm(VecEnv* env)
  {
    std::vector<Tensor> obs_vec, values_vec, logprob_vec, actions_vec, rewards_vec, terminals_vec;
    PufferEvalResult result;
    BEGIN_LIBTORCH_CATCH
    {
      double total_env_cpu_ms = 0.0;
      double total_to_device_copy_ms = 0.0;
      double total_lstm_forward_ms = 0.0;
      double total_to_cpu_copy_ms = 0.0;

      // Pre-allocate result tensors
      int total_horizon = opt->bptt_horizon * eval_batch_count;

      obs_vec.reserve(total_horizon);
      values_vec.reserve(total_horizon);
      logprob_vec.reserve(total_horizon);
      actions_vec.reserve(total_horizon);
      rewards_vec.reserve(total_horizon);
      terminals_vec.reserve(total_horizon);

      for (int i = 0; i < eval_batch_count; i++)
      {
        auto* state = env_states[i];
        obs_vec.push_back(torch::stack(state->obs_horizon, /*dim=*/0));
        values_vec.push_back(torch::stack(state->values_horizon, /*dim=*/0));
        logprob_vec.push_back(torch::stack(state->logprob_horizon, /*dim=*/0));
        actions_vec.push_back(torch::stack(state->actions_horizon, /*dim=*/0));
        rewards_vec.push_back(torch::stack(state->rewards_horizon, /*dim=*/0));
        terminals_vec.push_back(torch::stack(state->terminals_horizon, /*dim=*/0));

        total_env_cpu_ms += state->perf_env_cpu.duration.count();
        total_to_device_copy_ms += state->perf_to_device_copy.duration.count();
        total_lstm_forward_ms += state->perf_lstm_forward.duration.count();
        total_to_cpu_copy_ms += state->perf_to_cpu_copy.duration.count();

        // Prepare for next run.
        state->lstm_wrapper = nullptr;
      }

      // Concatenate all at once
      result.obs = torch::stack(obs_vec, /*dim=*/0);
      c_print_tensor_info(result.obs, "Final Obs Tensor", false);
      result.values = torch::stack(values_vec, /*dim=*/0);
      c_print_tensor_info(result.values, "Final values Tensor", false);
      result.logprob = torch::stack(logprob_vec, /*dim=*/0);
      c_print_tensor_info(result.logprob, "Final logprob Tensor", false);
      result.actions = torch::stack(actions_vec, /*dim=*/0);
      c_print_tensor_info(result.actions, "Final actions Tensor", false);
      result.rewards = torch::stack(rewards_vec, /*dim=*/0);
      c_print_tensor_info(result.rewards, "Final rewards Tensor", false);
      result.terminals = torch::stack(terminals_vec, /*dim=*/0);
      c_print_tensor_info(result.terminals, "Final terminals Tensor", false);
      result.stats_millis.push_back({"env_cpu", total_env_cpu_ms});
      result.stats_millis.push_back({"to_device_copy", total_to_device_copy_ms});
      result.stats_millis.push_back({"lstm_forward", total_lstm_forward_ms});
      result.stats_millis.push_back({"to_cpu_copy", total_to_cpu_copy_ms});
    }
    END_LIBTORCH_CATCH
    return result;
  }

private:
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
      if (state->bptt_segment >= this_ptr->opt->bptt_horizon) { return; }
      // printf(" Batch %d: Running BPTT segment %d / %d\n", batch_index, state->bptt_segment, opt->bptt_horizon);
      // Ok to perform synchronously as we need the obs tensor + forward eval before we can start env steps.
      this_ptr->copy_obs_forward_eval_batch(batch_index);
    }
    END_LIBTORCH_CATCH
  }

  //! @brief Async multi-threaded copy + forward eval pass for an entire batch of obs.
  void copy_obs_forward_eval_batch(int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      // We must do this per thread work as it's TLS guarded.
      torch::NoGradGuard no_grad;
      auto* state = env_states[batch_index];
#ifdef PUFFER_CUDA
      if (device == torch::kCUDA)
      {
        // TODO: CUDA stream synchronize the copy / forward / cpu transfers?
      }
#endif
      {
        RECORD_FUNCTION("batch_copy_to_device", std::vector<c10::IValue>({static_cast<uint64_t>(batch_index)}));
        state->perf_to_device_copy.start();

        // printf("batch obs copy: %d\n", batch_index);
        // NOTE: Env observations are memory mapped to the full_obs_cpu tensor already. 
        // Once it's on device, changes are no longer reflected unless we copy again.
        state->obs_device = state->obs_cpu.to(device);
        state->obs_horizon.push_back(state->obs_device);
        state->perf_to_device_copy.stop();
      }
      // TODO: Use non-blocking and await when the obs are in the GPU? May be not...
      //       Currently, we use this thread to block until the copy is done. 
      //       We maximize the number of parallel copies, so this should already be optimal?
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
      auto hidden = encoder->forward(obs_tensor);
      auto hc = lstm_cell->forward(hidden, std::make_tuple(state->h, state->c));
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
          //actions[i] = static_cast<int>(action_sample[0][i].item<float>());
        }
        throw std::runtime_error("Continuous action space not implemented yet.");
        // TODO(perumaal): Need to update state->logits as well and verify this with the puffernet impl.
      }
      else
      {
        // TODO: Parallelize these two forwards? Probably not worth it as these are just linear layers.
        auto logits = decoder->forward(h);
        auto values = value->forward(h);
        // Put into a tuple of num_actions tensors, each with N logits.
        // Shape after split and stack: [num_actions, num_envs, logit_size]
        auto split_logits = logits.split(at::IntArrayRef(opt->logit_sizes, opt->num_actions), /*dim=*/1);
        logits = torch::stack(split_logits, /*dim=*/0);
        auto normalized_logits = logits - logits.logsumexp(/*dim=*/-1, /*keepdim=*/true);
        auto logprob = torch::log_softmax(logits, /* dim=*/ -1);
        auto probs = logprob.exp();
        probs = torch::nan_to_num(probs, /*nan=*/0.0, /*posinf=*/1e8, /*neginf=*/-1e8);
        auto actions = torch::multinomial(probs.reshape({-1, probs.size(-1)}), /*num_samples=*/1, /*replacement=*/
          true);
        actions = actions.reshape({probs.size(0), probs.size(1)});
        actions = actions.transpose(0, 1).to(torch::kInt32);
        state->values_horizon.push_back(values);
        state->logprob_horizon.push_back(logprob.sum(0));
        state->actions_horizon.push_back(actions);
        const auto actions_int = actions.to(torch::kCPU);
        for (int i = 0; i < state->env_count; i++)
        {
          const int env_index = state->env_start_index + i;
          Env* env = state->vec_env->envs[env_index];
          int* actions_ptr = get_actions_ptr(env);
          for (int j = 0; j < opt->num_actions; j++)
          {
            actions_ptr[j] = actions_int[i][j].item<int>();
          }
        }
      }

      state->perf_lstm_forward.stop();

      state->perf_env_cpu.start();
      // Run the batch's env steps independently in different threads. 
      // Once all envs from this batch have completed, continue on to run the next BPTT segment.
      c_add_work_batched(vec_env, batch_env_step, state,
        state->env_start_index, state->env_start_index + state->env_count - 1,
        std::make_shared<BatchCompletion>([](void* arg)
        {
          auto* state = static_cast<PufferEnvState*>(arg);
          BEGIN_LIBTORCH_CATCH
          {
            state->perf_env_cpu.stop();
            state->bptt_segment++;
            auto rewards_arr = static_cast<float*>(state->rewards_cpu.data_ptr());
            auto terminals_arr = static_cast<float*>(state->terminals_cpu.data_ptr());
            for (int i = 0; i < state->env_count; i++)
            {
              const int env_index = state->env_start_index + i;
              Env* env = state->vec_env->envs[env_index];
              float& r = get_rewards_ptr(env)[0];
              r = std::max(-1.0f, std::min(1.0f, r));
              unsigned char* terminals_ptr = get_terminals_ptr(env);
              float t = (terminals_ptr[0] != 0 ? 1.0f : 0.0f);
              rewards_arr[i] = r;
              terminals_arr[i] = t;
            }
            state->rewards_horizon.push_back(state->rewards_cpu);
            state->terminals_horizon.push_back(state->terminals_cpu);
          }
          END_LIBTORCH_CATCH

          run_next_bptt_segment(state->lstm_wrapper, state->batch_index);
        }));
    }
    END_LIBTORCH_CATCH
  }

  //! @brief Async multi-threaded env step per env (in a batch).
  static inline void batch_env_step(void* arg, int env_index)
  {
    PufferEnvState* state = static_cast<PufferEnvState*>(arg);
    // printf("batch env #: %d\n", env_index);

    PUFFER_ASSERT(env_index >= state->env_start_index && env_index < state->env_start_index + state->env_count,
      "Invalid env index for batch.");
    // The obs_torch tensor array(s) are mapped to each env's observations float array via pointer ref in CPU side.
    // So any changes here are reflected in the CPU tensor automatically.
    Env* env = state->vec_env->envs[env_index];
    c_step_glue(env);
  }

  // All of these are multi-thread safe during a single eval call (except for update_model_weights).
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
  // Full obs space across all envs.
  Tensor full_obs_cpu;
  VecEnv* vec_env;
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
  for (int i = 0; i < num_actions; i++) { options->logit_sizes[i] = num_logits; }
  options->input_size = input_size;
  options->hidden_size = hidden_size;
  options->is_continuous = is_continuous;
}

void c_cleanup_pufferoptions(VecEnv* vec_env)
{
  if (vec_env->opts.logit_sizes)
  {
    delete[] vec_env->opts.logit_sizes;
    vec_env->opts.logit_sizes = nullptr;
  }
  vec_env->opts = {};
}


PufferTorch* c_torch_alloc(VecEnv* vec_env)
{
  BEGIN_LIBTORCH_CATCH
  {
    PufferOptions* opts = &vec_env->opts;
    PUFFER_ASSERT(
      opts != nullptr && opts->num_actions > 0 && opts->num_atns == 0 && opts->logit_sizes != nullptr && opts->
      enable_native_libtorch,
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
    delete pt->model;
    pt->model = nullptr;
    delete pt;
  }
  END_LIBTORCH_CATCH
}

void c_torch_start_eval_lstm(uintptr_t vec_env_ptr,
  Tensor full_obs_cpu, Tensor full_rewards_cpu, Tensor full_terminals_cpu,
  Tensor encoder_linear_w, Tensor encoder_linear_b,
  Tensor decoder_linear_w, Tensor decoder_linear_b,
  Tensor value_w, Tensor value_b,
  Tensor weight_ih, Tensor weight_hh,
  Tensor bias_ih, Tensor bias_hh)
{
  VecEnv* vec_env = (VecEnv*)vec_env_ptr;
  PufferTorch* puff_torch = vec_env->puff_torch;
  PUFFER_ASSERT(puff_torch != nullptr && puff_torch->model != nullptr, "Invalid state.");

  puff_torch->model->start_batch_eval_lstm(vec_env, full_obs_cpu, full_rewards_cpu, full_terminals_cpu,
    encoder_linear_w, encoder_linear_b, decoder_linear_w, decoder_linear_b, value_w, value_b,
    weight_ih, weight_hh, bias_ih, bias_hh);
}

//! @brief Performs action (inference) + step segmented across a BPTT horizon batched by envs.
//! Waits for the entire run to finish. TODO: Clarify - full bptt horizon ? or a single segment? TODO: log timing perf metrics
void c_torch_run_fulleval(uintptr_t vec_env_ptr)
{
  BEGIN_LIBTORCH_CATCH
  {
    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferTorch* pt = vec_env->puff_torch;
    PUFFER_ASSERT(
      pt != nullptr && pt->model != nullptr && vec_env->num_envs > 0 && vec_env->envs != nullptr &&
      vec_env->threading != nullptr, "Invalid state/inputs.");
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
        if (last_count == 1) { done_cv.notify_all(); }
        while (!(num_threads.load() == 0 || !work_items.empty())) { work_cv.wait(lock); }
        // Shortcuts to exit or try again in case we got woken up but no work.
        if (num_threads.load() == 0) { break; }
        if (work_items.empty()) { continue; }
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

      auto* batch_completion = work.batch_completion.get();
      if (batch_completion != nullptr)
      {
        batch_completion->check_call_done(work.arg, work.end_index - work.start_index + 1);
      }

      last_count = work_count.fetch_sub(1);
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
    if (num_threads.load() == 0) { return; } // TODO: Throw?
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
      if (thread.joinable()) { thread.join(); }
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
    delete vec_env->threading;
    vec_env->threading = nullptr;
  }
}

void c_start_work(struct VecEnv* vec_env)
{
  PUFFER_ASSERT(vec_env->threading != nullptr, "Invalid threading state.");
  vec_env->threading->check_empty();
}

//! Internal function to add batched work with optional batch group (if provided, batch group will be first setup to track total tasks). 
//! Use the optional batch group to queue up a completion routine on the full batch of work added.
void c_add_work_batched(VecEnv* vec_env, work_func func, void* arg, int start_index, int end_index,
  std::shared_ptr<BatchCompletion> batch_completion)
{
  PUFFER_ASSERT(vec_env->threading != nullptr && end_index >= start_index, "Invalid threading state.");
  const auto num_threads = vec_env->threading->num_threads.load();
  if (batch_completion != nullptr && batch_completion->batch_completion_cb != nullptr)
  {
    std::unique_lock<std::mutex> lock(batch_completion->mutex);
    // Note: a work item may add more work items, so we have to do this upfront and with minimal locking.
    batch_completion->batch_total_tasks.fetch_add(end_index - start_index + 1);
  }
  else
  {
    // If no callback was provided, avoid extra work.
    batch_completion = nullptr;
  }
  if (end_index == start_index)
  {
    vec_env->threading->add_work({
      .func = func, .arg = arg, .start_index = start_index, .end_index = end_index, .batch_completion = batch_completion
    });
    return;
  }
  const int batch_size = (end_index - start_index + 1 + num_threads) / num_threads;
  for (; start_index < end_index; start_index += batch_size)
  {
    int item_end = start_index + batch_size;
    if (item_end >= end_index) { item_end = end_index; }
    else { item_end--; }
    vec_env->threading->add_work({
      .func = func, .arg = arg, .start_index = start_index, .end_index = item_end, .batch_completion = batch_completion
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
      .def_readwrite("stats_millis", &PufferEvalResult::stats_millis);


  import_array();
  PyModule_AddFunctions(m.ptr(), get_c_env_binding_methods());
  m.def("libtorch_info", &c_libtorch_info, "Print libtorch info to stdout.");
  m.def("torch_start_eval_lstm", &c_torch_start_eval_lstm, py::arg("vec_env"),
    py::arg("full_obs_cpu"),
    // Full observation tensor on CPU across all horizons/envs with shape [envs, horizon, obs_count].
    py::arg("full_rewards_cpu"),   // Full rewards tensor on CPU across all horizons/envs [envs, horizon, 1].
    py::arg("full_terminals_cpu"), // Full terminals tensor on CPU across all horizons/envs [envs, horizon, 1].
    py::arg("encoder_linear_w"), py::arg("encoder_linear_b"), py::arg("decoder_linear_w"),
    py::arg("decoder_linear_b"), py::arg("value_w"), py::arg("value_b"), py::arg("weight_ih"), py::arg("weight_hh"),
    py::arg("bias_ih"), py::arg("bias_hh"), "Start the initial torch eval (before starting the horizon segments).");

  m.def("torch_run_fulleval", &c_torch_run_fulleval, py::arg("vec_env"),
    "Runs the full forward eval pass using libtorch for all segments in the horizon.");

  m.def("torch_finish_eval_lstm", &c_torch_finish_eval_lstm, py::arg("vec_env"),
    "Finish the torch eval (after all segments in the horizon are done).");
}

#endif
