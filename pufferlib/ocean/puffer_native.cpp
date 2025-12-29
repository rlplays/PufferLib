#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include "puffer_native.h"

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <torch/torch.h>

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

using namespace std;
using torch::Tensor;
using namespace std;


#ifdef PUFFER_CUDA
// Enable multi-threaded CUDA streams by default.
constexpr bool global_cuda_async = true;
// Enable multiple streams per batch by default. 2 means double-buffering etc.
// Very useful doc: https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management
// Set to 0 to disable multiple cuda streams (and instead use the default TLS one).
constexpr int global_max_num_cuda_streams = 32;
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
using namespace ::c10::cuda;
// Uncomment this to print memory info while debugging.
//#define PUFFER_CUDA_MEMCHECK 1
#else
constexpr bool global_cuda_async = false;
#endif


#include "puffer_threads.h"
#include "puffer_utils.h"

struct LSTMWrapper;


//! @brief Holds the state for a batch of envs.
struct PufferBatchState
{
  // Batch index within the envs.
  int batch_index;
  // The envs within this batch.
  int env_start_index;
  int env_count;
  int min_num_envs_per_batch;
  // For the LSTM wrapper.
  Tensor h1, c1;
  // The following can be released after a segment is processed.
  Tensor obs_cpu, obs_device;
  Tensor actions_cpu;
  Tensor rewards_cpu, terminals_cpu;
  Tensor logits_entropy_unused;

  // Stores the intermediate segments across a horizon for copying into the out tensors.
  // One set of threads write to the arr[bptt_segment] while the other thread reads/copies over the tensors.
  Tensor *values_horizon, *logprob_horizon, *actions_horizon, *rewards_horizon, *terminals_horizon;


  // Output tensors preallocated to avoid cuda malloc / stream synchronization overhead.
  // Forward pass - encoder output.
  Tensor hidden_out;
  // Forward pass - LSTM output (/input)
  // Double-buffer h1/c1 <-> h2/c2 to avoid cudaMallocs/stream syncs. Each batch proceeds linearly
  // where segment1 uses h1/c1 to generate h2/c2, segment2 uses h2/c2 to generate h1/c1 etc.
  Tensor h2, c2;
  // For our custom LSTM kernel.
  Tensor igates, hgates, workspace;
  Tensor decoder_out;
  Tensor values_out;
  Tensor logprobs_out, actions_out;
  // Global params for quick referencing.
  LSTMWrapper* lstm_wrapper;
  atomic_int bptt_segment;
  VecEnv* vec_env;
  PerfTimer perf_env_cpu;
  PerfTimer perf_to_device_copy; // Copy obs to GPU.
  PerfTimer perf_lstm_forward;
  PerfTimer perf_post_batch_copy; // Copy all the results back to the passed in Tensors.
};

struct LSTMWrapper : torch::nn::Module
{
  // Per-eval batch size (# of envs / batch) and count (# of batches).
  int eval_batch_size;
  int eval_batch_count;
  int num_cuda_streams;
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

    torch::globalContext().setDeterministicCuDNN(false);

    // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090) (copied from pufferlib)
    torch::globalContext().setAllowTF32CuBLAS(true);
    torch::globalContext().setAllowTF32CuDNN(true);


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
    eval_batch_count = std::max(1, std::min(num_envs, opt->num_gpu_batches));
    eval_batch_size = (num_envs + eval_batch_count - 1) / eval_batch_count;

#if PUFFER_CUDA
    num_cuda_streams = std::min(global_max_num_cuda_streams, eval_batch_count * opt->bptt_horizon);
#endif
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
    c_print_tensor_infos(to, from, "to (1) <- from (2)");

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
    // Only allocate at the beginning of a BPTT horizon (one epoch). Free all tensors before training.
    BEGIN_LIBTORCH_CATCH
    {
      torch::NoGradGuard no_grad;
      env_states = new PufferBatchState*[eval_batch_count];
      for (int i = 0; i < eval_batch_count; i++)
      {
        auto* state = (env_states[i] = new PufferBatchState());
        const int start_idx = i * eval_batch_size;
        int env_count = eval_batch_size;
        if (i == eval_batch_count - 1)
        {
          env_count = num_envs - start_idx;
        }
        state->batch_index = i;
        state->env_start_index = start_idx;
        state->env_count = env_count;
        // For 'fat' envs, we could go as low as 1 env per thread if needed. So for now, 2 is a good sweet spot.
        state->min_num_envs_per_batch = 2; 
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

      encoder_bias = encoder_linear->bias.unsqueeze(1);
      decoder_bias = decoder->bias.unsqueeze(1);
      value_bias = value->bias.unsqueeze(1);
#if defined(PUFFER_CUDA)
      //Tensor out = torch::zeros({},
      //                          torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
#endif
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
        state->obs_cpu = full_obs_cpu.narrow(0, state->env_start_index, state->env_count).requires_grad_(false);
        state->rewards_cpu = full_rewards_cpu.narrow(0, state->env_start_index, state->env_count).requires_grad_(false);
        state->terminals_cpu = full_terminals_cpu.narrow(0, state->env_start_index, state->env_count).
                                                  requires_grad_(false);
        state->actions_cpu = Tensor{};
        alloc_tensor_arr(&state->values_horizon);
        alloc_tensor_arr(&state->logprob_horizon);
        alloc_tensor_arr(&state->rewards_horizon);
        alloc_tensor_arr(&state->actions_horizon);
        alloc_tensor_arr(&state->terminals_horizon);


        // H/C state is tracked per batch across segments for the current horizon.
        state->h1 = torch::zeros({state->env_count, opt->hidden_size},
          torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
        state->c1 = torch::zeros({state->env_count, opt->hidden_size},
          torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
        state->logits_entropy_unused = Tensor{};
        state->lstm_wrapper = this;
        state->vec_env = vec_env;

        state->perf_env_cpu = make_timer("env_cpu");
        state->perf_to_device_copy = make_timer("to_device_copy");
        state->perf_lstm_forward = make_timer("lstm_forward");
        state->perf_post_batch_copy = make_timer("post_batch_copy");
        state->logprobs_out = torch::zeros({state->env_count},
          torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();

        state->actions_out = torch::zeros(
          (opt->num_actions == 1
             ? at::IntArrayRef({state->env_count})
             : at::IntArrayRef({state->env_count, opt->num_actions})),
          torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();

#ifdef PUFFER_CUDA
        if (device.type() == torch::kCUDA)
        {
          cuda_streams = {};
          for (int j = 0; j < num_cuda_streams; j++)
          {
            // We have one CUDA stream per thread already (TLS based), however, that is not sufficient
            // as we want each segment to proceed independently. We use a pool of streams (so we don't really
            // need ( N * M ) streams for N batches and M segments - as it results in fragmentation/holding memory inside libtorch).
            cuda_streams.push_back(std::make_shared<CUDAStream>(getStreamFromPool(/*isHighPriority=*/true)));
          }
          // Output tensors for fused CUDA kernels.
          state->hidden_out = torch::zeros({opt->hidden_size, state->env_count},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
          // Double-buffer to prevent allocations: Use h1,c1 to generate h2,c2 for the next segment and vice versa (per batch).
          state->h2 = torch::zeros({state->env_count, opt->hidden_size},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
          state->c2 = torch::zeros({state->env_count, opt->hidden_size},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
          // LSTM stuff:
          // See RNN.cpp (usage of _thnn_fused_lstm_cell):
          //  igates = hidden {env_count, hidden_size } * w_ih.transpose() { hidden_size, input_size*4 } 
          //  = { env_count, input_size*4 }
          state->igates = torch::zeros({state->env_count, 4 * opt->input_size},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();

          // hgates = state->h1 { env_count, hidden_size } * w_hh.transpose() { hidden_size, input_size*4 } 
          //  = { env_count, input_size*4 }
          state->hgates = torch::zeros({state->env_count, 4 * opt->input_size},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();

          state->workspace =
              torch::empty({state->env_count, opt->hidden_size * 4},
                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();

          state->decoder_out = torch::zeros({state->env_count, opt->num_atns},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();

          state->values_out = torch::zeros({state->env_count, 1},
            torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
        }
#endif
      }
      perf_total_forward_eval = {.name = "total_forward_eval"};
    }
    END_LIBTORCH_CATCH
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
          state->values_horizon[seg] = Tensor{};
          state->logprob_horizon[seg] = Tensor{};
          state->rewards_horizon[seg] = Tensor{};
          state->terminals_horizon[seg] = Tensor{};
          state->actions_horizon[seg] = Tensor{};
        }
        calc_total_perf_duration(i, result, state->perf_env_cpu, opt->num_threads);
        calc_total_perf_duration(i, result, state->perf_to_device_copy, eval_batch_count);
        calc_total_perf_duration(i, result, state->perf_lstm_forward, eval_batch_count);
        calc_total_perf_duration(i, result, state->perf_post_batch_copy, eval_batch_count);
#if PUFFER_CUDA
        for (auto& stream : cuda_streams)
        {
          if (stream != nullptr) { stream->synchronize(); }
          stream = nullptr;
        }
        cuda_streams = {};
#endif

        state->obs_cpu = Tensor{};
        state->obs_device = Tensor{};
        state->rewards_cpu = Tensor{};
        state->actions_cpu = Tensor{};
        state->terminals_cpu = Tensor{};
        state->logits_entropy_unused = Tensor{};
        state->h1 = Tensor{};
        state->c1 = Tensor{};
        state->h2 = Tensor{};
        state->c2 = Tensor{};
        state->igates = Tensor{};
        state->hgates = Tensor{};
        state->workspace = Tensor{};
        state->decoder_out = Tensor{};
        state->hidden_out = Tensor{};
        state->values_out = Tensor{};
        state->actions_out = Tensor{};
        state->logprobs_out = Tensor{};
        DELETE_ARRAY(state->values_horizon);
        DELETE_ARRAY(state->logprob_horizon);
        DELETE_ARRAY(state->actions_horizon);
        DELETE_ARRAY(state->rewards_horizon);
        DELETE_ARRAY(state->terminals_horizon);

        // Prepare for next run.
        state->lstm_wrapper = nullptr;
      }
      result.stats_millis.push_back({perf_total_forward_eval.name, perf_total_forward_eval.get_duration_millis()});
      result.step_count = this->horizon_steps;
      result.total_steps = this->total_steps;
      vec_env = nullptr;
      final_obs = Tensor{};
      final_actions = Tensor{};
      final_logprobs = Tensor{};
      final_rewards = Tensor{};
      final_terminals = Tensor{};
      final_values = Tensor{};
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
      // We can start off with putting this whole thing in a for loop (i.e. each iteration, wait for all done) to begin
      // with. I think ideally, some stuff should just start going forward.
      c_add_work_batched(vec_env, run_next_bptt_segment, this, 0, eval_batch_count - 1,
        /* batch_completion*/ nullptr, /* min_num_items_per_batch */ 1);
      c_wait_all_done(vec_env);
      perf_total_forward_eval.stop();
    } END_LIBTORCH_CATCH
  }

private:
  void alloc_tensor_arr(Tensor** arr) const
  {
    *arr = new Tensor[opt->bptt_horizon];
    for (int i = 0; i < opt->bptt_horizon; i++) { (*arr)[i] = Tensor{}; }
  }


  [[nodiscard]] torch::nn::Linear layer_init(torch::nn::Linear layer, const double std = std::sqrt(2.0),
    const double bias_const = 0.0) const
  {
    torch::nn::init::orthogonal_(layer->weight, std);
    torch::nn::init::constant_(layer->bias, bias_const);
    return layer;
  }

#ifdef PUFFER_CUDA
  CUDAStream get_cuda_stream(const int batch_index, const int segment) const
  {
    if (num_cuda_streams == 0) { return getDefaultCUDAStream(); }
    auto stream_index = ((segment * eval_batch_count) + batch_index) % num_cuda_streams;
    // printf("---Using stream %d [S %d B %d]\n", stream_index, segment, batch_index);
    return *(cuda_streams[stream_index]);
  }
#endif

  static void run_next_bptt_segment(void* arg, int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      auto* this_ptr = static_cast<LSTMWrapper*>(arg);
      // We must do this per thread work as it's TLS guarded.
      torch::NoGradGuard no_grad;
      auto* state = this_ptr->env_states[batch_index];
      auto segment = state->bptt_segment.load();
      print_cuda_mem_info("bptt_segment_S" + std::to_string(segment) + "_B" + std::to_string(batch_index), false);
      if (segment >= this_ptr->opt->bptt_horizon) { return; }
      // printf(" Batch %d: Running BPTT segment %d / %d\n", batch_index, state->bptt_segment, opt->bptt_horizon);
      // Ok to perform synchronously as we need the obs tensor + forward eval before we can start env steps.
#ifdef PUFFER_CUDA
      if (this_ptr->device == torch::kCUDA && this_ptr->num_cuda_streams > 0)
      {
        // Choose one of the CUDA streams we have alloted to the segments in a round-robin fashion.
        auto stream = this_ptr->get_cuda_stream(batch_index, segment);
        CUDAStreamGuard guard(stream);
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
  void copy_to_final_buffers_async(PufferBatchState* state, int segment)
  {
    BEGIN_LIBTORCH_CATCH
    {
      torch::NoGradGuard no_grad;
      state->perf_post_batch_copy.start();

#ifdef PUFFER_CUDA
      if (device == torch::kCUDA && num_cuda_streams > 0)
      {
        {
          CUDAStreamGuard guard(get_cuda_stream(state->batch_index, segment));
          copy_to_final_buffers(state, segment);
        }
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
  void copy_to_final_buffers(PufferBatchState* state, const int segment)
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
      final_values.narrow(0, env_start, n).select(1, segment).copy_(state->values_horizon[segment], non_blocking);
      final_logprobs.narrow(0, env_start, n).select(1, segment).copy_(state->logprob_horizon[segment], non_blocking);
      final_rewards.narrow(0, env_start, n).select(1, segment).copy_(state->rewards_horizon[segment], non_blocking);
      final_terminals.narrow(0, env_start, n).select(1, segment).copy_(state->terminals_horizon[segment], non_blocking);
      final_actions.narrow(0, env_start, n).select(1, segment).copy_(state->actions_horizon[segment], non_blocking);
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
        print_cuda_mem_info("copy_obs_pre_S" + std::to_string(segment) + "_B" + std::to_string(batch_index), false);

        // NOTE: Env observations are memory mapped to the full_obs_cpu tensor already.
        // Once it's on device, changes are no longer reflected unless we copy again.
        const int64_t env_start = state->env_start_index;
        const int64_t n = state->env_count;
        state->obs_device = final_obs.narrow(0, env_start, n).select(1, segment);
        state->obs_device.copy_(state->obs_cpu, false);
        // Must copy blocking as the obs will be overwritten by the envs next.
        state->perf_to_device_copy.stop();
        print_cuda_mem_info("copy_obs_post_S" + std::to_string(segment) + "_B" + std::to_string(batch_index), false);
      }
#if PUFFER_CUDA
      cuda_batch_forward_eval(batch_index);
#else
      torch_batch_forward_eval(batch_index);
#endif
      run_envs(state);
    }
    END_LIBTORCH_CATCH
  }


  void proceed_to_next_batch(PufferBatchState* state)
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
      state->actions_cpu = Tensor{};
    }
    END_LIBTORCH_CATCH

    // Schedule this work for the next segment. (We could reuse this thread, but let's yield to 
    // let the OS manage the priorities naturally).
    auto segment = atomic_fetch_add(&state->bptt_segment, 1);

    // Queue up two work items:
    // 1) Copy to final buffers (async) for the previous segment.
    // 2) Run next BPTT segment forward eval for the next segment.
    c_add_work_batched(state->vec_env,
      [segment](void* arg, int _2)
      {
        auto* state = static_cast<PufferBatchState*>(arg);
        state->lstm_wrapper->copy_to_final_buffers_async(state, segment);
      },
      state, segment, segment, /* batch_completion_cb */ nullptr, /* min_num_items_per_batch */ 1);

    c_add_work_batched(state->vec_env, run_next_bptt_segment, state->lstm_wrapper,
      state->batch_index, state->batch_index, /* batch_completion_cb */ nullptr, /* min_num_items_per_batch */ 1);
  }

  //! @brief Async non-cuda multi-threaded forward eval pass for an entire batch of obs.
  void torch_batch_forward_eval(int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      RECORD_FUNCTION("batch_forward_eval", std::vector<c10::IValue>({static_cast<uint64_t>(batch_index)}));

      // We must do this per thread work as it's TLS guarded.
      torch::NoGradGuard no_grad;
      auto* state = env_states[batch_index];
      auto segment = state->bptt_segment.load();

      print_cuda_mem_info(
        "torch_batch_forward_eval_pre_S" + std::to_string(segment) + "_B" + std::to_string(batch_index), false);
      state->perf_lstm_forward.start();
      auto obs_tensor = state->obs_device;
      state->obs_device = Tensor{};

      auto hidden = encoder->forward(obs_tensor);

      // Non-fused, just copy h1/c1 over all the time, ignore h2/c2
      std::tuple<Tensor, Tensor> hc = lstm_cell->forward(hidden, std::make_tuple(state->h1, state->c1));
      hidden = Tensor{};
      state->h1 = std::get<0>(hc);
      state->c1 = std::get<1>(hc);
      if (opt->is_continuous)
      {
        PUFFER_ASSERT(!opt->is_continuous, "Only supports (multi)discrete for now.");
        throw std::runtime_error("Continuous action space not implemented yet.");
        // TODO(perumaal): Need to update state->logits as well and verify this with the puffernet impl.
      }
      else
      {
        Tensor logits = decoder->forward(state->h1);
        Tensor values = value->forward(state->h1);
        values = values.flatten();
        state->values_horizon[segment] = values;

        sample_logits(logits, opt->num_actions, opt->logit_sizes, state->actions_out, state->logprobs_out);
        state->logprob_horizon[segment] = state->logprobs_out;

        state->actions_horizon[segment] = state->actions_out;

        // Keep the actions on device, but use the CPU tensor below locally.
        // Copy and hold on to the actions (and rewards/terminals) until the batch env steps are done asynchronously.
        state->actions_cpu = state->actions_horizon[segment].to(torch::kCPU, /*non_blocking=*/false, /*copy=*/true,
          {c10::MemoryFormat::Contiguous});
      }

      state->perf_lstm_forward.stop();

      state->perf_env_cpu.start();
    }
    END_LIBTORCH_CATCH
  }

#if PUFFER_CUDA

  void cuda_batch_forward_eval(int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      RECORD_FUNCTION("batch_forward_eval", std::vector<c10::IValue>({static_cast<uint64_t>(batch_index)}));

      // We must do this per thread work as it's TLS guarded.
      torch::NoGradGuard no_grad;
      auto* state = env_states[batch_index];
      auto segment = state->bptt_segment.load();

      print_cuda_mem_info(
        "torch_batch_forward_eval_pre_S" + std::to_string(segment) + "_B" + std::to_string(batch_index), false);
      state->perf_lstm_forward.start();
      auto obs_tensor = state->obs_device;
      state->obs_device = Tensor{};


      at::_addmm_activation_out(state->hidden_out, encoder_bias, encoder_linear->weight,
        obs_tensor.transpose(0, 1), 1, 1, /*use_gelu*/ true);

      auto hidden_transposed = state->hidden_out.transpose(0, 1);

      // Use double-buffering to switch between h1/c1 and h2/c2.
      Tensor h1, c1, h2, c2;
      if ((segment % 2) == 0)
      {
        h1 = state->h1;
        c1 = state->c1;
        h2 = state->h2;
        c2 = state->c2;
      }
      else
      {
        h1 = state->h2;
        c1 = state->c2;
        h2 = state->h1;
        c2 = state->c1;
      }

      at::matmul_out(state->igates, hidden_transposed, lstm_cell->weight_ih.transpose(0, 1));
      at::matmul_out(state->hgates, h1, lstm_cell->weight_hh.transpose(0, 1));
      lstm_forward_impl(state->igates, state->hgates, lstm_cell->bias_ih, lstm_cell->bias_hh,
        c1, h2, c2, state->workspace);

      // Now the h2/c2 (mapped to state->h1/h2 and state->c1/c2 as needed) has the results.
      if (opt->is_continuous)
      {
        PUFFER_ASSERT(!opt->is_continuous, "Only supports (multi)discrete for now.");
        throw std::runtime_error("Continuous action space not implemented yet.");
        // TODO(perumaal): Need to update state->logits as well and verify this with the puffernet impl.
      }
      else
      {
        launch_linear_forward(h2, decoder->weight, decoder_bias, state->decoder_out,
          get_cuda_stream(state->batch_index, segment));
        auto logits = state->decoder_out;

        launch_linear_forward(state->h1, value->weight, value->bias, state->values_out,
          get_cuda_stream(state->batch_index, segment));
        c_print_tensor_info(state->values_out, "state->values_out");
        auto values = state->values_out.flatten();

        state->values_horizon[segment] = values;

        {
          constexpr int COUNT = 1;
          auto t1 = start_timer_laps("sample_logits", COUNT);
          for (int i = 0; i < COUNT; i++)
          {
            sample_logits(logits, opt->num_actions, opt->logit_sizes, state->actions_out, state->logprobs_out);
            t1.lap();
          }
          t1.stop(); //.print(COUNT);
        }
        state->logprob_horizon[segment] = state->logprobs_out;

        state->actions_horizon[segment] = state->actions_out;

        // Keep the actions on device, but use the CPU tensor below locally.
        // Copy and hold on to the actions (and rewards/terminals) until the batch env steps are done asynchronously.
        state->actions_cpu = state->actions_horizon[segment].to(torch::kCPU, /*non_blocking=*/false, /*copy=*/true,
          {c10::MemoryFormat::Contiguous});
        print_cuda_mem_info(
          "torch_batch_forward_eval_post_S" + std::to_string(segment) + "_B" + std::to_string(batch_index), true, {
          {"h", state->h},
          {"c", state->c},
          {"logits", logits},
          {"values", values},
          {"state->actions_cpu", state->actions_cpu},
          {"state->rewards_cpu", state->rewards_cpu},
          {"state->terminals_cpu", state->terminals_cpu},
          {"state->obs_device", obs_tensor},
          {"state->values_horizon_s", state->values_horizon[segment]},
          {"final_obs", final_obs}
          });
      }

      state->perf_lstm_forward.stop();

      state->perf_env_cpu.start();
    }
    END_LIBTORCH_CATCH
  }
#endif

  void run_envs(PufferBatchState* state)
  {
    const auto segment = state->bptt_segment.load();

    // Run a batch of env steps independently on different threads.
    // Once all envs from this batch have completed, proceed to run the next BPTT segment.
    auto num_actions = opt->num_actions;
    // All these arrays are valid until the env step is done. The next segment for this batch won't
    // proceed until after.
    auto* rewards_arr = static_cast<float*>(state->rewards_cpu.data_ptr());
    auto* terminals_arr = static_cast<float*>(state->terminals_cpu.data_ptr());
    auto* actions_arr = static_cast<int*>(state->actions_cpu.data_ptr());
    const int env_start_index = state->env_start_index;
    c_add_work_batched(vec_env,
      [num_actions, rewards_arr, terminals_arr, actions_arr, env_start_index](void* envs, int env_index)
      {
        c_step_batch(envs, env_index, (env_index - env_start_index), actions_arr, num_actions, rewards_arr,
          terminals_arr);
      }, state->vec_env->envs, state->env_start_index,
      state->env_start_index + state->env_count - 1,
      [state, segment](void* _) // Unused as it's per-env, we need the batch captured state.
      {
        auto this_ptr = state->lstm_wrapper;
#ifdef PUFFER_CUDA
        if (this_ptr->device == torch::kCUDA && this_ptr->num_cuda_streams > 0)
        {
          {
            CUDAStreamGuard guard(this_ptr->get_cuda_stream(state->batch_index, segment));
            this_ptr->proceed_to_next_batch(state);
          }
        }
        else // fallthrough
#endif
        {
          this_ptr->proceed_to_next_batch(state);
        }
      }, /* min_num_items_per_batch */ state->min_num_envs_per_batch);
  }

private:
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
  Tensor decoder_logstd{nullptr};

  // LSTM Policy on top of the encoder/decoder above.
  torch::nn::LSTMCell lstm_cell{nullptr};
  torch::Device device = torch::kCPU;


  // These may be accessed from any thread during eval.
  PufferBatchState** env_states;
  VecEnv* vec_env;
  Tensor final_obs, final_actions, final_logprobs, final_rewards, final_terminals, final_values;

  Tensor encoder_bias, decoder_bias, value_bias;
  PerfTimer perf_total_forward_eval;

#ifdef PUFFER_CUDA
  // Using shared_ptr since there isn't a default constructor; plus avoids having a lock for the stream itself.
  // Stream 1 for copying obs to device and forward eval.
  std::vector<std::shared_ptr<CUDAStream>> cuda_streams;
#endif
};


// Separate out the API stuff from this.
#include <puffer_api.cpp>
