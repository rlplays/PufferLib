#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include "puffer_native_eval.h"

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

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
using namespace ::c10::cuda;

// Enable multiple streams per batch by default. 2 means double-buffering etc.
// Very useful doc: https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management
// Set to 0 to disable multiple cuda streams (and instead use the default TLS one).
constexpr int global_max_num_cuda_streams = 32;

#include "puffer_threads.h"
#include "puffer_utils.h"

struct LSTMWrapper;
struct LSTMPolicyModule; // This is the old Torch-based NN Module - not used anymore.

//! @brief Holds the state for a batch of envs.
struct PufferBatchState
{
  // Note: All Tensors are on-device unless that have a _cpu suffix.
  //       _out suffix means preallocated output tensors.
  //       _horizon suffix means intermediate storage per-step across the horizon
  //       (helps avoid `narrow`/`select` calls in hotpath).

  // Batch index within the envs.
  int batch_index;
  // The envs within this batch.
  int env_start_index;
  int env_count;
  int min_num_envs_per_batch;
  // The following can be released after a segment is processed.
  Tensor obs_cpu, obs_device;
  Tensor actions_cpu;
  Tensor rewards_cpu, terminals_cpu;

  // Stores the intermediate segments across a horizon for copying into the out tensors.
  // One set of threads write to the arr[bptt_segment] while the other thread reads/copies over the tensors.
  Tensor *values_horizon, *logprob_horizon, *actions_horizon, *terminals_horizon, *rewards_horizon, *obs_horizon;
  Tensor* random_vals_horizon;

  Tensor values_horizon_out, logprob_horizon_out, actions_horizon_out;
  Tensor random_vals_horizon_in;

  // Output tensors preallocated to avoid cuda malloc / stream synchronization overhead.
  // Forward pass - encoder output.
  Tensor hidden_out, hidden_transposed_out;
  // Forward pass - LSTM output (/input)
  // Double-buffer h1/c1 <-> h2/c2 to avoid cudaMallocs/stream syncs. Each batch proceeds linearly
  // where segment1 uses h1/c1 to generate h2/c2, segment2 uses h2/c2 to generate h1/c1 etc.
  Tensor h1, c1;
  Tensor h2, c2;

  // For our custom LSTM kernel.
  Tensor igates, hgates, workspace;
  Tensor decoder_out;

  // Global params for quick referencing.
  LSTMWrapper* lstm_wrapper;
  atomic_int bptt_segment;

  VecEnv* vec_env;
  PerfTimer perf_env_cpu;
  PerfTimer perf_to_device_copy; // Copy obs to GPU.
  PerfTimer perf_lstm_forward;
};

// Legacy torch::nn::Module-based policy for reference. Not used.
#include <puffer_policy.cpp>

struct LSTMWrapper
{
public:
  // Per-eval batch size (# of envs / batch) and count (# of batches).
  int eval_batch_size;
  int eval_batch_count;
  int num_cuda_streams;
  PufferOptions* opt{nullptr};
  int num_envs;

public:
  LSTMWrapper(VecEnv* vec_env, PufferOptions* opt, int num_envs) : opt(opt), num_envs(num_envs)
  {
    if (!torch::cuda::is_available()) { throw runtime_error("LSTMWrapper requires CUDA device."); }
    BEGIN_LIBTORCH_CATCH
    {
      torch::globalContext().setBenchmarkCuDNN(false);
      torch::globalContext().setDeterministicCuDNN(false);

      // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090)
      torch::globalContext().setAllowTF32CuBLAS(true);
      torch::globalContext().setAllowTF32CuDNN(true);

      // Enable faster FP16 reductions
      torch::globalContext().setAllowFP16ReductionCuBLAS(true);

      // BF16 reduction (if using bfloat16)
      torch::globalContext().setAllowBF16ReductionCuBLAS(true);

      // Enable memory history recording for detailed snapshots
#if PUFFER_CUDA_MEMCHECK
      CUDACachingAllocator::recordHistory(true, nullptr, 1024 * 1024 * 100, CUDACachingAllocator::RecordContext::NEVER, true);
#endif
      torch::NoGradGuard no_grad;
      device = torch::kCUDA;
      encoder_linear_weight = torch::zeros({opt->input_size, opt->obs_size}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      encoder_linear_bias = torch::zeros({opt->hidden_size, 1}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
      if (opt->is_continuous) { throw runtime_error("Continuous action spaces not yet supported in native LSTMWrapper."); }
      else
      {
        opt->num_atns = 0;
        vector<int64_t> sizes_vec(opt->num_actions);
        vector<int64_t> offsets_vec(opt->num_actions);
        int64_t cumulative = 0;

        for (int i = 0; i < opt->num_actions; i++)
        {
          // TODO(perumaal): No padding/etc for now, all logits must be the same size.
          PUFFER_ASSERT(opt->logit_sizes[i] > 0 && opt->logit_sizes[i] == opt->logit_sizes[0],
            "Logit sizes must be > 0 and must be all have the same number of logits.");
          opt->num_atns += opt->logit_sizes[i];
          sizes_vec[i] = opt->logit_sizes[i];
          offsets_vec[i] = cumulative;
          cumulative += opt->logit_sizes[i];
        }
        decoder_weight = torch::zeros({opt->num_atns, opt->hidden_size}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        decoder_bias = torch::zeros({opt->num_atns}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        value_weight = torch::zeros({1, opt->hidden_size}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        value_bias = torch::zeros({1}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        lstm_cell_weight_ih = torch::zeros({4 * opt->hidden_size, opt->input_size}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        lstm_cell_weight_hh = torch::zeros({4 * opt->hidden_size, opt->hidden_size}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        lstm_cell_bias_ih = torch::zeros({4 * opt->hidden_size}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        lstm_cell_bias_hh = torch::zeros({4 * opt->hidden_size}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        logits_sizes_gpu = torch::from_blob(sizes_vec.data(), {opt->num_actions}, torch::kInt64).clone().to(torch::kCUDA).contiguous();
        logits_offsets_gpu = torch::from_blob(offsets_vec.data(), {opt->num_actions}, torch::kInt64).clone().to(torch::kCUDA).contiguous();
      }
      eval_batch_count = max(1, min(num_envs, opt->num_gpu_batches));
      eval_batch_size = (num_envs + eval_batch_count - 1) / eval_batch_count;
      num_cuda_streams = min(global_max_num_cuda_streams, eval_batch_count);
      this->vec_env = vec_env;
      alloc_tensors();
      printf("[Eanbled native multithreading/libtorch eval: %d envs on %d threads (batch size = max %d envs/batch; "
        "total %d batches/batch threads)%s (%d cuda streams) (Torch %s)]\n",
        vec_env->num_envs, opt->num_threads_env, eval_batch_size, eval_batch_count,
        (global_debug_mode ? " [Debug Mode]" : " [Release Mode]"), num_cuda_streams, TORCH_VERSION);
    }
    END_LIBTORCH_CATCH
  }

  ~LSTMWrapper() { dealloc_tensors(); }

  inline void alloc_tensors()
  {
    env_states = new PufferBatchState*[eval_batch_count];
    int max_batch_size = 0;
    for (int i = 0; i < eval_batch_count; i++)
    {
      auto* state = (env_states[i] = new PufferBatchState());
      const int start_idx = i * eval_batch_size;
      int env_count = eval_batch_size;
      if (i == eval_batch_count - 1)
      {
        env_count = num_envs - start_idx;
      }
      max_batch_size = max(env_count, max_batch_size);
      state->batch_index = i;
      state->env_start_index = start_idx;
      state->env_count = env_count;
      // For 'fat' envs, we could go as low as 1 env per thread if needed. So for now, 2 is a good sweet spot.
      state->min_num_envs_per_batch = 2;
    }
    full_random_vals = torch::zeros({eval_batch_count, opt->bptt_horizon, max_batch_size}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false);
    for (int i = 0; i < eval_batch_count; i++)
    {
      auto* state = env_states[i];
      state->bptt_segment = 0;

      // Per-batch/per-bptt-segment slices.
      state->obs_device = Tensor{};
      alloc_tensor_arr(&state->values_horizon);
      alloc_tensor_arr(&state->logprob_horizon);
      alloc_tensor_arr(&state->actions_horizon);
      alloc_tensor_arr(&state->terminals_horizon);
      alloc_tensor_arr(&state->rewards_horizon);
      alloc_tensor_arr(&state->obs_horizon);
      alloc_tensor_arr(&state->random_vals_horizon);

      for (int segment = 0; segment < opt->bptt_horizon; segment++)
      {
        state->random_vals_horizon[segment] = torch::rand({state->env_count}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
      }

      // TODO(perumaal): Now that we have granular h/c we can trigger a reset upon `c_reset`.
      // H/C state is tracked per batch across segments for the current horizon.
      state->h1 = torch::zeros({state->env_count, opt->hidden_size}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->c1 = torch::zeros({state->env_count, opt->hidden_size}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->lstm_wrapper = this;
      state->vec_env = vec_env;


      cuda_streams = {};
      for (int j = 0; j < num_cuda_streams; j++)
      {
        // We have one CUDA stream per thread already (TLS based), however, that is not sufficient as we want each segment to proceed independently. 
        // We use a pool of streams (so we don't really need ( N * M ) streams for N batches and M segments - as it results in fragmentation/holding 
        // memory inside libtorch).
        cuda_streams.push_back(make_shared<CUDAStream>(getStreamFromPool(/*isHighPriority=*/true)));
      }
      // Output tensors for fused CUDA kernels.
      state->hidden_out = torch::zeros({state->env_count, opt->hidden_size}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->hidden_transposed_out = state->hidden_out.transpose(0, 1);
      PUFFER_ASSERT(state->hidden_transposed_out.data_ptr() == state->hidden_out.data_ptr(), "Should not realloc hidden_out.");

      // Double-buffer to prevent allocations: Use h1,c1 to generate h2,c2 for the next segment and vice versa (per
      // batch).
      state->h2 = torch::zeros({state->env_count, opt->hidden_size}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->c2 = torch::zeros({state->env_count, opt->hidden_size}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->igates = torch::zeros({state->env_count, 4 * opt->input_size}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->hgates = torch::zeros({state->env_count, 4 * opt->input_size},torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->workspace = torch::empty({state->env_count, opt->hidden_size * 4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->decoder_out = torch::zeros({state->env_count, opt->num_atns},  torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32)).requires_grad_(false).contiguous();
      state->actions_cpu = torch::zeros((opt->num_actions == 1 ? at::IntArrayRef({state->env_count}) : at::IntArrayRef({state->env_count, opt->num_actions})), torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32)).requires_grad_(false).contiguous().pin_memory();
    }
  }

  inline void dealloc_tensors()
  {
    for (auto& stream : cuda_streams) 
    { 
      if (stream != nullptr) { stream->synchronize(); }
      stream = nullptr;
    }
    cuda_streams = {};
    for (int i = 0; i < eval_batch_count; i++)
    {
      auto* state = env_states[i];
      for (int seg = 0; seg < opt->bptt_horizon; seg++)
      {
        state->values_horizon[seg] = Tensor{};
        state->logprob_horizon[seg] = Tensor{};
        state->actions_horizon[seg] = Tensor{};
        state->rewards_horizon[seg] = Tensor{};
        state->terminals_horizon[seg] = Tensor{};
        state->random_vals_horizon[seg] = Tensor{};
        state->obs_horizon[seg] = Tensor{};
      }
      DELETE_ARRAY(state->values_horizon);
      DELETE_ARRAY(state->logprob_horizon);
      DELETE_ARRAY(state->actions_horizon);
      DELETE_ARRAY(state->terminals_horizon);
      DELETE_ARRAY(state->random_vals_horizon);
      DELETE_ARRAY(state->obs_horizon);
      DELETE_ARRAY(state->rewards_horizon);
      DELETE_PTR(env_states[i]);
    }
    DELETE_ARRAY(env_states);
    env_states = nullptr;
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
      ++epoch;

      c_setup_log(vec_env);
      this->horizon_steps = 0;
      Tensor enc_bias = encoder_linear_bias.squeeze(1);

      // Try to get weights from training wrapper first (if native training is enabled).
      if (!assign_training_weights(
        vec_env, encoder_linear_weight, enc_bias, decoder_weight, decoder_bias, value_weight,
        value_bias, lstm_cell_weight_ih, lstm_cell_weight_hh, lstm_cell_bias_ih, lstm_cell_bias_hh, encoder_linear_w,
        encoder_linear_b, decoder_linear_w, decoder_linear_b, value_w, value_b, weight_ih, weight_hh, bias_ih, bias_hh))
      {
        throw runtime_error("Failed to assign training weights to LSTMWrapper.");
      }
            
      weight_ih_transposed = lstm_cell_weight_ih.transpose(0, 1).contiguous();
      weight_hh_transposed = lstm_cell_weight_hh.transpose(0, 1).contiguous();
      // print_tensors(encoder_linear->weight, encoder_linear->bias, "encoder_linear w and b", true);
      // print_tensors(decoder->weight, decoder->bias, "decoder_linear w and b", true);
      // print_tensors(value->weight, value->bias, "value w and b", true);
      PUFFER_ASSERT(actions_out.dtype() == torch::kInt32, "Actions must be of discrete int32 dtype.");

      PUFFER_ASSERT(obs_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon, opt->obs_size}), "Obs tensor size mismatch.");
      if (opt->num_actions == 1)
      {
        PUFFER_ASSERT(actions_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}), "Actions (discrete) tensor size mismatch.");
      }
      else
      {
        PUFFER_ASSERT(actions_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon, opt->num_actions}), "Actions (multidiscrete) tensor size mismatch.");
      }
      PUFFER_ASSERT(logprobs_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}), "logprobs tensor size mismatch.");
        PUFFER_ASSERT(rewards_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}), "rewards tensor size mismatch.");
      PUFFER_ASSERT(terminals_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}), "terminals tensor size mismatch.");
      PUFFER_ASSERT(values_out.sizes() == at::IntArrayRef({vec_env->num_envs, opt->bptt_horizon}), "values tensor size mismatch.");
      final_obs = obs_out;
      final_actions = actions_out;
      final_logprobs = logprobs_out;
      final_rewards = rewards_out;
      final_terminals = terminals_out;
      final_values = values_out;

      // uniform has a significant overhead. 5us per call (2080RTX cuda12.9).
      // So just initialize one large array and use it for all batches/segments.
      //    e.g. 64 segments*8 batches = 512 calls to uniform_ per epoch. 512*5us=2.5ms overhead.
      full_random_vals.uniform_(0.0, 1.0);
      for (int batch_idx = 0; batch_idx < eval_batch_count; batch_idx++)
      {
        auto* state = env_states[batch_idx];
        add_work_batched(
          vec_env, [state, full_obs_cpu, full_rewards_cpu, full_terminals_cpu](void* this_ptr, int _)
          {
            static_cast<LSTMWrapper*>(this_ptr)->setup_batch(state, full_obs_cpu, full_rewards_cpu, full_terminals_cpu);
          },
          this, batch_idx, batch_idx, nullptr, 1, PufferWorkType::BatchWork);
      }
      c_wait_all_done(vec_env);
      // Wait for the copies to finish before we proceed.
      getDefaultCUDAStream().synchronize();
      perf_total_forward_eval = {.name = "total_forward_eval"};
    } END_LIBTORCH_CATCH
  }

  //! @brief Setup the per-batch state for the next BPTT segment.
  void setup_batch(PufferBatchState* state, Tensor full_obs_cpu, Tensor full_rewards_cpu, Tensor full_terminals_cpu)
  {
    BEGIN_LIBTORCH_CATCH
    {
      torch::NoGradGuard no_grad;
      state->bptt_segment = 0;

      // Per-batch/per-bptt-segment slices.
      state->obs_device = Tensor{};
      state->obs_cpu = full_obs_cpu.narrow(0, state->env_start_index, state->env_count);
      state->rewards_cpu = full_rewards_cpu.narrow(0, state->env_start_index, state->env_count);
      state->terminals_cpu = full_terminals_cpu.narrow(0, state->env_start_index, state->env_count);
      PUFFER_ASSERT(state->obs_cpu.is_pinned() && state->obs_cpu.is_contiguous(), "Input obs tensor must be pinned memory / contiguous for async copy.");
      PUFFER_ASSERT(state->rewards_cpu.is_pinned() && state->rewards_cpu.dtype() == torch::kFloat32 && state->rewards_cpu.is_contiguous(), "Input rewards tensor must be pinned memory / float32 / contiguous for async copy.");
      PUFFER_ASSERT(state->terminals_cpu.is_pinned() && state->terminals_cpu.dtype() == torch::kFloat32 && state->terminals_cpu.is_contiguous(), "Input terminals tensor must be pinned memory / float32 / contiguous for async copy.");
      // Each narrow call is 1us on a 2080RTX cuda 12.9. 64 segments * 8 batches (e.g.) is a lot;
      // instead just use per-batch narrow, then per-segment select.
      auto batch_rnd = full_random_vals.select(0, state->batch_index);
      const int64_t env_start = state->env_start_index;
      const int64_t n = state->env_count;
      auto batch_values = final_values.narrow(0, env_start, n);
      auto batch_logprob = final_logprobs.narrow(0, env_start, n);
      auto batch_actions = final_actions.narrow(0, env_start, n);
      auto batch_rewards = final_rewards.narrow(0, env_start, n);
      auto batch_terminals = final_terminals.narrow(0, env_start, n);
      auto batch_obs = final_obs.narrow(0, env_start, n);
      for (int seg_idx = 0; seg_idx < opt->bptt_horizon; seg_idx++)
      {
        // TODO(perumaal): Evaluate AoS vs SoA here as the narrow/select may result in large strides (?)
        //                 GPU L2 cache friendliness matters here (arrange [segments, batch] instead of the other way
        //                 round?) `train` may require it the other way round though.
        state->values_horizon[seg_idx] = batch_values.select(1, seg_idx);
        state->logprob_horizon[seg_idx] = batch_logprob.select(1, seg_idx);
        state->actions_horizon[seg_idx] = batch_actions.select(1, seg_idx);
        state->rewards_horizon[seg_idx] = batch_rewards.select(1, seg_idx);
        state->terminals_horizon[seg_idx] = batch_terminals.select(1, seg_idx);
        state->obs_horizon[seg_idx] = batch_obs.select(1, seg_idx);
        // Reinitialize random values so we get fresh set per epoch. Much cheaper than having to rand() PER segment PER
        // env PER action!
        state->random_vals_horizon[seg_idx] = batch_rnd.select(0, seg_idx);
        if (batch_rnd.size(1) > n)
        {
          // Narrow only when needed, it's expensive per call.
          state->random_vals_horizon[seg_idx] = state->random_vals_horizon[seg_idx].narrow(0, 0, n);
        }
      }

      // H/C state is tracked per batch across segments for the current horizon.
      // TODO: Do ablation : We don't have to zero here as we zero after a terminal reset.
       state->h1.zero_();
       state->c1.zero_();
       state->h2.zero_();
       state->c2.zero_();

      const int num_perf_laps = min(4, opt->bptt_horizon / 4);
      state->perf_env_cpu = make_timer("env_cpu", num_perf_laps);
      state->perf_to_device_copy = make_timer("to_device_copy", num_perf_laps);
      state->perf_lstm_forward = make_timer("lstm_forward", num_perf_laps);
    }
    END_LIBTORCH_CATCH
  }

  //! @brief Returns all the tensors (on target device) plus stats across all batches.
  PufferEvalResult finish_batch_eval_lstm(VecEnv* env)
  {
    PufferEvalResult result;

    BEGIN_LIBTORCH_CATCH
    {
      for (auto& stream : cuda_streams) { if (stream != nullptr) { stream->synchronize(); } }

      for (int i = 0; i < eval_batch_count; i++)
      {
        auto* state = env_states[i];
        calc_total_perf_duration(i, result, state->perf_env_cpu, opt->num_threads_env);
        calc_total_perf_duration(i, result, state->perf_to_device_copy, eval_batch_count);
        calc_total_perf_duration(i, result, state->perf_lstm_forward, eval_batch_count);
      }
      result.perf_stats.push_back(
        {perf_total_forward_eval.name, 1, perf_total_forward_eval.get_duration_millis(), {}, {}, {}});
      result.step_count = this->horizon_steps;
      result.total_steps = this->total_steps;
      final_obs = Tensor{};
      final_actions = Tensor{};
      final_logprobs = Tensor{};
      final_rewards = Tensor{};
      final_terminals = Tensor{};
      final_values = Tensor{};
    }
    END_LIBTORCH_CATCH
    return result;
  }

  //! @brief Batched env forward eval. This starts the process per segment in the horizon. Waits for all segments to finish and
  //! then return the batched tensor set back.
  void forward_eval_batch(VecEnv* vec_env)
  {
    BEGIN_LIBTORCH_CATCH
    {
      torch::NoGradGuard no_grad;
      // This is effectively useless as all the work is done in other threads, but keep it for safety.
      perf_total_forward_eval.start();
      num_batches_done = 0;
      c_start_work(vec_env);
      // Start with segment 0 for each batch. Once each one is done, it will enqueue the next segment
      // until all segments are done.
      add_work_batched(vec_env, run_next_bptt_segment, this, 0, eval_batch_count - 1, /* batch_completion*/ nullptr, /* min_num_items_per_batch */ 1, PufferWorkType::BatchWork);

      // Note because different threads may enqueue work, the queue(s) might be empty intermittently,
      // so the c_wait_all_done may exit prematurely...
      c_wait_all_done(vec_env);

      // ...so block the main thread and wait here until the batches are done.
      {
        unique_lock lock(done_batches_mutex);
        while (num_batches_done != eval_batch_count) { done_batches.wait(lock); }
      }

      perf_total_forward_eval.stop();
    }
    END_LIBTORCH_CATCH
  }

  void alloc_tensor_arr(Tensor** arr) const
  {
    *arr = new Tensor[opt->bptt_horizon];
    for (int i = 0; i < opt->bptt_horizon; i++)
    {
      (*arr)[i] = Tensor{};
    }
  }

  CUDAStream get_cuda_stream(const int batch_index) const
  {
    if (num_cuda_streams == 0) { return getDefaultCUDAStream(); }
    auto stream_index = (batch_index) % num_cuda_streams;
    // printf("---Using stream %d [S %d B %d]\n", stream_index, segment, batch_index);
    return *(cuda_streams[stream_index]);
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
      print_cuda_mem_info("bptt_segment_S" + to_string(segment) + "_B" + to_string(batch_index), false);
      if (segment == this_ptr->opt->bptt_horizon)
      {
        auto count = this_ptr->num_batches_done.fetch_add(1);
        if (count + 1 == this_ptr->eval_batch_count)
        {
          // Only lock when the last segment finishes.
          unique_lock lock(this_ptr->done_batches_mutex);
          // printf("Batch %d: Done with all segments! Notifying main thread. \n", batch_index);
          this_ptr->done_batches.notify_one();
        }
        return;
      }
      // printf(" Batch %d: Running BPTT segment %d / %d\n", batch_index, state->bptt_segment, opt->bptt_horizon);
      {
        auto stream = this_ptr->get_cuda_stream(batch_index);
        CUDAStreamGuard guard(stream);
        stream.synchronize();
        this_ptr->copy_obs_forward_eval_batch(batch_index);
        this_ptr->run_envs(state);
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
      auto stream = get_cuda_stream(state->batch_index);
      CUDAStreamGuard guard(stream);
      const auto segment = state->bptt_segment.load();
      {
        state->perf_to_device_copy.start();
        // printf("batch obs copy: B %d S %d \n", batch_index, state->bptt_segment.load());
        print_cuda_mem_info("copy_obs_pre_S" + to_string(segment) + "_B" + to_string(batch_index), false);

        // Kickoff rewards/terminals from the previous run to device copy while we do the obs copy.
        state->rewards_horizon[segment].copy_(state->rewards_cpu, /*non_blocking*/ true);
        state->terminals_horizon[segment].copy_(state->terminals_cpu, /*non_blocking*/ true);
        state->obs_device = state->obs_horizon[segment];
        state->obs_device = state->obs_device.copy_(state->obs_cpu, /*non_blocking*/ true).transpose(0, 1);
        stream.synchronize();

        // Must copy blocking as the obs will be overwritten by the envs next.
        state->perf_to_device_copy.stop();
        print_cuda_mem_info("copy_obs_post_S" + to_string(segment) + "_B" + to_string(batch_index), false);
      }

      state->perf_lstm_forward.start();

      state->random_vals_horizon_in = state->random_vals_horizon[segment];
      state->values_horizon_out = state->values_horizon[segment].unsqueeze(1);
      state->logprob_horizon_out = state->logprob_horizon[segment];
      state->actions_horizon_out = state->actions_horizon[segment];

      cuda_batch_forward_eval(batch_index);
      // Just reverse LSTM states (double buffering).
      swap(state->h1, state->h2);
      swap(state->c1, state->c2);
      // The values_horizon, actions_horizon, logprob_horizon are memory mapped tensors already, so no need to copy here.

      // Keep the actions on device, but use the CPU tensor below locally (and we shouldn't have to wait for this copy).
      state->actions_cpu.copy_(state->actions_horizon[segment], /* non_blocking */ true);
      state->perf_lstm_forward.stop();

      // Must wait for the actions to be present fully before we proceed to run the envs.
      stream.synchronize();
    }
    END_LIBTORCH_CATCH
  }


  void cuda_batch_forward_eval(int batch_index)
  {
    BEGIN_LIBTORCH_CATCH
    {
      // We must do this per thread work as it's TLS guarded.
      torch::NoGradGuard no_grad;
      auto* state = env_states[batch_index];
      // NOTE: This uses GELU approximations so the values do not match the standard encoder->forward exactly.
      //       Error is about ~10e-3. Verified via tests and full e2e train perf scores that this is acceptable.
      //       Moreover,  the actual C code uses the same trick anyway.
      at::_addmm_activation_out(state->hidden_transposed_out, encoder_linear_bias, encoder_linear_weight, state->obs_device, 1, 1, /*use_gelu*/ true);
      at::matmul_out(state->igates, state->hidden_out, weight_ih_transposed);
      at::matmul_out(state->hgates, state->h1, weight_hh_transposed);

      // TODO(perumaal): Now that we have granular h/c we can trigger a reset selectively upon `c_reset` using a per-env mask.
      lstm_forward_impl(state->igates, state->hgates, lstm_cell_bias_ih, lstm_cell_bias_hh, state->c1, state->h2, state->c2, state->workspace);

      // Now the h2/c2 (mapped to state->h1/h2 and state->c1/c2 as needed) has the results.
      if (opt->is_continuous)
      {
        PUFFER_ASSERT(!opt->is_continuous, "Only supports (multi)discrete for now.");
        throw runtime_error("Continuous action space not implemented yet.");
        // TODO(perumaal): Need to update state->logits as well and verify this with the puffernet impl.
      }
      else
      {
        launch_dual_linear_forward(state->h2, decoder_weight, decoder_bias, state->decoder_out, value_weight, value_bias,state->values_horizon_out);
        launch_sample_logits_kernel(state->random_vals_horizon_in, logits_sizes_gpu, logits_offsets_gpu, state->decoder_out, opt->num_actions, 
          state->actions_horizon_out, state->logprob_horizon_out);
      }
    }
    END_LIBTORCH_CATCH
  }

  void run_envs(PufferBatchState* state)
  {
    const auto segment = state->bptt_segment.load();
    state->perf_env_cpu.start();

    // Run a batch of env steps independently on different threads.
    // Once all envs from this batch have completed, proceed to run the next BPTT segment.
    auto num_actions = opt->num_actions;

    // All these arrays are valid until the env step is done. The next segment for this batch won't
    // proceed until after.
    auto* rewards_arr = static_cast<float*>(state->rewards_cpu.data_ptr());
    auto* terminals_arr = static_cast<float*>(state->terminals_cpu.data_ptr());
    PUFFER_ASSERT(state->actions_cpu.dtype() == torch::kInt32, "Actions must be 32-bit int type.");
    auto* actions_arr = static_cast<int*>(state->actions_cpu.data_ptr());
    const int env_start_index = state->env_start_index;
    const int horizon_segment = state->bptt_segment.load();
    // Main env step threading work done on the EnvWork thread group independent of the batching work.
    add_work_batched(
      vec_env,
      [state, num_actions, rewards_arr, terminals_arr, actions_arr, env_start_index, horizon_segment](void* vec_env, int env_index)
      {
        auto local_env_index = env_index - env_start_index;
        c_step_batch(vec_env, env_index, local_env_index, actions_arr, num_actions, rewards_arr, terminals_arr, horizon_segment);

        // Clear RNN/LSTM states for envs that are in terminal state - before we proceed.
        // This is done in the same bg thread as the env so it won't impact other threads.
        // TODO: Measure perf w/wo this on breakout etc. With grid, it's worse. I am inclined to not do this yet.
        // if (terminals_arr[local_env_index] > 0.5f)
        // {
        //   state->h1[local_env_index].zero_();
        //   state->c1[local_env_index].zero_();
        // }
      },
      state->vec_env, state->env_start_index, state->env_start_index + state->env_count - 1,
      [state, segment](void* _) // Unused as it's per-env, we need the batch captured state.
      {
        state->perf_env_cpu.stop();
        state->bptt_segment.fetch_add(1);
        state->lstm_wrapper->total_steps += state->env_count;
        state->lstm_wrapper->horizon_steps += state->env_count;
        // Run next BPTT segment forward eval for the next segment.
        add_work_batched(state->vec_env, run_next_bptt_segment, state->lstm_wrapper, state->batch_index,
          state->batch_index, /* batch_completion_cb */ nullptr, /* min_num_items_per_batch */ 1,
          PufferWorkType::BatchWork);
      },
      /* min_num_items_per_batch */ state->min_num_envs_per_batch, PufferWorkType::EnvWork);
  }

  void sync_cuda_stream(const int batch_index)
  {
    auto stream = get_cuda_stream(batch_index);
    stream.synchronize();
  }

private:
  int64_t total_steps = 0;
  int64_t horizon_steps = 0;
  int epoch = 0;
  torch::Device device = torch::kCPU;

  // These may be accessed from any thread during eval.
  PufferBatchState** env_states;
  VecEnv* vec_env;
  Tensor final_obs, final_actions, final_logprobs, final_rewards, final_terminals, final_values;
  // Holds an entire (segments*envs) set of random values for sampling actions.
  Tensor full_random_vals;

  Tensor weight_ih_transposed, weight_hh_transposed;

  // Used by the sample_logits kernel
  Tensor logits_sizes_gpu, logits_offsets_gpu;

  Tensor encoder_linear_weight, encoder_linear_bias;
  Tensor decoder_weight, decoder_bias;
  Tensor value_weight, value_bias;
  Tensor lstm_cell_weight_ih, lstm_cell_weight_hh, lstm_cell_bias_ih, lstm_cell_bias_hh;
  PerfTimer perf_total_forward_eval;

  atomic_int num_batches_done = 0;
  mutex done_batches_mutex;
  condition_variable done_batches;
  // Using shared_ptr since there isn't a default constructor; plus avoids having a lock for the stream itself.
  // Stream 1 for copying obs to device and forward eval.
  vector<shared_ptr<CUDAStream>> cuda_streams;
};

// TODO(perumaal): "include"ing the CPP is terrible but that's the easiest way to keep everything in one place
// especially as setup.py is a bit finnicky to configure.
#include <puffer_native_train.cpp>

// Separate out the API stuff from this.
#include <puffer_api.cpp>
