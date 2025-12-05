#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include "puffer_nativecpp.h"

#include <torch/torch.h>
#include "puffernet.h"
#include <cassert>
#include <iostream>

using torch::Tensor;

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
void c_print_tensor_info(Tensor tensor)
{
  std::cout << "Tensor info:" << std::endl;
  std::cout << " - Device: " << tensor.device() << std::endl;
  std::cout << " - Dtype: " << tensor.dtype() << std::endl;
  std::cout << " - Size: " << tensor.sizes() << std::endl;
  std::cout << " - Stride: " << tensor.strides() << std::endl;
  std::cout << " - Is contiguous: " << (tensor.is_contiguous() ? "Yes" : "No") << std::endl;
}


void c_print_tensor_infos(Tensor tensor1, Tensor tensor2)
{
  std::cout << "Tensor 1 info:" << std::endl;
  c_print_tensor_info(tensor1);
  std::cout << "Tensor 2 info:" << std::endl;
  c_print_tensor_info(tensor2);
}

struct PufferEnvState
{
  // Full obs space across all envs, indexed for this particular env by i.
  Tensor full_obs;
  int index;
  // For the LSTM wrapper.
  Tensor h;
  Tensor c;
  Tensor values;
  Tensor logits;
  Tensor logprob;
  Tensor entropy;
  Tensor actions;
};

struct LSTMWrapper : torch::nn::Module
{
  LSTMWrapper(PufferOptions* opt) : opt(opt)
  {
    torch::NoGradGuard no_grad;
    device_ = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
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
  }

  [[nodiscard]] torch::nn::Linear layer_init(torch::nn::Linear layer, const double std = std::sqrt(2.0),
    const double bias_const = 0.0) const
  {
    torch::nn::init::orthogonal_(layer->weight, std);
    torch::nn::init::constant_(layer->bias, bias_const);
    return layer;
  }

  void weights_to_tensor(Weights* weights, const size_t num_weights, Tensor& tensor_to)
  {
    auto* arr = get_weights(weights, num_weights);
    PUFFER_ASSERT(num_weights == tensor_to.numel(), "Tensor does not match weights.");
    auto w = torch::from_blob(arr, {static_cast<int64_t>(num_weights)}, torch::kFloat32);
    if (w.device() == tensor_to.device() && w.dtype() == tensor_to.dtype())
    {
      tensor_to = tensor_to.set_(w.reshape(tensor_to.sizes()));
    }
    else
    {
      tensor_to = tensor_to.copy_(w.reshape(tensor_to.sizes()).to(tensor_to.device(), tensor_to.dtype()));
    }
  }

  void weights_to_linear(Weights* weights, const size_t input_dim, int output_dim, torch::nn::Linear& layer)
  {
    weights_to_tensor(weights, static_cast<uint64_t>(input_dim) * static_cast<uint64_t>(output_dim), layer->weight);
    weights_to_tensor(weights, static_cast<uint64_t>(output_dim), layer->bias);
  }

  void update_model_weights(Weights* weights)
  {
    PUFFER_ASSERT(weights != nullptr && opt != nullptr && opt->num_atns > 0, "Invalid input/state.");
    PUFFER_ASSERT(!opt->is_continuous, "Only supports multidiscrete for now.");
    torch::NoGradGuard no_grad;
    weights_to_linear(weights, opt->obs_size, opt->hidden_size, encoder_linear);
    weights_to_linear(weights, opt->hidden_size, opt->num_atns, decoder);
    weights_to_linear(weights, opt->hidden_size, 1, value);
    weights_to_tensor(weights, static_cast<uint64_t>(opt->hidden_size) * static_cast<uint64_t>(opt->input_size) * 4,
      lstm_cell->weight_ih);
    weights_to_tensor(weights, static_cast<uint64_t>(opt->hidden_size) * static_cast<uint64_t>(opt->input_size) * 4,
      lstm_cell->weight_hh);
    weights_to_tensor(weights, static_cast<uint64_t>(opt->hidden_size) * 4, lstm_cell->bias_ih);
    weights_to_tensor(weights, static_cast<uint64_t>(opt->hidden_size) * 4, lstm_cell->bias_hh);
    PUFFER_ASSERT(weights->idx == weights->size, "Must have used all weights exactly.");
  }

  // Single env forward eval
  void forward_eval_single_env(float* obs, int* actions)
  {
    auto state = env_states_[0];
    // Assumes obs_size_ for obs, and num_actions_ for actions_out already initialized.
    torch::NoGradGuard no_grad;
    auto obs_tensor = torch::from_blob(obs, {opt->obs_size}, torch::kFloat32).to(device_);
    auto hidden = encoder->forward(obs_tensor);
    auto hc = lstm_cell->forward(hidden.unsqueeze(0), std::make_tuple(state->h, state->c));
    auto h = std::get<0>(hc);
    auto c = std::get<1>(hc);
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
        actions[i] = static_cast<int>(action_sample[0][i].item<float>());
      }
      // TODO(perumaal): Need to update state->logits as well and verify this with the puffernet impl.
    }
    else
    {
      state->logits = decoder->forward(h);
      // Put into a tuple of num_actions tensors, each with N logits.
      // Shape after split and stack: [num_actions, 1, logit_size], squeeze to [num_actions, logit_size]
      auto split_logits = state->logits.split(at::IntArrayRef(opt->logit_sizes, opt->num_actions), /*dim=*/1);
      state->logits = torch::stack(split_logits, /*dim=*/0).squeeze(1);

      // Ensure 2D shape [num_actions, logit_size] for multinomial
      if (state->logits.dim() == 1)
      {
        state->logits = state->logits.unsqueeze(0);
      }

      auto normalized_logits = state->logits - state->logits.logsumexp(/*dim=*/1, /*keepdim=*/true);
      state->logprob = torch::log_softmax(state->logits, /* dim=*/ 1);
      auto probs = state->logprob.exp();
      state->actions = torch::multinomial(probs, /*num_samples=*/1, /*replacement=*/true).squeeze(-1);
      PUFFER_ASSERT(state->actions.numel() == opt->num_actions, "Invalid action size.");

      for (int i = 0; i < opt->num_actions; i++) { actions[i] = state->actions[i].item<int>(); }
      state->entropy = -(state->logprob * state->logprob.exp()).sum(1);
    }
  }

  // Batched env forward eval.
  void forward_eval_batch(PufferEnvState* state) {}

  void info() const
  {
    for (auto& np : this->named_parameters())
    {
      std::cout << np.key() << ": " << np.value().sizes() << std::endl;
    }
  }

  void init_state(Tensor full_obs)
  {
    auto state = env_states_[0];
    state->full_obs = full_obs;
    state->index = 0;
    state->h = torch::zeros({1, opt->hidden_size}).to(device_);
    state->c = torch::zeros({1, opt->hidden_size}).to(device_);
    state->values = Tensor{};
    state->logits = Tensor{};
    state->logprob = Tensor{};
    state->entropy = Tensor{};
    state->actions = Tensor{};
  }

  void start_eval_lstm(Tensor full_obs, Tensor encoder_linear_w, Tensor encoder_linear_b,
    Tensor decoder_linear_w, Tensor decoder_linear_b,
    Tensor value_w, Tensor value_b,
    Tensor weight_ih, Tensor weight_hh, Tensor bias_ih, Tensor bias_hh)
  {
    torch::NoGradGuard no_grad;

    // c_print_tensor_infos(this->encoder_linear->weight, encoder_linear_w);
    // c_print_tensor_infos(this->encoder_linear->bias, encoder_linear_b);
    // c_print_tensor_infos(this->decoder->weight, decoder_linear_w);
    // c_print_tensor_infos(this->decoder->bias, decoder_linear_b);
    // c_print_tensor_infos(this->value->weight, value_w);
    // c_print_tensor_infos(this->value->bias, value_b);
    // c_print_tensor_infos(this->lstm_cell->weight_ih, weight_ih);
    // c_print_tensor_infos(this->lstm_cell->weight_hh, weight_hh);
    // c_print_tensor_infos(this->lstm_cell->bias_ih, bias_ih);
    // c_print_tensor_infos(this->lstm_cell->bias_hh, bias_hh);
    // Update the model weights with the provided tensors

    this->encoder_linear->weight = encoder_linear_w;
    this->encoder_linear->bias = encoder_linear_b;
    this->decoder->weight = decoder_linear_w;
    this->decoder->bias = decoder_linear_b;
    this->value->weight = value_w;
    this->value->bias = value_b;
    this->lstm_cell->weight_ih = weight_ih;
    this->lstm_cell->weight_hh = weight_hh;
    this->lstm_cell->bias_ih = bias_ih;
    this->lstm_cell->bias_hh = bias_hh;
  }

private:
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
  torch::Device device_ = torch::kCPU;

  PufferOptions* opt{nullptr};
  PufferEnvState** env_states_;
};

struct PufferTorch
{
  // Could hold other models too, but for now, just one.
  LSTMWrapper* model;
  
  // Per-eval batch size (# of envs / batch) and count (# of batches).
  int eval_batch_size;
  int eval_batch_count;
};

void c_setup_pufferoptions(PufferOptions* options, const int num_actions, const int num_logits, const int input_size,
  const int hidden_size, const bool is_continuous, const int batch_chunk_size_mb)
{
  options->num_actions = num_actions;
  options->num_logits = num_logits;
  options->logit_sizes = new int64_t[num_actions];
  options->batch_chunk_size_mb = batch_chunk_size_mb;
  for (int i = 0; i < num_actions; i++) { options->logit_sizes[i] = num_logits; }
  options->input_size = input_size;
  options->hidden_size = hidden_size;
  options->is_continuous = is_continuous;
}

void c_cleanup_pufferoptions(PufferOptions* options)
{
  if (!options) { return; }
  if (options->logit_sizes)
  {
    delete[] options->logit_sizes;
    options->logit_sizes = nullptr;
  }
}

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
      throw;                      \
    }

#else
#define END_LIBTORCH_CATCH
#endif


PufferTorch* c_torch_alloc(PufferOptions* opt, VecEnv* vec_env)
{
  BEGIN_LIBTORCH_CATCH
  {
    PUFFER_ASSERT(opt != nullptr && opt->num_actions > 0 && opt->num_atns == 0 && opt->logit_sizes != nullptr && opt->enable_native_libtorch,
      "Invalid options.");
    auto* ptorch = new PufferTorch();
    opt->batch_chunk_size_mb = std::max(1, opt->batch_chunk_size_mb);
    ptorch->model = new LSTMWrapper(opt);
    int batch_chunk_size = (opt->batch_chunk_size_mb * 1024 * 1024) / (opt->obs_size * sizeof(float));
    if (batch_chunk_size < 1) { batch_chunk_size = 1; }
    ptorch->eval_batch_size = batch_chunk_size;
    ptorch->eval_batch_count = (vec_env->num_envs + batch_chunk_size - 1) / batch_chunk_size;
    printf(
      "Enabled native multithreading + native libtorch support with %d threads across %d envs (batch size = max %d envs per batch; %d batches).\n",
      opt->num_threads, vec_env->num_envs, ptorch->eval_batch_size, ptorch->eval_batch_count);

    return ptorch;
  }
  END_LIBTORCH_CATCH
}

void c_torch_load_weights(PufferTorch* pt, Weights* weights)
{
  BEGIN_LIBTORCH_CATCH
  {
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr && weights != nullptr, "Invalid state/inputs.");
    pt->model->update_model_weights(weights);
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



// Single-env eval (only for testing purposes).
void c_evalenv(PufferTorch* pt, float* obs, int* actions)
{
  BEGIN_LIBTORCH_CATCH
  {
    PUFFER_ASSERT(pt != nullptr && pt->model != nullptr && actions != nullptr && obs != nullptr,
      "Invalid state/inputs.");
    pt->model->forward_eval_single_env(obs, actions);
  }
  END_LIBTORCH_CATCH
}

// APIs to separate env_multithread/env_binding stuff from libtorch cleanly.

#ifndef PUFFER_EXTERN
// Silliness as the header is included in both C and C++ files (and from binding.c from each env). Makes it very hard to separate it.
struct Env;
struct VecEnv;
#define PUFFER_EXTERN extern "C"
PUFFER_EXTERN void c_step(Env* env);
#endif

PUFFER_EXTERN float* get_obs_ptr(Env* env);
PUFFER_EXTERN int* get_actions_ptr(Env* env);
PUFFER_EXTERN float* get_rewards_ptr(Env* env);
PUFFER_EXTERN unsigned char* get_terminals_ptr(Env* env);


void c_torch_start_eval_lstm(uintptr_t vec_env_ptr, Tensor full_obs_torch, Tensor encoder_linear_w,
  Tensor encoder_linear_b,
  Tensor decoder_linear_w, Tensor decoder_linear_b,
  Tensor value_w, Tensor value_b,
  Tensor weight_ih, Tensor weight_hh, Tensor bias_ih, Tensor bias_hh)
{
  VecEnv* vec_env = (VecEnv*)vec_env_ptr;
  torch::NoGradGuard no_grad;
  PufferTorch* puff_torch = vec_env->puff_torch;
  PUFFER_ASSERT(puff_torch != nullptr && puff_torch->model != nullptr, "Invalid state.");

  puff_torch->model->start_eval_lstm(full_obs_torch, encoder_linear_w, encoder_linear_b,
    decoder_linear_w, decoder_linear_b, value_w, value_b, weight_ih, weight_hh, bias_ih, bias_hh);
}

struct PufferEvalResult {
  Tensor values;
  Tensor logits;
  Tensor logprob;
  Tensor entropy;
  Tensor actions;
};

//! @brief Performs action (inference) + step segmented across a BPTT horizon batched by envs.
PufferEvalResult c_run_native_fulleval(uintptr_t vec_env_ptr)
{
  BEGIN_LIBTORCH_CATCH
  {
    auto* vec_env = reinterpret_cast<VecEnv*>(vec_env_ptr);
    PufferTorch* pt = vec_env->puff_torch;
    PUFFER_ASSERT(
      pt != nullptr && pt->eval_batch_count > 0 && pt->eval_batch_size > 0 && pt->model != nullptr &&
      vec_env-> num_envs > 1 && vec_env->envs != nullptr &&
      vec_env->threading != nullptr, "Invalid state/inputs.");


    torch::NoGradGuard no_grad;
    PufferEvalResult result = { };
    return result;
  }
  END_LIBTORCH_CATCH
}

void c_torch_finish_eval_lstm(uintptr_t vec_env_ptr)
{
  VecEnv* vec_env = (VecEnv*)vec_env_ptr;
  torch::NoGradGuard no_grad;
  PufferTorch* puff_torch = vec_env->puff_torch;
  PUFFER_ASSERT(puff_torch != nullptr && puff_torch->model != nullptr, "Invalid state.");
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


void c_init_multithreading(PufferOptions* options, VecEnv* vec_env)
{
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

void c_add_work_batched(VecEnv* vec_env, work_func func, void* arg, int start_index, int end_index)
{
  PUFFER_ASSERT(vec_env->threading != nullptr && end_index >= start_index, "Invalid threading state.");
  const auto num_threads = vec_env->threading->num_threads.load();
  if (end_index == start_index)
  {
    vec_env->threading->add_work({.func = func, .arg = arg, .start_index = start_index, .end_index = end_index});
    return;
  }
  const int batch_size = (end_index - start_index + 1 + num_threads) / num_threads;
  for (; start_index < end_index; start_index += batch_size)
  {
    int item_end = start_index + batch_size;
    if (item_end >= end_index) { item_end = end_index; }
    else { item_end--; }
    vec_env->threading->add_work({.func = func, .arg = arg, .start_index = start_index, .end_index = item_end});
  }
}

void c_wait_all_done(VecEnv* vec_env)
{
  PUFFER_ASSERT(vec_env->threading != nullptr, "Invalid threading state.");
  vec_env->threading->wait_all_done();
}
