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

#include "puffer_threads.h"
#include "puffer_utils.h"


#if DEBUG
// Uncomment this to check CUDA fused kernels with their slower counterparts (evaluate both).
// #define PUFFER_DBG_CHECK_NETWORK_SLOW 1
#endif
struct PufferBatchState;

struct LSTMPolicyModule : torch::nn::Module
{
  int eval_batch_size;
  int eval_batch_count;
  int num_cuda_streams;
  PufferOptions* opt{nullptr};

  int num_envs;

  LSTMPolicyModule(VecEnv* vec_env, PufferOptions* opt, int num_envs, torch::Device device) :
      opt(opt), num_envs(num_envs)
  {
    encoder_linear = layer_init(torch::nn::Linear(opt->obs_size, opt->hidden_size));
    encoder_linear->to(device);
    encoder_gelu = torch::nn::GELU();
    encoder_gelu->to(device);
    encoder = register_module("encoder", torch::nn::Sequential(encoder_linear, encoder_gelu));
    encoder->to(device);
    if (opt->is_continuous)
    {
      decoder_mean =
        register_module("decoder_mean", layer_init(torch::nn::Linear(opt->hidden_size, opt->num_actions), 0.01));
      decoder_logstd = register_parameter("decoder_logstd", torch::zeros({1, opt->num_actions}));
      throw std::runtime_error("Continuous action spaces not yet supported in native LSTMWrapper.");
    }
    else
    {

      decoder = register_module("decoder", layer_init(torch::nn::Linear(opt->hidden_size, opt->num_atns), 0.01));
      decoder->to(device);
    }
    value = register_module("value", layer_init(torch::nn::Linear(opt->hidden_size, 1), 1.0));
    value->to(device);
    lstm_cell = register_module("lstmcell", torch::nn::LSTMCell(opt->input_size, opt->hidden_size));
    lstm_cell->to(device);
  }

  // Sentinel to fill in and check for after running a CUDA op.
  // NOTE: 42 doesn't work in envs like puffer_go because it's a legit action value (square/grid number).
  constexpr static int PUFFER_CHECK_SENTINEL_VALUE = 42123;

  void old_lstm_network_forward_eval(PufferBatchState** env_states, int batch_index)
  {
    torch::NoGradGuard no_grad;
    auto* state = env_states[batch_index];
    state->hidden_out = encoder->forward(state->obs_device.transpose(0, 1));
    auto [h2_new, c2_new] = lstm_cell->forward(state->hidden_out, std::tuple(state->h1, state->c1));
    state->h2 = h2_new;
    state->c2 = c2_new;
    Tensor decoder_out = decoder->forward(state->h2);
    Tensor values_out = value->forward(state->h2);
    sample_logits(decoder_out, opt->num_actions, opt->logit_sizes, state->actions_horizon_out,
                  state->logprob_horizon_out);
    state->values_horizon_out.copy_(values_out);
  }


  void info() const
  {
    for (auto& np : this->named_parameters())
    {
      std::cout << np.key() << ": " << np.value().sizes() << std::endl;
    }
  }

private:
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
};
