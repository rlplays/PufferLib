#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <torch/torch.h>
#include "puffer_native.h"
#include "puffer_threads.h"
#include "puffer_utils.h"
#include "puffer_cuda.h"

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

// TODO(perumaal): This chain of .cpp includes is really messy, really need some build system to fix this.
#include <muon.cpp>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>


using namespace std;
using torch::Tensor;
using namespace ::c10::cuda;


struct LSTMTrainWrapper : torch::nn::Module
{
  LSTMTrainWrapper(VecEnv* vec_env, PufferOptions* opt, const PufferTrainOpts& config, int num_envs) :
    opt(opt), config(config), num_envs(num_envs), vec_env(vec_env)
  {
    if (!torch::cuda::is_available())
    {
      throw std::runtime_error("LSTMWrapper requires CUDA device.");
    }
    std::cout << "[Enabling native CUDA training - LSTM model]" << std::endl;
    torch::globalContext().setDeterministicCuDNN(false);

    // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090) (copied from pufferlib)
    torch::globalContext().setAllowTF32CuBLAS(true);
    torch::globalContext().setAllowTF32CuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(true);

    this->config = config;

    prio_beta0 = config.get_double("prio_beta0", 0.0);
    prio_alpha = config.get_double("prio_alpha", 0.0);
    clip_coef = config.get_double("clip_coef", 0.2);
    vf_clip_coef = config.get_double("vf_clip_coef", 0.0);
    vf_coef = config.get_double("vf_coef", 0.5);
    ent_coef = config.get_double("ent_coef", 0.01);
    gamma = config.get_double("gamma", 0.99);
    gae_lambda = config.get_double("gae_lambda", 0.95);
    vtrace_rho_clip = config.get_double("vtrace_rho_clip", 1.0);
    vtrace_c_clip = config.get_double("vtrace_c_clip", 1.0);
    anneal_lr = config.get_bool("anneal_lr", false);
    learning_rate = config.get_double("learning_rate", 0.0015);
    min_lr_ratio = config.get_double("min_lr_ratio", 0.1);

    device = torch::kCUDA;
    encoder_linear = layer_init(torch::nn::Linear(opt->obs_size, opt->hidden_size));
    encoder_gelu = torch::nn::GELU();
    encoder = register_module("encoder", torch::nn::Sequential(encoder_linear, encoder_gelu));
    if (opt->is_continuous)
    {
      throw std::runtime_error("Continuous action spaces not yet supported in native LSTMTrainWrapper.");
    }
    else
    {
      opt->num_atns = 0;
      std::vector<int64_t> sizes_vec(opt->num_actions);
      std::vector<int64_t> offsets_vec(opt->num_actions);
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
      decoder = register_module("decoder", layer_init(torch::nn::Linear(opt->hidden_size, opt->num_atns), 0.01));
    }
    value = register_module("value", layer_init(torch::nn::Linear(opt->hidden_size, 1), 1.0));
    lstm = register_module("lstm", torch::nn::LSTM(opt->input_size, opt->hidden_size));

    ratio = torch::ones({vec_env->num_envs, opt->bptt_horizon}, device);
    ep_lengths = torch::zeros({vec_env->num_envs}, device);
    ep_indices = torch::zeros({vec_env->num_envs}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    advantages = torch::zeros({vec_env->num_envs, opt->bptt_horizon}, device);

    free_idx = vec_env->num_envs;
    // TODO(perumaal): Is this correct?
    double initial_lr = config.get_double("initial_lr", 0.0);
    MuonOptions muon_opts(/* */ initial_lr);
    muon_opts.weight_decay(config.get_double("weight_decay", 0.0));
    muon_opts.eps(config.get_double("adam_eps", 1e-8));
    muon_opts.momentum(config.get_double("adam_beta1", 0.9));

    muon = std::make_unique<Muon>(parameters(), muon_opts);
  }

  void train_model(int epoch, int total_epochs, int segments, int total_minibatches, int minibatch_segments,
    int accumulate_minibatches,
    Tensor obs, Tensor actions, Tensor logprobs, Tensor rewards, Tensor terminals, Tensor values,
    Tensor encoder_linear_w, Tensor encoder_linear_b,
    Tensor decoder_linear_w, Tensor decoder_linear_b,
    Tensor value_w, Tensor value_b, Tensor weight_ih, Tensor weight_hh, Tensor bias_ih, Tensor bias_hh)
  {
    PUFFER_ASSERT(epoch < total_epochs && total_epochs > 0, "Invalid epoch/total_epochs.");
    PUFFER_ASSERT(accumulate_minibatches > 0, "accumulate_minibatches must be > 0");

    losses = {};

    anneal_beta = prio_beta0 + ((1.0 - prio_beta0) * prio_alpha * (static_cast<double>(epoch) / static_cast<double>(
      total_epochs)));
    ratio.fill_(1.0);

    if (anneal_lr)
    {
      float lr_min = min_lr_ratio * learning_rate;
      float lr = cosine_annealing(learning_rate, lr_min, epoch, (double)total_epochs);
      muon->lr.fill_(lr);
    }

    // TODO: Optimize
    Tensor values_cpu = values.to(torch::kCPU);
    Tensor rewards_cpu = rewards.to(torch::kCPU);
    Tensor terminals_cpu = terminals.to(torch::kCPU);
    

    for (int mb = 0; mb < total_minibatches; mb++)
    {
      advantages.zero_();
      
      Tensor idx , mb_prio , mb_obs, mb_actions, mb_logprobs, mb_values, returns, mb_advantages;
      
      { // Compute advantages
        torch::NoGradGuard no_grad;
        compute_puff_advantage(values_cpu, rewards_cpu, terminals_cpu, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip,
          vtrace_c_clip);
        Tensor prio_probs;
        compute_priority_weights(advantages, prio_alpha, prio_probs);
        idx = torch::multinomial(prio_probs, minibatch_segments);
        mb_prio = (segments * prio_probs[idx, /*dim*/ 0]).pow(-anneal_beta);
        mb_obs = obs.index_select(0, idx);
        mb_actions = actions.index_select(0, idx);
        mb_logprobs = logprobs.index_select(0, idx);
        mb_values = values.index_select(0, idx);
        mb_advantages = advantages.index_select(0, idx);
        returns = mb_values + mb_advantages;
        c_print_tensor_info(idx, "idx", true);
        c_print_tensor_info(mb_prio, "mb_prio", true);
        c_print_tensor_info(returns, "returns", true);
        c_print_tensor_info(mb_advantages, "mb_advantages", true);
      }
    }
  }
  void compute_priority_weights(Tensor advantages, double prio_alpha, Tensor& prio_probs_out)
  {
    torch::NoGradGuard no_grad;
    Tensor adv = advantages.abs().sum(/* axis */ 1);
    Tensor prio_weights = torch::nan_to_num(adv.pow(prio_alpha), 0, 0, 0);
    // TODO: optimize - no allocs?
    prio_probs_out = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6);
  }
private:
  // Copied from pufferlib.
  static float cosine_annealing(float lr_base, float lr_min, int t, int T)
  {
    if (T == 0) return lr_base; // avoid division by zero
    float ratio = static_cast<float>(t) / static_cast<float>(T);
    ratio = std::max(0.0f, std::min(1.0f, ratio)); // clamp to [0, 1]
    return lr_min + 0.5f * (lr_base - lr_min) * (1.0f + std::cos(M_PI * ratio));
  }

private:
  torch::Device device = torch::kCPU;
  int num_envs;
  PufferOptions* opt{nullptr};
  VecEnv* vec_env;
  torch::nn::Sequential encoder{nullptr};
  torch::nn::Linear encoder_linear{nullptr};
  torch::nn::GELU encoder_gelu{nullptr};
  torch::nn::Linear decoder{nullptr};
  torch::nn::Linear value{nullptr};
  torch::nn::LSTM lstm{nullptr};
  PufferTrainOpts config;
  PufferTrainResult result;

  // Config params
  double prio_beta0{0.0};
  double prio_alpha{0.0};
  double clip_coef{0.2};
  double vf_clip_coef{0.0};
  double vf_coef{0.5};
  double ent_coef{0.01};
  double gamma{0.99};
  double gae_lambda{0.95};
  double vtrace_rho_clip{1.0};
  double vtrace_c_clip{1.0};
  int epoch, total_epochs;
  int segments, total_minibatches, minibatch_segments, accumulate_minibatches;
  bool anneal_lr;
  double learning_rate, min_lr_ratio;
  std::map<std::string, double> losses;

  // Training-time tensors.
  Tensor ratio, ep_lengths, ep_indices;
  Tensor advantages;
  int free_idx;

  // Training-time state.
  double anneal_beta{0.0};
  std::unique_ptr<Muon> muon;
};
