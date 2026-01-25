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
    torch::AutoGradMode enable_grad(true);

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
    encoder->to(device);
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
      decoder->to(device);
    }
    value = register_module("value", layer_init(torch::nn::Linear(opt->hidden_size, 1), 1.0));
    value->to(device);
    lstm = register_module("lstm", torch::nn::LSTM(opt->input_size, opt->hidden_size));
    lstm->to(device);

    ratio = torch::ones({vec_env->num_envs, opt->bptt_horizon}, device);
    ep_lengths = torch::zeros({vec_env->num_envs}, device);
    ep_indices = torch::zeros({vec_env->num_envs}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    advantages = torch::zeros({vec_env->num_envs, opt->bptt_horizon}, device);
    free_idx = vec_env->num_envs;


    // TODO(perumaal): Is this correct?
    double initial_lr = config.get_double("learning_rate", 0.0);
    MuonOptions muon_opts(/* */ initial_lr);
    muon_opts.weight_decay(config.get_double("weight_decay", 0.0));
    muon_opts.eps(config.get_double("adam_eps", 1e-8));
    muon_opts.momentum(config.get_double("adam_beta1", 0.9));

    muon = std::make_unique<Muon>(parameters(), muon_opts);
  }

  PufferTrainResult train_model(int epoch, int total_epochs, int segments, int total_minibatches,
    int minibatch_segments, int accumulate_minibatches,
    Tensor obs, Tensor actions, Tensor logprobs, Tensor rewards, Tensor terminals, Tensor values)
  {
    BEGIN_LIBTORCH_CATCH
    {
      torch::AutoGradMode enable_grad(true);

      PufferTrainResult result = {};
      PUFFER_ASSERT(accumulate_minibatches > 0, "accumulate_minibatches must be > 0");

      double max_grad_norm = config.get_double("max_grad_norm", 0.5);

      anneal_beta = prio_beta0 + ((1.0 - prio_beta0) * prio_alpha * (static_cast<double>(epoch) / static_cast<double>(
        total_epochs)));
      ratio.fill_(1.0);

      if (anneal_lr)
      {
        float lr_min = min_lr_ratio * learning_rate;
        float lr = cosine_annealing(learning_rate, lr_min, epoch, (double)total_epochs);
        muon->lr.fill_(lr);
      }
      auto params = lstm->named_parameters();
      losses = {};
      losses["policy_loss"] = 0.0;
      losses["value_loss"] = 0.0;
      losses["entropy"] = 0.0;
      losses["old_approx_kl"] = 0.0;
      losses["approx_kl"] = 0.0;
      losses["clipfrac"] = 0.0;
      losses["importance"] = 0.0;
      getDefaultCUDAStream().synchronize();
      for (int mb = 0; mb < total_minibatches; mb++)
      {
        advantages.zero_();

        Tensor idx, mb_prio, mb_obs, mb_actions, mb_logprobs, mb_values, mb_returns, mb_advantages;

        { // No grad buffers: Compute advantages & priority weights
          compute_puff_advantage_cuda(values, rewards, terminals, ratio, advantages, gamma, gae_lambda,
            vtrace_rho_clip, vtrace_c_clip);
          Tensor adv = advantages.abs().sum(/* axis */ 1);
          Tensor prio_weights = torch::nan_to_num(adv.pow(prio_alpha), 0, 0, 0);
          Tensor prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6);
          idx = torch::multinomial(prio_probs, minibatch_segments);
          mb_prio = (segments * prio_probs.index_select(0, idx).unsqueeze(1)).pow(-anneal_beta);
          mb_obs = obs.index_select(0, idx);
          mb_actions = actions.index_select(0, idx);
          mb_logprobs = logprobs.index_select(0, idx);
          mb_values = values.index_select(0, idx);
          mb_advantages = advantages.index_select(0, idx);
          mb_returns = mb_values + mb_advantages;
        }

        { // Backprop grad buffers used here: Actual policy/action sampling.
          Tensor newlogprob, newvalues, entropy;
          forward_sample_logits(mb_obs, mb_actions, newlogprob, entropy, newvalues);
          newlogprob = newlogprob.reshape_as(mb_logprobs);
          newvalues = newvalues.reshape_as(mb_values);
          Tensor logratio = newlogprob - mb_logprobs;
          Tensor newratio = logratio.exp();
          ratio.index_copy_(0, idx, newratio.detach());

          //
          // This is the most important part of training: PPO!
          //

          // Compue PPO loss.
          Tensor old_approx_kl, approx_kl, clipfrac;
          {
            old_approx_kl = (-logratio).mean();
            auto tmp1 = (newratio - 1);
            approx_kl = (tmp1 - logratio).mean();
            clipfrac = (tmp1.abs() > clip_coef).to(torch::kFloat32).mean();
          }
          // Weight advantages by priority and normalize
          Tensor adv = mb_prio * (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8);

          // Policy loss
          Tensor pg_loss1 = -adv * newratio;
          Tensor pg_loss2 = -adv * torch::clamp(newratio, 1 - clip_coef, 1 + clip_coef);
          Tensor pg_loss = torch::max(pg_loss1, pg_loss2).mean();

          // Value loss
          Tensor v_clipped = mb_values + torch::clamp(newvalues - mb_values, -vf_clip_coef, vf_clip_coef);
          Tensor v_loss_unclipped = (newvalues - mb_returns).pow(2);
          Tensor v_loss_clipped = (v_clipped - mb_returns).pow(2);
          Tensor v_loss = 0.5 * torch::max(v_loss_unclipped, v_loss_clipped).mean();

          Tensor entropy_loss = entropy.mean();
          Tensor loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss;
          values.index_copy_(0, idx, newvalues.detach());

          double total = total_minibatches;
          losses["policy_loss"] += (pg_loss.item<double>() / total);
          losses["value_loss"] += (v_loss.item<double>() / total);
          losses["entropy"] += (entropy_loss.item<double>() / total);
          losses["old_approx_kl"] += (old_approx_kl.item<double>() / total);
          losses["approx_kl"] += (approx_kl.item<double>() / total);
          losses["clipfrac"] += (clipfrac.item<double>() / total);
          losses["importance"] += (newratio.mean().item<double>() / total);

          loss.backward();
          if ((mb + 1) % accumulate_minibatches == 0)
          {
            // Add gradient clipping before optimizer step
            torch::nn::utils::clip_grad_norm_(parameters(), max_grad_norm);

            muon->step();
            muon->zero_grad();
          }
        }
      }
      getDefaultCUDAStream().synchronize();

      for (auto& [k, v] : losses)
      {
        PufferTrainStat stat;
        stat.name = k;
        stat.value_dbl = v;
        result.train_stats.push_back(stat);
      }
      return result;
    }
    END_LIBTORCH_CATCH
  }

  void compute_priority_weights(Tensor advantages, double prio_alpha, Tensor& prio_probs_out)
  {
    Tensor adv = advantages.abs().sum(/* axis */ 1);
    Tensor prio_weights = torch::nan_to_num(adv.pow(prio_alpha), 0, 0, 0);
    // TODO: optimize - no allocs?
    prio_probs_out = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6);
  }

  void forward_sample_logits(Tensor obs, Tensor& actions_in, Tensor& logprobs_out, Tensor& entropy_out,
    Tensor& values_out)
  {
    torch::AutoGradMode enable_grad(true);
    PUFFER_ASSERT(obs.dim() == 3, "Obs must be [num_envs, bptt_horizon, obs_size] shaped Tensor");
    auto B = obs.sizes()[0];
    auto TT = obs.sizes()[1];

    Tensor x = obs.reshape(at::IntArrayRef{B * TT, obs.sizes()[2]});;
    Tensor hidden = encoder->forward(x);
    PUFFER_ASSERT(hidden.sizes()[0] == B * TT && hidden.sizes()[1] == opt->hidden_size,
      "Encoder output has invalid shape.");
    hidden = hidden.reshape(at::IntArrayRef{B, TT, opt->hidden_size}).transpose(0, 1).contiguous();
    std::tuple<Tensor, std::tuple<Tensor, Tensor>> lstm_out = lstm->forward(hidden);
    Tensor hidden_new = std::get<0>(lstm_out).to(torch::kFloat32).transpose(0, 1).contiguous();
    Tensor h2 = std::get<0>(std::get<1>(lstm_out));
    Tensor c2 = std::get<1>(std::get<1>(lstm_out));
    auto flat_hidden = hidden_new.reshape({B * TT, opt->hidden_size});
    Tensor decoder_out = decoder->forward(flat_hidden);
    values_out = value->forward(hidden_new);
    values_out = values_out.squeeze(-1).transpose(0, 1);
    sample_logits_entropy(decoder_out, opt->num_actions, opt->logit_sizes, actions_in, logprobs_out, entropy_out);
  }

  bool assign_training_weights(torch::Tensor& encoder_linear_w, torch::Tensor& encoder_linear_b,
    torch::Tensor& decoder_linear_w, torch::Tensor& decoder_linear_b,
    torch::Tensor& value_w, torch::Tensor& value_b,
    torch::Tensor& weight_ih, torch::Tensor& weight_hh,
    torch::Tensor& bias_ih, torch::Tensor& bias_hh, Tensor encoder_linear_w_in, torch::Tensor encoder_linear_b_in,
  torch::Tensor decoder_linear_w_in, torch::Tensor decoder_linear_b_in,
  torch::Tensor value_w_in, torch::Tensor value_b_in,
  torch::Tensor weight_ih_in, torch::Tensor weight_hh_in,
  torch::Tensor bias_ih_in, torch::Tensor bias_hh_in)
  {
    auto lstm_params = lstm->named_parameters();
    assign_tensors(encoder_linear_w, encoder_linear->weight,  "encoder_linear_w");
    assign_tensors(encoder_linear_b, encoder_linear->bias, "encoder_linear_b");
    assign_tensors(decoder_linear_w, decoder->weight, "decoder_linear_w");
    assign_tensors(decoder_linear_b, decoder->bias, "decoder_linear_b");
    assign_tensors(value_w, value->weight, "value_w");
    assign_tensors(value_b, value->bias, "value_b");
    assign_tensors(weight_ih, lstm_params["weight_ih_l0"], "weight_ih_l0");
    assign_tensors(weight_hh, lstm_params["weight_hh_l0"], "weight_hh_l0");
    assign_tensors(bias_ih, lstm_params["bias_ih_l0"], "bias_ih_l0");
    assign_tensors(bias_hh, lstm_params["bias_hh_l0"], "bias_hh_l0");
    return true;
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

LSTMTrainWrapper* get_train_wrapper(PufferTorch* pt);

bool assign_training_weights(PufferTorch* pt, torch::Tensor& encoder_linear_w, torch::Tensor& encoder_linear_b,
  torch::Tensor& decoder_linear_w, torch::Tensor& decoder_linear_b, torch::Tensor& value_w, torch::Tensor& value_b,
  torch::Tensor& weight_ih, torch::Tensor& weight_hh, torch::Tensor& bias_ih, torch::Tensor& bias_hh,
  Tensor encoder_linear_w_in, torch::Tensor encoder_linear_b_in,
  torch::Tensor decoder_linear_w_in, torch::Tensor decoder_linear_b_in,
  torch::Tensor value_w_in, torch::Tensor value_b_in,
  torch::Tensor weight_ih_in, torch::Tensor weight_hh_in,
  torch::Tensor bias_ih_in, torch::Tensor bias_hh_in) 
{
  auto* train_model = get_train_wrapper(pt);
  if (train_model == nullptr) { 
    assign_tensors(encoder_linear_w, encoder_linear_w_in, "encoder_linear_w");
    assign_tensors(encoder_linear_b, encoder_linear_b_in, "encoder_linear_b");
    assign_tensors(decoder_linear_w, decoder_linear_w_in, "decoder_linear_w");
    assign_tensors(decoder_linear_b, decoder_linear_b_in, "decoder_linear_b");
    assign_tensors(value_w, value_w_in, "value_w");
    assign_tensors(value_b, value_b_in, "value_b");
    assign_tensors(weight_ih, weight_ih_in, "weight_ih");
    assign_tensors(weight_hh, weight_hh_in, "weight_hh");
    assign_tensors(bias_ih, bias_ih_in, "bias_ih");
    assign_tensors(bias_hh, bias_hh_in, "bias_hh");
    return true;
   }
  return train_model->assign_training_weights(
    encoder_linear_w, encoder_linear_b,
    decoder_linear_w, decoder_linear_b,
    value_w, value_b,
    weight_ih, weight_hh,
    bias_ih, bias_hh, encoder_linear_w_in, encoder_linear_b_in,
    decoder_linear_w_in, decoder_linear_b_in,
    value_w_in, value_b_in,
    weight_ih_in, weight_hh_in,
    bias_ih_in, bias_hh_in);
}