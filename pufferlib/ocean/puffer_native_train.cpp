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

#ifndef _WIN32
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

// TODO(perumaal): This chain of .cpp includes is really messy, really need some build system to fix this.
#include <muon.cpp>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
using namespace ::c10::cuda;


using namespace std;
using torch::Tensor;
using namespace std;


struct LSTMTrainWrapper : torch::nn::Module
{
  LSTMTrainWrapper(VecEnv* vec_env, PufferOptions* opt, int num_envs) : opt(opt), num_envs(num_envs), vec_env(vec_env)
  {
    if (!torch::cuda::is_available())
    {
      throw std::runtime_error("LSTMWrapper requires CUDA device.");
    }
    std::cout << "-- Using native LSTM train wrapper with libtorch " << TORCH_VERSION << std::endl;
    torch::globalContext().setDeterministicCuDNN(false);

    // Enable TF32 for faster FP32 math (uses Tensor Cores on 4090) (copied from pufferlib)
    torch::globalContext().setAllowTF32CuBLAS(true);
    torch::globalContext().setAllowTF32CuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(true);

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
  }

  void prepare_train(VecEnv* vec_env, PufferTrainOpts config)
  {
    if (train_opts.config.size() == 0)
    {
      for (auto& [k, v] : train_opts.config)
      {
        std::cout << "-- Train config: " << k << " = " << v << std::endl;
      }
      this->train_opts = train_opts;
    }
  }

  PufferTrainResult train_model(VecEnv* vec_env) { return {}; }

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
  PufferTrainOpts train_opts;
};
