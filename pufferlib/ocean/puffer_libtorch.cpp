#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include "puffer_libtorch.h"
#include <torch/torch.h>

#include <iostream>

namespace pufferlib
{
void c_libtorch_info()
{
  std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
  std::cout << "cuDNN available: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;
  if (torch::cuda::is_available())
  {
    std::cout << "Number of CUDA devices: " << torch::cuda::device_count() << std::endl;
  }
  torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  torch::Tensor test_tensor = torch::zeros({2, 2}, device);
  std::cout << "Test tensor device: " << test_tensor.device() << std::endl;
}

struct LSTMWrapper : torch::nn::Module
{
  LSTMWrapper(const PufferOptions& opt) : opt_(opt)
  {
    // Cannot be multidiscrete and continuous at the same time.
    assert(opt.is_multidiscrete || !opt.is_continuous);
    encoder = register_module("encoder",
      torch::nn::Sequential(layer_init(torch::nn::Linear(opt.obs_size, opt.hidden_size)), torch::nn::GELU()));
    if (opt.is_multidiscrete || opt.is_continuous)
    {
      decoder = register_module("decoder", layer_init(torch::nn::Linear(opt.hidden_size, opt.num_actions), 0.01));
    }
    else
    {
      decoder_mean = register_module("decoder_mean",
        layer_init(torch::nn::Linear(opt.hidden_size, opt.num_actions), 0.01));
      decoder_logstd = register_parameter("decoder_logstd", torch::zeros({1, opt.num_actions}));
    }
    lstm = register_module("lstm", torch::nn::LSTM(opt.input_size, opt.hidden_size));
    lstm_cell = register_module("lstmcell", torch::nn::LSTMCell(opt.input_size, opt.hidden_size));

    for (auto& np : this->named_parameters())
    {
      std::cout << np.key() << ": " << np.value().sizes() << std::endl;
    }
  }

  [[nodiscard]] torch::nn::Linear layer_init(torch::nn::Linear layer, const double std = std::sqrt(2.0),
    const double bias_const = 0.0) const
  {
    torch::nn::init::orthogonal_(layer->weight, std);
    torch::nn::init::constant_(layer->bias, bias_const);
    return layer;
  }

  void update_model_weights(float* h, float* c)
  {
    // h is of size hidden_size_, while c is of size .
  }

  void forward_eval(float* obs, float* actions_out)
  {
    // Assumes obs_size_ for obs, and num_actions_ for actions_out already initialized.
    auto obs_tensor = torch::from_blob(obs, {opt_.obs_size}, torch::kFloat32);
    torch::Tensor hidden_tensor = encoder->forward(obs_tensor);
    // Copy model weights to LSTM cell before use.
  }

  // Inference only for now (need to copy weights from trained model)
  torch::nn::Sequential encoder{nullptr};
  torch::nn::LSTMCell lstm_cell{nullptr};
  torch::nn::Linear decoder{nullptr};
  torch::nn::Linear decoder_mean{nullptr};
  at::Tensor decoder_logstd{nullptr};
  // Unused for now (mainly by training, but used here to match with lstm_cell)
  torch::nn::LSTM lstm{nullptr};

  PufferOptions opt_ = {};
};

struct PufferTorch
{
  LSTMWrapper* model;
};

PufferTorch* c_torch_alloc(const PufferOptions& opt)
{
  auto* ptorch = new PufferTorch();
  ptorch->model = new LSTMWrapper(opt);
  return ptorch;
}

void c_torch_free(const PufferTorch* pt)
{
  if (!pt) return;
  delete pt->model;
  delete pt;
}

void c_eval(const PufferTorch* pt)
{
  if (!pt) return;
  // Perform evaluation using ptorch->model
}
}
