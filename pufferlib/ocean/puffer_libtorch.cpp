#include "puffer_libtorch.h"
#include <torch/torch.h>

#include <iostream>

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
  LSTMWrapper(const int obs_size, int logit_sizes[], const int num_actions, const int input_size = 128,
    const int hidden_size = 128) :
    hidden_size_(hidden_size), input_size_(input_size), obs_size_(obs_size), num_actions_(num_actions)
  {
    // TODO: Assumes multidiscrete.
    encoder = register_module("encoder", torch::nn::Sequential(
      layer_init(torch::nn::Linear(obs_size, hidden_size)),
      torch::nn::GELU()));
    lstm_cell = register_module("lstmcell", torch::nn::LSTMCell(input_size, hidden_size));
    //lstm_cell->weight_hh 
    decoder = register_module(
      "decoder", torch::nn::Linear(hidden_size, std::accumulate(logit_sizes, logit_sizes + num_actions, 0)));
  }

  torch::nn::Linear layer_init(torch::nn::Linear layer, const double std = std::sqrt(2.0),
    const double bias_const = 0.0)
  {
    torch::nn::init::orthogonal_(layer->weight, std);
    torch::nn::init::constant_(layer->bias, bias_const);
    return layer;
  }
  
  void update_model_weights(float* h, float* c)
  {
    
  }

  void forward_eval(float* obs, float* actions_out)
  {
    // Assumes obs_size_ for obs, and num_actions_ for actions_out already initialized.
    auto obs_tensor = torch::from_blob(obs, {obs_size_}, torch::kFloat32);
    torch::Tensor hidden_tensor = encoder->forward(obs_tensor);
    // Copy model weights to LSTM cell before use.
    
  }

  // Inference only for now (need to copy weights from trained model)
  torch::nn::Sequential encoder{nullptr};
  torch::nn::LSTMCell lstm_cell{nullptr};
  torch::nn::Linear decoder{nullptr};
  int hidden_size_, input_size_, obs_size_, num_actions_;
};

void c_eval() {}
