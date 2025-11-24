#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include "puffer_libtorch.h"
#include <torch/torch.h>
#include "puffernet.h"
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
  LSTMWrapper(PufferOptions* opt) : opt_(opt)
  {
    // Cannot be multidiscrete and continuous at the same time.
    assert(opt->is_multidiscrete || !opt->is_continuous);
    encoder_linear = layer_init(torch::nn::Linear(opt->obs_size, opt->hidden_size));
    encoder_gelu = torch::nn::GELU();
    encoder = register_module("encoder", torch::nn::Sequential(encoder_linear, encoder_gelu));
    if (opt->is_multidiscrete || opt->is_continuous)
    {
      decoder = register_module("decoder", layer_init(torch::nn::Linear(opt->hidden_size, opt->num_actions), 0.01));
    }
    else
    {
      decoder_mean = register_module("decoder_mean",
        layer_init(torch::nn::Linear(opt->hidden_size, opt->num_actions), 0.01));
      decoder_logstd = register_parameter("decoder_logstd", torch::zeros({1, opt->num_actions}));
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

  torch::Tensor weights_to_tensor(Weights* weights, const size_t num_weights)
  {
    auto w = torch::from_blob(get_weights(weights, num_weights), {static_cast<int64_t>(num_weights)}, torch::kFloat32).clone();
    return w;
  }
  void update_model_weights(Weights* weights)
  {
    encoder_linear->weight.data().copy_(weights_to_tensor(weights, opt_->input_size*opt_->hidden_size));
    encoder_linear->bias.data().copy_(weights_to_tensor(weights, opt_->hidden_size));
  }

  void forward_eval(float* obs, float* actions_out)
  {
    // Assumes obs_size_ for obs, and num_actions_ for actions_out already initialized.
    auto obs_tensor = torch::from_blob(obs, {opt_->obs_size}, torch::kFloat32);
    torch::Tensor hidden_tensor = encoder->forward(obs_tensor);
    // Copy model weights to LSTM cell before use.
  }


  void info() const
  {
    for (auto& np : this->named_parameters())
    {
      std::cout << np.key() << ": " << np.value().sizes() << std::endl;
    }
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
  torch::nn::Linear decoder_mean{nullptr};
  at::Tensor decoder_logstd{nullptr};

  // LSTM Policy on top of the encoder/decoder above.
  torch::nn::LSTMCell lstm_cell{nullptr};

  PufferOptions* opt_{nullptr};
};

struct PufferTorch
{
  LSTMWrapper* model;
};

void c_setup_pufferoptions(PufferOptions* options, const int num_logits)
{
  options->logit_sizes = new int[num_logits];
}

void c_cleanup_pufferoptions(PufferOptions* options)
{
  if (options->logit_sizes)
  {
    delete[] options->logit_sizes;
    options->logit_sizes = nullptr;
  }
}

PufferTorch* c_torch_alloc(PufferOptions* opt)
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

void c_torch_load_weights(PufferTorch* pt, Weights* weights)
{
  if (!pt || !pt->model || !weights) return;
  pt->model->update_model_weights(weights);
}

void c_eval(const PufferTorch* pt)
{
  if (!pt) return;
  // Perform evaluation using ptorch->model
}
}
