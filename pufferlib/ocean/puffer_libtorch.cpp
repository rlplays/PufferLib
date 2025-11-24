#pragma warning(disable : 4624) // Class Destructor Not visible
#pragma warning(disable : 4805) // Comparing bool and int
#pragma warning(disable : 4067) // Extra /Za preprocessor command

#include "puffer_libtorch.h"
#include <torch/torch.h>
#include "puffernet.h"
#include <cassert>
#include <iostream>

namespace pufferlib
{
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
      if (opt->is_multidiscrete)
      {
        for (int i = 0; i < opt_->num_actions; i++) { opt_->num_atns += opt_->logit_sizes[i]; }
      }
      else { opt_->num_atns = opt->num_actions; }
      decoder = register_module("decoder", layer_init(torch::nn::Linear(opt->hidden_size, opt_->num_atns), 0.01));
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
    PUFFER_ASSERT(weights != nullptr && opt_ != nullptr && opt_->num_atns > 0, "Invalid input/state.");
    PUFFER_ASSERT(!opt_->is_continuous, "Only supports multidiscrete for now.");
    torch::NoGradGuard no_grad;
    weights_to_linear(weights, opt_->obs_size, opt_->hidden_size, encoder_linear);
    weights_to_linear(weights, opt_->hidden_size, opt_->num_atns, decoder);
    weights_to_linear(weights, opt_->hidden_size, 1, value);
    weights_to_tensor(weights, opt_->hidden_size * opt_->input_size * 4, lstm_cell->weight_ih);
    weights_to_tensor(weights, opt_->hidden_size * opt_->input_size * 4, lstm_cell->weight_hh);
    weights_to_tensor(weights, opt_->hidden_size * 4, lstm_cell->bias_ih);
    weights_to_tensor(weights, opt_->hidden_size * 4, lstm_cell->bias_hh);
    PUFFER_ASSERT(weights->idx == weights->size, "Must have precisely used all weights.");
  }

  void forward_eval(float* obs, int* actions)
  {
    // Assumes obs_size_ for obs, and num_actions_ for actions_out already initialized.
    auto obs_tensor = torch::from_blob(obs, {opt_->obs_size}, torch::kFloat32);
    auto t1 = encoder->forward(obs_tensor);
    auto t2 = encoder_gelu->forward(t1);
    // TODO tomorrow auto t3 = lstm_cell->forward(t2);
    //auto t4 = decoder->forward(lstm_cell->)
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

// LibTorch throws exceptions on errors, log them correctly in debug mode only.
#if DEBUG
#define BEGIN_LIBTORCH_CATCH try {
#else
#define BEGIN_LIBTORCH_CATCH {
#endif

#if DEBUG
#define END_LIBTORCH_CATCH        \
    }                             \
    catch (const c10::Error& e)   \
    {                             \
      std::cerr << "Error from libtorch: " << e.what() << std::endl;\
      throw;                      \
    }

#else
#define END_LIBTORCH_CATCH }
#endif


PufferTorch* c_torch_alloc(PufferOptions* opt)
{
  BEGIN_LIBTORCH_CATCH
    auto* ptorch = new PufferTorch();
    ptorch->model = new LSTMWrapper(opt);
    return ptorch;
  END_LIBTORCH_CATCH
}

void c_torch_free(const PufferTorch* pt)
{
  BEGIN_LIBTORCH_CATCH
    PUFFER_ASSERT(!pt, "Invalid state.");
    delete pt->model;
    delete pt;
  END_LIBTORCH_CATCH
}

void c_torch_load_weights(PufferTorch* pt, Weights* weights)
{
  BEGIN_LIBTORCH_CATCH
    PUFFER_ASSERT(!(!pt || !pt->model || !weights), "Invalid state/inputs.");
    pt->model->update_model_weights(weights);
  END_LIBTORCH_CATCH
}

void c_eval(const PufferTorch* pt, float* obs, int* actions)
{
  BEGIN_LIBTORCH_CATCH
    PUFFER_ASSERT(!(!pt || !pt->model || !actions || !obs), "Invalid state/inputs.");
    pt->model->forward_eval(obs, actions);
  END_LIBTORCH_CATCH
}
}
