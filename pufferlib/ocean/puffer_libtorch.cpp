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
  LSTMWrapper(const int input_size=128, const int hidden_size=128)
  {
    lstm_cell = register_module("lstmcell", torch::nn::LSTMCell(input_size, hidden_size));
    //lstm_cell->forward(torch::tensor)
  }
  
  // For inference only.
  torch::nn::LSTMCell lstm_cell{nullptr};
};

void c_eval()
{
  
  
  
}
