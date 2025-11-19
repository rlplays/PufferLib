#include "puffer_libtorch.h"
#include <torch/torch.h>

#include <iostream>
void c_libtorch_info()
{
    // Check if CUDA is available
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
    
    // Check if cuDNN is available
    std::cout << "cuDNN available: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;
    
    // Number of CUDA devices
    if (torch::cuda::is_available()) {
        std::cout << "Number of CUDA devices: " << torch::cuda::device_count() << std::endl;
    }
    
    
    // Create a tensor and check its device
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    torch::Tensor test_tensor = torch::zeros({2, 2}, device);
    std::cout << "Test tensor device: " << test_tensor.device() << std::endl;  
}

void c_eval()
{
    
}