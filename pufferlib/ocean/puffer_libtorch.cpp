#include "puffer_libtorch.h"
#include <torch/torch.h>

#include <iostream>
void c_eval()
{
    torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}