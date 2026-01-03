#include <puffer_cuda.h>

#include <pybind11/pybind11.h>
#include <torch/extension.h>



PYBIND11_MODULE(native, m)
{
  m.doc() = "PufferLib native CUDA API for testing purposes.";
  m.def("launch_linear_forward", &launch_linear_forward, py::arg("input"), 
      py::arg("weight"), py::arg("bias"), py::arg("output"),
    "Launches a linear forward cuda kernel.");
}

