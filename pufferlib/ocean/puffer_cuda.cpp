#include <puffer_cuda.h>

#include <pybind11/pybind11.h>
#include <torch/extension.h>


PYBIND11_MODULE(native, m)
{
  m.doc() = "PufferLib native CUDA API for testing purposes.";
  m.def("launch_dual_linear_forward", &launch_dual_linear_forward, py::arg("input"), py::arg("weight1"),
        py::arg("bias1"), py::arg("output1"), py::arg("weight2"), py::arg("bias2"), py::arg("output2"),
        "Launches a dual linear forward that outputs decoder (logits) + values.");
  m.def("launch_sample_logits_kernel", &launch_sample_logits_kernel, py::arg("random_vals"), py::arg("sizes_gpu"),
        py::arg("offsets_gpu"), py::arg("logits"), py::arg("num_actions"), py::arg("actions"),
        py::arg("logprobs"),
        "Launches a logits kernels that samples actions into output actions and assigns logprobs for those actions.");
}
