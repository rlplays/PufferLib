// Include the CPP here so the tests etc don't have to pull in pybind etc.
#include <puffer_nativecpp.cpp>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Suggested by Claude to avoid pybind/C++ using import_array/numpy here while env_binding uses just the PyAPI alone (using non pybind).
#define PY_ARRAY_UNIQUE_SYMBOL puffer_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "puffer_nativecpp.h"
#include <torch/torch.h>
#include "puffernet.h"
#include <cassert>
#include <iostream>

using torch::Tensor;

// Forward declaration for env_multithread.h stuff to avoid circular references. Especially as binding.c (C only) includes C code that wraps C++ code/objects underneath.
extern "C" PyMethodDef* get_c_env_binding_methods();

PYBIND11_MODULE(binding, m)
{
  m.doc() = "PufferLib Libtorch API";

  import_array();
  PyModule_AddFunctions(m.ptr(), get_c_env_binding_methods());
  m.def("libtorch_info", &c_libtorch_info, "Print libtorch info to stdout.");
  m.def("print_tensor_info", &c_print_tensor_info, py::arg("tensor"), "Print tensor info to stdout.");
  m.def("torch_start_eval_lstm", &c_torch_start_eval_lstm,
      py::arg("vec_env"),
      py::arg("obs_torch"), // Full observation tensor across all envs.
      py::arg("encoder_linear_w"),
      py::arg("encoder_linear_b"),
      py::arg("decoder_linear_w"),
      py::arg("decoder_linear_b"),
      py::arg("value_w"),
      py::arg("value_b"),
      py::arg("weight_ih"),
      py::arg("weight_hh"),
      py::arg("bias_ih"),
      py::arg("bias_hh"),
      "Start the initial torch eval (before starting the horizon segments).");

 m.def("torch_finish_eval_lstm", &c_torch_finish_eval_lstm,
      py::arg("vec_env"),
      "Finish the torch eval (after all segments in the horizon are done).");
}


