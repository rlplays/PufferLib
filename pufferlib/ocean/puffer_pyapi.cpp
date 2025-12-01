// Include the CPP here so the tests etc don't have to pull in pybind etc.
#include <puffer_libtorch.cpp>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Suggested by Claude to avoid pybind/C++ using import_array/numpy here while env_binding uses just the PyAPI alone (using non pybind).
#define PY_ARRAY_UNIQUE_SYMBOL puffer_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "puffer_libtorch.h"
#include <torch/torch.h>
#include "puffernet.h"
#include <cassert>
#include <iostream>

using torch::Tensor;

// Forward declaration for env_multithread.h stuff to avoid circular references. Especially as binding.c (C only) includes C code that wraps C++ code/objects underneath.
extern "C" PyMethodDef* get_methods();
extern "C" struct PufferTorch* get_puffertorch(VecEnv* vec_env);
extern "C" struct PufferEnvState* get_envstate(VecEnv* vec_env, int env_index);

void c_torch_start_eval_lstm(VecEnv* vec_env, Tensor weight_ih, Tensor weight_hh, Tensor bias_ih, Tensor bias_hh)
{
  torch::NoGradGuard no_grad;
  PufferTorch* puff_torch = get_puffertorch(vec_env);
  PUFFER_ASSERT(puff_torch != nullptr && puff_torch->model != nullptr, "Invalid state.");
  puff_torch->model->start_eval_lstm(weight_ih, weight_hh, bias_ih,  bias_hh);
}

PYBIND11_MODULE(binding, m)
{
    m.doc() = "PufferLib Libtorch API";

    import_array();
    PyModule_AddFunctions(m.ptr(), get_methods());
    m.def("libtorch_info", &c_libtorch_info, "Print libtorch info to stdout.");
    m.def("print_tensor_info", &c_print_tensor_info, py::arg("tensor"), "Print tensor info to stdout.");
    m.def("torch_start_eval_lstm", &c_torch_start_eval_lstm, py::arg("tensor"), "Start the initial torch eval.");
}


