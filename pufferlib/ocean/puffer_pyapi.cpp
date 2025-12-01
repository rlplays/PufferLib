// Include the CPP here so the tests etc don't have to pull in pybind etc.
#include <puffer_libtorch.cpp>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Suggested by Claude to avoid pybind/C++ using import_array/numpy here while env_binding uses just the PyAPI alone (using non pybind).
#define PY_ARRAY_UNIQUE_SYMBOL puffer_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

extern "C" PyMethodDef* get_methods();

PYBIND11_MODULE(binding, m)
{
    m.doc() = "PufferLib Libtorch API";

    import_array();
    PyModule_AddFunctions(m.ptr(), get_methods());
    m.def("libtorch_info", &c_libtorch_info, "Print libtorch info to stdout.");
    m.def("print_tensor_info", &c_print_tensor_info, py::arg("tensor"), "Print tensor info to stdout.");
}


