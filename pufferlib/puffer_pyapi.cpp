// Include the CPP here so the tests etc don't have to pull in pybind etc.
#include <puffer_libtorch.cpp>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(puffer_pyapi, m)
{
    m.doc() = "PufferLib Libtorch API";

    m.def("libtorch_info", &pufferlib::c_libtorch_info, "Print libtorch info to stdout.");

    pybind11::class_<pufferlib::PufferTorch>(m, "PufferTorch")
        .def_static("alloc", &pufferlib::c_torch_alloc, "Allocate a PufferTorch model.", pybind11::arg("opt"))
        .def("load_weights", &pufferlib::c_torch_load_weights, "Load weights into the model.", pybind11::arg("weights"))
        .def("free", &pufferlib::c_torch_free, "Free the PufferTorch model.");

    pybind11::class_<pufferlib::PufferEnvState>(m, "PufferEnvState")
        .def_static("initenv", &pufferlib::c_initenv, "Initialize a new environment state.", pybind11::arg("pt"))
        .def("freeenv", &pufferlib::c_freeenv, "Free the environment state.", pybind11::arg("pt"));
}


