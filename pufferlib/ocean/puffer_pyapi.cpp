// Include the CPP here so the tests etc don't have to pull in pybind etc.
#include <puffer_libtorch.cpp>
#include <pybind11/pybind11.h>

extern "C" PyMethodDef* get_methods();

// NumPy import_array needs special handling - it may return or set error
static int init_numpy() {
    import_array1(-1);
    return 0;
}

PYBIND11_MODULE(binding, m)
{
    m.doc() = "PufferLib Libtorch API";

    if (init_numpy() < 0) {
        throw pybind11::error_already_set();
    }

    PyModule_AddFunctions(m.ptr(), get_methods());
    m.def("libtorch_info", &c_libtorch_info, "Print libtorch info to stdout.");

    pybind11::class_<PufferTorch>(m, "PufferTorch")
        .def_static("alloc", &c_torch_alloc, "Allocate a PufferTorch model.", pybind11::arg("opt"))
        .def("free", &c_torch_free, "Free the PufferTorch model.");

    pybind11::class_<PufferEnvState>(m, "PufferEnvState")
        .def_static("initenv", &c_initenv, "Initialize a new environment state.", pybind11::arg("pt"))
        .def("freeenv", &c_freeenv, "Free the environment state.", pybind11::arg("pt"));
}


