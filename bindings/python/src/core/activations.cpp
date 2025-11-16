#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_activations(py::module_ &m) {
    m.doc() = "Thor activations";
    m.def("Elu", []() { return "temp"; });
    m.def("Exponential", []() { return "temp"; });
    m.def("Gelu", []() { return "temp"; });
    m.def("HardSigmoid", []() { return "temp"; });
    m.def("Relu", []() { return "temp"; });
    m.def("Selu", []() { return "temp"; });
    m.def("Sigmoid", []() { return "temp"; });
    m.def("SoftMax", []() { return "temp"; });
    m.def("SoftPlus", []() { return "temp"; });
    m.def("SoftSign", []() { return "temp"; });
    m.def("Swish", []() { return "temp"; });
    m.def("Tanh", []() { return "temp"; });
}
