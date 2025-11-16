#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_layers(py::module_ &m) {
    m.doc() = "Thor layers";
    m.def("fully_connected", []() { return "temp"; });
    m.def("convolution_2d", []() { return "temp"; });
}
