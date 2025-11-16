#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_layers(py::module_ &m) {
    m.doc() = "Thor layers";
    m.def("BatchNormalization", []() { return "temp"; });
    m.def("Concatenate", []() { return "temp"; });
    m.def("Convolution2d", []() { return "temp"; });
    m.def("DropOut", []() { return "temp"; });
    m.def("Flatten", []() { return "temp"; });
    m.def("FullyConnected", []() { return "temp"; });
    m.def("Inception", []() { return "temp"; });
    m.def("NetworkInput", []() { return "temp"; });
    m.def("NetworkOutput", []() { return "temp"; });
    m.def("Pooling", []() { return "temp"; });
    m.def("Reshape", []() { return "temp"; });
    m.def("Stub", []() { return "temp"; });
    m.def("TypeConverter", []() { return "temp"; });
}
