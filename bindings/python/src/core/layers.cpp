#include <nanobind/nanobind.h>
namespace nb = nanobind;

void bind_layers(nb::module_ &m) {
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
