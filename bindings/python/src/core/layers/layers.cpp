#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

void bind_drop_out(nb::module_ &m);
void bind_batch_normalization(nb::module_ &m);

void bind_layers(nb::module_ &layers) {
    using Thor::DropOut;
    using Thor::Layer;
    using Thor::Network;
    using Thor::Tensor;

    layers.doc() = "Thor layers";

    nb::class_<Layer>(layers, "Layer");

    bind_batch_normalization(layers);
    bind_drop_out(layers);

    layers.def("Concatenate", []() { return "temp"; });
    layers.def("Convolution2d", []() { return "temp"; });
    layers.def("Flatten", []() { return "temp"; });
    layers.def("FullyConnected", []() { return "temp"; });
    layers.def("Inception", []() { return "temp"; });
    layers.def("NetworkInput", []() { return "temp"; });
    layers.def("NetworkOutput", []() { return "temp"; });
    layers.def("Pooling", []() { return "temp"; });
    layers.def("Reshape", []() { return "temp"; });
    layers.def("Stub", []() { return "temp"; });
    layers.def("TypeConverter", []() { return "temp"; });
}
