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
void bind_concatenate(nb::module_ &m);
void bind_convolution_2d(nb::module_ &m);
void bind_flatten(nb::module_ &m);
void bind_fully_connected(nb::module_ &m);
void bind_inception(nb::module_ &m);
void bind_network_input(nb::module_ &m);
void bind_network_output(nb::module_ &m);
void bind_pooling(nb::module_ &m);
void bind_reshape(nb::module_ &m);
void bind_stub(nb::module_ &m);
void bind_type_converter(nb::module_ &m);

void bind_layers(nb::module_ &layers) {
    using Thor::DropOut;
    using Thor::Layer;
    using Thor::Network;
    using Thor::Tensor;

    layers.doc() = "Thor layers";

    nb::class_<Layer>(layers, "Layer");

    bind_batch_normalization(layers);
    bind_drop_out(layers);
    bind_concatenate(layers);
    layers.def("Convolution2d", []() { return "temp"; });
    bind_flatten(layers);
    layers.def("FullyConnected", []() { return "temp"; });
    layers.def("Inception", []() { return "temp"; });
    bind_network_input(layers);
    bind_network_output(layers);
    layers.def("Pooling", []() { return "temp"; });
    layers.def("Reshape", []() { return "temp"; });
    layers.def("Stub", []() { return "temp"; });
    layers.def("TypeConverter", []() { return "temp"; });
}
