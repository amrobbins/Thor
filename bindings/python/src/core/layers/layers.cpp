#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_drop_out(nb::module_ &m);
void bind_batch_normalization(nb::module_ &m);
void bind_concatenate(nb::module_ &m);
void bind_convolution_2d(nb::module_ &m);
void bind_flatten(nb::module_ &m);
void bind_fully_connected(nb::module_ &m);
void bind_network_input(nb::module_ &m);
void bind_network_output(nb::module_ &m);
void bind_pooling(nb::module_ &m);
void bind_reshape(nb::module_ &m);
void bind_stub(nb::module_ &m);
void bind_type_converter(nb::module_ &m);

void bind_debug(nb::module_ &m);

void bind_layers(nb::module_ &layers) {
    layers.doc() = "Thor layers";

    nb::class_<Layer>(layers, "Layer");
    nb::class_<MultiConnectionLayer, Layer>(layers, "MultiConnectionLayer");
    nb::class_<TrainableWeightsBiasesLayer, MultiConnectionLayer>(layers, "TrainableWeightsBiasesLayer");

    bind_debug(layers);

    bind_batch_normalization(layers);
    bind_drop_out(layers);
    bind_concatenate(layers);
    layers.def("Convolution2d", []() { return "temp"; });
    bind_flatten(layers);
    bind_fully_connected(layers);
    bind_network_input(layers);
    bind_network_output(layers);
    bind_pooling(layers);
    layers.def("Reshape", []() { return "temp"; });
    layers.def("Stub", []() { return "temp"; });
    bind_type_converter(layers);
}
