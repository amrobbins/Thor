#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
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
void bind_custom_layer(nb::module_ &m);
void bind_flatten(nb::module_ &m);
void bind_fully_connected(nb::module_ &m);
void bind_network_input(nb::module_ &m);
void bind_network_output(nb::module_ &m);
void bind_pooling(nb::module_ &layers);
void bind_reshape(nb::module_ &m);
void bind_stub(nb::module_ &m);
void bind_type_converter(nb::module_ &m);

void bind_layers(nb::module_ &layers) {
    layers.doc() = "Thor layers";

    auto layer = nb::class_<Layer>(layers, "Layer");
    layer.attr("__module__") = "thor.layers";
    auto multi_connection_layer = nb::class_<MultiConnectionLayer, Layer>(layers, "MultiConnectionLayer");
    multi_connection_layer.attr("__module__") = "thor.layers";
    auto trainable_layer = nb::class_<TrainableLayer, MultiConnectionLayer>(layers, "TrainableLayer");
    trainable_layer.attr("__module__") = "thor.layers";

    bind_batch_normalization(layers);
    bind_drop_out(layers);
    bind_concatenate(layers);
    bind_convolution_2d(layers);
    bind_custom_layer(layers);
    bind_flatten(layers);
    bind_fully_connected(layers);
    bind_network_input(layers);
    bind_network_output(layers);
    bind_pooling(layers);
    bind_reshape(layers);
    bind_stub(layers);
    bind_type_converter(layers);
}
