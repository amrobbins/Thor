#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_drop_out(nb::module_ &m);
void bind_adaptive_layer_norm(nb::module_ &m);
void bind_attention(nb::module_ &m);
void bind_batch_normalization(nb::module_ &m);
void bind_concatenate(nb::module_ &m);
void bind_convolution_2d(nb::module_ &m);
void bind_convolution_3d(nb::module_ &m);
void bind_custom_layer(nb::module_ &m);
void bind_flatten(nb::module_ &m);
void bind_finite_check(nb::module_ &m);
void bind_embedding(nb::module_ &m);
void bind_fully_connected(nb::module_ &m);
void bind_instance_norm(nb::module_ &m);
void bind_layer_norm(nb::module_ &m);
void bind_rms_norm(nb::module_ &m);
void bind_scaled_dot_product_attention(nb::module_ &m);
void bind_stop_gradient(nb::module_ &m);
void bind_network_input(nb::module_ &m);
void bind_network_output(nb::module_ &m);
void bind_pooling(nb::module_ &layers);
void bind_reshape(nb::module_ &m);
void bind_stub(nb::module_ &m);
void bind_type_converter(nb::module_ &m);
void bind_transpose(nb::module_ &m);

void bind_layers(nb::module_ &layers) {
    layers.doc() = "Thor layers";

    auto layer = nb::class_<Layer>(layers, "Layer");
    layer.attr("__module__") = "thor.layers";
    layer.def("get_id", &Layer::getId);
    auto multi_connection_layer = nb::class_<MultiConnectionLayer, Layer>(layers, "MultiConnectionLayer");
    multi_connection_layer.attr("__module__") = "thor.layers";
    auto trainable_layer = nb::class_<TrainableLayer, MultiConnectionLayer>(layers, "TrainableLayer");
    trainable_layer.attr("__module__") = "thor.layers";
    trainable_layer.def("freeze_training", &TrainableLayer::freezeTraining);
    trainable_layer.def("unfreeze_training", &TrainableLayer::unfreezeTraining);
    trainable_layer.def("is_training_frozen", &TrainableLayer::isTrainingFrozen);
    trainable_layer.def("get_parameters", &TrainableLayer::getParameters, nb::rv_policy::reference_internal);
    trainable_layer.def("get_bound_parameter", &TrainableLayer::getBoundParameter, "placed_network"_a, "name"_a);
    trainable_layer.def("get_bound_parameters", &TrainableLayer::getBoundParameters, "placed_network"_a);
    trainable_layer.def("get_parameter_reference", &TrainableLayer::getParameterReference, "name"_a);
    trainable_layer.def("get_parameter_references", &TrainableLayer::getParameterReferences, "trainable_only"_a = true, "training_enabled_only"_a = true);

    bind_custom_layer(layers);
    bind_adaptive_layer_norm(layers);
    bind_attention(layers);
    bind_batch_normalization(layers);
    bind_drop_out(layers);
    bind_concatenate(layers);
    bind_convolution_2d(layers);
    bind_convolution_3d(layers);
    bind_flatten(layers);
    bind_finite_check(layers);
    bind_embedding(layers);
    bind_fully_connected(layers);
    bind_instance_norm(layers);
    bind_layer_norm(layers);
    bind_rms_norm(layers);
    bind_scaled_dot_product_attention(layers);
    bind_stop_gradient(layers);
    bind_network_input(layers);
    bind_network_output(layers);
    bind_pooling(layers);
    bind_reshape(layers);
    bind_stub(layers);
    bind_type_converter(layers);
    bind_transpose(layers);
}
