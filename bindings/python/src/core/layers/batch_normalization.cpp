#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "bindings/python/src/core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_batch_normalization(nb::module_ &m) {
    nb::class_<BatchNormalization, Layer>(m, "BatchNormalization")
        .def(
            "__init__",
            [](BatchNormalization *self,
               Network &network,
               TensorList feature_inputs,
               float exponential_running_average_factor = 0.05,
               float epsilon = 0.0001) {
                BatchNormalization::Builder builder;
                builder.network(network).exponentialRunningAverageFactor(exponential_running_average_factor).epsilon(epsilon);

                for (nb::handle h : feature_inputs) {
                    Tensor &t = nb::cast<Tensor &>(h);
                    builder.featureInput(t);
                }

                // Move the batchNormalization layer into the pre-allocated but uninitialized memory at self
                new (self) BatchNormalization(std::move(builder.build()));
            },
            "network"_a,
            "feature_input"_a,
            "exponential_running_average_factor"_a,
            "epsilon"_a,

            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "feature_inputs: list[thor.Tensor], "
                    "exponential_running_average_factor: float = 0.05, "
                    "epsilon: float = 0.0001"
                    ") -> None"),

            R"nbdoc(
            Create and attach a BatchNormalization layer to a Network.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor
                Input feature tensor for this layer.
            exponential_running_average_factor : float
                FIXME.
            epsilon : float
                FIXME.
            )nbdoc")
        //.def("get_feature_output", &BatchNormalization::getFeatureOutput)
        .def("get_feature_outputs", &BatchNormalization::getFeatureOutputs)
        .def("get_exponential_running_average_factor", &BatchNormalization::getExponentialRunningAverageFactor)
        .def("get_epsilon", &BatchNormalization::getEpsilon);
}