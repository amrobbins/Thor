#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

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
               const Tensor &feature_input,
               float exponential_running_average_factor,
               float epsilon) {
                BatchNormalization::Builder builder;

                BatchNormalization built = builder.network(network)
                                               .featureInput(feature_input)
                                               .exponentialRunningAverageFactor(exponential_running_average_factor)
                                               .epsilon(epsilon)
                                               .build();

                // Move the batchNormalization layer into the pre-allocated but uninitialized memory at self
                new (self) BatchNormalization(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "exponential_running_average_factor"_a,
            "epsilon"_a,
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
        .def("get_exponential_running_average_factor", &BatchNormalization::getExponentialRunningAverageFactor)
        .def("get_epsilon", &BatchNormalization::getEpsilon);
}