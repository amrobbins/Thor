#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_batch_normalization(nb::module_ &m) {
    auto batch_normalization = nb::class_<BatchNormalization, TrainableWeightsBiasesLayer>(m, "BatchNormalization");
    batch_normalization.attr("__module__") = "thor.layers";

    batch_normalization.def(
        "__init__",
        [](BatchNormalization *self, Network &network, Tensor feature_input, float exponential_running_average_factor, float epsilon) {
            // EMA factor bounds: (0, 1]
            if (!(exponential_running_average_factor > 0.0f && exponential_running_average_factor <= 1.0f)) {
                string msg =
                    "BatchNormalization instance: exponential_running_average_factor must satisfy "
                    "0 < factor <= 1. factor: " +
                    to_string(exponential_running_average_factor);
                throw nb::value_error(msg.c_str());
            }

            // Epsilon must be positive
            if (!(epsilon > 0.0f)) {
                string msg = "BatchNormalization instance: epsilon must be > 0. epsilon: " + to_string(epsilon);
                throw nb::value_error(msg.c_str());
            }

            BatchNormalization::Builder builder;
            builder.network(network).exponentialRunningAverageFactor(exponential_running_average_factor).epsilon(epsilon);

            builder.featureInput(feature_input);

            // Move the batchNormalization layer into the pre-allocated but uninitialized memory at self
            new (self) BatchNormalization(std::move(builder.build()));
        },
        "network"_a,
        "feature_input"_a,
        "exponential_running_average_factor"_a = 0.05,
        "epsilon"_a = 0.0001f);

    batch_normalization.def(
        "get_feature_output",
        [](BatchNormalization &self) -> Tensor {
            Optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.get();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor
                The feature output tensor handle.
            )nbdoc");

    batch_normalization.def("get_exponential_running_average_factor", [](BatchNormalization &self) -> optional<float> {
        Optional<double> maybe_eraf = self.getExponentialRunningAverageFactor();
        if (maybe_eraf.isPresent())
            return maybe_eraf.get();
        return optional<float>();
    });

    batch_normalization.def("get_epsilon", [](BatchNormalization &self) -> optional<float> {
        Optional<double> maybe_epsilon = self.getEpsilon();
        if (maybe_epsilon.isPresent())
            return maybe_epsilon.get();
        return optional<float>();
    });

    batch_normalization.attr("__doc__") = R"nbdoc(
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
            )nbdoc";
}
