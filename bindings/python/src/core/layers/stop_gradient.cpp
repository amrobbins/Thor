#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/StopGradient.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_stop_gradient(nb::module_ &m) {
    auto stop_gradient = nb::class_<StopGradient, Layer>(m, "StopGradient");
    stop_gradient.attr("__module__") = "thor.layers";

    stop_gradient.def(
        "__init__",
        [](StopGradient *self, Network &network, const Tensor &feature_input) {
            StopGradient::Builder builder;
            StopGradient built = builder.network(network).featureInput(feature_input).build();
            new (self) StopGradient(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        R"nbdoc(
Create and attach a StopGradient layer to a Network.

Forward is an identity alias of ``feature_input``. Backward does not propagate
an error tensor through this layer, making the gradient barrier explicit in the
network graph.
)nbdoc");

    stop_gradient.def(
        "get_feature_output",
        [](StopGradient &self) -> Tensor {
            std::optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(
Return the output tensor produced by this layer.
)nbdoc");
}
