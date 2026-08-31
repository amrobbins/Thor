#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/StopGradient.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

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
        [](StopGradient *self, Network &network, nb::object feature_input) {
            StopGradient::Builder builder;
            builder.network(network);
            if (nb::isinstance<RaggedTensor>(feature_input)) {
                builder.featureInput(nb::cast<RaggedTensor>(feature_input));
            } else if (nb::isinstance<Tensor>(feature_input)) {
                builder.featureInput(nb::cast<Tensor>(feature_input));
            } else {
                throw nb::type_error("StopGradient feature_input must be thor.Tensor or thor.RaggedTensor.");
            }
            StopGradient built = builder.build();
            new (self) StopGradient(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        R"nbdoc(
Create and attach a StopGradient layer to a Network.

Forward is an identity alias of ``feature_input``. Backward does not propagate
an error tensor through this layer, making the gradient barrier explicit in the
network graph. Ragged inputs preserve their exact row partition.
)nbdoc");

    stop_gradient.def(
        "get_feature_output",
        [](StopGradient &self) -> nb::object {
            if (std::optional<RaggedTensor> raggedOutput = self.getRaggedFeatureOutput(); raggedOutput.has_value()) {
                return nb::cast(raggedOutput.value());
            }
            return nb::cast(self.getFeatureOutput().value());
        },
        R"nbdoc(
Return the logical output produced by this layer. Ragged inputs return a thor.RaggedTensor.
)nbdoc");

    stop_gradient.def("get_use_ragged", &StopGradient::getUseRagged);
}
