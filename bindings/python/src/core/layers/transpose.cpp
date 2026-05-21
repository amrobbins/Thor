#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Transpose.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <optional>
#include <string>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_transpose(nb::module_ &m) {
    auto transpose = nb::class_<Transpose, Layer>(m, "Transpose");
    transpose.attr("__module__") = "thor.layers";

    transpose.def(
        "__init__",
        [](Transpose *self, Network &network, const Tensor &feature_input) {
            const auto &dims = feature_input.getDimensions();
            if (dims.size() < 2) {
                throw nb::value_error("Transpose instance: feature_input must have rank >= 2.");
            }

            Transpose::Builder builder;
            Transpose built = builder.network(network).featureInput(feature_input).build();
            new (self) Transpose(std::move(built));
        },
        "network"_a,
        "feature_input"_a);

    transpose.def(
        "get_feature_output",
        [](Transpose &self) -> Tensor {
            std::optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor
                The feature output tensor handle.
            )nbdoc");

    transpose.attr("__doc__") = R"nbdoc(
            Create and attach a Transpose layer to a Network.

            The layer swaps the last two feature dimensions. The network batch
            dimension is preserved by the underlying physical expression, so a
            feature tensor shaped [X, Y] is materialized as [Y, X], while the
            stamped physical tensor behaves as [B, X, Y] -> [B, Y, X].

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor
                Input feature tensor for this layer. Must have rank >= 2.
            )nbdoc";
}
