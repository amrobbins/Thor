#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_scale_gradient(nb::module_ &m) {
    auto scale_gradient = nb::class_<ScaleGradient, Layer>(m, "ScaleGradient");
    scale_gradient.attr("__module__") = "thor.layers";

    scale_gradient.def(
        "__init__",
        [](ScaleGradient *self, Network &network, const Tensor &feature_input, float scale) {
            ScaleGradient::Builder builder;
            ScaleGradient built = builder.network(network).featureInput(feature_input).scale(scale).build();
            new (self) ScaleGradient(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "scale"_a,
        R"nbdoc(
Create and attach a ScaleGradient layer to a Network.

Forward is an identity alias of ``feature_input``. During backward propagation,
the gradient passed upstream is multiplied by ``scale``. The downstream branch
and any trainable layers after ScaleGradient receive their ordinary gradients.

``scale=0`` blocks the numerical gradient while preserving a backward tensor
path. Negative scales are allowed and can be used for gradient reversal.
)nbdoc");

    scale_gradient.def(
        "get_feature_output",
        [](ScaleGradient &self) -> Tensor {
            std::optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            return maybeFeatureOutput.value();
        },
        R"nbdoc(
Return the output tensor produced by this layer.
)nbdoc");

    scale_gradient.def("get_scale", &ScaleGradient::getScale, R"nbdoc(Return the backward gradient scale.)nbdoc");
}
