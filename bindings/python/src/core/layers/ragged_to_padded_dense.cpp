#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Utility/RaggedToPaddedDense.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_ragged_to_padded_dense(nb::module_& m) {
    auto layer = nb::class_<RaggedToPaddedDense, MultiConnectionLayer>(m, "RaggedToPaddedDense");
    layer.attr("__module__") = "thor.layers";
    layer.def(
        "__init__",
        [](RaggedToPaddedDense* self, Network& network, RaggedTensor featureInput, double paddingValue) {
            RaggedToPaddedDense built = RaggedToPaddedDense::Builder()
                                            .network(network)
                                            .featureInput(featureInput)
                                            .paddingValue(paddingValue)
                                            .build();
            new (self) RaggedToPaddedDense(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "padding_value"_a = 0.0,
        R"nbdoc(
Convert a rank-1 RaggedTensor to a normal padded dense tensor.

``feature_input`` must declare ``max_values_per_row``. The logical output shape
is ``[max_values_per_row, *trailing]`` and the physical stamped shape is
``[B, max_values_per_row, *trailing]``. Active tokens are copied row-by-row and
all remaining padded positions are filled with ``padding_value``. Backward
ignores gradients in padded positions and returns only active packed gradients.
)nbdoc");
    layer.def("get_feature_output", &RaggedToPaddedDense::getPaddedFeatureOutput);
    layer.def("get_feature_input", &RaggedToPaddedDense::getRaggedFeatureInput);
    layer.def_prop_ro("padding_value", &RaggedToPaddedDense::getPaddingValue);
}
