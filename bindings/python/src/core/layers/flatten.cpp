#include <nanobind/nanobind.h>

#include <optional>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_flatten(nb::module_ &m) {
    auto flatten = nb::class_<Flatten, Layer>(m, "Flatten");
    flatten.attr("__module__") = "thor.layers";

    flatten.def(
        "__init__",
        [](Flatten *self, Network &network, nb::object feature_input, uint32_t num_output_dimensions) {
            vector<uint64_t> dims;
            Flatten::Builder builder;
            builder.network(network);
            if (nb::isinstance<RaggedTensor>(feature_input)) {
                RaggedTensor ragged = nb::cast<RaggedTensor>(feature_input);
                dims = ragged.getTrailingDimensions();
                builder.featureInput(ragged);
            } else if (nb::isinstance<Tensor>(feature_input)) {
                Tensor tensor = nb::cast<Tensor>(feature_input);
                dims = tensor.getDimensions();
                builder.featureInput(tensor);
            } else {
                throw nb::type_error("Flatten feature_input must be thor.Tensor or thor.RaggedTensor.");
            }

            if (dims.empty()) throw nb::value_error("Flatten instance: feature_input must have at least 1 dimension.");
            if (num_output_dimensions == 0) {
                throw nb::value_error("Flatten instance: num_output_dimensions must be >= 1.");
            }
            if (num_output_dimensions >= dims.size()) {
                throw nb::value_error("Flatten instance: num_output_dimensions must be < rank of feature_input.");
            }

            Flatten built = builder.numOutputDimensions(num_output_dimensions).build();
            new (self) Flatten(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "num_output_dimensions"_a,
        R"nbdoc(
            Create and attach a Flatten layer to a Network.

            For ragged inputs, num_output_dimensions applies only to the trailing
            per-token value shape. The packed row axis and offsets are preserved.
            )nbdoc")
        .def(
            "get_feature_output",
            [](Flatten &self) -> nb::object {
                if (auto ragged = self.getRaggedFeatureOutput(); ragged.has_value()) return nb::cast(ragged.value());
                return nb::cast(self.getFeatureOutput().value());
            },
            R"nbdoc(
            Return the output tensor produced by this layer.
            )nbdoc");
}
