#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <nanobind/stl/vector.h>
#include <limits>
#include <optional>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

namespace {

uint64_t checkedElements(const vector<uint64_t>& dims, const char* what) {
    uint64_t elements = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == 0) {
            const string suffix =
                string(what) == "new_dimensions" ? " must all be > 0." : " dimensions must all be > 0.";
            throw nb::value_error((string("Reshape instance: ") + what + suffix).c_str());
        }
        if (elements > numeric_limits<uint64_t>::max() / dims[i]) {
            throw nb::value_error((string("Reshape instance: overflow computing number of elements in ") + what + ".").c_str());
        }
        elements *= dims[i];
    }
    return elements;
}

}  // namespace

void bind_reshape(nb::module_ &m) {
    auto reshape = nb::class_<Reshape, Layer>(m, "Reshape");
    reshape.attr("__module__") = "thor.layers";

    reshape.def(
        "__init__",
        [](Reshape *self, Network &network, nb::object feature_input, vector<uint64_t> new_dimensions) {
            if (new_dimensions.empty()) throw nb::value_error("Reshape instance: new_dimensions must be non-empty.");
            const uint64_t new_elements = checkedElements(new_dimensions, "new_dimensions");

            Reshape::Builder builder;
            builder.network(network).newDimensions(new_dimensions);
            if (nb::isinstance<RaggedTensor>(feature_input)) {
                RaggedTensor ragged = nb::cast<RaggedTensor>(feature_input);
                if (checkedElements(ragged.getTrailingDimensions(), "ragged trailing input") != new_elements) {
                    throw nb::value_error("Reshape instance: ragged reshape must preserve elements per packed value.");
                }
                builder.featureInput(ragged);
            } else if (nb::isinstance<Tensor>(feature_input)) {
                Tensor tensor = nb::cast<Tensor>(feature_input);
                if (checkedElements(tensor.getDimensions(), "feature_input") != new_elements) {
                    throw nb::value_error("Reshape instance: number of elements must match.");
                }
                builder.featureInput(tensor);
            } else {
                throw nb::type_error("Reshape feature_input must be thor.Tensor or thor.RaggedTensor.");
            }

            Reshape built = builder.build();
            new (self) Reshape(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "new_dimensions"_a);

    reshape.def(
        "get_feature_output",
        [](Reshape &self) -> nb::object {
            if (auto ragged = self.getRaggedFeatureOutput(); ragged.has_value()) return nb::cast(ragged.value());
            return nb::cast(self.getFeatureOutput().value());
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor or thor.RaggedTensor
                Ragged inputs preserve their exact row partition and reshape only
                the trailing per-token value dimensions.
            )nbdoc");

    reshape.attr("__doc__") = R"nbdoc(
            Create and attach a Reshape layer to a Network.

            Dense inputs reshape the complete feature tensor. Ragged inputs reshape
            only the trailing dimensions of each packed value; the packed row axis
            and canonical offsets are preserved exactly.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor or thor.RaggedTensor
                Input feature tensor for this layer.
            new_dimensions : list[int]
                Dense output feature shape, or for ragged input the new per-token
                trailing shape. Element count must be preserved.
            )nbdoc";
}
