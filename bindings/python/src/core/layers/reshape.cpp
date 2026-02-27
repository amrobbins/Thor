#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_reshape(nb::module_ &m) {
    auto reshape = nb::class_<Reshape, Layer>(m, "Reshape");
    reshape.attr("__module__") = "thor.layers";

    reshape.def(
        "__init__",
        [](Reshape *self, Network &network, const Tensor &feature_input, vector<uint64_t> new_dimensions) {
            const auto &old_dims = feature_input.getDimensions();

            if (new_dimensions.empty()) {
                throw nb::value_error("Reshape instance: new_dimensions must be non-empty.");
            }

            // Validate new dims > 0 and compute element counts safely-ish
            auto mul_checked = [](uint64_t a, uint64_t b, uint64_t &out) -> bool {
                if (a == 0 || b == 0) {
                    out = 0;
                    return true;
                }
                if (a > std::numeric_limits<uint64_t>::max() / b)
                    return false;
                out = a * b;
                return true;
            };

            uint64_t old_elems = 1;
            for (size_t i = 0; i < old_dims.size(); ++i) {
                uint64_t tmp;
                if (!mul_checked(old_elems, old_dims[i], tmp)) {
                    throw nb::value_error("Reshape instance: overflow computing number of elements in feature_input.");
                }
                old_elems = tmp;
            }

            uint64_t new_elems = 1;
            for (size_t i = 0; i < new_dimensions.size(); ++i) {
                if (new_dimensions[i] == 0) {
                    string msg = "Reshape instance: new_dimensions must all be > 0, but new_dimensions[" + to_string(i) + "] == 0.";
                    throw nb::value_error(msg.c_str());
                }
                uint64_t tmp;
                if (!mul_checked(new_elems, new_dimensions[i], tmp)) {
                    throw nb::value_error("Reshape instance: overflow computing number of elements in new_dimensions.");
                }
                new_elems = tmp;
            }

            if (old_elems != new_elems) {
                string msg = "Reshape instance: number of elements must match. feature_input has " + to_string(old_elems) + " elements (" +
                             feature_input.getDescriptorString() + "), but new_dimensions has " + to_string(new_elems) + " elements.";
                throw nb::value_error(msg.c_str());
            }

            Reshape::Builder builder;
            Reshape built = builder.network(network).featureInput(feature_input).newDimensions(new_dimensions).build();

            // Move the reshape layer into the pre-allocated but uninitialized memory at self
            new (self) Reshape(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "new_dimensions"_a);

    reshape.def(
        "get_feature_output",
        [](Reshape &self) -> Tensor {
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

    reshape.attr("__doc__") = R"nbdoc(
            Create and attach a Reshape layer to a Network.
            The number of elements in the reshaped tensor must exactly equal the
            number of elements of the input tensor.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor
                Input feature tensor for this layer.
            new_dimensions : list[int]
                The new shape of the tensor.
            )nbdoc";
}
