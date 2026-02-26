#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Network/Network.h"
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
        [](Flatten *self, Network &network, const Tensor &feature_input, uint32_t num_output_dimensions) {
            const auto &dims = feature_input.getDimensions();
            const size_t rank = dims.size();

            if (rank == 0) {
                string msg = "Flatten instance: feature_input must have at least 1 dimension but tensor format is " +
                             feature_input.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            if (num_output_dimensions == 0) {
                string msg =
                    "Flatten instance: num_output_dimensions must be >= 1. You passed 0. "
                    "feature_input is " +
                    feature_input.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            if (num_output_dimensions >= rank) {
                string msg =
                    "Flatten instance: num_output_dimensions must be < rank of feature_input. "
                    "You passed num_output_dimensions " +
                    to_string(num_output_dimensions) + " for rank " + to_string(rank) + ". feature_input is " +
                    feature_input.getDescriptorString();
                throw nb::value_error(msg.c_str());
            }

            Flatten::Builder builder;
            Flatten built = builder.network(network).featureInput(feature_input).numOutputDimensions(num_output_dimensions).build();

            // Move the flatten layer into the pre-allocated but uninitialized memory at self
            new (self) Flatten(std::move(built));
        },
        "network"_a,
        "feature_input"_a,
        "num_output_dimensions"_a,

        R"nbdoc(
            Create and attach a Flatten layer to a Network.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor
                Input feature tensor for this layer.
            num_output_dimensions : float
                The number of leading dimesions that are kept from the input tensor by the output tensor.

                For example if the input Tensor has dimensions [10, 20, 30, 40] and num_output_dimensions == 2,
                then the ouput Tensor will have dimensions [10, 20, 1200].
            )nbdoc");
}
