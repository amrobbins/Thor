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
    nb::class_<Reshape, Layer>(m, "Reshape")
        .def(
            "__init__",
            [](Reshape *self, Network &network, const Tensor &feature_input, vector<uint64_t> new_dimensions) {
                Reshape::Builder builder;

                Reshape built = builder.network(network).featureInput(feature_input).newDimensions(new_dimensions).build();

                // Move the reshape layer into the pre-allocated but uninitialized memory at self
                new (self) Reshape(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "new_dimensions"_a,

            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "feature_input: thor.Tensor, "
                    "new_dimensions: list[int]"
                    ") -> None"),

            R"nbdoc(
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
            )nbdoc");
}