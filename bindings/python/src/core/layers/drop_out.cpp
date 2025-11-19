#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_drop_out(nb::module_ &m) {
    nb::class_<DropOut, Layer>(m, "DropOut")
        .def(
            "__init__",
            [](DropOut *self, Network &network, const Tensor &feature_input, float drop_proportion) {
                DropOut::Builder builder;

                DropOut built = builder.network(network).featureInput(feature_input).dropProportion(drop_proportion).build();

                // Move the dropout layer into the pre-allocated but uninitialized memory at self
                new (self) DropOut(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "drop_proportion"_a,

            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "feature_input: thor.Tensor, "
                    "drop_proportion: float"
                    ") -> None"),

            R"nbdoc(
            Create and attach a DropOut layer to a Network.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor
                Input feature tensor for this layer.
            drop_proportion : float
                Fraction of units to drop (0.0 <= p <= 1.0).
            )nbdoc")
        .def("get_drop_proportion", &DropOut::getDropProportion);
}