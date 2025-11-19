#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "bindings/python/src/core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_concatenate(nb::module_ &m) {
    nb::class_<Concatenate, Layer>(m, "Concatenate")
        .def(
            "__init__",
            [](Concatenate *self, Network &network, TensorList feature_inputs, uint32_t concatenation_axis) {
                Concatenate::Builder builder;
                builder.network(network).concatenationAxis(concatenation_axis);
                // Iterate Python list and cast each element to Tensor&
                for (nb::handle h : feature_inputs) {
                    Tensor &t = nb::cast<Tensor &>(h);
                    builder.featureInput(t);
                }

                // Move the concatenate layer into the pre-allocated but uninitialized memory at self
                new (self) Concatenate(std::move(builder.build()));
            },
            "network"_a,
            "feature_inputs"_a,
            "concatenation_axis"_a,

            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "feature_inputs: list[thor.Tensor], "
                    "concatenation_axis: int"
                    ") -> None"),

            R"nbdoc(
            Create and attach a Concatenate layer to a Network.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_inputs : list[thor.Tensor]
                List of input feature tensors for this layer.
            concatenation_axis : int
                Axis along which to concatenate the input tensors.

            For example, if your input tensors have dimensions:
                1. [2, 4, 5, 7]
                2. [2, 6, 5, 7]
                3. [2, 2, 5, 7]

            with concatenation_axis=1, then your output tensor will have dimensions:
                [2, 12, 5, 7]

            Note that all dimensions must match to perform Contcatenate, except for the concatenation_axis dimension.

            )nbdoc");
}