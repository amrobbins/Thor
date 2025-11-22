#include <nanobind/nanobind.h>

#include <optional>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Thor::Tensor::DataType;

Activation::Builder *getDefaultActivation() {
    static Relu::Builder defaultBuilder;
    return &defaultBuilder;
}

// void bind_debug(nb::module_ &m) {
//     m.def("debug_fullyconnected_layout", []() {
//         FullyConnected dummy;
//         std::size_t size = sizeof(FullyConnected);
//         std::size_t offset =
//             reinterpret_cast<char*>(&dummy.featureInputs) -
//             reinterpret_cast<char*>(&dummy);
//         //return std::make_tuple(size, offset);
//         printf("size %ld offset %ld\n", size, offset);
//     });
// }

void bind_fully_connected(nb::module_ &m) {
    nb::class_<FullyConnected, TrainableWeightsBiasesLayer>(m, "FullyConnected")
        .def(
            "__init__",
            [](FullyConnected *self,
               Network &network,
               Tensor feature_input,
               uint32_t numOutputFeatures,
               bool hasBias,
               optional<Activation::Builder *> activation) {
                FullyConnected::Builder builder;
                builder.network(network).featureInput(feature_input).numOutputFeatures(numOutputFeatures).hasBias(hasBias);

                if (!activation.has_value()) {
                    // Explicitly no activation applied
                    builder.noActivation();
                } else {
                    builder.activationBuilder(*(activation.value()));
                }

                FullyConnected built = builder.build();
                new (self) FullyConnected(std::move(built));
            },
            "network"_a,
            "feature_input"_a,
            "num_output_features"_a,
            "has_bias"_a = true,
            nb::arg("activation").none() = getDefaultActivation(),
            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "feature_input: thor.Tensor, "
                    "num_output_features: int, "
                    "has_bias: bool = True, "
                    "activation: thor.Activation | None = Relu()"
                    ") -> None"),

            R"nbdoc(
            Create and attach a FullyConnected to send data into a Network.

            Parameters
            ----------
            network : thor.Network
                The network that the layer should be added to.
            name : str
                Name of this network input.
            dimensions : list[int]
                Dimension sizes for the input tensor **excluding** the batch dimension.
                The batch dimension is added later when compiling the network.
                Note: the batch dimension is never specified in API layer tensors,
                      the batch dimension is only added when stamping down a physical network instance.
            data_type : thor.DataType
                Data type of the input tensor (e.g. thor.Tensor.DataType.fp16).
            )nbdoc");
}
