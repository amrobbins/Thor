#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "bindings/python/src/core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Thor::Tensor::DataType;

void bind_network_input(nb::module_ &m) {
    nb::class_<NetworkInput, Layer>(m, "NetworkInput")
        .def(
            "__init__",
            [](NetworkInput *self, Network &network, const string &name, const vector<uint64_t> &dimensions, const DataType &data_type) {
                NetworkInput::Builder builder;
                NetworkInput built = builder.network(network).name(name).dimensions(dimensions).dataType(data_type).build();

                // Move the networkInput layer into the pre-allocated but uninitialized memory at self
                new (self) NetworkInput(std::move(built));
            },
            "network"_a,
            "name"_a,
            "dimensions"_a,
            "data_type"_a,

            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "name: str, "
                    "dimensions: list[int], "
                    "data_type: thor.DataType"
                    ") -> None"),

            R"nbdoc(
            Create and attach a NetworkInput to send data into a Network.

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
                Data type of the input tensor (e.g. thor.DataType.fp16).
            )nbdoc")
        .def("get_feature_output", &NetworkInput::getFeatureOutput);
}