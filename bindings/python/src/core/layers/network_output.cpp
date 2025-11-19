#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "bindings/python/src/core/binding_types.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Thor::Tensor::DataType;

void bind_network_output(nb::module_ &m) {
    nb::class_<NetworkOutput, Layer>(m, "NetworkOutput")
        .def(
            "__init__",
            [](NetworkOutput *self, Network &network, const string &name, const Thor::Tensor &input_tensor, const DataType &data_type) {
                NetworkOutput::Builder builder;
                NetworkOutput built = builder.network(network).name(name).inputTensor(input_tensor).dataType(data_type).build();

                // Move the networkOutput layer into the pre-allocated but uninitialized memory at self
                new (self) NetworkOutput(std::move(built));
            },
            "network"_a,
            "name"_a,
            "input_tensor"_a,
            "data_type"_a,

            nb::sig("def __init__(self, "
                    "network: thor.Network, "
                    "name: str, "
                    "input_tensor: thor.Tensor, "
                    "data_type: thor.DataType"
                    ") -> None"),

            R"nbdoc(
            Create and attach a NetworkOutput to send data out of a Network.

            Parameters
            ----------
            network : thor.Network
                The network that the layer should be added to.
            name : str
                Name of this network output.
            input_tensor : thor.Tensor
                The tensor whose data the network output will send out of the network.
            data_type : thor.DataType
                Data type of the output tensor (e.g. thor.DataType.fp16).
            )nbdoc")
        .def("get_feature_output", &NetworkOutput::getFeatureOutput);
}