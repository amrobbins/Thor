#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::TensorDescriptor::DataType;

void bind_network_output(nb::module_ &m) {
    auto network_output = nb::class_<NetworkOutput, Layer>(m, "NetworkOutput");
    network_output.attr("__module__") = "thor.layers";

    network_output.def(
        "__init__",
        [](NetworkOutput *self, Network &network, const string &name, const Tensor &input_tensor, const DataType &data_type) {
            if (name.length() == 0) {
                string msg = "Network Output instance: name must have non-zero length but name=\"\" was passed in.";
                throw nb::value_error(msg.c_str());
            }

            NetworkOutput::Builder builder;
            NetworkOutput built = builder.network(network).name(name).inputTensor(input_tensor).dataType(data_type).build();

            // Move the networkOutput layer into the pre-allocated but uninitialized memory at self
            new (self) NetworkOutput(std::move(built));
        },
        "network"_a,
        "name"_a,
        "input_tensor"_a,
        "data_type"_a);

    network_output.def(
        "get_feature_output",
        [](NetworkOutput &self) -> Tensor {
            Optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            // if (!maybeFeatureOutput.isPresent())
            //     return nullopt;
            // Network output creates featureOutput always, straight away.
            return maybeFeatureOutput.get();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor
                The feature output tensor handle.
            )nbdoc");

    network_output.def("version", &NetworkInput::getLayerVersion);

    network_output.attr("__doc__") = R"nbdoc(
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
            )nbdoc";
}
