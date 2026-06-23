#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;

void bind_network_input(nb::module_ &m) {
    auto network_input = nb::class_<NetworkInput, Layer>(m, "NetworkInput");
    network_input.attr("__module__") = "thor.layers";

    network_input.def(
        "__init__",
        [](NetworkInput *self,
           Network &network,
           const string &name,
           const vector<uint64_t> &dimensions,
           const DataType &data_type,
           bool dimensions_include_batch,
           std::optional<Tensor> pass_through_source) {
            if (name.length() == 0) {
                string msg = "Network Input instance: name must have non-zero length but name=\"\" was passed in.";
                throw nb::value_error(msg.c_str());
            }
            if (dimensions.size() == 0) {
                string msg = "Network Input instance: dimensions must be non-zero, but dimensions of size 0 was passed in.";
                throw nb::value_error(msg.c_str());
            }

            NetworkInput::Builder builder;
            builder.network(network)
                .name(name)
                .dimensions(dimensions)
                .dataType(data_type)
                .dimensionsIncludeBatch(dimensions_include_batch);
            if (pass_through_source.has_value()) {
                builder.passThroughSource(pass_through_source.value());
            }
            NetworkInput built = builder.build();

            // Move the networkInput layer into the pre-allocated but uninitialized memory at self
            new (self) NetworkInput(std::move(built));
        },
        "network"_a,
        "name"_a,
        "dimensions"_a,
        "data_type"_a,
        "dimensions_include_batch"_a = false,
        "pass_through_source"_a = nb::none());

    network_input.def(
        "get_feature_output",
        [](NetworkInput &self) -> Tensor {
            std::optional<Tensor> maybeFeatureOutput = self.getFeatureOutput();
            // if (!maybeFeatureOutput.has_value())
            //     return nullopt;
            // Network input creates featureOutput always, straight away.
            return maybeFeatureOutput.value();
        },
        R"nbdoc(
            Return the output tensor produced by this layer.

            Returns
            -------
            thor.Tensor
                The feature output tensor handle.
            )nbdoc");

    network_input.def("version", &Layer::getLayerVersion);

    network_input.attr("__doc__") = R"nbdoc(
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
            dimensions_include_batch : bool, default False
                When True, ``dimensions`` already includes the batch dimension.
                This is primarily for internal network-composition runtimes.
            pass_through_source : thor.Tensor | None, default None
                Internal network-composition hook.  When supplied, this NetworkInput
                is an API-level pass-through alias of the source tensor.  It is not
                stamped as an external network input and does not allocate or copy
                through an input staging tensor.
            )nbdoc";
}
