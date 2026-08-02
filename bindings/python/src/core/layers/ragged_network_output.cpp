#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>

#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace Thor;

void bind_ragged_network_output(nb::module_& m) {
    auto output = nb::class_<RaggedNetworkOutput>(m, "RaggedNetworkOutput");
    output.attr("__module__") = "thor.layers";

    output.def(
        "__init__",
        [](RaggedNetworkOutput* self, Network& network, const std::string& name, const RaggedTensor& input_tensor) {
            if (name.empty()) {
                throw nb::value_error("RaggedNetworkOutput name must not be empty.");
            }
            new (self) RaggedNetworkOutput(
                RaggedNetworkOutput::Builder().network(network).name(name).inputTensor(input_tensor).build());
        },
        "network"_a,
        "name"_a,
        "input_tensor"_a,
        R"nbdoc(
Expose one logical ragged result from a Network.

The packed values and row-partition offsets are materialized internally as a
paired output, but inference returns one
``thor.physical.PhysicalRaggedTensor`` under ``name`` rather than exposing the
component output names.
        )nbdoc");

    output.def("get_name", &RaggedNetworkOutput::getName);
    output.def("get_input", &RaggedNetworkOutput::getInput);
    output.def("get_feature_output", &RaggedNetworkOutput::getFeatureOutput);
}
