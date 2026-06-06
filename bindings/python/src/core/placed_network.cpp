#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <utility>

#include "DeepLearning/Api/Network/PlacedNetwork.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace Thor;

void bind_placed_network(nb::module_ &thor) {
    auto placed_network = nb::class_<PlacedNetwork>(thor, "PlacedNetwork");
    placed_network.attr("__module__") = "thor";

    placed_network.def("save", &PlacedNetwork::save, "directory"_a, "overwrite"_a = false, "save_optimizer_state"_a = false);

    placed_network.def("get_num_stamps", &PlacedNetwork::getNumStamps);

    placed_network.def(
        "infer",
        [](PlacedNetwork& self, std::map<std::string, ThorImplementation::Tensor> batch_inputs, uint64_t stamp_index) {
            nb::gil_scoped_release release;
            return self.infer(std::move(batch_inputs), stamp_index);
        },
        "batch_inputs"_a,
        "stamp_index"_a = 0,
        R"nbdoc(
Run one inference batch through this placed network stamp.

Parameters
----------
batch_inputs : dict[str, thor.physical.PhysicalTensor]
    Full batched input tensors keyed by NetworkInput name.
stamp_index : int, default 0
    Stamped network instance to execute.

Returns
-------
dict[str, thor.physical.PhysicalTensor]
    Full batched output tensors keyed by NetworkOutput name. Network outputs are CPU tensors.
)nbdoc");

    placed_network.def("get_stamped_network", &PlacedNetwork::getStampedNetwork, "i"_a, nb::rv_policy::reference_internal);

    placed_network.def("get_network_name", &PlacedNetwork::getNetworkName);

    placed_network.def("get_num_trainable_layers", &PlacedNetwork::getNumTrainableLayers);
    placed_network.def("resolve_parameter_reference", &PlacedNetwork::resolveParameterReference, "parameter_reference"_a);
    placed_network.def("resolve_parameter_references", &PlacedNetwork::resolveParameterReferences, "parameter_references"_a);
    placed_network.def("has_api_tensor", &PlacedNetwork::hasApiTensor, "tensor"_a);
    placed_network.def("resolve_api_tensor", &PlacedNetwork::resolveApiTensor, "tensor"_a);
    placed_network.def("resolve_api_tensors", &PlacedNetwork::resolveApiTensors, "tensors"_a);
    placed_network.def("has_network_input", &PlacedNetwork::hasNetworkInput, "name"_a);
    placed_network.def("get_network_input_names", &PlacedNetwork::getNetworkInputNames, "stamp_index"_a = 0);
}
