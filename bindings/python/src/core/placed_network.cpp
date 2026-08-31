#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <utility>

#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace Thor;

void bind_placed_network(nb::module_ &thor) {
    auto placed_network = nb::class_<PlacedNetwork>(thor, "PlacedNetwork");
    placed_network.attr("__module__") = "thor";

    placed_network.def("save", &PlacedNetwork::save, "directory"_a, "overwrite"_a = false, "save_optimizer_state"_a = false);

    placed_network.def("get_num_stamps", &PlacedNetwork::getNumStamps);
    placed_network.def("set_training_dropout_enabled",
                       &PlacedNetwork::setTrainingDropoutEnabled,
                       "enabled"_a,
                       R"nbdoc(
Drain work already submitted by this placement, then enable or disable training-time dropout
for all controllable physical layers. Configured dropout probabilities are unchanged. Callers
must not submit batches concurrently with this operation.
)nbdoc");
    placed_network.def("is_training_dropout_enabled", &PlacedNetwork::isTrainingDropoutEnabled);
    placed_network.def("get_num_training_dropout_controllable_layers",
                       &PlacedNetwork::getNumTrainingDropoutControllableLayers);

    placed_network.def(
        "infer",
        [](PlacedNetwork& self, nb::dict batch_inputs, uint64_t stamp_index) {
            Batch batch;
            for (auto item : batch_inputs) {
                const std::string name = nb::cast<std::string>(item.first);
                if (nb::isinstance<ThorImplementation::Tensor>(item.second)) {
                    batch.insert(name, nb::cast<ThorImplementation::Tensor>(item.second));
                } else if (nb::isinstance<ThorImplementation::RaggedTensor>(item.second)) {
                    batch.insert(name, nb::cast<ThorImplementation::RaggedTensor>(item.second));
                } else {
                    throw nb::type_error(
                        "PlacedNetwork.infer input values must be thor.physical.PhysicalTensor or thor.physical.PhysicalRaggedTensor.");
                }
            }

            std::map<std::string, InferenceOutputValue> outputs;
            {
                nb::gil_scoped_release release;
                outputs = self.inferLogical(batch, stamp_index);
            }
            nb::dict result;
            for (auto& [name, value] : outputs) {
                if (std::holds_alternative<ThorImplementation::Tensor>(value)) {
                    result[nb::str(name.c_str())] = nb::cast(std::get<ThorImplementation::Tensor>(value));
                } else {
                    result[nb::str(name.c_str())] = nb::cast(std::get<ThorImplementation::RaggedTensor>(value));
                }
            }
            return result;
        },
        "batch_inputs"_a,
        "stamp_index"_a = 0,
        R"nbdoc(
Run one inference batch through this placed network stamp.

Parameters
----------
batch_inputs : dict[str, thor.physical.PhysicalTensor | thor.physical.PhysicalRaggedTensor]
    Logical dense or ragged input fields keyed by NetworkInput/RaggedNetworkInput name.
    A RaggedNetworkInput declared with ``partition=...`` may be supplied as just
    its packed PhysicalTensor values; the referenced partition-owning input
    supplies offsets for that batch.
stamp_index : int, default 0
    Stamped network instance to execute.

Returns
-------
dict[str, thor.physical.PhysicalTensor | thor.physical.PhysicalRaggedTensor]
    Logical external outputs keyed by NetworkOutput/RaggedNetworkOutput name. Ragged component tensors remain implementation details.
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
