#include <nanobind/nanobind.h>

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_network(nb::module_ &m) {
    auto network = nb::class_<Network>(m, "Network");
    network.attr("__module__") = "thor";

    auto network_status_code_type = nb::enum_<Network::StatusCode>(m, "StatusCode")
                                        .value("success", Network::StatusCode::SUCCESS)
                                        .value("floating_input", Network::StatusCode::FLOATING_INPUT)
                                        .value("dangling_output", Network::StatusCode::DANGLING_OUTPUT)
                                        .value("gpu_out_of_memory", Network::StatusCode::GPU_OUT_OF_MEMORY)
                                        .value("duplicate_named_network_input", Network::StatusCode::DUPLICATE_NAMED_NETWORK_INPUT)
                                        .value("duplicate_named_network_output", Network::StatusCode::DUPLICATE_NAMED_NETWORK_OUTPUT)
                                        .value("deadlock_cycle", Network::StatusCode::DEADLOCK_CYCLE);
    network_status_code_type.attr("__qualname__") = "Network.StatusCode";
    network.attr("StatusCode") = network_status_code_type;

    network.def(
        "__init__",
        [](Network *self, const std::string &name) {
            // Create the network in the pre-allocated but uninitialized memory at self
            new (self) Network(name);
        },
        "name"_a,
        R"nbdoc(
A Network that contains layers. FIXME.
)nbdoc");

    network.def("get_network_name", &Network::getNetworkName);
    network.def("get_num_stamps", &Network::getNumStamps);
    network.def("status_code_to_string", &Network::statusCodeToString, "status_code"_a);

    network.def("save", &Network::save, "directory"_a, "overwrite"_a = false, "save_optimizer_state"_a = true);
    network.def("load", &Network::load, "directory"_a);

    network.def("get_default_optimizer", &Network::getDefaultOptimizer);

    network.def(
        "place",
        [](Network &self,
           uint32_t batch_size,
           bool inference_only,
           std::vector<int32_t> forced_devices,
           uint32_t forced_num_stamps_per_gpu) {
            std::vector<Event> init_done_events;
            Network::StatusCode status_code =
                self.place(batch_size, init_done_events, inference_only, forced_devices, forced_num_stamps_per_gpu);

            // On the python side, synchronize on the host here for simplicity, host sync vs stream sync is not the hot path here.
            if (status_code == Network::StatusCode::SUCCESS) {
                nb::gil_scoped_release release;
                for (Event &init_done_event : init_done_events) {
                    init_done_event.synchronize();
                }
            }

            return status_code;
        },
        "batch_size"_a,
        "inference_only"_a = false,
        "forced_devices"_a = std::vector<int32_t>{},
        "forced_num_stamps_per_gpu"_a = 0,
        R"nbdoc(
Place / compile the network for execution.

Parameters
----------
batch_size : int
inference_only : bool, default False
forced_devices : list[int], default []
    Device ids to force placement onto. Use Network.CPU for CPU.
forced_num_stamps_per_gpu : int, default 0

Returns
-------
thor.Network.StatusCode
)nbdoc");
}
