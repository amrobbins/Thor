#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "Utilities/ComputeTopology/MachineEvaluator.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_machine_evaluator(nb::module_ &physical) {
    auto gcr = nb::class_<GpuConnectionRanking>(physical, "GpuConnectionRanking");
    gcr.attr("__module__") = "thor.physical";

    gcr.def(nb::init<>());

    // Expose fields as read/write attributes
    gcr.def_rw("peer_gpu_num", &GpuConnectionRanking::peerGpuNum);
    gcr.def_rw("is_peer_to_peer_supported", &GpuConnectionRanking::isPeerToPeerSupported);
    gcr.def_rw("peer_to_peer_speed_ranking", &GpuConnectionRanking::peerToPeerSpeedRanking);

    // Sorting support
    gcr.def("__lt__", [](const GpuConnectionRanking &a, const GpuConnectionRanking &b) { return a < b; });

    gcr.def("__repr__", [](const GpuConnectionRanking &self) {
        return "GpuConnectionRanking(peer_gpu_num=" + std::to_string(self.peerGpuNum) +
               ", is_peer_to_peer_supported=" + std::string(self.isPeerToPeerSupported ? "True" : "False") +
               ", peer_to_peer_speed_ranking=" + std::to_string(self.peerToPeerSpeedRanking) + ")";
    });

    auto machine_evaluator = nb::class_<MachineEvaluator>(physical, "MachineEvaluator");
    machine_evaluator.attr("__module__") = "thor.physical";

    // No public constructor binding => not constructible from Python.

    machine_evaluator.def_static(
        "instance",
        []() -> MachineEvaluator & { return MachineEvaluator::instance(); },
        nb::rv_policy::reference,
        R"nbdoc(
Return the singleton MachineEvaluator instance.
)nbdoc");

    machine_evaluator.def("get_current_gpu_num", &MachineEvaluator::getCurrentGpuNum);

    // Connection / topology info
    machine_evaluator.def("get_connection_speed_rankings", &MachineEvaluator::getConnectionSpeedRankings, "source_gpu_num"_a);
    machine_evaluator.def("is_peer_to_peer_available", &MachineEvaluator::isPeerToPeerAvailable, "source_gpu_num"_a, "dest_gpu_num"_a);

    // GPU type + PCI bus utilities
    machine_evaluator.def("get_gpu_type", nb::overload_cast<int>(&MachineEvaluator::getGpuType), "gpu_num"_a);
    machine_evaluator.def(
        "get_gpu_type", nb::overload_cast<>(&MachineEvaluator::getGpuType), R"nbdoc(Return the GPU type for the current GPU.)nbdoc");

    machine_evaluator.def("get_gpu_pci_bus_id", &MachineEvaluator::getGpuPciBusId, "gpu_num"_a);
    machine_evaluator.def("get_gpu_num_from_bus_id", &MachineEvaluator::getGpuNumFromBusId, "gpu_bus_id"_a);

    // adjacency + ordering
    machine_evaluator.def("get_adjacent_higher_gpu", &MachineEvaluator::getAdjacentHigherGpu, "gpu_num"_a);
    machine_evaluator.def("get_adjacent_lower_gpu", &MachineEvaluator::getAdjacentLowerGpu, "gpu_num"_a);
    machine_evaluator.def("get_ordered_gpus", &MachineEvaluator::getOrderedGpus);

    // device counts
    machine_evaluator.def("get_num_gpus", &MachineEvaluator::getNumGpus);

    machine_evaluator.def("get_num_multi_processors", nb::overload_cast<int>(&MachineEvaluator::getNumMultiProcessors), "gpu_num"_a);
    machine_evaluator.def("get_num_multi_processors",
                          nb::overload_cast<>(&MachineEvaluator::getNumMultiProcessors),
                          R"nbdoc(Return SM count for the current GPU.)nbdoc");

    // memory
    machine_evaluator.def("get_total_global_mem_bytes", &MachineEvaluator::getTotalGlobalMemBytes, "gpu_num"_a);
    machine_evaluator.def("get_free_mem_bytes", &MachineEvaluator::getFreeMemBytes, "gpu_num"_a);

    // static utility
    machine_evaluator.def_static("swap_active_device", &MachineEvaluator::swapActiveDevice, "new_gpu_num"_a);

    gcr.attr("__qualname__") = "MachineEvaluator.GpuConnectionRanking";
    machine_evaluator.attr("GpuConnectionRanking") = gcr;
    nb::delattr(physical, "GpuConnectionRanking");
}
