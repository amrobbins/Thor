#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "DeepLearning/Api/Network/PlacedNetwork.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace Thor;

void bind_placed_network(nb::module_ &thor) {
    auto placed_network = nb::class_<PlacedNetwork>(thor, "PlacedNetwork");
    placed_network.attr("__module__") = "thor";

    placed_network.def("save", &PlacedNetwork::save, "directory"_a, "overwrite"_a = false, "save_optimizer_state"_a = false);

    placed_network.def("get_num_stamps", &PlacedNetwork::getNumStamps);

    placed_network.def("get_stamped_network", &PlacedNetwork::getStampedNetwork, "i"_a, nb::rv_policy::reference_internal);

    placed_network.def("get_network_name", &PlacedNetwork::getNetworkName);

    placed_network.def("get_num_trainable_layers", &PlacedNetwork::getNumTrainableLayers);
}
