#include <nanobind/nanobind.h>

#include <nanobind/stl/vector.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_network(nb::module_ &m) {
    auto network_class = nb::class_<Network>(m, "Network")
                             .def(
                                 "__init__",
                                 [](Network *self) {
                                     // Create the network in the pre-allocated but uninitialized memory at self
                                     new (self) Network();
                                 },

                                 nb::sig("def __init__(self) -> None"),

                                 R"nbdoc(
        A Network that contains layers. FIXME.
        )nbdoc")
                             .def("get_network_name", &Network::getNetworkName);
}
