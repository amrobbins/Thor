#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;


void bind_layers(nb::module_ &layers) {
    using Thor::Layer;
    using Thor::DropOut;
    using Thor::Network;
    using Thor::Tensor;

    layers.doc() = "Thor layers";

    nb::class_<Layer>(layers, "Layer");
    nb::class_<DropOut, Layer>(layers, "DropOutLayer")
        .def("get_drop_proportion", &DropOut::getDropProportion);

    layers.def(
        "DropOut",
        [](Network &network,
           Tensor feature_input,
           float drop_proportion) -> std::shared_ptr<Layer> {
            DropOut::Builder b;

            // Call the C++ builder with your invariants
            b.network(network)
                .featureInput(feature_input)
                .dropProportion(drop_proportion);

            // This builds a DropOut, adds it to the network, and returns by value
            DropOut layer = b.build();

            // Expose it to Python as a shared_ptr<Layer>
            return make_shared<DropOut>(move(layer));
        },
        "network"_a,
        "feature_input"_a,
        "drop_proportion"_a,
        R"nbdoc(
            Create and attach a DropOut layer to a Network.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            feature_input : thor.Tensor
                Input feature tensor for this layer.
            drop_proportion : float
                Fraction of units to drop (0.0 <= p <= 1.0).
        )nbdoc"
    );

    layers.def("BatchNormalization", []() { return "temp"; });
    layers.def("Concatenate", []() { return "temp"; });
    layers.def("Convolution2d", []() { return "temp"; });
    layers.def("Flatten", []() { return "temp"; });
    layers.def("FullyConnected", []() { return "temp"; });
    layers.def("Inception", []() { return "temp"; });
    layers.def("NetworkInput", []() { return "temp"; });
    layers.def("NetworkOutput", []() { return "temp"; });
    layers.def("Pooling", []() { return "temp"; });
    layers.def("Reshape", []() { return "temp"; });
    layers.def("Stub", []() { return "temp"; });
    layers.def("TypeConverter", []() { return "temp"; });
}
