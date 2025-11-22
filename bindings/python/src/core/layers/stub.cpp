#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_stub(nb::module_ &m) {
    nb::class_<Stub, Layer>(m, "Stub").def(
        "__init__",
        [](Stub *self, Network &network, const Tensor &input_tensor) {
            Stub::Builder builder;

            Stub built = builder.network(network).inputTensor(input_tensor).build();

            // Move the stub layer into the pre-allocated but uninitialized memory at self
            new (self) Stub(std::move(built));
        },
        "network"_a,
        "input_tensor"_a,

        nb::sig("def __init__(self, "
                "network: thor.Network, "
                "feature_input: thor.Tensor"
                ") -> None"),

        R"nbdoc(
            Create and attach a Stub layer to a Network.
            When there is a dangling tensor in the execution graph (i.e. it is not
            connected to the input of anything else in the network) then the graph
            compiler will complain about a dangling tensor and abort. If you want
            to graph to compile you can tell network that you are aware and ok with
            the dangling tensor by attaching it as the input to a Stub layer, and
            then the network will compile.

            Parameters
            ----------
            network : thor.Network
                Network the layer should be added to.
            input_tensor : thor.Tensor
                Input feature tensor for this layer.
            )nbdoc");
}