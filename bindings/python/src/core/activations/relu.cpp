#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_relu(nb::module_ &m) {
    auto relu = nb::class_<Relu, Activation>(m, "Relu");

    relu.def_static(
        "__new__",
        [](nb::handle /*cls*/) -> std::shared_ptr<Relu> {
            Relu::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Relu> r = std::dynamic_pointer_cast<Relu>(base);
            if (!r)
                throw nb::type_error("Relu builder did not return a Relu instance");
            return r;
        },
        "cls"_a,
        nb::sig("def __new__(cls) -> thor.layers.activations.Relu"),
        R"nbdoc(Construct a ReLU activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    relu.def(
        "__init__",
        [](Relu *) {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a ReLU activation (construction happens in __new__).)nbdoc");

    relu.attr("__doc__") = R"doc(
Rectified Linear Unit (ReLU) activation.

Applied elementwise, ReLU is defined as

    f(x) = max(0, x)

ReLU preserves positive inputs and sets negative inputs to zero. It is
computationally cheap, helps mitigate vanishing gradients compared to
saturating activations (e.g., sigmoid/tanh), and is widely used in deep
networks. A potential drawback is "dead" neurons when inputs remain
negative for long periods, causing zero gradients in that region.
)doc";
}
