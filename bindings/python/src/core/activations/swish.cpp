#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Swish.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_swish(nb::module_ &m) {
    auto swish = nb::class_<Swish, Activation>(m, "Swish");
    swish.attr("__module__") = "thor.activations";

    swish.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<Swish> {
            Swish::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Swish> s = std::dynamic_pointer_cast<Swish>(base);
            if (!s)
                throw nb::type_error("Swish builder did not return a Swish instance");
            return s;
        },
        "cls"_a,
        nb::sig("def __new__(cls) -> thor.layers.activations.Swish"),
        R"nbdoc(Construct a Swish (SiLU) activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    swish.def(
        "__init__",
        [](Swish *) {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a Swish (SiLU) activation (construction happens in __new__).)nbdoc");

    swish.attr("__doc__") = R"doc(
Swish (SiLU) activation.

Swish is a smooth, non-monotonic activation defined elementwise as

    f(x) = x * sigmoid(x)
         = x / (1 + exp(-x))

It behaves roughly like a smoothed, self-gated ReLU: small negative inputs
are softly suppressed, while large positive inputs pass through almost
linearly. Swish (also known as SiLU) has been shown to work well in a
variety of deep architectures, particularly in modern convolutional and
transformer-based models.
)doc";
}
