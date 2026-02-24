#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_soft_plus(nb::module_ &m) {
    auto soft_plus = nb::class_<SoftPlus, Activation>(m, "SoftPlus");
    soft_plus.attr("__module__") = "thor.activations";

    soft_plus.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<SoftPlus> {
            SoftPlus::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<SoftPlus> sp = std::dynamic_pointer_cast<SoftPlus>(base);
            if (!sp)
                throw nb::type_error("SoftPlus builder did not return a SoftPlus instance");
            return sp;
        },
        "cls"_a,
        // nb::sig("def __new__(cls) -> thor.layers.activations.SoftPlus"),
        R"nbdoc(Construct a SoftPlus activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    soft_plus.def(
        "__init__",
        [](SoftPlus *self) -> void {
            // no-op: constructed in __new__
        },
        // nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a SoftPlus activation (construction happens in __new__).)nbdoc");

    soft_plus.attr("__doc__") = R"doc(
SoftPlus activation.

SoftPlus is a smooth approximation to ReLU, defined elementwise as

    f(x) = log(1 + exp(x))

It maps real inputs to positive outputs and grows roughly linearly for
large positive x, while remaining strictly positive and differentiable
everywhere. Compared to ReLU, SoftPlus avoids a hard kink at zero.
)doc";
}
