#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_hard_sigmoid(nb::module_ &m) {
    auto hard_sigmoid = nb::class_<HardSigmoid, Activation>(m, "HardSigmoid");
    hard_sigmoid.attr("__module__") = "thor.activations";

    hard_sigmoid.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<HardSigmoid> {
            HardSigmoid::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<HardSigmoid> hs = std::dynamic_pointer_cast<HardSigmoid>(base);
            if (!hs)
                throw nb::type_error("HardSigmoid builder did not return a HardSigmoid instance");
            return hs;
        },
        "cls"_a,
        // nb::sig("def __new__(cls) -> thor.layers.activations.HardSigmoid"),
        R"nbdoc(Construct a HardSigmoid activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    hard_sigmoid.def(
        "__init__",
        [](HardSigmoid *self) -> void {
            // no-op: constructed in __new__
        },
        // nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a HardSigmoid activation (construction happens in __new__).)nbdoc");

    hard_sigmoid.attr("__doc__") = R"doc(
Hard sigmoid activation.

Hard-sigmoid is a piecewise-linear approximation to the standard sigmoid,
defined elementwise as a clipped line:

    f(x) ≈ clip(a * x + b, 0, 1)

where ``a`` and ``b`` are chosen so that the function transitions smoothly
from 0 to 1 over a finite interval. Compared to the standard sigmoid,
hard-sigmoid is cheaper to evaluate and has constant slopes in the central
region.
)doc";
}
