#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Selu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_selu(nb::module_ &m) {
    auto selu = nb::class_<Selu, Activation>(m, "Selu");

    selu.def_static(
        "__new__",
        [](nb::handle /*cls*/) -> std::shared_ptr<Selu> {
            Selu::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Selu> s = std::dynamic_pointer_cast<Selu>(base);
            if (!s)
                throw nb::type_error("Selu builder did not return a Selu instance");
            return s;
        },
        "cls"_a,
        nb::sig("def __new__(cls) -> thor.layers.activations.Selu"),
        R"nbdoc(Construct a SELU activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    selu.def(
        "__init__",
        [](Selu *) {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a SELU activation (construction happens in __new__).)nbdoc");

    selu.attr("__doc__") = R"doc(
Scaled Exponential Linear Unit (SELU) activation.

SELU is defined elementwise as

    f(x) = λ * x                if x > 0
           λ * α * (exp(x) - 1) if x <= 0

where λ (lambda) and α (alpha) are fixed positive constants
(α ≈ 1.67326 and λ ≈ 1.05070). With appropriate weight
initialization and architecture constraints, SELU can encourage
self-normalizing behavior, keeping activations close to zero mean
and unit variance throughout deep networks.
)doc";
}
