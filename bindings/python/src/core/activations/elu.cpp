#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Elu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_elu(nb::module_ &m) {
    auto elu = nb::class_<Elu, Activation>(m, "Elu");

    elu.def_static(
        "__new__",
        [](nb::handle cls, float alpha) -> std::shared_ptr<Elu> {
            Elu::Builder b;
            b.alpha(alpha);

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Elu> e = std::dynamic_pointer_cast<Elu>(base);
            if (!e)
                throw nb::type_error("Elu builder did not return an Elu instance");
            return e;  // nanobind converts shared_ptr<Elu> to an Elu Python object
        },
        "cls"_a,
        "alpha"_a = 1.0f,
        nb::sig("def __new__(cls, alpha: float = 1.0) -> thor.layers.activations.Elu"),
        R"nbdoc(Construct an ELU activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    elu.def(
        "__init__",
        [](Elu *self, nb::args args, nb::kwargs kwargs) -> void {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self, *args, **kwargs) -> None"),
        R"nbdoc(No-op; constructed in __new__.)nbdoc");

    elu.attr("__doc__") = R"doc(
Exponential Linear Unit (ELU) activation.

ELU is defined elementwise as

    f(x) = x                    if x > 0
           alpha * (exp(x) - 1) if x <= 0

where ``alpha`` is a positive constant (typically ``alpha = 1``).
Compared to ReLU, ELU has negative outputs for negative inputs.
)doc";
}
