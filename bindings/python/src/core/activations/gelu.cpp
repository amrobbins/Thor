#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_gelu(nb::module_ &m) {
    auto gelu = nb::class_<Gelu, Activation>(m, "Gelu");

    gelu.def_static(
        "__new__",
        [](nb::handle /*cls*/) -> std::shared_ptr<Gelu> {
            Gelu::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Gelu> g = std::dynamic_pointer_cast<Gelu>(base);
            if (!g)
                throw nb::type_error("Gelu builder did not return a Gelu instance");
            return g;
        },
        "cls"_a,
        nb::sig("def __new__(cls) -> thor.layers.activations.Gelu"),
        R"nbdoc(Construct a GELU activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    gelu.def(
        "__init__",
        [](Gelu *) {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a GELU activation (construction happens in __new__).)nbdoc");

    gelu.attr("__doc__") = R"doc(
Gaussian Error Linear Unit (GELU) activation.

Applied elementwise, the exact GELU is defined as

    f(x) = x * Φ(x)

where Φ(x) is the CDF of a standard normal distribution. A common
tanh-based approximation is

    f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³))).
)doc";
}
