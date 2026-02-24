#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Exponential.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_exponential(nb::module_ &m) {
    auto exponential = nb::class_<Exponential, Activation>(m, "Exponential");
    exponential.attr("__module__") = "thor.activations";

    exponential.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<Exponential> {
            Exponential::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Exponential> e = std::dynamic_pointer_cast<Exponential>(base);
            if (!e)
                throw nb::type_error("Exponential builder did not return an Exponential instance");
            return e;
        },
        "cls"_a,
        // nb::sig("def __new__(cls) -> thor.layers.activations.Exponential"),
        R"nbdoc(Construct an Exponential activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    exponential.def(
        "__init__",
        [](Exponential *self) -> void {
            // no-op: constructed in __new__
        },
        // nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize an Exponential activation (construction happens in __new__).)nbdoc");

    exponential.attr("__doc__") = R"doc(
Exponential activation.

Applied elementwise, this activation is defined as

    f(x) = exp(x)

It maps all real inputs to positive outputs and grows rapidly for
large positive values. This can be useful in certain architectures,
but it may also lead to exploding activations if not combined with
appropriate normalization or regularization.
)doc";
}
