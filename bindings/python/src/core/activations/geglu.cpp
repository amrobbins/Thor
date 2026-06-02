#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Geglu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_geglu(nb::module_ &m) {
    auto activation = nb::class_<Geglu, Activation>(m, "Geglu");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<Geglu> {
            Geglu::Builder b;

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<Geglu> concrete = std::dynamic_pointer_cast<Geglu>(base);
            if (!concrete)
                throw nb::type_error("Geglu builder did not return a Geglu instance");
            return concrete;
        },
        "cls"_a,
        R"nbdoc(Construct a Geglu activation.)nbdoc");

    activation.def(
        "__init__",
        [](Geglu *self) -> void {
            (void)self;
        },
        R"nbdoc(Initialize a Geglu activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
GELU gated linear unit activation.

This activation splits the final feature dimension into two equal halves,
then returns the first half multiplied by a transformed gate half.
It is intended as a standalone shape-changing activation layer.
)doc";
}
