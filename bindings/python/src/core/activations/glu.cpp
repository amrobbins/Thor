#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Glu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_glu(nb::module_ &m) {
    auto activation = nb::class_<Glu, Activation>(m, "Glu");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<Glu> {
            Glu::Builder b;

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<Glu> concrete = std::dynamic_pointer_cast<Glu>(base);
            if (!concrete)
                throw nb::type_error("Glu builder did not return a Glu instance");
            return concrete;
        },
        "cls"_a,
        R"nbdoc(Construct a Glu activation.)nbdoc");

    activation.def(
        "__init__",
        [](Glu *self) -> void {
            (void)self;
        },
        R"nbdoc(Initialize a Glu activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
Gated Linear Unit activation.

This activation splits the final feature dimension into two equal halves,
then returns the first half multiplied by a transformed gate half.
It is intended as a standalone shape-changing activation layer.
)doc";
}
