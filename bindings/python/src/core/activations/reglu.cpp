#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Reglu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_reglu(nb::module_ &m) {
    auto activation = nb::class_<Reglu, Activation>(m, "Reglu");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<Reglu> {
            Reglu::Builder b;

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<Reglu> concrete = std::dynamic_pointer_cast<Reglu>(base);
            if (!concrete)
                throw nb::type_error("Reglu builder did not return a Reglu instance");
            return concrete;
        },
        "cls"_a,
        R"nbdoc(Construct a Reglu activation.)nbdoc");

    activation.def(
        "__init__",
        [](Reglu *self) -> void {
            (void)self;
        },
        R"nbdoc(Initialize a Reglu activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
Rectified gated linear unit activation.

This activation splits the final feature dimension into two equal halves,
then returns the first half multiplied by a transformed gate half.
It is intended as a standalone shape-changing activation layer.
)doc";
}
