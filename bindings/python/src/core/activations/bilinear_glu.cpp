#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/BilinearGlu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_bilinear_glu(nb::module_ &m) {
    auto activation = nb::class_<BilinearGlu, Activation>(m, "BilinearGlu");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<BilinearGlu> {
            BilinearGlu::Builder b;

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<BilinearGlu> concrete = std::dynamic_pointer_cast<BilinearGlu>(base);
            if (!concrete)
                throw nb::type_error("BilinearGlu builder did not return a BilinearGlu instance");
            return concrete;
        },
        "cls"_a,
        R"nbdoc(Construct a BilinearGlu activation.)nbdoc");

    activation.def(
        "__init__",
        [](BilinearGlu *self) -> void {
            (void)self;
        },
        R"nbdoc(Initialize a BilinearGlu activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
Bilinear gated linear unit activation.

This activation splits the final feature dimension into two equal halves,
then returns the first half multiplied by a transformed gate half.
It is intended as a standalone shape-changing activation layer.
)doc";
}
