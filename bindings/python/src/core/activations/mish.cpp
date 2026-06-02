#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Mish.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_mish(nb::module_ &m) {
    auto activation = nb::class_<Mish, Activation>(m, "Mish");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<Mish> {
            Mish::Builder b;

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<Mish> concrete = std::dynamic_pointer_cast<Mish>(base);
            if (!concrete)
                throw nb::type_error("Mish builder did not return a Mish instance");
            return concrete;
        },
        "cls"_a,
        R"nbdoc(Construct a Mish activation.)nbdoc");

    activation.def(
        "__init__",
        [](Mish *self) -> void {
            (void)self;
        },

        R"nbdoc(Initialize a Mish activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
Mish activation.

Applied elementwise, Mish is defined as

    f(x) = x * tanh(softplus(x))

It is a smooth, non-monotonic activation.
)doc";
}
