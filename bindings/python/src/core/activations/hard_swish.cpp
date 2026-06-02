#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/HardSwish.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_hard_swish(nb::module_ &m) {
    auto activation = nb::class_<HardSwish, Activation>(m, "HardSwish");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<HardSwish> {
            HardSwish::Builder b;

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<HardSwish> concrete = std::dynamic_pointer_cast<HardSwish>(base);
            if (!concrete)
                throw nb::type_error("HardSwish builder did not return a HardSwish instance");
            return concrete;
        },
        "cls"_a,
        R"nbdoc(Construct a HardSwish activation.)nbdoc");

    activation.def(
        "__init__",
        [](HardSwish *self) -> void {
            (void)self;
        },

        R"nbdoc(Initialize a HardSwish activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
Hard swish activation.

Applied elementwise, hard-swish is defined as

    f(x) = x * relu6(x + 3) / 6

It is a piecewise-linear approximation to swish.
)doc";
}
