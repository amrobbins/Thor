#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu6.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_relu6(nb::module_ &m) {
    auto activation = nb::class_<Relu6, Activation>(m, "Relu6");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<Relu6> {
            Relu6::Builder b;

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<Relu6> concrete = std::dynamic_pointer_cast<Relu6>(base);
            if (!concrete)
                throw nb::type_error("Relu6 builder did not return a Relu6 instance");
            return concrete;
        },
        "cls"_a,
        R"nbdoc(Construct a Relu6 activation.)nbdoc");

    activation.def(
        "__init__",
        [](Relu6 *self) -> void {
            (void)self;
        },

        R"nbdoc(Initialize a Relu6 activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
ReLU6 activation.

Applied elementwise, ReLU6 is defined as

    f(x) = min(max(x, 0), 6)

It is commonly used in mobile-efficient networks.
)doc";
}
