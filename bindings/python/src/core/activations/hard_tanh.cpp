#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/HardTanh.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_hard_tanh(nb::module_ &m) {
    auto activation = nb::class_<HardTanh, Activation>(m, "HardTanh");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls, double min_value, double max_value) -> std::shared_ptr<HardTanh> {
            HardTanh::Builder b;
            if (min_value > max_value) {
                throw nb::value_error("HardTanh requires min_value <= max_value");
            }
            b.minValue(min_value).maxValue(max_value);

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<HardTanh> concrete = std::dynamic_pointer_cast<HardTanh>(base);
            if (!concrete)
                throw nb::type_error("HardTanh builder did not return a HardTanh instance");
            return concrete;
        },
        "cls"_a, "min_value"_a = -1.0, "max_value"_a = 1.0,
        R"nbdoc(Construct a HardTanh activation.)nbdoc");

    activation.def(
        "__init__",
        [](HardTanh *self, double min_value, double max_value) -> void {
            (void)self;
        },
        "min_value"_a = -1.0, "max_value"_a = 1.0,
        R"nbdoc(Initialize a HardTanh activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
Hard tanh activation.

Applied elementwise, hard-tanh clamps x to [min_value, max_value].
)doc";
}
