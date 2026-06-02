#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Threshold.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_threshold(nb::module_ &m) {
    auto activation = nb::class_<Threshold, Activation>(m, "Threshold");
    activation.attr("__module__") = "thor.activations";

    activation.def_static(
        "__new__",
        [](nb::handle cls, double threshold, double value) -> std::shared_ptr<Threshold> {
            Threshold::Builder b;
            b.threshold(threshold).value(value);

            std::shared_ptr<Activation> base = b.build();
            std::shared_ptr<Threshold> concrete = std::dynamic_pointer_cast<Threshold>(base);
            if (!concrete)
                throw nb::type_error("Threshold builder did not return a Threshold instance");
            return concrete;
        },
        "cls"_a, "threshold"_a = 0.0, "value"_a = 0.0,
        R"nbdoc(Construct a Threshold activation.)nbdoc");

    activation.def(
        "__init__",
        [](Threshold *self, double threshold, double value) -> void {
            (void)self;
        },
        "threshold"_a = 0.0, "value"_a = 0.0,
        R"nbdoc(Initialize a Threshold activation (construction happens in __new__).)nbdoc");

    activation.attr("__doc__") = R"doc(
Threshold activation.

Applied elementwise, threshold returns x when x > threshold, otherwise value.
)doc";
}
