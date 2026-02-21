#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/SoftSign.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_soft_sign(nb::module_ &m) {
    auto soft_sign = nb::class_<SoftSign, Activation>(m, "SoftSign");
    soft_sign.attr("__module__") = "thor.activations";

    soft_sign.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<SoftSign> {
            SoftSign::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<SoftSign> ss = std::dynamic_pointer_cast<SoftSign>(base);
            if (!ss)
                throw nb::type_error("SoftSign builder did not return a SoftSign instance");
            return ss;
        },
        "cls"_a,
        nb::sig("def __new__(cls) -> thor.layers.activations.SoftSign"),
        R"nbdoc(Construct a SoftSign activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    soft_sign.def(
        "__init__",
        [](SoftSign *self) -> void {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a SoftSign activation (construction happens in __new__).)nbdoc");

    soft_sign.attr("__doc__") = R"doc(
SoftSign activation.

SoftSign is a smooth, bounded activation defined elementwise as

    f(x) = x / (1 + |x|)

It squashes large positive and negative values toward +1 and -1,
respectively, while remaining smooth and differentiable everywhere.
Compared to tanh, SoftSign has polynomial rather than exponential
tails, which can lead to slightly different gradient behavior for
large |x|.
)doc";
}
