#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_tanh(nb::module_ &m) {
    auto tanh = nb::class_<Tanh, Activation>(m, "Tanh");

    tanh.def_static(
        "__new__",
        [](nb::handle /*cls*/) -> std::shared_ptr<Tanh> {
            Tanh::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Tanh> t = std::dynamic_pointer_cast<Tanh>(base);
            if (!t)
                throw nb::type_error("Tanh builder did not return a Tanh instance");
            return t;
        },
        "cls"_a,
        nb::sig("def __new__(cls) -> thor.layers.activations.Tanh"),
        R"nbdoc(Construct a Tanh activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    tanh.def(
        "__init__",
        [](Tanh *) {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a Tanh activation (construction happens in __new__).)nbdoc");

    tanh.attr("__doc__") = R"doc(
Hyperbolic tangent (tanh) activation.

Applied elementwise, tanh is defined as

    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

It squashes real-valued inputs into the range (-1, 1) with approximately
linear behavior around zero and saturation for large |x|. Tanh is often
used in recurrent networks and can be viewed as a zero-centered alternative
to the logistic sigmoid.
)doc";
}
