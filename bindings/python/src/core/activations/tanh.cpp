#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Tanh.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_tanh(nb::module_ &m) {
    nb::class_<Tanh::Builder, Activation::Builder>(m, "Tanh").def(
        "__init__",
        [](Tanh::Builder *self) {
            // Create a tanh builder in the pre-allocated but uninitialized memory at self
            new (self) Tanh::Builder();
        },

        nb::sig("def __init__(self) -> None"),

        R"nbdoc(
            Hyperbolic tangent (tanh) activation.

            Applied elementwise, tanh is defined as

                f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

            It squashes real-valued inputs into the range (-1, 1) with
            approximately linear behavior around zero and saturation for large
            |x|. Tanh is often used in recurrent networks and can be viewed as
            a zero-centered alternative to the logistic sigmoid.
            )nbdoc");
}
