#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_soft_plus(nb::module_ &m) {
    nb::class_<SoftPlus::Builder, Activation::Builder>(m, "SoftPlus")
        .def(
            "__init__",
            [](SoftPlus::Builder *self) {
                // Create a softPlus builder in the pre-allocated but uninitialized memory at self
                new (self) SoftPlus::Builder();
            },

            nb::sig("def __init__(self) -> None"),

            R"nbdoc(
            SoftPlus activation.

            SoftPlus is a smooth approximation to ReLU, defined
            elementwise as

                f(x) = log(1 + exp(x))

            It maps real inputs to positive outputs and grows roughly linearly
            for large positive x, while remaining strictly positive and
            differentiable everywhere. Compared to ReLU, SoftPlus avoids a
            hard kink at zero, which can be beneficial in some optimization
            settings.
            )nbdoc");
}
