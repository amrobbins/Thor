#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Swish.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_swish(nb::module_ &m) {
    nb::class_<Swish::Builder, Activation::Builder>(m, "Swish").def(
        "__init__",
        [](Swish::Builder *self) {
            // Create a swish builder in the pre-allocated but uninitialized memory at self
            new (self) Swish::Builder();
        },

        nb::sig("def __init__(self) -> None"),

        R"nbdoc(
            Swish (SiLU) activation.

            Swish is a smooth, non-monotonic activation defined elementwise as

                f(x) = x * sigmoid(x)
                     = x / (1 + exp(-x))

            It behaves roughly like a smoothed, self-gated ReLU: small negative
            inputs are softly suppressed, while large positive inputs pass
            through almost linearly. Swish (also known as SiLU) has been shown
            to work well in a variety of deep architectures, particularly in
            modern convolutional and transformer-based models.
            )nbdoc");
}
