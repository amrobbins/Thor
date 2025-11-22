#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Exponential.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_exponential(nb::module_ &m) {
    nb::class_<Exponential::Builder, Activation::Builder>(m, "Exponential").def(
        "__init__",
        [](Exponential::Builder *self) {
            // Create a exponential builder in the pre-allocated but uninitialized memory at self
            new (self) Exponential::Builder();
        },

        nb::sig("def __init__(self) -> None"),

        R"nbdoc(
            Exponential activation.

            Applied elementwise, this activation is defined as

                f(x) = exp(x)

            It maps all real inputs to positive outputs and grows rapidly for
            large positive values. This can be useful in certain architectures,
            but it may also lead to exploding activations if not combined with
            appropriate normalization or regularization.
            )nbdoc");
}
