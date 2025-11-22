#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Selu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_selu(nb::module_ &m) {
    nb::class_<Selu::Builder, Activation::Builder>(m, "Selu").def(
        "__init__",
        [](Selu::Builder *self) {
            // Create a selu builder in the pre-allocated but uninitialized memory at self
            new (self) Selu::Builder();
        },

        nb::sig("def __init__(self) -> None"),

        R"nbdoc(
            Scaled Exponential Linear Unit (SELU) activation.

            SELU is defined elementwise as

                f(x) = λ * x                          if x > 0
                       λ * α * (exp(x) - 1)           if x <= 0

            where λ (lambda) and α (alpha) are fixed positive constants
            (α ≈ 1.67326 and λ ≈ 1.05070). With appropriate weight
            initialization and architecture constraints, SELU can encourage
            self-normalizing behavior, keeping activations close to zero mean
            and unit variance throughout deep networks.
            )nbdoc");
}
