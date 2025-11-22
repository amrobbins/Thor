#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Gelu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_gelu(nb::module_ &m) {
    nb::class_<Gelu::Builder, Activation::Builder>(m, "Gelu").def(
        "__init__",
        [](Gelu::Builder *self) {
            // Create a gelu builder in the pre-allocated but uninitialized memory at self
            new (self) Gelu::Builder();
        },

        nb::sig("def __init__(self) -> None"),

        R"nbdoc(
            Gaussian Error Linear Unit (GELU) activation.

            Applied elementwise, the exact GELU is defined as

                f(x) = x * Φ(x)

            where Φ(x) is the CDF of a standard normal distribution. A common
            tanh-based approximation is

                f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

            GELU smoothly gates inputs based on their magnitude, and is widely
            used in modern transformer and deep NLP architectures.
            )nbdoc");
}
