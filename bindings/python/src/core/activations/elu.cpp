#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Elu.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_elu(nb::module_ &m) {
    nb::class_<Elu::Builder, Activation::Builder>(m, "Elu").def(
        "__init__",
        [](Elu::Builder *self, float alpha) {
            Elu::Builder builder;
            builder.alpha(alpha);
            // Create a elu builder in the pre-allocated but uninitialized memory at self
            new (self) Elu::Builder(std::move(builder));
        },
        "alpha"_a = 1.0f,

        nb::sig("def __init__(self, "
                "alpha: float = 1.0"
                ") -> None"),

        R"nbdoc(
        Exponential Linear Unit (ELU) activation.

        ELU is defined elementwise as

            f(x) = x                    if x > 0
                   alpha * (exp(x) - 1) if x <= 0

        where ``alpha`` is a positive constant (typically ``alpha = 1``).
        Compared to ReLU, ELU has negative outputs for negative inputs,
        which can help reduce bias shift and improve convergence in some
        networks.
        )nbdoc");
}
