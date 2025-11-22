#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_hard_sigmoid(nb::module_ &m) {
    nb::class_<HardSigmoid::Builder, Activation::Builder>(m, "HardSigmoid").def(
        "__init__",
        [](HardSigmoid::Builder *self) {
            // Create a hard_sigmoid builder in the pre-allocated but uninitialized memory at self
            new (self) HardSigmoid::Builder();
        },

        nb::sig("def __init__(self) -> None"),

        R"nbdoc(
            Hard sigmoid activation.

            Hard-sigmoid is a piecewise-linear approximation to the
            standard sigmoid, defined elementwise as a clipped line:

                f(x) ≈ clip(a * x + b, 0, 1)

            where ``a`` and ``b`` are chosen so that the function transitions
            smoothly from 0 to 1 over a finite interval. Compared to the
            standard sigmoid, hard-sigmoid is cheaper to evaluate and has
            constant slopes in the central region, which can be useful in
            recurrent networks or other performance-sensitive settings.
            )nbdoc");
}
