#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Softmax.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_softmax(nb::module_ &m) {
    nb::class_<Softmax::Builder, Activation::Builder>(m, "Softmax")
        .def(
            "__init__",
            [](Softmax::Builder *self) {
                // Create a softmax builder in the pre-allocated but uninitialized memory at self
                new (self) Softmax::Builder();
            },

            nb::sig("def __init__(self) -> None"),

            R"nbdoc(
            Softmax activation.

            Softmax is typically applied along the last (feature) dimension of
            a tensor to convert raw scores (logits) into a probability
            distribution. For an input vector x, softmax is defined as

                softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

            for each component i. The outputs are all positive and sum to 1,
            making softmax a natural choice for multi-class classification
            outputs. A numerically stable form (subtracting max(x) before
            exponentiation) is used to avoid overflow.
            )nbdoc");
}
