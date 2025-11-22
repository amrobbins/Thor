#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_sigmoid(nb::module_ &m) {
    nb::class_<Sigmoid::Builder, Activation::Builder>(m, "Sigmoid").def(
        "__init__",
        [](Sigmoid::Builder *self) {
            // Create a sigmoid builder in the pre-allocated but uninitialized memory at self
            new (self) Sigmoid::Builder();
        },

        nb::sig("def __init__(self) -> None"),

        R"nbdoc(
            Sigmoid activation.

            Applied elementwise, this activation is defined as

                f(x) = 1 / (1 + exp(-x))

            It maps real-valued inputs into the interval (0, 1) and is
            commonly used when outputs are interpreted as probabilities or
            gates (e.g., in recurrent networks). Note that sigmoid can suffer
            from saturation for large |x|, which may slow down learning if not
            combined with appropriate initialization or normalization.
            )nbdoc");
}
