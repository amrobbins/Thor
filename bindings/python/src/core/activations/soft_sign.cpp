#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/SoftSign.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

void bind_soft_sign(nb::module_ &m) {
    nb::class_<SoftSign::Builder, Activation::Builder>(m, "SoftSign")
        .def(
            "__init__",
            [](SoftSign::Builder *self) {
                // Create a softSign builder in the pre-allocated but uninitialized memory at self
                new (self) SoftSign::Builder();
            },

            nb::sig("def __init__(self) -> None"),

            R"nbdoc(
            SoftSign activation.

            SoftSign is a smooth, bounded activation defined elementwise as

                f(x) = x / (1 + |x|)

            It squashes large positive and negative values toward +1 and -1,
            respectively, while remaining smooth and differentiable everywhere.
            Compared to tanh, SoftSign has polynomial rather than exponential
            tails, which can lead to slightly different gradient behavior for
            large |x|.
            )nbdoc");
}
