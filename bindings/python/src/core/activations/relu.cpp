#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Thor::Tensor::DataType;

void bind_relu(nb::module_ &m) {
    nb::class_<Relu::Builder, Activation::Builder>(m, "Relu").def(
        "__init__",
        [](Relu::Builder *self) {
            // Create a relu builder in the pre-allocated but uninitialized memory at self
            new (self) Relu::Builder();
        },

        nb::sig("def __init__(self) -> None"),

        R"nbdoc(
            Create a relu activation.

            FIXME: Show math
            ----------
            ...
            )nbdoc");
}
