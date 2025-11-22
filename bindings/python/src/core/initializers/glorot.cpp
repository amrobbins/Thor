#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Implementation/Initializers/Glorot.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_glorot(nb::module_ &m) {
    auto glorot_class = nb::class_<Glorot::Builder, Initializer::Builder>(m, "Glorot")
                            .def(
                                "__init__",
                                [](Glorot::Builder *self, ThorImplementation::Glorot::Mode mode) {
                                    // Create the glorot in the pre-allocated but uninitialized memory at self
                                    Glorot::Builder builder;
                                    builder.mode(mode);
                                    new (self) Glorot::Builder(std::move(builder));
                                },
                                "mode"_a = ThorImplementation::Glorot::Mode::UNIFORM,

                                nb::sig("def __init__(self, "
                                        "mode: thor.initializers.Glorot.Mode"
                                        ") -> None"),

                                R"nbdoc(
        A Glorot (Xavier) weight initializer.

        Draws each weight from a uniform (or normal) distribution. The uniform version is the following.

        U[-limit, limit], where limit = sqrt(6 / (fan_in + fan_out)),

        with fan_in and fan_out the number of input and output units of the weight tensor.
        This choice keeps the variance of activations roughly constant across layers and
        reduces vanishing/exploding gradients in deep networks.

        Introduced in:
        X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,”
        AISTATS 2010. https://proceedings.mlr.press/v9/glorot10a.html
        )nbdoc");

    nb::enum_<ThorImplementation::Glorot::Mode>(glorot_class, "Mode")
        .value("NORMAL", ThorImplementation::Glorot::Mode::NORMAL)
        .value("UNIFORM", ThorImplementation::Glorot::Mode::UNIFORM)
        .export_values();
}
