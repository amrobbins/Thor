#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Implementation/Initializers/Glorot.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_glorot(nb::module_ &m) {
    auto glorot = nb::class_<Glorot, Initializer>(m, "Glorot");
    glorot.attr("__module__") = "thor.initializers";

    nb::enum_<ThorImplementation::Glorot::Mode>(glorot, "Mode")
        .value("NORMAL", ThorImplementation::Glorot::Mode::NORMAL)
        .value("UNIFORM", ThorImplementation::Glorot::Mode::UNIFORM)
        .export_values();

    glorot.def_static(
        "__new__",
        [](nb::handle cls, ThorImplementation::Glorot::Mode mode) -> std::shared_ptr<Glorot> {
            Glorot::Builder b;
            b.mode(mode);
            shared_ptr<Initializer> base = b.build();
            shared_ptr<Glorot> g = std::dynamic_pointer_cast<Glorot>(base);
            if (!g)
                throw nb::type_error("Glorot builder did not return a Glorot instance");
            return g;  // nanobind converts shared_ptr<Glorot> to a Glorot Python object
        },
        "cls"_a,
        "mode"_a = ThorImplementation::Glorot::Mode::UNIFORM,
        // nb::sig(
        //     "def __new__(cls, mode: thor.initializers.Glorot.Mode = thor.initializers.Glorot.Mode.UNIFORM) -> thor.initializers.Glorot"),
        R"nbdoc(Construct a Glorot initializer.)nbdoc");

    // make __init__ a no-op
    glorot.def(
        "__init__",
        [](Glorot *, ThorImplementation::Glorot::Mode) {
            // no-op: constructed in __new__
        },
        "mode"_a = ThorImplementation::Glorot::Mode::UNIFORM,
        // nb::sig("def __init__(self, mode: thor.initializers.Glorot.Mode = thor.initializers.Glorot.Mode.UNIFORM) -> None"),
        R"nbdoc(Initialize a Glorot initializer (construction happens in __new__).)nbdoc");

    glorot.attr("__doc__") = R"doc(
    A Glorot (Xavier) initializer.

    Draws each weight from a uniform (or normal) distribution. See: Glorot.Mode.

    References:
    X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,”
    AISTATS 2010. https://proceedings.mlr.press/v9/glorot10a.html
)doc";
}
