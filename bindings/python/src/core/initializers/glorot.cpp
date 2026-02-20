#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Implementation/Initializers/Glorot.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

struct GlorotFactory {};

void bind_glorot(nb::module_ &m) {
    // Create a Python type for the factory object
    auto gf = nb::class_<GlorotFactory>(m, "_GlorotFactory", nb::dynamic_attr());

    // Attach enum as nested attribute of the factory TYPE
    nb::enum_<ThorImplementation::Glorot::Mode>(gf, "Mode")
        .value("NORMAL", ThorImplementation::Glorot::Mode::NORMAL)
        .value("UNIFORM", ThorImplementation::Glorot::Mode::UNIFORM)
        .export_values();

    gf.def(nb::init<>());

    gf.def(
        "__call__",
        [](GlorotFactory &, ThorImplementation::Glorot::Mode mode) -> std::shared_ptr<Initializer> {
            Glorot::Builder b;
            b.mode(mode);
            return b.build();
        },
        "mode"_a = ThorImplementation::Glorot::Mode::UNIFORM,

        // Signature shown for Glorot(...)
        nb::sig("def __call__(self, mode: thor.initializers.Glorot.Mode = thor.initializers.Glorot.Mode.UNIFORM) -> "
                "thor.initializers.Initializer"),

        R"nbdoc(

)nbdoc");

    nb::object inst = nb::cast(GlorotFactory{});
    inst.attr("__name__") = "Glorot";
    inst.attr("__qualname__") = "Glorot";
    inst.attr("__doc__") = R"doc(
A Glorot (Xavier) weight initializer factory.

Glorot(mode: thor.initializers.Glorot.Mode = thor.initializers.Glorot.Mode.UNIFORM) -> thor.initializers.Initializer

See: Glorot.Mode

------------------------------------------------

Returns:
    A Glorot (Xavier) initializer.

    Draws each weight from a uniform (or normal) distribution.

    References:
    X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,”
    AISTATS 2010. https://proceedings.mlr.press/v9/glorot10a.html
)doc";

    m.attr("Glorot") = inst;
}
