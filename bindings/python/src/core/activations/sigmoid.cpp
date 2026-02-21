#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_sigmoid(nb::module_ &m) {
    auto sigmoid = nb::class_<Sigmoid, Activation>(m, "Sigmoid");

    sigmoid.def_static(
        "__new__",
        [](nb::handle cls) -> std::shared_ptr<Sigmoid> {
            Sigmoid::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Sigmoid> s = std::dynamic_pointer_cast<Sigmoid>(base);
            if (!s)
                throw nb::type_error("Sigmoid builder did not return a Sigmoid instance");
            return s;
        },
        "cls"_a,
        nb::sig("def __new__(cls) -> thor.layers.activations.Sigmoid"),
        R"nbdoc(Construct a Sigmoid activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    sigmoid.def(
        "__init__",
        [](Sigmoid *self) -> void {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a Sigmoid activation (construction happens in __new__).)nbdoc");

    sigmoid.attr("__doc__") = R"doc(
Sigmoid activation.

Applied elementwise, this activation is defined as

    f(x) = 1 / (1 + exp(-x))

It maps real-valued inputs into the interval (0, 1) and is commonly used
when outputs are interpreted as probabilities or gates (e.g., in recurrent
networks). Note that sigmoid can suffer from saturation for large |x|,
which may slow down learning if not combined with appropriate initialization
or normalization.
)doc";
}
