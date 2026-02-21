#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Softmax.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_softmax(nb::module_ &m) {
    auto softmax = nb::class_<Softmax, Activation>(m, "Softmax");

    softmax.def_static(
        "__new__",
        [](nb::handle /*cls*/) -> std::shared_ptr<Softmax> {
            Softmax::Builder b;

            std::shared_ptr<Activation> base = b.build();  // Builder returns shared_ptr<Activation>
            std::shared_ptr<Softmax> sm = std::dynamic_pointer_cast<Softmax>(base);
            if (!sm)
                throw nb::type_error("Softmax builder did not return a Softmax instance");
            return sm;
        },
        "cls"_a,
        nb::sig("def __new__(cls) -> thor.layers.activations.Softmax"),
        R"nbdoc(Construct a Softmax activation.)nbdoc");

    // No-op __init__ (construction happens in __new__)
    softmax.def(
        "__init__",
        [](Softmax *) {
            // no-op: constructed in __new__
        },
        nb::sig("def __init__(self) -> None"),
        R"nbdoc(Initialize a Softmax activation (construction happens in __new__).)nbdoc");

    softmax.attr("__doc__") = R"doc(
Softmax activation.

Softmax is typically applied along the last (feature) dimension of a tensor
to convert raw scores (logits) into a probability distribution. For an input
vector x, softmax is defined as

    softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

for each component i. The outputs are all positive and sum to 1, making
softmax a natural choice for multi-class classification outputs. A numerically
stable form (subtracting max(x) before exponentiation) is used to avoid overflow.
)doc";
}
