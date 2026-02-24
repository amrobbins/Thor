#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_uniform_random(nb::module_ &m) {
    auto uniform_random = nb::class_<UniformRandom, Initializer>(m, "UniformRandom");
    uniform_random.attr("__module__") = "thor.initializers";

    uniform_random.def_static(
        "__new__",
        [](nb::handle /*cls*/, float min_value, float max_value) -> std::shared_ptr<UniformRandom> {
            UniformRandom::Builder b;
            b.minValue(min_value).maxValue(max_value);

            shared_ptr<Initializer> base = b.build();
            shared_ptr<UniformRandom> ur = std::dynamic_pointer_cast<UniformRandom>(base);
            if (!ur)
                throw nb::type_error("UniformRandom builder did not return a UniformRandom instance");
            return ur;  // nanobind converts shared_ptr<UniformRandom> to a UniformRandom Python object
        },
        "cls"_a,
        "min_value"_a,
        "max_value"_a,
        // nb::sig("def __new__(cls, min_value: float, max_value: float) -> thor.initializers.UniformRandom"),
        R"nbdoc(Construct a UniformRandom initializer.)nbdoc");

    // Make __init__ a no-op (construction happens in __new__)
    uniform_random.def(
        "__init__",
        [](UniformRandom *, float, float) {
            // no-op: constructed in __new__
        },
        "min_value"_a,
        "max_value"_a,
        // nb::sig("def __init__(self, min_value: float, max_value: float) -> None"),
        R"nbdoc(Initialize a UniformRandom initializer.)nbdoc");

    uniform_random.attr("__doc__") = R"doc(
A uniform random initializer.

Draws each weight from a uniform distribution:

    U[min_value, max_value]

Where min_value <= max_value. When min_value == max_value, the constant is written to each weight.
)doc";
}
