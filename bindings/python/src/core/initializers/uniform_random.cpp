#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

struct UniformRandomFactory {};

void bind_uniform_random(nb::module_ &m) {
    auto uf = nb::class_<UniformRandomFactory>(m, "_UniformRandomFactory", nb::dynamic_attr())
                  .def(nb::init<>())
                  .def(
                      "__call__",
                      [](UniformRandomFactory &, float min_value, float max_value) -> std::shared_ptr<Initializer> {
                          UniformRandom::Builder b;
                          b.minValue(min_value).maxValue(max_value);
                          return b.build();  // <-- must exist and return shared_ptr<Initializer>
                      },
                      "min_value"_a,
                      "max_value"_a,

                      nb::sig("def __call__(self, min_value: float, max_value: float) -> thor.initializers.Initializer"));

    // Export a single callable instance named "UniformRandom"
    nb::object inst = nb::cast(UniformRandomFactory{});
    inst.attr("__name__") = "UniformRandom";
    inst.attr("__qualname__") = "UniformRandom";
    inst.attr("__doc__") = R"doc(
Uniform random initializer factory.

UniformRandom(min_value: float, max_value: float) -> thor.initializers.Initializer

-------------------------------------------------------

Returns:
    A uniform random initializer.

    Draws each weight from a uniform distribution.

    U[min_value, max_value]

    Where min_value <= max_value. When min_value == max_value, the constant is written.
)doc";

    m.attr("UniformRandom") = inst;
}
