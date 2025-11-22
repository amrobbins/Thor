#include <nanobind/nanobind.h>


#include "DeepLearning/Api/Initializers/UniformRandom.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_uniform_random(nb::module_ &m) {
    auto uniformRandom_class = nb::class_<UniformRandom::Builder, Initializer::Builder>(m, "UniformRandom")
                                   .def(
                                       "__init__",
                                       [](UniformRandom::Builder *self, uint32_t min_value, uint32_t max_value) {
                                           // Create the uniformRandom in the pre-allocated but uninitialized memory at self
                                           UniformRandom::Builder builder;
                                           builder.minValue(min_value).maxValue(max_value);
                                           new (self) UniformRandom::Builder(std::move(builder));
                                       },
                                       "min_value"_a,
                                       "max_value"_a,

                                       nb::sig("def __init__(self, "
                                               "min_value: float, "
                                               "max_value: float"
                                               ") -> None"),

                                       R"nbdoc(
        Draws each weight from a uniform distribution.

        U[min_value, max_value]

        Where min_value <= max_value. When min_value == max_value, the constant is simply written to each weight.
        )nbdoc");
}
