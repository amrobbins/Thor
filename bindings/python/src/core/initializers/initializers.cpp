#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Initializers/Initializer.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

void bind_glorot(nb::module_ &m);
void bind_uniform_random(nb::module_ &m);

void bind_initializers(nb::module_ &initializers) {
    initializers.doc() = "Thor initializers";

    nb::class_<Initializer::Builder>(initializers, "Initializer");

    bind_glorot(initializers);
    bind_uniform_random(initializers);
}
