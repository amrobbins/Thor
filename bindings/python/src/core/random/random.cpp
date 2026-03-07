#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_full_period_random(nb::module_ &random);

void bind_random(nb::module_ &random) {
    random.doc() = "Thor random utilities";

    bind_full_period_random(random);
}
