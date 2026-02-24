#include <nanobind/nanobind.h>
#include "ThorVersion.h"

namespace nb = nanobind;

void bind_version(nb::module_ &thor) {
    // CMake-defined versions
    thor.def("version", []() { return THOR_VERSION; });

    thor.def("git_version", []() { return THOR_GIT_VERSION; });

    thor.attr("__version__") = THOR_VERSION;
    thor.attr("__git_version__") = THOR_GIT_VERSION;
}
