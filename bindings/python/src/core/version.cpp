#include <nanobind/nanobind.h>
#include "ThorVersion.h"

namespace nb = nanobind;

void bind_version(nb::module_ &m) {
    // CMake-defined versions
    m.def("version", []() { return THOR_VERSION; });

    m.def("git_version", []() { return THOR_GIT_VERSION; });

    m.attr("__version__") = THOR_VERSION;
    m.attr("__git_version__") = THOR_GIT_VERSION;
}
