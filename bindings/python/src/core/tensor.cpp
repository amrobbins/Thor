#include <pybind11/pybind11.h>
#include "ThorVersion.h"

namespace py = pybind11;

void bind_tensor(py::module_ &m) {
    m.def("Tensor", []() { return "temp"; });
}
