#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_optimizers(py::module_ &m) {
    m.doc() = "Thor optimizers";

    m.def("Sgd", []() { return "temp"; });
    m.def("Adam", []() { return "temp"; });
}
