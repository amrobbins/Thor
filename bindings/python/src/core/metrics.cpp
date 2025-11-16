#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_metrics(py::module_ &m) {
    m.doc() = "Thor losses";
    m.def("BinaryAccuracy", []() { return "temp"; });
    m.def("CategoricalAccuracy", []() { return "temp"; });
}
