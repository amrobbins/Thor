#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_losses(py::module_ &m) {
    m.doc() = "Thor losses";
    m.def("BinaryCrossEntropy", []() { return "temp"; });
    m.def("CategoricalCrossEntropy", []() { return "temp"; });
    m.def("MeanAbsoluteError", []() { return "temp"; });
    m.def("MeanAbsolutePercentageError", []() { return "temp"; });
    m.def("MeanSquaredError", []() { return "temp"; });
}
