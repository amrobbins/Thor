#include <nanobind/nanobind.h>
namespace nb = nanobind;

void bind_losses(nb::module_ &m) {
    m.doc() = "Thor losses";
    m.def("BinaryCrossEntropy", []() { return "temp"; });
    m.def("CategoricalCrossEntropy", []() { return "temp"; });
    m.def("MeanAbsoluteError", []() { return "temp"; });
    m.def("MeanAbsolutePercentageError", []() { return "temp"; });
    m.def("MeanSquaredError", []() { return "temp"; });
}
