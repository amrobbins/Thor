#include <nanobind/nanobind.h>
namespace nb = nanobind;

void bind_metrics(nb::module_ &m) {
    m.doc() = "Thor losses";
    m.def("BinaryAccuracy", []() { return "temp"; });
    m.def("CategoricalAccuracy", []() { return "temp"; });
}
