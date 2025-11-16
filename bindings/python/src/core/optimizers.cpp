#include <nanobind/nanobind.h>
namespace nb = nanobind;

void bind_optimizers(nb::module_ &m) {
    m.doc() = "Thor optimizers";

    m.def("Sgd", []() { return "temp"; });
    m.def("Adam", []() { return "temp"; });
}
