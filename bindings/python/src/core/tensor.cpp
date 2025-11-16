#include <nanobind/nanobind.h>
namespace nb = nanobind;

void bind_tensor(nb::module_ &m) {
    m.def("Tensor", []() { return "temp"; });
}
