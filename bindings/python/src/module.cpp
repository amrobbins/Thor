#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for per-feature binders
void bind_version(py::module_ &m);
void bind_tensor(py::module_ &m);  // will be stubbed first
void bind_model(py::module_ &m);   // stub

PYBIND11_MODULE(_thor, m) {
    m.doc() = "Thor Python bindings";

    bind_version(m);
    bind_tensor(m);
    bind_model(m);
}