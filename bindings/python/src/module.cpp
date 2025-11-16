#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for per-feature binders
void bind_version(py::module_ &m);
void bind_layers(py::module_ &m);
void bind_optimizers(py::module_ &m);

PYBIND11_MODULE(thor, m) {
    m.doc() = "Thor Python bindings";

    bind_version(m);

    auto layers = m.def_submodule("layers", "Thor layers");
    bind_layers(layers);

    auto optimizers = m.def_submodule("optimizers", "Thor optimizers");
    bind_optimizers(optimizers);
}
