#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations for per-feature binders
void bind_version(py::module_ &m);
void bind_tensor(py::module_ &m);
void bind_activations(py::module_ &m);
void bind_layers(py::module_ &m);
void bind_losses(py::module_ &m);
void bind_metrics(py::module_ &m);
void bind_optimizers(py::module_ &m);

PYBIND11_MODULE(thor, m) {
    m.doc() = "Thor Python bindings";

    bind_version(m);
    bind_tensor(m);

    auto activations = m.def_submodule("activations");
    bind_activations(activations);

    auto layers = m.def_submodule("layers");
    bind_layers(layers);

    auto losses = m.def_submodule("losses");
    bind_losses(losses);

    auto metrics = m.def_submodule("metrics");
    bind_metrics(metrics);

    auto optimizers = m.def_submodule("optimizers");
    bind_optimizers(optimizers);
}
