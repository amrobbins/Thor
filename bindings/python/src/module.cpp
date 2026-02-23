#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declarations for per-feature binders
void bind_version(nb::module_ &m);
void bind_network(nb::module_ &m);
void bind_tensor(nb::module_ &thor);

void bind_activations(nb::module_ &m);
void bind_initializers(nb::module_ &initializers);
void bind_layers(nb::module_ &layers);
void bind_losses(nb::module_ &losses);
void bind_metrics(nb::module_ &m);
void bind_optimizers(nb::module_ &m);

NB_MODULE(_thor, m) {
    m.doc() = "Thor Python bindings";

    auto DEFAULT = nb::capsule((void *)0x1, "thor.default");
    m.attr("DEFAULT") = DEFAULT;

    bind_version(m);
    bind_tensor(m);
    bind_network(m);

    auto activations = m.def_submodule("activations");
    bind_activations(activations);

    auto initializers = m.def_submodule("initializers");
    bind_initializers(initializers);

    auto layers = m.def_submodule("layers");
    bind_layers(layers);

    auto losses = m.def_submodule("losses");
    bind_losses(losses);

    auto metrics = m.def_submodule("metrics");
    bind_metrics(metrics);

    auto optimizers = m.def_submodule("optimizers");
    bind_optimizers(optimizers);
}
