#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Optimizers/Optimizer.h"
namespace nb = nanobind;

void bind_sgd(nb::module_ &optimizers);
void bind_adam(nb::module_ &optimizers);

void bind_optimizers(nb::module_ &optimizers) {
    optimizers.doc() = "Thor optimizers";

    auto optimizer = nb::class_<Thor::Optimizer>(optimizers, "Optimizer");
    optimizer.attr("__module__") = "thor.optimizers";

    bind_sgd(optimizers);
    bind_adam(optimizers);
}
