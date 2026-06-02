#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Optimizers/Optimizer.h"
namespace nb = nanobind;

void bind_sgd(nb::module_ &optimizers);
void bind_adam(nb::module_ &optimizers);
void bind_adamw(nb::module_ &optimizers);
void bind_adamax(nb::module_ &optimizers);
void bind_nadam(nb::module_ &optimizers);
void bind_radam(nb::module_ &optimizers);
void bind_adagrad(nb::module_ &optimizers);
void bind_adadelta(nb::module_ &optimizers);
void bind_adafactor(nb::module_ &optimizers);
void bind_rmsprop(nb::module_ &optimizers);
void bind_lamb(nb::module_ &optimizers);
void bind_muon(nb::module_ &optimizers);

void bind_optimizers(nb::module_ &optimizers) {
    optimizers.doc() = "Thor optimizers";

    auto optimizer = nb::class_<Thor::Optimizer>(optimizers, "Optimizer");
    optimizer.attr("__module__") = "thor.optimizers";

    bind_sgd(optimizers);
    bind_adam(optimizers);
    bind_adamw(optimizers);
    bind_adamax(optimizers);
    bind_nadam(optimizers);
    bind_radam(optimizers);
    bind_adagrad(optimizers);
    bind_adadelta(optimizers);
    bind_adafactor(optimizers);
    bind_rmsprop(optimizers);
    bind_lamb(optimizers);
    bind_muon(optimizers);
}
