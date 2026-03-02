#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Optimizers/Optimizer.h"
namespace nb = nanobind;

void bind_physical_tensor(nb::module_ &physical);
void bind_event(nb::module_ &physical);
void bind_stream(nb::module_ &physical);
void bind_machine_evaluator(nb::module_ &physical);

void bind_physical(nb::module_ &physical) {
    physical.doc() = "Thor physical layer";

    bind_physical_tensor(physical);
    bind_event(physical);
    bind_stream(physical);
    bind_machine_evaluator(physical);
}
