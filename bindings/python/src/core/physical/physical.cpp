#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Optimizers/Optimizer.h"
namespace nb = nanobind;

void bind_physical_tensor(nb::module_ &physical);
void bind_event(nb::module_ &physical);
void bind_stream(nb::module_ &physical);
void bind_machine_evaluator(nb::module_ &physical);
void bind_scoped_gpu(nb::module_ &physical);
void bind_physical_expression(nb::module_ &physical);
void bind_fused_equation(nb::module_ &physical);
void bind_stamped_equation(nb::module_ &physical);
void bind_physical_compile(nb::module_ &physical);

void bind_physical(nb::module_ &physical) {
    physical.doc() = "Thor physical layer";

    bind_physical_tensor(physical);
    bind_event(physical);
    bind_stream(physical);
    bind_machine_evaluator(physical);
    bind_scoped_gpu(physical);

    bind_physical_expression(physical);
    bind_fused_equation(physical);
    bind_stamped_equation(physical);
    bind_physical_compile(physical);
}
