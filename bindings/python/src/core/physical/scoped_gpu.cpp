#include <nanobind/nanobind.h>

#include "Utilities/ComputeTopology/MachineEvaluator.h"

namespace nb = nanobind;
using namespace nb::literals;

struct PyScopedGpu {
    int prev = -1;
    int cur = -1;
    bool active = false;

    explicit PyScopedGpu(int gpu) {
        cur = gpu;
        prev = MachineEvaluator::swapActiveDevice(gpu);
        active = true;
    }

    void exit() {
        if (active && prev >= 0 && prev != cur) {
            MachineEvaluator::swapActiveDevice(prev);
        }
        active = false;
    }

    ~PyScopedGpu() { exit(); }  // fallback only
};

void bind_scoped_gpu(nb::module_ &physical) {
    auto scoped_gpu = nb::class_<PyScopedGpu>(physical, "ScopedGpu");
    scoped_gpu.attr("__module__") = "thor.physical";

    scoped_gpu.def("__init__", [](PyScopedGpu *self, int gpu_num) { new (self) PyScopedGpu(gpu_num); }, "gpu_num"_a);
    scoped_gpu.def("__enter__", [](PyScopedGpu &self) -> PyScopedGpu & { return self; }, nb::rv_policy::reference_internal);
    scoped_gpu.def(
        "__exit__",
        [](PyScopedGpu &self, nb::handle, nb::handle, nb::handle) { self.exit(); },
        "exc_type"_a.none() = nb::none(),
        "exc"_a.none() = nb::none(),
        "tb"_a.none() = nb::none());
}
