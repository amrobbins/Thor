#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>

#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

namespace nb = nanobind;
using namespace nb::literals;

using Placement = ThorImplementation::TensorPlacement;

void bind_stream(nb::module_ &physical) {
    auto stream = nb::class_<Stream>(physical, "Stream");
    stream.attr("__module__") = "thor.physical";

    // Constructors
    stream.def(
        "__init__",
        [](Stream *self, int32_t gpu_num) {
            if (gpu_num < 0)
                throw nb::value_error("gpu_num must be >= 0");
            new (self) Stream(gpu_num, Stream::Priority::REGULAR);
        },
        "gpu_num"_a = 0,
        R"nbdoc(
Stream(gpu_num=0)

Create a CUDA stream on the specified GPU.
Priority is always REGULAR in the Python API.

Parameters
----------
gpu_num : int, default 0
)nbdoc");

    stream.def(
        "__init__",
        [](Stream *self, const Placement &placement) { new (self) Stream(placement, Stream::Priority::REGULAR); },
        "placement"_a,
        R"nbdoc(
Stream(placement)

Create a CUDA stream based on a tensor placement.
Priority is always REGULAR in the Python API.

Parameters
----------
placement : thor.physical.Placement
)nbdoc");

    // Copy support
    stream.def("__copy__", [](const Stream &self) { return Stream(self); });
    stream.def("__deepcopy__", [](const Stream &self, nb::handle /*memo*/) { return Stream(self); }, "memo"_a);

    // Introspection
    stream.def("get_gpu_num", &Stream::getGpuNum);
    stream.def("get_id", &Stream::getId);

    // Synchronization
    stream.def(
        "synchronize",
        [](Stream &self) {
            nb::gil_scoped_release release;
            self.synchronize();
        },
        R"nbdoc(
Block until all queued work on this stream has completed.
)nbdoc");

    stream.def_static(
        "device_synchronize",
        [](int32_t gpu_num) {
            if (gpu_num < 0)
                throw nb::value_error("gpu_num must be >= 0");
            nb::gil_scoped_release release;
            Stream::deviceSynchronize(gpu_num);
        },
        "gpu_num"_a = 0,
        R"nbdoc(
device_synchronize(gpu_num=0)

Block until all work on the specified device has completed.
)nbdoc");

    // Events
    stream.def(
        "put_event",
        [](Stream &self, bool enable_timing, bool expecting_host_to_wait) {
            // Event creation/record doesn't need the GIL; but usually it's fast.
            // If you prefer, you can also release the GIL here.
            return self.putEvent(enable_timing, expecting_host_to_wait);
        },
        "enable_timing"_a = false,
        "expecting_host_to_wait"_a = false,
        R"nbdoc(
Create and record an event on this stream.

Returns
-------
thor.physical.Event
)nbdoc");

    stream.def(
        "wait_event",
        [](Stream &self, const Event &event) { self.waitEvent(event); },
        "event"_a,
        R"nbdoc(
Make this stream wait until the given event is completed.
)nbdoc");

    stream.def("__repr__", [](const Stream &self) {
        return "Stream(gpu_num=" + std::to_string(self.getGpuNum()) + ", id=" + std::to_string(self.getId()) + ")";
    });
}
