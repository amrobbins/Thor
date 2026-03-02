#include <nanobind/nanobind.h>

#include <nanobind/stl/string.h>

#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_event(nb::module_ &physical) {
    auto ev = nb::class_<Event>(physical, "Event");
    ev.attr("__module__") = "thor.physical";

    // Constructors
    ev.def(
        "__init__",
        [](Event *self, int32_t gpu_num, bool enable_timing, bool expecting_host_to_wait) {
            if (gpu_num < 0)
                throw nb::value_error("gpu_num must be >= 0");
            new (self) Event(gpu_num, enable_timing, expecting_host_to_wait);
        },
        "gpu_num"_a = 0,
        "enable_timing"_a = false,
        "expecting_host_to_wait"_a = false,
        R"nbdoc(
Event(gpu_num=0, enable_timing=False, expecting_host_to_wait=False)

Create a CUDA event.

Parameters
----------
gpu_num : int, default 0
enable_timing : bool, default False
expecting_host_to_wait : bool, default False
)nbdoc");

    // Copy support
    ev.def("__copy__", [](const Event &self) { return Event(self); });
    ev.def("__deepcopy__", [](const Event &self, nb::handle /*memo*/) { return Event(self); }, "memo"_a);

    // Core API
    ev.def("get_gpu_num", &Event::getGpuNum);
    ev.def("get_id", &Event::getId);

    ev.def(
        "record",
        [](Event &self, Stream stream) {
            // record is usually fast; no need to release GIL
            self.record(stream);
        },
        "stream"_a,
        R"nbdoc(
Record this event on the given stream.
)nbdoc");

    ev.def(
        "synchronize",
        [](Event &self) {
            nb::gil_scoped_release release;
            self.synchronize();
        },
        R"nbdoc(
Block until this event is completed.
)nbdoc");

    ev.def(
        "synchronize_and_report_elapsed_time_ms",
        [](Event &self, const Event &start_event) {
            nb::gil_scoped_release release;
            return self.synchronizeAndReportElapsedTimeInMilliseconds(start_event);
        },
        "start_event"_a,
        R"nbdoc(
Synchronize this event and return elapsed time in milliseconds since start_event.

Returns
-------
float
)nbdoc");

    ev.def("__repr__", [](const Event &self) {
        return "Event(gpu_num=" + std::to_string(self.getGpuNum()) + ", id=" + std::to_string(self.getId()) + ")";
    });
}
