#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_box_iou_losses(nb::module_ &detection);

void bind_detection_losses(nb::module_ &detection) {
    detection.doc() = "Thor object detection losses";

    bind_box_iou_losses(detection);
}
