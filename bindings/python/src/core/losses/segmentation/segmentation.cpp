#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_dice_loss(nb::module_ &segmentation);
void bind_tversky_loss(nb::module_ &segmentation);
void bind_focal_tversky_loss(nb::module_ &segmentation);

void bind_segmentation_losses(nb::module_ &segmentation) {
    segmentation.doc() = "Thor segmentation losses";

    bind_dice_loss(segmentation);
    bind_tversky_loss(segmentation);
    bind_focal_tversky_loss(segmentation);
}
