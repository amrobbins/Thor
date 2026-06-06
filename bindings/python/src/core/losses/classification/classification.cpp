#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_binary_focal_loss(nb::module_ &classification);
void bind_categorical_focal_loss(nb::module_ &classification);

void bind_classification_losses(nb::module_ &classification) {
    classification.doc() = "Thor classification losses";

    bind_binary_focal_loss(classification);
    bind_categorical_focal_loss(classification);
}
