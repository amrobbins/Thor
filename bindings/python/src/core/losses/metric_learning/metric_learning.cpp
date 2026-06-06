#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_contrastive_loss(nb::module_ &metric_learning);
void bind_info_nce_loss(nb::module_ &metric_learning);
void bind_triplet_loss(nb::module_ &metric_learning);
void bind_cosine_embedding_loss(nb::module_ &metric_learning);

void bind_metric_learning_losses(nb::module_ &metric_learning) {
    metric_learning.doc() = "Thor metric learning losses";

    bind_contrastive_loss(metric_learning);
    bind_info_nce_loss(metric_learning);
    bind_triplet_loss(metric_learning);
    bind_cosine_embedding_loss(metric_learning);
}
