#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_list_net_loss(nb::module_ &ranking);
void bind_listwise_softmax_cross_entropy_loss(nb::module_ &ranking);
void bind_margin_ranking_loss(nb::module_ &ranking);

void bind_ranking_losses(nb::module_ &ranking) {
    ranking.doc() = "Thor ranking losses";

    bind_list_net_loss(ranking);
    bind_listwise_softmax_cross_entropy_loss(ranking);
    bind_margin_ranking_loss(ranking);
}
