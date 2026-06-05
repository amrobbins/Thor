#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Loss/Loss.h"

namespace nb = nanobind;

void bind_custom_loss(nb::module_ &losses);
void bind_binary_cross_entropy(nb::module_ &losses);
void bind_categorical_cross_entropy(nb::module_ &losses);
void bind_mean_absolute_error(nb::module_ &losses);
void bind_mean_absolute_percentage_error(nb::module_ &losses);
void bind_mean_squared_error(nb::module_ &losses);
void bind_smooth_l1_loss(nb::module_ &losses);
void bind_huber_loss(nb::module_ &losses);
void bind_contrastive_loss(nb::module_ &losses);
void bind_info_nce_loss(nb::module_ &losses);
void bind_triplet_loss(nb::module_ &losses);
void bind_cosine_embedding_loss(nb::module_ &losses);
void bind_soft_target_cross_entropy(nb::module_ &losses);
void bind_kl_div_loss(nb::module_ &losses);
void bind_binary_focal_loss(nb::module_ &losses);
void bind_categorical_focal_loss(nb::module_ &losses);
void bind_dice_loss(nb::module_ &losses);
void bind_tversky_loss(nb::module_ &losses);
void bind_focal_tversky_loss(nb::module_ &losses);

using namespace Thor;

void bind_losses(nb::module_ &losses) {
    losses.doc() = "Thor losses";

    auto loss = nb::class_<Loss>(losses, "Loss");
    loss.attr("__module__") = "thor.losses";
    loss.def("get_predictions", &Loss::getPredictions);
    loss.def("get_labels", &Loss::getLabels);
    loss.def("get_loss", &Loss::getLoss);

    auto label_type =
        nb::enum_<Loss::LabelType>(losses, "LabelType").value("index", Loss::LabelType::INDEX).value("one_hot", Loss::LabelType::ONE_HOT);
    label_type.attr("__module__") = "thor.losses";
    // label_type.attr("__qualname__") = "Loss.LabelType";
    loss.attr("LabelType") = label_type;

    auto loss_shape = nb::enum_<Loss::LossShape>(losses, "LossShape")
                          .value("batch", Loss::LossShape::BATCH)
                          .value("classwise", Loss::LossShape::CLASSWISE)
                          .value("elementwise", Loss::LossShape::ELEMENTWISE)
                          .value("raw", Loss::LossShape::RAW);
    loss_shape.attr("__module__") = "thor.losses";
    // loss_shape.attr("__qualname__") = "Loss.LossShape";
    loss.attr("LossShape") = loss_shape;

    bind_custom_loss(losses);
    bind_binary_cross_entropy(losses);
    bind_categorical_cross_entropy(losses);
    bind_mean_absolute_error(losses);
    bind_mean_absolute_percentage_error(losses);
    bind_mean_squared_error(losses);
    bind_smooth_l1_loss(losses);
    bind_huber_loss(losses);
    bind_contrastive_loss(losses);
    bind_info_nce_loss(losses);
    bind_triplet_loss(losses);
    bind_cosine_embedding_loss(losses);
    bind_soft_target_cross_entropy(losses);
    bind_kl_div_loss(losses);
    bind_binary_focal_loss(losses);
    bind_categorical_focal_loss(losses);
    bind_dice_loss(losses);
    bind_tversky_loss(losses);
    bind_focal_tversky_loss(losses);
}
