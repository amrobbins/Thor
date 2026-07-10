#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/Loss.h"

namespace nb = nanobind;

void bind_custom_loss(nb::module_ &losses);
void bind_binary_cross_entropy(nb::module_ &losses);
void bind_categorical_cross_entropy(nb::module_ &losses);
void bind_mean_absolute_error(nb::module_ &losses);
void bind_mean_absolute_percentage_error(nb::module_ &losses);
void bind_mean_squared_error(nb::module_ &losses);
void bind_mean_power_error(nb::module_ &losses);
void bind_smooth_l1_loss(nb::module_ &losses);
void bind_huber_loss(nb::module_ &losses);
void bind_soft_target_cross_entropy(nb::module_ &losses);
void bind_kl_div_loss(nb::module_ &losses);
void bind_ctc_loss(nb::module_ &losses);
void bind_quantile_loss(nb::module_ &losses);
void bind_expectile_loss(nb::module_ &losses);
void bind_asymmetric_power_loss(nb::module_ &losses);

void bind_classification_losses(nb::module_ &classification);
void bind_detection_losses(nb::module_ &detection);
void bind_distribution_losses(nb::module_ &distribution);
void bind_gan_losses(nb::module_ &gan);
void bind_metric_learning_losses(nb::module_ &metric_learning);
void bind_ranking_losses(nb::module_ &ranking);
void bind_segmentation_losses(nb::module_ &segmentation);

using namespace Thor;

void bind_losses(nb::module_ &losses) {
    losses.doc() = "Thor losses";

    auto loss = nb::class_<Loss>(losses, "Loss");
    loss.attr("__module__") = "thor.losses";
    loss.def("get_predictions", &Loss::getPredictions);
    loss.def("get_labels", &Loss::getLabels);
    loss.def("get_loss", &Loss::getLoss);
    loss.def("get_example_weights", &Loss::getExampleWeights);
    loss.def_prop_ro("loss_weight", &Loss::getLossWeight);
    loss.def_prop_ro("example_weights", &Loss::getExampleWeights);

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
    bind_mean_power_error(losses);
    bind_smooth_l1_loss(losses);
    bind_huber_loss(losses);
    bind_soft_target_cross_entropy(losses);
    bind_kl_div_loss(losses);
    bind_ctc_loss(losses);
    bind_quantile_loss(losses);
    bind_expectile_loss(losses);
    bind_asymmetric_power_loss(losses);

    auto classification = losses.def_submodule("classification");
    bind_classification_losses(classification);

    auto detection = losses.def_submodule("detection");
    bind_detection_losses(detection);

    auto distribution = losses.def_submodule("distribution");
    bind_distribution_losses(distribution);

    auto gan = losses.def_submodule("gan");
    bind_gan_losses(gan);

    auto metric_learning = losses.def_submodule("metric_learning");
    bind_metric_learning_losses(metric_learning);

    auto ranking = losses.def_submodule("ranking");
    bind_ranking_losses(ranking);

    auto segmentation = losses.def_submodule("segmentation");
    bind_segmentation_losses(segmentation);
}
