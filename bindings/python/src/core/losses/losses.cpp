#include <nanobind/nanobind.h>

#include "DeepLearning/Api/Layers/Loss/Loss.h"

namespace nb = nanobind;

void bind_binary_cross_entropy(nb::module_ &losses);
void bind_categorical_cross_entropy(nb::module_ &losses);
void bind_mean_absolute_error(nb::module_ &losses);
void bind_mean_absolute_percentage_error(nb::module_ &losses);
void bind_mean_squared_error(nb::module_ &losses);

using namespace Thor;

void bind_losses(nb::module_ &losses) {
    losses.doc() = "Thor losses";

    nb::class_<Loss>(losses, "Loss");

    bind_binary_cross_entropy(losses);
    // bind_categorical_cross_entropy(losses);
    // bind_mean_absolute_error(losses);
    // bind_mean_absolute_percentage_error(losses);
    // bind_mean_squared_error(losses);

    // losses.def("BinaryCrossEntropy", []() { return "temp"; });
    // losses.def("CategoricalCrossEntropy", []() { return "temp"; });
    // losses.def("MeanAbsoluteError", []() { return "temp"; });
    // losses.def("MeanAbsolutePercentageError", []() { return "temp"; });
    // losses.def("MeanSquaredError", []() { return "temp"; });

    auto label_type =
        nb::enum_<Loss::LabelType>(losses, "LabelType").value("index", Loss::LabelType::INDEX).value("one_hot", Loss::LabelType::ONE_HOT);

    auto loss_shape = nb::enum_<Loss::LossShape>(losses, "LossShape")
                          .value("batch", Loss::LossShape::BATCH)
                          .value("classwise", Loss::LossShape::CLASSWISE)
                          .value("elementwise", Loss::LossShape::ELEMENTWISE)
                          .value("raw", Loss::LossShape::RAW);
}
