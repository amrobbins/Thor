#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/CategoricalFocalLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string &loss_name) {
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::CLASSWISE &&
        reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(CategoricalFocalLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::CLASSWISE) {
        builder.reportsPerOutputLoss();
    } else if (reported_loss_shape == LossShape::ELEMENTWISE) {
        builder.reportsElementwiseLoss();
    } else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void validateCategoricalFocalLossArguments(const string &loss_name,
                                           Tensor predictions,
                                           Tensor labels,
                                           float gamma,
                                           float alpha,
                                           optional<DataType> loss_data_type,
                                           LossShape reported_loss_shape) {
    if (predictions.getDimensions().size() != 1) {
        string error_message = loss_name + ": predictions must be a 1 dimensional logits tensor but predictions is " +
                               predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (labels.getDimensions() != predictions.getDimensions()) {
        string error_message = loss_name + ": labels dimensions " + labels.getDescriptorString() +
                               " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (predictions.getDataType() != DataType::FP16 && predictions.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": predictions must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (labels.getDataType() != DataType::FP16 && labels.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": labels must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (gamma < 0.0f) {
        string error_message = loss_name + ": gamma must be non-negative";
        throw nb::value_error(error_message.c_str());
    }
    if (alpha < 0.0f) {
        string error_message = loss_name + ": alpha must be non-negative";
        throw nb::value_error(error_message.c_str());
    }
    DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
    if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_categorical_focal_loss(nb::module_ &losses) {
    auto categorical_focal_loss = nb::class_<CategoricalFocalLoss, Loss>(losses, "CategoricalFocalLoss");
    categorical_focal_loss.attr("__module__") = "thor.losses.classification";

    categorical_focal_loss.def(
        "__init__",
        [](CategoricalFocalLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float gamma,
           float alpha,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "CategoricalFocalLoss instance";
            validateCategoricalFocalLossArguments(loss_name, predictions, labels, gamma, alpha, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            CategoricalFocalLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .focusingParameter(gamma)
                .alpha(alpha)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            CategoricalFocalLoss built = builder.build();

            new (self) CategoricalFocalLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "gamma"_a = 2.0f,
        "alpha"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a categorical focal loss from logits and dense targets.)nbdoc");

    categorical_focal_loss.def_prop_ro("gamma", &CategoricalFocalLoss::getGamma);
    categorical_focal_loss.def_prop_ro("alpha", &CategoricalFocalLoss::getAlpha);

    categorical_focal_loss.attr("__doc__") = R"nbdoc(
Categorical focal loss from logits and dense target distributions.

The predictions tensor contains unnormalized logits and the labels tensor contains a dense
one-hot or soft target distribution with the same class dimension. The raw loss is:

    -alpha * target * (1 - softmax(logits)) ** gamma * log_softmax(logits)

For sparse class-index targets, use a sparse focal wrapper later rather than this dense-target loss.
)nbdoc";
}
