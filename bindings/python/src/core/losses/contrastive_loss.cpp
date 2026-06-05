#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/ContrastiveLoss.h"
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

void setReportedLossShape(ContrastiveLoss::Builder &builder, LossShape reported_loss_shape) {
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

bool isBinaryLabelDType(DataType dtype) {
    return dtype == DataType::BOOLEAN || dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::FP16 || dtype == DataType::FP32;
}

void validateContrastiveLossArguments(const string &loss_name,
                                      Tensor predictions,
                                      Tensor labels,
                                      float margin,
                                      optional<DataType> loss_data_type,
                                      LossShape reported_loss_shape) {
    if (predictions.getDimensions().size() != 1) {
        string error_message = loss_name + ": predictions must be a 1 dimensional distance tensor but predictions is " +
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
    if (!isBinaryLabelDType(labels.getDataType())) {
        string error_message = loss_name + ": labels must use bool, uint8, uint16, uint32, fp16, or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (margin <= 0.0f) {
        string error_message = loss_name + ": margin must be greater than zero";
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

void bind_contrastive_loss(nb::module_ &losses) {
    auto contrastive_loss = nb::class_<ContrastiveLoss, Loss>(losses, "ContrastiveLoss");
    contrastive_loss.attr("__module__") = "thor.losses";

    contrastive_loss.def(
        "__init__",
        [](ContrastiveLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float margin,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "ContrastiveLoss instance";
            validateContrastiveLossArguments(loss_name, predictions, labels, margin, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            ContrastiveLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).margin(margin).lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            ContrastiveLoss built = builder.build();

            new (self) ContrastiveLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "margin"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a distance-based contrastive loss.)nbdoc");

    contrastive_loss.def_prop_ro("margin", &ContrastiveLoss::getMargin);

    contrastive_loss.attr("__doc__") = R"nbdoc(
Distance-based contrastive loss.

The predictions tensor contains pair distances and the labels tensor contains binary pair labels.
Labels greater than 0.5 are treated as positive/similar pairs. The raw loss is:

    distance ** 2                         if label > 0.5
    max(margin - distance, 0) ** 2        otherwise

The predictions tensor is expected to contain non-negative distances; Thor does not clamp it.
)nbdoc";
}
