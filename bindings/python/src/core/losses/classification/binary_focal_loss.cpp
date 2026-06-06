#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/BinaryFocalLoss.h"
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
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(BinaryFocalLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
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

void validateBinaryFocalLossArguments(const string &loss_name,
                                      Tensor predictions,
                                      Tensor labels,
                                      float gamma,
                                      float alpha,
                                      optional<DataType> loss_data_type,
                                      LossShape reported_loss_shape) {
    if (predictions.getDimensions().size() != 1 || predictions.getDimensions()[0] != 1) {
        string error_message = loss_name + ": predictions must be a 1 dimensional logits tensor of size one but predictions is " +
                               predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (labels.getDimensions().size() != 1 || labels.getDimensions()[0] != 1) {
        string error_message = loss_name + ": labels must be a 1 dimensional tensor of size one but labels is " +
                               labels.getDescriptorString();
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
    if (gamma < 0.0f) {
        string error_message = loss_name + ": gamma must be non-negative";
        throw nb::value_error(error_message.c_str());
    }
    if (alpha < 0.0f || alpha > 1.0f) {
        string error_message = loss_name + ": alpha must be in the range [0, 1]";
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

void bind_binary_focal_loss(nb::module_ &losses) {
    auto binary_focal_loss = nb::class_<BinaryFocalLoss, Loss>(losses, "BinaryFocalLoss");
    binary_focal_loss.attr("__module__") = "thor.losses.classification";

    binary_focal_loss.def(
        "__init__",
        [](BinaryFocalLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float gamma,
           float alpha,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "BinaryFocalLoss instance";
            validateBinaryFocalLossArguments(loss_name, predictions, labels, gamma, alpha, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            BinaryFocalLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .focusingParameter(gamma)
                .alpha(alpha)
                .lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            BinaryFocalLoss built = builder.build();

            new (self) BinaryFocalLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "gamma"_a = 2.0f,
        "alpha"_a = 0.25f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a binary focal loss from logits.)nbdoc");

    binary_focal_loss.def_prop_ro("gamma", &BinaryFocalLoss::getGamma);
    binary_focal_loss.def_prop_ro("alpha", &BinaryFocalLoss::getAlpha);

    binary_focal_loss.attr("__doc__") = R"nbdoc(
Binary focal loss from logits.

The predictions tensor contains one unnormalized logit per example and the labels tensor contains
binary targets. The raw loss is:

    alpha_t * (1 - p_t) ** gamma * BCEWithLogits(logit, target)

where alpha_t is alpha for positive targets and 1 - alpha for negative targets.
)nbdoc";
}
