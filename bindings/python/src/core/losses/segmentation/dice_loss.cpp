#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/DiceLoss.h"
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

void setReportedLossShape(DiceLoss::Builder &builder, LossShape reported_loss_shape) {
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
}  // namespace

void bind_dice_loss(nb::module_ &losses) {
    auto dice_loss = nb::class_<DiceLoss, Loss>(losses, "DiceLoss");
    dice_loss.attr("__module__") = "thor.losses.segmentation";

    dice_loss.def(
        "__init__",
        [](DiceLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float smooth,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "DiceLoss instance";
            if (predictions.getDimensions().empty()) {
                string error_message = loss_name + ": predictions must have at least one feature dimension but predictions is " +
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
            if (labels.getDataType() != DataType::BOOLEAN && labels.getDataType() != DataType::UINT8 &&
                labels.getDataType() != DataType::UINT16 && labels.getDataType() != DataType::UINT32 &&
                labels.getDataType() != DataType::FP16 && labels.getDataType() != DataType::FP32) {
                string error_message = loss_name + ": labels must use bool, uint8, uint16, uint32, fp16, or fp32 dtype";
                throw nb::value_error(error_message.c_str());
            }
            if (smooth < 0.0f) {
                string error_message = loss_name + ": smooth must be non-negative";
                throw nb::value_error(error_message.c_str());
            }
            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
                string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
                throw nb::value_error(error_message.c_str());
            }
            validateReportedLossShape(reported_loss_shape, loss_name);

            DiceLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).smooth(smooth).lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            DiceLoss built = builder.build();

            new (self) DiceLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "smooth"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a Dice loss.)nbdoc");

    dice_loss.def_prop_ro("smooth", &DiceLoss::getSmooth);

    dice_loss.attr("__doc__") = R"nbdoc(
Soft Dice loss over dense probability tensors.

Predictions are expected to already be probabilities, not logits. For a one-dimensional feature tensor [N], Dice is computed globally over N for each sample. For [C, ...spatial], Dice is computed per class/channel C by reducing only the spatial axes:

    1 - (2 * sum_spatial(prediction * label) + smooth) / (sum_spatial(prediction) + sum_spatial(label) + smooth)

Use a sigmoid/softmax activation before this loss if the model output is logits.
)nbdoc";
}
