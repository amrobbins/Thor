#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/FocalTverskyLoss.h"
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

void setReportedLossShape(FocalTverskyLoss::Builder &builder, LossShape reported_loss_shape) {
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

void validateProbabilityLossTensors(const string &loss_name, Tensor predictions, Tensor labels) {
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
}
}  // namespace

void bind_focal_tversky_loss(nb::module_ &losses) {
    auto focal_tversky_loss = nb::class_<FocalTverskyLoss, Loss>(losses, "FocalTverskyLoss");
    focal_tversky_loss.attr("__module__") = "thor.losses";

    focal_tversky_loss.def(
        "__init__",
        [](FocalTverskyLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float alpha,
           float beta,
           float gamma,
           float smooth,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "FocalTverskyLoss instance";
            validateProbabilityLossTensors(loss_name, predictions, labels);
            if (alpha < 0.0f) {
                string error_message = loss_name + ": alpha must be non-negative";
                throw nb::value_error(error_message.c_str());
            }
            if (beta < 0.0f) {
                string error_message = loss_name + ": beta must be non-negative";
                throw nb::value_error(error_message.c_str());
            }
            if (gamma <= 0.0f) {
                string error_message = loss_name + ": gamma must be greater than zero";
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

            FocalTverskyLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .alpha(alpha)
                .beta(beta)
                .gamma(gamma)
                .smooth(smooth)
                .lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            FocalTverskyLoss built = builder.build();

            new (self) FocalTverskyLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "alpha"_a = 0.3f,
        "beta"_a = 0.7f,
        "gamma"_a = 0.75f,
        "smooth"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a focal Tversky loss.)nbdoc");

    focal_tversky_loss.def_prop_ro("alpha", &FocalTverskyLoss::getAlpha);
    focal_tversky_loss.def_prop_ro("beta", &FocalTverskyLoss::getBeta);
    focal_tversky_loss.def_prop_ro("gamma", &FocalTverskyLoss::getGamma);
    focal_tversky_loss.def_prop_ro("smooth", &FocalTverskyLoss::getSmooth);

    focal_tversky_loss.attr("__doc__") = R"nbdoc(
Focal Tversky loss over dense probability tensors.

Predictions are expected to already be probabilities, not logits. For a one-dimensional feature tensor [N], Focal Tversky is computed globally over N for each sample. For [C, ...spatial], Focal Tversky is computed per class/channel C by reducing only the spatial axes:

    (1 - TverskyIndex) ** gamma

where TverskyIndex = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth).
)nbdoc";
}
