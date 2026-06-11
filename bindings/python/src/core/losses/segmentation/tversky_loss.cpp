#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/TverskyLoss.h"
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

void setReportedLossShape(TverskyLoss::Builder &builder, LossShape reported_loss_shape) {
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

void bind_tversky_loss(nb::module_ &losses) {
    auto tversky_loss = nb::class_<TverskyLoss, Loss>(losses, "TverskyLoss");
    tversky_loss.attr("__module__") = "thor.losses.segmentation";

    tversky_loss.def(
        "__init__",
        [](TverskyLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float alpha,
           float beta,
           float smooth,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "TverskyLoss instance";
            validateProbabilityLossTensors(loss_name, predictions, labels);
            if (alpha < 0.0f) {
                string error_message = loss_name + ": alpha must be non-negative";
                throw nb::value_error(error_message.c_str());
            }
            if (beta < 0.0f) {
                string error_message = loss_name + ": beta must be non-negative";
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

            TverskyLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .alpha(alpha)
                .beta(beta)
                .smooth(smooth)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            TverskyLoss built = builder.build();

            new (self) TverskyLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "alpha"_a = 0.5f,
        "beta"_a = 0.5f,
        "smooth"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a Tversky loss.)nbdoc");

    tversky_loss.def_prop_ro("alpha", &TverskyLoss::getAlpha);
    tversky_loss.def_prop_ro("beta", &TverskyLoss::getBeta);
    tversky_loss.def_prop_ro("smooth", &TverskyLoss::getSmooth);

    tversky_loss.attr("__doc__") = R"nbdoc(
Tversky loss over dense probability tensors.

Predictions are expected to already be probabilities, not logits. For a one-dimensional feature tensor [N], Tversky is computed globally over N for each sample. For [C, ...spatial], Tversky is computed per class/channel C by reducing only the spatial axes:

    1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

alpha weights false positives and beta weights false negatives.
)nbdoc";
}
