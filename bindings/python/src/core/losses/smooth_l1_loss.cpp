#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/SmoothL1Loss.h"
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

void setReportedLossShape(SmoothL1Loss::Builder &builder, LossShape reported_loss_shape) {
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

void bind_smooth_l1_loss(nb::module_ &losses) {
    auto smooth_l1_loss = nb::class_<SmoothL1Loss, Loss>(losses, "SmoothL1Loss");
    smooth_l1_loss.attr("__module__") = "thor.losses";

    smooth_l1_loss.def(
        "__init__",
        [](SmoothL1Loss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float beta,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "SmoothL1Loss instance";
            if (predictions.getDimensions().size() != 1) {
                string error_message = loss_name + ": predictions must be a 1 dimensional tensor but predictions is " +
                                       predictions.getDescriptorString();
                throw nb::value_error(error_message.c_str());
            }
            if (labels.getDimensions() != predictions.getDimensions()) {
                string error_message = loss_name + ": labels dimensions " + labels.getDescriptorString() +
                                       " must match predictions dimensions " + predictions.getDescriptorString();
                throw nb::value_error(error_message.c_str());
            }
            if (beta <= 0.0f) {
                string error_message = loss_name + ": beta must be greater than zero";
                throw nb::value_error(error_message.c_str());
            }
            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
                string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
                throw nb::value_error(error_message.c_str());
            }
            validateReportedLossShape(reported_loss_shape, loss_name);

            SmoothL1Loss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).beta(beta).lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            SmoothL1Loss built = builder.build();

            new (self) SmoothL1Loss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "beta"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a Smooth L1 loss.)nbdoc");

    smooth_l1_loss.def_prop_ro("beta", &SmoothL1Loss::getBeta);

    smooth_l1_loss.attr("__doc__") = R"nbdoc(
Smooth L1 loss.

SmoothL1Loss uses the PyTorch-style beta parameterization:

    0.5 * (prediction - label)^2 / beta    if |prediction - label| < beta
    |prediction - label| - 0.5 * beta      otherwise

HuberLoss(delta=beta) is beta times SmoothL1Loss(beta=beta).
)nbdoc";
}
