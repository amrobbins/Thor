#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/HuberLoss.h"
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

void setReportedLossShape(HuberLoss::Builder &builder, LossShape reported_loss_shape) {
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

void bind_huber_loss(nb::module_ &losses) {
    auto huber_loss = nb::class_<HuberLoss, Loss>(losses, "HuberLoss");
    huber_loss.attr("__module__") = "thor.losses";

    huber_loss.def(
        "__init__",
        [](HuberLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float delta,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "HuberLoss instance";
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
            if (delta <= 0.0f) {
                string error_message = loss_name + ": delta must be greater than zero";
                throw nb::value_error(error_message.c_str());
            }
            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
                string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
                throw nb::value_error(error_message.c_str());
            }
            validateReportedLossShape(reported_loss_shape, loss_name);

            HuberLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).delta(delta).lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            HuberLoss built = builder.build();

            new (self) HuberLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "delta"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a Huber loss.)nbdoc");

    huber_loss.def_prop_ro("delta", &HuberLoss::getDelta);

    huber_loss.attr("__doc__") = R"nbdoc(
Huber loss.

HuberLoss uses the standard delta parameterization:

    0.5 * (prediction - label)^2                    if |prediction - label| <= delta
    delta * (|prediction - label| - 0.5 * delta)    otherwise

HuberLoss(delta=beta) is beta times SmoothL1Loss(beta=beta).
)nbdoc";
}
