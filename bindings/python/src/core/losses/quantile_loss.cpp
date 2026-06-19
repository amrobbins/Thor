#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
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

void setReportedLossShape(QuantileLoss::Builder &builder, LossShape reported_loss_shape) {
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

void bind_quantile_loss(nb::module_ &losses) {
    auto quantile_loss = nb::class_<QuantileLoss, Loss>(losses, "QuantileLoss");
    quantile_loss.attr("__module__") = "thor.losses";

    quantile_loss.def(
        "__init__",
        [](QuantileLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float quantile,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "QuantileLoss instance";
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
            if (quantile <= 0.0f || quantile >= 1.0f) {
                string error_message = loss_name + ": quantile must be greater than zero and less than one";
                throw nb::value_error(error_message.c_str());
            }
            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
                string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
                throw nb::value_error(error_message.c_str());
            }
            validateReportedLossShape(reported_loss_shape, loss_name);

            QuantileLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).quantile(quantile).lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            QuantileLoss built = builder.build();

            new (self) QuantileLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "quantile"_a = 0.5f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a Quantile / pinball loss.)nbdoc");

    quantile_loss.def_prop_ro("quantile", &QuantileLoss::getQuantile);

    losses.attr("PinballLoss") = quantile_loss;

    quantile_loss.attr("__doc__") = R"nbdoc(
Quantile / pinball loss.

For quantile q and error y_true - y_pred:

    q * error          if error > 0
    (q - 1) * error    otherwise

The subgradient at zero error is defined as 0.
)nbdoc";
}
