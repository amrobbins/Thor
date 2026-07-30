#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/MeanPowerError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "bindings/python/src/core/losses/regression_loss_dtype.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;
using LabelType = Loss::LabelType;

namespace {

void validateReportedLossShape(LossShape reported_loss_shape, const string& loss_name) {
    if (reported_loss_shape != LossShape::NONE && reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::PER_EXAMPLE &&
        reported_loss_shape != LossShape::PER_OUTPUT && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(MeanPowerError::Builder& builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::NONE) {
        builder.reportsNoLoss();
    } else if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::PER_EXAMPLE) {
        builder.reportsPerExampleLoss();
    } else if (reported_loss_shape == LossShape::PER_OUTPUT) {
        builder.reportsPerOutputLoss();
    } else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void maybeSetExampleWeights(MeanPowerError::Builder& builder,
                            Tensor predictions,
                            Tensor labels,
                            std::optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions || example_weights.value() == labels)
        throw nb::value_error("MeanPowerError instance: example_weights must be distinct from predictions and labels.");
    ThorPython::RegressionLossDType::validateExampleWeights("MeanPowerError instance", example_weights.value());
    const std::vector<uint64_t>& dims = example_weights.value().getDimensions();
    if (dims != std::vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        string error_message = "MeanPowerError instance: example_weights dimensions must be [1] for per-example weights or match predictions. "
                               "example_weights tensor is " +
                               example_weights.value().getDescriptorString() + "; predictions tensor is " +
                               predictions.getDescriptorString() + ".";
        throw nb::value_error(error_message.c_str());
    }
    builder.exampleWeights(example_weights.value());
}

}  // namespace

void bind_mean_power_error(nb::module_ &losses) {
    auto mean_power_error = nb::class_<MeanPowerError, Loss>(losses, "MeanPowerError");
    mean_power_error.attr("__module__") = "thor.losses";

    mean_power_error.def(
        "__init__",
        [](MeanPowerError *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float exponent,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight,
           std::optional<Tensor> example_weights) {
            if (!std::isfinite(exponent) || exponent < 1.0f) {
                throw nb::value_error("MeanPowerError instance: exponent must be finite and greater than or equal to 1.0.");
            }

            const string loss_name = "MeanPowerError instance";
            ThorPython::RegressionLossDType::validatePredictions(loss_name, predictions);
            ThorPython::RegressionLossDType::validateLabels(loss_name, labels);
            const DataType effectiveLossDataType =
                ThorPython::RegressionLossDType::effectiveLossDType(loss_name, predictions.getDataType(), loss_data_type);
            validateReportedLossShape(reported_loss_shape, loss_name);

            MeanPowerError::Builder builder;

            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .exponent(exponent)
                .lossDataType(effectiveLossDataType);
            builder.lossWeight(loss_weight.value_or(1.0f));
            maybeSetExampleWeights(builder, predictions, labels, example_weights);

            if (predictions.getDimensions() != labels.getDimensions()) {
                string error_message = "MeanPowerError instance: predictions and labels dimensions must match. predictions tensor is " +
                                       predictions.getDescriptorString() + "; labels tensor is " + labels.getDescriptorString() + ".";
                throw nb::value_error(error_message.c_str());
            }

            setReportedLossShape(builder, reported_loss_shape);

            MeanPowerError built = builder.build();

            new (self) MeanPowerError(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "exponent"_a = 1.5f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a MeanPowerError loss.)nbdoc");

    mean_power_error.def_prop_ro("exponent", &MeanPowerError::getExponent);

    mean_power_error.attr("__doc__") = R"nbdoc(
MeanPowerError loss.

MeanPowerError computes the mean absolute residual raised to a configurable
power:

    loss = mean(abs(prediction - label) ** exponent)

The exponent must be finite and greater than or equal to 1.0. The most useful
range for ordinary regression losses is usually 1.0 <= exponent <= 2.0:

    MeanPowerError(exponent=1.0) is MeanAbsoluteError / MAE.
    MeanPowerError(exponent=2.0) is MeanSquaredError / MSE.
    1.0 < exponent < 2.0 gives behavior between MAE and MSE.

Values greater than 2.0 are allowed for cases that intentionally give very large
errors more leverage than MSE, but they are more outlier-sensitive.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
exponent : float, default 1.5
    Power applied to abs(prediction - label). Must be >= 1.0.
loss_data_type : thor.DataType | None, default fp16 for fp16 predictions, otherwise fp32
reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
    Controls the reported loss tensor:

    * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
    * ``batch`` averages over the batch after summing all non-batch values.
    * ``per_example`` sums all non-batch values independently for each example.
    * ``per_output`` averages over the batch and preserves every non-batch dimension.
    * ``raw`` reports the unreduced pointwise loss.

)nbdoc";
}
