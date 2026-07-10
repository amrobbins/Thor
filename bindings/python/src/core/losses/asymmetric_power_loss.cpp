#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/AsymmetricPowerLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string& loss_name) {
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::CLASSWISE &&
        reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(AsymmetricPowerLoss::Builder& builder, LossShape reported_loss_shape) {
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

void maybeSetExampleWeights(AsymmetricPowerLoss::Builder& builder,
                            Tensor predictions,
                            Tensor labels,
                            std::optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions || example_weights.value() == labels)
        throw nb::value_error("AsymmetricPowerLoss instance: example_weights must be distinct from predictions and labels.");
    DataType dtype = example_weights.value().getDataType();
    if (dtype != DataType::FP16 && dtype != DataType::FP32)
        throw nb::value_error("AsymmetricPowerLoss instance: example_weights must be fp16 or fp32.");
    const std::vector<uint64_t>& dims = example_weights.value().getDimensions();
    if (dims != std::vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        string error_message =
            "AsymmetricPowerLoss instance: example_weights dimensions must be [1] for per-example weights or match predictions. "
            "example_weights tensor is " +
            example_weights.value().getDescriptorString() + "; predictions tensor is " + predictions.getDescriptorString() + ".";
        throw nb::value_error(error_message.c_str());
    }
    builder.exampleWeights(example_weights.value());
}
}  // namespace

void bind_asymmetric_power_loss(nb::module_& losses) {
    auto asymmetric_power_loss = nb::class_<AsymmetricPowerLoss, Loss>(losses, "AsymmetricPowerLoss");
    asymmetric_power_loss.attr("__module__") = "thor.losses";

    asymmetric_power_loss.def(
        "__init__",
        [](AsymmetricPowerLoss* self,
           Network& network,
           Tensor predictions,
           Tensor labels,
           float level,
           float exponent,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight,
           std::optional<Tensor> example_weights) {
            const string loss_name = "AsymmetricPowerLoss instance";
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
            if (!std::isfinite(level) || level <= 0.0f || level >= 1.0f) {
                throw nb::value_error(
                    "AsymmetricPowerLoss instance: level must be finite, greater than zero, and less than one.");
            }
            if (!std::isfinite(exponent) || exponent < 1.0f) {
                throw nb::value_error(
                    "AsymmetricPowerLoss instance: exponent must be finite and greater than or equal to 1.0.");
            }
            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
                string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
                throw nb::value_error(error_message.c_str());
            }
            validateReportedLossShape(reported_loss_shape, loss_name);

            AsymmetricPowerLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .level(level)
                .exponent(exponent)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            maybeSetExampleWeights(builder, predictions, labels, example_weights);
            setReportedLossShape(builder, reported_loss_shape);
            AsymmetricPowerLoss built = builder.build();

            new (self) AsymmetricPowerLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "level"_a = 0.5f,
        "exponent"_a = 1.5f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct an asymmetric absolute-power regression loss.)nbdoc");

    asymmetric_power_loss.def_prop_ro("level", &AsymmetricPowerLoss::getLevel);
    asymmetric_power_loss.def_prop_ro("exponent", &AsymmetricPowerLoss::getExponent);

    asymmetric_power_loss.attr("__doc__") = R"nbdoc(
Asymmetric absolute-power regression loss.

For level tau, exponent p, and error y_true - y_pred, Thor uses:

    2 * tau       * abs(error)**p    if error > 0
    2 * (1 - tau) * abs(error)**p    otherwise

The normalization gives these exact relationships:

    AsymmetricPowerLoss(level=0.5, exponent=p) == MeanPowerError(exponent=p)
    AsymmetricPowerLoss(level=tau, exponent=2) == ExpectileLoss(expectile=tau)

At exponent=1 this is twice the conventional pinball loss. That constant does not
change the fitted optimum, and preserves exact equality with MeanPowerError at the
central level. Exponents between 1 and 2 provide asymmetric bounds with robustness
between quantile loss and expectile loss.
)nbdoc";
}
