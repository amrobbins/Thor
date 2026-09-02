#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <cmath>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/MeanPowerError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

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
           nb::object predictionsObject,
           nb::object labelsObject,
           float exponent,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight,
           std::optional<Tensor> example_weights) {
            if (!std::isfinite(exponent) || exponent < 1.0f) {
                throw nb::value_error("MeanPowerError instance: exponent must be finite and greater than or equal to 1.0.");
            }

            const string loss_name = "MeanPowerError instance";
            validateReportedLossShape(reported_loss_shape, loss_name);

            MeanPowerError::Builder builder;
            builder.network(network).exponent(exponent);

            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                ThorPython::RegressionLossDType::validatePredictions(loss_name, predictions);
                ThorPython::RegressionLossDType::validateLabels(loss_name, labels);
                const DataType effectiveLossDataType =
                    ThorPython::RegressionLossDType::effectiveLossDType(loss_name, predictions.getDataType(), loss_data_type);
                builder.predictions(predictions).labels(labels).lossDataType(effectiveLossDataType);
                maybeSetExampleWeights(builder, predictions, labels, example_weights);

                if (predictions.getDimensions() != labels.getDimensions()) {
                    string error_message = "MeanPowerError instance: predictions and labels dimensions must match. predictions tensor is " +
                                           predictions.getDescriptorString() + "; labels tensor is " + labels.getDescriptorString() + ".";
                    throw nb::value_error(error_message.c_str());
                }
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                ThorPython::RegressionLossDType::validatePredictions(loss_name, predictions.getValues());
                ThorPython::RegressionLossDType::validateLabels(loss_name, labels.getValues());
                const DataType effectiveLossDataType = ThorPython::RegressionLossDType::effectiveLossDType(
                    loss_name, predictions.getValuesDataType(), loss_data_type);
                if (example_weights.has_value()) {
                    if (example_weights.value() == predictions.getValues() || example_weights.value() == labels.getValues())
                        throw nb::value_error("MeanPowerError instance: ragged example_weights must be distinct from predictions and labels values.");
                    ThorPython::RegressionLossDType::validateExampleWeights("MeanPowerError instance", example_weights.value());
                    if (example_weights->getDimensions() != std::vector<uint64_t>{1})
                        throw nb::value_error("MeanPowerError instance: ragged example_weights dimensions must be [1] for one scalar weight per logical row.");
                }
                if (reported_loss_shape == LossShape::PER_OUTPUT) {
                    throw nb::value_error("MeanPowerError instance: per_output reporting is undefined for ragged predictions.");
                }
                if (predictions.getOffsets() != labels.getOffsets()) {
                    throw nb::value_error("MeanPowerError instance: ragged predictions and labels must use the exact same row partition tensor.");
                }
                if (predictions.getBatchSize() != labels.getBatchSize() ||
                    predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
                    predictions.getTrailingDimensions() != labels.getTrailingDimensions()) {
                    throw nb::value_error("MeanPowerError instance: ragged predictions and labels must have identical value geometry.");
                }
                builder.predictions(predictions).labels(labels).lossDataType(effectiveLossDataType);
                if (example_weights.has_value())
                    builder.exampleWeights(example_weights.value());
            } else {
                throw nb::type_error("MeanPowerError predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

            builder.lossWeight(loss_weight.value_or(1.0f));
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
        R"nbdoc(Construct a dense or rank-1 ragged MeanPowerError loss.)nbdoc");

    // Shadow the dense Loss accessors on MeanPowerError so Python receives the logical
    // RaggedTensor for ragged predictions/labels and RAW reported loss.
    mean_power_error.def("get_predictions", [](const MeanPowerError& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    mean_power_error.def("get_labels", [](const MeanPowerError& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    mean_power_error.def("get_raw_loss", [](const MeanPowerError& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    mean_power_error.def("get_loss", [](const MeanPowerError& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    mean_power_error.def_prop_ro("is_ragged", &MeanPowerError::isRagged);
    mean_power_error.def_prop_ro("exponent", &MeanPowerError::getExponent);

    mean_power_error.attr("__doc__") = R"nbdoc(
MeanPowerError loss.

``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or
rank-1 ``thor.RaggedTensor`` objects. Ragged inputs must share the exact same
row-partition tensor. Ragged loss reporting supports ``none``, ``raw``,
``per_example``, and ``batch``; ``per_output`` is intentionally undefined.

The exponent must be finite and greater than or equal to 1.0. For ragged
inputs, ``raw`` preserves the partition, ``per_example`` returns one dense
scalar per logical row, and ``batch`` averages those row sums over valid
logical examples rather than active tokens. Dense ``[1]`` example weights are
broadcast to active tokens in each row and scale both loss and prediction
gradient.
)nbdoc";
}
