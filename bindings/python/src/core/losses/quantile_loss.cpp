#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
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

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string &loss_name) {
    if (reported_loss_shape != LossShape::NONE && reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::PER_OUTPUT &&
        reported_loss_shape != LossShape::PER_EXAMPLE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(QuantileLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::NONE) {
        builder.reportsNoLoss();
    } else if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::PER_OUTPUT) {
        builder.reportsPerOutputLoss();
    } else if (reported_loss_shape == LossShape::PER_EXAMPLE) {
        builder.reportsPerExampleLoss();
    } else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void maybeSetExampleWeights(QuantileLoss::Builder &builder,
                            Tensor predictions,
                            Tensor labels,
                            std::optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions || example_weights.value() == labels)
        throw nb::value_error("QuantileLoss instance: example_weights must be distinct from predictions and labels.");
    ThorPython::RegressionLossDType::validateExampleWeights("QuantileLoss instance", example_weights.value());
    const std::vector<uint64_t>& dims = example_weights.value().getDimensions();
    if (dims != std::vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        string error_message = "QuantileLoss instance: example_weights dimensions must be [1] for per-example weights or match predictions. "
                               "example_weights tensor is " +
                               example_weights.value().getDescriptorString() + "; predictions tensor is " +
                               predictions.getDescriptorString() + ".";
        throw nb::value_error(error_message.c_str());
    }
    builder.exampleWeights(example_weights.value());
}

void maybeSetRaggedExampleWeights(QuantileLoss::Builder& builder,
                                  RaggedTensor predictions,
                                  RaggedTensor labels,
                                  std::optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions.getValues() || example_weights.value() == labels.getValues())
        throw nb::value_error("QuantileLoss instance: ragged example_weights must be distinct from predictions and labels values.");
    ThorPython::RegressionLossDType::validateExampleWeights("QuantileLoss instance", example_weights.value());
    if (example_weights->getDimensions() != std::vector<uint64_t>{1})
        throw nb::value_error("QuantileLoss instance: ragged example_weights dimensions must be [1] for one scalar weight per logical row.");
    builder.exampleWeights(example_weights.value());
}
}  // namespace

void bind_quantile_loss(nb::module_ &losses) {
    auto quantile_loss = nb::class_<QuantileLoss, Loss>(losses, "QuantileLoss");
    quantile_loss.attr("__module__") = "thor.losses";

    quantile_loss.def(
        "__init__",
        [](QuantileLoss *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           float quantile,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight,
           std::optional<Tensor> example_weights) {
            const string loss_name = "QuantileLoss instance";
            if (quantile <= 0.0f || quantile >= 1.0f) {
                string error_message = loss_name + ": quantile must be greater than zero and less than one";
                throw nb::value_error(error_message.c_str());
            }
            validateReportedLossShape(reported_loss_shape, loss_name);

            QuantileLoss::Builder builder;
            builder.network(network).quantile(quantile);

            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                ThorPython::RegressionLossDType::validatePredictions(loss_name, predictions);
                ThorPython::RegressionLossDType::validateLabels(loss_name, labels);
                if (predictions.getDimensions().empty()) {
                    string error_message = loss_name + ": predictions must have at least one per-example dimension but predictions is " +
                                           predictions.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                if (labels.getDimensions() != predictions.getDimensions()) {
                    string error_message = loss_name + ": labels dimensions " + labels.getDescriptorString() +
                                           " must match predictions dimensions " + predictions.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                const DataType effectiveLossDataType =
                    ThorPython::RegressionLossDType::effectiveLossDType(loss_name, predictions.getDataType(), loss_data_type);
                builder.predictions(predictions).labels(labels).lossDataType(effectiveLossDataType);
                maybeSetExampleWeights(builder, predictions, labels, example_weights);
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                ThorPython::RegressionLossDType::validatePredictions(loss_name, predictions.getValues());
                ThorPython::RegressionLossDType::validateLabels(loss_name, labels.getValues());
                if (reported_loss_shape == LossShape::PER_OUTPUT)
                    throw nb::value_error("QuantileLoss instance: per_output reporting is undefined for ragged predictions.");
                if (predictions.getOffsets() != labels.getOffsets())
                    throw nb::value_error("QuantileLoss instance: ragged predictions and labels must use the exact same row partition tensor.");
                if (predictions.getBatchSize() != labels.getBatchSize() ||
                    predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
                    predictions.getTrailingDimensions() != labels.getTrailingDimensions())
                    throw nb::value_error("QuantileLoss instance: ragged predictions and labels must have identical value geometry.");
                const DataType effectiveLossDataType = ThorPython::RegressionLossDType::effectiveLossDType(
                    loss_name, predictions.getValuesDataType(), loss_data_type);
                builder.predictions(predictions).labels(labels).lossDataType(effectiveLossDataType);
                maybeSetRaggedExampleWeights(builder, predictions, labels, example_weights);
            } else {
                throw nb::type_error("QuantileLoss predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

            builder.lossWeight(loss_weight.value_or(1.0f));
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
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged Quantile / pinball loss.)nbdoc");

    quantile_loss.def("get_predictions", [](const QuantileLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    quantile_loss.def("get_labels", [](const QuantileLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    quantile_loss.def("get_raw_loss", [](const QuantileLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    quantile_loss.def("get_loss", [](const QuantileLoss& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    quantile_loss.def_prop_ro("is_ragged", &QuantileLoss::isRagged);

    quantile_loss.def_prop_ro("quantile", &QuantileLoss::getQuantile);

    losses.attr("PinballLoss") = quantile_loss;

    quantile_loss.attr("__doc__") = R"nbdoc(
Quantile / pinball loss.

For quantile q and error y_true - y_pred:

    q * error          if error > 0
    (q - 1) * error    otherwise

The subgradient at zero error is defined as 0.


Predictions and labels may both be dense tensors or rank-1 ragged tensors. Ragged
inputs must share the exact row partition. Ragged reporting supports none, raw,
per-example, and batch; per-output is undefined. Dense [1] example weights are
broadcast over each logical row's active tokens.
)nbdoc";
}
