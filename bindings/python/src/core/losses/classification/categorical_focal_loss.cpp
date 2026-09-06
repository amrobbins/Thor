#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/CategoricalFocalLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

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

void setReportedLossShape(CategoricalFocalLoss::Builder &builder, LossShape reported_loss_shape) {
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

void validateDType(const string& loss_name, const char* what, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        string error_message = loss_name + ": " + what + " must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
}

void validateLossDataType(const string& loss_name, DataType dtype) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
}

void validateFocalScalars(const string& loss_name, float gamma, float alpha) {
    if (gamma < 0.0f) {
        string error_message = loss_name + ": gamma must be non-negative";
        throw nb::value_error(error_message.c_str());
    }
    if (alpha < 0.0f) {
        string error_message = loss_name + ": alpha must be non-negative";
        throw nb::value_error(error_message.c_str());
    }
}

void validateDenseArguments(const string &loss_name,
                            Tensor predictions,
                            Tensor labels,
                            float gamma,
                            float alpha,
                            optional<DataType> loss_data_type,
                            LossShape reported_loss_shape) {
    if (predictions.getDimensions().size() != 1) {
        string error_message = loss_name + ": predictions must be a 1 dimensional logits tensor but predictions is " +
                               predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (labels.getDimensions() != predictions.getDimensions()) {
        string error_message = loss_name + ": labels dimensions " + labels.getDescriptorString() +
                               " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    validateDType(loss_name, "predictions", predictions.getDataType());
    validateDType(loss_name, "labels", labels.getDataType());
    validateFocalScalars(loss_name, gamma, alpha);
    validateLossDataType(loss_name, loss_data_type.value_or(predictions.getDataType()));
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void validateRaggedArguments(const string& loss_name,
                             const RaggedTensor& predictions,
                             const RaggedTensor& labels,
                             float gamma,
                             float alpha,
                             optional<DataType> loss_data_type,
                             LossShape reported_loss_shape) {
    if (predictions.getTrailingDimensions().size() != 1 || predictions.getTrailingDimensions().back() <= 1)
        throw nb::value_error("CategoricalFocalLoss instance: ragged predictions must have exactly one trailing class dimension greater than one.");
    if (!predictions.sharesPartitionWith(labels))
        throw nb::value_error("CategoricalFocalLoss instance: ragged predictions and labels must use the exact same row partition.");
    if (predictions.getBatchSize() != labels.getBatchSize() ||
        predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
        predictions.getTrailingDimensions() != labels.getTrailingDimensions())
        throw nb::value_error("CategoricalFocalLoss instance: ragged predictions and labels must have identical value geometry.");
    if (reported_loss_shape == LossShape::PER_OUTPUT)
        throw nb::value_error("CategoricalFocalLoss instance: per_output reporting is undefined for ragged predictions.");
    validateDType(loss_name, "predictions", predictions.getValuesDataType());
    validateDType(loss_name, "labels", labels.getValuesDataType());
    validateFocalScalars(loss_name, gamma, alpha);
    validateLossDataType(loss_name, loss_data_type.value_or(predictions.getValuesDataType()));
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_categorical_focal_loss(nb::module_ &losses) {
    auto categorical_focal_loss = nb::class_<CategoricalFocalLoss, Loss>(losses, "CategoricalFocalLoss");
    categorical_focal_loss.attr("__module__") = "thor.losses.classification";

    categorical_focal_loss.def(
        "__init__",
        [](CategoricalFocalLoss *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           float gamma,
           float alpha,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "CategoricalFocalLoss instance";
            CategoricalFocalLoss::Builder builder;
            builder.network(network);
            DataType effectiveLossDataType;

            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                validateDenseArguments(loss_name, predictions, labels, gamma, alpha, loss_data_type, reported_loss_shape);
                effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
                builder.predictions(predictions).labels(labels);
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                validateRaggedArguments(loss_name, predictions, labels, gamma, alpha, loss_data_type, reported_loss_shape);
                effectiveLossDataType = loss_data_type.value_or(predictions.getValuesDataType());
                builder.predictions(predictions).labels(labels);
            } else {
                throw nb::type_error("CategoricalFocalLoss predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

            builder.focusingParameter(gamma)
                .alpha(alpha)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            CategoricalFocalLoss built = builder.build();
            new (self) CategoricalFocalLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "gamma"_a = 2.0f,
        "alpha"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged categorical focal loss from logits and dense targets.)nbdoc");

    categorical_focal_loss.def_prop_ro("gamma", &CategoricalFocalLoss::getGamma);
    categorical_focal_loss.def_prop_ro("alpha", &CategoricalFocalLoss::getAlpha);
    categorical_focal_loss.def("get_predictions", [](const CategoricalFocalLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    categorical_focal_loss.def("get_labels", [](const CategoricalFocalLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    categorical_focal_loss.def("get_raw_loss", [](const CategoricalFocalLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    categorical_focal_loss.def("get_loss", [](const CategoricalFocalLoss& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    categorical_focal_loss.def_prop_ro("is_ragged", &CategoricalFocalLoss::isRagged);

    categorical_focal_loss.attr("__doc__") = R"nbdoc(
Categorical focal loss from logits and dense target distributions.

``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or
rank-1 ``thor.RaggedTensor`` objects. Ragged values must have exactly one
trailing class dimension ``[C]`` and share the exact same row partition.

The raw loss is:

    -alpha * target * (1 - softmax(logits)) ** gamma * log_softmax(logits)

For ragged inputs, ``raw`` preserves the partition, ``per_example`` sums over
all active tokens/classes in each logical row, and ``batch`` averages those row
sums over valid logical examples. ``per_output`` is undefined for ragged input.
For sparse class-index targets, use a sparse focal wrapper later rather than
this dense-target loss.
)nbdoc";
}
