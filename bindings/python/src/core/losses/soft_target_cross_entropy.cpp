#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/SoftTargetCrossEntropy.h"
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

void setReportedLossShape(SoftTargetCrossEntropy::Builder &builder, LossShape reported_loss_shape) {
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

void validateDenseArguments(const string &loss_name,
                            Tensor predictions,
                            Tensor labels,
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
    validateLossDataType(loss_name, loss_data_type.value_or(predictions.getDataType()));
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void validateRaggedArguments(const string& loss_name,
                             const RaggedTensor& predictions,
                             const RaggedTensor& labels,
                             optional<DataType> loss_data_type,
                             LossShape reported_loss_shape) {
    if (predictions.getTrailingDimensions().size() != 1 || predictions.getTrailingDimensions().back() <= 1)
        throw nb::value_error("SoftTargetCrossEntropy instance: ragged predictions must have exactly one trailing class dimension greater than one.");
    if (!predictions.sharesPartitionWith(labels))
        throw nb::value_error("SoftTargetCrossEntropy instance: ragged predictions and labels must use the exact same row partition.");
    if (predictions.getBatchSize() != labels.getBatchSize() ||
        predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
        predictions.getTrailingDimensions() != labels.getTrailingDimensions())
        throw nb::value_error("SoftTargetCrossEntropy instance: ragged predictions and labels must have identical value geometry.");
    if (reported_loss_shape == LossShape::PER_OUTPUT)
        throw nb::value_error("SoftTargetCrossEntropy instance: per_output reporting is undefined for ragged predictions.");
    validateDType(loss_name, "predictions", predictions.getValuesDataType());
    validateDType(loss_name, "labels", labels.getValuesDataType());
    validateLossDataType(loss_name, loss_data_type.value_or(predictions.getValuesDataType()));
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_soft_target_cross_entropy(nb::module_ &losses) {
    auto soft_target_cross_entropy = nb::class_<SoftTargetCrossEntropy, Loss>(losses, "SoftTargetCrossEntropy");
    soft_target_cross_entropy.attr("__module__") = "thor.losses";

    soft_target_cross_entropy.def(
        "__init__",
        [](SoftTargetCrossEntropy *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "SoftTargetCrossEntropy instance";
            SoftTargetCrossEntropy::Builder builder;
            builder.network(network);
            DataType effectiveLossDataType;

            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                validateDenseArguments(loss_name, predictions, labels, loss_data_type, reported_loss_shape);
                effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
                builder.predictions(predictions).labels(labels);
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                validateRaggedArguments(loss_name, predictions, labels, loss_data_type, reported_loss_shape);
                effectiveLossDataType = loss_data_type.value_or(predictions.getValuesDataType());
                builder.predictions(predictions).labels(labels);
            } else {
                throw nb::type_error("SoftTargetCrossEntropy predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

            builder.lossDataType(effectiveLossDataType).lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            SoftTargetCrossEntropy built = builder.build();
            new (self) SoftTargetCrossEntropy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged soft-target cross entropy loss.)nbdoc");

    soft_target_cross_entropy.def("get_predictions", [](const SoftTargetCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    soft_target_cross_entropy.def("get_labels", [](const SoftTargetCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    soft_target_cross_entropy.def("get_raw_loss", [](const SoftTargetCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    soft_target_cross_entropy.def("get_loss", [](const SoftTargetCrossEntropy& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    soft_target_cross_entropy.def_prop_ro("is_ragged", &SoftTargetCrossEntropy::isRagged);

    soft_target_cross_entropy.attr("__doc__") = R"nbdoc(
Soft-target categorical cross entropy from logits.

``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or
rank-1 ``thor.RaggedTensor`` objects. Ragged values must have exactly one
trailing class dimension ``[C]`` and predictions/labels must share the exact
same row partition.

The raw loss is:

    -target * log_softmax(logits)

For ragged inputs, ``raw`` preserves the partition, ``per_example`` sums over
all active tokens/classes in each logical row, and ``batch`` averages those row
sums over valid logical examples. ``per_output`` is undefined for ragged input.
The gradient assumes targets are normalized distributions.
)nbdoc";
}
