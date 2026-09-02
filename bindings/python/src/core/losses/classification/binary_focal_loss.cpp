#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/BinaryFocalLoss.h"
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
    if (reported_loss_shape != LossShape::NONE && reported_loss_shape != LossShape::BATCH &&
        reported_loss_shape != LossShape::PER_EXAMPLE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(BinaryFocalLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::NONE) {
        builder.reportsNoLoss();
    } else if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::PER_EXAMPLE) {
        builder.reportsPerExampleLoss();
    } else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

bool isBinaryLabelDType(DataType dtype) {
    return dtype == DataType::BOOLEAN || dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           dtype == DataType::FP16 || dtype == DataType::FP32;
}

void validateBinaryFocalLossArguments(const string &loss_name,
                                      Tensor predictions,
                                      Tensor labels,
                                      float gamma,
                                      float alpha,
                                      optional<DataType> loss_data_type,
                                      LossShape reported_loss_shape) {
    const vector<uint64_t> prediction_dimensions = predictions.getDimensions();
    const vector<uint64_t> label_dimensions = labels.getDimensions();
    if (prediction_dimensions.empty() || prediction_dimensions[0] == 0) {
        throw nb::value_error((loss_name + ": predictions must have at least one nonempty per-example dimension").c_str());
    }
    if (label_dimensions.empty() || label_dimensions[0] == 0) {
        throw nb::value_error((loss_name + ": labels must have at least one nonempty per-example dimension").c_str());
    }
    if (prediction_dimensions != label_dimensions) {
        string error_message = loss_name + ": predictions and labels dimensions must match. predictions tensor is " +
                               predictions.getDescriptorString() + " and labels tensor is " + labels.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (predictions.getDataType() != DataType::FP16 && predictions.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": predictions must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (!isBinaryLabelDType(labels.getDataType())) {
        string error_message = loss_name + ": labels must use bool, uint8, uint16, uint32, fp16, or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (gamma < 0.0f) {
        string error_message = loss_name + ": gamma must be non-negative";
        throw nb::value_error(error_message.c_str());
    }
    if (alpha < 0.0f || alpha > 1.0f) {
        string error_message = loss_name + ": alpha must be in the range [0, 1]";
        throw nb::value_error(error_message.c_str());
    }
    DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
    if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_binary_focal_loss(nb::module_ &losses) {
    auto binary_focal_loss = nb::class_<BinaryFocalLoss, Loss>(losses, "BinaryFocalLoss");
    binary_focal_loss.attr("__module__") = "thor.losses.classification";

    binary_focal_loss.def(
        "__init__",
        [](BinaryFocalLoss *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           float gamma,
           float alpha,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "BinaryFocalLoss instance";
            if (gamma < 0.0f) throw nb::value_error((loss_name + ": gamma must be non-negative").c_str());
            if (alpha < 0.0f || alpha > 1.0f) throw nb::value_error((loss_name + ": alpha must be in the range [0, 1]").c_str());
            validateReportedLossShape(reported_loss_shape, loss_name);

            BinaryFocalLoss::Builder builder;
            builder.network(network).focusingParameter(gamma).alpha(alpha).lossWeight(loss_weight.value_or(1.0f));
            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                validateBinaryFocalLossArguments(loss_name, predictions, labels, gamma, alpha, loss_data_type, reported_loss_shape);
                builder.predictions(predictions).labels(labels);
                if (loss_data_type.has_value()) builder.lossDataType(loss_data_type.value());
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                if (predictions.getValuesDataType() != DataType::FP16 && predictions.getValuesDataType() != DataType::FP32)
                    throw nb::value_error((loss_name + ": predictions must use fp16 or fp32 dtype").c_str());
                if (!isBinaryLabelDType(labels.getValuesDataType()))
                    throw nb::value_error((loss_name + ": labels must use bool, uint8, uint16, uint32, fp16, or fp32 dtype").c_str());
                if (predictions.getOffsets() != labels.getOffsets())
                    throw nb::value_error((loss_name + ": ragged predictions and labels must use the exact same row partition tensor.").c_str());
                if (predictions.getBatchSize() != labels.getBatchSize() ||
                    predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
                    predictions.getTrailingDimensions() != labels.getTrailingDimensions())
                    throw nb::value_error((loss_name + ": ragged predictions and labels must have identical value geometry.").c_str());
                DataType effectiveLossDataType = loss_data_type.value_or(predictions.getValuesDataType());
                if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32)
                    throw nb::value_error((loss_name + ": loss_data_type must be fp16 or fp32").c_str());
                builder.predictions(predictions).labels(labels).lossDataType(effectiveLossDataType);
            } else {
                throw nb::type_error("BinaryFocalLoss predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

            setReportedLossShape(builder, reported_loss_shape);
            BinaryFocalLoss built = builder.build();
            new (self) BinaryFocalLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "gamma"_a = 2.0f,
        "alpha"_a = 0.25f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged binary focal loss.)nbdoc");

    binary_focal_loss.def("get_predictions", [](const BinaryFocalLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    binary_focal_loss.def("get_labels", [](const BinaryFocalLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    binary_focal_loss.def("get_raw_loss", [](const BinaryFocalLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    binary_focal_loss.def("get_loss", [](const BinaryFocalLoss& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    binary_focal_loss.def_prop_ro("is_ragged", &BinaryFocalLoss::isRagged);

    binary_focal_loss.def_prop_ro("gamma", &BinaryFocalLoss::getGamma);
    binary_focal_loss.def_prop_ro("alpha", &BinaryFocalLoss::getAlpha);

    binary_focal_loss.attr("__doc__") = R"nbdoc(
Binary focal loss from logits.

The predictions tensor contains a nonempty per-example tensor of independent unnormalized binary
logits, and the labels tensor contains matching binary targets. A shape of [1] is the standard
binary-classification case; wider or higher-rank tensors support multi-output, multilabel, and dense
prediction objectives. The raw loss is applied pointwise:

    alpha_t * (1 - p_t) ** gamma * BCEWithLogits(logit, target)

where alpha_t is alpha for positive targets and 1 - alpha for negative targets.

Predictions and labels may both be dense tensors or rank-1 ragged tensors. Ragged
inputs must share the exact row partition; raw loss preserves that partition and
per-example/batch reporting uses active tokens only.
)nbdoc";
}
