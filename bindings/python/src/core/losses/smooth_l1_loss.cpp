#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/SmoothL1Loss.h"
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
void validateReportedLossShape(LossShape shape, const string& name) {
    if (shape != LossShape::NONE && shape != LossShape::BATCH && shape != LossShape::PER_OUTPUT &&
        shape != LossShape::PER_EXAMPLE && shape != LossShape::RAW)
        throw nb::value_error(("Invalid reported_loss_shape passed to " + name + ".").c_str());
}

void setReportedLossShape(SmoothL1Loss::Builder& builder, LossShape shape) {
    if (shape == LossShape::NONE) builder.reportsNoLoss();
    else if (shape == LossShape::BATCH) builder.reportsBatchLoss();
    else if (shape == LossShape::PER_OUTPUT) builder.reportsPerOutputLoss();
    else if (shape == LossShape::PER_EXAMPLE) builder.reportsPerExampleLoss();
    else builder.reportsRawLoss();
}

void validatePredictionDType(DataType dtype, const string& name) {
    if (dtype != DataType::FP16 && dtype != DataType::FP32)
        throw nb::value_error((name + ": predictions must use fp16 or fp32").c_str());
}

void validateLabelDType(DataType dtype, const string& name) {
    if (dtype != DataType::BOOLEAN && dtype != DataType::UINT8 && dtype != DataType::UINT16 && dtype != DataType::UINT32 &&
        dtype != DataType::FP16 && dtype != DataType::FP32)
        throw nb::value_error((name + ": unsupported labels dtype").c_str());
}
}  // namespace

void bind_smooth_l1_loss(nb::module_ &losses) {
    auto loss = nb::class_<SmoothL1Loss, Loss>(losses, "SmoothL1Loss");
    loss.attr("__module__") = "thor.losses";

    loss.def(
        "__init__",
        [](SmoothL1Loss *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           float beta,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "SmoothL1Loss instance";
            if (beta <= 0.0f) throw nb::value_error("SmoothL1Loss instance: beta must be greater than zero");
            validateReportedLossShape(reported_loss_shape, loss_name);
            SmoothL1Loss::Builder builder;
            builder.network(network).beta(beta);

            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                if (predictions.getDimensions().empty())
                    throw nb::value_error((loss_name + ": predictions must have at least one per-example dimension").c_str());
                if (predictions.getDimensions() != labels.getDimensions()) {
                    string error_message = loss_name + ": labels dimensions " + labels.getDescriptorString() +
                                           " must match predictions dimensions " + predictions.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                validatePredictionDType(predictions.getDataType(), loss_name);
                validateLabelDType(labels.getDataType(), loss_name);
                DataType effective = loss_data_type.value_or(predictions.getDataType());
                if (effective != DataType::FP16 && effective != DataType::FP32)
                    throw nb::value_error((loss_name + ": loss_data_type must be fp16 or fp32").c_str());
                builder.predictions(predictions).labels(labels).lossDataType(effective);
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                validatePredictionDType(predictions.getValuesDataType(), loss_name);
                validateLabelDType(labels.getValuesDataType(), loss_name);
                if (reported_loss_shape == LossShape::PER_OUTPUT)
                    throw nb::value_error((loss_name + ": per_output reporting is undefined for ragged predictions").c_str());
                if (predictions.getOffsets() != labels.getOffsets())
                    throw nb::value_error((loss_name + ": ragged predictions and labels must use the exact same row partition tensor").c_str());
                if (predictions.getBatchSize() != labels.getBatchSize() ||
                    predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
                    predictions.getTrailingDimensions() != labels.getTrailingDimensions())
                    throw nb::value_error((loss_name + ": ragged predictions and labels must have identical value geometry").c_str());
                DataType effective = loss_data_type.value_or(predictions.getValuesDataType());
                if (effective != DataType::FP16 && effective != DataType::FP32)
                    throw nb::value_error((loss_name + ": loss_data_type must be fp16 or fp32").c_str());
                builder.predictions(predictions).labels(labels).lossDataType(effective);
            } else {
                throw nb::type_error("SmoothL1Loss predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

            builder.lossWeight(loss_weight.value_or(1.0f));
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
        R"nbdoc(Construct a dense or rank-1 ragged SmoothL1Loss loss.)nbdoc");

    loss.def("get_predictions", [](const SmoothL1Loss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    loss.def("get_labels", [](const SmoothL1Loss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    loss.def("get_raw_loss", [](const SmoothL1Loss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    loss.def("get_loss", [](const SmoothL1Loss& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    loss.def_prop_ro("is_ragged", &SmoothL1Loss::isRagged);
    loss.def_prop_ro("beta", &SmoothL1Loss::getBeta);

    loss.attr("__doc__") = R"nbdoc(
SmoothL1Loss loss.

``predictions`` and ``labels`` may both be dense ``thor.Tensor`` objects or rank-1
``thor.RaggedTensor`` objects. Ragged inputs must share the exact same row
partition. Ragged reporting supports ``none``, ``raw``, ``per_example``, and
``batch``; ``per_output`` is intentionally undefined. ``batch`` averages
per-row active-token sums over valid logical examples rather than active tokens.
SmoothL1Loss uses the PyTorch-style beta parameterization.
)nbdoc";
}
