#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
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
void validateReportedLossShape(LossShape reported_loss_shape, const string& loss_name) {
    if (reported_loss_shape != LossShape::NONE && reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::PER_EXAMPLE &&
        reported_loss_shape != LossShape::PER_OUTPUT && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(BinaryCrossEntropy::Builder& builder, LossShape reported_loss_shape) {
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
}  // namespace

void bind_binary_cross_entropy(nb::module_ &losses) {
    auto binary_cross_entropy = nb::class_<BinaryCrossEntropy, Loss>(losses, "BinaryCrossEntropy");
    binary_cross_entropy.attr("__module__") = "thor.losses";

    binary_cross_entropy.def(
        "__init__",
        [](BinaryCrossEntropy *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           DataType loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            validateReportedLossShape(reported_loss_shape, "BinaryCrossEntropy instance");
            if (loss_data_type != DataType::FP16 && loss_data_type != DataType::FP32)
                throw nb::value_error("BinaryCrossEntropy instance: loss_data_type must be fp16 or fp32");

            BinaryCrossEntropy::Builder builder;
            builder.network(network).lossDataType(loss_data_type).lossWeight(loss_weight.value_or(1.0f));
            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                if (predictions.getDimensions().empty()) {
                    string error_message =
                        "BinaryCrossEntropy instance: predictions must have at least one per-example dimension but predictions is " +
                        predictions.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                if (labels.getDimensions().empty()) {
                    string error_message = "BinaryCrossEntropy instance: labels must have at least one per-example dimension but labels is " +
                                           labels.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                if (predictions.getDimensions() != labels.getDimensions()) {
                    string error_message = "BinaryCrossEntropy instance: predictions and labels dimensions must match. predictions tensor is " +
                                           predictions.getDescriptorString() + "; labels tensor is " + labels.getDescriptorString() + ".";
                    throw nb::value_error(error_message.c_str());
                }
                builder.predictions(predictions).labels(labels);
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                if (predictions.getValuesDataType() != DataType::FP16 && predictions.getValuesDataType() != DataType::FP32)
                    throw nb::value_error("BinaryCrossEntropy instance: ragged predictions must use fp16 or fp32 dtype");
                if (reported_loss_shape == LossShape::PER_OUTPUT)
                    throw nb::value_error("BinaryCrossEntropy instance: per_output reporting is undefined for ragged predictions.");
                if (predictions.getOffsets() != labels.getOffsets())
                    throw nb::value_error("BinaryCrossEntropy instance: ragged predictions and labels must use the exact same row partition tensor.");
                if (predictions.getBatchSize() != labels.getBatchSize() ||
                    predictions.getMaxTotalValues() != labels.getMaxTotalValues() ||
                    predictions.getTrailingDimensions() != labels.getTrailingDimensions())
                    throw nb::value_error("BinaryCrossEntropy instance: ragged predictions and labels must have identical value geometry.");
                builder.predictions(predictions).labels(labels);
            } else {
                throw nb::type_error("BinaryCrossEntropy predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }

            setReportedLossShape(builder, reported_loss_shape);
            BinaryCrossEntropy built = builder.build();
            new (self) BinaryCrossEntropy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a = DataType::FP32,
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged Binary Cross Entropy loss.)nbdoc");

    binary_cross_entropy.def("get_predictions", [](const BinaryCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    binary_cross_entropy.def("get_labels", [](const BinaryCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    binary_cross_entropy.def("get_raw_loss", [](const BinaryCrossEntropy& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    binary_cross_entropy.def("get_loss", [](const BinaryCrossEntropy& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    binary_cross_entropy.def_prop_ro("is_ragged", &BinaryCrossEntropy::isRagged);

    binary_cross_entropy.attr("__doc__") = R"nbdoc(
Binary cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor or thor.RaggedTensor
labels : thor.Tensor or thor.RaggedTensor
loss_data_type : thor.DataType, default thor.DataType.fp32
reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
    Controls the reported loss tensor:

    * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
    * ``batch`` averages over the batch after summing all non-batch values.
    * ``per_example`` sums all non-batch values independently for each example.
    * ``per_output`` averages over the batch and preserves every non-batch dimension for dense inputs; it is undefined for ragged inputs.
    * ``raw`` reports the unreduced pointwise loss.

If you want to inspect mutually exclusive binary categories, it may be more convenient
to use SparseCategoricalCrossEntropy with two classes.
)nbdoc";
}
