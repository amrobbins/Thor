#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
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
           Tensor predictions,
           Tensor labels,
           DataType loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
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
            validateReportedLossShape(reported_loss_shape, "BinaryCrossEntropy instance");
            if (loss_data_type != DataType::FP16 && loss_data_type != DataType::FP32) {
                string error_message = "BinaryCrossEntropy instance: loss_data_type must be fp16 or fp32";
                throw nb::value_error(error_message.c_str());
            }

            BinaryCrossEntropy::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).lossDataType(loss_data_type)
                .lossWeight(loss_weight.value_or(1.0f));

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
        R"nbdoc(Construct a Binary Cross Entropy loss.)nbdoc");

    binary_cross_entropy.attr("__doc__") = R"nbdoc(
Binary cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
loss_data_type : thor.DataType, default thor.DataType.fp32
reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
    Controls the reported loss tensor:

    * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
    * ``batch`` averages over the batch after summing all non-batch values.
    * ``per_example`` sums all non-batch values independently for each example.
    * ``per_output`` averages over the batch and preserves every non-batch dimension.
    * ``raw`` reports the unreduced pointwise loss.

If you want to inspect mutually exclusive binary categories, it may be more convenient
to use SparseCategoricalCrossEntropy with two classes.
)nbdoc";
}
