#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/SoftTargetCrossEntropy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string &loss_name) {
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::CLASSWISE &&
        reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(SoftTargetCrossEntropy::Builder &builder, LossShape reported_loss_shape) {
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

void validateDistributionLossArguments(const string &loss_name,
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
    if (predictions.getDataType() != DataType::FP16 && predictions.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": predictions must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (labels.getDataType() != DataType::FP16 && labels.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": labels must use fp16 or fp32 dtype";
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

void bind_soft_target_cross_entropy(nb::module_ &losses) {
    auto soft_target_cross_entropy = nb::class_<SoftTargetCrossEntropy, Loss>(losses, "SoftTargetCrossEntropy");
    soft_target_cross_entropy.attr("__module__") = "thor.losses";

    soft_target_cross_entropy.def(
        "__init__",
        [](SoftTargetCrossEntropy *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "SoftTargetCrossEntropy instance";
            validateDistributionLossArguments(loss_name, predictions, labels, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            SoftTargetCrossEntropy::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            SoftTargetCrossEntropy built = builder.build();

            new (self) SoftTargetCrossEntropy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a soft-target cross entropy loss.)nbdoc");

    soft_target_cross_entropy.attr("__doc__") = R"nbdoc(
Soft-target categorical cross entropy from logits.

The predictions tensor contains unnormalized logits and the labels tensor contains
a dense target distribution with the same class dimension. The raw loss is:

    -target * log_softmax(logits)

The gradient assumes targets are normalized distributions.
)nbdoc";
}
