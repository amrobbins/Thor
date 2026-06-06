#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/ListwiseSoftmaxCrossEntropyLoss.h"
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

void setReportedLossShape(ListwiseSoftmaxCrossEntropyLoss::Builder &builder, LossShape reported_loss_shape) {
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

void validateListwiseSoftmaxCrossEntropyLossArguments(const string &loss_name,
                                                      Tensor predictions,
                                                      Tensor labels,
                                                      float temperature,
                                                      optional<DataType> loss_data_type,
                                                      LossShape reported_loss_shape,
                                                      optional<Tensor> mask) {
    if (predictions.getDimensions().size() != 1 || predictions.getDimensions()[0] <= 1) {
        string error_message = loss_name +
                               ": predictions must be a 1 dimensional fixed-size list score tensor with more than one document but predictions is " +
                               predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (labels.getDimensions() != predictions.getDimensions()) {
        string error_message = loss_name + ": labels dimensions " + labels.getDescriptorString() +
                               " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (mask.has_value() && mask.value().getDimensions() != predictions.getDimensions()) {
        string error_message = loss_name + ": mask dimensions " + mask.value().getDescriptorString() +
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
    if (mask.has_value()) {
        DataType maskDType = mask.value().getDataType();
        if (maskDType != DataType::BOOLEAN && maskDType != DataType::UINT8 && maskDType != DataType::FP16 && maskDType != DataType::FP32) {
            string error_message = loss_name + ": mask must use bool, uint8, fp16, or fp32 dtype";
            throw nb::value_error(error_message.c_str());
        }
    }
    if (temperature <= 0.0f) {
        string error_message = loss_name + ": temperature must be greater than zero";
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

void bind_listwise_softmax_cross_entropy_loss(nb::module_ &ranking) {
    auto listwise_softmax_cross_entropy_loss =
        nb::class_<ListwiseSoftmaxCrossEntropyLoss, Loss>(ranking, "ListwiseSoftmaxCrossEntropyLoss");
    listwise_softmax_cross_entropy_loss.attr("__module__") = "thor.losses.ranking";

    listwise_softmax_cross_entropy_loss.def(
        "__init__",
        [](ListwiseSoftmaxCrossEntropyLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float temperature,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<Tensor> mask) {
            const string loss_name = "ListwiseSoftmaxCrossEntropyLoss instance";
            validateListwiseSoftmaxCrossEntropyLossArguments(loss_name,
                                                             predictions,
                                                             labels,
                                                             temperature,
                                                             loss_data_type,
                                                             reported_loss_shape,
                                                             mask);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            ListwiseSoftmaxCrossEntropyLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).temperature(temperature).lossDataType(effectiveLossDataType);
            if (mask.has_value())
                builder.mask(mask.value());
            setReportedLossShape(builder, reported_loss_shape);
            ListwiseSoftmaxCrossEntropyLoss built = builder.build();

            new (self) ListwiseSoftmaxCrossEntropyLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "temperature"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        "mask"_a.none() = nb::none(),
        R"nbdoc(Construct a listwise softmax cross entropy loss over fixed-size query/document lists.)nbdoc");

    listwise_softmax_cross_entropy_loss.def_prop_ro("temperature", &ListwiseSoftmaxCrossEntropyLoss::getTemperature);

    listwise_softmax_cross_entropy_loss.attr("__doc__") = R"nbdoc(
Listwise softmax cross entropy over fixed-size query/document lists.

The predictions tensor contains unnormalized model scores for the documents in one fixed-size
list, and the labels tensor contains a target distribution or nonnegative target weights with
the same list dimension. An optional mask tensor may mark padded documents with 0 and valid
documents with 1. The prediction ranking distribution is computed from predictions /
temperature across the valid documents, and the raw loss is one scalar per list:

    -sum(labels * log_softmax(predictions / temperature))

When labels sum to one, this is standard listwise softmax cross entropy. When labels are
nonnegative weights, the gradient uses the per-list target mass.
)nbdoc";
}
