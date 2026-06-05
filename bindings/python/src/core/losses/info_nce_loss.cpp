#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/InfoNCELoss.h"
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

void setReportedLossShape(InfoNCELoss::Builder &builder, LossShape reported_loss_shape) {
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

void validateInfoNCELossArguments(const string &loss_name,
                                  Tensor predictions,
                                  Tensor labels,
                                  float temperature,
                                  optional<DataType> loss_data_type,
                                  LossShape reported_loss_shape) {
    if (predictions.getDimensions().size() != 1 || predictions.getDimensions()[0] <= 1) {
        string error_message = loss_name + ": predictions must be a 1 dimensional logits tensor with more than one candidate but predictions is " +
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

void bind_info_nce_loss(nb::module_ &losses) {
    auto info_nce_loss = nb::class_<InfoNCELoss, Loss>(losses, "InfoNCELoss");
    info_nce_loss.attr("__module__") = "thor.losses";

    info_nce_loss.def(
        "__init__",
        [](InfoNCELoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float temperature,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "InfoNCELoss instance";
            validateInfoNCELossArguments(loss_name, predictions, labels, temperature, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            InfoNCELoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .temperature(temperature)
                .lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            InfoNCELoss built = builder.build();

            new (self) InfoNCELoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "temperature"_a = 1.0f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct an InfoNCE loss from similarity logits and dense targets.)nbdoc");

    info_nce_loss.def_prop_ro("temperature", &InfoNCELoss::getTemperature);

    info_nce_loss.attr("__doc__") = R"nbdoc(
InfoNCE loss from similarity logits and dense target distributions.

The predictions tensor contains unnormalized similarity logits over the candidate set and
the labels tensor contains a dense one-hot, multi-hot, or soft target distribution with
the same candidate dimension. The raw loss is:

    -target * log_softmax(logits / temperature)

For the standard one-positive in-batch contrastive case, pass one-hot labels whose positive
entry selects the matching candidate for each batch item.
)nbdoc";
}
