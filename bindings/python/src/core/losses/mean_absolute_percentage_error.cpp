#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;
using LabelType = Loss::LabelType;

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string& loss_name) {
    if (reported_loss_shape != LossShape::NONE && reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::PER_EXAMPLE &&
        reported_loss_shape != LossShape::PER_OUTPUT && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(MAPE::Builder& builder, LossShape reported_loss_shape) {
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

void bind_mean_absolute_percentage_error(nb::module_ &losses) {
    auto mean_absolute_percentage_error = nb::class_<MAPE, Loss>(losses, "MAPE");
    mean_absolute_percentage_error.attr("__module__") = "thor.losses";

    mean_absolute_percentage_error.def(
        "__init__",
        [](MAPE *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "MAPE instance";
            if (predictions.getDimensions().empty())
                throw nb::value_error("MAPE instance: predictions must have at least one non-batch dimension");
            validateReportedLossShape(reported_loss_shape, loss_name);

            MAPE::Builder builder;

            builder.network(network).predictions(predictions).labels(labels);
            if (loss_data_type.has_value())
                builder.lossDataType(loss_data_type.value());
            builder.lossWeight(loss_weight.value_or(1.0f));

            if (predictions.getDimensions() != labels.getDimensions()) {
                string error_message = "MAPE instance: predictions and labels dimensions must match. predictions tensor is " +
                                       predictions.getDescriptorString() + "; labels tensor is " + labels.getDescriptorString() + ".";
                throw nb::value_error(error_message.c_str());
            }

            setReportedLossShape(builder, reported_loss_shape);

            MAPE built = builder.build();

            new (self) MAPE(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a MAPE loss.)nbdoc");

    mean_absolute_percentage_error.attr("__doc__") = R"nbdoc(
MAPE loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
loss_data_type : thor.DataType | None, default same data type as predictions
reported_loss_shape : thor.losses.LossShape, default thor.losses.LossShape.batch
    Controls the reported loss tensor:

    * ``none`` does not expose a reportable loss tensor; the raw loss remains the training objective.
    * ``batch`` averages over the batch after summing all non-batch values.
    * ``per_example`` sums all non-batch values independently for each example.
    * ``per_output`` averages over the batch and preserves every non-batch dimension.
    * ``raw`` reports the unreduced pointwise loss.

)nbdoc";
}
