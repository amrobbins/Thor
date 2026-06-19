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
           bool reportsElementwiseLoss,
           std::optional<float> loss_weight) {
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

            if (reportsElementwiseLoss)
                builder.reportsElementwiseLoss();

            MAPE built = builder.build();

            new (self) MAPE(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reportsElementwiseLoss"_a = false,
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
reports_elementwise_loss : Optional[bool], default None
    If True, report elementwise loss.
    When reports_batch_loss and reports_elementwise_loss are None, defaults to batch loss.

Notes
-----
Loss reductions available, meant to aid in hand analysis of a data set:

 * Batch [b][...] -> [1] - default
 * Elementwise [b][...] -> [b]

So you could send a single batch and check the loss per example using Elementwise.

)nbdoc";
}
