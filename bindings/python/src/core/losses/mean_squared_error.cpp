#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;
using LabelType = Loss::LabelType;

void bind_mean_squared_error(nb::module_ &losses) {
    auto mean_squared_error = nb::class_<MSE, Loss>(losses, "MSE");
    mean_squared_error.attr("__module__") = "thor.losses";

    mean_squared_error.def(
        "__init__",
        [](MSE *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           std::optional<DataType> loss_data_type,
           bool reportsElementwiseLoss,
           std::optional<float> loss_weight) {
            MSE::Builder builder;

            builder.network(network).predictions(predictions).labels(labels);
            if (loss_data_type.has_value())
                builder.lossDataType(loss_data_type.value());
            builder.lossWeight(loss_weight.value_or(1.0f));

            if (predictions.getDimensions() != labels.getDimensions()) {
                string error_message = "MSE instance: predictions and labels dimensions must match. predictions tensor is " +
                                       predictions.getDescriptorString() + "; labels tensor is " + labels.getDescriptorString() + ".";
                throw nb::value_error(error_message.c_str());
            }

            if (reportsElementwiseLoss)
                builder.reportsElementwiseLoss();

            MSE built = builder.build();

            new (self) MSE(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reportsElementwiseLoss"_a = false,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct a MSE loss.)nbdoc");

    mean_squared_error.attr("__doc__") = R"nbdoc(
MSE loss.

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
