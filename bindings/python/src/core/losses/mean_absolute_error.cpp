#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Tensor::DataType;
using LossShape = Loss::LossShape;
using LabelType = Loss::LabelType;

void bind_mean_absolute_error(nb::module_ &losses) {
    auto mean_absolute_error = nb::class_<MeanAbsoluteError, Loss>(losses, "MeanAbsoluteError");
    mean_absolute_error.attr("__module__") = "thor.losses";

    mean_absolute_error.def(
        "__init__",
        [](MeanAbsoluteError *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           optional<DataType> loss_data_type,
           bool reportsElementwiseLoss) {
            MeanAbsoluteError::Builder builder;

            builder.network(network).predictions(predictions).labels(labels);
            if (loss_data_type.has_value())
                builder.lossDataType(loss_data_type.value());

            if (labels.getDimensions().size() != 1 || labels.getDimensions()[0] != 1) {
                string error_message = "MeanAbsoluteError instance: labels tensor is not sized right. label tensor is " +
                                       labels.getDescriptorString() + ". labels must be a 1 dimensional tensor of size 1.";
                throw nb::value_error(error_message.c_str());
            }

            if (reportsElementwiseLoss)
                builder.reportsElementwiseLoss();

            MeanAbsoluteError built = builder.build();

            new (self) MeanAbsoluteError(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reportsElementwiseLoss"_a = false,
        R"nbdoc(Construct a Categorical Cross Entropy loss.)nbdoc");

    mean_absolute_error.attr("__doc__") = R"nbdoc(
Mean Absolute Error loss.

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

 * Batch [b][1] -> [1] - default
 * Elementwise [b][1] -> [b]

So you could send a single batch and check the loss per example using Elementwise.

)nbdoc";
}
