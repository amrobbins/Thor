#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = Tensor::DataType;

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
           bool reportsElementwiseLoss) {
            if (predictions.getDimensions().size() != 1 || predictions.getDimensions()[0] != 1) {
                string error_message =
                    "BinaryCrossEntropy instance: predictions must be a 1 dimensional tensor of size one but predictions is " +
                    predictions.getDescriptorString();
                throw nb::value_error(error_message.c_str());
            }
            if (labels.getDimensions().size() != 1 || labels.getDimensions()[0] != 1) {
                string error_message = "BinaryCrossEntropy instance: labels must be a 1 dimensional tensor of size one but labels is " +
                                       labels.getDescriptorString();
                throw nb::value_error(error_message.c_str());
            }
            if (loss_data_type != DataType::FP16 && loss_data_type != DataType::FP32) {
                string error_message = "BinaryCrossEntropy instance: loss_data_type must be fp16 or fp32";
                throw nb::value_error(error_message.c_str());
            }

            BinaryCrossEntropy::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).lossDataType(loss_data_type);

            if (reportsElementwiseLoss)
                builder.reportsElementwiseLoss();

            BinaryCrossEntropy built = builder.build();

            new (self) BinaryCrossEntropy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a = DataType::FP32,
        "reports_elementwise_loss"_a = false,
        // nb::sig("def __init__(self, "
        //         "network: thor.Network, "
        //         "predictions: thor.Tensor, "
        //         "labels: thor.Tensor, "
        //         "reports_batch_loss: bool | None = None, "
        //         "reports_elementwise_loss: bool | None = None, "
        //         "loss_data_type: thor.DataType = thor.DataType.fp32"
        //         ") -> None"),
        R"nbdoc(Construct a Binary Cross Entropy loss.)nbdoc");

    binary_cross_entropy.attr("__doc__") = R"nbdoc(
Binary cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
reports_batch_loss : Optional[bool], default None
    If True, report a single batch-aggregated loss.
    When reports_batch_loss and reports_elementwise_loss are None, defaults to batch loss.
reports_elementwise_loss : Optional[bool], default None
    If True, report elementwise loss.
    When reports_batch_loss and reports_elementwise_loss are None, defaults to batch loss.
loss_data_type : thor.DataType, default thor.DataType.FP32

Loss reductions available, meant to aid in hand analysis of a data set:

 * Batch [b][1] -> [1] - default
 * Elementwise [b][1] -> [b]

So you could send a single batch and check the loss per example using Elementwise.
If you want to see loss per category (categories 1, 0 in this case) it may be more
convenient to get that using CategoricalCrossEntropy, e.g. use index labels
with 2 classes and set reported_loss_shape=Loss.LossShape.classwise.
)nbdoc";
}
