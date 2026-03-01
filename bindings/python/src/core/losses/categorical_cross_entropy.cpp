#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "DeepLearning/Api/Layers/Loss/CategoricalCrossEntropy.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::TensorDescriptor::DataType;
using LossShape = Loss::LossShape;
using LabelType = Loss::LabelType;

void bind_categorical_cross_entropy(nb::module_ &losses) {
    auto categorical_cross_entropy = nb::class_<CategoricalCrossEntropy, Loss>(losses, "CategoricalCrossEntropy");
    categorical_cross_entropy.attr("__module__") = "thor.losses";

    categorical_cross_entropy.def(
        "__init__",
        [](CategoricalCrossEntropy *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           LabelType label_type,
           std::optional<int32_t> num_classes,
           DataType loss_data_type,
           LossShape reported_loss_shape) {
            CategoricalCrossEntropy::Builder builder;
            builder.network(network);

            // Ensure everything matches up.
            if (label_type == LabelType::ONE_HOT) {
                if (predictions.getDimensions().size() != 1) {
                    string error_message =
                        "CategoricalCrossEntropy instance: one_hot predictions must have 1 dimension but tensor format is " +
                        predictions.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                if (labels.getDimensions().size() != 1) {
                    string error_message = "CategoricalCrossEntropy instance: one_hot labels must have 1 dimension but tensor format is " +
                                           labels.getDescriptorString();
                    throw nb::value_error(error_message.c_str());
                }
                if (num_classes.has_value() &&
                    (uint64_t(num_classes.value()) != predictions.getDimensions()[0] || num_classes.value() <= 0)) {
                    string error_message = "CategoricalCrossEntropy instance: mismatch between num_classes " +
                                           to_string(num_classes.value()) + " and predictions tensor size " +
                                           to_string(predictions.getDimensions()[0]) +
                                           ". Either set num_classes to match, don't pass num_classes, or fix your predictions tensor.";
                    throw nb::value_error(error_message.c_str());
                }
                if (predictions.getDimensions()[0] != labels.getDimensions()[0]) {
                    string error_message = "CategoricalCrossEntropy instance: mismatch between predictions size " +
                                           to_string(predictions.getDimensions()[0]) + " and labels tensor size " +
                                           to_string(labels.getDimensions()[0]);
                    throw nb::value_error(error_message.c_str());
                }

                builder.receivesOneHotLabels();
            } else if (label_type == LabelType::INDEX) {
                if (!num_classes.has_value()) {
                    throw nb::value_error(
                        "CategoricalCrossEntropy instance: label_type set to LabelType.index but num_classes is None. You must pass "
                        "num_classes in this case.");
                } else if (num_classes.value() <= 0) {
                    string error_message =
                        "CategoricalCrossEntropy instance: num_classes must be a positive integer when using index labels. You passed "
                        "num_classes == " +
                        to_string(num_classes.value());
                    throw nb::value_error(error_message.c_str());
                }

                if (labels.getDimensions().size() != 1 || labels.getDimensions()[0] != 1) {
                    string error_message = "CategoricalCrossEntropy instance: labels tensor is not sized right. label tensor is " +
                                           labels.getDescriptorString() +
                                           ". labels must be a 1 dimensional tensor of size 1. since label_type == index";
                    throw nb::value_error(error_message.c_str());
                }

                builder.receivesClassIndexLabels(num_classes.value());
            } else {
                string error_message =
                    "Invalid value " + to_string((int)label_type) + " passed for enum LabelType to CategoricalCrossEntropy instance.";
                throw nb::value_error(error_message.c_str());
            }

            if (reported_loss_shape == LossShape::BATCH) {
                builder.reportsBatchLoss();
            } else if (reported_loss_shape == LossShape::CLASSWISE) {
                builder.reportsClasswiseLoss();
            } else if (reported_loss_shape == LossShape::ELEMENTWISE) {
                builder.reportsElementwiseLoss();
            } else if (reported_loss_shape == LossShape::RAW) {
                builder.reportsRawLoss();
            } else {
                string error_message = "Invalid value " + to_string((int)reported_loss_shape) +
                                       " passed for enum reported_loss_shape to CategoricalCrossEntropy instance.";
                throw nb::value_error(error_message.c_str());
            }

            builder.predictions(predictions).labels(labels).lossDataType(loss_data_type);
            CategoricalCrossEntropy built = builder.build();

            new (self) CategoricalCrossEntropy(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "label_type"_a,
        "num_classes"_a.none() = nb::none(),
        "loss_data_type"_a = DataType::FP32,
        "reported_loss_shape"_a = LossShape::BATCH,
        // Looks like nb::sig is not the way to go, let the real stuff dictate the docs, c++ is typed!
        // nb::sig("def __init__(self, "
        //         "network: thor.Network, "
        //         "predictions: thor.Tensor, "
        //         "labels: thor.Tensor, "
        //         "label_type: thor.losses.LabelType, "
        //         "num_classes: int | None = None, "
        //         "loss_data_type: DataType = DataType.fp32, "
        //         "reported_loss_shape: thor.losses.LossShape = thor.losses.LossShape.batch "
        //         ") -> None"),
        R"nbdoc(Construct a Categorical Cross Entropy loss.)nbdoc");

    categorical_cross_entropy.attr("__doc__") = R"nbdoc(
Categorical cross-entropy loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
loss_data_type : thor.DataType, default thor.DataType.FP32
num_classes : Optional[int], default None
    If True, report a single batch-aggregated loss.
    When reports_batch_loss and reports_elementwise_loss are None, defaults to batch loss.
reported_loss_shape : Optional[thor.losses.LossShape], default batch
    This setting does not affect training, this is for analysis.
    If you want to see the loss aggregated by class or by batch element or unaggregated,
    pass thor.losses.LossShape as classwise, elementwise or raw.
    Note: if you want to see the loss aggregated both classwise and elementwise, send raw
          here and use two loss shapers on the output loss to get both.


Notes
-----
This loss compares predicted logits ``z`` over classes to a target distribution ``y``
A softmax is applied internally to convert logits into probabilities:

    p_c = exp(z_c) / \sum_{j=1}^{C} exp(z_j)

The per-example categorical cross-entropy is then:

    L = -\sum_{c=1}^{C} y_c \log(p_c)

When labels are one-hot, this reduces to ``L = -\log(p_{true})``.

Loss reductions available, meant to aid in hand analysis of a data set:

 * Batch [b][c] -> [1]
 * Classwise [b][c] -> [c]
 * Elementwise [b][c] -> [b]
 * Raw [b][c] -> [b][c]

So for example you could check the loss per class using Classwise,
or you could send a single batch and check the loss per example using Elementwise.
)nbdoc";
}
