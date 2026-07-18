#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "bindings/python/src/core/losses/regression_loss_dtype.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;
using LabelType = Loss::LabelType;

namespace {

void maybeSetExampleWeights(MSE::Builder& builder,
                            Tensor predictions,
                            Tensor labels,
                            std::optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions || example_weights.value() == labels)
        throw nb::value_error("MSE instance: example_weights must be distinct from predictions and labels.");
    ThorPython::RegressionLossDType::validateExampleWeights("MSE instance", example_weights.value());
    const std::vector<uint64_t>& dims = example_weights.value().getDimensions();
    if (dims != std::vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        string error_message = "MSE instance: example_weights dimensions must be [1] for per-example weights or match predictions. "
                               "example_weights tensor is " +
                               example_weights.value().getDescriptorString() + "; predictions tensor is " +
                               predictions.getDescriptorString() + ".";
        throw nb::value_error(error_message.c_str());
    }
    builder.exampleWeights(example_weights.value());
}

}  // namespace

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
           std::optional<float> loss_weight,
           std::optional<Tensor> example_weights) {
            const string loss_name = "MSE instance";
            ThorPython::RegressionLossDType::validatePredictions(loss_name, predictions);
            ThorPython::RegressionLossDType::validateLabels(loss_name, labels);
            const DataType effectiveLossDataType =
                ThorPython::RegressionLossDType::effectiveLossDType(loss_name, predictions.getDataType(), loss_data_type);

            MSE::Builder builder;

            builder.network(network).predictions(predictions).labels(labels).lossDataType(effectiveLossDataType);
            builder.lossWeight(loss_weight.value_or(1.0f));
            maybeSetExampleWeights(builder, predictions, labels, example_weights);

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
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a MSE loss.)nbdoc");

    mean_squared_error.attr("__doc__") = R"nbdoc(
MSE loss.

Parameters
----------
network : thor.Network
predictions : thor.Tensor
labels : thor.Tensor
loss_data_type : thor.DataType | None, default fp16 for fp16 predictions, otherwise fp32
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
