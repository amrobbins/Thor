#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <utility>

#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "Utilities/Expression/DynamicExpression.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;
using DataType = ThorImplementation::DataType;
using DynamicExpression = ThorImplementation::DynamicExpression;

void bind_custom_loss(nb::module_& losses) {
    auto custom_loss = nb::class_<CustomLoss, Loss>(losses, "CustomLoss");
    custom_loss.attr("__module__") = "thor.losses";

    custom_loss.def(
        "__init__",
        [](CustomLoss* self,
           Network& network,
           DynamicExpression loss_expression,
           DynamicExpression gradient_expression,
           Tensor predictions,
           Tensor labels,
           DataType loss_data_type,
           Loss::LossShape reported_loss_shape,
           const std::string& predictions_name,
           const std::string& labels_name,
           const std::string& loss_name,
           const std::string& gradient_name) {
            CustomLoss::Builder builder;
            builder.network(network)
                .lossExpression(std::move(loss_expression))
                .gradientExpression(std::move(gradient_expression))
                .predictions(std::move(predictions))
                .labels(std::move(labels))
                .predictionsName(predictions_name)
                .labelsName(labels_name)
                .lossName(loss_name)
                .gradientName(gradient_name)
                .lossDataType(loss_data_type);

            switch (reported_loss_shape) {
                case Loss::LossShape::BATCH:
                    builder.reportsBatchLoss();
                    break;
                case Loss::LossShape::ELEMENTWISE:
                    builder.reportsElementwiseLoss();
                    break;
                case Loss::LossShape::CLASSWISE:
                    builder.reportsClasswiseLoss();
                    break;
                case Loss::LossShape::RAW:
                    builder.reportsRawLoss();
                    break;
                default:
                    throw nb::value_error("CustomLoss instance: unsupported reported_loss_shape");
            }

            CustomLoss built = builder.build();
            new (self) CustomLoss(std::move(built));
        },
        "network"_a,
        "loss_expression"_a,
        "gradient_expression"_a,
        "predictions"_a,
        "labels"_a,
        "loss_data_type"_a = DataType::FP32,
        "reported_loss_shape"_a = Loss::LossShape::BATCH,
        "predictions_name"_a = "predictions",
        "labels_name"_a = "labels",
        "loss_name"_a = "loss",
        "gradient_name"_a = "predictions_grad",
        R"nbdoc(Construct an expression-backed CustomLoss.)nbdoc");

    custom_loss.def_prop_ro("predictions_name", &CustomLoss::getPredictionsName);
    custom_loss.def_prop_ro("labels_name", &CustomLoss::getLabelsName);
    custom_loss.def_prop_ro("loss_name", &CustomLoss::getLossName);
    custom_loss.def_prop_ro("gradient_name", &CustomLoss::getGradientName);

    custom_loss.attr("__doc__") = R"nbdoc(
Expression-backed custom loss.

A CUDA-backed loss does not need a separate CudaKernelLoss type. Build one
CudaKernelExpression for the raw loss, another for dLoss/dPredictions, convert
both with ``as_dynamic_expression()``, and pass them here. The same
Network-level CUDA-kernel source inspection and save/load key policy applies.

Parameters
----------
network : thor.Network
loss_expression : thor.physical.DynamicExpression
    Expression that maps predictions and labels to a raw per-example loss tensor.
gradient_expression : thor.physical.DynamicExpression
    Expression that maps predictions and labels to dLoss/dPredictions. Its output descriptor must match predictions.
predictions : thor.Tensor
labels : thor.Tensor
loss_data_type : thor.DataType, default thor.DataType.FP32
reported_loss_shape : thor.losses.Loss.LossShape, default LossShape.batch
predictions_name : str, default "predictions"
labels_name : str, default "labels"
loss_name : str, default "loss"
gradient_name : str, default "predictions_grad"
)nbdoc";
}
