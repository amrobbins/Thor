#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/PoissonNLLLoss.h"
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
    if (reported_loss_shape != LossShape::NONE && reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::PER_OUTPUT &&
        reported_loss_shape != LossShape::PER_EXAMPLE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(PoissonNLLLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::NONE) {
        builder.reportsNoLoss();
    } else if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::PER_OUTPUT) {
        builder.reportsPerOutputLoss();
    } else if (reported_loss_shape == LossShape::PER_EXAMPLE) {
        builder.reportsPerExampleLoss();
    } else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

bool isFloatingDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

bool isPoissonTargetDType(DataType dtype) {
    return dtype == DataType::BOOLEAN || dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           isFloatingDType(dtype);
}

void validatePoissonNLLLossArguments(const string &loss_name,
                                     Tensor predictions,
                                     Tensor labels,
                                     std::optional<DataType> loss_data_type,
                                     LossShape reported_loss_shape,
                                     float eps) {
    if (predictions.getDimensions().empty()) {
        string error_message = loss_name + ": predictions must have at least one non-batch dimension";
        throw nb::value_error(error_message.c_str());
    }
    if (labels.getDimensions() != predictions.getDimensions()) {
        string error_message = loss_name + ": labels dimensions " + labels.getDescriptorString() +
                               " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (!isFloatingDType(predictions.getDataType())) {
        string error_message = loss_name + ": predictions must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (!isPoissonTargetDType(labels.getDataType())) {
        string error_message = loss_name + ": labels must use boolean, unsigned integer, fp16, or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
    if (!isFloatingDType(effectiveLossDataType)) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    if (eps <= 0.0f) {
        string error_message = loss_name + ": eps must be greater than zero";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void maybeSetPoissonExampleWeights(PoissonNLLLoss::Builder &builder,
                                   Tensor predictions,
                                   Tensor labels,
                                   std::optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions || example_weights.value() == labels)
        throw nb::value_error("PoissonNLLLoss instance: example_weights must be distinct from predictions and labels.");
    if (!isFloatingDType(example_weights.value().getDataType()))
        throw nb::value_error("PoissonNLLLoss instance: example_weights must use fp16 or fp32 dtype.");
    const std::vector<uint64_t>& dims = example_weights.value().getDimensions();
    if (dims != std::vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        string error_message =
            "PoissonNLLLoss instance: example_weights dimensions must be [1] for per-example weights or match predictions. "
            "example_weights tensor is " +
            example_weights.value().getDescriptorString() + "; predictions tensor is " + predictions.getDescriptorString() + ".";
        throw nb::value_error(error_message.c_str());
    }
    builder.exampleWeights(example_weights.value());
}

}  // namespace

void bind_poisson_nll_loss(nb::module_ &losses) {
    auto poisson_nll_loss = nb::class_<PoissonNLLLoss, Loss>(losses, "PoissonNLLLoss");
    poisson_nll_loss.attr("__module__") = "thor.losses.distribution";

    poisson_nll_loss.def(
        "__init__",
        [](PoissonNLLLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           bool log_input,
           bool full,
           float eps,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight,
           std::optional<Tensor> example_weights) {
            const string loss_name = "PoissonNLLLoss instance";
            validatePoissonNLLLossArguments(loss_name, predictions, labels, loss_data_type, reported_loss_shape, eps);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            PoissonNLLLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .logInput(log_input)
                .full(full)
                .eps(eps)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            maybeSetPoissonExampleWeights(builder, predictions, labels, example_weights);
            setReportedLossShape(builder, reported_loss_shape);
            PoissonNLLLoss built = builder.build();

            new (self) PoissonNLLLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "log_input"_a = true,
        "full"_a = false,
        "eps"_a = 1.0e-8f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a Poisson negative log-likelihood loss.)nbdoc");

    poisson_nll_loss.def_prop_ro("log_input", &PoissonNLLLoss::getLogInput);
    poisson_nll_loss.def_prop_ro("full", &PoissonNLLLoss::getFull);
    poisson_nll_loss.def_prop_ro("eps", &PoissonNLLLoss::getEps);

    poisson_nll_loss.attr("__doc__") = R"nbdoc(
Poisson negative log-likelihood loss.

When log_input is True, predictions are log-rates and the raw loss is:

    exp(predictions) - labels * predictions

When log_input is False, predictions are rates and the raw loss is:

    predictions - labels * log(predictions + eps)

If full is True, the Stirling approximation term for labels > 1 is included.

example_weights may be a [1] per-example weight tensor or match predictions
for elementwise weighting. Weights multiply both the raw loss and the
prediction gradient before loss-shape reduction.
)nbdoc";
}
