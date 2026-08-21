#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/GaussianNLLLoss.h"
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

void setReportedLossShape(GaussianNLLLoss::Builder &builder, LossShape reported_loss_shape) {
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

void validateGaussianNLLLossArguments(const string &loss_name,
                                      Tensor predictions,
                                      Tensor labels,
                                      Tensor variance,
                                      optional<DataType> loss_data_type,
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
    if (variance.getDimensions() != predictions.getDimensions()) {
        string error_message = loss_name + ": variance dimensions " + variance.getDescriptorString() +
                               " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (!isFloatingDType(predictions.getDataType()))
        throw nb::value_error("GaussianNLLLoss instance: predictions must use fp16 or fp32 dtype");
    if (!isFloatingDType(labels.getDataType()))
        throw nb::value_error("GaussianNLLLoss instance: labels must use fp16 or fp32 dtype");
    if (!isFloatingDType(variance.getDataType()))
        throw nb::value_error("GaussianNLLLoss instance: variance must use fp16 or fp32 dtype");
    DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
    if (!isFloatingDType(effectiveLossDataType))
        throw nb::value_error("GaussianNLLLoss instance: loss_data_type must be fp16 or fp32");
    if (eps <= 0.0f)
        throw nb::value_error("GaussianNLLLoss instance: eps must be greater than zero");
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void maybeSetExampleWeights(GaussianNLLLoss::Builder &builder,
                            Tensor predictions,
                            Tensor labels,
                            Tensor variance,
                            optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions || example_weights.value() == labels || example_weights.value() == variance)
        throw nb::value_error("GaussianNLLLoss instance: example_weights must be distinct from predictions, labels, and variance.");
    if (!isFloatingDType(example_weights.value().getDataType()))
        throw nb::value_error("GaussianNLLLoss instance: example_weights must use fp16 or fp32 dtype.");
    const vector<uint64_t>& dims = example_weights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        string error_message =
            "GaussianNLLLoss instance: example_weights dimensions must be [1] for per-example weights or match predictions. "
            "example_weights tensor is " +
            example_weights.value().getDescriptorString() + "; predictions tensor is " + predictions.getDescriptorString() + ".";
        throw nb::value_error(error_message.c_str());
    }
    builder.exampleWeights(example_weights.value());
}
}  // namespace

void bind_gaussian_nll_loss(nb::module_ &losses) {
    auto gaussian_nll_loss = nb::class_<GaussianNLLLoss, Loss>(losses, "GaussianNLLLoss");
    gaussian_nll_loss.attr("__module__") = "thor.losses.distribution";

    gaussian_nll_loss.def(
        "__init__",
        [](GaussianNLLLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           Tensor variance,
           bool full,
           float eps,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           optional<float> loss_weight,
           bool log_variance,
           optional<Tensor> example_weights) {
            const string loss_name = "GaussianNLLLoss instance";
            validateGaussianNLLLossArguments(loss_name, predictions, labels, variance, loss_data_type, reported_loss_shape, eps);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            GaussianNLLLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .variance(variance)
                .logVariance(log_variance)
                .full(full)
                .eps(eps)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            maybeSetExampleWeights(builder, predictions, labels, variance, example_weights);
            setReportedLossShape(builder, reported_loss_shape);
            GaussianNLLLoss built = builder.build();

            new (self) GaussianNLLLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "variance"_a,
        "full"_a = false,
        "eps"_a = 1.0e-6f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        "log_variance"_a = false,
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a Gaussian negative log-likelihood loss.)nbdoc");

    gaussian_nll_loss.def_prop_ro("variance", &GaussianNLLLoss::getVariance);
    gaussian_nll_loss.def_prop_ro("log_variance", &GaussianNLLLoss::getLogVariance);
    gaussian_nll_loss.def_prop_ro("full", &GaussianNLLLoss::getFull);
    gaussian_nll_loss.def_prop_ro("eps", &GaussianNLLLoss::getEps);

    gaussian_nll_loss.attr("__doc__") = R"nbdoc(
Gaussian negative log-likelihood loss.

predictions are means and labels are targets. By default variance contains a
positive per-element variance and is clamped to at least eps. With
log_variance=True, variance contains log(variance), so an unconstrained network
head can be trained directly and the raw loss is evaluated as:

    0.5 * (log_variance + (predictions - labels)^2 * exp(-log_variance))

If full is True, the constant 0.5 * log(2*pi) is included.

example_weights may be a [1] per-example weight tensor or match predictions for
elementwise weighting. Weights multiply the raw loss and both learned-parameter
gradients before loss-shape reduction.
)nbdoc";
}
