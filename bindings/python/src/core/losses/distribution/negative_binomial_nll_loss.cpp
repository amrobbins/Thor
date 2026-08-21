#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/NegativeBinomialNLLLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;
using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {
bool isFloatingDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

bool isCountTargetDType(DataType dtype) {
    return dtype == DataType::BOOLEAN || dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
           isFloatingDType(dtype);
}

void validateReportedLossShape(LossShape shape, const string& lossName) {
    if (shape != LossShape::NONE && shape != LossShape::BATCH && shape != LossShape::PER_OUTPUT && shape != LossShape::PER_EXAMPLE &&
        shape != LossShape::RAW) {
        string message = "Invalid value " + to_string((int)shape) + " passed for enum reported_loss_shape to " + lossName + ".";
        throw nb::value_error(message.c_str());
    }
}

void setReportedLossShape(NegativeBinomialNLLLoss::Builder& builder, LossShape shape) {
    if (shape == LossShape::NONE)
        builder.reportsNoLoss();
    else if (shape == LossShape::BATCH)
        builder.reportsBatchLoss();
    else if (shape == LossShape::PER_OUTPUT)
        builder.reportsPerOutputLoss();
    else if (shape == LossShape::PER_EXAMPLE)
        builder.reportsPerExampleLoss();
    else {
        THOR_THROW_IF_FALSE(shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void validateArguments(Tensor mean,
                       Tensor dispersion,
                       Tensor labels,
                       optional<DataType> lossDataType,
                       LossShape reportedLossShape,
                       float eps) {
    const string lossName = "NegativeBinomialNLLLoss instance";
    if (mean.getDimensions().empty())
        throw nb::value_error("NegativeBinomialNLLLoss instance: mean must have at least one non-batch dimension");
    if (dispersion.getDimensions() != mean.getDimensions()) {
        string message = lossName + ": dispersion dimensions " + dispersion.getDescriptorString() +
                         " must match mean dimensions " + mean.getDescriptorString();
        throw nb::value_error(message.c_str());
    }
    if (labels.getDimensions() != mean.getDimensions()) {
        string message = lossName + ": labels dimensions " + labels.getDescriptorString() + " must match mean dimensions " + mean.getDescriptorString();
        throw nb::value_error(message.c_str());
    }
    if (!isFloatingDType(mean.getDataType()))
        throw nb::value_error("NegativeBinomialNLLLoss instance: mean must use fp16 or fp32 dtype.");
    if (!isFloatingDType(dispersion.getDataType()))
        throw nb::value_error("NegativeBinomialNLLLoss instance: dispersion must use fp16 or fp32 dtype.");
    if (!isCountTargetDType(labels.getDataType()))
        throw nb::value_error("NegativeBinomialNLLLoss instance: labels must use boolean, unsigned integer, fp16, or fp32 dtype.");
    DataType effectiveLossDataType = lossDataType.value_or(mean.getDataType());
    if (!isFloatingDType(effectiveLossDataType))
        throw nb::value_error("NegativeBinomialNLLLoss instance: loss_data_type must be fp16 or fp32.");
    if (eps <= 0.0f)
        throw nb::value_error("NegativeBinomialNLLLoss instance: eps must be greater than zero.");
    validateReportedLossShape(reportedLossShape, lossName);
}

void maybeSetExampleWeights(NegativeBinomialNLLLoss::Builder& builder,
                            Tensor mean,
                            Tensor dispersion,
                            Tensor labels,
                            optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == mean || exampleWeights.value() == dispersion || exampleWeights.value() == labels)
        throw nb::value_error("NegativeBinomialNLLLoss instance: example_weights must be distinct from mean, dispersion, and labels.");
    if (!isFloatingDType(exampleWeights.value().getDataType()))
        throw nb::value_error("NegativeBinomialNLLLoss instance: example_weights must use fp16 or fp32 dtype.");
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != mean.getDimensions()) {
        string message =
            "NegativeBinomialNLLLoss instance: example_weights dimensions must be [1] for per-example weights or match mean. "
            "example_weights tensor is " +
            exampleWeights.value().getDescriptorString() + "; mean tensor is " + mean.getDescriptorString() + ".";
        throw nb::value_error(message.c_str());
    }
    builder.exampleWeights(exampleWeights.value());
}
}  // namespace

void bind_negative_binomial_nll_loss(nb::module_& losses) {
    auto lossClass = nb::class_<NegativeBinomialNLLLoss, Loss>(losses, "NegativeBinomialNLLLoss");
    lossClass.attr("__module__") = "thor.losses.distribution";

    lossClass.def(
        "__init__",
        [](NegativeBinomialNLLLoss* self,
           Network& network,
           Tensor mean,
           Tensor dispersion,
           Tensor labels,
           bool log_mean,
           bool log_dispersion,
           float eps,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           optional<float> loss_weight,
           optional<Tensor> example_weights) {
            validateArguments(mean, dispersion, labels, loss_data_type, reported_loss_shape, eps);

            NegativeBinomialNLLLoss::Builder builder;
            builder.network(network)
                .mean(mean)
                .dispersion(dispersion)
                .labels(labels)
                .logMean(log_mean)
                .logDispersion(log_dispersion)
                .eps(eps)
                .lossDataType(loss_data_type.value_or(mean.getDataType()))
                .lossWeight(loss_weight.value_or(1.0f));
            maybeSetExampleWeights(builder, mean, dispersion, labels, example_weights);
            setReportedLossShape(builder, reported_loss_shape);
            NegativeBinomialNLLLoss built = builder.build();
            new (self) NegativeBinomialNLLLoss(std::move(built));
        },
        "network"_a,
        "mean"_a,
        "dispersion"_a,
        "labels"_a,
        "log_mean"_a = true,
        "log_dispersion"_a = true,
        "eps"_a = 1.0e-8f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a Negative Binomial negative log-likelihood loss.)nbdoc");

    lossClass.def_prop_ro("mean", &NegativeBinomialNLLLoss::getMean);
    lossClass.def_prop_ro("dispersion", &NegativeBinomialNLLLoss::getDispersion);
    lossClass.def_prop_ro("log_mean", &NegativeBinomialNLLLoss::getLogMean);
    lossClass.def_prop_ro("log_dispersion", &NegativeBinomialNLLLoss::getLogDispersion);
    lossClass.def_prop_ro("eps", &NegativeBinomialNLLLoss::getEps);

    lossClass.attr("__doc__") = R"nbdoc(
Negative Binomial negative log-likelihood using the NB2 mean/dispersion parameterization.

The distribution is parameterized by mean mu and dispersion alpha:

    Var(Y) = mu + alpha * mu^2

Equivalently, the Negative Binomial concentration is r = 1 / alpha. By default
mean and dispersion tensors contain log(mu) and log(alpha), allowing both model
heads to be unconstrained. Set log_mean=False and/or log_dispersion=False when
supplying positive parameters directly; direct parameters are floored by eps.

labels must contain non-negative counts. Floating labels are accepted for
training pipelines that represent counts in fp16/fp32.

example_weights may be [1] for per-example weighting or may match mean for
elementwise weighting. Weights scale the raw NLL and both learned-parameter
gradients before loss-shape reduction.
)nbdoc";
}
