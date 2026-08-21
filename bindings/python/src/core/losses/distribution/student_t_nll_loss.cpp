#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <cmath>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/StudentTNLLLoss.h"
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

void validateReportedLossShape(LossShape shape, const string& lossName) {
    if (shape != LossShape::NONE && shape != LossShape::BATCH && shape != LossShape::PER_OUTPUT && shape != LossShape::PER_EXAMPLE &&
        shape != LossShape::RAW) {
        string message = "Invalid value " + to_string((int)shape) + " passed for enum reported_loss_shape to " + lossName + ".";
        throw nb::value_error(message.c_str());
    }
}

void setReportedLossShape(StudentTNLLLoss::Builder& builder, LossShape shape) {
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

void validateArguments(Tensor location,
                       Tensor logScale,
                       Tensor labels,
                       optional<float> degreesOfFreedom,
                       optional<Tensor> learnedLogDegreesOfFreedom,
                       float minimumDegreesOfFreedom,
                       optional<DataType> lossDataType,
                       LossShape reportedLossShape) {
    const string lossName = "StudentTNLLLoss instance";
    if (location.getDimensions().empty())
        throw nb::value_error("StudentTNLLLoss instance: location must have at least one non-batch dimension");
    if (logScale.getDimensions() != location.getDimensions()) {
        string message = lossName + ": log_scale dimensions " + logScale.getDescriptorString() +
                         " must match location dimensions " + location.getDescriptorString();
        throw nb::value_error(message.c_str());
    }
    if (labels.getDimensions() != location.getDimensions()) {
        string message = lossName + ": labels dimensions " + labels.getDescriptorString() +
                         " must match location dimensions " + location.getDescriptorString();
        throw nb::value_error(message.c_str());
    }
    if (learnedLogDegreesOfFreedom.has_value() && learnedLogDegreesOfFreedom.value().getDimensions() != location.getDimensions()) {
        string message = lossName + ": learned_log_degrees_of_freedom dimensions " +
                         learnedLogDegreesOfFreedom.value().getDescriptorString() + " must match location dimensions " +
                         location.getDescriptorString();
        throw nb::value_error(message.c_str());
    }
    if (!isFloatingDType(location.getDataType()))
        throw nb::value_error("StudentTNLLLoss instance: location must use fp16 or fp32 dtype.");
    if (!isFloatingDType(logScale.getDataType()))
        throw nb::value_error("StudentTNLLLoss instance: log_scale must use fp16 or fp32 dtype.");
    if (!isFloatingDType(labels.getDataType()))
        throw nb::value_error("StudentTNLLLoss instance: labels must use fp16 or fp32 dtype.");
    if (learnedLogDegreesOfFreedom.has_value() && !isFloatingDType(learnedLogDegreesOfFreedom.value().getDataType()))
        throw nb::value_error("StudentTNLLLoss instance: learned_log_degrees_of_freedom must use fp16 or fp32 dtype.");
    if (degreesOfFreedom.has_value() && learnedLogDegreesOfFreedom.has_value())
        throw nb::value_error(
            "StudentTNLLLoss instance: specify either fixed degrees_of_freedom or learned_log_degrees_of_freedom, not both.");
    if (degreesOfFreedom.has_value() && (!std::isfinite(degreesOfFreedom.value()) || degreesOfFreedom.value() <= 0.0f))
        throw nb::value_error("StudentTNLLLoss instance: degrees_of_freedom must be greater than zero.");
    if (!std::isfinite(minimumDegreesOfFreedom) || minimumDegreesOfFreedom < 0.0f)
        throw nb::value_error("StudentTNLLLoss instance: minimum_degrees_of_freedom must be finite and non-negative.");
    const float effectiveFixedDegreesOfFreedom = degreesOfFreedom.value_or(3.0f);
    if (!learnedLogDegreesOfFreedom.has_value() && effectiveFixedDegreesOfFreedom <= minimumDegreesOfFreedom)
        throw nb::value_error(
            "StudentTNLLLoss instance: fixed degrees_of_freedom must be greater than minimum_degrees_of_freedom.");
    DataType effectiveLossDataType = lossDataType.value_or(location.getDataType());
    if (!isFloatingDType(effectiveLossDataType))
        throw nb::value_error("StudentTNLLLoss instance: loss_data_type must be fp16 or fp32.");
    validateReportedLossShape(reportedLossShape, lossName);
}

void maybeSetExampleWeights(StudentTNLLLoss::Builder& builder,
                            Tensor location,
                            Tensor logScale,
                            Tensor labels,
                            optional<Tensor> learnedLogDegreesOfFreedom,
                            optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == location || exampleWeights.value() == logScale || exampleWeights.value() == labels ||
        (learnedLogDegreesOfFreedom.has_value() && exampleWeights.value() == learnedLogDegreesOfFreedom.value())) {
        throw nb::value_error(
            "StudentTNLLLoss instance: example_weights must be distinct from location, log_scale, labels, and learned degrees of freedom.");
    }
    if (!isFloatingDType(exampleWeights.value().getDataType()))
        throw nb::value_error("StudentTNLLLoss instance: example_weights must use fp16 or fp32 dtype.");
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != location.getDimensions()) {
        string message =
            "StudentTNLLLoss instance: example_weights dimensions must be [1] for per-example weights or match location. "
            "example_weights tensor is " +
            exampleWeights.value().getDescriptorString() + "; location tensor is " + location.getDescriptorString() + ".";
        throw nb::value_error(message.c_str());
    }
    builder.exampleWeights(exampleWeights.value());
}
}  // namespace

void bind_student_t_nll_loss(nb::module_& losses) {
    auto lossClass = nb::class_<StudentTNLLLoss, Loss>(losses, "StudentTNLLLoss");
    lossClass.attr("__module__") = "thor.losses.distribution";

    lossClass.def(
        "__init__",
        [](StudentTNLLLoss* self,
           Network& network,
           Tensor location,
           Tensor log_scale,
           Tensor labels,
           optional<float> degrees_of_freedom,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           float minimum_degrees_of_freedom,
           optional<Tensor> learned_log_degrees_of_freedom,
           optional<float> loss_weight,
           optional<Tensor> example_weights) {
            validateArguments(location,
                              log_scale,
                              labels,
                              degrees_of_freedom,
                              learned_log_degrees_of_freedom,
                              minimum_degrees_of_freedom,
                              loss_data_type,
                              reported_loss_shape);

            StudentTNLLLoss::Builder builder;
            builder.network(network)
                .location(location)
                .logScale(log_scale)
                .labels(labels)
                .minimumDegreesOfFreedom(minimum_degrees_of_freedom)
                .lossDataType(loss_data_type.value_or(location.getDataType()))
                .lossWeight(loss_weight.value_or(1.0f));
            if (degrees_of_freedom.has_value())
                builder.degreesOfFreedom(degrees_of_freedom.value());
            if (learned_log_degrees_of_freedom.has_value())
                builder.logDegreesOfFreedom(learned_log_degrees_of_freedom.value());
            maybeSetExampleWeights(builder, location, log_scale, labels, learned_log_degrees_of_freedom, example_weights);
            setReportedLossShape(builder, reported_loss_shape);
            StudentTNLLLoss built = builder.build();
            new (self) StudentTNLLLoss(std::move(built));
        },
        "network"_a,
        "location"_a,
        "log_scale"_a,
        "labels"_a,
        "degrees_of_freedom"_a.none() = nb::none(),
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "minimum_degrees_of_freedom"_a = 0.0f,
        "learned_log_degrees_of_freedom"_a.none() = nb::none(),
        "loss_weight"_a.none() = nb::none(),
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a Student-t negative log-likelihood loss.)nbdoc");

    lossClass.def_prop_ro("location", &StudentTNLLLoss::getLocation);
    lossClass.def_prop_ro("log_scale", &StudentTNLLLoss::getLogScale);
    lossClass.def_prop_ro("degrees_of_freedom", &StudentTNLLLoss::getDegreesOfFreedom);
    lossClass.def_prop_ro("learned_log_degrees_of_freedom", &StudentTNLLLoss::getLearnedLogDegreesOfFreedom);
    lossClass.def_prop_ro("minimum_degrees_of_freedom", &StudentTNLLLoss::getMinimumDegreesOfFreedom);

    lossClass.attr("__doc__") = R"nbdoc(
Student-t negative log-likelihood using location, log-scale, and fixed or learned degrees of freedom.

For location m, scale s > 0, degrees of freedom nu > 0, and standardized
residual z = (target - m) / s, the per-element negative log-likelihood is:

    log(s) + lgamma(nu / 2) - lgamma((nu + 1) / 2)
    + 0.5 * log(nu * pi)
    + 0.5 * (nu + 1) * log1p(z^2 / nu)

log_scale always contains log(s), allowing an unconstrained scale head. Supply
`degrees_of_freedom` for fixed nu. Alternatively supply
`learned_log_degrees_of_freedom`, whose tensor receives an analytical gradient.
With the default `minimum_degrees_of_freedom=0.0`, it contains log(nu). When a
positive minimum m is supplied, learned nu is parameterized as
`nu = m + exp(learned_log_degrees_of_freedom)`, so the tensor contains the log
of the degrees-of-freedom excess above the floor. If neither fixed nor learned
degrees of freedom is supplied, fixed nu defaults to 3.0. Fixed nu must be
greater than the configured minimum.

example_weights may be [1] for per-example weighting or may match location for
elementwise weighting. Weights scale the raw NLL and all learned-parameter
gradients before loss-shape reduction.
)nbdoc";
}
