#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <cmath>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/StudentTNLLLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"

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

void validateRaggedArguments(const RaggedTensor& location,
                            const RaggedTensor& logScale,
                            const RaggedTensor& labels,
                            optional<float> degreesOfFreedom,
                            optional<RaggedTensor> learnedLogDegreesOfFreedom,
                            float minimumDegreesOfFreedom,
                            optional<DataType> lossDataType,
                            LossShape reportedLossShape) {
    if (!isFloatingDType(location.getValuesDataType()) || !isFloatingDType(logScale.getValuesDataType()) ||
        !isFloatingDType(labels.getValuesDataType()))
        throw nb::value_error("StudentTNLLLoss instance: ragged location, log_scale, and labels must use fp16 or fp32 dtype.");
    if (location.getOffsets() != logScale.getOffsets() || location.getOffsets() != labels.getOffsets())
        throw nb::value_error("StudentTNLLLoss instance: ragged location, log_scale, and labels must use the exact same row partition tensor.");
    if (location.getBatchSize() != logScale.getBatchSize() || location.getBatchSize() != labels.getBatchSize() ||
        location.getMaxTotalValues() != logScale.getMaxTotalValues() || location.getMaxTotalValues() != labels.getMaxTotalValues() ||
        location.getTrailingDimensions() != logScale.getTrailingDimensions() || location.getTrailingDimensions() != labels.getTrailingDimensions())
        throw nb::value_error("StudentTNLLLoss instance: ragged location, log_scale, and labels must have identical value geometry.");
    if (learnedLogDegreesOfFreedom.has_value()) {
        const RaggedTensor& dof = learnedLogDegreesOfFreedom.value();
        if (!isFloatingDType(dof.getValuesDataType()))
            throw nb::value_error("StudentTNLLLoss instance: learned_log_degrees_of_freedom must use fp16 or fp32 dtype.");
        if (dof.getOffsets() != location.getOffsets())
            throw nb::value_error("StudentTNLLLoss instance: ragged learned_log_degrees_of_freedom must use the exact same row partition tensor.");
        if (dof.getBatchSize() != location.getBatchSize() || dof.getMaxTotalValues() != location.getMaxTotalValues() ||
            dof.getTrailingDimensions() != location.getTrailingDimensions())
            throw nb::value_error("StudentTNLLLoss instance: ragged learned_log_degrees_of_freedom must have identical value geometry.");
    }
    if (degreesOfFreedom.has_value() && learnedLogDegreesOfFreedom.has_value())
        throw nb::value_error("StudentTNLLLoss instance: specify either fixed degrees_of_freedom or learned_log_degrees_of_freedom, not both.");
    if (degreesOfFreedom.has_value() && (!std::isfinite(degreesOfFreedom.value()) || degreesOfFreedom.value() <= 0.0f))
        throw nb::value_error("StudentTNLLLoss instance: degrees_of_freedom must be greater than zero.");
    if (!std::isfinite(minimumDegreesOfFreedom) || minimumDegreesOfFreedom < 0.0f)
        throw nb::value_error("StudentTNLLLoss instance: minimum_degrees_of_freedom must be finite and non-negative.");
    const float effectiveFixedDegreesOfFreedom = degreesOfFreedom.value_or(3.0f);
    if (!learnedLogDegreesOfFreedom.has_value() && effectiveFixedDegreesOfFreedom <= minimumDegreesOfFreedom)
        throw nb::value_error("StudentTNLLLoss instance: fixed degrees_of_freedom must be greater than minimum_degrees_of_freedom.");
    DataType effectiveLossDataType = lossDataType.value_or(location.getValuesDataType());
    if (!isFloatingDType(effectiveLossDataType))
        throw nb::value_error("StudentTNLLLoss instance: loss_data_type must be fp16 or fp32.");
    validateReportedLossShape(reportedLossShape, "StudentTNLLLoss instance");
    if (reportedLossShape == LossShape::PER_OUTPUT)
        throw nb::value_error("StudentTNLLLoss instance: reported_loss_shape per_output is undefined for ragged sequences.");
}

void maybeSetRaggedExampleWeights(StudentTNLLLoss::Builder& builder,
                                  const RaggedTensor& location,
                                  const RaggedTensor& logScale,
                                  const RaggedTensor& labels,
                                  optional<RaggedTensor> learnedLogDegreesOfFreedom,
                                  optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value()) return;
    if (exampleWeights.value() == location.getValues() || exampleWeights.value() == logScale.getValues() ||
        exampleWeights.value() == labels.getValues() ||
        (learnedLogDegreesOfFreedom.has_value() && exampleWeights.value() == learnedLogDegreesOfFreedom->getValues()))
        throw nb::value_error("StudentTNLLLoss instance: example_weights must be distinct from ragged differentiable/label values.");
    if (!isFloatingDType(exampleWeights->getDataType()))
        throw nb::value_error("StudentTNLLLoss instance: example_weights must use fp16 or fp32 dtype.");
    if (exampleWeights->getDimensions() != vector<uint64_t>{1})
        throw nb::value_error("StudentTNLLLoss instance: ragged example_weights dimensions must be [1] for one scalar weight per logical row.");
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
           nb::object locationObject,
           nb::object logScaleObject,
           nb::object labelsObject,
           optional<float> degrees_of_freedom,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           float minimum_degrees_of_freedom,
           nb::object learnedLogDegreesOfFreedomObject,
           optional<float> loss_weight,
           optional<Tensor> example_weights) {
            StudentTNLLLoss::Builder builder;
            builder.network(network).lossWeight(loss_weight.value_or(1.0f));

            const bool noLearnedDof = learnedLogDegreesOfFreedomObject.is_none();
            if (nb::isinstance<Tensor>(locationObject) && nb::isinstance<Tensor>(logScaleObject) && nb::isinstance<Tensor>(labelsObject) &&
                (noLearnedDof || nb::isinstance<Tensor>(learnedLogDegreesOfFreedomObject))) {
                Tensor location = nb::cast<Tensor>(locationObject);
                Tensor logScale = nb::cast<Tensor>(logScaleObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                optional<Tensor> learnedDof = noLearnedDof ? nullopt : optional<Tensor>(nb::cast<Tensor>(learnedLogDegreesOfFreedomObject));
                validateArguments(location, logScale, labels, degrees_of_freedom, learnedDof, minimum_degrees_of_freedom,
                                  loss_data_type, reported_loss_shape);
                builder.location(location).logScale(logScale).labels(labels).lossDataType(loss_data_type.value_or(location.getDataType()));
                if (learnedDof.has_value()) builder.logDegreesOfFreedom(learnedDof.value());
                maybeSetExampleWeights(builder, location, logScale, labels, learnedDof, example_weights);
            } else if (nb::isinstance<RaggedTensor>(locationObject) && nb::isinstance<RaggedTensor>(logScaleObject) &&
                       nb::isinstance<RaggedTensor>(labelsObject) &&
                       (noLearnedDof || nb::isinstance<RaggedTensor>(learnedLogDegreesOfFreedomObject))) {
                RaggedTensor location = nb::cast<RaggedTensor>(locationObject);
                RaggedTensor logScale = nb::cast<RaggedTensor>(logScaleObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                optional<RaggedTensor> learnedDof = noLearnedDof ? nullopt : optional<RaggedTensor>(nb::cast<RaggedTensor>(learnedLogDegreesOfFreedomObject));
                validateRaggedArguments(location, logScale, labels, degrees_of_freedom, learnedDof, minimum_degrees_of_freedom,
                                        loss_data_type, reported_loss_shape);
                builder.location(location).logScale(logScale).labels(labels).lossDataType(loss_data_type.value_or(location.getValuesDataType()));
                if (learnedDof.has_value()) builder.logDegreesOfFreedom(learnedDof.value());
                maybeSetRaggedExampleWeights(builder, location, logScale, labels, learnedDof, example_weights);
            } else {
                throw nb::type_error("StudentTNLLLoss location, log_scale, labels, and learned_log_degrees_of_freedom must all use the same dense/ragged tensor kind.");
            }

            // The public validation above deliberately runs before these builder setters.
            // Builder checks are internal invariants and surface as RuntimeError through
            // nanobind; invalid user parameters must instead raise Python ValueError.
            builder.minimumDegreesOfFreedom(minimum_degrees_of_freedom);
            if (degrees_of_freedom.has_value()) builder.degreesOfFreedom(degrees_of_freedom.value());
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
        "learned_log_degrees_of_freedom"_a = nb::none(),
        "loss_weight"_a.none() = nb::none(),
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged Student-t negative log-likelihood loss.)nbdoc");

    lossClass.def("get_predictions", [](const StudentTNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    lossClass.def("get_labels", [](const StudentTNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    lossClass.def("get_raw_loss", [](const StudentTNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    lossClass.def("get_loss", [](const StudentTNLLLoss& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    lossClass.def_prop_ro("is_ragged", &StudentTNLLLoss::isRagged);
    lossClass.def_prop_ro("location", [](const StudentTNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.getLocation());
    });
    lossClass.def_prop_ro("log_scale", [](const StudentTNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLogScale());
        return nb::cast(self.getLogScale());
    });
    lossClass.def_prop_ro("degrees_of_freedom", &StudentTNLLLoss::getDegreesOfFreedom);
    lossClass.def_prop_ro("learned_log_degrees_of_freedom", [](const StudentTNLLLoss& self) -> nb::object {
        if (self.isRagged()) {
            auto value = self.getRaggedLearnedLogDegreesOfFreedom();
            return value.has_value() ? nb::cast(value.value()) : nb::none();
        }
        auto value = self.getLearnedLogDegreesOfFreedom();
        return value.has_value() ? nb::cast(value.value()) : nb::none();
    });
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

For dense inputs, example_weights may be [1] for per-example weighting or may
match location for elementwise weighting. Ragged inputs support dense [1]
per-row example weights only. Weights scale the raw NLL and all learned-parameter
gradients before loss-shape reduction.
)nbdoc";
}
