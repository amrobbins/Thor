#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/LaplaceNLLLoss.h"
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

void setReportedLossShape(LaplaceNLLLoss::Builder& builder, LossShape shape) {
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
                       Tensor scale,
                       Tensor labels,
                       optional<DataType> lossDataType,
                       LossShape reportedLossShape,
                       float eps) {
    const string lossName = "LaplaceNLLLoss instance";
    if (location.getDimensions().empty())
        throw nb::value_error("LaplaceNLLLoss instance: location must have at least one non-batch dimension");
    if (scale.getDimensions() != location.getDimensions()) {
        string message = lossName + ": scale dimensions " + scale.getDescriptorString() +
                         " must match location dimensions " + location.getDescriptorString();
        throw nb::value_error(message.c_str());
    }
    if (labels.getDimensions() != location.getDimensions()) {
        string message = lossName + ": labels dimensions " + labels.getDescriptorString() +
                         " must match location dimensions " + location.getDescriptorString();
        throw nb::value_error(message.c_str());
    }
    if (!isFloatingDType(location.getDataType()))
        throw nb::value_error("LaplaceNLLLoss instance: location must use fp16 or fp32 dtype.");
    if (!isFloatingDType(scale.getDataType()))
        throw nb::value_error("LaplaceNLLLoss instance: scale must use fp16 or fp32 dtype.");
    if (!isFloatingDType(labels.getDataType()))
        throw nb::value_error("LaplaceNLLLoss instance: labels must use fp16 or fp32 dtype.");
    DataType effectiveLossDataType = lossDataType.value_or(location.getDataType());
    if (!isFloatingDType(effectiveLossDataType))
        throw nb::value_error("LaplaceNLLLoss instance: loss_data_type must be fp16 or fp32.");
    if (eps <= 0.0f)
        throw nb::value_error("LaplaceNLLLoss instance: eps must be greater than zero.");
    validateReportedLossShape(reportedLossShape, lossName);
}

void maybeSetExampleWeights(LaplaceNLLLoss::Builder& builder,
                            Tensor location,
                            Tensor scale,
                            Tensor labels,
                            optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value())
        return;
    if (exampleWeights.value() == location || exampleWeights.value() == scale || exampleWeights.value() == labels)
        throw nb::value_error("LaplaceNLLLoss instance: example_weights must be distinct from location, scale, and labels.");
    if (!isFloatingDType(exampleWeights.value().getDataType()))
        throw nb::value_error("LaplaceNLLLoss instance: example_weights must use fp16 or fp32 dtype.");
    const vector<uint64_t>& dims = exampleWeights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != location.getDimensions()) {
        string message =
            "LaplaceNLLLoss instance: example_weights dimensions must be [1] for per-example weights or match location. "
            "example_weights tensor is " +
            exampleWeights.value().getDescriptorString() + "; location tensor is " + location.getDescriptorString() + ".";
        throw nb::value_error(message.c_str());
    }
    builder.exampleWeights(exampleWeights.value());
}

void validateRaggedArguments(const RaggedTensor& location,
                            const RaggedTensor& scale,
                            const RaggedTensor& labels,
                            optional<DataType> lossDataType,
                            LossShape reportedLossShape,
                            float eps) {
    const string lossName = "LaplaceNLLLoss instance";
    if (!isFloatingDType(location.getValuesDataType()))
        throw nb::value_error("LaplaceNLLLoss instance: location must use fp16 or fp32 dtype.");
    if (!isFloatingDType(scale.getValuesDataType()))
        throw nb::value_error("LaplaceNLLLoss instance: scale must use fp16 or fp32 dtype.");
    if (!isFloatingDType(labels.getValuesDataType()))
        throw nb::value_error("LaplaceNLLLoss instance: labels must use fp16 or fp32 dtype.");
    if (location.getOffsets() != scale.getOffsets() || location.getOffsets() != labels.getOffsets())
        throw nb::value_error("LaplaceNLLLoss instance: ragged location, scale, and labels must use the exact same row partition tensor.");
    if (location.getBatchSize() != scale.getBatchSize() || location.getBatchSize() != labels.getBatchSize() ||
        location.getMaxTotalValues() != scale.getMaxTotalValues() || location.getMaxTotalValues() != labels.getMaxTotalValues() ||
        location.getTrailingDimensions() != scale.getTrailingDimensions() ||
        location.getTrailingDimensions() != labels.getTrailingDimensions())
        throw nb::value_error("LaplaceNLLLoss instance: ragged location, scale, and labels must have identical value geometry.");
    const DataType effectiveLossDataType = lossDataType.value_or(location.getValuesDataType());
    if (!isFloatingDType(effectiveLossDataType))
        throw nb::value_error("LaplaceNLLLoss instance: loss_data_type must be fp16 or fp32.");
    if (eps <= 0.0f) throw nb::value_error("LaplaceNLLLoss instance: eps must be greater than zero.");
    validateReportedLossShape(reportedLossShape, lossName);
    if (reportedLossShape == LossShape::PER_OUTPUT)
        throw nb::value_error("LaplaceNLLLoss instance: reported_loss_shape per_output is undefined for ragged sequences.");
}

void maybeSetRaggedExampleWeights(LaplaceNLLLoss::Builder& builder,
                                  const RaggedTensor& location,
                                  const RaggedTensor& scale,
                                  const RaggedTensor& labels,
                                  optional<Tensor> exampleWeights) {
    if (!exampleWeights.has_value()) return;
    if (exampleWeights.value() == location.getValues() || exampleWeights.value() == scale.getValues() ||
        exampleWeights.value() == labels.getValues())
        throw nb::value_error("LaplaceNLLLoss instance: example_weights must be distinct from location, scale, and labels values.");
    if (!isFloatingDType(exampleWeights->getDataType()))
        throw nb::value_error("LaplaceNLLLoss instance: example_weights must use fp16 or fp32 dtype.");
    if (exampleWeights->getDimensions() != vector<uint64_t>{1})
        throw nb::value_error("LaplaceNLLLoss instance: ragged example_weights dimensions must be [1] for one scalar weight per logical row.");
    builder.exampleWeights(exampleWeights.value());
}
}  // namespace

void bind_laplace_nll_loss(nb::module_& losses) {
    auto lossClass = nb::class_<LaplaceNLLLoss, Loss>(losses, "LaplaceNLLLoss");
    lossClass.attr("__module__") = "thor.losses.distribution";

    lossClass.def(
        "__init__",
        [](LaplaceNLLLoss* self,
           Network& network,
           nb::object locationObject,
           nb::object scaleObject,
           nb::object labelsObject,
           bool log_scale,
           float eps,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           optional<float> loss_weight,
           optional<Tensor> example_weights) {
            if (eps <= 0.0f) throw nb::value_error("LaplaceNLLLoss instance: eps must be greater than zero");
            LaplaceNLLLoss::Builder builder;
            builder.network(network).logScale(log_scale).eps(eps).lossWeight(loss_weight.value_or(1.0f));

            if (nb::isinstance<Tensor>(locationObject) && nb::isinstance<Tensor>(scaleObject) &&
                nb::isinstance<Tensor>(labelsObject)) {
                Tensor location = nb::cast<Tensor>(locationObject);
                Tensor scale = nb::cast<Tensor>(scaleObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                validateArguments(location, scale, labels, loss_data_type, reported_loss_shape, eps);
                builder.location(location)
                    .scale(scale)
                    .labels(labels)
                    .lossDataType(loss_data_type.value_or(location.getDataType()));
                maybeSetExampleWeights(builder, location, scale, labels, example_weights);
            } else if (nb::isinstance<RaggedTensor>(locationObject) && nb::isinstance<RaggedTensor>(scaleObject) &&
                       nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor location = nb::cast<RaggedTensor>(locationObject);
                RaggedTensor scale = nb::cast<RaggedTensor>(scaleObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                validateRaggedArguments(location, scale, labels, loss_data_type, reported_loss_shape, eps);
                builder.location(location)
                    .scale(scale)
                    .labels(labels)
                    .lossDataType(loss_data_type.value_or(location.getValuesDataType()));
                maybeSetRaggedExampleWeights(builder, location, scale, labels, example_weights);
            } else {
                throw nb::type_error(
                    "LaplaceNLLLoss location, scale, and labels must all be thor.Tensor or all be thor.RaggedTensor.");
            }
            setReportedLossShape(builder, reported_loss_shape);
            LaplaceNLLLoss built = builder.build();
            new (self) LaplaceNLLLoss(std::move(built));
        },
        "network"_a,
        "location"_a,
        "scale"_a,
        "labels"_a,
        "log_scale"_a = true,
        "eps"_a = 1.0e-8f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged Laplace negative log-likelihood loss.)nbdoc");

    lossClass.def("get_predictions", [](const LaplaceNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    lossClass.def("get_labels", [](const LaplaceNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    lossClass.def("get_raw_loss", [](const LaplaceNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    lossClass.def("get_loss", [](const LaplaceNLLLoss& self) -> nb::object {
        if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss());
        return nb::cast(self.Loss::getLoss());
    });
    lossClass.def_prop_ro("is_ragged", &LaplaceNLLLoss::isRagged);
    lossClass.def_prop_ro("location", [](const LaplaceNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.getLocation());
    });
    lossClass.def_prop_ro("scale", [](const LaplaceNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedScale());
        return nb::cast(self.getScale());
    });
    lossClass.def_prop_ro("log_scale", &LaplaceNLLLoss::getLogScale);
    lossClass.def_prop_ro("eps", &LaplaceNLLLoss::getEps);

    lossClass.attr("__doc__") = R"nbdoc(
Laplace negative log-likelihood using location and scale parameters.

For location m and scale b > 0, the per-element negative log-likelihood is:

    log(2 * b) + abs(target - m) / b

By default scale contains log(b), allowing an unconstrained network head. Set
log_scale=False to supply positive scale directly; direct scale is floored by
eps for numerical stability.

example_weights may be [1] for per-example weighting or may match location for
elementwise weighting. Weights scale the raw NLL and both learned-parameter
gradients before loss-shape reduction.
)nbdoc";
}
