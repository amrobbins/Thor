#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <cmath>
#include <optional>
#include <vector>

#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/TweedieLoss.h"
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
void validateReportedLossShape(LossShape reported_loss_shape, const string &loss_name) {
    if (reported_loss_shape != LossShape::NONE && reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::PER_OUTPUT &&
        reported_loss_shape != LossShape::PER_EXAMPLE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

bool isFloatingDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

void setReportedLossShape(GammaNLLLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::NONE)
        builder.reportsNoLoss();
    else if (reported_loss_shape == LossShape::BATCH)
        builder.reportsBatchLoss();
    else if (reported_loss_shape == LossShape::PER_OUTPUT)
        builder.reportsPerOutputLoss();
    else if (reported_loss_shape == LossShape::PER_EXAMPLE)
        builder.reportsPerExampleLoss();
    else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void setReportedLossShape(TweedieLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::NONE)
        builder.reportsNoLoss();
    else if (reported_loss_shape == LossShape::BATCH)
        builder.reportsBatchLoss();
    else if (reported_loss_shape == LossShape::PER_OUTPUT)
        builder.reportsPerOutputLoss();
    else if (reported_loss_shape == LossShape::PER_EXAMPLE)
        builder.reportsPerExampleLoss();
    else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void validateMeanTargetLossArguments(const string &loss_name,
                                     Tensor predictions,
                                     Tensor labels,
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
    if (!isFloatingDType(predictions.getDataType())) {
        string error_message = loss_name + ": predictions must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (!isFloatingDType(labels.getDataType())) {
        string error_message = loss_name + ": labels must use fp16 or fp32 dtype";
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

void validateTweedieArguments(const string &loss_name,
                              Tensor predictions,
                              Tensor labels,
                              float power,
                              optional<DataType> loss_data_type,
                              LossShape reported_loss_shape,
                              float eps) {
    validateMeanTargetLossArguments(loss_name, predictions, labels, loss_data_type, reported_loss_shape, eps);
    if (!std::isfinite(power)) {
        string error_message = loss_name + ": power must be finite";
        throw nb::value_error(error_message.c_str());
    }
}

void maybeSetGammaDispersion(GammaNLLLoss::Builder &builder,
                             Tensor predictions,
                             Tensor labels,
                             optional<Tensor> dispersion,
                             bool log_dispersion) {
    if (!dispersion.has_value()) {
        if (log_dispersion)
            throw nb::value_error("GammaNLLLoss instance: log_dispersion=True requires dispersion.");
        return;
    }
    if (dispersion.value() == predictions || dispersion.value() == labels)
        throw nb::value_error("GammaNLLLoss instance: dispersion must be distinct from predictions and labels.");
    if (dispersion.value().getDimensions() != predictions.getDimensions()) {
        string error_message = "GammaNLLLoss instance: dispersion dimensions " + dispersion.value().getDescriptorString() +
                               " must match predictions dimensions " + predictions.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (!isFloatingDType(dispersion.value().getDataType()))
        throw nb::value_error("GammaNLLLoss instance: dispersion must use fp16 or fp32 dtype.");
    builder.dispersion(dispersion.value()).logDispersion(log_dispersion);
}

void maybeSetGammaExampleWeights(GammaNLLLoss::Builder &builder,
                                 Tensor predictions,
                                 Tensor labels,
                                 optional<Tensor> dispersion,
                                 optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions || example_weights.value() == labels ||
        (dispersion.has_value() && example_weights.value() == dispersion.value()))
        throw nb::value_error("GammaNLLLoss instance: example_weights must be distinct from predictions, labels, and dispersion.");
    if (!isFloatingDType(example_weights.value().getDataType()))
        throw nb::value_error("GammaNLLLoss instance: example_weights must use fp16 or fp32 dtype.");
    const vector<uint64_t>& dims = example_weights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        string error_message =
            "GammaNLLLoss instance: example_weights dimensions must be [1] for per-example weights or match predictions. "
            "example_weights tensor is " +
            example_weights.value().getDescriptorString() + "; predictions tensor is " + predictions.getDescriptorString() + ".";
        throw nb::value_error(error_message.c_str());
    }
    builder.exampleWeights(example_weights.value());
}

void maybeSetTweedieExampleWeights(TweedieLoss::Builder &builder,
                                   Tensor predictions,
                                   Tensor labels,
                                   optional<Tensor> example_weights) {
    if (!example_weights.has_value())
        return;
    if (example_weights.value() == predictions || example_weights.value() == labels)
        throw nb::value_error("TweedieLoss instance: example_weights must be distinct from predictions and labels.");
    if (!isFloatingDType(example_weights.value().getDataType()))
        throw nb::value_error("TweedieLoss instance: example_weights must use fp16 or fp32 dtype.");
    const vector<uint64_t>& dims = example_weights.value().getDimensions();
    if (dims != vector<uint64_t>{1} && dims != predictions.getDimensions()) {
        string error_message =
            "TweedieLoss instance: example_weights dimensions must be [1] for per-example weights or match predictions. "
            "example_weights tensor is " +
            example_weights.value().getDescriptorString() + "; predictions tensor is " + predictions.getDescriptorString() + ".";
        throw nb::value_error(error_message.c_str());
    }
    builder.exampleWeights(example_weights.value());
}
void validateRaggedMeanTarget(const string& loss_name, const RaggedTensor& predictions, const RaggedTensor& labels,
                              optional<DataType> loss_data_type, LossShape reported_loss_shape, float eps) {
    if (!isFloatingDType(predictions.getValuesDataType())) throw nb::value_error((loss_name + ": predictions must use fp16 or fp32 dtype").c_str());
    if (!isFloatingDType(labels.getValuesDataType())) throw nb::value_error((loss_name + ": labels must use fp16 or fp32 dtype").c_str());
    if (predictions.getOffsets() != labels.getOffsets()) throw nb::value_error((loss_name + ": ragged predictions and labels must use the exact same row partition tensor.").c_str());
    if (predictions.getBatchSize() != labels.getBatchSize() || predictions.getMaxTotalValues() != labels.getMaxTotalValues() || predictions.getTrailingDimensions() != labels.getTrailingDimensions())
        throw nb::value_error((loss_name + ": ragged predictions and labels must have identical value geometry.").c_str());
    if (reported_loss_shape == LossShape::PER_OUTPUT) throw nb::value_error((loss_name + ": per_output reporting is undefined for ragged predictions.").c_str());
    DataType effective = loss_data_type.value_or(predictions.getValuesDataType());
    if (!isFloatingDType(effective)) throw nb::value_error((loss_name + ": loss_data_type must be fp16 or fp32").c_str());
    if (eps <= 0.0f) throw nb::value_error((loss_name + ": eps must be greater than zero").c_str());
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void maybeSetRaggedDistributionWeights(const string& loss_name, optional<Tensor> example_weights,
                                       const RaggedTensor& predictions, const RaggedTensor& labels,
                                       const optional<RaggedTensor>& third, auto& builder) {
    if (!example_weights.has_value()) return;
    if (example_weights.value() == predictions.getValues() || example_weights.value() == labels.getValues() ||
        (third.has_value() && example_weights.value() == third->getValues()))
        throw nb::value_error((loss_name + ": example_weights must be distinct from ragged value inputs.").c_str());
    if (!isFloatingDType(example_weights->getDataType())) throw nb::value_error((loss_name + ": example_weights must use fp16 or fp32 dtype.").c_str());
    if (example_weights->getDimensions() != vector<uint64_t>{1}) throw nb::value_error((loss_name + ": ragged example_weights dimensions must be [1] for one scalar weight per logical row.").c_str());
    builder.exampleWeights(example_weights.value());
}

}  // namespace

void bind_gamma_tweedie_losses(nb::module_ &losses) {
    auto gamma_nll_loss = nb::class_<GammaNLLLoss, Loss>(losses, "GammaNLLLoss");
    gamma_nll_loss.attr("__module__") = "thor.losses.distribution";

    gamma_nll_loss.def(
        "__init__",
        [](GammaNLLLoss *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           float eps,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           optional<float> loss_weight,
           nb::object dispersionObject,
           bool log_mean,
           bool log_dispersion,
           optional<Tensor> example_weights) {
            const string loss_name = "GammaNLLLoss instance";
            if (eps <= 0.0f) throw nb::value_error("GammaNLLLoss instance: eps must be greater than zero");
            GammaNLLLoss::Builder builder;
            builder.network(network).logMean(log_mean).eps(eps).lossWeight(loss_weight.value_or(1.0f));
            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject);
                Tensor labels = nb::cast<Tensor>(labelsObject);
                validateMeanTargetLossArguments(loss_name, predictions, labels, loss_data_type, reported_loss_shape, eps);
                builder.predictions(predictions).labels(labels).lossDataType(loss_data_type.value_or(predictions.getDataType()));
                optional<Tensor> dispersion;
                if (!dispersionObject.is_none()) {
                    if (!nb::isinstance<Tensor>(dispersionObject)) throw nb::type_error("GammaNLLLoss dense predictions require dense dispersion.");
                    dispersion = nb::cast<Tensor>(dispersionObject);
                }
                maybeSetGammaDispersion(builder, predictions, labels, dispersion, log_dispersion);
                maybeSetGammaExampleWeights(builder, predictions, labels, dispersion, example_weights);
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject);
                RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                validateRaggedMeanTarget(loss_name, predictions, labels, loss_data_type, reported_loss_shape, eps);
                builder.predictions(predictions).labels(labels).lossDataType(loss_data_type.value_or(predictions.getValuesDataType()));
                optional<RaggedTensor> dispersion;
                if (!dispersionObject.is_none()) {
                    if (!nb::isinstance<RaggedTensor>(dispersionObject)) throw nb::type_error("GammaNLLLoss ragged predictions require ragged dispersion.");
                    dispersion = nb::cast<RaggedTensor>(dispersionObject);
                    if (!isFloatingDType(dispersion->getValuesDataType())) throw nb::value_error("GammaNLLLoss instance: dispersion must use fp16 or fp32 dtype.");
                    if (dispersion->getOffsets() != predictions.getOffsets()) throw nb::value_error("GammaNLLLoss instance: ragged dispersion must use the exact same row partition tensor as predictions.");
                    if (dispersion->getBatchSize() != predictions.getBatchSize() || dispersion->getMaxTotalValues() != predictions.getMaxTotalValues() || dispersion->getTrailingDimensions() != predictions.getTrailingDimensions())
                        throw nb::value_error("GammaNLLLoss instance: ragged dispersion must have identical value geometry to predictions.");
                    builder.dispersion(dispersion.value());
                } else if (log_dispersion) {
                    throw nb::value_error("GammaNLLLoss instance: log_dispersion=True requires dispersion.");
                }
                builder.logDispersion(log_dispersion);
                maybeSetRaggedDistributionWeights(loss_name, example_weights, predictions, labels, dispersion, builder);
            } else {
                throw nb::type_error("GammaNLLLoss predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }
            setReportedLossShape(builder, reported_loss_shape);
            GammaNLLLoss built = builder.build();
            new (self) GammaNLLLoss(std::move(built));
        },
        "network"_a, "predictions"_a, "labels"_a, "eps"_a = 1.0e-6f,
        "loss_data_type"_a.none() = nb::none(), "reported_loss_shape"_a = LossShape::BATCH, nb::kw_only(),
        "loss_weight"_a.none() = nb::none(), "dispersion"_a = nb::none(), "log_mean"_a = false,
        "log_dispersion"_a = false, "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged Gamma negative log-likelihood loss.)nbdoc");

    gamma_nll_loss.def("get_predictions", [](const GammaNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    gamma_nll_loss.def("get_labels", [](const GammaNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    gamma_nll_loss.def("get_raw_loss", [](const GammaNLLLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    gamma_nll_loss.def("get_loss", [](const GammaNLLLoss& self) -> nb::object { if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss()); return nb::cast(self.Loss::getLoss()); });
    gamma_nll_loss.def_prop_ro("is_ragged", &GammaNLLLoss::isRagged);

    gamma_nll_loss.def_prop_ro("dispersion", [](const GammaNLLLoss& self) -> nb::object {
        if (self.isRagged()) { auto d = self.getRaggedDispersion(); return d.has_value() ? nb::cast(d.value()) : nb::none(); }
        auto d = self.getDispersion(); return d.has_value() ? nb::cast(d.value()) : nb::none();
    });
    gamma_nll_loss.def_prop_ro("log_mean", &GammaNLLLoss::getLogMean);
    gamma_nll_loss.def_prop_ro("log_dispersion", &GammaNLLLoss::getLogDispersion);
    gamma_nll_loss.def_prop_ro("eps", &GammaNLLLoss::getEps);
    gamma_nll_loss.attr("__doc__") = R"nbdoc(
Gamma negative log-likelihood loss in mean/dispersion parameterization.

Without dispersion, this preserves Thor's legacy unit-dispersion (shape=1)
Gamma/exponential loss:

    log(mean) + labels / mean

When dispersion is supplied, Thor uses Var(Y) = dispersion * mean^2, with
concentration = 1 / dispersion and scale = mean * dispersion, and evaluates
the full per-element Gamma NLL. log_mean=True and log_dispersion=True allow
unconstrained network heads to supply log-parameters directly.

example_weights may be a [1] per-example weight tensor or match predictions for
elementwise weighting. Weights scale the raw loss and all learned-parameter
gradients before loss-shape reduction.
)nbdoc";

    auto tweedie_loss = nb::class_<TweedieLoss, Loss>(losses, "TweedieLoss");
    tweedie_loss.attr("__module__") = "thor.losses.distribution";

    tweedie_loss.def(
        "__init__",
        [](TweedieLoss *self,
           Network &network,
           nb::object predictionsObject,
           nb::object labelsObject,
           float power,
           float eps,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           optional<float> loss_weight,
           optional<Tensor> example_weights) {
            const string loss_name = "TweedieLoss instance";
            if (!std::isfinite(power)) throw nb::value_error("TweedieLoss instance: power must be finite");
            if (eps <= 0.0f) throw nb::value_error("TweedieLoss instance: eps must be greater than zero");
            TweedieLoss::Builder builder;
            builder.network(network).power(power).eps(eps).lossWeight(loss_weight.value_or(1.0f));
            if (nb::isinstance<Tensor>(predictionsObject) && nb::isinstance<Tensor>(labelsObject)) {
                Tensor predictions = nb::cast<Tensor>(predictionsObject); Tensor labels = nb::cast<Tensor>(labelsObject);
                validateTweedieArguments(loss_name, predictions, labels, power, loss_data_type, reported_loss_shape, eps);
                builder.predictions(predictions).labels(labels).lossDataType(loss_data_type.value_or(predictions.getDataType()));
                maybeSetTweedieExampleWeights(builder, predictions, labels, example_weights);
            } else if (nb::isinstance<RaggedTensor>(predictionsObject) && nb::isinstance<RaggedTensor>(labelsObject)) {
                RaggedTensor predictions = nb::cast<RaggedTensor>(predictionsObject); RaggedTensor labels = nb::cast<RaggedTensor>(labelsObject);
                validateRaggedMeanTarget(loss_name, predictions, labels, loss_data_type, reported_loss_shape, eps);
                builder.predictions(predictions).labels(labels).lossDataType(loss_data_type.value_or(predictions.getValuesDataType()));
                optional<RaggedTensor> noThird;
                maybeSetRaggedDistributionWeights(loss_name, example_weights, predictions, labels, noThird, builder);
            } else {
                throw nb::type_error("TweedieLoss predictions and labels must both be thor.Tensor or both be thor.RaggedTensor.");
            }
            setReportedLossShape(builder, reported_loss_shape);
            TweedieLoss built = builder.build();
            new (self) TweedieLoss(std::move(built));
        },
        "network"_a, "predictions"_a, "labels"_a, "power"_a = 1.5f, "eps"_a = 1.0e-6f,
        "loss_data_type"_a.none() = nb::none(), "reported_loss_shape"_a = LossShape::BATCH, nb::kw_only(),
        "loss_weight"_a.none() = nb::none(), "example_weights"_a.none() = nb::none(),
        R"nbdoc(Construct a dense or rank-1 ragged Tweedie unit-deviance loss.)nbdoc");

    tweedie_loss.def("get_predictions", [](const TweedieLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedPredictions());
        return nb::cast(self.Loss::getPredictions());
    });
    tweedie_loss.def("get_labels", [](const TweedieLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedLabels());
        return nb::cast(self.Loss::getLabels());
    });
    tweedie_loss.def("get_raw_loss", [](const TweedieLoss& self) -> nb::object {
        if (self.isRagged()) return nb::cast(self.getRaggedRawLoss());
        return nb::cast(self.Loss::getRawLoss());
    });
    tweedie_loss.def("get_loss", [](const TweedieLoss& self) -> nb::object { if (self.isRagged() && self.getLossShape() == LossShape::RAW) return nb::cast(self.getRaggedLoss()); return nb::cast(self.Loss::getLoss()); });
    tweedie_loss.def_prop_ro("is_ragged", &TweedieLoss::isRagged);

    tweedie_loss.def_prop_ro("power", &TweedieLoss::getPower);
    tweedie_loss.def_prop_ro("eps", &TweedieLoss::getEps);
    tweedie_loss.attr("__doc__") = R"nbdoc(
Tweedie unit-deviance loss for positive mean predictions.

This is a Tweedie unit deviance objective, not a full normalized Tweedie
negative log-likelihood. Predictions are per-element means and labels are
targets. Predictions are clamped to at least eps for numerical stability.
power selects the Tweedie variance power; powers 0, 1, and 2 use direct special
cases.
)nbdoc";
}
