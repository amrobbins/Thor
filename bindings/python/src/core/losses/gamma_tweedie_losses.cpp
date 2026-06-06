#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <cmath>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/TweedieLoss.h"
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
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::CLASSWISE &&
        reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

bool isFloatingDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

void setReportedLossShape(GammaNLLLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::CLASSWISE) {
        builder.reportsPerOutputLoss();
    } else if (reported_loss_shape == LossShape::ELEMENTWISE) {
        builder.reportsElementwiseLoss();
    } else {
        THOR_THROW_IF_FALSE(reported_loss_shape == LossShape::RAW);
        builder.reportsRawLoss();
    }
}

void setReportedLossShape(TweedieLoss::Builder &builder, LossShape reported_loss_shape) {
    if (reported_loss_shape == LossShape::BATCH) {
        builder.reportsBatchLoss();
    } else if (reported_loss_shape == LossShape::CLASSWISE) {
        builder.reportsPerOutputLoss();
    } else if (reported_loss_shape == LossShape::ELEMENTWISE) {
        builder.reportsElementwiseLoss();
    } else {
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
    if (predictions.getDimensions().size() != 1) {
        string error_message = loss_name + ": predictions must be a 1 dimensional mean tensor but predictions is " +
                               predictions.getDescriptorString();
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
}  // namespace

void bind_gamma_tweedie_losses(nb::module_ &losses) {
    auto gamma_nll_loss = nb::class_<GammaNLLLoss, Loss>(losses, "GammaNLLLoss");
    gamma_nll_loss.attr("__module__") = "thor.losses";

    gamma_nll_loss.def(
        "__init__",
        [](GammaNLLLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float eps,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "GammaNLLLoss instance";
            validateMeanTargetLossArguments(loss_name, predictions, labels, loss_data_type, reported_loss_shape, eps);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            GammaNLLLoss::Builder builder;
            builder.network(network).predictions(predictions).labels(labels).eps(eps).lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            GammaNLLLoss built = builder.build();

            new (self) GammaNLLLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "eps"_a = 1.0e-6f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a Gamma negative log-likelihood loss.)nbdoc");

    gamma_nll_loss.def_prop_ro("eps", &GammaNLLLoss::getEps);
    gamma_nll_loss.attr("__doc__") = R"nbdoc(
Gamma negative log-likelihood loss for positive mean predictions.

Predictions are per-element means and labels are targets. Predictions are
clamped to at least eps for numerical stability. Target-independent constants
are omitted. The raw loss is:

    log(max(predictions, eps)) + labels / max(predictions, eps)
)nbdoc";

    auto tweedie_loss = nb::class_<TweedieLoss, Loss>(losses, "TweedieLoss");
    tweedie_loss.attr("__module__") = "thor.losses";

    tweedie_loss.def(
        "__init__",
        [](TweedieLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           float power,
           float eps,
           optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "TweedieLoss instance";
            validateTweedieArguments(loss_name, predictions, labels, power, loss_data_type, reported_loss_shape, eps);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            TweedieLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .power(power)
                .eps(eps)
                .lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            TweedieLoss built = builder.build();

            new (self) TweedieLoss(std::move(built));
        },
        "network"_a,
        "predictions"_a,
        "labels"_a,
        "power"_a = 1.5f,
        "eps"_a = 1.0e-6f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct a Tweedie deviance loss.)nbdoc");

    tweedie_loss.def_prop_ro("power", &TweedieLoss::getPower);
    tweedie_loss.def_prop_ro("eps", &TweedieLoss::getEps);
    tweedie_loss.attr("__doc__") = R"nbdoc(
Tweedie unit deviance loss for positive mean predictions.

Predictions are per-element means and labels are targets. Predictions are
clamped to at least eps for numerical stability. power selects the Tweedie
variance power. Special cases are handled directly for power 0, 1, and 2.
)nbdoc";
}
