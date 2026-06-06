#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

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
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::CLASSWISE &&
        reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(GaussianNLLLoss::Builder &builder, LossShape reported_loss_shape) {
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

bool isFloatingDType(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::FP32; }

void validateGaussianNLLLossArguments(const string &loss_name,
                                      Tensor predictions,
                                      Tensor labels,
                                      Tensor variance,
                                      std::optional<DataType> loss_data_type,
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
    if (variance.getDimensions() != predictions.getDimensions()) {
        string error_message = loss_name + ": variance dimensions " + variance.getDescriptorString() +
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
    if (!isFloatingDType(variance.getDataType())) {
        string error_message = loss_name + ": variance must use fp16 or fp32 dtype";
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
}  // namespace

void bind_gaussian_nll_loss(nb::module_ &losses) {
    auto gaussian_nll_loss = nb::class_<GaussianNLLLoss, Loss>(losses, "GaussianNLLLoss");
    gaussian_nll_loss.attr("__module__") = "thor.losses";

    gaussian_nll_loss.def(
        "__init__",
        [](GaussianNLLLoss *self,
           Network &network,
           Tensor predictions,
           Tensor labels,
           Tensor variance,
           bool full,
           float eps,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "GaussianNLLLoss instance";
            validateGaussianNLLLossArguments(loss_name, predictions, labels, variance, loss_data_type, reported_loss_shape, eps);

            DataType effectiveLossDataType = loss_data_type.value_or(predictions.getDataType());
            GaussianNLLLoss::Builder builder;
            builder.network(network)
                .predictions(predictions)
                .labels(labels)
                .variance(variance)
                .full(full)
                .eps(eps)
                .lossDataType(effectiveLossDataType);
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
        R"nbdoc(Construct a Gaussian negative log-likelihood loss.)nbdoc");

    gaussian_nll_loss.def_prop_ro("variance", &GaussianNLLLoss::getVariance);
    gaussian_nll_loss.def_prop_ro("full", &GaussianNLLLoss::getFull);
    gaussian_nll_loss.def_prop_ro("eps", &GaussianNLLLoss::getEps);

    gaussian_nll_loss.attr("__doc__") = R"nbdoc(
Gaussian negative log-likelihood loss.

Predictions are means, labels are targets, and variance is the per-element
variance tensor. Variance is clamped to at least eps for numerical stability.
The raw loss is:

    0.5 * (log(max(variance, eps)) + (predictions - labels)^2 / max(variance, eps))

If full is True, the constant 0.5 * log(2*pi) is included.
)nbdoc";
}
