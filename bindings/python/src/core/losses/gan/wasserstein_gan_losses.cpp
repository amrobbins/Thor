#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticGradientPenaltyLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANGeneratorLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace std;

using namespace Thor;

using DataType = ThorImplementation::DataType;
using LossShape = Loss::LossShape;

namespace {
void validateReportedLossShape(LossShape reported_loss_shape, const string& loss_name) {
    if (reported_loss_shape != LossShape::BATCH && reported_loss_shape != LossShape::CLASSWISE &&
        reported_loss_shape != LossShape::ELEMENTWISE && reported_loss_shape != LossShape::RAW) {
        string error_message =
            "Invalid value " + to_string((int)reported_loss_shape) + " passed for enum reported_loss_shape to " + loss_name + ".";
        throw nb::value_error(error_message.c_str());
    }
}

void setReportedLossShape(WassersteinGANCriticLoss::Builder& builder, LossShape reported_loss_shape) {
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

void setReportedLossShape(WassersteinGANGeneratorLoss::Builder& builder, LossShape reported_loss_shape) {
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

void setReportedLossShape(WassersteinGANCriticGradientPenaltyLoss::Builder& builder, LossShape reported_loss_shape) {
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

void validateFloatingDType(const string& loss_name, const string& tensor_name, Tensor tensor) {
    if (tensor.getDataType() != DataType::FP16 && tensor.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": " + tensor_name + " must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
}

void validateLossDataType(const string& loss_name, optional<DataType> loss_data_type, DataType default_dtype) {
    DataType effectiveLossDataType = loss_data_type.value_or(default_dtype);
    if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
}

void validateWassersteinGANCriticLossArguments(const string& loss_name,
                                               Tensor real_scores,
                                               Tensor fake_scores,
                                               optional<DataType> loss_data_type,
                                               LossShape reported_loss_shape) {
    if (real_scores.getDimensions().size() != 1 || real_scores.getDimensions()[0] == 0) {
        string error_message = loss_name + ": real_scores must be a non-empty 1D score tensor but real_scores is " +
                               real_scores.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (fake_scores.getDimensions() != real_scores.getDimensions()) {
        string error_message = loss_name + ": fake_scores dimensions " + fake_scores.getDescriptorString() +
                               " must match real_scores dimensions " + real_scores.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (real_scores == fake_scores) {
        string error_message = loss_name + ": real_scores and fake_scores must be distinct tensors";
        throw nb::value_error(error_message.c_str());
    }
    validateFloatingDType(loss_name, "real_scores", real_scores);
    if (fake_scores.getDataType() != real_scores.getDataType()) {
        string error_message = loss_name + ": real_scores and fake_scores must use the same fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    validateLossDataType(loss_name, loss_data_type, real_scores.getDataType());
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void validateWassersteinGANGeneratorLossArguments(const string& loss_name,
                                                  Tensor fake_scores,
                                                  optional<DataType> loss_data_type,
                                                  LossShape reported_loss_shape) {
    if (fake_scores.getDimensions().size() != 1 || fake_scores.getDimensions()[0] == 0) {
        string error_message = loss_name + ": fake_scores must be a non-empty 1D score tensor but fake_scores is " +
                               fake_scores.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    validateFloatingDType(loss_name, "fake_scores", fake_scores);
    validateLossDataType(loss_name, loss_data_type, fake_scores.getDataType());
    validateReportedLossShape(reported_loss_shape, loss_name);
}

void validateWassersteinGANCriticGradientPenaltyLossArguments(const string& loss_name,
                                                              Tensor real_scores,
                                                              Tensor fake_scores,
                                                              Tensor sample_gradients,
                                                              float gradient_penalty_weight,
                                                              float target_gradient_norm,
                                                              float eps,
                                                              optional<DataType> loss_data_type,
                                                              LossShape reported_loss_shape) {
    if (real_scores.getDimensions().size() != 1 || real_scores.getDimensions()[0] != 1) {
        string error_message = loss_name + ": real_scores must be a scalar 1D score tensor [1] for WGAN-GP but real_scores is " +
                               real_scores.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (fake_scores.getDimensions() != real_scores.getDimensions()) {
        string error_message = loss_name + ": fake_scores dimensions " + fake_scores.getDescriptorString() +
                               " must match real_scores dimensions " + real_scores.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (sample_gradients.getDimensions().empty()) {
        string error_message = loss_name + ": sample_gradients must have at least one feature dimension";
        throw nb::value_error(error_message.c_str());
    }
    if (real_scores == fake_scores || real_scores == sample_gradients || fake_scores == sample_gradients) {
        string error_message = loss_name + ": real_scores, fake_scores, and sample_gradients must be distinct tensors";
        throw nb::value_error(error_message.c_str());
    }
    validateFloatingDType(loss_name, "real_scores", real_scores);
    if (fake_scores.getDataType() != real_scores.getDataType()) {
        string error_message = loss_name + ": real_scores and fake_scores must use the same fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    validateFloatingDType(loss_name, "sample_gradients", sample_gradients);
    if (gradient_penalty_weight < 0.0f) {
        string error_message = loss_name + ": gradient_penalty_weight must be non-negative";
        throw nb::value_error(error_message.c_str());
    }
    if (target_gradient_norm <= 0.0f) {
        string error_message = loss_name + ": target_gradient_norm must be greater than zero";
        throw nb::value_error(error_message.c_str());
    }
    if (eps <= 0.0f) {
        string error_message = loss_name + ": eps must be greater than zero";
        throw nb::value_error(error_message.c_str());
    }
    validateLossDataType(loss_name, loss_data_type, real_scores.getDataType());
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_wasserstein_gan_losses(nb::module_& losses) {
    auto critic_loss = nb::class_<WassersteinGANCriticLoss, Loss>(losses, "WassersteinGANCriticLoss");
    critic_loss.attr("__module__") = "thor.losses.gan";

    critic_loss.def(
        "__init__",
        [](WassersteinGANCriticLoss* self,
           Network& network,
           Tensor real_scores,
           Tensor fake_scores,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "WassersteinGANCriticLoss instance";
            validateWassersteinGANCriticLossArguments(loss_name, real_scores, fake_scores, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(real_scores.getDataType());
            WassersteinGANCriticLoss::Builder builder;
            builder.network(network).realScores(real_scores).fakeScores(fake_scores).lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            WassersteinGANCriticLoss built = builder.build();

            new (self) WassersteinGANCriticLoss(std::move(built));
        },
        "network"_a,
        "real_scores"_a,
        "fake_scores"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct the critic-side Wasserstein GAN loss over real and fake critic scores.)nbdoc");

    critic_loss.def("get_real_scores", &WassersteinGANCriticLoss::getRealScores);
    critic_loss.def("get_fake_scores", &WassersteinGANCriticLoss::getFakeScores);
    critic_loss.attr("__doc__") = R"nbdoc(
Critic-side Wasserstein GAN loss.

The raw elementwise loss is:

    fake_scores - real_scores

Minimizing this critic loss maximizes the Wasserstein distance estimate. For
original WGAN weight clipping, use a post-update parameter constraint hook; this
loss intentionally only models the differentiable score objective.
)nbdoc";

    auto generator_loss = nb::class_<WassersteinGANGeneratorLoss, Loss>(losses, "WassersteinGANGeneratorLoss");
    generator_loss.attr("__module__") = "thor.losses.gan";

    generator_loss.def(
        "__init__",
        [](WassersteinGANGeneratorLoss* self,
           Network& network,
           Tensor fake_scores,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "WassersteinGANGeneratorLoss instance";
            validateWassersteinGANGeneratorLossArguments(loss_name, fake_scores, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(fake_scores.getDataType());
            WassersteinGANGeneratorLoss::Builder builder;
            builder.network(network).fakeScores(fake_scores).lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            WassersteinGANGeneratorLoss built = builder.build();

            new (self) WassersteinGANGeneratorLoss(std::move(built));
        },
        "network"_a,
        "fake_scores"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct the generator-side Wasserstein GAN loss over fake critic scores.)nbdoc");

    generator_loss.def("get_fake_scores", &WassersteinGANGeneratorLoss::getFakeScores);
    generator_loss.attr("__doc__") = R"nbdoc(
Generator-side Wasserstein GAN loss.

The raw elementwise loss is:

    -fake_scores
)nbdoc";

    auto gp_loss = nb::class_<WassersteinGANCriticGradientPenaltyLoss, Loss>(losses, "WassersteinGANCriticGradientPenaltyLoss");
    gp_loss.attr("__module__") = "thor.losses.gan";

    gp_loss.def(
        "__init__",
        [](WassersteinGANCriticGradientPenaltyLoss* self,
           Network& network,
           Tensor real_scores,
           Tensor fake_scores,
           Tensor sample_gradients,
           float gradient_penalty_weight,
           float target_gradient_norm,
           float eps,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           std::optional<float> loss_weight) {
            const string loss_name = "WassersteinGANCriticGradientPenaltyLoss instance";
            validateWassersteinGANCriticGradientPenaltyLossArguments(loss_name,
                                                                     real_scores,
                                                                     fake_scores,
                                                                     sample_gradients,
                                                                     gradient_penalty_weight,
                                                                     target_gradient_norm,
                                                                     eps,
                                                                     loss_data_type,
                                                                     reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(real_scores.getDataType());
            WassersteinGANCriticGradientPenaltyLoss::Builder builder;
            builder.network(network)
                .realScores(real_scores)
                .fakeScores(fake_scores)
                .sampleGradients(sample_gradients)
                .gradientPenaltyWeight(gradient_penalty_weight)
                .targetGradientNorm(target_gradient_norm)
                .epsilon(eps)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            WassersteinGANCriticGradientPenaltyLoss built = builder.build();

            new (self) WassersteinGANCriticGradientPenaltyLoss(std::move(built));
        },
        "network"_a,
        "real_scores"_a,
        "fake_scores"_a,
        "sample_gradients"_a,
        "gradient_penalty_weight"_a = 10.0f,
        "target_gradient_norm"_a = 1.0f,
        "eps"_a = 1.0e-12f,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct the WGAN-GP critic loss over scalar scores and a materialized per-sample input-gradient tensor.)nbdoc");

    gp_loss.def("get_real_scores", &WassersteinGANCriticGradientPenaltyLoss::getRealScores);
    gp_loss.def("get_fake_scores", &WassersteinGANCriticGradientPenaltyLoss::getFakeScores);
    gp_loss.def("get_sample_gradients", &WassersteinGANCriticGradientPenaltyLoss::getSampleGradients);
    gp_loss.def_prop_ro("gradient_penalty_weight", &WassersteinGANCriticGradientPenaltyLoss::getGradientPenaltyWeight);
    gp_loss.def_prop_ro("target_gradient_norm", &WassersteinGANCriticGradientPenaltyLoss::getTargetGradientNorm);
    gp_loss.def_prop_ro("eps", &WassersteinGANCriticGradientPenaltyLoss::getEps);
    gp_loss.attr("__doc__") = R"nbdoc(
Critic-side Wasserstein GAN loss with gradient penalty.

The raw per-example scalar loss is:

    fake_scores - real_scores + gradient_penalty_weight * (||sample_gradients||_2 - target_gradient_norm)^2

``sample_gradients`` is expected to be the gradient of the critic output with
respect to interpolated input samples. This class deliberately consumes that
gradient tensor; trainer/autodiff scaffolding must produce it as a differentiable
first-class tensor for end-to-end WGAN-GP training.
)nbdoc";
}
