#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/LSGANDiscriminatorLoss.h"
#include "DeepLearning/Api/Layers/Loss/LSGANGeneratorLoss.h"
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

void setReportedLossShape(LSGANDiscriminatorLoss::Builder& builder, LossShape reported_loss_shape) {
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

void setReportedLossShape(LSGANGeneratorLoss::Builder& builder, LossShape reported_loss_shape) {
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

void validateLSGANDiscriminatorLossArguments(const string& loss_name,
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

void validateLSGANGeneratorLossArguments(const string& loss_name,
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
}  // namespace

void bind_lsgan_losses(nb::module_& losses) {
    auto discriminator_loss = nb::class_<LSGANDiscriminatorLoss, Loss>(losses, "LSGANDiscriminatorLoss");
    discriminator_loss.attr("__module__") = "thor.losses.gan";

    discriminator_loss.def(
        "__init__",
        [](LSGANDiscriminatorLoss* self,
           Network& network,
           Tensor real_scores,
           Tensor fake_scores,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           float real_target,
           float fake_target,
           std::optional<float> loss_weight) {
            const string loss_name = "LSGANDiscriminatorLoss instance";
            validateLSGANDiscriminatorLossArguments(loss_name, real_scores, fake_scores, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(real_scores.getDataType());
            LSGANDiscriminatorLoss::Builder builder;
            builder.network(network)
                .realScores(real_scores)
                .fakeScores(fake_scores)
                .realTarget(real_target)
                .fakeTarget(fake_target)
                .lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            LSGANDiscriminatorLoss built = builder.build();

            new (self) LSGANDiscriminatorLoss(std::move(built));
        },
        "network"_a,
        "real_scores"_a,
        "fake_scores"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        "real_target"_a = 1.0f,
        "fake_target"_a = 0.0f,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct the discriminator-side least-squares GAN loss over real and fake discriminator scores.)nbdoc");

    discriminator_loss.def("get_real_scores", &LSGANDiscriminatorLoss::getRealScores);
    discriminator_loss.def("get_fake_scores", &LSGANDiscriminatorLoss::getFakeScores);
    discriminator_loss.def_prop_ro("real_target", &LSGANDiscriminatorLoss::getRealTarget);
    discriminator_loss.def_prop_ro("fake_target", &LSGANDiscriminatorLoss::getFakeTarget);
    discriminator_loss.attr("__doc__") = R"nbdoc(
Discriminator-side least-squares GAN loss.

The raw elementwise loss is:

    0.5 * ((real_scores - real_target)^2 + (fake_scores - fake_target)^2)

The default targets are real_target=1.0 and fake_target=0.0. Gradients are
produced for both real_scores and fake_scores. Use StopGradient on generated
samples feeding the discriminator when the discriminator step should not
backpropagate into the generator.
)nbdoc";

    auto generator_loss = nb::class_<LSGANGeneratorLoss, Loss>(losses, "LSGANGeneratorLoss");
    generator_loss.attr("__module__") = "thor.losses.gan";

    generator_loss.def(
        "__init__",
        [](LSGANGeneratorLoss* self,
           Network& network,
           Tensor fake_scores,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape,
           float target,
           std::optional<float> loss_weight) {
            const string loss_name = "LSGANGeneratorLoss instance";
            validateLSGANGeneratorLossArguments(loss_name, fake_scores, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(fake_scores.getDataType());
            LSGANGeneratorLoss::Builder builder;
            builder.network(network).fakeScores(fake_scores).target(target).lossDataType(effectiveLossDataType)
                .lossWeight(loss_weight.value_or(1.0f));
            setReportedLossShape(builder, reported_loss_shape);
            LSGANGeneratorLoss built = builder.build();

            new (self) LSGANGeneratorLoss(std::move(built));
        },
        "network"_a,
        "fake_scores"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        "target"_a = 1.0f,
        nb::kw_only(),
        "loss_weight"_a.none() = nb::none(),
        R"nbdoc(Construct the generator-side least-squares GAN loss over fake discriminator scores.)nbdoc");

    generator_loss.def("get_fake_scores", &LSGANGeneratorLoss::getFakeScores);
    generator_loss.def_prop_ro("target", &LSGANGeneratorLoss::getTarget);
    generator_loss.attr("__doc__") = R"nbdoc(
Generator-side least-squares GAN loss.

The raw elementwise loss is:

    0.5 * (fake_scores - target)^2

The default target is 1.0.
)nbdoc";
}
