#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/HingeGANDiscriminatorLoss.h"
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

void setReportedLossShape(HingeGANDiscriminatorLoss::Builder& builder, LossShape reported_loss_shape) {
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

void validateHingeGANDiscriminatorLossArguments(const string& loss_name,
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
    if (real_scores.getDataType() != DataType::FP16 && real_scores.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": real_scores must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    if (fake_scores.getDataType() != real_scores.getDataType()) {
        string error_message = loss_name + ": real_scores and fake_scores must use the same fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    DataType effectiveLossDataType = loss_data_type.value_or(real_scores.getDataType());
    if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_hinge_gan_discriminator_loss(nb::module_& losses) {
    auto hinge_gan_discriminator_loss = nb::class_<HingeGANDiscriminatorLoss, Loss>(losses, "HingeGANDiscriminatorLoss");
    hinge_gan_discriminator_loss.attr("__module__") = "thor.losses.gan";

    hinge_gan_discriminator_loss.def(
        "__init__",
        [](HingeGANDiscriminatorLoss* self,
           Network& network,
           Tensor real_scores,
           Tensor fake_scores,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "HingeGANDiscriminatorLoss instance";
            validateHingeGANDiscriminatorLossArguments(loss_name, real_scores, fake_scores, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(real_scores.getDataType());
            HingeGANDiscriminatorLoss::Builder builder;
            builder.network(network).realScores(real_scores).fakeScores(fake_scores).lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            HingeGANDiscriminatorLoss built = builder.build();

            new (self) HingeGANDiscriminatorLoss(std::move(built));
        },
        "network"_a,
        "real_scores"_a,
        "fake_scores"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct the discriminator-side hinge GAN loss over real and fake discriminator scores.)nbdoc");

    hinge_gan_discriminator_loss.def("get_real_scores", &HingeGANDiscriminatorLoss::getRealScores);
    hinge_gan_discriminator_loss.def("get_fake_scores", &HingeGANDiscriminatorLoss::getFakeScores);

    hinge_gan_discriminator_loss.attr("__doc__") = R"nbdoc(
Discriminator-side hinge GAN loss.

The raw elementwise loss is:

    relu(1 - real_scores) + relu(1 + fake_scores)

Gradients are produced for both real_scores and fake_scores. Use StopGradient on
the generated samples feeding the discriminator when the discriminator step should
not backpropagate into the generator.
)nbdoc";
}
