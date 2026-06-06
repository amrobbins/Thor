#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>

#include "DeepLearning/Api/Layers/Loss/HingeGANGeneratorLoss.h"
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

void setReportedLossShape(HingeGANGeneratorLoss::Builder& builder, LossShape reported_loss_shape) {
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

void validateHingeGANGeneratorLossArguments(const string& loss_name,
                                            Tensor fake_scores,
                                            optional<DataType> loss_data_type,
                                            LossShape reported_loss_shape) {
    if (fake_scores.getDimensions().size() != 1 || fake_scores.getDimensions()[0] == 0) {
        string error_message = loss_name + ": fake_scores must be a non-empty 1D score tensor but fake_scores is " +
                               fake_scores.getDescriptorString();
        throw nb::value_error(error_message.c_str());
    }
    if (fake_scores.getDataType() != DataType::FP16 && fake_scores.getDataType() != DataType::FP32) {
        string error_message = loss_name + ": fake_scores must use fp16 or fp32 dtype";
        throw nb::value_error(error_message.c_str());
    }
    DataType effectiveLossDataType = loss_data_type.value_or(fake_scores.getDataType());
    if (effectiveLossDataType != DataType::FP16 && effectiveLossDataType != DataType::FP32) {
        string error_message = loss_name + ": loss_data_type must be fp16 or fp32";
        throw nb::value_error(error_message.c_str());
    }
    validateReportedLossShape(reported_loss_shape, loss_name);
}
}  // namespace

void bind_hinge_gan_generator_loss(nb::module_& losses) {
    auto hinge_gan_generator_loss = nb::class_<HingeGANGeneratorLoss, Loss>(losses, "HingeGANGeneratorLoss");
    hinge_gan_generator_loss.attr("__module__") = "thor.losses.gan";

    hinge_gan_generator_loss.def(
        "__init__",
        [](HingeGANGeneratorLoss* self,
           Network& network,
           Tensor fake_scores,
           std::optional<DataType> loss_data_type,
           LossShape reported_loss_shape) {
            const string loss_name = "HingeGANGeneratorLoss instance";
            validateHingeGANGeneratorLossArguments(loss_name, fake_scores, loss_data_type, reported_loss_shape);

            DataType effectiveLossDataType = loss_data_type.value_or(fake_scores.getDataType());
            HingeGANGeneratorLoss::Builder builder;
            builder.network(network).fakeScores(fake_scores).lossDataType(effectiveLossDataType);
            setReportedLossShape(builder, reported_loss_shape);
            HingeGANGeneratorLoss built = builder.build();

            new (self) HingeGANGeneratorLoss(std::move(built));
        },
        "network"_a,
        "fake_scores"_a,
        "loss_data_type"_a.none() = nb::none(),
        "reported_loss_shape"_a = LossShape::BATCH,
        R"nbdoc(Construct the generator-side hinge GAN loss over fake discriminator scores.)nbdoc");

    hinge_gan_generator_loss.def("get_fake_scores", &HingeGANGeneratorLoss::getFakeScores);

    hinge_gan_generator_loss.attr("__doc__") = R"nbdoc(
Generator-side hinge GAN loss.

The raw elementwise loss is:

    -fake_scores

Gradients are produced for fake_scores so that the generator step can backpropagate
through the discriminator graph to the generator, while step-scoped update sets
control which parameters are updated.
)nbdoc";
}
