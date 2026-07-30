#include "DeepLearning/Api/Layers/Loss/HingeGANDiscriminatorLoss.h"
#include "DeepLearning/Api/Layers/Loss/HingeGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Loss/LSGANDiscriminatorLoss.h"
#include "DeepLearning/Api/Layers/Loss/LSGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANGeneratorLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Api = Thor;

namespace {

const std::vector<uint64_t> kPatchScoreDimensions = {1, 8, 8};

template <typename BuildLoss>
void expectDiscriminatorAcceptsPatchScores(const std::string& networkName, BuildLoss buildLoss) {
    Api::Network network(networkName);
    Api::Tensor realScores(Api::DataType::FP32, kPatchScoreDimensions);
    Api::Tensor fakeScores(Api::DataType::FP32, kPatchScoreDimensions);

    const Api::Tensor rawLoss = buildLoss(network, realScores, fakeScores);
    EXPECT_EQ(rawLoss.getDimensions(), kPatchScoreDimensions);
}

template <typename BuildLoss>
void expectGeneratorAcceptsPatchScores(const std::string& networkName, BuildLoss buildLoss) {
    Api::Network network(networkName);
    Api::Tensor fakeScores(Api::DataType::FP32, kPatchScoreDimensions);

    const Api::Tensor rawLoss = buildLoss(network, fakeScores);
    EXPECT_EQ(rawLoss.getDimensions(), kPatchScoreDimensions);
}

}  // namespace

TEST(GanLossShapeContract, PointwiseGanLossesAcceptPatchScoreTensors) {
    expectDiscriminatorAcceptsPatchScores(
        "hinge_discriminator_patch_scores",
        [](Api::Network& network, Api::Tensor realScores, Api::Tensor fakeScores) {
            return Api::HingeGANDiscriminatorLoss::Builder()
                .network(network)
                .realScores(realScores)
                .fakeScores(fakeScores)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectGeneratorAcceptsPatchScores(
        "hinge_generator_patch_scores", [](Api::Network& network, Api::Tensor fakeScores) {
            return Api::HingeGANGeneratorLoss::Builder()
                .network(network)
                .fakeScores(fakeScores)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectDiscriminatorAcceptsPatchScores(
        "lsgan_discriminator_patch_scores",
        [](Api::Network& network, Api::Tensor realScores, Api::Tensor fakeScores) {
            return Api::LSGANDiscriminatorLoss::Builder()
                .network(network)
                .realScores(realScores)
                .fakeScores(fakeScores)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectGeneratorAcceptsPatchScores(
        "lsgan_generator_patch_scores", [](Api::Network& network, Api::Tensor fakeScores) {
            return Api::LSGANGeneratorLoss::Builder()
                .network(network)
                .fakeScores(fakeScores)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectDiscriminatorAcceptsPatchScores(
        "wasserstein_critic_patch_scores",
        [](Api::Network& network, Api::Tensor realScores, Api::Tensor fakeScores) {
            return Api::WassersteinGANCriticLoss::Builder()
                .network(network)
                .realScores(realScores)
                .fakeScores(fakeScores)
                .reportsRawLoss()
                .build()
                .getLoss();
        });

    expectGeneratorAcceptsPatchScores(
        "wasserstein_generator_patch_scores", [](Api::Network& network, Api::Tensor fakeScores) {
            return Api::WassersteinGANGeneratorLoss::Builder()
                .network(network)
                .fakeScores(fakeScores)
                .reportsRawLoss()
                .build()
                .getLoss();
        });
}

TEST(GanLossShapeContract, PerOutputReportingPreservesPatchLayout) {
    Api::Network network("hinge_discriminator_per_output_patch_scores");
    Api::Tensor realScores(Api::DataType::FP32, kPatchScoreDimensions);
    Api::Tensor fakeScores(Api::DataType::FP32, kPatchScoreDimensions);

    Api::HingeGANDiscriminatorLoss loss = Api::HingeGANDiscriminatorLoss::Builder()
                                              .network(network)
                                              .realScores(realScores)
                                              .fakeScores(fakeScores)
                                              .reportsPerOutputLoss()
                                              .build();

    EXPECT_EQ(loss.getLoss().getDimensions(), kPatchScoreDimensions);
}
