#include "DeepLearning/Api/Layers/Loss/BinaryFocalLoss.h"
#include "DeepLearning/Api/Layers/Loss/BoxIouLoss.h"
#include "DeepLearning/Api/Layers/Loss/CategoricalFocalLoss.h"
#include "DeepLearning/Api/Layers/Loss/ContrastiveLoss.h"
#include "DeepLearning/Api/Layers/Loss/CosineEmbeddingLoss.h"
#include "DeepLearning/Api/Layers/Loss/DiceLoss.h"
#include "DeepLearning/Api/Layers/Loss/FocalTverskyLoss.h"
#include "DeepLearning/Api/Layers/Loss/GammaNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/GaussianNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/HingeGANDiscriminatorLoss.h"
#include "DeepLearning/Api/Layers/Loss/HingeGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Loss/InfoNCELoss.h"
#include "DeepLearning/Api/Layers/Loss/LSGANDiscriminatorLoss.h"
#include "DeepLearning/Api/Layers/Loss/LSGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Loss/ListNetLoss.h"
#include "DeepLearning/Api/Layers/Loss/ListwiseSoftmaxCrossEntropyLoss.h"
#include "DeepLearning/Api/Layers/Loss/MarginRankingLoss.h"
#include "DeepLearning/Api/Layers/Loss/PoissonNLLLoss.h"
#include "DeepLearning/Api/Layers/Loss/TripletLoss.h"
#include "DeepLearning/Api/Layers/Loss/TverskyLoss.h"
#include "DeepLearning/Api/Layers/Loss/TweedieLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticGradientPenaltyLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANCriticLoss.h"
#include "DeepLearning/Api/Layers/Loss/WassersteinGANGeneratorLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {

NetworkInput input(Network& network, const string& name, const vector<uint64_t>& dimensions, DataType dataType) {
    return NetworkInput::Builder().network(network).name(name).dimensions(dimensions).dataType(dataType).build();
}

NetworkInput fp32Input(Network& network, const string& name, const vector<uint64_t>& dimensions) {
    return input(network, name, dimensions, DataType::FP32);
}

NetworkInput int32Input(Network& network, const string& name, const vector<uint64_t>& dimensions) {
    return input(network, name, dimensions, DataType::INT32);
}

void consumeLoss(Network& network, const string& name, const Tensor& lossTensor) {
    NetworkOutput::Builder().network(network).name(name).inputTensor(lossTensor).dataType(lossTensor.getDataType()).build();
}

filesystem::path makeUniqueTestArchiveDir(const string& testName) {
    const auto now = chrono::high_resolution_clock::now().time_since_epoch().count();
    filesystem::path dir = filesystem::temp_directory_path() / (testName + "_" + to_string(now));
    filesystem::remove_all(dir);
    filesystem::create_directories(dir);
    return dir;
}

void normalizeGeneratedIds(json& value) {
    if (value.is_object()) {
        value.erase("id");
        value.erase("layer_name");
        for (auto& item : value.items()) {
            normalizeGeneratedIds(item.value());
        }
        return;
    }
    if (value.is_array()) {
        for (json& item : value) {
            normalizeGeneratedIds(item);
        }
    }
}

json canonicalSupportLayers(const Network& network) {
    json result = json::array();
    json architecture = network.architectureJson();
    for (json layer : architecture.at("layers")) {
        const string layerType = layer.at("layer_type").get<string>();
        if (layerType == "network_input" || layerType == "network_output")
            continue;
        normalizeGeneratedIds(layer);
        result.push_back(layer);
    }
    return result;
}

struct DomainLossFixture {
    size_t numLosses = 0;
};

template <typename LossT>
void rememberLoss(DomainLossFixture& fixture, Network& network, const LossT& loss) {
    json publicJson = loss.architectureJson();
    if (!publicJson.contains("loss_weight")) {
        throw runtime_error("Expected public loss JSON to contain loss_weight: " + publicJson.dump());
    }
    ++fixture.numLosses;

    const string outputName = publicJson.at("layer_type").get<string>() + "_" + to_string(fixture.numLosses) + "_output";
    consumeLoss(network, outputName, loss.getLoss());
}

DomainLossFixture addAllNewDomainLosses(Network& network) {
    DomainLossFixture fixture;

    {
        auto predictions = fp32Input(network, "binary_focal_predictions", {1});
        auto labels = fp32Input(network, "binary_focal_labels", {1});
        BinaryFocalLoss loss = BinaryFocalLoss::Builder()
                                   .network(network)
                                   .predictions(predictions.getFeatureOutput().value())
                                   .labels(labels.getFeatureOutput().value())
                                   .focusingParameter(1.375f)
                                   .alpha(0.625f)
                                   .reportsRawLoss()
                                   .lossDataType(DataType::FP32)
                                   .lossWeight(1.125f)
                                   .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "categorical_focal_predictions", {4});
        auto labels = fp32Input(network, "categorical_focal_labels", {4});
        CategoricalFocalLoss loss = CategoricalFocalLoss::Builder()
                                        .network(network)
                                        .predictions(predictions.getFeatureOutput().value())
                                        .labels(labels.getFeatureOutput().value())
                                        .focusingParameter(1.25f)
                                        .alpha(0.875f)
                                        .reportsPerOutputLoss()
                                        .lossDataType(DataType::FP32)
                                        .lossWeight(1.25f)
                                        .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "dice_predictions", {2, 3});
        auto labels = fp32Input(network, "dice_labels", {2, 3});
        DiceLoss loss = DiceLoss::Builder()
                            .network(network)
                            .predictions(predictions.getFeatureOutput().value())
                            .labels(labels.getFeatureOutput().value())
                            .smooth(0.375f)
                            .reportsElementwiseLoss()
                            .lossDataType(DataType::FP32)
                            .lossWeight(1.375f)
                            .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "tversky_predictions", {2, 3});
        auto labels = fp32Input(network, "tversky_labels", {2, 3});
        TverskyLoss loss = TverskyLoss::Builder()
                               .network(network)
                               .predictions(predictions.getFeatureOutput().value())
                               .labels(labels.getFeatureOutput().value())
                               .alpha(0.2f)
                               .beta(0.8f)
                               .smooth(0.25f)
                               .reportsRawLoss()
                               .lossDataType(DataType::FP32)
                               .lossWeight(1.5f)
                               .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "focal_tversky_predictions", {2, 3});
        auto labels = fp32Input(network, "focal_tversky_labels", {2, 3});
        FocalTverskyLoss loss = FocalTverskyLoss::Builder()
                                    .network(network)
                                    .predictions(predictions.getFeatureOutput().value())
                                    .labels(labels.getFeatureOutput().value())
                                    .alpha(0.4f)
                                    .beta(0.6f)
                                    .gamma(1.35f)
                                    .smooth(0.5f)
                                    .reportsBatchLoss()
                                    .lossDataType(DataType::FP32)
                                    .lossWeight(1.625f)
                                    .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "iou_predictions", {2, 4});
        auto labels = fp32Input(network, "iou_labels", {2, 4});
        IoULoss loss = IoULoss::Builder()
                           .network(network)
                           .predictions(predictions.getFeatureOutput().value())
                           .labels(labels.getFeatureOutput().value())
                           .eps(1.0e-5f)
                           .reportsRawLoss()
                           .lossDataType(DataType::FP32)
                           .lossWeight(1.75f)
                           .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "giou_predictions", {2, 4});
        auto labels = fp32Input(network, "giou_labels", {2, 4});
        GIoULoss loss = GIoULoss::Builder()
                            .network(network)
                            .predictions(predictions.getFeatureOutput().value())
                            .labels(labels.getFeatureOutput().value())
                            .eps(2.0e-5f)
                            .reportsBatchLoss()
                            .lossDataType(DataType::FP32)
                            .lossWeight(1.875f)
                            .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "diou_predictions", {2, 4});
        auto labels = fp32Input(network, "diou_labels", {2, 4});
        DIoULoss loss = DIoULoss::Builder()
                            .network(network)
                            .predictions(predictions.getFeatureOutput().value())
                            .labels(labels.getFeatureOutput().value())
                            .eps(3.0e-5f)
                            .reportsElementwiseLoss()
                            .lossDataType(DataType::FP32)
                            .lossWeight(2.0f)
                            .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "ciou_predictions", {2, 4});
        auto labels = fp32Input(network, "ciou_labels", {2, 4});
        CIoULoss loss = CIoULoss::Builder()
                            .network(network)
                            .predictions(predictions.getFeatureOutput().value())
                            .labels(labels.getFeatureOutput().value())
                            .eps(4.0e-5f)
                            .reportsPerOutputLoss()
                            .lossDataType(DataType::FP32)
                            .lossWeight(2.125f)
                            .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto mean = fp32Input(network, "gamma_mean", {3});
        auto target = fp32Input(network, "gamma_target", {3});
        GammaNLLLoss loss = GammaNLLLoss::Builder()
                                .network(network)
                                .mean(mean.getFeatureOutput().value())
                                .target(target.getFeatureOutput().value())
                                .eps(1.0e-4f)
                                .reportsRawLoss()
                                .lossDataType(DataType::FP32)
                                .lossWeight(2.25f)
                                .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto mean = fp32Input(network, "tweedie_mean", {3});
        auto target = fp32Input(network, "tweedie_target", {3});
        TweedieLoss loss = TweedieLoss::Builder()
                               .network(network)
                               .mean(mean.getFeatureOutput().value())
                               .target(target.getFeatureOutput().value())
                               .power(1.7f)
                               .eps(2.0e-4f)
                               .reportsElementwiseLoss()
                               .lossDataType(DataType::FP32)
                               .lossWeight(2.375f)
                               .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto mean = fp32Input(network, "poisson_mean", {3});
        auto target = fp32Input(network, "poisson_target", {3});
        PoissonNLLLoss loss = PoissonNLLLoss::Builder()
                                  .network(network)
                                  .predictions(mean.getFeatureOutput().value())
                                  .labels(target.getFeatureOutput().value())
                                  .logInput(false)
                                  .full(true)
                                  .eps(3.0e-4f)
                                  .reportsBatchLoss()
                                  .lossDataType(DataType::FP32)
                                  .lossWeight(2.5f)
                                  .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto mean = fp32Input(network, "gaussian_mean", {3});
        auto target = fp32Input(network, "gaussian_target", {3});
        auto variance = fp32Input(network, "gaussian_variance", {3});
        GaussianNLLLoss loss = GaussianNLLLoss::Builder()
                                   .network(network)
                                   .mean(mean.getFeatureOutput().value())
                                   .target(target.getFeatureOutput().value())
                                   .variance(variance.getFeatureOutput().value())
                                   .full(true)
                                   .eps(4.0e-4f)
                                   .reportsRawLoss()
                                   .lossDataType(DataType::FP32)
                                   .lossWeight(2.625f)
                                   .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto realScores = fp32Input(network, "hinge_real_scores", {3});
        auto fakeScores = fp32Input(network, "hinge_fake_scores", {3});
        HingeGANDiscriminatorLoss loss = HingeGANDiscriminatorLoss::Builder()
                                             .network(network)
                                             .realScores(realScores.getFeatureOutput().value())
                                             .fakeScores(fakeScores.getFeatureOutput().value())
                                             .reportsRawLoss()
                                             .lossDataType(DataType::FP32)
                                             .lossWeight(2.75f)
                                             .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto fakeScores = fp32Input(network, "hinge_generator_fake_scores", {3});
        HingeGANGeneratorLoss loss = HingeGANGeneratorLoss::Builder()
                                         .network(network)
                                         .fakeScores(fakeScores.getFeatureOutput().value())
                                         .reportsBatchLoss()
                                         .lossDataType(DataType::FP32)
                                         .lossWeight(2.875f)
                                         .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto realScores = fp32Input(network, "lsgan_real_scores", {3});
        auto fakeScores = fp32Input(network, "lsgan_fake_scores", {3});
        LSGANDiscriminatorLoss loss = LSGANDiscriminatorLoss::Builder()
                                          .network(network)
                                          .realScores(realScores.getFeatureOutput().value())
                                          .fakeScores(fakeScores.getFeatureOutput().value())
                                          .realTarget(0.9f)
                                          .fakeTarget(0.1f)
                                          .reportsElementwiseLoss()
                                          .lossDataType(DataType::FP32)
                                          .lossWeight(3.0f)
                                          .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto fakeScores = fp32Input(network, "lsgan_generator_fake_scores", {3});
        LSGANGeneratorLoss loss = LSGANGeneratorLoss::Builder()
                                      .network(network)
                                      .fakeScores(fakeScores.getFeatureOutput().value())
                                      .target(0.85f)
                                      .reportsPerOutputLoss()
                                      .lossDataType(DataType::FP32)
                                      .lossWeight(3.125f)
                                      .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto realScores = fp32Input(network, "wgan_real_scores", {3});
        auto fakeScores = fp32Input(network, "wgan_fake_scores", {3});
        WassersteinGANCriticLoss loss = WassersteinGANCriticLoss::Builder()
                                            .network(network)
                                            .realScores(realScores.getFeatureOutput().value())
                                            .fakeScores(fakeScores.getFeatureOutput().value())
                                            .reportsRawLoss()
                                            .lossDataType(DataType::FP32)
                                            .lossWeight(3.25f)
                                            .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto fakeScores = fp32Input(network, "wgan_generator_fake_scores", {3});
        WassersteinGANGeneratorLoss loss = WassersteinGANGeneratorLoss::Builder()
                                               .network(network)
                                               .fakeScores(fakeScores.getFeatureOutput().value())
                                               .reportsElementwiseLoss()
                                               .lossDataType(DataType::FP32)
                                               .lossWeight(3.375f)
                                               .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto realScores = fp32Input(network, "wgan_gp_real_scores", {1});
        auto fakeScores = fp32Input(network, "wgan_gp_fake_scores", {1});
        auto sampleGradients = fp32Input(network, "wgan_gp_sample_gradients", {2, 3});
        WassersteinGANCriticGradientPenaltyLoss loss = WassersteinGANCriticGradientPenaltyLoss::Builder()
                                                           .network(network)
                                                           .realScores(realScores.getFeatureOutput().value())
                                                           .fakeScores(fakeScores.getFeatureOutput().value())
                                                           .sampleGradients(sampleGradients.getFeatureOutput().value())
                                                           .gradientPenaltyWeight(7.5f)
                                                           .targetGradientNorm(1.25f)
                                                           .eps(1.0e-6f)
                                                           .reportsBatchLoss()
                                                           .lossDataType(DataType::FP32)
                                                           .lossWeight(3.5f)
                                                           .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto distances = fp32Input(network, "contrastive_distances", {4});
        auto labels = fp32Input(network, "contrastive_labels", {4});
        ContrastiveLoss loss = ContrastiveLoss::Builder()
                                   .network(network)
                                   .predictions(distances.getFeatureOutput().value())
                                   .labels(labels.getFeatureOutput().value())
                                   .margin(1.625f)
                                   .reportsRawLoss()
                                   .lossDataType(DataType::FP32)
                                   .lossWeight(3.625f)
                                   .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto logits = fp32Input(network, "info_nce_logits", {4});
        auto labels = fp32Input(network, "info_nce_labels", {4});
        InfoNCELoss loss = InfoNCELoss::Builder()
                               .network(network)
                               .predictions(logits.getFeatureOutput().value())
                               .labels(labels.getFeatureOutput().value())
                               .temperature(0.375f)
                               .reportsPerOutputLoss()
                               .lossDataType(DataType::FP32)
                               .lossWeight(3.75f)
                               .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto anchor = fp32Input(network, "triplet_anchor", {3});
        auto positive = fp32Input(network, "triplet_positive", {3});
        auto negative = fp32Input(network, "triplet_negative", {3});
        TripletLoss loss = TripletLoss::Builder()
                               .network(network)
                               .anchor(anchor.getFeatureOutput().value())
                               .positive(positive.getFeatureOutput().value())
                               .negative(negative.getFeatureOutput().value())
                               .margin(0.875f)
                               .eps(3.0e-5f)
                               .reportsBatchLoss()
                               .lossDataType(DataType::FP32)
                               .lossWeight(3.875f)
                               .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto input1 = fp32Input(network, "cosine_input1", {3});
        auto input2 = fp32Input(network, "cosine_input2", {3});
        auto target = int32Input(network, "cosine_target", {1});
        CosineEmbeddingLoss loss = CosineEmbeddingLoss::Builder()
                                       .network(network)
                                       .input1(input1.getFeatureOutput().value())
                                       .input2(input2.getFeatureOutput().value())
                                       .target(target.getFeatureOutput().value())
                                       .margin(0.125f)
                                       .eps(4.0e-5f)
                                       .reportsElementwiseLoss()
                                       .lossDataType(DataType::FP32)
                                       .lossWeight(4.0f)
                                       .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto input1 = fp32Input(network, "margin_ranking_input1", {4});
        auto input2 = fp32Input(network, "margin_ranking_input2", {4});
        auto target = int32Input(network, "margin_ranking_target", {4});
        MarginRankingLoss loss = MarginRankingLoss::Builder()
                                     .network(network)
                                     .input1(input1.getFeatureOutput().value())
                                     .input2(input2.getFeatureOutput().value())
                                     .target(target.getFeatureOutput().value())
                                     .margin(0.625f)
                                     .reportsRawLoss()
                                     .lossDataType(DataType::FP32)
                                     .lossWeight(4.125f)
                                     .build();
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "list_net_predictions", {5});
        auto labels = fp32Input(network, "list_net_labels", {5});
        auto mask = fp32Input(network, "list_net_mask", {5});
        ListNetLoss loss = ListNetLoss::Builder()
                               .network(network)
                               .predictions(predictions.getFeatureOutput().value())
                               .labels(labels.getFeatureOutput().value())
                               .mask(mask.getFeatureOutput().value())
                               .scoreTemperature(0.625f)
                               .labelTemperature(0.4375f)
                               .reportsRawLoss()
                               .lossDataType(DataType::FP32)
                               .lossWeight(4.25f)
                               .build();
        json publicJson = loss.architectureJson();
        if (!publicJson.at("has_mask").get<bool>() || !publicJson.contains("mask_tensor")) {
            throw runtime_error("Expected ListNetLoss JSON to preserve its mask tensor: " + publicJson.dump());
        }
        EXPECT_FLOAT_EQ(publicJson.at("score_temperature").get<float>(), 0.625f);
        EXPECT_FLOAT_EQ(publicJson.at("label_temperature").get<float>(), 0.4375f);
        rememberLoss(fixture, network, loss);
    }
    {
        auto predictions = fp32Input(network, "listwise_softmax_predictions", {5});
        auto labels = fp32Input(network, "listwise_softmax_labels", {5});
        auto mask = fp32Input(network, "listwise_softmax_mask", {5});
        ListwiseSoftmaxCrossEntropyLoss loss = ListwiseSoftmaxCrossEntropyLoss::Builder()
                                                   .network(network)
                                                   .predictions(predictions.getFeatureOutput().value())
                                                   .labels(labels.getFeatureOutput().value())
                                                   .mask(mask.getFeatureOutput().value())
                                                   .temperature(0.5625f)
                                                   .reportsElementwiseLoss()
                                                   .lossDataType(DataType::FP32)
                                                   .lossWeight(4.375f)
                                                   .build();
        json publicJson = loss.architectureJson();
        if (!publicJson.at("has_mask").get<bool>() || !publicJson.contains("mask_tensor")) {
            throw runtime_error("Expected ListwiseSoftmaxCrossEntropyLoss JSON to preserve its mask tensor: " + publicJson.dump());
        }
        EXPECT_FLOAT_EQ(publicJson.at("temperature").get<float>(), 0.5625f);
        rememberLoss(fixture, network, loss);
    }

    return fixture;
}

}  // namespace

TEST(DomainLossSerializationRoundTrip, NetworkSaveLoadPreservesSupportGraphsForNonDefaultDomainLossFields) {
    const string networkName = "domain_loss_support_graph_save_load_round_trip";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Network source(networkName);
        DomainLossFixture fixture = addAllNewDomainLosses(source);
        ASSERT_EQ(fixture.numLosses, 27u);
        const json expectedSupportLayers = canonicalSupportLayers(source);
        ASSERT_FALSE(expectedSupportLayers.empty());
        source.save(archiveDir.string(), true);

        Network loaded(networkName);
        loaded.load(archiveDir.string());

        EXPECT_EQ(canonicalSupportLayers(loaded), expectedSupportLayers);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }

    filesystem::remove_all(archiveDir);
}
