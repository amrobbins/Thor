#include "DeepLearning/Api/Layers/Loss/AsymmetricPowerLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <limits>
#include <stdexcept>
#include <vector>

namespace Api = Thor;

TEST(AsymmetricPowerLossApi, ConstructsWithDefaultsAndSerializesParameters) {
    Api::Network network("asymmetric_power_loss_default");
    Api::Tensor predictions(Api::DataType::FP32, {1});
    Api::Tensor labels(Api::DataType::FP32, {1});

    Api::AsymmetricPowerLoss loss =
        Api::AsymmetricPowerLoss::Builder().network(network).predictions(predictions).labels(labels).build();

    EXPECT_TRUE(loss.isInitialized());
    EXPECT_FLOAT_EQ(loss.getLevel(), 0.5f);
    EXPECT_FLOAT_EQ(loss.getExponent(), 1.5f);
    EXPECT_EQ(loss.getPredictions(), predictions);
    EXPECT_EQ(loss.getLabels(), labels);
    EXPECT_EQ(loss.architectureJson().at("layer_type").get<std::string>(), "asymmetric_power_loss");
    EXPECT_FLOAT_EQ(loss.architectureJson().at("level").get<float>(), 0.5f);
    EXPECT_FLOAT_EQ(loss.architectureJson().at("exponent").get<float>(), 1.5f);
}

TEST(AsymmetricPowerLossApi, ConstructsForecastHorizonWidth100RawLossWithExampleWeights) {
    Api::Network network("asymmetric_power_loss_width_100");
    Api::Tensor predictions(Api::DataType::FP32, {100});
    Api::Tensor labels(Api::DataType::FP32, {100});
    Api::Tensor exampleWeights(Api::DataType::FP16, {1});

    Api::AsymmetricPowerLoss loss = Api::AsymmetricPowerLoss::Builder()
                                        .network(network)
                                        .predictions(predictions)
                                        .labels(labels)
                                        .exampleWeights(exampleWeights)
                                        .level(0.9f)
                                        .exponent(1.5f)
                                        .lossDataType(Api::DataType::FP32)
                                        .reportsRawLoss()
                                        .lossWeight(2.6667f)
                                        .build();

    EXPECT_TRUE(loss.isInitialized());
    EXPECT_FLOAT_EQ(loss.getLevel(), 0.9f);
    EXPECT_FLOAT_EQ(loss.getExponent(), 1.5f);
    EXPECT_EQ(loss.getLoss().getDimensions(), std::vector<uint64_t>({100}));
    ASSERT_TRUE(loss.getExampleWeights().has_value());
    EXPECT_EQ(loss.getExampleWeights().value(), exampleWeights);
    ASSERT_TRUE(loss.getLossWeight().has_value());
    EXPECT_FLOAT_EQ(loss.getLossWeight().value(), 2.6667f);
}

TEST(AsymmetricPowerLossApi, RejectsInvalidLevelsAndExponents) {
    Api::Network network("asymmetric_power_loss_invalid_parameters");
    Api::Tensor predictions(Api::DataType::FP32, {1});
    Api::Tensor labels(Api::DataType::FP32, {1});

    EXPECT_THROW(Api::AsymmetricPowerLoss::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labels)
                     .level(0.0f)
                     .build(),
                 std::logic_error);
    EXPECT_THROW(Api::AsymmetricPowerLoss::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labels)
                     .level(1.0f)
                     .build(),
                 std::logic_error);
    EXPECT_THROW(Api::AsymmetricPowerLoss::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labels)
                     .level(std::numeric_limits<float>::quiet_NaN())
                     .build(),
                 std::logic_error);
    EXPECT_THROW(Api::AsymmetricPowerLoss::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labels)
                     .exponent(0.999f)
                     .build(),
                 std::logic_error);
    EXPECT_THROW(Api::AsymmetricPowerLoss::Builder()
                     .network(network)
                     .predictions(predictions)
                     .labels(labels)
                     .exponent(std::numeric_limits<float>::infinity())
                     .build(),
                 std::logic_error);
}

TEST(AsymmetricPowerLossApi, RejectsMismatchedDimensions) {
    Api::Network network("asymmetric_power_loss_mismatched_dims");
    Api::Tensor predictions(Api::DataType::FP32, {100});
    Api::Tensor labels(Api::DataType::FP32, {99});

    EXPECT_THROW(Api::AsymmetricPowerLoss::Builder().network(network).predictions(predictions).labels(labels).build(),
                 std::logic_error);
}
