#include "DeepLearning/Api/Layers/Loss/ExpectileLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <stdexcept>
#include <vector>

namespace Api = Thor;

TEST(ExpectileLossApi, ConstructsWithDefaultsAndMatchesMseAtCenter) {
    Api::Network network("expectile_loss_default");
    Api::Tensor predictions(Api::DataType::FP32, {1});
    Api::Tensor labels(Api::DataType::FP32, {1});

    Api::ExpectileLoss loss = Api::ExpectileLoss::Builder().network(network).predictions(predictions).labels(labels).build();

    EXPECT_TRUE(loss.isInitialized());
    EXPECT_FLOAT_EQ(loss.getExpectile(), 0.5f);
    EXPECT_EQ(loss.getPredictions(), predictions);
    EXPECT_EQ(loss.getLabels(), labels);
    EXPECT_EQ(loss.architectureJson().at("layer_type").get<std::string>(), "expectile_loss");
    EXPECT_FLOAT_EQ(loss.architectureJson().at("expectile").get<float>(), 0.5f);
}

TEST(ExpectileLossApi, ConstructsForecastHorizonWidth100RawLoss) {
    Api::Network network("expectile_loss_width_100");
    Api::Tensor predictions(Api::DataType::FP32, {100});
    Api::Tensor labels(Api::DataType::FP32, {100});
    Api::Tensor exampleWeights(Api::DataType::FP16, {1});

    Api::ExpectileLoss loss = Api::ExpectileLoss::Builder()
                                  .network(network)
                                  .predictions(predictions)
                                  .labels(labels)
                                  .exampleWeights(exampleWeights)
                                  .expectile(0.9f)
                                  .lossDataType(Api::DataType::FP32)
                                  .reportsRawLoss()
                                  .lossWeight(2.6667f)
                                  .build();

    EXPECT_TRUE(loss.isInitialized());
    EXPECT_FLOAT_EQ(loss.getExpectile(), 0.9f);
    EXPECT_EQ(loss.getLoss().getDimensions(), std::vector<uint64_t>({100}));
    ASSERT_TRUE(loss.getExampleWeights().has_value());
    EXPECT_EQ(loss.getExampleWeights().value(), exampleWeights);
    ASSERT_TRUE(loss.getLossWeight().has_value());
    EXPECT_FLOAT_EQ(loss.getLossWeight().value(), 2.6667f);
    EXPECT_FLOAT_EQ(loss.architectureJson().at("expectile").get<float>(), 0.9f);
}

TEST(ExpectileLossApi, RejectsInvalidExpectiles) {
    Api::Network network("expectile_loss_invalid_expectile");
    Api::Tensor predictions(Api::DataType::FP32, {1});
    Api::Tensor labels(Api::DataType::FP32, {1});

    EXPECT_THROW(Api::ExpectileLoss::Builder().network(network).predictions(predictions).labels(labels).expectile(0.0f).build(),
                 std::logic_error);
    EXPECT_THROW(Api::ExpectileLoss::Builder().network(network).predictions(predictions).labels(labels).expectile(1.0f).build(),
                 std::logic_error);
}

TEST(ExpectileLossApi, RejectsMismatchedDimensions) {
    Api::Network network("expectile_loss_mismatched_dims");
    Api::Tensor predictions(Api::DataType::FP32, {100});
    Api::Tensor labels(Api::DataType::FP32, {99});

    EXPECT_THROW(Api::ExpectileLoss::Builder().network(network).predictions(predictions).labels(labels).build(), std::logic_error);
}
