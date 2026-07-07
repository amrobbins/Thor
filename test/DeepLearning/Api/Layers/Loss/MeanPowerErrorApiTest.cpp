#include "DeepLearning/Api/Layers/Loss/MeanPowerError.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace Thor;
using json = nlohmann::json;

TEST(MeanPowerErrorApi, BuildsAndSerializesExponent) {
    Network network("mean_power_error_api");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});
    Tensor weights(DataType::FP32, {1});

    MeanPowerError loss = MeanPowerError::Builder()
                              .network(network)
                              .predictions(predictions)
                              .labels(labels)
                              .exampleWeights(weights)
                              .exponent(1.5f)
                              .lossWeight(2.0f)
                              .reportsRawLoss()
                              .build();

    ASSERT_TRUE(loss.isInitialized());
    ASSERT_EQ(loss.getPredictions(), predictions);
    ASSERT_EQ(loss.getLabels(), labels);
    ASSERT_EQ(loss.getExampleWeights().value(), weights);
    ASSERT_FLOAT_EQ(loss.getExponent(), 1.5f);
    ASSERT_EQ(loss.getLoss().getDataType(), DataType::FP32);
    ASSERT_EQ(loss.getLoss().getDimensions(), vector<uint64_t>({3}));

    shared_ptr<Layer> cloneLayer = loss.clone();
    MeanPowerError* clone = dynamic_cast<MeanPowerError*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->isInitialized());
    ASSERT_FLOAT_EQ(clone->getExponent(), 1.5f);

    json lossJson = loss.architectureJson();
    ASSERT_EQ(lossJson.at("factory").get<string>(), Layer::Factory::Loss.value());
    ASSERT_EQ(lossJson.at("layer_type").get<string>(), string("mean_power_error"));
    ASSERT_EQ(lossJson.at("loss_shape").get<string>(), string("raw"));
    ASSERT_FLOAT_EQ(lossJson.at("exponent").get<float>(), 1.5f);
    ASSERT_FLOAT_EQ(lossJson.at("loss_weight").get<float>(), 2.0f);
    ASSERT_TRUE(lossJson.contains("example_weights_tensor"));
}

TEST(MeanPowerErrorApi, ExponentOneAndTwoBuild) {
    Network network("mean_power_error_api_exponent_one_two");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    MeanPowerError maeLike = MeanPowerError::Builder().network(network).predictions(predictions).labels(labels).exponent(1.0f).build();
    MeanPowerError mseLike = MeanPowerError::Builder().network(network).predictions(predictions).labels(labels).exponent(2.0f).build();

    ASSERT_FLOAT_EQ(maeLike.getExponent(), 1.0f);
    ASSERT_FLOAT_EQ(mseLike.getExponent(), 2.0f);
}

TEST(MeanPowerErrorApi, RejectsExponentBelowOne) {
    Network network("mean_power_error_api_bad_exponent");
    Tensor predictions(DataType::FP32, {3});
    Tensor labels(DataType::FP32, {3});

    EXPECT_THROW(MeanPowerError::Builder().network(network).predictions(predictions).labels(labels).exponent(0.5f).build(),
                 std::exception);
}
