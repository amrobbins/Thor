#include "DeepLearning/Api/Layers/Loss/QuantileLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include "gtest/gtest.h"

#include <stdexcept>
#include <vector>

namespace Api = Thor;

TEST(QuantileLossApi, ConstructsWithDefaults) {
    Api::Network network("quantile_loss_default");
    Api::Tensor predictions(Api::DataType::FP32, {1});
    Api::Tensor labels(Api::DataType::FP32, {1});

    Api::QuantileLoss loss = Api::QuantileLoss::Builder().network(network).predictions(predictions).labels(labels).build();

    EXPECT_TRUE(loss.isInitialized());
    EXPECT_FLOAT_EQ(loss.getQuantile(), 0.5f);
    EXPECT_EQ(loss.getPredictions(), predictions);
    EXPECT_EQ(loss.getLabels(), labels);
}

TEST(QuantileLossApi, ConstructsForecastHorizonWidth100RawLoss) {
    Api::Network network("quantile_loss_width_100");
    Api::Tensor predictions(Api::DataType::FP32, {100});
    Api::Tensor labels(Api::DataType::FP32, {100});

    Api::QuantileLoss loss = Api::QuantileLoss::Builder()
                                 .network(network)
                                 .predictions(predictions)
                                 .labels(labels)
                                 .quantile(0.9f)
                                 .lossDataType(Api::DataType::FP32)
                                 .reportsRawLoss()
                                 .lossWeight(2.6667f)
                                 .build();

    EXPECT_TRUE(loss.isInitialized());
    EXPECT_FLOAT_EQ(loss.getQuantile(), 0.9f);
    EXPECT_EQ(loss.getLoss().getDimensions(), std::vector<uint64_t>({100}));
    ASSERT_TRUE(loss.getLossWeight().has_value());
    EXPECT_FLOAT_EQ(loss.getLossWeight().value(), 2.6667f);
}

TEST(QuantileLossApi, RejectsInvalidQuantiles) {
    Api::Network network("quantile_loss_invalid_quantile");
    Api::Tensor predictions(Api::DataType::FP32, {1});
    Api::Tensor labels(Api::DataType::FP32, {1});

    EXPECT_THROW(Api::QuantileLoss::Builder().network(network).predictions(predictions).labels(labels).quantile(0.0f).build(),
                 std::logic_error);
    EXPECT_THROW(Api::QuantileLoss::Builder().network(network).predictions(predictions).labels(labels).quantile(1.0f).build(),
                 std::logic_error);
}

TEST(QuantileLossApi, RejectsMismatchedDimensions) {
    Api::Network network("quantile_loss_mismatched_dims");
    Api::Tensor predictions(Api::DataType::FP32, {100});
    Api::Tensor labels(Api::DataType::FP32, {99});

    EXPECT_THROW(Api::QuantileLoss::Builder().network(network).predictions(predictions).labels(labels).build(), std::logic_error);
}
