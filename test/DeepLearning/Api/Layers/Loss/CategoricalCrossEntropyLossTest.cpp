#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using std::shared_ptr;

using namespace Thor;

TEST(CategoricalCrossEntropyLoss, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(rand() % 1000);
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    Tensor featureInput(dataType, dimensions);

    Optional<float> lossScalingFactor;
    if (rand() % 2)
        lossScalingFactor = (rand() % 100) / 10.0f;

    CategoricalCrossEntropyLoss::Builder crossEntropyBuilder =
        CategoricalCrossEntropyLoss::Builder().network(network).featureInput(featureInput);
    if (lossScalingFactor.isPresent())
        crossEntropyBuilder.lossScalingFactor(lossScalingFactor);

    CategoricalCrossEntropyLoss crossEntropy = crossEntropyBuilder.build();

    ASSERT_TRUE(crossEntropy.isInitialized());

    Optional<Tensor> actualInput = crossEntropy.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualPredictions = crossEntropy.getPredictions();
    ASSERT_TRUE(actualPredictions.isPresent());
    ASSERT_EQ(actualPredictions.get().getDataType(), dataType);
    ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

    Optional<Tensor> actualLoss = crossEntropy.getLoss();
    ASSERT_TRUE(actualLoss.isPresent());
    ASSERT_EQ(actualLoss.get().getDataType(), dataType);
    ASSERT_TRUE(actualLoss.get().getDimensions().empty());

    Optional<float> actualLossScalingFactor = crossEntropy.getLossScalingFactor();
    ASSERT_TRUE(actualLossScalingFactor.isPresent() == lossScalingFactor.isPresent());
    if (lossScalingFactor.isPresent())
        assert(lossScalingFactor.get() == actualLossScalingFactor.get());

    shared_ptr<Layer> cloneLayer = crossEntropy.clone();
    CategoricalCrossEntropyLoss *clone = dynamic_cast<CategoricalCrossEntropyLoss *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> clonePredictions = clone->getPredictions();
    ASSERT_TRUE(clonePredictions.isPresent());
    ASSERT_EQ(clonePredictions.get().getDataType(), dataType);
    ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

    Optional<Tensor> cloneLoss = clone->getLoss();
    ASSERT_TRUE(cloneLoss.isPresent());
    ASSERT_EQ(cloneLoss.get().getDataType(), dataType);
    ASSERT_TRUE(cloneLoss.get().getDimensions().empty());

    Optional<float> cloneLossScalingFactor = clone->getLossScalingFactor();
    ASSERT_TRUE(cloneLossScalingFactor.isPresent() == lossScalingFactor.isPresent());
    if (lossScalingFactor.isPresent())
        assert(lossScalingFactor.get() == cloneLossScalingFactor.get());

    ASSERT_EQ(crossEntropy.getId(), clone->getId());
    ASSERT_GT(crossEntropy.getId(), 1u);

    ASSERT_TRUE(crossEntropy == *clone);
    ASSERT_FALSE(crossEntropy != *clone);
    ASSERT_FALSE(crossEntropy > *clone);
    ASSERT_FALSE(crossEntropy < *clone);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
