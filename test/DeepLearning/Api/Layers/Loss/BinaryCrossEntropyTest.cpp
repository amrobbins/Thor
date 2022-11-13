#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace std;

using namespace Thor;

TEST(BinaryCrossEntropy, BatchLossBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions = {1UL + (rand() % 300)};
        vector<uint64_t> lossDimensions = {1};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 2;
        if (r == 0)
            labelsDataType = Tensor::DataType::FP16;
        else if (r == 1)
            labelsDataType = Tensor::DataType::FP32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        BinaryCrossEntropy::Builder crossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                              .network(network)
                                                              .predictions(predictions)
                                                              .reportsBatchLoss()
                                                              .lossDataType(lossDataType)
                                                              .labels(labels);
        BinaryCrossEntropy crossEntropy = crossEntropyBuilder.build();

        ASSERT_TRUE(crossEntropy.isInitialized());

        Optional<Tensor> actualInput = crossEntropy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLabels = crossEntropy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualPredictions = crossEntropy.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

        Optional<Tensor> actualLoss = crossEntropy.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(actualLoss.get().getDimensions(), lossDimensions);

        shared_ptr<Layer> cloneLayer = crossEntropy.clone();
        BinaryCrossEntropy *clone = dynamic_cast<BinaryCrossEntropy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(cloneLoss.get().getDimensions(), lossDimensions);

        ASSERT_EQ(crossEntropy.getId(), clone->getId());
        ASSERT_GT(crossEntropy.getId(), 1u);

        ASSERT_TRUE(crossEntropy == *clone);
        ASSERT_FALSE(crossEntropy != *clone);
        ASSERT_FALSE(crossEntropy > *clone);
        ASSERT_FALSE(crossEntropy < *clone);
    }
}

TEST(BinaryCrossEntropy, ElementwiseLossBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions = {1UL + (rand() % 300)};
        vector<uint64_t> lossDimensions = dimensions;
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 2;
        if (r == 0)
            labelsDataType = Tensor::DataType::FP16;
        else if (r == 1)
            labelsDataType = Tensor::DataType::FP32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        BinaryCrossEntropy::Builder crossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                              .network(network)
                                                              .predictions(predictions)
                                                              .reportsElementwiseLoss()
                                                              .lossDataType(lossDataType)
                                                              .labels(labels);
        BinaryCrossEntropy crossEntropy = crossEntropyBuilder.build();

        ASSERT_TRUE(crossEntropy.isInitialized());

        Optional<Tensor> actualInput = crossEntropy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLabels = crossEntropy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualPredictions = crossEntropy.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

        Optional<Tensor> actualLoss = crossEntropy.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(actualLoss.get().getDimensions(), lossDimensions);

        shared_ptr<Layer> cloneLayer = crossEntropy.clone();
        BinaryCrossEntropy *clone = dynamic_cast<BinaryCrossEntropy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(cloneLoss.get().getDimensions(), lossDimensions);

        ASSERT_EQ(crossEntropy.getId(), clone->getId());
        ASSERT_GT(crossEntropy.getId(), 1u);

        ASSERT_TRUE(crossEntropy == *clone);
        ASSERT_FALSE(crossEntropy != *clone);
        ASSERT_FALSE(crossEntropy > *clone);
        ASSERT_FALSE(crossEntropy < *clone);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
