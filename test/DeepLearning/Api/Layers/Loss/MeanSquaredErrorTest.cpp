#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using std::shared_ptr;

using namespace Thor;

TEST(MeanSquaredError, Builds) {
    srand(time(nullptr));

    for (uint32_t i = 0; i < 10; ++i) {
        Network network;

        vector<uint64_t> dimensions;
        uint32_t numDimensions = 2;
        for (uint32_t j = 0; j < numDimensions; ++j)
            dimensions.push_back(1 + (rand() % 1000));
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor predictions(predictionsDataType, dimensions);

        Tensor::DataType labelsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor labels(labelsDataType, dimensions);

        bool reportBatchLoss = rand() % 2 == 0;
        bool setLossDataType = rand() % 2 == 0;
        Tensor::DataType lossDataType;

        MeanSquaredError::Builder meanSquaredErrorBuilder =
            MeanSquaredError::Builder().network(network).predictions(predictions).labels(labels);

        if (reportBatchLoss)
            meanSquaredErrorBuilder.reportsBatchLoss();
        else
            meanSquaredErrorBuilder.reportsElementwiseLoss();

        if (setLossDataType) {
            lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
            meanSquaredErrorBuilder.lossDataType(lossDataType);
        }

        MeanSquaredError meanSquaredError = meanSquaredErrorBuilder.build();

        ASSERT_TRUE(meanSquaredError.isInitialized());

        Optional<Tensor> actualLabels = meanSquaredError.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualPredictions = meanSquaredError.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

        Optional<Tensor> actualLoss = meanSquaredError.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), setLossDataType ? lossDataType : predictionsDataType);
        if (reportBatchLoss) {
            vector<uint64_t> batchDimensions = {dimensions[1]};
            ASSERT_EQ(actualLoss.get().getDimensions(), batchDimensions);
        } else {
            ASSERT_EQ(actualLoss.get().getDimensions(), dimensions);
        }

        ASSERT_TRUE(meanSquaredError.getPredictions() == meanSquaredError.getFeatureInput());
        ASSERT_TRUE(meanSquaredError.getLoss() == meanSquaredError.getFeatureOutput());

        shared_ptr<Layer> cloneLayer = meanSquaredError.clone();
        MeanSquaredError *clone = dynamic_cast<MeanSquaredError *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), setLossDataType ? lossDataType : predictionsDataType);
        if (reportBatchLoss) {
            vector<uint64_t> batchDimensions = {dimensions[1]};
            ASSERT_EQ(cloneLoss.get().getDimensions(), batchDimensions);
        } else {
            ASSERT_EQ(cloneLoss.get().getDimensions(), dimensions);
        }

        ASSERT_TRUE(clone->getPredictions() == meanSquaredError.getFeatureInput());
        ASSERT_TRUE(clone->getLoss() == meanSquaredError.getFeatureOutput());

        ASSERT_EQ(meanSquaredError.getId(), clone->getId());
        ASSERT_GT(meanSquaredError.getId(), 1u);

        ASSERT_TRUE(meanSquaredError == *clone);
        ASSERT_FALSE(meanSquaredError != *clone);
        ASSERT_FALSE(meanSquaredError > *clone);
        ASSERT_FALSE(meanSquaredError < *clone);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
