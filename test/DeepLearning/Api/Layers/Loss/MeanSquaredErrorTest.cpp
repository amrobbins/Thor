#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;

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

        bool setLossDataType = rand() % 2 == 0;
        Tensor::DataType lossDataType;

        MeanSquaredError::Builder meanSquaredErrorBuilder =
            MeanSquaredError::Builder().network(network).predictions(predictions).labels(labels);

        uint32_t shape = rand() % 4;
        if (shape == 0) {
            meanSquaredErrorBuilder.reportsBatchLoss();
        } else if (shape == 1) {
            meanSquaredErrorBuilder.reportsElementwiseLoss();
        } else if (shape == 2) {
            meanSquaredErrorBuilder.reportsPerOutputLoss();
        } else if (shape == 3) {
            meanSquaredErrorBuilder.reportsRawLoss();
        } else {
            assert(false);
        }
        vector<uint64_t> batchDimensions = {1};
        vector<uint64_t> elementwiseDimensions = {dimensions[0]};
        vector<uint64_t> perOutputDimensions = {dimensions[1]};
        vector<uint64_t> rawLossDimensions = dimensions;

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
        if (shape == 0) {
            ASSERT_EQ(actualLoss.get().getDimensions(), batchDimensions);
        } else if (shape == 1) {
            ASSERT_EQ(actualLoss.get().getDimensions(), elementwiseDimensions);
        } else if (shape == 2) {
            ASSERT_EQ(actualLoss.get().getDimensions(), perOutputDimensions);
        } else if (shape == 3) {
            ASSERT_EQ(actualLoss.get().getDimensions(), rawLossDimensions);
        } else {
            assert(false);
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
        if (shape == 0) {
            ASSERT_EQ(cloneLoss.get().getDimensions(), batchDimensions);
        } else if (shape == 1) {
            ASSERT_EQ(cloneLoss.get().getDimensions(), elementwiseDimensions);
        } else if (shape == 2) {
            ASSERT_EQ(cloneLoss.get().getDimensions(), perOutputDimensions);
        } else if (shape == 3) {
            ASSERT_EQ(cloneLoss.get().getDimensions(), rawLossDimensions);
        } else {
            assert(false);
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
