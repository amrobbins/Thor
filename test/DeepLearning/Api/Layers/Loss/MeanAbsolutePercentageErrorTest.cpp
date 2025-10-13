#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Loss/MeanAbsolutePercentageError.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;

TEST(MeanAbsolutePercentageError, Builds) {
    srand(time(nullptr));

    for (uint32_t i = 0; i < 10; ++i) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions.push_back(1 + (rand() % 1000));
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor predictions(predictionsDataType, dimensions);

        Tensor::DataType labelsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor labels(labelsDataType, dimensions);

        MeanAbsolutePercentageError::Builder meanAbsolutePercentageErrorBuilder =
            MeanAbsolutePercentageError::Builder().network(network).predictions(predictions).labels(labels);

        uint32_t shape = rand() % 4;
        if (shape == 0) {
            meanAbsolutePercentageErrorBuilder.reportsBatchLoss();
        } else if (shape == 1) {
            meanAbsolutePercentageErrorBuilder.reportsElementwiseLoss();
        } else if (shape == 2) {
            meanAbsolutePercentageErrorBuilder.reportsPerOutputLoss();
        } else if (shape == 3) {
            meanAbsolutePercentageErrorBuilder.reportsRawLoss();
        } else {
            assert(false);
        }
        vector<uint64_t> batchDimensions = {1};
        vector<uint64_t> elementwiseDimensions = {1};
        vector<uint64_t> perOutputDimensions = {dimensions[0]};
        vector<uint64_t> rawLossDimensions = dimensions;

        MeanAbsolutePercentageError meanAbsolutePercentageError = meanAbsolutePercentageErrorBuilder.build();

        ASSERT_TRUE(meanAbsolutePercentageError.isInitialized());

        Optional<Tensor> actualLabels = meanAbsolutePercentageError.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualPredictions = meanAbsolutePercentageError.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

        Optional<Tensor> actualLoss = meanAbsolutePercentageError.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), predictionsDataType);
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

        ASSERT_TRUE(meanAbsolutePercentageError.getPredictions() == meanAbsolutePercentageError.getFeatureInput());
        ASSERT_TRUE(meanAbsolutePercentageError.getLoss() == meanAbsolutePercentageError.getFeatureOutput());

        shared_ptr<Layer> cloneLayer = meanAbsolutePercentageError.clone();
        MeanAbsolutePercentageError *clone = dynamic_cast<MeanAbsolutePercentageError *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), predictionsDataType);
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

        ASSERT_TRUE(clone->getPredictions() == meanAbsolutePercentageError.getFeatureInput());
        ASSERT_TRUE(clone->getLoss() == meanAbsolutePercentageError.getFeatureOutput());

        ASSERT_EQ(meanAbsolutePercentageError.getId(), clone->getId());
        ASSERT_GT(meanAbsolutePercentageError.getId(), 1u);

        ASSERT_TRUE(meanAbsolutePercentageError == *clone);
        ASSERT_FALSE(meanAbsolutePercentageError != *clone);
        ASSERT_FALSE(meanAbsolutePercentageError > *clone);
        ASSERT_FALSE(meanAbsolutePercentageError < *clone);
    }
}
