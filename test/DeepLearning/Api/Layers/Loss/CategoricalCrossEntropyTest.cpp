#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace std;

using namespace Thor;

TEST(CategoricalCrossEntropy, ClassIndexLabelsBatchLossBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        // API layer does not have a batch dimension
        vector<uint64_t> labelDimensions = {1};
        uint64_t numClasses = 1UL + (rand() % 1000);
        vector<uint64_t> predictionsDimensions = {numClasses};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 3;
        if (r == 0)
            labelsDataType = Tensor::DataType::UINT8;
        else if (r == 1)
            labelsDataType = Tensor::DataType::UINT16;
        else
            labelsDataType = Tensor::DataType::UINT32;

        Tensor predictions(predictionsDataType, predictionsDimensions);
        Tensor labels(labelsDataType, labelDimensions);

        CategoricalCrossEntropy::Builder crossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                   .network(network)
                                                                   .predictions(predictions)
                                                                   .reportsBatchLoss()
                                                                   .receivesClassIndexLabels(2 + (rand() % 10000))
                                                                   .lossDataType(lossDataType)
                                                                   .labels(labels);
        CategoricalCrossEntropy crossEntropy = crossEntropyBuilder.build();

        ASSERT_TRUE(crossEntropy.isInitialized());

        Optional<Tensor> actualInput = crossEntropy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), predictionsDimensions);

        Optional<Tensor> actualLabels = crossEntropy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), labelDimensions);

        Optional<Tensor> actualPredictions = crossEntropy.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), predictionsDimensions);

        Optional<Tensor> actualLoss = crossEntropy.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(actualLoss.get().getDimensions(), vector<uint64_t>(1, 1));

        shared_ptr<Layer> cloneLayer = crossEntropy.clone();
        CategoricalCrossEntropy *clone = dynamic_cast<CategoricalCrossEntropy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), predictionsDimensions);

        Optional<Tensor> cloneLabels = clone->getLabels();
        ASSERT_TRUE(cloneLabels.isPresent());
        ASSERT_EQ(cloneLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(cloneLabels.get().getDimensions(), labelDimensions);

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), predictionsDimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(cloneLoss.get().getDimensions(), vector<uint64_t>(1, 1));

        ASSERT_EQ(crossEntropy.getId(), clone->getId());
        ASSERT_GT(crossEntropy.getId(), 1u);

        ASSERT_TRUE(crossEntropy == *clone);
        ASSERT_FALSE(crossEntropy != *clone);
        ASSERT_FALSE(crossEntropy > *clone);
        ASSERT_FALSE(crossEntropy < *clone);
    }
}

TEST(CategoricalCrossEntropy, OneHotLabelsClasswiseLossBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions = {2UL + (rand() % 300)};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 5;
        if (r == 0)
            labelsDataType = Tensor::DataType::UINT8;
        else if (r == 1)
            labelsDataType = Tensor::DataType::UINT16;
        else if (r == 2)
            labelsDataType = Tensor::DataType::UINT32;
        else if (r == 3)
            labelsDataType = Tensor::DataType::FP16;
        else
            labelsDataType = Tensor::DataType::FP32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        CategoricalCrossEntropy::Builder crossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                   .network(network)
                                                                   .predictions(predictions)
                                                                   .reportsClasswiseLoss()
                                                                   .receivesOneHotLabels()
                                                                   .lossDataType(lossDataType)
                                                                   .labels(labels);
        CategoricalCrossEntropy crossEntropy = crossEntropyBuilder.build();

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
        ASSERT_EQ(actualLoss.get().getDimensions(), dimensions);

        shared_ptr<Layer> cloneLayer = crossEntropy.clone();
        CategoricalCrossEntropy *clone = dynamic_cast<CategoricalCrossEntropy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLabels = clone->getLabels();
        ASSERT_TRUE(cloneLabels.isPresent());
        ASSERT_EQ(cloneLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(cloneLabels.get().getDimensions(), dimensions);

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(cloneLoss.get().getDimensions(), dimensions);

        ASSERT_EQ(crossEntropy.getId(), clone->getId());
        ASSERT_GT(crossEntropy.getId(), 1u);

        ASSERT_TRUE(crossEntropy == *clone);
        ASSERT_FALSE(crossEntropy != *clone);
        ASSERT_FALSE(crossEntropy > *clone);
        ASSERT_FALSE(crossEntropy < *clone);
    }
}

TEST(CategoricalCrossEntropy, OneHotLabelsElementwiseLossBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions = {2UL + (rand() % 300)};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 5;
        if (r == 0)
            labelsDataType = Tensor::DataType::UINT8;
        else if (r == 1)
            labelsDataType = Tensor::DataType::UINT16;
        else if (r == 2)
            labelsDataType = Tensor::DataType::UINT32;
        else if (r == 3)
            labelsDataType = Tensor::DataType::FP16;
        else if (r == 4)
            labelsDataType = Tensor::DataType::FP32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        CategoricalCrossEntropy::Builder crossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                   .network(network)
                                                                   .predictions(predictions)
                                                                   .reportsElementwiseLoss()
                                                                   .receivesOneHotLabels()
                                                                   .lossDataType(lossDataType)
                                                                   .labels(labels);
        CategoricalCrossEntropy crossEntropy = crossEntropyBuilder.build();

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

        vector<uint64_t> batchLossDimensions = {1UL};
        Optional<Tensor> actualLoss = crossEntropy.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(actualLoss.get().getDimensions(), batchLossDimensions);

        shared_ptr<Layer> cloneLayer = crossEntropy.clone();
        CategoricalCrossEntropy *clone = dynamic_cast<CategoricalCrossEntropy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLabels = clone->getLabels();
        ASSERT_TRUE(cloneLabels.isPresent());
        ASSERT_EQ(cloneLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(cloneLabels.get().getDimensions(), dimensions);

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(cloneLoss.get().getDimensions(), batchLossDimensions);

        ASSERT_EQ(crossEntropy.getId(), clone->getId());
        ASSERT_GT(crossEntropy.getId(), 1u);

        ASSERT_TRUE(crossEntropy == *clone);
        ASSERT_FALSE(crossEntropy != *clone);
        ASSERT_FALSE(crossEntropy > *clone);
        ASSERT_FALSE(crossEntropy < *clone);
    }
}

TEST(CategoricalCrossEntropy, ClassIndexLabelsRawLossBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions = {1};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 3;
        if (r == 0)
            labelsDataType = Tensor::DataType::UINT8;
        else if (r == 1)
            labelsDataType = Tensor::DataType::UINT16;
        else
            labelsDataType = Tensor::DataType::UINT32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        uint32_t numClasses = 2 + (rand() % 10000);
        CategoricalCrossEntropy::Builder crossEntropyBuilder = CategoricalCrossEntropy::Builder()
                                                                   .network(network)
                                                                   .predictions(predictions)
                                                                   .reportsRawLoss()
                                                                   .receivesClassIndexLabels(numClasses)
                                                                   .lossDataType(lossDataType)
                                                                   .labels(labels);
        CategoricalCrossEntropy crossEntropy = crossEntropyBuilder.build();

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
        ASSERT_EQ(actualLoss.get().getDimensions(), vector<uint64_t>(1, numClasses));

        shared_ptr<Layer> cloneLayer = crossEntropy.clone();
        CategoricalCrossEntropy *clone = dynamic_cast<CategoricalCrossEntropy *>(cloneLayer.get());
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
        ASSERT_EQ(cloneLoss.get().getDimensions(), vector<uint64_t>(1, numClasses));

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
