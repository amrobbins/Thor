#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace std;

using namespace Thor;

TEST(BinaryAccuracy, Builds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions = {1};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType accuracyDataType = Tensor::DataType::FP32;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 8;
        if (r == 0)
            labelsDataType = Tensor::DataType::UINT8;
        else if (r == 1)
            labelsDataType = Tensor::DataType::UINT16;
        else if (r == 2)
            labelsDataType = Tensor::DataType::UINT32;
        else if (r == 3)
            labelsDataType = Tensor::DataType::INT8;
        else if (r == 4)
            labelsDataType = Tensor::DataType::INT16;
        else if (r == 5)
            labelsDataType = Tensor::DataType::INT32;
        else if (r == 6)
            labelsDataType = Tensor::DataType::FP16;
        else if (r == 7)
            labelsDataType = Tensor::DataType::FP32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        BinaryAccuracy::Builder binaryAccuracyBuilder = BinaryAccuracy::Builder().network(network).predictions(predictions).labels(labels);
        BinaryAccuracy binaryAccuracy = binaryAccuracyBuilder.build();

        ASSERT_TRUE(binaryAccuracy.isInitialized());

        Optional<Tensor> actualInput = binaryAccuracy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLabels = binaryAccuracy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualAccuracy = binaryAccuracy.getFeatureOutput();
        ASSERT_TRUE(actualAccuracy.isPresent());
        ASSERT_EQ(actualAccuracy.get().getDataType(), accuracyDataType);
        ASSERT_EQ(actualAccuracy.get().getDimensions(), vector<uint64_t>({1}));

        shared_ptr<Layer> cloneLayer = binaryAccuracy.clone();
        BinaryAccuracy *clone = dynamic_cast<BinaryAccuracy *>(cloneLayer.get());
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

        Optional<Tensor> cloneAccuracy = clone->getFeatureOutput();
        ASSERT_TRUE(cloneAccuracy.isPresent());
        ASSERT_EQ(cloneAccuracy.get().getDataType(), accuracyDataType);
        ASSERT_EQ(cloneAccuracy.get().getDimensions(), vector<uint64_t>({1}));

        ASSERT_EQ(binaryAccuracy.getId(), clone->getId());
        ASSERT_GT(binaryAccuracy.getId(), 1u);

        ASSERT_TRUE(binaryAccuracy == *clone);
        ASSERT_FALSE(binaryAccuracy != *clone);
        ASSERT_FALSE(binaryAccuracy > *clone);
        ASSERT_FALSE(binaryAccuracy < *clone);
    }
}
