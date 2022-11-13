#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace std;

using namespace Thor;

TEST(CategoricalAccuracy, ClassIndexLabelBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions = {1UL + (rand() % 1000), 1};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType accuracyDataType = Tensor::DataType::FP32;

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

        CategoricalAccuracy::Builder categoricalAccuracyBuilder =
            CategoricalAccuracy::Builder().network(network).predictions(predictions).labels(labels).receivesClassIndexLabels();
        CategoricalAccuracy categoricalAccuracy = categoricalAccuracyBuilder.build();

        ASSERT_TRUE(categoricalAccuracy.isInitialized());

        Optional<Tensor> actualInput = categoricalAccuracy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLabels = categoricalAccuracy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualAccuracy = categoricalAccuracy.getFeatureOutput();
        ASSERT_TRUE(actualAccuracy.isPresent());
        ASSERT_EQ(actualAccuracy.get().getDataType(), accuracyDataType);
        ASSERT_EQ(actualAccuracy.get().getDimensions(), vector<uint64_t>({1}));

        shared_ptr<Layer> cloneLayer = categoricalAccuracy.clone();
        CategoricalAccuracy *clone = dynamic_cast<CategoricalAccuracy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneAccuracy = clone->getFeatureOutput();
        ASSERT_TRUE(cloneAccuracy.isPresent());
        ASSERT_EQ(cloneAccuracy.get().getDataType(), accuracyDataType);
        ASSERT_EQ(cloneAccuracy.get().getDimensions(), vector<uint64_t>({1}));

        ASSERT_EQ(categoricalAccuracy.getId(), clone->getId());
        ASSERT_GT(categoricalAccuracy.getId(), 1u);

        ASSERT_TRUE(categoricalAccuracy == *clone);
        ASSERT_FALSE(categoricalAccuracy != *clone);
        ASSERT_FALSE(categoricalAccuracy > *clone);
        ASSERT_FALSE(categoricalAccuracy < *clone);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
