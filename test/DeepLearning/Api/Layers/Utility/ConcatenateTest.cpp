#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Layers/Utility/Concatenate.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;

TEST(UtilityApi, ConcatenateBuilds) {
    srand(time(nullptr));

    Network network;

    uint32_t numDimensions = 1 + (rand() % 4);
    uint32_t concatenationAxis = rand() % numDimensions;
    uint32_t concatenationAxisSize = 1 + (rand() % 50);
    vector<uint64_t> concatenatedDimensions(numDimensions, 0U);
    uint32_t numTensors = 1 + (rand() % 5);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    vector<uint64_t> fixedDimensionSize;
    for(uint32_t d = 0; d < numDimensions; ++d) {
        fixedDimensionSize.push_back(1 + (rand() % 5));
    }
    concatenatedDimensions = fixedDimensionSize;
    concatenatedDimensions[concatenationAxis] = 0;

    vector<vector<uint64_t>> tensorDimensions;
    for (uint32_t t = 0; t < numTensors; ++t) {
        tensorDimensions.emplace_back();
        for(uint32_t d = 0; d < numDimensions; ++d) {
            uint64_t size;
            if (d == concatenationAxis) {
                size = 1 + (rand() % 5);
                concatenatedDimensions[concatenationAxis] += size;
            } else {
                size = fixedDimensionSize[d];
            }
            tensorDimensions[t].push_back(size);
        }
    }

    vector<Tensor> tensors;
    for (uint32_t t = 0; t < numTensors; ++t) {
        tensors.push_back(Tensor(dataType, tensorDimensions[t]));
    }

    Concatenate::Builder concatenateBuilder = Concatenate::Builder().network(network).concatenationAxis(concatenationAxis);
    for (uint32_t t = 0; t < numTensors; ++t) {
        concatenateBuilder.featureInput(tensors[t]);
    }
    Concatenate concatenate = concatenateBuilder.build();

    ASSERT_TRUE(concatenate.isInitialized());

    vector<Tensor> actualInputs = concatenate.getFeatureInputs();
    ASSERT_EQ(actualInputs.size(), numTensors);
    for (uint32_t t = 0; t < numTensors; ++t) {
        ASSERT_EQ(actualInputs[t].getDataType(), dataType);
        ASSERT_EQ(actualInputs[t].getDimensions(), tensorDimensions[t]);
    }

    vector<Tensor> actualOutputs = concatenate.getFeatureOutputs();
    ASSERT_EQ(actualOutputs.size(), 1U);
    ASSERT_EQ(actualOutputs[0].getDataType(), dataType);
    vector<uint64_t> outputDimensions = actualOutputs[0].getDimensions();
    ASSERT_EQ(outputDimensions.size(), numDimensions);
    ASSERT_EQ(actualOutputs[0].getDimensions(), concatenatedDimensions);


    shared_ptr<Layer> cloneLayer = concatenate.clone();
    Concatenate *clone = dynamic_cast<Concatenate *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    vector<Tensor> cloneInputs = clone->getFeatureInputs();
    ASSERT_EQ(cloneInputs.size(), numTensors);
    for (uint32_t t = 0; t < numTensors; ++t) {
        ASSERT_EQ(cloneInputs[t].getDataType(), dataType);
        ASSERT_EQ(cloneInputs[t].getDimensions(), tensorDimensions[t]);
    }

    ASSERT_EQ(concatenate.getId(), clone->getId());
    ASSERT_GT(concatenate.getId(), 1u);

    vector<Tensor> cloneOutputs = clone->getFeatureOutputs();
    ASSERT_EQ(cloneOutputs.size(), 1U);
    ASSERT_EQ(cloneOutputs[0].getDataType(), dataType);
    outputDimensions.clear();
    outputDimensions = cloneOutputs[0].getDimensions();
    ASSERT_EQ(outputDimensions.size(), numDimensions);
    ASSERT_EQ(cloneOutputs[0].getDimensions(), concatenatedDimensions);

    ASSERT_TRUE(concatenate == *clone);
    ASSERT_FALSE(concatenate != *clone);
    ASSERT_FALSE(concatenate > *clone);
    ASSERT_FALSE(concatenate < *clone);
}
