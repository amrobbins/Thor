//#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
//
//#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
//
//#include "gtest/gtest.h"
//
//#include <stdio.h>
//#include <memory>
//
//using namespace Thor;
//using namespace std;
//
//TEST(UtilityApi, ConcatenateBuilds) {
//    // FIXME THIS IS FLATTEN
//    srand(time(nullptr));
//
//    Network network;
//
//    vector<uint64_t> inputDimensions;
//    int numInputDimensions = 2 + rand() % 6;
//    for (int i = 0; i < numInputDimensions; ++i)
//        inputDimensions.push_back(1 + (rand() % 1000));
//
//    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
//
//    Tensor featureInput(dataType, inputDimensions);
//    uint32_t numOutputDimensions = (rand() % (numInputDimensions - 1)) + 1;
//    Flatten flatten = Flatten::Builder().network(network).featureInput(featureInput).numOutputDimensions(numOutputDimensions).build();
//
//    ASSERT_TRUE(flatten.isInitialized());
//
//    Optional<Tensor> actualInput = flatten.getFeatureInput();
//    ASSERT_TRUE(actualInput.isPresent());
//    ASSERT_EQ(actualInput.get().getDataType(), dataType);
//    ASSERT_EQ(actualInput.get().getDimensions(), inputDimensions);
//
//    Optional<Tensor> actualOutput = flatten.getFeatureOutput();
//    ASSERT_TRUE(actualOutput.isPresent());
//    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
//    vector<uint64_t> outputDimensions = actualOutput.get().getDimensions();
//    ASSERT_EQ(outputDimensions.size(), numOutputDimensions);
//    uint64_t totalInputElements = 1;
//    for (uint32_t i = 0; i < inputDimensions.size(); ++i)
//        totalInputElements *= inputDimensions[i];
//    uint64_t totalOutputElements = 1;
//    for (uint32_t i = 0; i < outputDimensions.size(); ++i)
//        totalOutputElements *= outputDimensions[i];
//    ASSERT_EQ(totalInputElements, totalOutputElements);
//
//    shared_ptr<Layer> cloneLayer = flatten.clone();
//    Flatten *clone = dynamic_cast<Flatten *>(cloneLayer.get());
//    assert(clone != nullptr);
//
//    ASSERT_TRUE(clone->isInitialized());
//
//    Optional<Tensor> cloneInput = clone->getFeatureInput();
//    ASSERT_TRUE(cloneInput.isPresent());
//    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
//    ASSERT_EQ(cloneInput.get().getDimensions(), inputDimensions);
//
//    ASSERT_EQ(flatten.getId(), clone->getId());
//    ASSERT_GT(flatten.getId(), 1u);
//
//    actualOutput = clone->getFeatureOutput();
//    ASSERT_TRUE(actualOutput.isPresent());
//    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
//    outputDimensions = actualOutput.get().getDimensions();
//    ASSERT_EQ(outputDimensions.size(), numOutputDimensions);
//    totalInputElements = 1;
//    for (uint32_t i = 0; i < inputDimensions.size(); ++i)
//        totalInputElements *= inputDimensions[i];
//    totalOutputElements = 1;
//    for (uint32_t i = 0; i < outputDimensions.size(); ++i)
//        totalOutputElements *= outputDimensions[i];
//    ASSERT_EQ(totalInputElements, totalOutputElements);
//
//    ASSERT_TRUE(flatten == *clone);
//    ASSERT_FALSE(flatten != *clone);
//    ASSERT_FALSE(flatten > *clone);
//    ASSERT_FALSE(flatten < *clone);
//}