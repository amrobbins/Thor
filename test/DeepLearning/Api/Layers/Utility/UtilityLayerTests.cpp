#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using std::shared_ptr;

using namespace Thor;

TEST(NetworkInput, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    NetworkInput networkInput = NetworkInput::Builder().network(network).dimensions(dimensions).dataType(dataType).build();

    ASSERT_TRUE(networkInput.isInitialized());

    Optional<Tensor> actualInput = networkInput.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = networkInput.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = networkInput.clone();
    NetworkInput *clone = dynamic_cast<NetworkInput *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(networkInput.getId(), clone->getId());
    ASSERT_GT(networkInput.getId(), 1u);

    ASSERT_TRUE(networkInput == *clone);
    ASSERT_FALSE(networkInput != *clone);
    ASSERT_FALSE(networkInput > *clone);
    ASSERT_FALSE(networkInput < *clone);
}

TEST(NetworkOutput, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    Tensor::DataType outputDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    NetworkOutput networkOutput = NetworkOutput::Builder().network(network).inputTensor(featureInput).dataType(outputDataType).build();

    ASSERT_TRUE(networkOutput.isInitialized());

    Optional<Tensor> actualInput = networkOutput.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = networkOutput.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), outputDataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = networkOutput.clone();
    NetworkOutput *clone = dynamic_cast<NetworkOutput *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), outputDataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    ASSERT_EQ(networkOutput.getId(), clone->getId());
    ASSERT_GT(networkOutput.getId(), 1u);

    ASSERT_TRUE(networkOutput == *clone);
    ASSERT_FALSE(networkOutput != *clone);
    ASSERT_FALSE(networkOutput > *clone);
    ASSERT_FALSE(networkOutput < *clone);
}

TEST(Stub, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Stub stub = Stub::Builder().network(network).inputTensor(featureInput).build();

    ASSERT_TRUE(stub.isInitialized());

    Optional<Tensor> actualInput = stub.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = stub.clone();
    Stub *clone = dynamic_cast<Stub *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    ASSERT_EQ(stub.getId(), clone->getId());
    ASSERT_GT(stub.getId(), 1u);

    ASSERT_TRUE(stub == *clone);
    ASSERT_FALSE(stub != *clone);
    ASSERT_FALSE(stub > *clone);
    ASSERT_FALSE(stub < *clone);
}

TEST(Flatten, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> inputDimensions;
    int numInputDimensions = 2 + rand() % 6;
    for (int i = 0; i < numInputDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, inputDimensions);
    uint32_t numOutputDimensions = (rand() % (numInputDimensions - 1)) + 1;
    Flatten flatten = Flatten::Builder().network(network).featureInput(featureInput).numOutputDimensions(numOutputDimensions).build();

    ASSERT_TRUE(flatten.isInitialized());

    Optional<Tensor> actualInput = flatten.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), inputDimensions);

    Optional<Tensor> actualOutput = flatten.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    vector<uint64_t> outputDimensions = actualOutput.get().getDimensions();
    ASSERT_EQ(outputDimensions.size(), numOutputDimensions);
    uint64_t totalInputElements = 1;
    for (uint32_t i = 0; i < inputDimensions.size(); ++i)
        totalInputElements *= inputDimensions[i];
    uint64_t totalOutputElements = 1;
    for (uint32_t i = 0; i < outputDimensions.size(); ++i)
        totalOutputElements *= outputDimensions[i];
    ASSERT_EQ(totalInputElements, totalOutputElements);

    shared_ptr<Layer> cloneLayer = flatten.clone();
    Flatten *clone = dynamic_cast<Flatten *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), inputDimensions);

    ASSERT_EQ(flatten.getId(), clone->getId());
    ASSERT_GT(flatten.getId(), 1u);

    actualOutput = clone->getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    outputDimensions = actualOutput.get().getDimensions();
    ASSERT_EQ(outputDimensions.size(), numOutputDimensions);
    totalInputElements = 1;
    for (uint32_t i = 0; i < inputDimensions.size(); ++i)
        totalInputElements *= inputDimensions[i];
    totalOutputElements = 1;
    for (uint32_t i = 0; i < outputDimensions.size(); ++i)
        totalOutputElements *= outputDimensions[i];
    ASSERT_EQ(totalInputElements, totalOutputElements);

    ASSERT_TRUE(flatten == *clone);
    ASSERT_FALSE(flatten != *clone);
    ASSERT_FALSE(flatten > *clone);
    ASSERT_FALSE(flatten < *clone);
}

TEST(DropOut, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    float dropProportion = ((rand() % 100) + 1) / 1000.0f;

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    DropOut dropOut = DropOut::Builder().network(network).featureInput(featureInput).dropProportion(dropProportion).build();

    ASSERT_TRUE(dropOut.isInitialized());

    Optional<Tensor> actualInput = dropOut.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = dropOut.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    float actualDropProportion = dropOut.getDropProportion();
    ASSERT_EQ(actualDropProportion, dropProportion);

    shared_ptr<Layer> cloneLayer = dropOut.clone();
    DropOut *clone = dynamic_cast<DropOut *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    float cloneDropProportion = clone->getDropProportion();
    ASSERT_EQ(cloneDropProportion, dropProportion);

    ASSERT_EQ(dropOut.getId(), clone->getId());
    ASSERT_GT(dropOut.getId(), 1u);

    ASSERT_TRUE(dropOut == *clone);
    ASSERT_FALSE(dropOut != *clone);
    ASSERT_FALSE(dropOut > *clone);
    ASSERT_FALSE(dropOut < *clone);
}

TEST(BatchNormalizationSingleFeatureInput, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);

    double exponentialRunningAverageFactor = (1 + (rand() % 100)) / 1000.0f;

    double epsilon = (1 + (rand() % 100)) / 100000.0f;

    BatchNormalization batchNormalization = BatchNormalization::Builder()
                                                .network(network)
                                                .featureInput(featureInput)
                                                .exponentialRunningAverageFactor(exponentialRunningAverageFactor)
                                                .epsilon(epsilon)
                                                .build();

    ASSERT_TRUE(batchNormalization.isInitialized());

    Optional<Tensor> actualInput = batchNormalization.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = batchNormalization.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    double actualExponentialRunningAverageFactor = batchNormalization.getExponentialRunningAverageFactor();
    ASSERT_EQ(actualExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double actualEpsilon = batchNormalization.getEpsilon();
    ASSERT_EQ(actualEpsilon, epsilon);

    shared_ptr<Layer> cloneLayer = batchNormalization.clone();
    BatchNormalization *clone = dynamic_cast<BatchNormalization *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    double cloneExponentialRunningAverageFactor = clone->getExponentialRunningAverageFactor();
    ASSERT_EQ(cloneExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double cloneEpsilon = clone->getEpsilon();
    ASSERT_EQ(cloneEpsilon, epsilon);

    ASSERT_EQ(batchNormalization.getId(), clone->getId());
    ASSERT_GT(batchNormalization.getId(), 1u);

    ASSERT_TRUE(batchNormalization == *clone);
    ASSERT_FALSE(batchNormalization != *clone);
    ASSERT_FALSE(batchNormalization > *clone);
    ASSERT_FALSE(batchNormalization < *clone);
}

TEST(BatchNormalizationMultipleFeatureInputs, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions0 = 1 + rand() % 6;
    for (int i = 0; i < numDimensions0; ++i)
        dimensions.push_back(1 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    Tensor featureInput0(dataType, dimensions);
    Tensor featureInput1(dataType, dimensions);

    double exponentialRunningAverageFactor = (1 + (rand() % 100)) / 1000.0f;

    double epsilon = (1 + (rand() % 100)) / 100000.0f;

    BatchNormalization batchNormalization = BatchNormalization::Builder()
                                                .network(network)
                                                .featureInput(featureInput0)
                                                .featureInput(featureInput1)
                                                .exponentialRunningAverageFactor(exponentialRunningAverageFactor)
                                                .epsilon(epsilon)
                                                .build();

    ASSERT_TRUE(batchNormalization.isInitialized());

    vector<Tensor> featureInputs = batchNormalization.getFeatureInputs();
    vector<Tensor> featureOutputs = batchNormalization.getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(batchNormalization.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[1].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[1].getDimensions(), dimensions);

    double actualExponentialRunningAverageFactor = batchNormalization.getExponentialRunningAverageFactor();
    ASSERT_EQ(actualExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double actualEpsilon = batchNormalization.getEpsilon();
    ASSERT_EQ(actualEpsilon, epsilon);

    shared_ptr<Layer> cloneLayer = batchNormalization.clone();
    BatchNormalization *clone = dynamic_cast<BatchNormalization *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    featureInputs.clear();
    featureOutputs.clear();
    featureInputs = clone->getFeatureInputs();
    featureOutputs = clone->getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(batchNormalization.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[1].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[1].getDimensions(), dimensions);

    double cloneExponentialRunningAverageFactor = clone->getExponentialRunningAverageFactor();
    ASSERT_EQ(cloneExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double cloneEpsilon = clone->getEpsilon();
    ASSERT_EQ(cloneEpsilon, epsilon);

    ASSERT_EQ(batchNormalization.getId(), clone->getId());
    ASSERT_GT(batchNormalization.getId(), 1u);

    ASSERT_TRUE(batchNormalization == *clone);
    ASSERT_FALSE(batchNormalization != *clone);
    ASSERT_FALSE(batchNormalization > *clone);
    ASSERT_FALSE(batchNormalization < *clone);
}

TEST(PoolingNoPadding, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(10 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    uint32_t windowHeight = 1 + (rand() % dimensions[1]);
    uint32_t windowWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 10);

    Tensor featureInput(dataType, dimensions);
    Pooling pooling = Pooling::Builder()
                          .network(network)
                          .featureInput(featureInput)
                          .type(Pooling::Type::AVERAGE)
                          .windowHeight(windowHeight)
                          .windowWidth(windowWidth)
                          .verticalStride(verticalStride)
                          .horizontalStride(horizontalStride)
                          .noPadding()
                          .build();

    ASSERT_TRUE(pooling.isInitialized());

    Optional<Tensor> actualInput = pooling.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    vector<uint64_t> outputDimensions;
    uint32_t outputHeight = Pooling::Builder::computeOutputDimension(dimensions[1], verticalStride, windowHeight, 0);
    uint32_t outputWidth = Pooling::Builder::computeOutputDimension(dimensions[2], horizontalStride, windowWidth, 0);
    outputDimensions.push_back(dimensions[0]);
    outputDimensions.push_back(outputHeight);
    outputDimensions.push_back(outputWidth);

    Optional<Tensor> actualOutput = pooling.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(pooling.getWindowHeight(), windowHeight);
    ASSERT_EQ(pooling.getWindowWidth(), windowWidth);
    ASSERT_EQ(pooling.getVerticalStride(), verticalStride);
    ASSERT_EQ(pooling.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(pooling.getVerticalPadding(), 0u);
    ASSERT_EQ(pooling.getHorizontalPadding(), 0u);

    shared_ptr<Layer> cloneLayer = pooling.clone();
    Pooling *clone = dynamic_cast<Pooling *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(clone->getWindowHeight(), windowHeight);
    ASSERT_EQ(clone->getWindowWidth(), windowWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), 0u);
    ASSERT_EQ(clone->getHorizontalPadding(), 0u);

    ASSERT_EQ(pooling.getId(), clone->getId());
    ASSERT_GT(pooling.getId(), 1u);

    ASSERT_TRUE(pooling == *clone);
    ASSERT_FALSE(pooling != *clone);
    ASSERT_FALSE(pooling > *clone);
    ASSERT_FALSE(pooling < *clone);
}

TEST(PoolingSamePadding, Builds) {
    srand(time(nullptr));

    for (int test = 0; test < 50; ++test) {
        Network network;

        vector<uint64_t> dimensions;
        int numDimensions = 3;
        for (int i = 0; i < numDimensions; ++i)
            dimensions.push_back(10 + (rand() % 1000));
        Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        uint32_t windowHeight = 1 + (rand() % dimensions[1]);
        uint32_t windowWidth = 1 + (rand() % dimensions[2]);
        uint32_t verticalStride = 1;  // due to same padding
        uint32_t horizontalStride = 1;

        Tensor featureInput(dataType, dimensions);
        Pooling pooling = Pooling::Builder()
                              .network(network)
                              .featureInput(featureInput)
                              .type(Pooling::Type::MAX)
                              .windowHeight(windowHeight)
                              .windowWidth(windowWidth)
                              .verticalStride(verticalStride)
                              .horizontalStride(horizontalStride)
                              .samePadding()
                              .build();

        ASSERT_TRUE(pooling.isInitialized());

        Optional<Tensor> actualInput = pooling.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), dataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualOutput = pooling.getFeatureOutput();
        ASSERT_TRUE(actualOutput.isPresent());
        ASSERT_EQ(actualOutput.get().getDataType(), dataType);
        ASSERT_EQ(actualOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(actualOutput.get().getDimensions()[0], dimensions[0]);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = actualOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        uint32_t verticalPadding = Pooling::Builder::computeSamePadding(dimensions[1], verticalStride, windowHeight);
        uint32_t horizontalPadding = Pooling::Builder::computeSamePadding(dimensions[2], horizontalStride, windowWidth);
        ASSERT_EQ(pooling.getWindowHeight(), windowHeight);
        ASSERT_EQ(pooling.getWindowWidth(), windowWidth);
        ASSERT_EQ(pooling.getVerticalStride(), verticalStride);
        ASSERT_EQ(pooling.getHorizontalStride(), horizontalStride);
        ASSERT_EQ(pooling.getVerticalPadding(), verticalPadding);
        ASSERT_EQ(pooling.getHorizontalPadding(), horizontalPadding);

        shared_ptr<Layer> cloneLayer = pooling.clone();
        Pooling *clone = dynamic_cast<Pooling *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), dataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneOutput = clone->getFeatureOutput();
        ASSERT_TRUE(cloneOutput.isPresent());
        ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
        ASSERT_EQ(cloneOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(cloneOutput.get().getDimensions()[0], dimensions[0]);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = cloneOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        ASSERT_EQ(clone->getWindowHeight(), windowHeight);
        ASSERT_EQ(clone->getWindowWidth(), windowWidth);
        ASSERT_EQ(clone->getVerticalStride(), verticalStride);
        ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
        ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
        ASSERT_EQ(clone->getHorizontalPadding(), horizontalPadding);

        ASSERT_EQ(pooling.getId(), clone->getId());
        ASSERT_GT(pooling.getId(), 1u);

        ASSERT_TRUE(pooling == *clone);
        ASSERT_FALSE(pooling != *clone);
        ASSERT_FALSE(pooling > *clone);
        ASSERT_FALSE(pooling < *clone);
    }
}

TEST(PoolingDefaultPadding, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(10 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    uint32_t windowHeight = 1 + (rand() % dimensions[1]);
    uint32_t windowWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 10);

    Tensor featureInput(dataType, dimensions);
    Pooling pooling = Pooling::Builder()
                          .network(network)
                          .featureInput(featureInput)
                          .type(Pooling::Type::AVERAGE)
                          .windowHeight(windowHeight)
                          .windowWidth(windowWidth)
                          .verticalStride(verticalStride)
                          .horizontalStride(horizontalStride)
                          .build();

    ASSERT_TRUE(pooling.isInitialized());

    Optional<Tensor> actualInput = pooling.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    vector<uint64_t> outputDimensions;
    uint32_t outputHeight = Pooling::Builder::computeOutputDimension(dimensions[1], verticalStride, windowHeight, 0);
    uint32_t outputWidth = Pooling::Builder::computeOutputDimension(dimensions[2], horizontalStride, windowWidth, 0);
    outputDimensions.push_back(dimensions[0]);
    outputDimensions.push_back(outputHeight);
    outputDimensions.push_back(outputWidth);

    Optional<Tensor> actualOutput = pooling.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(pooling.getWindowHeight(), windowHeight);
    ASSERT_EQ(pooling.getWindowWidth(), windowWidth);
    ASSERT_EQ(pooling.getVerticalStride(), verticalStride);
    ASSERT_EQ(pooling.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(pooling.getVerticalPadding(), 0u);
    ASSERT_EQ(pooling.getHorizontalPadding(), 0u);

    shared_ptr<Layer> cloneLayer = pooling.clone();
    Pooling *clone = dynamic_cast<Pooling *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(clone->getWindowHeight(), windowHeight);
    ASSERT_EQ(clone->getWindowWidth(), windowWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), 0u);
    ASSERT_EQ(clone->getHorizontalPadding(), 0u);

    ASSERT_EQ(pooling.getId(), clone->getId());
    ASSERT_GT(pooling.getId(), 1u);

    ASSERT_TRUE(pooling == *clone);
    ASSERT_FALSE(pooling != *clone);
    ASSERT_FALSE(pooling > *clone);
    ASSERT_FALSE(pooling < *clone);
}

TEST(PoolingSpecifiedPadding, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(10 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    uint32_t windowHeight = 1 + (rand() % dimensions[1]);
    uint32_t windowWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 10);
    uint32_t verticalPadding = rand() % windowHeight;
    uint32_t horizontalPadding = rand() % windowWidth;

    Tensor featureInput(dataType, dimensions);
    Pooling pooling = Pooling::Builder()
                          .network(network)
                          .featureInput(featureInput)
                          .type(Pooling::Type::MAX)
                          .windowHeight(windowHeight)
                          .windowWidth(windowWidth)
                          .verticalStride(verticalStride)
                          .horizontalStride(horizontalStride)
                          .verticalPadding(verticalPadding)
                          .horizontalPadding(horizontalPadding)
                          .build();

    ASSERT_TRUE(pooling.isInitialized());

    Optional<Tensor> actualInput = pooling.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    vector<uint64_t> outputDimensions;
    uint32_t outputHeight = Pooling::Builder::computeOutputDimension(dimensions[1], verticalStride, windowHeight, verticalPadding);
    uint32_t outputWidth = Pooling::Builder::computeOutputDimension(dimensions[2], horizontalStride, windowWidth, horizontalPadding);
    outputDimensions.push_back(dimensions[0]);
    outputDimensions.push_back(outputHeight);
    outputDimensions.push_back(outputWidth);

    Optional<Tensor> actualOutput = pooling.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(pooling.getWindowHeight(), windowHeight);
    ASSERT_EQ(pooling.getWindowWidth(), windowWidth);
    ASSERT_EQ(pooling.getVerticalStride(), verticalStride);
    ASSERT_EQ(pooling.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(pooling.getVerticalPadding(), verticalPadding);
    ASSERT_EQ(pooling.getHorizontalPadding(), horizontalPadding);

    shared_ptr<Layer> cloneLayer = pooling.clone();
    Pooling *clone = dynamic_cast<Pooling *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(clone->getWindowHeight(), windowHeight);
    ASSERT_EQ(clone->getWindowWidth(), windowWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
    ASSERT_EQ(clone->getHorizontalPadding(), horizontalPadding);

    ASSERT_EQ(pooling.getId(), clone->getId());
    ASSERT_GT(pooling.getId(), 1u);

    ASSERT_TRUE(pooling == *clone);
    ASSERT_FALSE(pooling != *clone);
    ASSERT_FALSE(pooling > *clone);
    ASSERT_FALSE(pooling < *clone);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
