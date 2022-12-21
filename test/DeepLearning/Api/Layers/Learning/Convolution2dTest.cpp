#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "Thor.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using std::shared_ptr;

using namespace Thor;
using namespace std;

TEST(Convolution2dSingleFeatureInputNoPadding, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);

    uint32_t numOutputChannels = 1 + (rand() % 1000);

    uint32_t filterHeight = 1 + (rand() % dimensions[1]);
    uint32_t filterWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 10);

    bool hasBias = rand() % 2;

    UniformRandom::Builder uniformRandomInitializerBuilder;
    Tanh::Builder tanhBuilder;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
    double exponentialRunningAverageFactor = (1 + (rand() % 1000)) / 1000.0f;
    double epsilon = (1 + (rand() % 1000)) / 1000.0f;

    Convolution2d convolution2d = Convolution2d::Builder()
                                      .network(network)
                                      .featureInput(featureInput)
                                      .numOutputChannels(numOutputChannels)
                                      .filterHeight(filterHeight)
                                      .filterWidth(filterWidth)
                                      .verticalStride(verticalStride)
                                      .horizontalStride(horizontalStride)
                                      .noPadding()
                                      .hasBias(hasBias)
                                      .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                                      .biasInitializerBuilder(uniformRandomInitializerBuilder)
                                      .activationBuilder(tanhBuilder)
                                      .batchNormalization(exponentialRunningAverageFactor, epsilon)
                                      .dropOut(dropProportion)
                                      .build();

    ASSERT_TRUE(convolution2d.isInitialized());

    Optional<Tensor> actualInput = convolution2d.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    uint64_t outputHeight = Convolution2d::Builder::computeOutputDimension(dimensions[1], verticalStride, filterHeight, 0);
    uint64_t outputWidth = Convolution2d::Builder::computeOutputDimension(dimensions[2], horizontalStride, filterWidth, 0);
    vector<uint64_t> outputDimensions = {numOutputChannels, outputHeight, outputWidth};
    Optional<Tensor> actualOutput = convolution2d.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(convolution2d.getFilterHeight(), filterHeight);
    ASSERT_EQ(convolution2d.getFilterWidth(), filterWidth);
    ASSERT_EQ(convolution2d.getVerticalStride(), verticalStride);
    ASSERT_EQ(convolution2d.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(convolution2d.getVerticalPadding(), 0u);
    ASSERT_EQ(convolution2d.getHoriztonalPadding(), 0u);

    shared_ptr<Layer> cloneLayer = convolution2d.clone();
    Convolution2d *clone = dynamic_cast<Convolution2d *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(clone->getFilterHeight(), filterHeight);
    ASSERT_EQ(clone->getFilterWidth(), filterWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), 0u);
    ASSERT_EQ(clone->getHoriztonalPadding(), 0u);

    ASSERT_EQ(convolution2d.getId(), clone->getId());
    ASSERT_GT(convolution2d.getId(), 1u);

    ASSERT_TRUE(convolution2d == *clone);
    ASSERT_FALSE(convolution2d != *clone);
    ASSERT_FALSE(convolution2d > *clone);
    ASSERT_FALSE(convolution2d < *clone);
}

TEST(Convolution2dSingleFeatureInputSpecifiedPadding, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);

    uint32_t numOutputChannels = 1 + (rand() % 1000);

    uint32_t filterHeight = 1 + (rand() % dimensions[1]);
    uint32_t filterWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 10);

    uint32_t verticalPadding = rand() % filterHeight;
    uint32_t horizontalPadding = rand() % filterWidth;

    bool hasBias = rand() % 2;

    UniformRandom::Builder uniformRandomInitializerBuilder;
    Tanh::Builder tanhBuilder;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
    double exponentialRunningAverageFactor = (1 + (rand() % 1000)) / 1000.0f;
    double epsilon = (1 + (rand() % 1000)) / 1000.0f;

    Convolution2d convolution2d = Convolution2d::Builder()
                                      .network(network)
                                      .featureInput(featureInput)
                                      .numOutputChannels(numOutputChannels)
                                      .filterHeight(filterHeight)
                                      .filterWidth(filterWidth)
                                      .verticalStride(verticalStride)
                                      .horizontalStride(horizontalStride)
                                      .verticalPadding(verticalPadding)
                                      .horizontalPadding(horizontalPadding)
                                      .hasBias(hasBias)
                                      .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                                      .biasInitializerBuilder(uniformRandomInitializerBuilder)
                                      .activationBuilder(tanhBuilder)
                                      .batchNormalization(exponentialRunningAverageFactor, epsilon)
                                      .dropOut(dropProportion)
                                      .build();

    ASSERT_TRUE(convolution2d.isInitialized());

    Optional<Tensor> actualInput = convolution2d.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    uint64_t outputHeight = Convolution2d::Builder::computeOutputDimension(dimensions[1], verticalStride, filterHeight, verticalPadding);
    uint64_t outputWidth = Convolution2d::Builder::computeOutputDimension(dimensions[2], horizontalStride, filterWidth, horizontalPadding);
    vector<uint64_t> outputDimensions = {numOutputChannels, outputHeight, outputWidth};
    Optional<Tensor> actualOutput = convolution2d.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(actualOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(convolution2d.getFilterHeight(), filterHeight);
    ASSERT_EQ(convolution2d.getFilterWidth(), filterWidth);
    ASSERT_EQ(convolution2d.getVerticalStride(), verticalStride);
    ASSERT_EQ(convolution2d.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(convolution2d.getVerticalPadding(), verticalPadding);
    ASSERT_EQ(convolution2d.getHoriztonalPadding(), horizontalPadding);

    shared_ptr<Layer> cloneLayer = convolution2d.clone();
    Convolution2d *clone = dynamic_cast<Convolution2d *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(cloneOutput.get().getDimensions(), outputDimensions);

    ASSERT_EQ(clone->getFilterHeight(), filterHeight);
    ASSERT_EQ(clone->getFilterWidth(), filterWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
    ASSERT_EQ(clone->getHoriztonalPadding(), horizontalPadding);

    ASSERT_EQ(convolution2d.getId(), clone->getId());
    ASSERT_GT(convolution2d.getId(), 1u);

    ASSERT_TRUE(convolution2d == *clone);
    ASSERT_FALSE(convolution2d != *clone);
    ASSERT_FALSE(convolution2d > *clone);
    ASSERT_FALSE(convolution2d < *clone);
}

TEST(Convolution2dSingleFeatureInputSamePadding, Builds) {
    srand(time(nullptr));

    for (uint32_t test = 0; test < 25; ++test) {
        Network network;

        vector<uint64_t> dimensions;
        int numDimensions = 3;
        for (int i = 0; i < numDimensions; ++i)
            dimensions.push_back(1 + (rand() % 1000));

        Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor featureInput(dataType, dimensions);

        uint32_t numOutputChannels = 1 + (rand() % 1000);

        uint32_t filterHeight = 1 + (rand() % dimensions[1]);
        uint32_t filterWidth = 1 + (rand() % dimensions[2]);
        uint32_t verticalStride = 1;  // due to same padding
        uint32_t horizontalStride = 1;

        bool hasBias = rand() % 2;

        float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
        double exponentialRunningAverageFactor = (1 + (rand() % 1000)) / 1000.0f;
        double epsilon = (1 + (rand() % 1000)) / 1000.0f;

        Convolution2d convolution2d = Convolution2d::Builder()
                                          .network(network)
                                          .featureInput(featureInput)
                                          .numOutputChannels(numOutputChannels)
                                          .filterHeight(filterHeight)
                                          .filterWidth(filterWidth)
                                          .verticalStride(verticalStride)
                                          .horizontalStride(horizontalStride)
                                          .samePadding()
                                          .hasBias(hasBias)
                                          .batchNormalization(exponentialRunningAverageFactor, epsilon)
                                          .dropOut(dropProportion)
                                          .build();

        ASSERT_TRUE(convolution2d.isInitialized());

        Optional<Tensor> actualInput = convolution2d.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), dataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        uint32_t verticalPadding = Convolution2d::Builder::computeSamePadding(dimensions[1], verticalStride, filterHeight);
        uint32_t horizontalPadding = Convolution2d::Builder::computeSamePadding(dimensions[2], horizontalStride, filterWidth);
        Optional<Tensor> actualOutput = convolution2d.getFeatureOutput();
        ASSERT_TRUE(actualOutput.isPresent());
        ASSERT_EQ(actualOutput.get().getDataType(), Tensor::DataType::FP16);
        ASSERT_EQ(actualOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(actualOutput.get().getDimensions()[0], numOutputChannels);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = actualOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        ASSERT_EQ(convolution2d.getFilterHeight(), filterHeight);
        ASSERT_EQ(convolution2d.getFilterWidth(), filterWidth);
        ASSERT_EQ(convolution2d.getVerticalStride(), verticalStride);
        ASSERT_EQ(convolution2d.getHorizontalStride(), horizontalStride);
        ASSERT_EQ(convolution2d.getVerticalPadding(), verticalPadding);
        ASSERT_EQ(convolution2d.getHoriztonalPadding(), horizontalPadding);

        shared_ptr<Layer> cloneLayer = convolution2d.clone();
        Convolution2d *clone = dynamic_cast<Convolution2d *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), dataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneOutput = clone->getFeatureOutput();
        ASSERT_TRUE(cloneOutput.isPresent());
        ASSERT_EQ(cloneOutput.get().getDataType(), Tensor::DataType::FP16);
        ASSERT_EQ(cloneOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(cloneOutput.get().getDimensions()[0], numOutputChannels);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = cloneOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        ASSERT_EQ(clone->getFilterHeight(), filterHeight);
        ASSERT_EQ(clone->getFilterWidth(), filterWidth);
        ASSERT_EQ(clone->getVerticalStride(), verticalStride);
        ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
        ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
        ASSERT_EQ(clone->getHoriztonalPadding(), horizontalPadding);

        ASSERT_EQ(convolution2d.getId(), clone->getId());
        ASSERT_GT(convolution2d.getId(), 1u);

        ASSERT_TRUE(convolution2d == *clone);
        ASSERT_FALSE(convolution2d != *clone);
        ASSERT_FALSE(convolution2d > *clone);
        ASSERT_FALSE(convolution2d < *clone);
    }
}

TEST(Convolution2dSingleFeatureInputDefaultPadding, Builds) {
    srand(time(nullptr));

    for (uint32_t test = 0; test < 25; ++test) {
        Network network;

        vector<uint64_t> dimensions;
        int numDimensions = 3;
        for (int i = 0; i < numDimensions; ++i)
            dimensions.push_back(1 + (rand() % 1000));

        Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor featureInput(dataType, dimensions);

        uint32_t numOutputChannels = 1 + (rand() % 1000);

        uint32_t filterHeight = 1 + (rand() % dimensions[1]);
        uint32_t filterWidth = 1 + (rand() % dimensions[2]);
        uint32_t verticalStride = 1;  // due to same padding
        uint32_t horizontalStride = 1;

        bool hasBias = rand() % 2;

        float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
        double exponentialRunningAverageFactor = (1 + (rand() % 1000)) / 1000.0f;
        double epsilon = (1 + (rand() % 1000)) / 1000.0f;

        Convolution2d convolution2d = Convolution2d::Builder()
                                          .network(network)
                                          .featureInput(featureInput)
                                          .numOutputChannels(numOutputChannels)
                                          .filterHeight(filterHeight)
                                          .filterWidth(filterWidth)
                                          .verticalStride(verticalStride)
                                          .horizontalStride(horizontalStride)
                                          .hasBias(hasBias)
                                          .batchNormalization(exponentialRunningAverageFactor, epsilon)
                                          .dropOut(dropProportion)
                                          .build();

        ASSERT_TRUE(convolution2d.isInitialized());

        Optional<Tensor> actualInput = convolution2d.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), dataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        uint32_t verticalPadding = Convolution2d::Builder::computeSamePadding(dimensions[1], verticalStride, filterHeight);
        uint32_t horizontalPadding = Convolution2d::Builder::computeSamePadding(dimensions[2], horizontalStride, filterWidth);
        Optional<Tensor> actualOutput = convolution2d.getFeatureOutput();
        ASSERT_TRUE(actualOutput.isPresent());
        ASSERT_EQ(actualOutput.get().getDataType(), Tensor::DataType::FP16);
        ASSERT_EQ(actualOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(actualOutput.get().getDimensions()[0], numOutputChannels);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = actualOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        ASSERT_EQ(convolution2d.getFilterHeight(), filterHeight);
        ASSERT_EQ(convolution2d.getFilterWidth(), filterWidth);
        ASSERT_EQ(convolution2d.getVerticalStride(), verticalStride);
        ASSERT_EQ(convolution2d.getHorizontalStride(), horizontalStride);
        ASSERT_EQ(convolution2d.getVerticalPadding(), verticalPadding);
        ASSERT_EQ(convolution2d.getHoriztonalPadding(), horizontalPadding);

        shared_ptr<Layer> cloneLayer = convolution2d.clone();
        Convolution2d *clone = dynamic_cast<Convolution2d *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), dataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneOutput = clone->getFeatureOutput();
        ASSERT_TRUE(cloneOutput.isPresent());
        ASSERT_EQ(cloneOutput.get().getDataType(), Tensor::DataType::FP16);
        ASSERT_EQ(cloneOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(cloneOutput.get().getDimensions()[0], numOutputChannels);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = cloneOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        ASSERT_EQ(clone->getFilterHeight(), filterHeight);
        ASSERT_EQ(clone->getFilterWidth(), filterWidth);
        ASSERT_EQ(clone->getVerticalStride(), verticalStride);
        ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
        ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
        ASSERT_EQ(clone->getHoriztonalPadding(), horizontalPadding);

        ASSERT_EQ(convolution2d.getId(), clone->getId());
        ASSERT_GT(convolution2d.getId(), 1u);

        ASSERT_TRUE(convolution2d == *clone);
        ASSERT_FALSE(convolution2d != *clone);
        ASSERT_FALSE(convolution2d > *clone);
        ASSERT_FALSE(convolution2d < *clone);
    }
}

TEST(Convolution2dSingleFeatureInputSamePaddingV2, Builds) {
    srand(time(nullptr));

    for (uint32_t test = 0; test < 25; ++test) {
        Network network;

        vector<uint64_t> dimensions;
        int numDimensions = 3;
        for (int i = 0; i < numDimensions; ++i)
            dimensions.push_back(1 + (rand() % 1000));

        Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor featureInput(dataType, dimensions);

        uint32_t numOutputChannels = 1 + (rand() % 1000);

        uint32_t filterHeight = 1 + (rand() % dimensions[1]);
        uint32_t filterWidth = 1 + (rand() % dimensions[2]);
        uint32_t verticalStride = 1;  // due to same padding
        uint32_t horizontalStride = 1;

        bool hasBias = rand() % 2;

        float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
        double exponentialRunningAverageFactor = (1 + (rand() % 1000)) / 1000.0f;
        double epsilon = (1 + (rand() % 1000)) / 1000.0f;

        Convolution2d convolution2d = Convolution2d::Builder()
                                          .network(network)
                                          .featureInput(featureInput)
                                          .numOutputChannels(numOutputChannels)
                                          .filterHeight(filterHeight)
                                          .filterWidth(filterWidth)
                                          .verticalStride(verticalStride)
                                          .horizontalStride(horizontalStride)
                                          .verticalSamePadding()
                                          .horizontalSamePadding()
                                          .hasBias(hasBias)
                                          .batchNormalization(exponentialRunningAverageFactor, epsilon)
                                          .dropOut(dropProportion)
                                          .build();

        ASSERT_TRUE(convolution2d.isInitialized());

        Optional<Tensor> actualInput = convolution2d.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), dataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        uint32_t verticalPadding = Convolution2d::Builder::computeSamePadding(dimensions[1], verticalStride, filterHeight);
        uint32_t horizontalPadding = Convolution2d::Builder::computeSamePadding(dimensions[2], horizontalStride, filterWidth);
        Optional<Tensor> actualOutput = convolution2d.getFeatureOutput();
        ASSERT_TRUE(actualOutput.isPresent());
        ASSERT_EQ(actualOutput.get().getDataType(), Tensor::DataType::FP16);
        ASSERT_EQ(actualOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(actualOutput.get().getDimensions()[0], numOutputChannels);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = actualOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        ASSERT_EQ(convolution2d.getFilterHeight(), filterHeight);
        ASSERT_EQ(convolution2d.getFilterWidth(), filterWidth);
        ASSERT_EQ(convolution2d.getVerticalStride(), verticalStride);
        ASSERT_EQ(convolution2d.getHorizontalStride(), horizontalStride);
        ASSERT_EQ(convolution2d.getVerticalPadding(), verticalPadding);
        ASSERT_EQ(convolution2d.getHoriztonalPadding(), horizontalPadding);

        shared_ptr<Layer> cloneLayer = convolution2d.clone();
        Convolution2d *clone = dynamic_cast<Convolution2d *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), dataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneOutput = clone->getFeatureOutput();
        ASSERT_TRUE(cloneOutput.isPresent());
        ASSERT_EQ(cloneOutput.get().getDataType(), Tensor::DataType::FP16);
        ASSERT_EQ(cloneOutput.get().getDimensions().size(), dimensions.size());
        ASSERT_EQ(cloneOutput.get().getDimensions()[0], numOutputChannels);
        for (uint32_t d = 1; d < dimensions.size(); ++d) {
            uint32_t diff = cloneOutput.get().getDimensions()[d] - dimensions[d];
            ASSERT_GE(diff, 0u);
            ASSERT_LE(diff, 1u);
        }

        ASSERT_EQ(clone->getFilterHeight(), filterHeight);
        ASSERT_EQ(clone->getFilterWidth(), filterWidth);
        ASSERT_EQ(clone->getVerticalStride(), verticalStride);
        ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
        ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
        ASSERT_EQ(clone->getHoriztonalPadding(), horizontalPadding);

        ASSERT_EQ(convolution2d.getId(), clone->getId());
        ASSERT_GT(convolution2d.getId(), 1u);

        ASSERT_TRUE(convolution2d == *clone);
        ASSERT_FALSE(convolution2d != *clone);
        ASSERT_FALSE(convolution2d > *clone);
        ASSERT_FALSE(convolution2d < *clone);
    }
}

TEST(Convolution2dMultipleFeatureInputs, Builds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 3;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput0(dataType, dimensions);
    Tensor featureInput1(dataType, dimensions);

    uint32_t numOutputChannels = 1 + (rand() % 1000);

    uint32_t filterHeight = 1 + (rand() % dimensions[1]);
    uint32_t filterWidth = 1 + (rand() % dimensions[2]);
    uint32_t verticalStride = 1 + (rand() % 10);
    uint32_t horizontalStride = 1;  // due to same padding

    uint32_t verticalPadding = rand() % filterHeight;

    bool hasBias = rand() % 2;

    UniformRandom::Builder uniformRandomInitializerBuilder;
    Tanh::Builder tanhBuilder;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
    double exponentialRunningAverageFactor = (1 + (rand() % 1000)) / 1000.0f;
    double epsilon = (1 + (rand() % 1000)) / 1000.0f;

    Convolution2d convolution2d = Convolution2d::Builder()
                                      .network(network)
                                      .featureInput(featureInput0)
                                      .featureInput(featureInput1)
                                      .numOutputChannels(numOutputChannels)
                                      .filterHeight(filterHeight)
                                      .filterWidth(filterWidth)
                                      .verticalStride(verticalStride)
                                      .horizontalStride(horizontalStride)
                                      .verticalPadding(verticalPadding)
                                      .horizontalSamePadding()
                                      .hasBias(hasBias)
                                      .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                                      .biasInitializerBuilder(uniformRandomInitializerBuilder)
                                      .activationBuilder(tanhBuilder)
                                      .batchNormalization(exponentialRunningAverageFactor, epsilon)
                                      .dropOut(dropProportion)
                                      .build();

    ASSERT_TRUE(convolution2d.isInitialized());

    uint64_t outputHeight = Convolution2d::Builder::computeOutputDimension(dimensions[1], verticalStride, filterHeight, verticalPadding);
    uint32_t horizontalPadding = Convolution2d::Builder::computeSamePadding(dimensions[2], horizontalStride, filterWidth);
    vector<uint64_t> outputDimensions = {numOutputChannels, outputHeight, dimensions[2]};
    uint32_t diff;

    vector<Tensor> featureInputs = convolution2d.getFeatureInputs();
    vector<Tensor> featureOutputs = convolution2d.getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(convolution2d.getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(convolution2d.getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(convolution2d.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(convolution2d.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(featureOutputs[0].getDimensions().size(), dimensions.size());
    ASSERT_EQ(featureOutputs[0].getDimensions()[0], outputDimensions[0]);
    ASSERT_EQ(featureOutputs[0].getDimensions()[1], outputDimensions[1]);
    diff = featureOutputs[0].getDimensions()[2] - dimensions[2];
    ASSERT_GE(diff, 0u);
    ASSERT_LE(diff, 1u);

    ASSERT_EQ(featureOutputs[1].getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(featureOutputs[1].getDimensions().size(), dimensions.size());
    ASSERT_EQ(featureOutputs[1].getDimensions()[0], outputDimensions[0]);
    ASSERT_EQ(featureOutputs[1].getDimensions()[1], outputDimensions[1]);
    diff = featureOutputs[1].getDimensions()[2] - dimensions[2];
    ASSERT_GE(diff, 0u);
    ASSERT_LE(diff, 1u);

    ASSERT_EQ(convolution2d.getFilterHeight(), filterHeight);
    ASSERT_EQ(convolution2d.getFilterWidth(), filterWidth);
    ASSERT_EQ(convolution2d.getVerticalStride(), verticalStride);
    ASSERT_EQ(convolution2d.getHorizontalStride(), horizontalStride);
    ASSERT_EQ(convolution2d.getVerticalPadding(), verticalPadding);
    ASSERT_EQ(convolution2d.getHoriztonalPadding(), horizontalPadding);

    shared_ptr<Layer> cloneLayer = convolution2d.clone();
    Convolution2d *clone = dynamic_cast<Convolution2d *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    featureInputs.clear();
    featureOutputs.clear();
    featureInputs = clone->getFeatureInputs();
    featureOutputs = clone->getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(clone->getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(clone->getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(clone->getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(clone->getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(featureOutputs[0].getDimensions().size(), dimensions.size());
    ASSERT_EQ(featureOutputs[0].getDimensions()[0], outputDimensions[0]);
    ASSERT_EQ(featureOutputs[0].getDimensions()[1], outputDimensions[1]);
    diff = featureOutputs[0].getDimensions()[2] - dimensions[2];
    ASSERT_GE(diff, 0u);
    ASSERT_LE(diff, 1u);

    ASSERT_EQ(featureOutputs[1].getDataType(), Tensor::DataType::FP16);
    ASSERT_EQ(featureOutputs[1].getDimensions().size(), dimensions.size());
    ASSERT_EQ(featureOutputs[1].getDimensions()[0], outputDimensions[0]);
    ASSERT_EQ(featureOutputs[1].getDimensions()[1], outputDimensions[1]);
    diff = featureOutputs[1].getDimensions()[2] - dimensions[2];
    ASSERT_GE(diff, 0u);
    ASSERT_LE(diff, 1u);

    ASSERT_EQ(clone->getFilterHeight(), filterHeight);
    ASSERT_EQ(clone->getFilterWidth(), filterWidth);
    ASSERT_EQ(clone->getVerticalStride(), verticalStride);
    ASSERT_EQ(clone->getHorizontalStride(), horizontalStride);
    ASSERT_EQ(clone->getVerticalPadding(), verticalPadding);
    ASSERT_EQ(clone->getHoriztonalPadding(), horizontalPadding);

    ASSERT_EQ(convolution2d.getId(), clone->getId());
    ASSERT_GT(convolution2d.getId(), 1u);

    ASSERT_TRUE(convolution2d == *clone);
    ASSERT_FALSE(convolution2d != *clone);
    ASSERT_FALSE(convolution2d > *clone);
    ASSERT_FALSE(convolution2d < *clone);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
