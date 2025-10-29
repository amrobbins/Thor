#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>

using std::shared_ptr;
using json = nlohmann::json;

using namespace Thor;
using namespace std;

TEST(Convolution2d, SingleFeatureInputNoPaddingBuilds) {
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

TEST(Convolution2d, SingleFeatureInputSpecifiedPaddingBuilds) {
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

TEST(Convolution2d, SingleFeatureInputSamePaddingBuilds) {
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

TEST(Convolution2d, SingleFeatureInputDefaultPaddingBuilds) {
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

TEST(Convolution2d, SingleFeatureInputSamePaddingV2Builds) {
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

TEST(Convolution2d, MultipleFeatureInputsBuilds) {
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

TEST(Convolution2d, SerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;

    Tensor::DataType dataType = Tensor::DataType::FP16;

    vector<uint64_t> inputDimensions = {3UL + (rand() % 16), 100UL + (rand() % 16), 100UL + (rand() % 3)};

    uint32_t numOutputChannels = 1 + (rand() % 10);
    bool hasBias = rand() % 2;

    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;

    bool use_batch_norm = rand() % 2;

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    uint32_t filterWidth = 1 + (rand() % 10);
    uint32_t filterHeight = 1 + (rand() % 10);
    uint32_t horizontalStride = 1 + (rand() % 2);
    uint32_t verticalStride = 1 + (rand() % 2);
    uint32_t horizontalPadding = rand() % filterWidth;
    uint32_t verticalPadding = rand() % filterHeight;

    Convolution2d::Builder convolution2dBuilder = Convolution2d::Builder()
                                                      .network(initialNetwork)
                                                      .featureInput(networkInput.getFeatureOutput())
                                                      .numOutputChannels(numOutputChannels)
                                                      .filterWidth(filterWidth)
                                                      .filterHeight(filterHeight)
                                                      .horizontalStride(horizontalStride)
                                                      .verticalStride(verticalStride)
                                                      .horizontalPadding(horizontalPadding)
                                                      .verticalPadding(verticalPadding)
                                                      .hasBias(hasBias)
                                                      .dropOut(dropProportion);
    if (use_batch_norm) {
        convolution2dBuilder.batchNormalization();
    }
    Convolution2d convolution2d = convolution2dBuilder.build();

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(convolution2d.getFeatureOutputs()[0])
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(convolution2d.isInitialized());

    vector<Tensor> featureInputs = convolution2d.getFeatureInputs();
    vector<Tensor> featureOutputs = convolution2d.getFeatureOutputs();
    assert(featureInputs[0] == networkInput.getFeatureOutput());

    ASSERT_EQ(convolution2d.getFeatureOutput(networkInput.getFeatureOutput()), featureOutputs[0]);

    assert(convolution2d.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    initialNetwork.place(batchSize, initDoneEvents);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the convolution connected layer from the network and write to its weights
    vector<ThorImplementation::StampedNetwork> stampedNetworks = initialNetwork.getStampedNetworks();
    ASSERT_EQ(stampedNetworks.size(), 1UL);
    ThorImplementation::StampedNetwork stampedNetwork = stampedNetworks[0];
    vector<shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer>> trainableLayers = stampedNetwork.getTrainableLayers();
    ASSERT_EQ(trainableLayers.size(), use_batch_norm ? 2UL : 1UL);
    shared_ptr<ThorImplementation::Convolution2d> physicalConvLayer =
        dynamic_pointer_cast<ThorImplementation::Convolution2d>(trainableLayers[0]);
    if (use_batch_norm) {
        if (physicalConvLayer == nullptr)
            physicalConvLayer = dynamic_pointer_cast<ThorImplementation::Convolution2d>(trainableLayers[1]);
    }
    ASSERT_TRUE(physicalConvLayer != nullptr);

    ThorImplementation::Tensor weights = physicalConvLayer->getWeights();
    ThorImplementation::Tensor weightsCpu = weights.clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    half *weightsCpuMem = (half *)weightsCpu.getMemPtr();
    for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
        weightsCpuMem[i] = i;
    }
    weights.copyFromAsync(weightsCpu, stream);

    ThorImplementation::Tensor biases;
    ThorImplementation::Tensor biasesCpu;
    if (hasBias) {
        biases = physicalConvLayer->getBiases();
        biasesCpu = biases.clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
        half *biasesCpuMem = (half *)biasesCpu.getMemPtr();
        for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
            biasesCpuMem[i] = i * i + 6;
        }
        biases.copyFromAsync(biasesCpu, stream);
    }

    json convolution2dJ = convolution2d.serialize("/tmp/", stream);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    ASSERT_EQ(convolution2dJ["version"], "1.0.0");
    ASSERT_EQ(convolution2dJ["layer_type"], "convolution_2d");
    ASSERT_EQ(convolution2dJ["data_layout"], "NCHW");

    EXPECT_TRUE(convolution2dJ.contains("num_output_channels"));
    EXPECT_TRUE(convolution2dJ.contains("has_bias"));
    EXPECT_FALSE(convolution2dJ.contains("activation"));
    EXPECT_FALSE(convolution2dJ.contains("drop_out"));
    EXPECT_FALSE(convolution2dJ.contains("batch_normalization"));
    EXPECT_FALSE(convolution2dJ.contains("activation"));
    EXPECT_EQ(convolution2dJ.contains("biases_tensor"), hasBias);
    EXPECT_TRUE(convolution2dJ.contains("weights_tensor"));
    EXPECT_TRUE(convolution2dJ.contains("inputs"));
    EXPECT_TRUE(convolution2dJ.contains("outputs"));
    EXPECT_TRUE(convolution2dJ.contains("filter_width"));
    EXPECT_TRUE(convolution2dJ.contains("filter_height"));
    EXPECT_TRUE(convolution2dJ.contains("horizontal_stride"));
    EXPECT_TRUE(convolution2dJ.contains("vertical_stride"));
    EXPECT_TRUE(convolution2dJ.contains("horizontal_padding"));
    EXPECT_TRUE(convolution2dJ.contains("vertical_padding"));

    ASSERT_TRUE(convolution2dJ.at("num_output_channels").is_number_integer());
    ASSERT_TRUE(convolution2dJ.at("has_bias").is_boolean());
    ASSERT_TRUE(convolution2dJ.at("weights_tensor").is_string());
    ASSERT_TRUE(convolution2dJ.at("inputs").is_array());
    ASSERT_TRUE(convolution2dJ.at("outputs").is_array());
    ASSERT_TRUE(convolution2dJ.at("filter_width").is_number_integer());
    ASSERT_TRUE(convolution2dJ.at("filter_height").is_number_integer());
    ASSERT_TRUE(convolution2dJ.at("horizontal_stride").is_number_integer());
    ASSERT_TRUE(convolution2dJ.at("vertical_stride").is_number_integer());
    ASSERT_TRUE(convolution2dJ.at("horizontal_padding").is_number_integer());
    ASSERT_TRUE(convolution2dJ.at("vertical_padding").is_number_integer());

    EXPECT_EQ(convolution2dJ.at("num_output_channels").get<uint32_t>(), numOutputChannels);
    EXPECT_EQ(convolution2dJ.at("has_bias").get<bool>(), hasBias);
    EXPECT_EQ(convolution2dJ.at("filter_width").get<uint32_t>(), filterWidth);
    EXPECT_EQ(convolution2dJ.at("filter_height").get<uint32_t>(), filterHeight);
    EXPECT_EQ(convolution2dJ.at("horizontal_stride").get<uint32_t>(), horizontalStride);
    EXPECT_EQ(convolution2dJ.at("vertical_stride").get<uint32_t>(), verticalStride);
    EXPECT_EQ(convolution2dJ.at("horizontal_padding").get<uint32_t>(), horizontalPadding);
    EXPECT_EQ(convolution2dJ.at("vertical_padding").get<uint32_t>(), verticalPadding);

    const auto &inputs = convolution2dJ.at("inputs");
    ASSERT_EQ(inputs.size(), 1U) << "Expect exactly one input";
    const auto &in0 = inputs.at(0);
    ASSERT_TRUE(in0.is_object());
    ASSERT_TRUE(in0.at("data_type").is_string());
    EXPECT_EQ(in0.at("data_type").get<string>(), "fp16");

    ASSERT_TRUE(in0.at("dimensions").is_array());
    ASSERT_EQ(in0.at("dimensions").size(), 3U);
    EXPECT_TRUE(in0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(in0.at("dimensions").at(0).get<uint32_t>(), inputDimensions[0]);

    ASSERT_TRUE(in0.at("id").is_number_integer());

    const auto &outputs = convolution2dJ.at("outputs");
    ASSERT_EQ(outputs.size(), 1U) << "Expect exactly one output";
    const auto &out0 = outputs.at(0);
    ASSERT_TRUE(out0.is_object());
    ASSERT_TRUE(out0.at("data_type").is_string());
    EXPECT_EQ(out0.at("data_type").get<string>(), "fp16");

    vector<uint64_t> physicalOutputDimensions = physicalConvLayer->getFeatureOutputs()[0].get().getDimensions();
    vector<uint64_t> apiOutputDimensions;
    // skip batch dimension
    for (uint32_t i = 1; i < 4; ++i) {
        apiOutputDimensions.push_back(physicalOutputDimensions[i]);
    }

    ASSERT_TRUE(out0.at("dimensions").is_array());
    ASSERT_EQ(out0.at("dimensions").size(), 3U);
    // EXPECT_TRUE(out0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(out0.at("dimensions").get<vector<uint64_t>>(), apiOutputDimensions);

    ASSERT_TRUE(out0.at("id").is_number_integer());

    EXPECT_FALSE(convolution2dJ.at("weights_tensor").get<string>().empty());
    if (hasBias) {
        EXPECT_FALSE(convolution2dJ.at("biases_tensor").get<string>().empty());
    }

    // printf("%s\n", networkInputJ.dump(4).c_str());
    // printf("%s\n", convolution2dJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Convolution2d::deserialize(convolution2dJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    newNetwork.place(batchSize, initDoneEvents);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    stampedNetworks.clear();
    stampedNetworks = newNetwork.getStampedNetworks();
    ASSERT_EQ(stampedNetworks.size(), 1UL);
    stampedNetwork = stampedNetworks[0];
    trainableLayers.clear();
    trainableLayers = stampedNetwork.getTrainableLayers();

    ASSERT_EQ(trainableLayers.size(), 1UL);
    shared_ptr<ThorImplementation::Convolution2d> physicalConvLayerDes =
        dynamic_pointer_cast<ThorImplementation::Convolution2d>(trainableLayers[0]);
    ASSERT_TRUE(physicalConvLayerDes != nullptr);

    ThorImplementation::Tensor weightsDes = physicalConvLayerDes->getWeights();
    ThorImplementation::Tensor weightsCpuDes = weightsDes.clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    weightsCpuDes.copyFromAsync(weightsDes, stream);

    ThorImplementation::Tensor biasesDes;
    ThorImplementation::Tensor biasesCpuDes;
    if (hasBias) {
        biasesDes = physicalConvLayerDes->getBiases();
        biasesCpuDes = biasesDes.clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
        biasesCpuDes.copyFromAsync(biasesDes, stream);
    }

    stream.synchronize();

    ASSERT_NE(weightsDes, weights);
    ASSERT_EQ(weightsDes.getDimensions(), weights.getDimensions());
    ASSERT_EQ(weightsDes.getDataType(), weights.getDataType());
    ASSERT_TRUE(weightsDes.getPlacement() == weights.getPlacement());

    half *weightsCpuMemDes = (half *)weightsCpuDes.getMemPtr();
    for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
        ASSERT_EQ(weightsCpuMemDes[i], half(i));
    }

    if (hasBias) {
        ASSERT_NE(biasesDes, biases);
        ASSERT_EQ(biasesDes.getDimensions(), biases.getDimensions());
        ASSERT_EQ(biasesDes.getDataType(), biases.getDataType());
        ASSERT_TRUE(biasesDes.getPlacement() == biases.getPlacement());

        half *biasesCpuMemDes = (half *)biasesCpuDes.getMemPtr();
        for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
            ASSERT_EQ(biasesCpuMemDes[i], half(i * i + 6));
        }
    }
}
