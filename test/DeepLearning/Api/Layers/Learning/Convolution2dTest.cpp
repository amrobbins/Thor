#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2d.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

using ApiDataType = Api::Tensor::DataType;
using ImplDataType = Impl::TensorDescriptor::DataType;

Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);

std::vector<uint64_t> expectedConvOutputDims(uint32_t numOutputChannels,
                                             uint32_t inputHeight,
                                             uint32_t inputWidth,
                                             uint32_t filterHeight,
                                             uint32_t filterWidth,
                                             uint32_t verticalStride,
                                             uint32_t horizontalStride,
                                             uint32_t verticalPadding,
                                             uint32_t horizontalPadding) {
    const uint32_t outputHeight =
        Api::Convolution2d::Builder::computeOutputDimension(inputHeight, verticalStride, filterHeight, verticalPadding);
    const uint32_t outputWidth =
        Api::Convolution2d::Builder::computeOutputDimension(inputWidth, horizontalStride, filterWidth, horizontalPadding);

    return {numOutputChannels, outputHeight, outputWidth};
}

void expectSingleInputOutput(const Api::Convolution2d& convolution,
                             const Api::Tensor& featureInput,
                             const std::vector<uint64_t>& expectedOutputDims) {
    Optional<Api::Tensor> actualInput = convolution.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    EXPECT_EQ(actualInput.get(), featureInput);
    EXPECT_EQ(actualInput.get().getDataType(), featureInput.getDataType());
    EXPECT_EQ(actualInput.get().getDimensions(), featureInput.getDimensions());

    Optional<Api::Tensor> actualOutput = convolution.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    EXPECT_EQ(actualOutput.get().getDataType(), ApiDataType::FP16);
    EXPECT_EQ(actualOutput.get().getDimensions(), expectedOutputDims);

    EXPECT_EQ(convolution.getFeatureOutput(featureInput), actualOutput.get());
    EXPECT_EQ(convolution.getFeatureInput(actualOutput.get()), featureInput);
}

std::shared_ptr<Api::Convolution2d> findOnlyApiConvolution(Api::Network& network) {
    std::shared_ptr<Api::Convolution2d> found;
    for (uint32_t i = 0; i < network.getNumTrainableLayers(); ++i) {
        std::shared_ptr<Api::Convolution2d> candidate = std::dynamic_pointer_cast<Api::Convolution2d>(network.getTrainableLayer(i));
        if (candidate != nullptr) {
            EXPECT_EQ(found, nullptr);
            found = candidate;
        }
    }
    return found;
}

std::shared_ptr<Impl::Convolution2d> findOnlyPhysicalConvolution(Impl::StampedNetwork& stampedNetwork) {
    std::shared_ptr<Impl::Convolution2d> found;
    for (uint64_t i = 0; i < stampedNetwork.getNumTrainableLayers(); ++i) {
        std::shared_ptr<Impl::Convolution2d> candidate =
            std::dynamic_pointer_cast<Impl::Convolution2d>(stampedNetwork.getTrainableLayer(i));
        if (candidate != nullptr) {
            EXPECT_EQ(found, nullptr);
            found = candidate;
        }
    }
    return found;
}

void synchronizeEvents(std::vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

}  // namespace

TEST(Convolution2dApi, SingleFeatureInputNoPaddingBuildsAndClones) {
    Api::Network network("testNetwork");

    const std::vector<uint64_t> inputDims{3, 8, 9};
    Api::Tensor featureInput(ApiDataType::FP16, inputDims);

    constexpr uint32_t numOutputChannels = 7;
    constexpr uint32_t filterHeight = 3;
    constexpr uint32_t filterWidth = 5;
    constexpr uint32_t verticalStride = 2;
    constexpr uint32_t horizontalStride = 1;
    constexpr bool hasBias = true;

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(featureInput)
                                         .numOutputChannels(numOutputChannels)
                                         .filterHeight(filterHeight)
                                         .filterWidth(filterWidth)
                                         .verticalStride(verticalStride)
                                         .horizontalStride(horizontalStride)
                                         .noPadding()
                                         .hasBias(hasBias)
                                         .noActivation()
                                         .build();

    ASSERT_TRUE(convolution.isInitialized());

    const std::vector<uint64_t> outputDims = expectedConvOutputDims(
        numOutputChannels, inputDims[1], inputDims[2], filterHeight, filterWidth, verticalStride, horizontalStride, 0, 0);
    expectSingleInputOutput(convolution, featureInput, outputDims);

    EXPECT_EQ(convolution.getFilterHeight(), filterHeight);
    EXPECT_EQ(convolution.getFilterWidth(), filterWidth);
    EXPECT_EQ(convolution.getVerticalStride(), verticalStride);
    EXPECT_EQ(convolution.getHorizontalStride(), horizontalStride);
    EXPECT_EQ(convolution.getVerticalPadding(), 0u);
    EXPECT_EQ(convolution.getHoriztonalPadding(), 0u);

    std::shared_ptr<Api::Layer> cloneLayer = convolution.clone();
    std::shared_ptr<Api::Convolution2d> clone = std::dynamic_pointer_cast<Api::Convolution2d>(cloneLayer);
    ASSERT_NE(clone, nullptr);

    ASSERT_TRUE(clone->isInitialized());
    expectSingleInputOutput(*clone, featureInput, outputDims);

    EXPECT_EQ(clone->getFilterHeight(), filterHeight);
    EXPECT_EQ(clone->getFilterWidth(), filterWidth);
    EXPECT_EQ(clone->getVerticalStride(), verticalStride);
    EXPECT_EQ(clone->getHorizontalStride(), horizontalStride);
    EXPECT_EQ(clone->getVerticalPadding(), 0u);
    EXPECT_EQ(clone->getHoriztonalPadding(), 0u);

    EXPECT_EQ(convolution.getId(), clone->getId());
    EXPECT_GT(convolution.getId(), 1u);
    EXPECT_TRUE(convolution == *clone);
    EXPECT_FALSE(convolution != *clone);
    EXPECT_FALSE(convolution > *clone);
    EXPECT_FALSE(convolution < *clone);
}

TEST(Convolution2dApi, SingleFeatureInputSpecifiedPaddingBuilds) {
    Api::Network network("testNetwork");

    const std::vector<uint64_t> inputDims{4, 10, 11};
    Api::Tensor featureInput(ApiDataType::FP16, inputDims);

    constexpr uint32_t numOutputChannels = 5;
    constexpr uint32_t filterHeight = 4;
    constexpr uint32_t filterWidth = 3;
    constexpr uint32_t verticalStride = 2;
    constexpr uint32_t horizontalStride = 3;
    constexpr uint32_t verticalPadding = 1;
    constexpr uint32_t horizontalPadding = 2;

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(featureInput)
                                         .numOutputChannels(numOutputChannels)
                                         .filterHeight(filterHeight)
                                         .filterWidth(filterWidth)
                                         .verticalStride(verticalStride)
                                         .horizontalStride(horizontalStride)
                                         .verticalPadding(verticalPadding)
                                         .horizontalPadding(horizontalPadding)
                                         .hasBias(false)
                                         .noActivation()
                                         .build();

    ASSERT_TRUE(convolution.isInitialized());

    const std::vector<uint64_t> outputDims = expectedConvOutputDims(numOutputChannels,
                                                                    inputDims[1],
                                                                    inputDims[2],
                                                                    filterHeight,
                                                                    filterWidth,
                                                                    verticalStride,
                                                                    horizontalStride,
                                                                    verticalPadding,
                                                                    horizontalPadding);
    expectSingleInputOutput(convolution, featureInput, outputDims);

    EXPECT_EQ(convolution.getFilterHeight(), filterHeight);
    EXPECT_EQ(convolution.getFilterWidth(), filterWidth);
    EXPECT_EQ(convolution.getVerticalStride(), verticalStride);
    EXPECT_EQ(convolution.getHorizontalStride(), horizontalStride);
    EXPECT_EQ(convolution.getVerticalPadding(), verticalPadding);
    EXPECT_EQ(convolution.getHoriztonalPadding(), horizontalPadding);
}

TEST(Convolution2dApi, SamePaddingBuildsWithExpectedPaddingAndOutputShape) {
    Api::Network network("testNetwork");

    const std::vector<uint64_t> inputDims{3, 8, 10};
    Api::Tensor featureInput(ApiDataType::FP16, inputDims);

    constexpr uint32_t numOutputChannels = 6;
    constexpr uint32_t filterHeight = 3;
    constexpr uint32_t filterWidth = 5;
    constexpr uint32_t verticalStride = 1;
    constexpr uint32_t horizontalStride = 1;

    const uint32_t verticalPadding = Api::Convolution2d::Builder::computeSamePadding(inputDims[1], verticalStride, filterHeight);
    const uint32_t horizontalPadding = Api::Convolution2d::Builder::computeSamePadding(inputDims[2], horizontalStride, filterWidth);

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(featureInput)
                                         .numOutputChannels(numOutputChannels)
                                         .filterHeight(filterHeight)
                                         .filterWidth(filterWidth)
                                         .verticalStride(verticalStride)
                                         .horizontalStride(horizontalStride)
                                         .samePadding()
                                         .hasBias(true)
                                         .noActivation()
                                         .build();

    ASSERT_TRUE(convolution.isInitialized());

    EXPECT_EQ(verticalPadding, 1u);
    EXPECT_EQ(horizontalPadding, 2u);
    EXPECT_EQ(convolution.getVerticalPadding(), verticalPadding);
    EXPECT_EQ(convolution.getHoriztonalPadding(), horizontalPadding);

    expectSingleInputOutput(convolution, featureInput, {numOutputChannels, inputDims[1], inputDims[2]});
}

TEST(Convolution2dApi, MultipleFeatureInputsBuildAndMapToDistinctOutputs) {
    Api::Network network("testNetwork");

    const std::vector<uint64_t> inputDims{2, 9, 7};
    Api::Tensor featureInput0(ApiDataType::FP16, inputDims);
    Api::Tensor featureInput1(ApiDataType::FP16, inputDims);

    constexpr uint32_t numOutputChannels = 4;
    constexpr uint32_t filterHeight = 3;
    constexpr uint32_t filterWidth = 3;
    constexpr uint32_t verticalStride = 1;
    constexpr uint32_t horizontalStride = 2;
    constexpr uint32_t verticalPadding = 1;
    constexpr uint32_t horizontalPadding = 0;

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(featureInput0)
                                         .featureInput(featureInput1)
                                         .numOutputChannels(numOutputChannels)
                                         .filterHeight(filterHeight)
                                         .filterWidth(filterWidth)
                                         .verticalStride(verticalStride)
                                         .horizontalStride(horizontalStride)
                                         .verticalPadding(verticalPadding)
                                         .horizontalPadding(horizontalPadding)
                                         .hasBias(true)
                                         .noActivation()
                                         .build();

    ASSERT_TRUE(convolution.isInitialized());

    std::vector<Api::Tensor> featureInputs = convolution.getFeatureInputs();
    std::vector<Api::Tensor> featureOutputs = convolution.getFeatureOutputs();

    ASSERT_EQ(featureInputs.size(), 2u);
    ASSERT_EQ(featureOutputs.size(), 2u);

    EXPECT_EQ(featureInputs[0], featureInput0);
    EXPECT_EQ(featureInputs[1], featureInput1);

    EXPECT_EQ(convolution.getFeatureOutput(featureInput0), featureOutputs[0]);
    EXPECT_EQ(convolution.getFeatureOutput(featureInput1), featureOutputs[1]);
    EXPECT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    EXPECT_EQ(convolution.getFeatureInput(featureOutputs[0]), featureInput0);
    EXPECT_EQ(convolution.getFeatureInput(featureOutputs[1]), featureInput1);

    const std::vector<uint64_t> outputDims = expectedConvOutputDims(numOutputChannels,
                                                                    inputDims[1],
                                                                    inputDims[2],
                                                                    filterHeight,
                                                                    filterWidth,
                                                                    verticalStride,
                                                                    horizontalStride,
                                                                    verticalPadding,
                                                                    horizontalPadding);

    for (const Api::Tensor& output : featureOutputs) {
        EXPECT_EQ(output.getDataType(), ApiDataType::FP16);
        EXPECT_EQ(output.getDimensions(), outputDims);
    }

    std::shared_ptr<Api::Convolution2d> clone = std::dynamic_pointer_cast<Api::Convolution2d>(convolution.clone());
    ASSERT_NE(clone, nullptr);

    std::vector<Api::Tensor> cloneFeatureOutputs = clone->getFeatureOutputs();
    ASSERT_EQ(cloneFeatureOutputs.size(), 2u);
    EXPECT_EQ(clone->getFeatureOutput(featureInput0), cloneFeatureOutputs[0]);
    EXPECT_EQ(clone->getFeatureOutput(featureInput1), cloneFeatureOutputs[1]);
    EXPECT_EQ(cloneFeatureOutputs[0].getDimensions(), outputDims);
    EXPECT_EQ(cloneFeatureOutputs[1].getDimensions(), outputDims);
}

TEST(Convolution2dApi, CompoundLayerUsesOriginalInputAndFinalOutput) {
    Api::Network network("testNetwork");

    const std::vector<uint64_t> inputDims{3, 8, 8};
    Api::Tensor fp32FeatureInput(ApiDataType::FP32, inputDims);

    constexpr uint32_t numOutputChannels = 5;

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(fp32FeatureInput)
                                         .numOutputChannels(numOutputChannels)
                                         .filterHeight(3)
                                         .filterWidth(3)
                                         .verticalStride(1)
                                         .horizontalStride(1)
                                         .samePadding()
                                         .hasBias(true)
                                         .build();

    ASSERT_TRUE(convolution.isInitialized());

    Optional<Api::Tensor> publicInput = convolution.getFeatureInput();
    ASSERT_TRUE(publicInput.isPresent());
    EXPECT_EQ(publicInput.get(), fp32FeatureInput);
    EXPECT_EQ(publicInput.get().getDataType(), ApiDataType::FP32);
    EXPECT_EQ(publicInput.get().getDimensions(), inputDims);

    Optional<Api::Tensor> publicOutput = convolution.getFeatureOutput();
    ASSERT_TRUE(publicOutput.isPresent());
    EXPECT_EQ(publicOutput.get().getDataType(), ApiDataType::FP16);
    EXPECT_EQ(publicOutput.get().getDimensions(), (std::vector<uint64_t>{numOutputChannels, inputDims[1], inputDims[2]}));

    EXPECT_EQ(convolution.getFeatureOutput(fp32FeatureInput), publicOutput.get());
    EXPECT_EQ(convolution.getFeatureInput(publicOutput.get()), fp32FeatureInput);

    // The public compound layer should have added support layers plus the standalone convolution.
    EXPECT_GT(network.getNumLayers(), 1u);
    EXPECT_EQ(network.getNumTrainableLayers(), 1u);
}

TEST(Convolution2dApi, ArchitectureJsonContainsExpectedFieldsAndLayerOptimizer) {
    Api::Network network("testNetwork");

    const std::vector<uint64_t> inputDims{3, 8, 9};
    Api::Tensor featureInput(ApiDataType::FP16, inputDims);

    std::shared_ptr<Api::Sgd> sgd =
        Api::Sgd::Builder().initialLearningRate(0.125f).decay(0.25f).momentum(0.5f).useNesterovMomentum(true).build();

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(featureInput)
                                         .numOutputChannels(7)
                                         .filterHeight(3)
                                         .filterWidth(5)
                                         .verticalStride(2)
                                         .horizontalStride(1)
                                         .verticalPadding(1)
                                         .horizontalPadding(2)
                                         .hasBias(true)
                                         .optimizer(sgd)
                                         .noActivation()
                                         .build();

    json j = convolution.architectureJson();

    ASSERT_EQ(j.at("factory").get<std::string>(), Api::Layer::Factory::Learning.value());
    ASSERT_EQ(j.at("version").get<std::string>(), "1.0.0");
    ASSERT_EQ(j.at("layer_type").get<std::string>(), "convolution_2d");
    ASSERT_EQ(j.at("data_layout").get<std::string>(), "NCHW");

    EXPECT_EQ(j.at("filter_height").get<uint32_t>(), 3u);
    EXPECT_EQ(j.at("filter_width").get<uint32_t>(), 5u);
    EXPECT_EQ(j.at("vertical_stride").get<uint32_t>(), 2u);
    EXPECT_EQ(j.at("horizontal_stride").get<uint32_t>(), 1u);
    EXPECT_EQ(j.at("vertical_padding").get<uint32_t>(), 1u);
    EXPECT_EQ(j.at("horizontal_padding").get<uint32_t>(), 2u);
    EXPECT_EQ(j.at("num_output_channels").get<uint32_t>(), 7u);
    EXPECT_TRUE(j.at("has_bias").get<bool>());

    ASSERT_TRUE(j.contains("inputs"));
    ASSERT_TRUE(j.at("inputs").is_array());
    ASSERT_EQ(j.at("inputs").size(), 1u);
    EXPECT_EQ(j.at("inputs")[0].at("data_type").get<ApiDataType>(), ApiDataType::FP16);
    EXPECT_EQ(j.at("inputs")[0].at("dimensions").get<std::vector<uint64_t>>(), inputDims);

    ASSERT_TRUE(j.contains("outputs"));
    ASSERT_TRUE(j.at("outputs").is_array());
    ASSERT_EQ(j.at("outputs").size(), 1u);
    EXPECT_EQ(j.at("outputs")[0].at("data_type").get<ApiDataType>(), ApiDataType::FP16);
    EXPECT_EQ(j.at("outputs")[0].at("dimensions").get<std::vector<uint64_t>>(), (std::vector<uint64_t>{7, 4, 9}));

    ASSERT_TRUE(j.contains("weights_initializer"));
    ASSERT_TRUE(j.contains("biases_initializer"));

    ASSERT_TRUE(j.contains("optimizer"));
    const json& optimizer = j.at("optimizer");
    EXPECT_EQ(optimizer.at("optimizer_type").get<std::string>(), "sgd");
    EXPECT_FLOAT_EQ(optimizer.at("initial_learning_rate").get<float>(), 0.125f);
    EXPECT_FLOAT_EQ(optimizer.at("decay").get<float>(), 0.25f);
    EXPECT_FLOAT_EQ(optimizer.at("momentum").get<float>(), 0.5f);
    EXPECT_TRUE(optimizer.at("use_nesterov").get<bool>());
}

TEST(Convolution2dApi, ArchitectureJsonDeserializeRebuildsApiLayerAndOptimizer) {
    Api::Network initialNetwork("initialNetwork");

    Api::NetworkInput networkInput =
        Api::NetworkInput::Builder().network(initialNetwork).name("input").dimensions({3, 8, 9}).dataType(ApiDataType::FP16).build();

    std::shared_ptr<Api::Sgd> sgd =
        Api::Sgd::Builder().initialLearningRate(0.05f).decay(0.1f).momentum(0.2f).useNesterovMomentum(false).build();

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(initialNetwork)
                                         .featureInput(networkInput.getFeatureOutput())
                                         .numOutputChannels(4)
                                         .filterHeight(3)
                                         .filterWidth(3)
                                         .verticalStride(1)
                                         .horizontalStride(1)
                                         .samePadding()
                                         .hasBias(true)
                                         .optimizer(sgd)
                                         .noActivation()
                                         .build();

    json networkInputJ = networkInput.architectureJson();
    json convolutionJ = convolution.architectureJson();

    Api::Network newNetwork("newNetwork");
    std::shared_ptr<thor_file::TarReader> archiveReader;

    Api::Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
    Api::Layer::deserialize(archiveReader, convolutionJ, &newNetwork);

    ASSERT_EQ(newNetwork.getNumTrainableLayers(), 1u);

    std::shared_ptr<Api::Convolution2d> deserialized = std::dynamic_pointer_cast<Api::Convolution2d>(newNetwork.getTrainableLayer(0));
    ASSERT_NE(deserialized, nullptr);

    EXPECT_TRUE(deserialized->isInitialized());
    EXPECT_EQ(deserialized->getFilterHeight(), 3u);
    EXPECT_EQ(deserialized->getFilterWidth(), 3u);
    EXPECT_EQ(deserialized->getVerticalStride(), 1u);
    EXPECT_EQ(deserialized->getHorizontalStride(), 1u);
    EXPECT_EQ(deserialized->getVerticalPadding(), 1u);
    EXPECT_EQ(deserialized->getHoriztonalPadding(), 1u);

    Optional<Api::Tensor> output = deserialized->getFeatureOutput();
    ASSERT_TRUE(output.isPresent());
    EXPECT_EQ(output.get().getDataType(), ApiDataType::FP16);
    EXPECT_EQ(output.get().getDimensions(), (std::vector<uint64_t>{4, 8, 9}));

    ASSERT_TRUE(deserialized->hasOptimizer());
    std::shared_ptr<Api::Sgd> deserializedSgd = std::dynamic_pointer_cast<Api::Sgd>(deserialized->getOptimizer());
    ASSERT_NE(deserializedSgd, nullptr);
    EXPECT_FLOAT_EQ(deserializedSgd->getInitialLearningRate(), 0.05f);
    EXPECT_FLOAT_EQ(deserializedSgd->getDecay(), 0.1f);
    EXPECT_FLOAT_EQ(deserializedSgd->getMomentum(), 0.2f);
    EXPECT_FALSE(deserializedSgd->getUseNesterovMomentum());
}

TEST(Convolution2dApi, PlaceCompilesPhysicalConvolutionParametersAndOptimizers) {
    Api::Network network("placeNetwork");

    Api::NetworkInput networkInput =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3, 8, 8}).dataType(ApiDataType::FP16).build();

    std::shared_ptr<Api::Sgd> sgd =
        Api::Sgd::Builder().initialLearningRate(0.1f).decay(0.0f).momentum(0.0f).useNesterovMomentum(false).build();

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(networkInput.getFeatureOutput())
                                         .numOutputChannels(5)
                                         .filterHeight(3)
                                         .filterWidth(3)
                                         .verticalStride(1)
                                         .horizontalStride(1)
                                         .samePadding()
                                         .hasBias(true)
                                         .optimizer(sgd)
                                         .noActivation()
                                         .build();

    Api::NetworkOutput networkOutput = Api::NetworkOutput::Builder()
                                           .network(network)
                                           .name("output")
                                           .inputTensor(convolution.getFeatureOutput())
                                           .dataType(ApiDataType::FP16)
                                           .build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/false, {0}, 1);
    ASSERT_NE(placedNetwork, nullptr);

    synchronizeEvents(initDoneEvents);

    ASSERT_EQ(placedNetwork->getNumStamps(), 1u);
    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);

    std::shared_ptr<Impl::Convolution2d> physicalConvolution = findOnlyPhysicalConvolution(stampedNetwork);
    ASSERT_NE(physicalConvolution, nullptr);

    Impl::Tensor weights = physicalConvolution->getWeights();
    Optional<Impl::Tensor> biases = physicalConvolution->getBiases();

    EXPECT_EQ(weights.getPlacement(), gpuPlacement);
    EXPECT_EQ(weights.getDataType(), ImplDataType::FP16);
    EXPECT_EQ(weights.getDimensions(), (std::vector<uint64_t>{5, 3, 3, 3}));

    ASSERT_TRUE(biases.isPresent());
    EXPECT_EQ(biases.get().getPlacement(), gpuPlacement);
    EXPECT_EQ(biases.get().getDataType(), ImplDataType::FP16);
    EXPECT_EQ(biases.get().getDimensions(), (std::vector<uint64_t>{5}));

    ASSERT_TRUE(physicalConvolution->getParameter("weights")->hasOptimizer());
    ASSERT_TRUE(physicalConvolution->getParameter("biases")->hasOptimizer());

    std::shared_ptr<Impl::Sgd> weightsOptimizer =
        std::dynamic_pointer_cast<Impl::Sgd>(physicalConvolution->getParameter("weights")->getOptimizer());
    std::shared_ptr<Impl::Sgd> biasesOptimizer =
        std::dynamic_pointer_cast<Impl::Sgd>(physicalConvolution->getParameter("biases")->getOptimizer());

    ASSERT_NE(weightsOptimizer, nullptr);
    ASSERT_NE(biasesOptimizer, nullptr);

    EXPECT_EQ(weightsOptimizer->getId(), sgd->getId());
    EXPECT_EQ(biasesOptimizer->getId(), sgd->getId());
    EXPECT_TRUE(weightsOptimizer->isCompiled());
    EXPECT_TRUE(biasesOptimizer->isCompiled());

    Optional<Impl::Tensor> weightsGradient = weightsOptimizer->getWeightsGradient();
    Optional<Impl::Tensor> biasesGradient = biasesOptimizer->getWeightsGradient();

    ASSERT_TRUE(weightsGradient.isPresent());
    ASSERT_TRUE(biasesGradient.isPresent());

    EXPECT_EQ(weightsGradient.get().getDimensions(), weights.getDimensions());
    EXPECT_EQ(biasesGradient.get().getDimensions(), biases.get().getDimensions());
}

TEST(Convolution2dApi, SerializePlacedLayerIncludesStateFilesAndPerParameterOptimizers) {
    Api::Network network("serializeNetwork");

    Api::NetworkInput networkInput =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3, 8, 8}).dataType(ApiDataType::FP16).build();

    std::shared_ptr<Api::Sgd> sgd =
        Api::Sgd::Builder().initialLearningRate(0.2f).decay(0.1f).momentum(0.0f).useNesterovMomentum(false).build();

    Api::Convolution2d convolution = Api::Convolution2d::Builder()
                                         .network(network)
                                         .featureInput(networkInput.getFeatureOutput())
                                         .numOutputChannels(5)
                                         .filterHeight(3)
                                         .filterWidth(3)
                                         .verticalStride(1)
                                         .horizontalStride(1)
                                         .samePadding()
                                         .hasBias(true)
                                         .optimizer(sgd)
                                         .noActivation()
                                         .build();

    Api::NetworkOutput networkOutput = Api::NetworkOutput::Builder()
                                           .network(network)
                                           .name("output")
                                           .inputTensor(convolution.getFeatureOutput())
                                           .dataType(ApiDataType::FP16)
                                           .build();

    std::vector<Event> initDoneEvents;
    std::shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(/*batchSize=*/2, initDoneEvents, /*inferenceOnly=*/false, {0}, 1);
    ASSERT_NE(placedNetwork, nullptr);

    synchronizeEvents(initDoneEvents);

    ASSERT_EQ(placedNetwork->getNumStamps(), 1u);
    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);

    thor_file::TarWriter archiveWriter("convolution2d_api_test");
    Stream stream(gpuPlacement);

    json j = convolution.serialize(archiveWriter, stream, /*saveOptimizerState=*/true, stampedNetwork);

    ASSERT_EQ(j.at("factory").get<std::string>(), Api::Layer::Factory::Learning.value());
    ASSERT_EQ(j.at("version").get<std::string>(), "1.0.0");
    ASSERT_EQ(j.at("layer_type").get<std::string>(), "convolution_2d");
    ASSERT_EQ(j.at("data_layout").get<std::string>(), "NCHW");

    const std::string layerName = "layer" + std::to_string(convolution.getId());

    ASSERT_TRUE(j.contains("weights_tensor"));
    EXPECT_EQ(j.at("weights_tensor").get<std::string>(), layerName + "_weights.gds");

    ASSERT_TRUE(j.contains("biases_tensor"));
    EXPECT_EQ(j.at("biases_tensor").get<std::string>(), layerName + "_biases.gds");

    ASSERT_TRUE(j.contains("optimizer"));
    ASSERT_TRUE(j.contains("weights_optimizer"));
    ASSERT_TRUE(j.contains("biases_optimizer"));

    EXPECT_EQ(j.at("optimizer").at("optimizer_type").get<std::string>(), "sgd");
    EXPECT_EQ(j.at("weights_optimizer").at("optimizer_type").get<std::string>(), "sgd");
    EXPECT_EQ(j.at("biases_optimizer").at("optimizer_type").get<std::string>(), "sgd");

    EXPECT_EQ(j.at("weights_optimizer").at("id").get<uint64_t>(), sgd->getId());
    EXPECT_EQ(j.at("biases_optimizer").at("id").get<uint64_t>(), sgd->getId());
    EXPECT_FLOAT_EQ(j.at("weights_optimizer").at("initial_learning_rate").get<float>(), 0.2f);
    EXPECT_FLOAT_EQ(j.at("biases_optimizer").at("initial_learning_rate").get<float>(), 0.2f);
}
