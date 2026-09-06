#include <optional>
#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "Utilities/Expression/RaggedExpression.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(Activations, SoftmaxBuilds) {
    srand(time(nullptr));

    Network network("testNetwork");

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Softmax::Builder softmaxBuilder;
    softmaxBuilder.network(network);
    softmaxBuilder.featureInput(featureInput);
    shared_ptr<Softmax> softmax = dynamic_pointer_cast<Softmax>(softmaxBuilder.build());

    ASSERT_TRUE(softmax->isInitialized());

    std::optional<Tensor> actualInput = softmax->getFeatureInput();
    ASSERT_TRUE(actualInput.has_value());
    ASSERT_EQ(actualInput.value().getDataType(), dataType);
    ASSERT_EQ(actualInput.value().getDimensions(), dimensions);

    std::optional<Tensor> actualOutput = softmax->getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), dataType);
    ASSERT_EQ(actualOutput.value().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = softmax->clone();
    Softmax *clone = dynamic_cast<Softmax *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    std::optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.has_value());
    ASSERT_EQ(cloneInput.value().getDataType(), dataType);
    ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);

    std::optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.has_value());
    ASSERT_EQ(cloneOutput.value().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.value().getDimensions(), dimensions);

    ASSERT_NE(softmax->getId(), clone->getId());
    ASSERT_GT(softmax->getId(), 1u);
}


TEST(Activations, RaggedSoftmaxBuildCloneAndExpressionPreservePartitionWithOrdinaryFinalAxisSemantics) {
    Network network("raggedSoftmaxBuildCloneAndExpression");
    RaggedTensor featureInput(DataType::FP32, {2, 3}, 2, 8, DataType::UINT64);

    shared_ptr<Softmax> softmax = dynamic_pointer_cast<Softmax>(
        Softmax::Builder().network(network).featureInput(featureInput).build());
    ASSERT_NE(softmax, nullptr);
    ASSERT_TRUE(softmax->isInitialized());
    EXPECT_TRUE(softmax->supportsRaggedStandalone());
    EXPECT_FALSE(softmax->supportsRaggedLearningLayerFusion());
    EXPECT_TRUE(softmax->getUseRagged());
    ASSERT_TRUE(softmax->getRaggedFeatureInput().has_value());
    ASSERT_TRUE(softmax->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(softmax->getRaggedFeatureInput()->sharesPartitionWith(featureInput));
    EXPECT_TRUE(softmax->getRaggedFeatureOutput()->sharesPartitionWith(featureInput));
    EXPECT_EQ(softmax->getRaggedFeatureOutput()->getRowPartitionToken(), featureInput.getRowPartitionToken());
    EXPECT_EQ(softmax->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{8, 2, 3}));

    shared_ptr<Layer> cloneLayer = softmax->clone();
    auto* clone = dynamic_cast<Softmax*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->getRaggedFeatureInput().has_value());
    ASSERT_TRUE(clone->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(clone->getRaggedFeatureInput()->sharesPartitionWith(featureInput));
    EXPECT_TRUE(clone->getRaggedFeatureOutput()->sharesPartitionWith(featureInput));
    EXPECT_NE(clone->getId(), softmax->getId());

    const json architecture = softmax->architectureJson();
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_EQ(architecture.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>());

    ThorImplementation::RaggedTensorDescriptor descriptor(
        DataType::FP32, {2, 3}, 2, 8, DataType::UINT64);
    ThorImplementation::RaggedExpression inputExpression =
        ThorImplementation::RaggedExpression::input("tokens.values",
                                                     "tokens.active_count",
                                                     descriptor,
                                                     ThorImplementation::RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
    ThorImplementation::RaggedExpression outputExpression = softmax->toRaggedExpression(inputExpression);
    EXPECT_EQ(outputExpression.getDescriptor(), descriptor);
    EXPECT_TRUE(outputExpression.getOffsets().isSameLogicalNode(inputExpression.getOffsets()));

    const ThorImplementation::PhysicalExpression physical = outputExpression.getValues().expression();
    const ThorImplementation::ExprNode& extent = physical.nodes.at(physical.output_node);
    ASSERT_EQ(extent.op, ThorImplementation::ExprOp::RAGGED_VALUEWISE_EXTENT);
    ASSERT_LT(extent.lhs, physical.nodes.size());
    const ThorImplementation::ExprNode& ordinarySoftmax = physical.nodes.at(extent.lhs);
    ASSERT_EQ(ordinarySoftmax.op, ThorImplementation::ExprOp::SOFTMAX);
    EXPECT_EQ(ordinarySoftmax.softmax_algorithm, CUDNN_SOFTMAX_ACCURATE);
    EXPECT_EQ(ordinarySoftmax.softmax_mode, CUDNN_SOFTMAX_MODE_CHANNEL);
}


TEST(Activations, RaggedSoftmaxStandaloneMaterializesManagedActiveCountWithoutFullOffsets) {
    constexpr uint32_t batchSize = 3;
    Network network("raggedSoftmaxManagedActiveCount");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT64)
                             .trailingDimensions({5})
                             .maxTotalValues(11)
                             .batchSize(batchSize)
                             .build();

    auto softmax = dynamic_pointer_cast<Softmax>(
        Softmax::Builder().network(network).featureInput(input).build());
    ASSERT_NE(softmax, nullptr);
    ASSERT_TRUE(softmax->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(softmax->getRaggedFeatureOutput()->sharesPartitionWith(input));
    RaggedNetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(softmax->getRaggedFeatureOutput().value())
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) event.synchronize();

    const auto& stamp = placed->getStampedNetwork(0);
    EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(input.getRowPartitionId()), nullptr);
    auto activeCount = stamp.getManagedPartitionActiveCountInputForTest(input.getRowPartitionId());
    ASSERT_NE(activeCount, nullptr);
    ASSERT_TRUE(activeCount->getFeatureOutput().has_value());
    EXPECT_EQ(activeCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(activeCount->getFeatureOutput()->getDataType(), DataType::UINT64);
}

TEST(Activations, SoftmaxSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    DataType dataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1;
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Softmax::Builder softmaxBuilder = Softmax::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput().value());
    shared_ptr<Softmax> softmax = dynamic_pointer_cast<Softmax>(softmaxBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(softmax->getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(softmax->isInitialized());

    Tensor featureInput = softmax->getFeatureInput().value();
    Tensor featureOutput = softmax->getFeatureOutput().value();
    assert(featureInput == networkInput.getFeatureOutput());

    ASSERT_TRUE(softmax->getFeatureOutput().has_value());
    ASSERT_EQ(softmax->getFeatureOutput().value(), featureOutput);

    ASSERT_TRUE(softmax->getFeatureInput().has_value());
    assert(softmax->getFeatureInput().value() == featureInput);

    ASSERT_EQ(featureInput.getDataType(), dataType);
    ASSERT_EQ(featureInput.getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutput.getDataType(), dataType);
    ASSERT_EQ(featureOutput.getDimensions(), inputDimensions);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> initialPlacedNetwork = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(initialPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the layer from the network
    ASSERT_EQ(initialPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialPlacedNetwork->getStampedNetwork(0);

    thor_file::TarWriter archiveWriter("testModel");

    json softmaxJ = softmax->serialize(archiveWriter, stream);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = softmax.get();
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(softmaxJ, fromLayerJ);

    ASSERT_EQ(softmaxJ["factory"], "activation");
    ASSERT_EQ(softmaxJ["version"], "1.0.0");
    ASSERT_EQ(softmaxJ["layer_type"], "softmax");

    EXPECT_TRUE(softmaxJ.contains("feature_input"));
    EXPECT_TRUE(softmaxJ.contains("feature_output"));

    const auto &input = softmaxJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    ASSERT_TRUE(input.at("data_type").is_string());
    string dataTypeString = dataType == DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(input.at("dimensions").is_array());
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = softmaxJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    ASSERT_TRUE(output.at("data_type").is_string());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(output.at("dimensions").is_array());
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    //     printf("%s\n", networkInputJ.dump(4).c_str());
    //     printf("%s\n", softmaxJ.dump(4).c_str());
    //     printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork("newNetwork");

    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Softmax::deserialize(softmaxJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &newStamp = newPlacedNetwork->getStampedNetwork(0);

    ASSERT_EQ(newStamp.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::CustomLayer> stampedSoftmax = dynamic_pointer_cast<ThorImplementation::CustomLayer>(newStamp.getTrainableLayer(0));
    ASSERT_NE(stampedSoftmax, nullptr);
    ASSERT_EQ(stampedSoftmax->getLayerType(), "CustomLayer<Softmax>");

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = newStamp.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = newStamp.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);

    ASSERT_TRUE(stampedInput->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedSoftmax->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedOutput->getFeatureOutput().has_value());
    ASSERT_EQ(stampedInput->getFeatureOutput().value(), stampedSoftmax->getFeatureInput().value());
    ASSERT_EQ(stampedSoftmax->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());

    filesystem::remove("/tmp/testModel.thor.tar");
}

TEST(Activations, SoftmaxRegistered) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    DataType dataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1;
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Softmax::Builder softmaxBuilder = Softmax::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput().value());
    shared_ptr<Softmax> softmax = dynamic_pointer_cast<Softmax>(softmaxBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(softmax->getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(softmax->isInitialized());

    thor_file::TarWriter archiveWriter("testModel");

    Stream stream(0);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json softmaxJ = softmax->serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Test that it is registered with Activation to deserialize
    Network newNetwork("newNetwork");
    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Activation::deserialize(softmaxJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    vector<Event> initDoneEvents;
    uint32_t batchSize = 1 + (rand() % 16);
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = newPlacedNetwork->getStampedNetwork(0);

    ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::CustomLayer> stampedSoftmax = dynamic_pointer_cast<ThorImplementation::CustomLayer>(stampedNetwork.getTrainableLayer(0));
    ASSERT_NE(stampedSoftmax, nullptr);
    ASSERT_EQ(stampedSoftmax->getLayerType(), "CustomLayer<Softmax>");
    filesystem::remove("/tmp/testModel.thor.tar");
}
