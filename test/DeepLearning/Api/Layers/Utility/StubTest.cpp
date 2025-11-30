#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, StubBuilds) {
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

TEST(UtilityApiLayers, StubSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Stream stream(0);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    string dataTypeString = dataType == Tensor::DataType::FP32 ? "fp32" : "fp16";

    uint32_t numDimensions = 1 + (rand() % 4);

    vector<uint64_t> dimensions;
    for (uint32_t d = 0; d < numDimensions; ++d) {
        dimensions.push_back(1 + (rand() % 5));
    }

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(dimensions).dataType(dataType).build();

    Stub stub = Stub::Builder().network(initialNetwork).inputTensor(networkInput.getFeatureOutput().get()).build();
    ASSERT_TRUE(stub.isInitialized());

    json stubJ = stub.serialize("/tmp/", stream);

    // printf("%s\n", stubJ.dump(4).c_str());

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &stub;
    json fromLayerJ = layer->serialize("/tmp/", stream);
    ASSERT_EQ(stubJ, fromLayerJ);

    json networkInputJ = networkInput.serialize("/tmp/", stream);

    ASSERT_EQ(stubJ["factory"], Layer::Factory::Layer.value());
    ASSERT_EQ(stubJ["version"], "1.0.0");
    ASSERT_EQ(stubJ["layer_type"], "stub");

    const auto &input = stubJ.at("input_tensor");
    ASSERT_TRUE(input.is_object());
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), dimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    Network newNetwork;

    Layer::deserialize(networkInputJ, &newNetwork);
    Layer::deserialize(stubJ, &newNetwork);

    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode;
    statusCode = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork stampedNetwork = newNetwork.getStampedNetwork(0);
    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 0U);

    // There is no stamped stub layer.
    // The fact that the network successfully places, with an otherwise dangling output, is the test that the stub layer worked.
    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    vector<uint64_t> stampedDimensions = {batchSize};
    for (uint32_t d = 0; d < numDimensions; ++d)
        stampedDimensions.push_back(dimensions[d]);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 0U);

    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(stampedInput, nullptr);
    ASSERT_TRUE(stampedInput->getFeatureOutput().isEmpty());
}
