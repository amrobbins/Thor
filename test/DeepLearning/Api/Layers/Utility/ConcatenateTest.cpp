#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, ConcatenateBuilds) {
    srand(time(nullptr));

    Network network;

    uint32_t numDimensions = 1 + (rand() % 4);
    uint32_t concatenationAxis = rand() % numDimensions;
    uint32_t concatenationAxisSize = 1 + (rand() % 50);
    vector<uint64_t> concatenatedDimensions(numDimensions, 0U);
    uint32_t numTensors = 1 + (rand() % 5);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    vector<uint64_t> fixedDimensionSize;
    for (uint32_t d = 0; d < numDimensions; ++d) {
        fixedDimensionSize.push_back(1 + (rand() % 5));
    }
    concatenatedDimensions = fixedDimensionSize;
    concatenatedDimensions[concatenationAxis] = 0;

    vector<vector<uint64_t>> tensorDimensions;
    for (uint32_t t = 0; t < numTensors; ++t) {
        tensorDimensions.emplace_back();
        for (uint32_t d = 0; d < numDimensions; ++d) {
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

TEST(UtilityApiLayers, ConcatenateSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Stream stream(0);

    uint32_t numDimensions = 1 + (rand() % 4);
    uint32_t concatenationAxis = rand() % numDimensions;
    vector<uint64_t> concatenatedDimensions(numDimensions, 0U);
    uint64_t numTensors = 2 + (rand() % 4);

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    string dataTypeString = dataType == Tensor::DataType::FP32 ? "fp32" : "fp16";

    vector<uint64_t> fixedDimensionSize;
    for (uint32_t d = 0; d < numDimensions; ++d) {
        fixedDimensionSize.push_back(1 + (rand() % 5));
    }
    concatenatedDimensions = fixedDimensionSize;
    concatenatedDimensions[concatenationAxis] = 0;

    vector<vector<uint64_t>> tensorDimensions;
    for (uint32_t t = 0; t < numTensors; ++t) {
        tensorDimensions.emplace_back();
        for (uint32_t d = 0; d < numDimensions; ++d) {
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

    vector<NetworkInput> networkInputs;
    for (uint32_t t = 0; t < numTensors; ++t) {
        NetworkInput networkInput = NetworkInput::Builder()
                                        .network(initialNetwork)
                                        .name("testInput" + t)
                                        .dimensions(tensorDimensions[t])
                                        .dataType(dataType)
                                        .build();
        networkInputs.push_back(networkInput);
    }

    Concatenate::Builder concatenateBuilder = Concatenate::Builder().network(initialNetwork).concatenationAxis(concatenationAxis);
    for (uint32_t t = 0; t < numTensors; ++t) {
        concatenateBuilder.featureInput(networkInputs[t].getFeatureOutput().get());
    }
    Concatenate concatenate = concatenateBuilder.build();
    ASSERT_TRUE(concatenate.isInitialized());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(concatenate.getFeatureOutputs()[0])
                                      .dataType(dataType)
                                      .build();

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

    json concatenateJ = concatenate.serialize(archiveWriter, stream);

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &concatenate;
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(concatenateJ, fromLayerJ);

    vector<json> networkInputJs;
    for (uint32_t t = 0; t < numTensors; ++t) {
        json networkInputJ = networkInputs[t].serialize(archiveWriter, stream);
        networkInputJs.push_back(networkInputJ);
    }
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    ASSERT_EQ(concatenateJ["factory"], Layer::Factory::Layer.value());
    ASSERT_EQ(concatenateJ["version"], "1.0.0");
    ASSERT_EQ(concatenateJ["layer_type"], "concatenate");

    ASSERT_EQ(concatenateJ["concatenation_axis"], concatenationAxis);
    EXPECT_TRUE(concatenateJ.contains("inputs"));
    EXPECT_TRUE(concatenateJ.contains("outputs"));
    ASSERT_TRUE(concatenateJ.at("inputs").is_array());
    ASSERT_TRUE(concatenateJ.at("outputs").is_array());

    // printf("%s\n", concatenateJ.dump(4).c_str());

    const auto &inputs = concatenateJ.at("inputs");
    ASSERT_EQ(inputs.size(), numTensors);
    for (uint32_t t = 0; t < numTensors; ++t) {
        const auto &input = inputs.at(t);
        ASSERT_TRUE(input.is_object());
        ASSERT_TRUE(input.at("data_type").is_string());
        EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
        ASSERT_TRUE(input.at("id").is_number_integer());

        ASSERT_TRUE(input.at("dimensions").is_array());
        ASSERT_EQ(input.at("dimensions").size(), numDimensions);
        EXPECT_EQ(input.at("dimensions").get<vector<uint64_t>>(), tensorDimensions[t]);
    }

    const auto &outputs = concatenateJ.at("outputs");
    ASSERT_EQ(outputs.size(), 1U) << "Expect exactly one output";
    const auto &out0 = outputs.at(0);
    ASSERT_TRUE(out0.is_object());
    ASSERT_TRUE(out0.at("data_type").is_string());
    EXPECT_EQ(out0.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(out0.at("id").is_number_integer());

    ASSERT_TRUE(out0.at("dimensions").is_array());
    ASSERT_EQ(out0.at("dimensions").size(), numDimensions);
    EXPECT_TRUE(out0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(out0.at("dimensions").get<vector<uint64_t>>(), concatenatedDimensions);

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    Network newNetwork;

    for (uint32_t t = 0; t < numTensors; ++t)
        NetworkInput::deserialize(networkInputJs[t], &newNetwork);

    archiveWriter.finishArchive();
    shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");

    Layer::deserialize(archiveReader, concatenateJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

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
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::Concatenate> stampedConcatenate = dynamic_pointer_cast<ThorImplementation::Concatenate>(otherLayers[0]);
    ASSERT_NE(stampedConcatenate, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), numTensors);
    for (uint32_t t = 0; t < numTensors; ++t) {
        vector<uint64_t> stampedDimensions = {batchSize};
        for (uint32_t d = 0; d < numDimensions; ++d)
            stampedDimensions.push_back(tensorDimensions[t][d]);

        shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[t]);
        ASSERT_NE(stampedInput, nullptr);
        ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
        ASSERT_EQ(stampedInput->getFeatureOutput().get().getDimensions(), stampedDimensions);
    }

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);
    vector<uint64_t> stampedDimensions = {batchSize};
    for (uint32_t d = 0; d < numDimensions; ++d) {
        stampedDimensions.push_back(concatenatedDimensions[d]);
    }
    ASSERT_EQ(stampedOutput->getFeatureOutput().get().getDimensions(), stampedDimensions);

    for (uint32_t t = 0; t < numTensors; ++t) {
        ASSERT_TRUE(inputLayers[t]->getFeatureOutput().isPresent());
    }
    ASSERT_EQ(stampedConcatenate->getFeatureOutputs().size(), 1U);
    ASSERT_TRUE(stampedOutput->getFeatureOutput().isPresent());

    // Ensure 1. that they are all connected and 2. that they are in the same order as pre-serialization
    for (uint32_t t = 0; t < numTensors; ++t) {
        ASSERT_TRUE(inputLayers[t]->getFeatureOutput().isPresent());
        ASSERT_TRUE(stampedConcatenate->getFeatureInputs()[t].isPresent());
        EXPECT_EQ(inputLayers[t]->getFeatureOutput().get().getTensorId(), stampedConcatenate->getFeatureInputs()[t].get().getTensorId());
        EXPECT_EQ(inputLayers[t]->getFeatureOutput().get(), stampedConcatenate->getFeatureInputs()[t].get());
    }
    ASSERT_EQ(stampedConcatenate->getFeatureOutputs()[0].get(), stampedOutput->getFeatureInput().get());
}
