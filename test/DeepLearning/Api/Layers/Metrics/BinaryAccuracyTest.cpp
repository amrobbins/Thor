#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Metrics/BinaryAccuracy.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>
#include <nlohmann/json.hpp>

using namespace std;
using namespace Thor;
using json = nlohmann::json;

TEST(BinaryAccuracy, Builds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions = {1};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType accuracyDataType = Tensor::DataType::FP32;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 8;
        if (r == 0)
            labelsDataType = Tensor::DataType::UINT8;
        else if (r == 1)
            labelsDataType = Tensor::DataType::UINT16;
        else if (r == 2)
            labelsDataType = Tensor::DataType::UINT32;
        else if (r == 3)
            labelsDataType = Tensor::DataType::INT8;
        else if (r == 4)
            labelsDataType = Tensor::DataType::INT16;
        else if (r == 5)
            labelsDataType = Tensor::DataType::INT32;
        else if (r == 6)
            labelsDataType = Tensor::DataType::FP16;
        else if (r == 7)
            labelsDataType = Tensor::DataType::FP32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        BinaryAccuracy::Builder binaryAccuracyBuilder = BinaryAccuracy::Builder().network(network).predictions(predictions).labels(labels);
        BinaryAccuracy binaryAccuracy = binaryAccuracyBuilder.build();

        ASSERT_TRUE(binaryAccuracy.isInitialized());

        Optional<Tensor> actualInput = binaryAccuracy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLabels = binaryAccuracy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualAccuracy = binaryAccuracy.getFeatureOutput();
        ASSERT_TRUE(actualAccuracy.isPresent());
        ASSERT_EQ(actualAccuracy.get().getDataType(), accuracyDataType);
        ASSERT_EQ(actualAccuracy.get().getDimensions(), vector<uint64_t>({1}));

        shared_ptr<Layer> cloneLayer = binaryAccuracy.clone();
        BinaryAccuracy *clone = dynamic_cast<BinaryAccuracy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLabels = clone->getLabels();
        ASSERT_TRUE(cloneLabels.isPresent());
        ASSERT_EQ(cloneLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(cloneLabels.get().getDimensions(), dimensions);

        Optional<Tensor> cloneAccuracy = clone->getFeatureOutput();
        ASSERT_TRUE(cloneAccuracy.isPresent());
        ASSERT_EQ(cloneAccuracy.get().getDataType(), accuracyDataType);
        ASSERT_EQ(cloneAccuracy.get().getDimensions(), vector<uint64_t>({1}));

        ASSERT_EQ(binaryAccuracy.getId(), clone->getId());
        ASSERT_GT(binaryAccuracy.getId(), 1u);

        ASSERT_TRUE(binaryAccuracy == *clone);
        ASSERT_FALSE(binaryAccuracy != *clone);
        ASSERT_FALSE(binaryAccuracy > *clone);
        ASSERT_FALSE(binaryAccuracy < *clone);
    }
}

TEST(Activations, BinaryAccuracySerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;

    Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    vector<uint64_t> predictionsDimensions = {1};
    NetworkInput predictionsNetworkInput = NetworkInput::Builder()
                                               .network(initialNetwork)
                                               .name("predictionsInput")
                                               .dimensions(predictionsDimensions)
                                               .dataType(predictionsDataType)
                                               .build();

    Tensor::DataType labelsDataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    vector<uint64_t> labelsDimensions = {1U};
    NetworkInput labelsNetworkInput =
        NetworkInput::Builder().network(initialNetwork).name("labelsInput").dimensions(labelsDimensions).dataType(labelsDataType).build();

    BinaryAccuracy binaryAccuracy = BinaryAccuracy::Builder()
                                        .network(initialNetwork)
                                        .predictions(predictionsNetworkInput.getFeatureOutput())
                                        .labels(labelsNetworkInput.getFeatureOutput())
                                        .build();

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(binaryAccuracy.getMetric())
                                      .dataType(labelsDataType)
                                      .build();

    ASSERT_TRUE(binaryAccuracy.isInitialized());

    Tensor predictionsInput = binaryAccuracy.getPredictions();
    Tensor labelsInput = binaryAccuracy.getLabels();
    Tensor metricTensor = binaryAccuracy.getMetric();
    assert(predictionsInput == predictionsNetworkInput.getFeatureOutput());
    assert(labelsInput == labelsNetworkInput.getFeatureOutput());
    assert(metricTensor == networkOutput.getFeatureInput());

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    Network::StatusCode placementStatus;
    placementStatus = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the layer from the network
    ASSERT_EQ(initialNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialNetwork.getStampedNetwork(0);

    thor_file::TarWriter archiveWriter("testModel", "/tmp/", true);

    json binaryAccuracyJ = binaryAccuracy.serialize(archiveWriter, stream);
    json predictionsNetworkInputJ = predictionsNetworkInput.serialize(archiveWriter, stream);
    json labelsNetworkInputJ = labelsNetworkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // printf("%s\n", predictionsNetworkInputJ.dump(4).c_str());
    // printf("%s\n", labelsNetworkInputJ.dump(4).c_str());
    // printf("%s\n", binaryAccuracyJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());

    ASSERT_EQ(binaryAccuracyJ["factory"], "metric");
    ASSERT_EQ(binaryAccuracyJ["version"], "1.0.0");
    ASSERT_EQ(binaryAccuracyJ["layer_type"], "binary_accuracy");

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &binaryAccuracy;
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(binaryAccuracyJ, fromLayerJ);

    EXPECT_TRUE(binaryAccuracyJ.contains("predictions"));
    EXPECT_TRUE(binaryAccuracyJ.contains("labels"));
    EXPECT_TRUE(binaryAccuracyJ.contains("metric"));

    const auto &predictions = binaryAccuracyJ.at("predictions");
    ASSERT_TRUE(predictions.is_object());
    ASSERT_TRUE(predictions.at("data_type").is_string());
    string predictionsDataTypeString = predictionsDataType == Tensor::DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(predictions.at("data_type").get<string>(), predictionsDataTypeString);
    ASSERT_TRUE(predictions.at("dimensions").is_array());
    ASSERT_EQ(predictions.at("dimensions").get<vector<uint64_t>>(), predictionsDimensions);
    ASSERT_TRUE(predictions.at("id").is_number_integer());

    const auto &labels = binaryAccuracyJ.at("labels");
    ASSERT_TRUE(labels.is_object());
    ASSERT_TRUE(labels.at("data_type").is_string());
    string labelsDataTypeString = labelsDataType == Tensor::DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(labels.at("data_type").get<string>(), labelsDataTypeString);
    ASSERT_TRUE(labels.at("dimensions").is_array());
    ASSERT_EQ(labels.at("dimensions").get<vector<uint64_t>>(), labelsDimensions);
    ASSERT_TRUE(labels.at("id").is_number_integer());

    const auto &metric = binaryAccuracyJ.at("metric");
    ASSERT_TRUE(metric.is_object());
    ASSERT_TRUE(metric.at("data_type").is_string());
    EXPECT_EQ(metric.at("data_type").get<string>(), "fp32");
    ASSERT_TRUE(metric.at("dimensions").is_array());
    ASSERT_EQ(metric.at("dimensions").get<vector<uint64_t>>(), predictionsDimensions);
    ASSERT_TRUE(metric.at("id").is_number_integer());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    archiveWriter.finishArchive();
    shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");

    Layer::deserialize(archiveReader, predictionsNetworkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, labelsNetworkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, binaryAccuracyJ, &newNetwork);
    Layer::deserialize(archiveReader, networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    placementStatus = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    stampedNetwork = newNetwork.getStampedNetwork(0);

    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = stampedNetwork.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::BinaryAccuracy> stampedBinaryAccuracy =
        dynamic_pointer_cast<ThorImplementation::BinaryAccuracy>(otherLayers[0]);
    ASSERT_NE(stampedBinaryAccuracy, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = stampedNetwork.getInputs();
    ASSERT_EQ(inputLayers.size(), 2U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput0 = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput1 = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[1]);
    ASSERT_NE(inputLayers[1], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = stampedNetwork.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);

    ASSERT_TRUE(stampedInput0->getFeatureOutput().isPresent());
    ASSERT_TRUE(stampedInput1->getFeatureOutput().isPresent());
    ASSERT_TRUE(stampedOutput->getFeatureInput().isPresent());

    if (stampedInput0->getFeatureOutput().get() == stampedBinaryAccuracy->getFeatureInput().get()) {
        ASSERT_EQ(stampedInput1->getFeatureOutput().get(), stampedBinaryAccuracy->getLabelsInput().get());
    } else {
        ASSERT_EQ(stampedInput0->getFeatureOutput().get(), stampedBinaryAccuracy->getLabelsInput().get());
        ASSERT_EQ(stampedInput1->getFeatureOutput().get(), stampedBinaryAccuracy->getFeatureInput().get());
    }
    ASSERT_EQ(stampedBinaryAccuracy->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());
}
