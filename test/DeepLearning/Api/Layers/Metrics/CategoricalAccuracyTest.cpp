#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Metrics/CategoricalAccuracy.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>
#include <nlohmann/json.hpp>

using namespace std;
using namespace Thor;
using json = nlohmann::json;

TEST(CategoricalAccuracy, ClassIndexLabelBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> labelDimensions = {1};
        vector<uint64_t> dimensions;
        uint64_t numClasses = 2UL + (rand() % 1000);
        dimensions = {numClasses};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType accuracyDataType = Tensor::DataType::FP32;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 3;
        if (r == 0)
            labelsDataType = Tensor::DataType::UINT8;
        else if (r == 1)
            labelsDataType = Tensor::DataType::UINT16;
        else
            labelsDataType = Tensor::DataType::UINT32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, labelDimensions);

        CategoricalAccuracy::Builder categoricalAccuracyBuilder =
            CategoricalAccuracy::Builder().network(network).predictions(predictions).labels(labels).receivesClassIndexLabels(numClasses);
        CategoricalAccuracy categoricalAccuracy = categoricalAccuracyBuilder.build();

        ASSERT_TRUE(categoricalAccuracy.isInitialized());

        Optional<Tensor> actualInput = categoricalAccuracy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLabels = categoricalAccuracy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), labelDimensions);

        Optional<Tensor> actualAccuracy = categoricalAccuracy.getFeatureOutput();
        ASSERT_TRUE(actualAccuracy.isPresent());
        ASSERT_EQ(actualAccuracy.get().getDataType(), accuracyDataType);
        ASSERT_EQ(actualAccuracy.get().getDimensions(), vector<uint64_t>({1}));

        shared_ptr<Layer> cloneLayer = categoricalAccuracy.clone();
        CategoricalAccuracy *clone = dynamic_cast<CategoricalAccuracy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLabels = clone->getLabels();
        ASSERT_TRUE(cloneLabels.isPresent());
        ASSERT_EQ(cloneLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(cloneLabels.get().getDimensions(), labelDimensions);

        Optional<Tensor> cloneAccuracy = clone->getFeatureOutput();
        ASSERT_TRUE(cloneAccuracy.isPresent());
        ASSERT_EQ(cloneAccuracy.get().getDataType(), accuracyDataType);
        ASSERT_EQ(cloneAccuracy.get().getDimensions(), vector<uint64_t>({1}));

        ASSERT_EQ(categoricalAccuracy.getId(), clone->getId());
        ASSERT_GT(categoricalAccuracy.getId(), 1u);

        ASSERT_TRUE(categoricalAccuracy == *clone);
        ASSERT_FALSE(categoricalAccuracy != *clone);
        ASSERT_FALSE(categoricalAccuracy > *clone);
        ASSERT_FALSE(categoricalAccuracy < *clone);
    }
}

TEST(Activations, CategoricalAccuracySerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;

    Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    uint64_t numClasses = 2U + (rand() % 20);
    vector<uint64_t> predictionsDimensions = {numClasses};
    NetworkInput predictionsNetworkInput = NetworkInput::Builder()
                                               .network(initialNetwork)
                                               .name("predictionsInput")
                                               .dimensions(predictionsDimensions)
                                               .dataType(predictionsDataType)
                                               .build();

    Tensor::DataType labelsDataType = rand() % 2 ? Tensor::DataType::UINT16 : Tensor::DataType::UINT32;
    bool oneHotLabels = rand() % 2;
    vector<uint64_t> labelsDimensions;
    if (oneHotLabels)
        labelsDimensions.push_back(numClasses);
    else
        labelsDimensions.push_back(1);
    NetworkInput labelsNetworkInput =
        NetworkInput::Builder().network(initialNetwork).name("labelsInput").dimensions(labelsDimensions).dataType(labelsDataType).build();

    CategoricalAccuracy::Builder builder = CategoricalAccuracy::Builder()
                                               .network(initialNetwork)
                                               .predictions(predictionsNetworkInput.getFeatureOutput())
                                               .labels(labelsNetworkInput.getFeatureOutput());

    if (oneHotLabels)
        builder.receivesOneHotLabels();
    else
        builder.receivesClassIndexLabels(numClasses);

    CategoricalAccuracy categoricalAccuracy = builder.build();

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(categoricalAccuracy.getMetric())
                                      .dataType(labelsDataType)
                                      .build();

    ASSERT_TRUE(categoricalAccuracy.isInitialized());

    Tensor predictionsInput = categoricalAccuracy.getPredictions();
    Tensor labelsInput = categoricalAccuracy.getLabels();
    Tensor metricTensor = categoricalAccuracy.getMetric();
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

    json categoricalAccuracyJ = categoricalAccuracy.serialize("/tmp/", stream);
    json predictionsNetworkInputJ = predictionsNetworkInput.serialize("/tmp/", stream);
    json labelsNetworkInputJ = labelsNetworkInput.serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    // printf("%s\n", predictionsNetworkInputJ.dump(4).c_str());
    // printf("%s\n", labelsNetworkInputJ.dump(4).c_str());
    // printf("%s\n", categoricalAccuracyJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());

    ASSERT_EQ(categoricalAccuracyJ["factory"], "metric");
    ASSERT_EQ(categoricalAccuracyJ["version"], "1.0.0");
    ASSERT_EQ(categoricalAccuracyJ["layer_type"], "categorical_accuracy");

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &categoricalAccuracy;
    json fromLayerJ = layer->serialize("/tmp/", stream);
    ASSERT_EQ(categoricalAccuracyJ, fromLayerJ);

    EXPECT_TRUE(categoricalAccuracyJ.contains("predictions"));
    EXPECT_TRUE(categoricalAccuracyJ.contains("labels"));
    EXPECT_TRUE(categoricalAccuracyJ.contains("metric"));
    EXPECT_TRUE(categoricalAccuracyJ.contains("label_type"));

    CategoricalAccuracy::LabelType jsonLabelType = categoricalAccuracyJ.at("label_type").get<CategoricalAccuracy::LabelType>();
    CategoricalAccuracy::LabelType labelType =
        oneHotLabels ? CategoricalAccuracy::LabelType::ONE_HOT : CategoricalAccuracy::LabelType::INDEX;
    ASSERT_EQ(jsonLabelType, labelType);

    const auto &predictions = categoricalAccuracyJ.at("predictions");
    ASSERT_TRUE(predictions.is_object());
    ASSERT_TRUE(predictions.at("data_type").is_string());
    string predictionsDataTypeString = predictionsDataType == Tensor::DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(predictions.at("data_type").get<string>(), predictionsDataTypeString);
    ASSERT_TRUE(predictions.at("dimensions").is_array());
    ASSERT_EQ(predictions.at("dimensions").get<vector<uint64_t>>(), predictionsDimensions);
    ASSERT_TRUE(predictions.at("id").is_number_integer());

    const auto &labels = categoricalAccuracyJ.at("labels");
    ASSERT_TRUE(labels.is_object());
    ASSERT_TRUE(labels.at("data_type").is_string());
    EXPECT_EQ(labels.at("data_type").get<Tensor::DataType>(), labelsDataType);
    ASSERT_TRUE(labels.at("dimensions").is_array());
    ASSERT_EQ(labels.at("dimensions").get<vector<uint64_t>>(), labelsDimensions);
    ASSERT_TRUE(labels.at("id").is_number_integer());

    vector<uint64_t> metricDimensions = {1U};
    const auto &metric = categoricalAccuracyJ.at("metric");
    ASSERT_TRUE(metric.is_object());
    ASSERT_TRUE(metric.at("data_type").is_string());
    EXPECT_EQ(metric.at("data_type").get<string>(), "fp32");
    ASSERT_TRUE(metric.at("dimensions").is_array());
    ASSERT_EQ(metric.at("dimensions").get<vector<uint64_t>>(), metricDimensions);
    ASSERT_TRUE(metric.at("id").is_number_integer());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    NetworkInput::deserialize(predictionsNetworkInputJ, &newNetwork);
    NetworkInput::deserialize(labelsNetworkInputJ, &newNetwork);
    CategoricalAccuracy::deserialize(categoricalAccuracyJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

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
    shared_ptr<ThorImplementation::CategoricalAccuracy> stampedCategoricalAccuracy =
        dynamic_pointer_cast<ThorImplementation::CategoricalAccuracy>(otherLayers[0]);
    ASSERT_NE(stampedCategoricalAccuracy, nullptr);

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

    if (stampedInput0->getFeatureOutput().get() == stampedCategoricalAccuracy->getFeatureInput().get()) {
        ASSERT_EQ(stampedInput1->getFeatureOutput().get(), stampedCategoricalAccuracy->getLabelsInput().get());
    } else {
        ASSERT_EQ(stampedInput0->getFeatureOutput().get(), stampedCategoricalAccuracy->getLabelsInput().get());
        ASSERT_EQ(stampedInput1->getFeatureOutput().get(), stampedCategoricalAccuracy->getFeatureInput().get());
    }
    ASSERT_EQ(stampedCategoricalAccuracy->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());
}
