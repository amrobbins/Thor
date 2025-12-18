#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/BinaryCrossEntropy.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace std;

using namespace Thor;
using json = nlohmann::json;

TEST(BinaryCrossEntropy, BatchLossBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions = {1UL};
        vector<uint64_t> lossDimensions = {1};
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 2;
        if (r == 0)
            labelsDataType = Tensor::DataType::FP16;
        else if (r == 1)
            labelsDataType = Tensor::DataType::FP32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        BinaryCrossEntropy::Builder crossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                              .network(network)
                                                              .predictions(predictions)
                                                              .reportsBatchLoss()
                                                              .lossDataType(lossDataType)
                                                              .labels(labels);
        BinaryCrossEntropy crossEntropy = crossEntropyBuilder.build();

        ASSERT_TRUE(crossEntropy.isInitialized());

        Optional<Tensor> actualInput = crossEntropy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLabels = crossEntropy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualPredictions = crossEntropy.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

        Optional<Tensor> actualLoss = crossEntropy.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(actualLoss.get().getDimensions(), lossDimensions);

        shared_ptr<Layer> cloneLayer = crossEntropy.clone();
        BinaryCrossEntropy *clone = dynamic_cast<BinaryCrossEntropy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(cloneLoss.get().getDimensions(), lossDimensions);

        ASSERT_EQ(crossEntropy.getId(), clone->getId());
        ASSERT_GT(crossEntropy.getId(), 1u);

        ASSERT_TRUE(crossEntropy == *clone);
        ASSERT_FALSE(crossEntropy != *clone);
        ASSERT_FALSE(crossEntropy > *clone);
        ASSERT_FALSE(crossEntropy < *clone);
    }
}

TEST(BinaryCrossEntropy, ElementwiseLossBuilds) {
    srand(time(nullptr));

    for (uint32_t t = 0; t < 10; ++t) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions = {1UL};
        vector<uint64_t> lossDimensions = dimensions;
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

        Tensor::DataType labelsDataType;
        uint32_t r = rand() % 2;
        if (r == 0)
            labelsDataType = Tensor::DataType::FP16;
        else if (r == 1)
            labelsDataType = Tensor::DataType::FP32;

        Tensor predictions(predictionsDataType, dimensions);
        Tensor labels(labelsDataType, dimensions);

        BinaryCrossEntropy::Builder crossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                              .network(network)
                                                              .predictions(predictions)
                                                              .reportsElementwiseLoss()
                                                              .lossDataType(lossDataType)
                                                              .labels(labels);
        BinaryCrossEntropy crossEntropy = crossEntropyBuilder.build();

        ASSERT_TRUE(crossEntropy.isInitialized());

        Optional<Tensor> actualInput = crossEntropy.getFeatureInput();
        ASSERT_TRUE(actualInput.isPresent());
        ASSERT_EQ(actualInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLabels = crossEntropy.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualPredictions = crossEntropy.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

        Optional<Tensor> actualLoss = crossEntropy.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(actualLoss.get().getDimensions(), lossDimensions);

        shared_ptr<Layer> cloneLayer = crossEntropy.clone();
        BinaryCrossEntropy *clone = dynamic_cast<BinaryCrossEntropy *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> cloneInput = clone->getFeatureInput();
        ASSERT_TRUE(cloneInput.isPresent());
        ASSERT_EQ(cloneInput.get().getDataType(), predictionsDataType);
        ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), lossDataType);
        ASSERT_EQ(cloneLoss.get().getDimensions(), lossDimensions);

        ASSERT_EQ(crossEntropy.getId(), clone->getId());
        ASSERT_GT(crossEntropy.getId(), 1u);

        ASSERT_TRUE(crossEntropy == *clone);
        ASSERT_FALSE(crossEntropy != *clone);
        ASSERT_FALSE(crossEntropy > *clone);
        ASSERT_FALSE(crossEntropy < *clone);
    }
}

TEST(BinaryCrossEntropy, SerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Tensor::DataType dataType = Tensor::DataType::FP16;
    vector<uint64_t> inputDimensions = {1UL};
    Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;

    NetworkInput labelsInput =
        NetworkInput::Builder().network(initialNetwork).name("labelsInput").dimensions(inputDimensions).dataType(dataType).build();
    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("networkInput").dimensions(inputDimensions).dataType(dataType).build();

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(initialNetwork)
                                        .featureInput(networkInput.getFeatureOutput())
                                        .numOutputFeatures(1)
                                        .hasBias(false)
                                        .noActivation()
                                        .build();

    BinaryCrossEntropy::Builder binaryCrossEntropyBuilder = BinaryCrossEntropy::Builder()
                                                                .network(initialNetwork)
                                                                .labels(labelsInput.getFeatureOutput())
                                                                .predictions(fullyConnected.getFeatureOutput())
                                                                .lossDataType(lossDataType);

    uint32_t lossShape = 0;  // rand() % 2;
    if (lossShape == 0)
        binaryCrossEntropyBuilder.reportsBatchLoss();
    else
        // FIXME: Get the following to work, seems error output path issue
        binaryCrossEntropyBuilder.reportsElementwiseLoss();

    BinaryCrossEntropy binaryCrossEntropy = binaryCrossEntropyBuilder.build();

    shared_ptr<Sgd> sgd = Sgd::Builder().network(initialNetwork).initialLearningRate(0.1).decay(0.1).build();

    NetworkOutput lossOutput = NetworkOutput::Builder()
                                   .network(initialNetwork)
                                   .name("lossOutput")
                                   .inputTensor(binaryCrossEntropy.getLoss())
                                   .dataType(dataType)
                                   .build();

    ASSERT_TRUE(binaryCrossEntropy.isInitialized());

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

    // Find the sigmoid layer in the network so can serialize it for this test case
    shared_ptr<Sigmoid> sigmoid;
    shared_ptr<LossShaper> lossShaper;
    bool sigmoidFound = false;
    bool lossShaperFound = false;
    for (int32_t i = 0; i < initialNetwork.getNumLayers(); ++i) {
        shared_ptr<Layer> layer = initialNetwork.getLayer(i);
        if (!sigmoidFound) {
            sigmoid = dynamic_pointer_cast<Sigmoid>(layer);
            if (sigmoid)
                sigmoidFound = true;
        }
        if (!lossShaperFound) {
            lossShaper = dynamic_pointer_cast<LossShaper>(layer);
            if (lossShaper)
                lossShaperFound = true;
        }
        if (sigmoidFound && lossShaperFound)
            break;
    }
    ASSERT_TRUE(sigmoidFound);
    ASSERT_TRUE(lossShaperFound);

    json labelsInputJ = labelsInput.serialize("/tmp/", stream);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json sigmoidJ = sigmoid->serialize("/tmp/", stream);
    Layer *layer = &binaryCrossEntropy;
    json binaryCrossEntropyJ = layer->serialize("/tmp/", stream);
    json lossShaperJ = lossShaper->serialize("/tmp/", stream);
    json lossOutputJ = lossOutput.serialize("/tmp/", stream);

    ASSERT_EQ(binaryCrossEntropyJ["factory"], "loss");
    ASSERT_EQ(binaryCrossEntropyJ["version"], "1.0.0");
    ASSERT_EQ(binaryCrossEntropyJ["layer_type"], "binary_cross_entropy");
    EXPECT_TRUE(binaryCrossEntropyJ.contains("layer_name"));
    ASSERT_EQ(binaryCrossEntropyJ.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::ELEMENTWISE);
    ASSERT_EQ(binaryCrossEntropyJ.at("loss_data_type").get<Tensor::DataType>(), lossDataType);

    const json &labelsJ = binaryCrossEntropyJ["labels_tensor"];
    ASSERT_EQ(labelsJ.at("data_type").get<Tensor::DataType>(), dataType);
    ASSERT_EQ(labelsJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(labelsJ.at("id").is_number_integer());

    const json &predictionsJ = binaryCrossEntropyJ["predictions_tensor"];
    ASSERT_EQ(predictionsJ.at("data_type").get<Tensor::DataType>(), dataType);
    ASSERT_EQ(predictionsJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(predictionsJ.at("id").is_number_integer());

    const json &sigmoidOutputJ = binaryCrossEntropyJ["sigmoid_output_tensor"];
    ASSERT_EQ(sigmoidOutputJ.at("data_type").get<Tensor::DataType>(), dataType);
    ASSERT_EQ(sigmoidOutputJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(sigmoidOutputJ.at("id").is_number_integer());

    const json &lossShaperInputJ = binaryCrossEntropyJ["loss_shaper_input_tensor"];
    ASSERT_EQ(lossShaperInputJ.at("data_type").get<Tensor::DataType>(), lossDataType);
    ASSERT_EQ(lossShaperInputJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(lossShaperInputJ.at("id").is_number_integer());

    const json &lossJ = binaryCrossEntropyJ["loss_tensor"];
    ASSERT_EQ(lossJ.at("data_type").get<Tensor::DataType>(), lossDataType);
    ASSERT_EQ(lossJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(lossJ.at("id").is_number_integer());

    // The network attached the optimizer to its copy of the FC layer
    json fullyConnectedJ;
    bool fcFound = false;
    shared_ptr<FullyConnected> initalNetworkFC;
    for (int32_t i = 0; i < initialNetwork.getNumTrainableLayers(); ++i) {
        shared_ptr<TrainableWeightsBiasesLayer> layer = initialNetwork.getTrainableLayer(i);
        initalNetworkFC = dynamic_pointer_cast<FullyConnected>(layer);
        if (initalNetworkFC) {
            fullyConnectedJ = initalNetworkFC->serialize("/tmp", stream);
            fcFound = true;
            break;
        }
    }
    ASSERT_TRUE(fcFound);

    printf("%s\n", networkInputJ.dump(4).c_str());
    printf("%s\n", labelsInputJ.dump(4).c_str());
    printf("%s\n", fullyConnectedJ.dump(4).c_str());
    printf("%s\n", sigmoidJ.dump(4).c_str());
    printf("%s\n", binaryCrossEntropyJ.dump(4).c_str());
    printf("%s\n", lossShaperJ.dump(4).c_str());
    printf("%s\n", lossOutputJ.dump(4).c_str());

    // ////////////////////////////
    // // Deserialize
    // ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    // The sigmoid output is not loaded, probably it is restamped? Oh it needs to be serialized so that it looks like a single layer or
    // stamping will not work, I did this for FC etc.

    Layer::deserialize(networkInputJ, &newNetwork);
    Layer::deserialize(labelsInputJ, &newNetwork);
    Layer::deserialize(fullyConnectedJ, &newNetwork);
    Layer::deserialize(sigmoidJ, &newNetwork);
    Layer::deserialize(binaryCrossEntropyJ, &newNetwork);
    Layer::deserialize(lossShaperJ, &newNetwork);
    Layer::deserialize(lossOutputJ, &newNetwork);

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
    ASSERT_EQ(otherLayers.size(), 3U);
    // shared_ptr<ThorImplementation::CrossEntropy> stampedBinaryCrossEntropy =
    //     dynamic_pointer_cast<ThorImplementation::CrossEntropy>(otherLayers[0]);
    shared_ptr<ThorImplementation::Sigmoid> stampedSigmoid;
    shared_ptr<ThorImplementation::CrossEntropy> stampedBinaryCrossEntropy;
    shared_ptr<ThorImplementation::LossShaper> stampedLossShaper;
    sigmoidFound = false;
    bool crossEntropyFound = false;
    lossShaperFound = false;
    for (shared_ptr<ThorImplementation::Layer> layer : otherLayers) {
        if (!sigmoidFound) {
            stampedSigmoid = dynamic_pointer_cast<ThorImplementation::Sigmoid>(layer);
            if (stampedSigmoid != nullptr)
                sigmoidFound = true;
        }
        if (!crossEntropyFound) {
            stampedBinaryCrossEntropy = dynamic_pointer_cast<ThorImplementation::CrossEntropy>(layer);
            if (stampedBinaryCrossEntropy != nullptr)
                crossEntropyFound = true;
        }
        if (!lossShaperFound) {
            stampedLossShaper = dynamic_pointer_cast<ThorImplementation::LossShaper>(layer);
            if (stampedLossShaper != nullptr)
                lossShaperFound = true;
        }
    }
    assert(sigmoidFound);
    assert(crossEntropyFound);
    assert(lossShaperFound);

    ASSERT_NE(stampedBinaryCrossEntropy, nullptr);

    shared_ptr<ThorImplementation::FullyConnected> stampedFC =
        dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(0));
    ASSERT_NE(stampedFC, nullptr);

    // vector<shared_ptr<ThorImplementation::NetworkInput>> stampedInputs = stampedNetwork.getInputs();
    shared_ptr<ThorImplementation::NetworkInput> stampedLabelsInput;
    for (auto input : stampedNetwork.getInputs()) {
        if (input->getName() == "labelsInput") {
            stampedLabelsInput = input;
        }
    }
    ASSERT_TRUE(stampedLabelsInput != nullptr);

    shared_ptr<ThorImplementation::NetworkOutput> stampedLossOutput;
    stampedLossOutput = stampedNetwork.getOutputs()[0];

    ASSERT_EQ(stampedSigmoid->getFeatureInput().get(), stampedFC->getFeatureOutputs()[0].get());
    ASSERT_EQ(stampedBinaryCrossEntropy->getPredictionsInput().get(), stampedSigmoid->getFeatureOutput().get());
    ASSERT_EQ(stampedBinaryCrossEntropy->getLabelsInput().get(), stampedLabelsInput->getFeatureOutput().get());
    ASSERT_EQ(stampedBinaryCrossEntropy->getLossOutput().get(), stampedLossShaper->getFeatureInput().get());
    ASSERT_EQ(stampedLossShaper->getFeatureOutput().get(), stampedLossOutput->getFeatureInput().get());
}
