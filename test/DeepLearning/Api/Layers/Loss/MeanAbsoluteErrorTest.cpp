#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(MeanAbsoluteError, Builds) {
    srand(time(nullptr));

    for (uint32_t i = 0; i < 10; ++i) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions.push_back(1 + (rand() % 1000));
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor predictions(predictionsDataType, dimensions);

        Tensor::DataType labelsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor labels(labelsDataType, dimensions);

        MeanAbsoluteError::Builder meanAbsoluteErrorBuilder =
            MeanAbsoluteError::Builder().network(network).predictions(predictions).labels(labels);

        uint32_t shape = rand() % 4;
        if (shape == 0) {
            meanAbsoluteErrorBuilder.reportsBatchLoss();
        } else if (shape == 1) {
            meanAbsoluteErrorBuilder.reportsElementwiseLoss();
        } else if (shape == 2) {
            meanAbsoluteErrorBuilder.reportsPerOutputLoss();
        } else if (shape == 3) {
            meanAbsoluteErrorBuilder.reportsRawLoss();
        } else {
            assert(false);
        }
        vector<uint64_t> batchDimensions = {1};
        vector<uint64_t> elementwiseDimensions = {1};
        vector<uint64_t> perOutputDimensions = {dimensions[0]};
        vector<uint64_t> rawLossDimensions = dimensions;

        MeanAbsoluteError meanAbsoluteError = meanAbsoluteErrorBuilder.build();

        ASSERT_TRUE(meanAbsoluteError.isInitialized());

        Optional<Tensor> actualLabels = meanAbsoluteError.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualPredictions = meanAbsoluteError.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

        Optional<Tensor> actualLoss = meanAbsoluteError.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), predictionsDataType);
        if (shape == 0) {
            ASSERT_EQ(actualLoss.get().getDimensions(), batchDimensions);
        } else if (shape == 1) {
            ASSERT_EQ(actualLoss.get().getDimensions(), elementwiseDimensions);
        } else if (shape == 2) {
            ASSERT_EQ(actualLoss.get().getDimensions(), perOutputDimensions);
        } else if (shape == 3) {
            ASSERT_EQ(actualLoss.get().getDimensions(), rawLossDimensions);
        } else {
            assert(false);
        }

        ASSERT_TRUE(meanAbsoluteError.getPredictions() == meanAbsoluteError.getFeatureInput());
        ASSERT_TRUE(meanAbsoluteError.getLoss() == meanAbsoluteError.getFeatureOutput());

        shared_ptr<Layer> cloneLayer = meanAbsoluteError.clone();
        MeanAbsoluteError *clone = dynamic_cast<MeanAbsoluteError *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), predictionsDataType);
        if (shape == 0) {
            ASSERT_EQ(cloneLoss.get().getDimensions(), batchDimensions);
        } else if (shape == 1) {
            ASSERT_EQ(cloneLoss.get().getDimensions(), elementwiseDimensions);
        } else if (shape == 2) {
            ASSERT_EQ(cloneLoss.get().getDimensions(), perOutputDimensions);
        } else if (shape == 3) {
            ASSERT_EQ(cloneLoss.get().getDimensions(), rawLossDimensions);
        } else {
            assert(false);
        }

        ASSERT_TRUE(clone->getPredictions() == meanAbsoluteError.getFeatureInput());
        ASSERT_TRUE(clone->getLoss() == meanAbsoluteError.getFeatureOutput());

        ASSERT_EQ(meanAbsoluteError.getId(), clone->getId());
        ASSERT_GT(meanAbsoluteError.getId(), 1u);

        ASSERT_TRUE(meanAbsoluteError == *clone);
        ASSERT_FALSE(meanAbsoluteError != *clone);
        ASSERT_FALSE(meanAbsoluteError > *clone);
        ASSERT_FALSE(meanAbsoluteError < *clone);
    }
}

TEST(MeanAbsoluteError, SerializeDeserialize) {
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

    MeanAbsoluteError::Builder meanAbsoluteErrorBuilder = MeanAbsoluteError::Builder()
                                                              .network(initialNetwork)
                                                              .labels(labelsInput.getFeatureOutput())
                                                              .predictions(fullyConnected.getFeatureOutput())
                                                              .lossDataType(lossDataType);

    uint32_t lossShape = rand() % 4;
    if (lossShape == 0)
        meanAbsoluteErrorBuilder.reportsBatchLoss();
    else if (lossShape == 1)
        meanAbsoluteErrorBuilder.reportsPerOutputLoss();
    else if (lossShape == 2)
        meanAbsoluteErrorBuilder.reportsElementwiseLoss();
    else
        meanAbsoluteErrorBuilder.reportsRawLoss();

    MeanAbsoluteError meanAbsoluteError = meanAbsoluteErrorBuilder.build();

    shared_ptr<Sgd> sgd = Sgd::Builder().network(initialNetwork).initialLearningRate(0.1).decay(0.1).build();

    NetworkOutput lossOutput = NetworkOutput::Builder()
                                   .network(initialNetwork)
                                   .name("lossOutput")
                                   .inputTensor(meanAbsoluteError.getLoss())
                                   .dataType(dataType)
                                   .build();

    ASSERT_TRUE(meanAbsoluteError.isInitialized());

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

    Layer *layer = &meanAbsoluteError;
    json meanAbsoluteErrorJ = layer->serialize("/tmp/", stream);
    json labelsInputJ = labelsInput.serialize("/tmp/", stream);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json lossOutputJ = lossOutput.serialize("/tmp/", stream);

    ASSERT_EQ(meanAbsoluteErrorJ["factory"], "loss");
    ASSERT_EQ(meanAbsoluteErrorJ["version"], "1.0.0");
    ASSERT_EQ(meanAbsoluteErrorJ["layer_type"], "mean_absolute_error");
    EXPECT_TRUE(meanAbsoluteErrorJ.contains("layer_name"));
    if (lossShape == 0)
        ASSERT_EQ(meanAbsoluteErrorJ.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::BATCH);
    else if (lossShape == 1)
        ASSERT_EQ(meanAbsoluteErrorJ.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::CLASSWISE);
    else if (lossShape == 2)
        ASSERT_EQ(meanAbsoluteErrorJ.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::ELEMENTWISE);
    else
        ASSERT_EQ(meanAbsoluteErrorJ.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::RAW);
    ASSERT_EQ(meanAbsoluteErrorJ.at("loss_data_type").get<Tensor::DataType>(), lossDataType);

    const json &labelsJ = meanAbsoluteErrorJ["labels_tensor"];
    ASSERT_EQ(labelsJ.at("data_type").get<Tensor::DataType>(), dataType);
    ASSERT_EQ(labelsJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(labelsJ.at("id").is_number_integer());

    const json &predictionsJ = meanAbsoluteErrorJ["predictions_tensor"];
    ASSERT_EQ(predictionsJ.at("data_type").get<Tensor::DataType>(), dataType);
    ASSERT_EQ(predictionsJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(predictionsJ.at("id").is_number_integer());

    const json &lossJ = meanAbsoluteErrorJ["loss_tensor"];
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

    // printf("%s\n", networkInputJ.dump(4).c_str());
    // printf("%s\n", fullyConnectedJ.dump(4).c_str());
    // printf("%s\n", meanAbsoluteErrorJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());

    // ////////////////////////////
    // // Deserialize
    // ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    Layer::deserialize(networkInputJ, &newNetwork);
    Layer::deserialize(labelsInputJ, &newNetwork);
    Layer::deserialize(fullyConnectedJ, &newNetwork);
    Layer::deserialize(meanAbsoluteErrorJ, &newNetwork);
    // FIXME: Find, serialize and deserialize the loss shaper
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
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::MeanAbsoluteError> stampedMeanAbsoluteError =
        dynamic_pointer_cast<ThorImplementation::MeanAbsoluteError>(otherLayers[0]);
    ASSERT_NE(stampedMeanAbsoluteError, nullptr);

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

    ASSERT_EQ(stampedMeanAbsoluteError->getPredictionsInput().get(), stampedFC->getFeatureOutputs()[0].get());
    ASSERT_EQ(stampedMeanAbsoluteError->getLabelsInput().get(), stampedLabelsInput->getFeatureOutput().get());
    ASSERT_EQ(stampedMeanAbsoluteError->getLossOutput().get(), stampedLossOutput->getFeatureInput().get());
}
