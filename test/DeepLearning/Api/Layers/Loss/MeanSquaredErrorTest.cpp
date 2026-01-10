#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(MeanSquaredError, Builds) {
    srand(time(nullptr));

    for (uint32_t i = 0; i < 10; ++i) {
        Network network;

        vector<uint64_t> dimensions;
        dimensions.push_back(1 + (rand() % 1000));
        Tensor::DataType predictionsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor predictions(predictionsDataType, dimensions);

        Tensor::DataType labelsDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor labels(labelsDataType, dimensions);

        bool setLossDataType = rand() % 2 == 0;
        Tensor::DataType lossDataType;

        MeanSquaredError::Builder meanSquaredErrorBuilder =
            MeanSquaredError::Builder().network(network).predictions(predictions).labels(labels);

        uint32_t shape = rand() % 4;
        if (shape == 0) {
            meanSquaredErrorBuilder.reportsBatchLoss();
        } else if (shape == 1) {
            meanSquaredErrorBuilder.reportsElementwiseLoss();
        } else if (shape == 2) {
            meanSquaredErrorBuilder.reportsPerOutputLoss();
        } else if (shape == 3) {
            meanSquaredErrorBuilder.reportsRawLoss();
        } else {
            assert(false);
        }
        vector<uint64_t> batchDimensions = {1};
        vector<uint64_t> elementwiseDimensions = {1};
        vector<uint64_t> perOutputDimensions = {dimensions[0]};
        vector<uint64_t> rawLossDimensions = dimensions;

        if (setLossDataType) {
            lossDataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
            meanSquaredErrorBuilder.lossDataType(lossDataType);
        }

        MeanSquaredError meanSquaredError = meanSquaredErrorBuilder.build();

        ASSERT_TRUE(meanSquaredError.isInitialized());

        Optional<Tensor> actualLabels = meanSquaredError.getLabels();
        ASSERT_TRUE(actualLabels.isPresent());
        ASSERT_EQ(actualLabels.get().getDataType(), labelsDataType);
        ASSERT_EQ(actualLabels.get().getDimensions(), dimensions);

        Optional<Tensor> actualPredictions = meanSquaredError.getPredictions();
        ASSERT_TRUE(actualPredictions.isPresent());
        ASSERT_EQ(actualPredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(actualPredictions.get().getDimensions(), dimensions);

        Optional<Tensor> actualLoss = meanSquaredError.getLoss();
        ASSERT_TRUE(actualLoss.isPresent());
        ASSERT_EQ(actualLoss.get().getDataType(), setLossDataType ? lossDataType : predictionsDataType);
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

        ASSERT_TRUE(meanSquaredError.getPredictions() == meanSquaredError.getFeatureInput());
        ASSERT_TRUE(meanSquaredError.getLoss() == meanSquaredError.getFeatureOutput());

        shared_ptr<Layer> cloneLayer = meanSquaredError.clone();
        MeanSquaredError *clone = dynamic_cast<MeanSquaredError *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(clone->isInitialized());

        Optional<Tensor> clonePredictions = clone->getPredictions();
        ASSERT_TRUE(clonePredictions.isPresent());
        ASSERT_EQ(clonePredictions.get().getDataType(), predictionsDataType);
        ASSERT_EQ(clonePredictions.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLoss = clone->getLoss();
        ASSERT_TRUE(cloneLoss.isPresent());
        ASSERT_EQ(cloneLoss.get().getDataType(), setLossDataType ? lossDataType : predictionsDataType);
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

        ASSERT_TRUE(clone->getPredictions() == meanSquaredError.getFeatureInput());
        ASSERT_TRUE(clone->getLoss() == meanSquaredError.getFeatureOutput());

        ASSERT_EQ(meanSquaredError.getId(), clone->getId());
        ASSERT_GT(meanSquaredError.getId(), 1u);

        ASSERT_TRUE(meanSquaredError == *clone);
        ASSERT_FALSE(meanSquaredError != *clone);
        ASSERT_FALSE(meanSquaredError > *clone);
        ASSERT_FALSE(meanSquaredError < *clone);
    }
}

TEST(MeanSquaredError, SerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;
    Tensor::DataType dataType = Tensor::DataType::FP16;
    vector<uint64_t> inputDimensions = {1UL};
    Tensor::DataType lossDataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;

    NetworkInput labelsInput =
        NetworkInput::Builder().network(initialNetwork).name("labelsInput").dimensions(inputDimensions).dataType(dataType).build();
    NetworkInput predictionsInput =
        NetworkInput::Builder().network(initialNetwork).name("networkInput").dimensions(inputDimensions).dataType(dataType).build();

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(initialNetwork)
                                        .featureInput(predictionsInput.getFeatureOutput())
                                        .numOutputFeatures(1)
                                        .hasBias(false)
                                        .noActivation()
                                        .build();

    MeanSquaredError::Builder meanSquaredErrorBuilder = MeanSquaredError::Builder()
                                                            .network(initialNetwork)
                                                            .labels(labelsInput.getFeatureOutput())
                                                            .predictions(fullyConnected.getFeatureOutput())
                                                            .lossDataType(lossDataType);

    uint32_t lossShape = rand() % 4;
    lossShape = 2;  // FIXME
    if (lossShape == 0)
        meanSquaredErrorBuilder.reportsBatchLoss();
    else if (lossShape == 1)
        meanSquaredErrorBuilder.reportsPerOutputLoss();
    else if (lossShape == 2)
        meanSquaredErrorBuilder.reportsElementwiseLoss();
    else
        meanSquaredErrorBuilder.reportsRawLoss();

    MeanSquaredError meanSquaredError = meanSquaredErrorBuilder.build();

    shared_ptr<Sgd> sgd = Sgd::Builder().network(initialNetwork).initialLearningRate(0.1).decay(0.1).build();

    NetworkOutput lossOutput = NetworkOutput::Builder()
                                   .network(initialNetwork)
                                   .name("lossOutput")
                                   .inputTensor(meanSquaredError.getLoss())
                                   .dataType(dataType)
                                   .build();

    ASSERT_TRUE(meanSquaredError.isInitialized());

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

    Layer *layer = &meanSquaredError;
    json meanSquaredErrorJ = layer->serialize(archiveWriter, stream);
    json labelsInputJ = labelsInput.serialize(archiveWriter, stream);
    json predictionsInputJ = predictionsInput.serialize(archiveWriter, stream);
    json lossOutputJ = lossOutput.serialize(archiveWriter, stream);

    ASSERT_EQ(meanSquaredErrorJ["factory"], "loss");
    ASSERT_EQ(meanSquaredErrorJ["version"], "1.0.0");
    ASSERT_EQ(meanSquaredErrorJ["layer_type"], "mean_squared_error");
    EXPECT_TRUE(meanSquaredErrorJ.contains("layer_name"));
    ASSERT_EQ(meanSquaredErrorJ.at("loss_shape").get<Loss::LossShape>(), Loss::LossShape::RAW);
    ASSERT_EQ(meanSquaredErrorJ.at("loss_data_type").get<Tensor::DataType>(), lossDataType);

    const json &labelsJ = meanSquaredErrorJ["labels_tensor"];
    ASSERT_EQ(labelsJ.at("data_type").get<Tensor::DataType>(), dataType);
    ASSERT_EQ(labelsJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(labelsJ.at("id").is_number_integer());

    const json &predictionsJ = meanSquaredErrorJ["predictions_tensor"];
    ASSERT_EQ(predictionsJ.at("data_type").get<Tensor::DataType>(), dataType);
    ASSERT_EQ(predictionsJ.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(predictionsJ.at("id").is_number_integer());

    const json &lossJ = meanSquaredErrorJ["loss_tensor"];
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
            fullyConnectedJ = initalNetworkFC->serialize(archiveWriter, stream, true);
            fcFound = true;
            break;
        }
    }
    ASSERT_TRUE(fcFound);

    shared_ptr<LossShaper> lossShaper;
    for (uint32_t i = 0; i < initialNetwork.getNumLayers(); ++i) {
        shared_ptr<Layer> layer = initialNetwork.getLayer(i);
        lossShaper = dynamic_pointer_cast<LossShaper>(layer);
        if (lossShaper)
            break;
    }
    ASSERT_EQ(lossShaper == nullptr, lossShape == 3);
    json lossShaperJ;
    if (lossShaper)
        lossShaperJ = lossShaper->serialize(archiveWriter, stream);

    // printf("%s\n", predictionsInputJ.dump(4).c_str());
    // printf("%s\n", labelsInputJ.dump(4).c_str());
    // printf("%s\n", fullyConnectedJ.dump(4).c_str());
    // printf("%s\n", meanSquaredErrorJ.dump(4).c_str());
    // if (lossShaper)
    //     printf("%s\n", lossShaperJ.dump(4).c_str());
    // printf("%s\n", lossOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    archiveWriter.finishArchive();
    thor_file::TarReader archiveReader("testModel", "/tmp/");

    Layer::deserialize(archiveReader, predictionsInputJ, &newNetwork);
    Layer::deserialize(archiveReader, labelsInputJ, &newNetwork);
    Layer::deserialize(archiveReader, fullyConnectedJ, &newNetwork);
    Layer::deserialize(archiveReader, meanSquaredErrorJ, &newNetwork);
    if (lossShaper)
        Layer::deserialize(archiveReader, lossShaperJ, &newNetwork);
    Layer::deserialize(archiveReader, lossOutputJ, &newNetwork);

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
    if (lossShaper)
        ASSERT_EQ(otherLayers.size(), 2U);
    else
        ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::MeanSquaredError> stampedMeanSquaredError =
        dynamic_pointer_cast<ThorImplementation::MeanSquaredError>(otherLayers[0]);
    ASSERT_NE(stampedMeanSquaredError, nullptr);

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

    shared_ptr<ThorImplementation::LossShaper> stampedLossShaper;
    shared_ptr<ThorImplementation::Reshape> stampedReshape;
    if (lossShaper) {
        for (auto layer : stampedNetwork.getOtherLayers()) {
            stampedLossShaper = dynamic_pointer_cast<ThorImplementation::LossShaper>(layer);
            if (stampedLossShaper)
                break;
            stampedReshape = dynamic_pointer_cast<ThorImplementation::Reshape>(layer);
            if (stampedReshape)
                break;
        }
        ASSERT_EQ(stampedLossShaper == nullptr, lossShape == 2);
        ASSERT_EQ(stampedReshape == nullptr, lossShape != 2);
    }

    ASSERT_EQ(stampedMeanSquaredError->getPredictionsInput().get(), stampedFC->getFeatureOutputs()[0].get());
    ASSERT_EQ(stampedMeanSquaredError->getLabelsInput().get(), stampedLabelsInput->getFeatureOutput().get());
    if (stampedLossShaper) {
        ASSERT_EQ(stampedMeanSquaredError->getLossOutput().get(), stampedLossShaper->getFeatureInput().get());
        ASSERT_EQ(stampedLossShaper->getFeatureOutput().get(), stampedLossOutput->getFeatureInput().get());
    } else if (stampedReshape) {
        ASSERT_EQ(stampedMeanSquaredError->getLossOutput().get(), stampedReshape->getFeatureInput().get());
        ASSERT_EQ(stampedReshape->getFeatureOutput().get(), stampedLossOutput->getFeatureInput().get());
    } else {
        ASSERT_EQ(stampedMeanSquaredError->getLossOutput().get(), stampedLossOutput->getFeatureInput().get());
    }

    filesystem::remove("/tmp/testModel.thor");
}
