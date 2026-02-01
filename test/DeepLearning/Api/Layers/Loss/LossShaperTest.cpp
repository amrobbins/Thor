#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(LossShaper, Builds) {
    srand(time(nullptr));

    for (uint32_t i = 0; i < 10; ++i) {
        Network network("testNetwork");

        vector<uint64_t> dimensions;
        dimensions.push_back(2 + (rand() % 1000));
        Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
        Tensor lossInput(dataType, dimensions);

        LossShaper::Builder lossShaperBuilder = LossShaper::Builder().network(network).lossInput(lossInput);

        uint32_t shape = rand() % 3;
        ThorImplementation::LossShaper::OutputLossType outputLossType;
        if (shape == 0) {
            outputLossType = ThorImplementation::LossShaper::OutputLossType::BATCH;
            lossShaperBuilder.reportsBatchLoss();
        } else if (shape == 1) {
            outputLossType = ThorImplementation::LossShaper::OutputLossType::CLASSWISE;
            lossShaperBuilder.reportsClasswiseLoss();
        } else if (shape == 2) {
            outputLossType = ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE;
            lossShaperBuilder.reportsElementwiseLoss();
        } else {
            assert(false);
        }
        vector<uint64_t> batchDimensions = {1};
        vector<uint64_t> classwiseDimensions = {dimensions[0]};
        vector<uint64_t> elementwiseDimensions = {1};

        LossShaper lossShaper = lossShaperBuilder.build();

        ASSERT_TRUE(lossShaper.isInitialized());

        Optional<Tensor> actualLossInput = lossShaper.getLossInput();
        ASSERT_TRUE(actualLossInput.isPresent());
        ASSERT_EQ(actualLossInput.get().getDataType(), dataType);
        ASSERT_EQ(actualLossInput.get().getDimensions(), dimensions);

        Optional<Tensor> actualLossOutput = lossShaper.getLossOutput();
        ASSERT_TRUE(actualLossOutput.isPresent());
        ASSERT_EQ(actualLossOutput.get().getDataType(), dataType);
        if (outputLossType == ThorImplementation::LossShaper::OutputLossType::BATCH) {
            ASSERT_EQ(actualLossOutput.get().getDimensions(), batchDimensions);
        } else if (outputLossType == ThorImplementation::LossShaper::OutputLossType::CLASSWISE) {
            ASSERT_EQ(actualLossOutput.get().getDimensions(), classwiseDimensions);
        } else {
            assert(outputLossType == ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE);
            ASSERT_EQ(actualLossOutput.get().getDimensions(), elementwiseDimensions);
        }

        ASSERT_FALSE(actualLossInput.get() == actualLossOutput.get());

        ASSERT_TRUE(lossShaper.getLossInput() == lossShaper.getFeatureInput());
        ASSERT_TRUE(lossShaper.getLossOutput() == lossShaper.getFeatureOutput());

        shared_ptr<Layer> cloneLayer = lossShaper.clone();
        LossShaper *clone = dynamic_cast<LossShaper *>(cloneLayer.get());
        assert(clone != nullptr);

        ASSERT_TRUE(lossShaper.isInitialized());

        Optional<Tensor> cloneLossInput = clone->getLossInput();
        ASSERT_TRUE(cloneLossInput.isPresent());
        ASSERT_EQ(cloneLossInput.get().getDataType(), dataType);
        ASSERT_EQ(cloneLossInput.get().getDimensions(), dimensions);

        Optional<Tensor> cloneLossOutput = clone->getLossOutput();
        ASSERT_TRUE(cloneLossOutput.isPresent());
        ASSERT_EQ(cloneLossOutput.get().getDataType(), dataType);
        if (outputLossType == ThorImplementation::LossShaper::OutputLossType::BATCH) {
            ASSERT_EQ(cloneLossOutput.get().getDimensions(), batchDimensions);
        } else if (outputLossType == ThorImplementation::LossShaper::OutputLossType::CLASSWISE) {
            ASSERT_EQ(cloneLossOutput.get().getDimensions(), classwiseDimensions);
        } else if (outputLossType == ThorImplementation::LossShaper::OutputLossType::ELEMENTWISE) {
            ASSERT_EQ(cloneLossOutput.get().getDimensions(), elementwiseDimensions);
        }

        ASSERT_FALSE(cloneLossInput.get() == cloneLossOutput.get());

        ASSERT_TRUE(clone->getLossInput() == clone->getFeatureInput());
        ASSERT_TRUE(clone->getLossOutput() == clone->getFeatureOutput());

        ASSERT_EQ(lossShaper.getId(), clone->getId());
        ASSERT_GT(lossShaper.getId(), 1u);

        ASSERT_TRUE(lossShaper == *clone);
        ASSERT_FALSE(lossShaper != *clone);
        ASSERT_FALSE(lossShaper > *clone);
        ASSERT_FALSE(lossShaper < *clone);
    }
}

TEST(LossShaper, SerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP16 : Tensor::DataType::FP32;
    uint64_t numClasses = (rand() % 100) + 1;
    vector<uint64_t> inputDimensions = {numClasses};

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    LossShaper::Builder lossShaperBuilder = LossShaper::Builder().network(initialNetwork).lossInput(networkInput.getFeatureOutput());

    vector<uint64_t> outputDimensions;
    uint32_t outputLossType = rand() % 3;
    if (outputLossType == 0) {
        lossShaperBuilder.reportsBatchLoss();
        outputDimensions = {1UL};
    } else if (outputLossType == 1) {
        lossShaperBuilder.reportsClasswiseLoss();
        outputDimensions = inputDimensions;
    } else {
        lossShaperBuilder.reportsElementwiseLoss();
        outputDimensions = {1UL};
    }
    LossShaper lossShaper = lossShaperBuilder.build();

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(lossShaper.getFeatureOutput())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(lossShaper.isInitialized());

    Tensor featureInput = lossShaper.getFeatureInput();
    Tensor featureOutput = lossShaper.getFeatureOutput();
    assert(featureInput == networkInput.getFeatureOutput());

    ASSERT_TRUE(lossShaper.getFeatureOutput().isPresent());
    ASSERT_EQ(lossShaper.getFeatureOutput().get(), featureOutput);

    ASSERT_TRUE(lossShaper.getFeatureInput().isPresent());
    assert(lossShaper.getFeatureInput().get() == featureInput);

    ASSERT_EQ(featureInput.getDataType(), dataType);
    ASSERT_EQ(featureInput.getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutput.getDataType(), dataType);
    ASSERT_EQ(featureOutput.getDimensions(), outputDimensions);

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

    thor_file::TarWriter archiveWriter("testModel");

    json lossShaperJ = lossShaper.serialize(archiveWriter, stream);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = &lossShaper;
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(lossShaperJ, fromLayerJ);

    // printf("%s\n", networkInputJ.dump(4).c_str());
    // printf("%s\n", lossShaperJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());

    ASSERT_EQ(lossShaperJ["factory"], "loss");
    ASSERT_EQ(lossShaperJ["version"], "1.0.0");
    ASSERT_EQ(lossShaperJ["layer_type"], "loss_shaper");

    ASSERT_EQ(lossShaperJ.at("loss_shape").get<Loss::LossShape>() == Loss::LossShape::BATCH, outputLossType == 0);
    ASSERT_EQ(lossShaperJ.at("loss_shape").get<Loss::LossShape>() == Loss::LossShape::CLASSWISE, outputLossType == 1);
    ASSERT_EQ(lossShaperJ.at("loss_shape").get<Loss::LossShape>() == Loss::LossShape::ELEMENTWISE, outputLossType == 2);

    EXPECT_TRUE(lossShaperJ.contains("loss_input"));
    EXPECT_TRUE(lossShaperJ.contains("loss_output"));
    EXPECT_TRUE(lossShaperJ.contains("loss_shape"));

    const json &lossInput = lossShaperJ.at("loss_input");
    ASSERT_TRUE(lossInput.is_object());
    ASSERT_TRUE(lossInput.at("data_type").is_string());
    string dataTypeString = dataType == Tensor::DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(lossInput.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(lossInput.at("dimensions").is_array());
    ASSERT_EQ(lossInput.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(lossInput.at("id").is_number_integer());

    const auto &lossOutput = lossShaperJ.at("loss_output");
    ASSERT_TRUE(lossOutput.is_object());
    ASSERT_TRUE(lossOutput.at("data_type").is_string());
    EXPECT_EQ(lossOutput.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(lossOutput.at("dimensions").is_array());
    ASSERT_EQ(lossOutput.at("dimensions").get<vector<uint64_t>>(), outputDimensions);
    ASSERT_TRUE(lossOutput.at("id").is_number_integer());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork("newNetwork");

    // Write a dummy file with data into the archive since none of the layers wrote anything into it (no weights)
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {4});
    ThorImplementation::Tensor dummyData(cpuPlacement, descriptor);
    archiveWriter.addArchiveFile("dummy", dummyData);

    archiveWriter.createArchive("/tmp/", true);
    shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");

    Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
    Layer::deserialize(archiveReader, lossShaperJ, &newNetwork);
    Layer::deserialize(archiveReader, networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    placementStatus = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(placementStatus, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &newStamp = newNetwork.getStampedNetwork(0);

    vector<shared_ptr<ThorImplementation::Layer>> otherLayers = newStamp.getOtherLayers();
    ASSERT_EQ(otherLayers.size(), 1U);
    shared_ptr<ThorImplementation::LossShaper> stampedLossShaper;
    stampedLossShaper = dynamic_pointer_cast<ThorImplementation::LossShaper>(otherLayers[0]);
    ASSERT_NE(stampedLossShaper, nullptr);

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = newStamp.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = newStamp.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);

    ASSERT_TRUE(stampedInput->getFeatureOutput().isPresent());
    ASSERT_TRUE(stampedLossShaper->getFeatureOutput().isPresent());
    ASSERT_TRUE(stampedOutput->getFeatureOutput().isPresent());
    ASSERT_EQ(stampedInput->getFeatureOutput().get(), stampedLossShaper->getFeatureInput().get());
    ASSERT_EQ(stampedLossShaper->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());

    ASSERT_EQ(stampedInput->getFeatureOutput().get(), stampedLossShaper->getFeatureInput().get());
    ASSERT_EQ(stampedLossShaper->getFeatureOutput().get(), stampedOutput->getFeatureInput().get());

    filesystem::remove("/tmp/testModel.thor.tar");
}
