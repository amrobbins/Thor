#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Network/Network.h"

#include "gtest/gtest.h"

#include <stdio.h>
#include <memory>

#include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"

using namespace Thor;
using namespace std;
using json = nlohmann::json;

TEST(UtilityApiLayers, BatchNormalizationSingleFeatureInputBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;

    Tensor featureInput(dataType, dimensions);

    double exponentialRunningAverageFactor = (1 + (rand() % 100)) / 1000.0f;

    double epsilon = (1 + (rand() % 100)) / 100000.0f;

    BatchNormalization batchNormalization = BatchNormalization::Builder()
                                                .network(network)
                                                .featureInput(featureInput)
                                                .exponentialRunningAverageFactor(exponentialRunningAverageFactor)
                                                .epsilon(epsilon)
                                                .build();

    ASSERT_TRUE(batchNormalization.isInitialized());

    Optional<Tensor> actualInput = batchNormalization.getFeatureInput();
    ASSERT_TRUE(actualInput.isPresent());
    ASSERT_EQ(actualInput.get().getDataType(), dataType);
    ASSERT_EQ(actualInput.get().getDimensions(), dimensions);

    Optional<Tensor> actualOutput = batchNormalization.getFeatureOutput();
    ASSERT_TRUE(actualOutput.isPresent());
    ASSERT_EQ(actualOutput.get().getDataType(), dataType);
    ASSERT_EQ(actualOutput.get().getDimensions(), dimensions);

    double actualExponentialRunningAverageFactor = batchNormalization.getExponentialRunningAverageFactor();
    ASSERT_EQ(actualExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double actualEpsilon = batchNormalization.getEpsilon();
    ASSERT_EQ(actualEpsilon, epsilon);

    shared_ptr<Layer> cloneLayer = batchNormalization.clone();
    BatchNormalization *clone = dynamic_cast<BatchNormalization *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    Optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.isPresent());
    ASSERT_EQ(cloneInput.get().getDataType(), dataType);
    ASSERT_EQ(cloneInput.get().getDimensions(), dimensions);

    Optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.isPresent());
    ASSERT_EQ(cloneOutput.get().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.get().getDimensions(), dimensions);

    double cloneExponentialRunningAverageFactor = clone->getExponentialRunningAverageFactor();
    ASSERT_EQ(cloneExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double cloneEpsilon = clone->getEpsilon();
    ASSERT_EQ(cloneEpsilon, epsilon);

    ASSERT_EQ(batchNormalization.getId(), clone->getId());
    ASSERT_GT(batchNormalization.getId(), 1u);

    ASSERT_TRUE(batchNormalization == *clone);
    ASSERT_FALSE(batchNormalization != *clone);
    ASSERT_FALSE(batchNormalization > *clone);
    ASSERT_FALSE(batchNormalization < *clone);
}

TEST(UtilityApiLayers, BatchNormalizationMultipleFeatureInputsBuilds) {
    srand(time(nullptr));

    Network network;

    vector<uint64_t> dimensions;
    int numDimensions0 = 1 + rand() % 6;
    for (int i = 0; i < numDimensions0; ++i)
        dimensions.push_back(1 + (rand() % 1000));
    Tensor::DataType dataType = rand() % 2 ? Tensor::DataType::FP32 : Tensor::DataType::FP16;
    Tensor featureInput0(dataType, dimensions);
    Tensor featureInput1(dataType, dimensions);

    double exponentialRunningAverageFactor = (1 + (rand() % 100)) / 1000.0f;

    double epsilon = (1 + (rand() % 100)) / 100000.0f;

    BatchNormalization batchNormalization = BatchNormalization::Builder()
                                                .network(network)
                                                .featureInput(featureInput0)
                                                .featureInput(featureInput1)
                                                .exponentialRunningAverageFactor(exponentialRunningAverageFactor)
                                                .epsilon(epsilon)
                                                .build();

    ASSERT_TRUE(batchNormalization.isInitialized());

    vector<Tensor> featureInputs = batchNormalization.getFeatureInputs();
    vector<Tensor> featureOutputs = batchNormalization.getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(batchNormalization.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[1].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[1].getDimensions(), dimensions);

    double actualExponentialRunningAverageFactor = batchNormalization.getExponentialRunningAverageFactor();
    ASSERT_EQ(actualExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double actualEpsilon = batchNormalization.getEpsilon();
    ASSERT_EQ(actualEpsilon, epsilon);

    shared_ptr<Layer> cloneLayer = batchNormalization.clone();
    BatchNormalization *clone = dynamic_cast<BatchNormalization *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    featureInputs.clear();
    featureOutputs.clear();
    featureInputs = clone->getFeatureInputs();
    featureOutputs = clone->getFeatureOutputs();
    assert(featureInputs[0] == featureInput0);
    assert(featureInputs[1] == featureInput1);

    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput0), featureOutputs[0]);
    ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput1), featureOutputs[1]);
    ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());

    assert(batchNormalization.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
    assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureInputs[1].getDataType(), dataType);
    ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), dimensions);

    ASSERT_EQ(featureOutputs[1].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[1].getDimensions(), dimensions);

    double cloneExponentialRunningAverageFactor = clone->getExponentialRunningAverageFactor();
    ASSERT_EQ(cloneExponentialRunningAverageFactor, exponentialRunningAverageFactor);

    double cloneEpsilon = clone->getEpsilon();
    ASSERT_EQ(cloneEpsilon, epsilon);

    ASSERT_EQ(batchNormalization.getId(), clone->getId());
    ASSERT_GT(batchNormalization.getId(), 1u);

    ASSERT_TRUE(batchNormalization == *clone);
    ASSERT_FALSE(batchNormalization != *clone);
    ASSERT_FALSE(batchNormalization > *clone);
    ASSERT_FALSE(batchNormalization < *clone);
}

TEST(UtilityApiLayers, BatchNormalizationSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork;

    Tensor::DataType dataType = Tensor::DataType::FP16;
    string dataTypeString = dataType == Tensor::DataType::FP32 ? "fp32" : "fp16";

    vector<uint64_t> inputDimensions = {1UL + (rand() % 16)};

    float exponential_running_average_factor = (rand() % 1000) / 1000.0f;
    float epsilon = (rand() % 1000) / 1000.0f;

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    BatchNormalization::Builder batchNormalizationBuilder = BatchNormalization::Builder()
                                                                .network(initialNetwork)
                                                                .featureInput(networkInput.getFeatureOutput())
                                                                .exponentialRunningAverageFactor(exponential_running_average_factor)
                                                                .epsilon(epsilon);
    BatchNormalization batchNormalization = batchNormalizationBuilder.build();

    Tensor logits = batchNormalization.getFeatureOutputs()[0];
    uint32_t numClasses = logits.getDimensions()[0];
    NetworkInput labelsInput =
        NetworkInput::Builder().network(initialNetwork).name("labelsInput").dimensions({numClasses}).dataType(dataType).build();

    MeanAbsoluteError meanAbsoluteError = MeanAbsoluteError::Builder()
                                              .network(initialNetwork)
                                              .predictions(logits)
                                              .reportsRawLoss()
                                              .lossDataType(dataType)
                                              .labels(labelsInput.getFeatureOutput())
                                              .build();

    shared_ptr<Sgd> sgd = Sgd::Builder().network(initialNetwork).initialLearningRate(0.1).decay(0.1).build();
    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(meanAbsoluteError.getLoss())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(batchNormalization.isInitialized());

    vector<Tensor> featureInputs = batchNormalization.getFeatureInputs();
    vector<Tensor> featureOutputs = batchNormalization.getFeatureOutputs();
    assert(featureInputs[0] == networkInput.getFeatureOutput());

    ASSERT_EQ(batchNormalization.getFeatureOutput(networkInput.getFeatureOutput()), featureOutputs[0]);

    assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);

    ASSERT_EQ(featureInputs[0].getDataType(), dataType);
    ASSERT_EQ(featureInputs[0].getDimensions(), inputDimensions);

    // At the moment BatchNormalization data type is forced to fp16
    ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
    ASSERT_EQ(featureOutputs[0].getDimensions(), inputDimensions);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    Network::StatusCode statusCode;
    statusCode = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the physical batch norm layer from the stamped network and write to its weights
    ASSERT_EQ(initialNetwork.getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialNetwork.getStampedNetwork(0);
    ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNormLayer =
        dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getTrainableLayer(0));
    ASSERT_TRUE(physicalBatchNormLayer != nullptr);
    ThorImplementation::Tensor weights = physicalBatchNormLayer->getWeights();
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor weightsCpu = weights.clone(cpuPlacement);
    half *weightsCpuMem = (half *)weightsCpu.getMemPtr();
    for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
        weightsCpuMem[i] = i;
    }
    weights.copyFromAsync(weightsCpu, stream);

    ThorImplementation::Tensor biases = physicalBatchNormLayer->getBiases();
    ThorImplementation::Tensor biasesCpu = biases.clone(cpuPlacement);
    half *biasesCpuMem = (half *)biasesCpu.getMemPtr();
    for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
        biasesCpuMem[i] = i * i + 6;
    }
    biases.copyFromAsync(biasesCpu, stream);

    ThorImplementation::Tensor means = physicalBatchNormLayer->getResultRunningMean();
    ThorImplementation::Tensor meansCpu = means.clone(cpuPlacement);
    half *meansCpuMem = (half *)meansCpu.getMemPtr();
    for (uint32_t i = 0; i < means.getTotalNumElements(); ++i) {
        meansCpuMem[i] = i * i + 6;
    }
    means.copyFromAsync(meansCpu, stream);

    ThorImplementation::Tensor variances = physicalBatchNormLayer->getResultRunningVariance();
    ThorImplementation::Tensor variancesCpu = variances.clone(cpuPlacement);
    half *variancesCpuMem = (half *)variancesCpu.getMemPtr();
    for (uint32_t i = 0; i < variances.getTotalNumElements(); ++i) {
        variancesCpuMem[i] = i * i + 6;
    }
    variances.copyFromAsync(variancesCpu, stream);

    json meanAbsoluteErrorJ = meanAbsoluteError.serialize("/tmp/", stream);
    json networkInputJ = networkInput.serialize("/tmp/", stream);
    json labelsInputJ = labelsInput.serialize("/tmp/", stream);
    json networkOutputJ = networkOutput.serialize("/tmp/", stream);

    // The network attached the optimizer to its copy of the BN layer
    json batchNormalizationJ;
    bool bnFound = false;
    shared_ptr<BatchNormalization> initalNetworkBN;
    for (int32_t i = 0; i < initialNetwork.getNumTrainableLayers(); ++i) {
        shared_ptr<TrainableWeightsBiasesLayer> layer = initialNetwork.getTrainableLayer(i);
        initalNetworkBN = dynamic_pointer_cast<BatchNormalization>(layer);
        if (initalNetworkBN) {
            batchNormalizationJ = initalNetworkBN->serialize("/tmp", stream);
            bnFound = true;
            break;
        }
    }
    ASSERT_TRUE(bnFound);

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    shared_ptr<Layer> layer = initalNetworkBN;
    json fromLayerJ = layer->serialize("/tmp/", stream);
    ASSERT_EQ(batchNormalizationJ, fromLayerJ);

    ASSERT_EQ(batchNormalizationJ["version"], "1.0.0");
    ASSERT_EQ(batchNormalizationJ["layer_type"], "batch_normalization");

    EXPECT_TRUE(batchNormalizationJ.contains("inputs"));
    EXPECT_TRUE(batchNormalizationJ.contains("outputs"));
    EXPECT_TRUE(batchNormalizationJ.contains("exponential_running_average_factor"));
    EXPECT_TRUE(batchNormalizationJ.contains("epsilon"));

    ASSERT_TRUE(batchNormalizationJ.at("inputs").is_array());
    ASSERT_TRUE(batchNormalizationJ.at("outputs").is_array());
    ASSERT_TRUE(batchNormalizationJ.at("exponential_running_average_factor").is_number_float());
    ASSERT_TRUE(batchNormalizationJ.at("epsilon").is_number_float());

    EXPECT_EQ(batchNormalizationJ.at("exponential_running_average_factor").get<float>(), exponential_running_average_factor);
    EXPECT_EQ(batchNormalizationJ.at("epsilon").get<float>(), epsilon);

    const auto &inputs = batchNormalizationJ.at("inputs");
    ASSERT_EQ(inputs.size(), 1U) << "Expect exactly one input";
    const auto &in0 = inputs.at(0);
    ASSERT_TRUE(in0.is_object());
    ASSERT_TRUE(in0.at("data_type").is_string());
    EXPECT_EQ(in0.at("data_type").get<string>(), dataTypeString);

    ASSERT_TRUE(in0.at("dimensions").is_array());
    ASSERT_EQ(in0.at("dimensions").size(), 1U);
    EXPECT_TRUE(in0.at("dimensions").at(0).is_number_integer());
    EXPECT_EQ(in0.at("dimensions").at(0).get<uint32_t>(), inputDimensions[0]);

    ASSERT_TRUE(in0.at("id").is_number_integer());

    const auto &outputs = batchNormalizationJ.at("outputs");
    ASSERT_EQ(outputs.size(), 1U) << "Expect exactly one output";
    const auto &out0 = outputs.at(0);
    ASSERT_TRUE(out0.is_object());
    ASSERT_TRUE(out0.at("data_type").is_string());
    EXPECT_EQ(out0.at("data_type").get<string>(), dataType == Tensor::DataType::FP16 ? "fp16" : "fp32");

    ASSERT_TRUE(out0.at("dimensions").is_array());
    ASSERT_EQ(out0.at("dimensions").size(), inputDimensions.size());
    EXPECT_EQ(out0.at("dimensions").get<vector<uint64_t>>(), inputDimensions);

    ASSERT_TRUE(out0.at("id").is_number_integer());

    EXPECT_FALSE(batchNormalizationJ.at("weights_tensor").get<string>().empty());
    EXPECT_FALSE(batchNormalizationJ.at("biases_tensor").get<string>().empty());
    EXPECT_FALSE(batchNormalizationJ.at("means_tensor").get<string>().empty());
    EXPECT_FALSE(batchNormalizationJ.at("variances_tensor").get<string>().empty());

    string file_prefix = "/tmp/layer" + to_string(batchNormalization.getId());
    EXPECT_EQ(batchNormalizationJ.at("weights_tensor").get<string>(), file_prefix + "_weights.gds");
    EXPECT_EQ(batchNormalizationJ.at("biases_tensor").get<string>(), file_prefix + "_biases.gds");
    EXPECT_EQ(batchNormalizationJ.at("means_tensor").get<string>(), file_prefix + "_means.gds");
    EXPECT_EQ(batchNormalizationJ.at("variances_tensor").get<string>(), file_prefix + "_variances.gds");

    // printf("%s\n", networkInputJ.dump(4).c_str());
    // printf("%s\n", batchNormalizationJ.dump(4).c_str());
    // printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork;

    Layer::deserialize(networkInputJ, &newNetwork);
    Layer::deserialize(labelsInputJ, &newNetwork);
    Layer::deserialize(batchNormalizationJ, &newNetwork);
    Layer::deserialize(meanAbsoluteErrorJ, &newNetwork);
    Layer::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    statusCode = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newNetwork.getNumStamps(), 1UL);
    stampedNetwork = newNetwork.getStampedNetwork(0);
    ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNormLayerDes =
        dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getTrainableLayer(0));
    ASSERT_TRUE(physicalBatchNormLayerDes != nullptr);

    ThorImplementation::Tensor weightsDes = physicalBatchNormLayerDes->getWeights();
    ThorImplementation::Tensor weightsCpuDes = weightsDes.clone(cpuPlacement);
    weightsCpuDes.copyFromAsync(weightsDes, stream);

    ThorImplementation::Tensor biasesDes = physicalBatchNormLayerDes->getBiases();
    ThorImplementation::Tensor biasesCpuDes = biasesDes.clone(cpuPlacement);
    biasesCpuDes.copyFromAsync(biasesDes, stream);

    ThorImplementation::Tensor meansDes = physicalBatchNormLayerDes->getResultRunningMean();
    ThorImplementation::Tensor meansCpuDes = meansDes.clone(cpuPlacement);
    meansCpuDes.copyFromAsync(meansDes, stream);

    ThorImplementation::Tensor variancesDes = physicalBatchNormLayerDes->getResultRunningVariance();
    ThorImplementation::Tensor variancesCpuDes = variancesDes.clone(cpuPlacement);
    variancesCpuDes.copyFromAsync(variancesDes, stream);

    stream.synchronize();

    ASSERT_NE(weightsDes, weights);
    ASSERT_EQ(weightsDes.getDimensions(), weights.getDimensions());
    ASSERT_EQ(weightsDes.getDataType(), weights.getDataType());
    ASSERT_TRUE(weightsDes.getPlacement() == weights.getPlacement());

    half *weightsCpuMemDes = (half *)weightsCpuDes.getMemPtr();
    for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
        ASSERT_EQ(float(weightsCpuMemDes[i]), float(half(i)));
    }

    ASSERT_NE(biasesDes, biases);
    ASSERT_EQ(biasesDes.getDimensions(), biases.getDimensions());
    ASSERT_EQ(biasesDes.getDataType(), biases.getDataType());
    ASSERT_TRUE(biasesDes.getPlacement() == biases.getPlacement());

    half *biasesCpuMemDes = (half *)biasesCpuDes.getMemPtr();
    for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
        ASSERT_EQ(biasesCpuMemDes[i], half(i * i + 6));
    }

    ASSERT_NE(meansDes, means);
    ASSERT_EQ(meansDes.getDimensions(), means.getDimensions());
    ASSERT_EQ(meansDes.getDataType(), means.getDataType());
    ASSERT_TRUE(meansDes.getPlacement() == means.getPlacement());

    half *meansCpuMemDes = (half *)meansCpuDes.getMemPtr();
    for (uint32_t i = 0; i < means.getTotalNumElements(); ++i) {
        ASSERT_EQ(meansCpuMemDes[i], half(i * i + 6));
    }

    ASSERT_NE(variancesDes, variances);
    ASSERT_EQ(variancesDes.getDimensions(), variances.getDimensions());
    ASSERT_EQ(variancesDes.getDataType(), variances.getDataType());
    ASSERT_TRUE(variancesDes.getPlacement() == variances.getPlacement());

    half *variancesCpuMemDes = (half *)variancesCpuDes.getMemPtr();
    for (uint32_t i = 0; i < variances.getTotalNumElements(); ++i) {
        ASSERT_EQ(variancesCpuMemDes[i], half(i * i + 6));
    }
}
