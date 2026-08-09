// #include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
// #include "DeepLearning/Api/Network/Network.h"
// #include "DeepLearning/Api/Network/PlacedNetwork.h"
//
// #include "gtest/gtest.h"
//
// #include <stdio.h>
// #include <memory>
//
// #include "DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h"
// #include "DeepLearning/Api/Optimizers/Sgd.h"
//
// using namespace Thor;
// using namespace std;
// using json = nlohmann::json;
//
// TEST(UtilityApiLayers, BatchNormalizationSingleFeatureInputBuilds) {
//     srand(time(nullptr));
//
//     Network network("testNetwork");
//
//     vector<uint64_t> dimensions;
//     int numDimensions = 1 + rand() % 6;
//     for (int i = 0; i < numDimensions; ++i)
//         dimensions.push_back(1 + (rand() % 1000));
//
//     DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//
//     Tensor featureInput(dataType, dimensions);
//
//     double exponentialRunningAverageFactor = (1 + (rand() % 100)) / 1000.0f;
//
//     double epsilon = (1 + (rand() % 100)) / 100000.0f;
//
//     BatchNormalization batchNormalization = BatchNormalization::Builder()
//                                                 .network(network)
//                                                 .featureInput(featureInput)
//                                                 .exponentialRunningAverageFactor(exponentialRunningAverageFactor)
//                                                 .epsilon(epsilon)
//                                                 .build();
//
//     ASSERT_TRUE(batchNormalization.isInitialized());
//
//     std::optional<Tensor> actualInput = batchNormalization.getFeatureInput();
//     ASSERT_TRUE(actualInput.has_value());
//     ASSERT_EQ(actualInput.value().getDataType(), dataType);
//     ASSERT_EQ(actualInput.value().getDimensions(), dimensions);
//
//     std::optional<Tensor> actualOutput = batchNormalization.getFeatureOutput();
//     ASSERT_TRUE(actualOutput.has_value());
//     ASSERT_EQ(actualOutput.value().getDataType(), dataType);
//     ASSERT_EQ(actualOutput.value().getDimensions(), dimensions);
//
//     double actualExponentialRunningAverageFactor = batchNormalization.getExponentialRunningAverageFactor();
//     ASSERT_EQ(actualExponentialRunningAverageFactor, exponentialRunningAverageFactor);
//
//     double actualEpsilon = batchNormalization.getEpsilon();
//     ASSERT_EQ(actualEpsilon, epsilon);
//
//     shared_ptr<Layer> cloneLayer = batchNormalization.clone();
//     BatchNormalization *clone = dynamic_cast<BatchNormalization *>(cloneLayer.get());
//     assert(clone != nullptr);
//
//     ASSERT_TRUE(clone->isInitialized());
//
//     std::optional<Tensor> cloneInput = clone->getFeatureInput();
//     ASSERT_TRUE(cloneInput.has_value());
//     ASSERT_EQ(cloneInput.value().getDataType(), dataType);
//     ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);
//
//     std::optional<Tensor> cloneOutput = clone->getFeatureOutput();
//     ASSERT_TRUE(cloneOutput.has_value());
//     ASSERT_EQ(cloneOutput.value().getDataType(), dataType);
//     ASSERT_EQ(cloneOutput.value().getDimensions(), dimensions);
//
//     double cloneExponentialRunningAverageFactor = clone->getExponentialRunningAverageFactor();
//     ASSERT_EQ(cloneExponentialRunningAverageFactor, exponentialRunningAverageFactor);
//
//     double cloneEpsilon = clone->getEpsilon();
//     ASSERT_EQ(cloneEpsilon, epsilon);
//
//     ASSERT_EQ(batchNormalization.getId(), clone->getId());
//     ASSERT_GT(batchNormalization.getId(), 1u);
//
//     ASSERT_TRUE(batchNormalization == *clone);
//     ASSERT_FALSE(batchNormalization != *clone);
//     ASSERT_FALSE(batchNormalization > *clone);
//     ASSERT_FALSE(batchNormalization < *clone);
// }
//
// TEST(UtilityApiLayers, BatchNormalizationMultipleFeatureInputsBuilds) {
//     srand(time(nullptr));
//
//     Network network("testNetwork");
//
//     vector<uint64_t> dimensions;
//     int numDimensions0 = 1 + rand() % 6;
//     for (int i = 0; i < numDimensions0; ++i)
//         dimensions.push_back(1 + (rand() % 1000));
//     DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;
//     Tensor featureInput0(dataType, dimensions);
//     Tensor featureInput1(dataType, dimensions);
//
//     double exponentialRunningAverageFactor = (1 + (rand() % 100)) / 1000.0f;
//
//     double epsilon = (1 + (rand() % 100)) / 100000.0f;
//
//     BatchNormalization batchNormalization = BatchNormalization::Builder()
//                                                 .network(network)
//                                                 .featureInput(featureInput0)
//                                                 .featureInput(featureInput1)
//                                                 .exponentialRunningAverageFactor(exponentialRunningAverageFactor)
//                                                 .epsilon(epsilon)
//                                                 .build();
//
//     ASSERT_TRUE(batchNormalization.isInitialized());
//
//     vector<Tensor> featureInputs = batchNormalization.getFeatureInputs();
//     vector<Tensor> featureOutputs = batchNormalization.getFeatureOutputs();
//     assert(featureInputs[0] == featureInput0);
//     assert(featureInputs[1] == featureInput1);
//
//     ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput0), featureOutputs[0]);
//     ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput1), featureOutputs[1]);
//     ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());
//
//     assert(batchNormalization.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
//     assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);
//
//     ASSERT_EQ(featureInputs[0].getDataType(), dataType);
//     ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);
//
//     ASSERT_EQ(featureInputs[1].getDataType(), dataType);
//     ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);
//
//     ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
//     ASSERT_EQ(featureOutputs[0].getDimensions(), dimensions);
//
//     ASSERT_EQ(featureOutputs[1].getDataType(), dataType);
//     ASSERT_EQ(featureOutputs[1].getDimensions(), dimensions);
//
//     double actualExponentialRunningAverageFactor = batchNormalization.getExponentialRunningAverageFactor();
//     ASSERT_EQ(actualExponentialRunningAverageFactor, exponentialRunningAverageFactor);
//
//     double actualEpsilon = batchNormalization.getEpsilon();
//     ASSERT_EQ(actualEpsilon, epsilon);
//
//     shared_ptr<Layer> cloneLayer = batchNormalization.clone();
//     BatchNormalization *clone = dynamic_cast<BatchNormalization *>(cloneLayer.get());
//     assert(clone != nullptr);
//
//     ASSERT_TRUE(clone->isInitialized());
//
//     featureInputs.clear();
//     featureOutputs.clear();
//     featureInputs = clone->getFeatureInputs();
//     featureOutputs = clone->getFeatureOutputs();
//     assert(featureInputs[0] == featureInput0);
//     assert(featureInputs[1] == featureInput1);
//
//     ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput0), featureOutputs[0]);
//     ASSERT_EQ(batchNormalization.getFeatureOutput(featureInput1), featureOutputs[1]);
//     ASSERT_NE(featureOutputs[0].getId(), featureOutputs[1].getId());
//
//     assert(batchNormalization.getFeatureInput(featureOutputs[1]) == featureInputs[1]);
//     assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);
//
//     ASSERT_EQ(featureInputs[0].getDataType(), dataType);
//     ASSERT_EQ(featureInputs[0].getDimensions(), dimensions);
//
//     ASSERT_EQ(featureInputs[1].getDataType(), dataType);
//     ASSERT_EQ(featureInputs[1].getDimensions(), dimensions);
//
//     ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
//     ASSERT_EQ(featureOutputs[0].getDimensions(), dimensions);
//
//     ASSERT_EQ(featureOutputs[1].getDataType(), dataType);
//     ASSERT_EQ(featureOutputs[1].getDimensions(), dimensions);
//
//     double cloneExponentialRunningAverageFactor = clone->getExponentialRunningAverageFactor();
//     ASSERT_EQ(cloneExponentialRunningAverageFactor, exponentialRunningAverageFactor);
//
//     double cloneEpsilon = clone->getEpsilon();
//     ASSERT_EQ(cloneEpsilon, epsilon);
//
//     ASSERT_EQ(batchNormalization.getId(), clone->getId());
//     ASSERT_GT(batchNormalization.getId(), 1u);
//
//     ASSERT_TRUE(batchNormalization == *clone);
//     ASSERT_FALSE(batchNormalization != *clone);
//     ASSERT_FALSE(batchNormalization > *clone);
//     ASSERT_FALSE(batchNormalization < *clone);
// }
//
// TEST(UtilityApiLayers, BatchNormalizationSerializeDeserialize) {
//     srand(time(nullptr));
//
//     Stream stream(0);
//
//     for (uint32_t t = 0; t < 5; t++) {
//         stream.synchronize();
//
//         Network initialNetwork("initialNetwork");
//
//         DataType dataType = DataType::FP16;
//         string dataTypeString = dataType == DataType::FP32 ? "fp32" : "fp16";
//
//         vector<uint64_t> inputDimensions = {1UL + (rand() % 16)};
//
//         float exponential_running_average_factor = ((rand() % 1000) + 1) / 1001.0f;
//         float epsilon = ((rand() % 1000) + 1) / 1001.0f;
//
//         NetworkInput networkInput =
//             NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();
//
//         BatchNormalization::Builder batchNormalizationBuilder = BatchNormalization::Builder()
//                                                                     .network(initialNetwork)
//                                                                     .featureInput(networkInput.getFeatureOutput())
//                                                                     .exponentialRunningAverageFactor(exponential_running_average_factor)
//                                                                     .epsilon(epsilon);
//         BatchNormalization batchNormalization = batchNormalizationBuilder.build();
//
//         Tensor logits = batchNormalization.getFeatureOutputs()[0];
//         uint32_t numClasses = logits.getDimensions()[0];
//         NetworkInput labelsInput =
//             NetworkInput::Builder().network(initialNetwork).name("labelsInput").dimensions({numClasses}).dataType(dataType).build();
//
//         MeanAbsoluteError meanAbsoluteError = MeanAbsoluteError::Builder()
//                                                   .network(initialNetwork)
//                                                   .predictions(logits)
//                                                   .reportsRawLoss()
//                                                   .lossDataType(dataType)
//                                                   .labels(labelsInput.getFeatureOutput())
//                                                   .build();
//
//         shared_ptr<Sgd> sgd = Sgd::Builder().network(initialNetwork).initialLearningRate(0.1).decay(0.1).build();
//         NetworkOutput networkOutput = NetworkOutput::Builder()
//                                           .network(initialNetwork)
//                                           .name("testOutput")
//                                           .inputTensor(meanAbsoluteError.getLoss())
//                                           .dataType(dataType)
//                                           .build();
//
//         ASSERT_TRUE(batchNormalization.isInitialized());
//
//         vector<Tensor> featureInputs = batchNormalization.getFeatureInputs();
//         vector<Tensor> featureOutputs = batchNormalization.getFeatureOutputs();
//         assert(featureInputs[0] == networkInput.getFeatureOutput());
//
//         ASSERT_EQ(batchNormalization.getFeatureOutput(networkInput.getFeatureOutput()), featureOutputs[0]);
//
//         assert(batchNormalization.getFeatureInput(featureOutputs[0]) == featureInputs[0]);
//
//         ASSERT_EQ(featureInputs[0].getDataType(), dataType);
//         ASSERT_EQ(featureInputs[0].getDimensions(), inputDimensions);
//
//         ASSERT_EQ(featureOutputs[0].getDataType(), dataType);
//         ASSERT_EQ(featureOutputs[0].getDimensions(), inputDimensions);
//
//         // Now stamp the network and test serialization
//         uint32_t batchSize = 1 + (rand() % 16);
//         vector<Event> initDoneEvents;
//         shared_ptr<PlacedNetwork> initialPlacedNetwork = initialNetwork.place(batchSize, initDoneEvents);
//         ASSERT_TRUE(initialPlacedNetwork != nullptr);
//         for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
//             stream.waitEvent(initDoneEvents[i]);
//         }
//         initDoneEvents.clear();
//
//         // Fetch the physical batch norm layer from the stamped network and write to its weights
//         ASSERT_EQ(initialPlacedNetwork->getNumStamps(), 1UL);
//         ThorImplementation::StampedNetwork &stampedNetwork = initialPlacedNetwork->getStampedNetwork(0);
//         ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
//         shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNormLayer =
//             dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getTrainableLayer(0));
//         ASSERT_TRUE(physicalBatchNormLayer != nullptr);
//         ThorImplementation::Tensor weights = physicalBatchNormLayer->getWeights();
//         ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
//         ThorImplementation::Tensor weightsCpu = weights.clone(cpuPlacement);
//         float *weightsCpuMem = (float *)weightsCpu.getMemPtr();
//         for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
//             weightsCpuMem[i] = i;
//         }
//         weights.copyFromAsync(weightsCpu, stream);
//
//         ThorImplementation::Tensor biases = physicalBatchNormLayer->getBiases();
//         ThorImplementation::Tensor biasesCpu = biases.clone(cpuPlacement);
//         half *biasesCpuMem = (half *)biasesCpu.getMemPtr();
//         for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
//             biasesCpuMem[i] = i * i + 6;
//         }
//         biases.copyFromAsync(biasesCpu, stream);
//
//         ThorImplementation::Tensor means = physicalBatchNormLayer->getResultRunningMean();
//         ThorImplementation::Tensor meansCpu = means.clone(cpuPlacement);
//         half *meansCpuMem = (half *)meansCpu.getMemPtr();
//         for (uint32_t i = 0; i < means.getTotalNumElements(); ++i) {
//             meansCpuMem[i] = i * i + 10;
//         }
//         means.copyFromAsync(meansCpu, stream);
//
//         ThorImplementation::Tensor variances = physicalBatchNormLayer->getResultRunningVariance();
//         ThorImplementation::Tensor variancesCpu = variances.clone(cpuPlacement);
//         half *variancesCpuMem = (half *)variancesCpu.getMemPtr();
//         for (uint32_t i = 0; i < variances.getTotalNumElements(); ++i) {
//             variancesCpuMem[i] = i * i + 14;
//         }
//         variances.copyFromAsync(variancesCpu, stream);
//
//         stream.synchronize();
//
//         thor_file::TarWriter archiveWriter("testModel");
//
//         json meanAbsoluteErrorJ = meanAbsoluteError.serialize(archiveWriter, stream);
//         json networkInputJ = networkInput.serialize(archiveWriter, stream);
//         json labelsInputJ = labelsInput.serialize(archiveWriter, stream);
//         json networkOutputJ = networkOutput.serialize(archiveWriter, stream);
//
//         // The network attached the optimizer to its copy of the BN layer
//         json batchNormalizationJ;
//         bool bnFound = false;
//         shared_ptr<Layer> initalNetworkBN;
//         for (int32_t i = 0; i < initialNetwork.getNumTrainableLayers(); ++i) {
//             shared_ptr<TrainableWeightsBiasesLayer> layer = initialNetwork.getTrainableLayer(i);
//             initalNetworkBN = dynamic_pointer_cast<BatchNormalization>(layer);
//             if (initalNetworkBN) {
//                 batchNormalizationJ = initalNetworkBN->serialize(archiveWriter, stream, true,
//                 initialPlacedNetwork->getStampedNetwork(0)); bnFound = true; break;
//             }
//         }
//         ASSERT_TRUE(bnFound);
//
//         archiveWriter.createArchive("/tmp/", true);
//
//         ASSERT_EQ(batchNormalizationJ["version"], "1.0.0");
//         ASSERT_EQ(batchNormalizationJ["layer_type"], "batch_normalization");
//
//         EXPECT_TRUE(batchNormalizationJ.contains("inputs"));
//         EXPECT_TRUE(batchNormalizationJ.contains("outputs"));
//         EXPECT_TRUE(batchNormalizationJ.contains("exponential_running_average_factor"));
//         EXPECT_TRUE(batchNormalizationJ.contains("epsilon"));
//
//         ASSERT_TRUE(batchNormalizationJ.at("inputs").is_array());
//         ASSERT_TRUE(batchNormalizationJ.at("outputs").is_array());
//         ASSERT_TRUE(batchNormalizationJ.at("exponential_running_average_factor").is_number_float());
//         ASSERT_TRUE(batchNormalizationJ.at("epsilon").is_number_float());
//
//         EXPECT_EQ(batchNormalizationJ.at("exponential_running_average_factor").get<float>(), exponential_running_average_factor);
//         EXPECT_EQ(batchNormalizationJ.at("epsilon").get<float>(), epsilon);
//
//         const auto &inputs = batchNormalizationJ.at("inputs");
//         ASSERT_EQ(inputs.size(), 1U) << "Expect exactly one input";
//         const auto &in0 = inputs.at(0);
//         ASSERT_TRUE(in0.is_object());
//         ASSERT_TRUE(in0.at("data_type").is_string());
//         EXPECT_EQ(in0.at("data_type").get<string>(), dataTypeString);
//
//         ASSERT_TRUE(in0.at("dimensions").is_array());
//         ASSERT_EQ(in0.at("dimensions").size(), 1U);
//         EXPECT_TRUE(in0.at("dimensions").at(0).is_number_integer());
//         EXPECT_EQ(in0.at("dimensions").at(0).get<uint32_t>(), inputDimensions[0]);
//
//         ASSERT_TRUE(in0.at("id").is_number_integer());
//
//         const auto &outputs = batchNormalizationJ.at("outputs");
//         ASSERT_EQ(outputs.size(), 1U) << "Expect exactly one output";
//         const auto &out0 = outputs.at(0);
//         ASSERT_TRUE(out0.is_object());
//         ASSERT_TRUE(out0.at("data_type").is_string());
//         EXPECT_EQ(out0.at("data_type").get<string>(), dataType == DataType::FP16 ? "fp16" : "fp32");
//
//         ASSERT_TRUE(out0.at("dimensions").is_array());
//         ASSERT_EQ(out0.at("dimensions").size(), inputDimensions.size());
//         EXPECT_EQ(out0.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
//
//         ASSERT_TRUE(out0.at("id").is_number_integer());
//
//         EXPECT_FALSE(batchNormalizationJ.at("weights_tensor").get<string>().empty());
//         EXPECT_FALSE(batchNormalizationJ.at("biases_tensor").get<string>().empty());
//         EXPECT_FALSE(batchNormalizationJ.at("means_tensor").get<string>().empty());
//         EXPECT_FALSE(batchNormalizationJ.at("variances_tensor").get<string>().empty());
//
//         string file_prefix = "layer" + to_string(batchNormalization.getId());
//         EXPECT_EQ(batchNormalizationJ.at("weights_tensor").get<string>(), file_prefix + "_weights.tensor");
//         EXPECT_EQ(batchNormalizationJ.at("biases_tensor").get<string>(), file_prefix + "_biases.tensor");
//         EXPECT_EQ(batchNormalizationJ.at("means_tensor").get<string>(), file_prefix + "_means.tensor");
//         EXPECT_EQ(batchNormalizationJ.at("variances_tensor").get<string>(), file_prefix + "_variances.tensor");
//
//         // printf("%s\n", networkInputJ.dump(4).c_str());
//         // printf("%s\n", batchNormalizationJ.dump(4).c_str());
//         // printf("%s\n", networkOutputJ.dump(4).c_str());
//
//         ////////////////////////////
//         // Deserialize
//         ////////////////////////////
//         // Ensure the file is written before deserializing using a loader stream
//         stream.synchronize();
//
//         // Verify that the layer gets added to the network and that its weights are set to the correct values
//         Network newNetwork("newNetwork");
//
//         archiveWriter.createArchive("/tmp/", true);
//         shared_ptr<thor_file::TarReader> archiveReader = make_shared<thor_file::TarReader>("testModel", "/tmp/");
//
//         Layer::deserialize(archiveReader, networkInputJ, &newNetwork);
//         Layer::deserialize(archiveReader, labelsInputJ, &newNetwork);
//         Layer::deserialize(archiveReader, batchNormalizationJ, &newNetwork);
//         Layer::deserialize(archiveReader, meanAbsoluteErrorJ, &newNetwork);
//         Layer::deserialize(archiveReader, networkOutputJ, &newNetwork);
//
//         batchSize = 1 + (rand() % 16);
//         shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
//         ASSERT_TRUE(newPlacedNetwork != nullptr);
//         archiveReader->executeReadRequests();
//         for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
//             stream.waitEvent(initDoneEvents[i]);
//         }
//         initDoneEvents.clear();
//
//         ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
//         stampedNetwork = newPlacedNetwork->getStampedNetwork(0);
//         ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
//         shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNormLayerDes =
//             dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getTrainableLayer(0));
//         ASSERT_TRUE(physicalBatchNormLayerDes != nullptr);
//
//         ThorImplementation::Tensor weightsDes = physicalBatchNormLayerDes->getWeights();
//         ThorImplementation::Tensor weightsCpuDes = weightsDes.clone(cpuPlacement);
//         weightsCpuDes.copyFromAsync(weightsDes, stream);
//
//         ThorImplementation::Tensor biasesDes = physicalBatchNormLayerDes->getBiases();
//         ThorImplementation::Tensor biasesCpuDes = biasesDes.clone(cpuPlacement);
//         biasesCpuDes.copyFromAsync(biasesDes, stream);
//
//         ThorImplementation::Tensor meansDes = physicalBatchNormLayerDes->getResultRunningMean();
//         ThorImplementation::Tensor meansCpuDes = meansDes.clone(cpuPlacement);
//         meansCpuDes.copyFromAsync(meansDes, stream);
//
//         ThorImplementation::Tensor variancesDes = physicalBatchNormLayerDes->getResultRunningVariance();
//         ThorImplementation::Tensor variancesCpuDes = variancesDes.clone(cpuPlacement);
//         variancesCpuDes.copyFromAsync(variancesDes, stream);
//
//         stream.synchronize();
//
//         ASSERT_NE(weightsDes, weights);
//         ASSERT_EQ(weightsDes.getDimensions(), weights.getDimensions());
//         ASSERT_EQ(weightsDes.getDataType(), weights.getDataType());
//         ASSERT_TRUE(weightsDes.getPlacement() == weights.getPlacement());
//
//         float *weightsCpuMemDes = (float *)weightsCpuDes.getMemPtr();
//         for (uint32_t i = 0; i < weights.getTotalNumElements(); ++i) {
//             EXPECT_EQ(weightsCpuMemDes[i], float(i));
//             if (weightsCpuMemDes[i] != float(i))
//                 printf("i = %d\n", i);
//         }
//
//         ASSERT_NE(biasesDes, biases);
//         ASSERT_EQ(biasesDes.getDimensions(), biases.getDimensions());
//         ASSERT_EQ(biasesDes.getDataType(), biases.getDataType());
//         ASSERT_TRUE(biasesDes.getPlacement() == biases.getPlacement());
//
//         half *biasesCpuMemDes = (half *)biasesCpuDes.getMemPtr();
//         for (uint32_t i = 0; i < biases.getTotalNumElements(); ++i) {
//             EXPECT_EQ(biasesCpuMemDes[i], half(i * i + 6));
//         }
//
//         ASSERT_NE(meansDes, means);
//         ASSERT_EQ(meansDes.getDimensions(), means.getDimensions());
//         ASSERT_EQ(meansDes.getDataType(), means.getDataType());
//         ASSERT_TRUE(meansDes.getPlacement() == means.getPlacement());
//
//         half *meansCpuMemDes = (half *)meansCpuDes.getMemPtr();
//         for (uint32_t i = 0; i < means.getTotalNumElements(); ++i) {
//             EXPECT_EQ(meansCpuMemDes[i], half(i * i + 10));
//         }
//
//         ASSERT_NE(variancesDes, variances);
//         ASSERT_EQ(variancesDes.getDimensions(), variances.getDimensions());
//         ASSERT_EQ(variancesDes.getDataType(), variances.getDataType());
//         ASSERT_TRUE(variancesDes.getPlacement() == variances.getPlacement());
//
//         half *variancesCpuMemDes = (half *)variancesCpuDes.getMemPtr();
//         for (uint32_t i = 0; i < variances.getTotalNumElements(); ++i) {
//             EXPECT_EQ(variancesCpuMemDes[i], half(i * i + 14));
//         }
//     }
//
//     filesystem::remove("/tmp/testModel.thor.tar");
// }
#include <optional>

#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "Utilities/TarFile/TarWriter.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
using std::shared_ptr;
using std::string;
using std::vector;

Impl::TensorPlacement batchNormCpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

void synchronizeBatchNormEvents(vector<Event>& events) {
    for (Event& event : events) {
        event.synchronize();
    }
}

vector<float> readBatchNormParameter(const shared_ptr<Impl::PhysicalParameter>& parameter, Stream& stream) {
    EXPECT_NE(parameter, nullptr);
    if (parameter == nullptr || !parameter->getStorage().has_value()) {
        return {};
    }

    Impl::Tensor deviceTensor = parameter->getStorage().value();
    EXPECT_EQ(deviceTensor.getDataType(), DataType::FP32);
    Impl::Tensor cpuTensor = deviceTensor.clone(batchNormCpuPlacement);
    cpuTensor.copyFromAsync(deviceTensor, stream);
    stream.synchronize();

    const float* mem = cpuTensor.getMemPtr<float>();
    return vector<float>(mem, mem + cpuTensor.getTotalNumElements());
}

void setBatchNormParameter(const shared_ptr<Impl::PhysicalParameter>& parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());

    Impl::Tensor deviceTensor = parameter->getStorage().value();
    ASSERT_EQ(deviceTensor.getDataType(), DataType::FP32);
    ASSERT_EQ(deviceTensor.getTotalNumElements(), values.size());

    Impl::Tensor cpuTensor = deviceTensor.clone(batchNormCpuPlacement);
    float* mem = cpuTensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i) {
        mem[i] = values[i];
    }
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

template <typename LayerT>
shared_ptr<LayerT> findOnlyBatchNormTestLayerOfType(Api::Network& network) {
    shared_ptr<LayerT> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        shared_ptr<LayerT> candidate = std::dynamic_pointer_cast<LayerT>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1u);
    return found;
}

struct PlacedBatchNormFixture {
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    shared_ptr<Impl::BatchNormalization> physicalBatchNorm;
};

PlacedBatchNormFixture placeBatchNorm(Api::Network& network, const Api::BatchNormalization& batchNorm, uint32_t batchSize) {
    vector<Event> initDoneEvents;
    PlacedBatchNormFixture fixture;
    fixture.placedNetwork = network.place(batchSize, initDoneEvents, true);
    synchronizeBatchNormEvents(initDoneEvents);
    EXPECT_NE(fixture.placedNetwork, nullptr);
    if (fixture.placedNetwork == nullptr) {
        return fixture;
    }

    fixture.physicalBatchNorm = std::dynamic_pointer_cast<Impl::BatchNormalization>(
        fixture.placedNetwork->getStampedNetwork(0).getPhysicalLayerFromApiLayer(batchNorm.getId()));
    EXPECT_NE(fixture.physicalBatchNorm, nullptr);
    return fixture;
}

std::filesystem::path makeUniqueBatchNormArchiveDir(const string& testName) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir = std::filesystem::temp_directory_path() / (testName + "_" + std::to_string(now));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

}  // namespace

TEST(BatchNormalizationApi, BuilderRegistersAndInitializesAllPersistentState) {
    constexpr uint32_t batchSize = 2;
    constexpr uint64_t channels = 3;

    Api::Network network("batch_norm_initial_state");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({channels, 2, 2})
                                  .dataType(DataType::FP16)
                                  .build();
    Api::BatchNormalization batchNorm = Api::BatchNormalization::Builder()
                                                .network(network)
                                                .featureInput(input.getFeatureOutput().value())
                                                .build();
    Api::NetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(batchNorm.getFeatureOutput().value())
        .dataType(DataType::FP16)
        .build();

    EXPECT_EQ(batchNorm.listParameters(), (vector<string>{"weights", "biases", "running_mean", "running_variance"}));
    EXPECT_TRUE(batchNorm.getParameterSpecification("weights")->isTrainable());
    EXPECT_TRUE(batchNorm.getParameterSpecification("biases")->isTrainable());
    EXPECT_FALSE(batchNorm.getParameterSpecification("running_mean")->isTrainable());
    EXPECT_FALSE(batchNorm.getParameterSpecification("running_variance")->isTrainable());

    const nlohmann::json architecture = batchNorm.architectureJson();
    ASSERT_TRUE(architecture.at("parameters").is_object());
    EXPECT_EQ(architecture.at("parameters").size(), 4u);
    EXPECT_EQ(architecture.at("num_items_observed").get<uint64_t>(), 0u);

    PlacedBatchNormFixture fixture = placeBatchNorm(network, batchNorm, batchSize);
    ASSERT_NE(fixture.physicalBatchNorm, nullptr);
    EXPECT_EQ(fixture.physicalBatchNorm->listParameters(),
              (vector<string>{"weights", "biases", "running_mean", "running_variance"}));

    Stream stream = fixture.physicalBatchNorm->getStreams()[0];
    EXPECT_EQ(readBatchNormParameter(fixture.physicalBatchNorm->getParameter("weights"), stream), vector<float>(channels, 1.0f));
    EXPECT_EQ(readBatchNormParameter(fixture.physicalBatchNorm->getParameter("biases"), stream), vector<float>(channels, 0.0f));
    EXPECT_EQ(readBatchNormParameter(fixture.physicalBatchNorm->getParameter("running_mean"), stream), vector<float>(channels, 0.0f));
    EXPECT_EQ(readBatchNormParameter(fixture.physicalBatchNorm->getParameter("running_variance"), stream), vector<float>(channels, 1.0f));
}

TEST(BatchNormalizationApi, PlacedSaveLoadRoundTripRestoresParametersAndRunningState) {
    constexpr uint32_t batchSize = 2;
    constexpr uint64_t channels = 3;
    constexpr uint64_t itemsObserved = 37;
    const vector<float> weights = {1.25f, 0.75f, -0.5f};
    const vector<float> biases = {0.1f, -0.2f, 0.3f};
    const vector<float> runningMean = {2.0f, -1.0f, 0.5f};
    const vector<float> runningVariance = {4.0f, 2.5f, 0.25f};

    const string networkName = "batch_norm_state_round_trip";
    std::filesystem::path archiveDir = makeUniqueBatchNormArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("input")
                                      .dimensions({channels, 2, 2})
                                      .dataType(DataType::FP32)
                                      .build();
        Api::BatchNormalization batchNorm = Api::BatchNormalization::Builder()
                                                    .network(network)
                                                    .featureInput(input.getFeatureOutput().value())
                                                    .exponentialRunningAverageFactor(0.125)
                                                    .epsilon(0.0002)
                                                    .build();
        Api::NetworkOutput::Builder()
            .network(network)
            .name("output")
            .inputTensor(batchNorm.getFeatureOutput().value())
            .dataType(DataType::FP32)
            .build();

        PlacedBatchNormFixture fixture = placeBatchNorm(network, batchNorm, batchSize);
        ASSERT_NE(fixture.physicalBatchNorm, nullptr);
        Stream stream = fixture.physicalBatchNorm->getStreams()[0];
        setBatchNormParameter(fixture.physicalBatchNorm->getParameter("weights"), weights, stream);
        setBatchNormParameter(fixture.physicalBatchNorm->getParameter("biases"), biases, stream);
        setBatchNormParameter(fixture.physicalBatchNorm->getParameter("running_mean"), runningMean, stream);
        setBatchNormParameter(fixture.physicalBatchNorm->getParameter("running_variance"), runningVariance, stream);
        fixture.physicalBatchNorm->setNumItemsObserved(itemsObserved);
        stream.synchronize();

        thor_file::TarWriter formatWriter("batch_norm_serialization_format");
        nlohmann::json serialized =
            batchNorm.serialize(formatWriter, stream, false, fixture.placedNetwork->getStampedNetwork(0));
        EXPECT_EQ(serialized.at("num_items_observed").get<uint64_t>(), itemsObserved);
        ASSERT_TRUE(serialized.at("parameters").is_object());
        EXPECT_EQ(serialized.at("parameters").size(), 4u);
        for (const string& parameterName : vector<string>{"weights", "biases", "running_mean", "running_variance"}) {
            ASSERT_TRUE(serialized.at("parameters").contains(parameterName));
            EXPECT_TRUE(serialized.at("parameters").at(parameterName).contains("storage_file"));
        }
        EXPECT_FALSE(serialized.contains("weights_tensor"));
        EXPECT_FALSE(serialized.contains("biases_tensor"));
        EXPECT_FALSE(serialized.contains("means_tensor"));
        EXPECT_FALSE(serialized.contains("variances_tensor"));

        fixture.placedNetwork->save(archiveDir.string(), true, false);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());
        shared_ptr<Api::BatchNormalization> loadedBatchNorm = findOnlyBatchNormTestLayerOfType<Api::BatchNormalization>(loadedNetwork);
        ASSERT_NE(loadedBatchNorm, nullptr);
        EXPECT_DOUBLE_EQ(loadedBatchNorm->getExponentialRunningAverageFactor().value(), 0.125);
        EXPECT_DOUBLE_EQ(loadedBatchNorm->getEpsilon().value(), 0.0002);
        EXPECT_EQ(loadedBatchNorm->listParameters(), (vector<string>{"weights", "biases", "running_mean", "running_variance"}));

        const nlohmann::json loadedArchitecture = loadedBatchNorm->architectureJson();
        EXPECT_EQ(loadedArchitecture.at("num_items_observed").get<uint64_t>(), itemsObserved);
        ASSERT_TRUE(loadedArchitecture.at("parameters").is_object());
        EXPECT_EQ(loadedArchitecture.at("parameters").size(), 4u);

        PlacedBatchNormFixture loadedFixture = placeBatchNorm(loadedNetwork, *loadedBatchNorm, batchSize);
        ASSERT_NE(loadedFixture.physicalBatchNorm, nullptr);
        Stream loadedStream = loadedFixture.physicalBatchNorm->getStreams()[0];
        EXPECT_EQ(readBatchNormParameter(loadedFixture.physicalBatchNorm->getParameter("weights"), loadedStream), weights);
        EXPECT_EQ(readBatchNormParameter(loadedFixture.physicalBatchNorm->getParameter("biases"), loadedStream), biases);
        EXPECT_EQ(readBatchNormParameter(loadedFixture.physicalBatchNorm->getParameter("running_mean"), loadedStream), runningMean);
        EXPECT_EQ(readBatchNormParameter(loadedFixture.physicalBatchNorm->getParameter("running_variance"), loadedStream), runningVariance);
        EXPECT_EQ(loadedFixture.physicalBatchNorm->getNumItemsObserved(), itemsObserved);
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }

    std::filesystem::remove_all(archiveDir);
}
