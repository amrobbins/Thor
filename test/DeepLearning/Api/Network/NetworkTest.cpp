// #include "DeepLearning/Api/ExampleNetworks/AlexNet.h"
// #include "DeepLearning/Api/Initializers/UniformRandom.h"
// #include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
// #include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
// #include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
// #include "DeepLearning/Api/Network/Network.h"
// #include "DeepLearning/Api/Network/PlacedNetwork.h"
// #include "DeepLearning/Api/Optimizers/Sgd.h"
// #include "DeepLearning/Implementation/Layers/Activation/Sigmoid.h"
//
// #include "test/DeepLearning/Api/Helpers/GradientRivet.h"
//
// #include <stdio.h>
// #include <unistd.h>
// #include "cuda.h"
// #include "cuda_fp16.h"
// #include "cuda_runtime.h"
// #include "gtest/gtest.h"
//
// #include <math.h>
// #include <set>
// #include <vector>
//
// using namespace std;
//
// using namespace Thor;
//
// TEST(Network, SimplestNetworkProperlyFormed) {
//     Network network("testNetwork");
//     Tensor latestOutputTensor;
//     shared_ptr<Initializer> uniformRandomInitializer = UniformRandom::Builder().minValue(-0.1).maxValue(0.1).build();
//
//     NetworkInput networkInput =
//         NetworkInput::Builder().network(network).name("input").dimensions({1024}).dataType(DataType::FP16).build();
//     latestOutputTensor = networkInput.getFeatureOutput();
//
//     FullyConnected fullyConnected = FullyConnected::Builder()
//                                         .network(network)
//                                         .featureInput(latestOutputTensor)
//                                         .numOutputFeatures(1)
//                                         .hasBias(true)
//                                         .weightsInitializer(uniformRandomInitializer)
//                                         .biasInitializer(uniformRandomInitializer)
//                                         .noActivation()
//                                         .build();
//     latestOutputTensor = fullyConnected.getFeatureOutput();
//
//     std::shared_ptr<Sgd> sgd =
//         Sgd::Builder().initialLearningRate(0.01).decay(0).momentum(0).useNesterovMomentum(true).network(network).build();
//
//     NetworkInput label = NetworkInput::Builder().network(network).name("label").dimensions({1}).dataType(DataType::FP16).build();
//     MeanSquaredError meanSquaredError = MeanSquaredError::Builder()
//                                             .network(network)
//                                             .lossDataType(DataType::FP16)
//                                             .reportsRawLoss()
//                                             .predictions(fullyConnected.getFeatureOutput())
//                                             .labels(label.getFeatureOutput())
//                                             .build();
//
//     NetworkOutput networkOutput = NetworkOutput::Builder()
//                                       .network(network)
//                                       .name("output")
//                                       .inputTensor(meanSquaredError.getLoss())
//                                       .dataType(DataType::FP16)
//                                       .build();
//     Tensor networkOutputTensor = networkOutput.getFeatureOutput();
//
//     ThorImplementation::StampedNetwork stampedNetwork;
//     int gpuNum = 0;
//     int stampsPerGpu = 1;
//     int batchSize = 32;
//     vector<Event> initDoneEvents;
//     shared_ptr<PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, false, {gpuNum}, stampsPerGpu);
//     ASSERT_TRUE(placedNetwork != nullptr);
//     ASSERT_EQ(placedNetwork->getNumStamps(), 1UL);
//     stampedNetwork = placedNetwork->getStampedNetwork(0);
//
//     // Check network structure
//     ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1u);
//     shared_ptr<ThorImplementation::FullyConnected> fc =
//         dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getApiLayerToPhysicalLayer()[fullyConnected.getId()]);
//     ASSERT_NE(fc, nullptr);
//     // otherLayers holds loss layer
//     ASSERT_EQ(stampedNetwork.getOtherLayers().size(), 1u);
//     shared_ptr<ThorImplementation::MeanSquaredError> mse =
//         dynamic_pointer_cast<ThorImplementation::MeanSquaredError>(stampedNetwork.getApiLayerToPhysicalLayer()[meanSquaredError.getId()]);
//     ASSERT_NE(mse, nullptr);
//     ASSERT_EQ(stampedNetwork.getInputs().size(), 2u);
//     if (stampedNetwork.getInputs()[0]->getFeatureOutput().value() ==
//         stampedNetwork.getApiLayerToPhysicalLayer()[networkInput.getId()]->getFeatureOutput().value()) {
//         ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(),
//                   stampedNetwork.getApiLayerToPhysicalLayer()[networkInput.getId()]->getFeatureOutput().value());
//         ASSERT_EQ(stampedNetwork.getInputs()[0]->getName(), "input");
//         ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(), fc->getFeatureInputs()[0].value());
//
//         ASSERT_EQ(stampedNetwork.getInputs()[1]->getFeatureOutput().value(),
//                   stampedNetwork.getApiLayerToPhysicalLayer()[label.getId()]->getFeatureOutput().value());
//         ASSERT_EQ(stampedNetwork.getInputs()[1]->getName(), "label");
//         ASSERT_EQ(stampedNetwork.getInputs()[1]->getFeatureOutput().value(), mse->getLabelsInput().value());
//     } else {
//         ASSERT_EQ(stampedNetwork.getInputs()[1]->getFeatureOutput().value(),
//                   stampedNetwork.getApiLayerToPhysicalLayer()[networkInput.getId()]->getFeatureOutput().value());
//         ASSERT_EQ(stampedNetwork.getInputs()[1]->getName(), "input");
//         ASSERT_EQ(stampedNetwork.getInputs()[1]->getFeatureOutput().value(), fc->getFeatureInputs()[0].value());
//
//         ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(),
//                   stampedNetwork.getApiLayerToPhysicalLayer()[label.getId()]->getFeatureOutput().value());
//         ASSERT_EQ(stampedNetwork.getInputs()[0]->getName(), "label");
//         ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(), mse->getLabelsInput().value());
//     }
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureInputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureInputs()[0].value(), fc->getFeatureInputs()[0].value());
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureOutputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureOutputs()[0].value(), fc->getFeatureOutputs()[0].value());
//     ASSERT_EQ(mse->getPredictionsInput().has_value(), true);
//     ASSERT_EQ(mse->getPredictionsInput().value(), fc->getFeatureOutputs()[0].value());
//     ASSERT_EQ(stampedNetwork.getOutputs().size(), 1u);
//     ASSERT_EQ(mse->getLossOutput().has_value(), true);
//     ASSERT_EQ(mse->getLossOutput().value(), stampedNetwork.getOutputs()[0]->getFeatureInput().value());
//     ASSERT_EQ(stampedNetwork.getOutputs()[0]->getName(), "output");
//     ASSERT_EQ(stampedNetwork.getOutputs()[0]->getFeatureInput().value(),
//               stampedNetwork.getApiLayerToPhysicalLayer()[networkOutput.getId()]->getFeatureInput().value());
//
//     // Check weights initialization
//     ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
//     ThorImplementation::Tensor fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *weightsMem = (half *)fcWeights.getMemPtr();
//     for (uint32_t i = 0; i < 1024; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     ThorImplementation::Tensor fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *biasesMem = (half *)fcBiases.getMemPtr();
//     for (uint32_t i = 0; i < 1; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
// }
//
// TEST(Network, SimplestNetworkWithGlorotUniformProperlyFormed) {
//     Network network("testNetwork");
//     Tensor latestOutputTensor;
//     shared_ptr<Initializer> glorot = Glorot::Builder().mode(ThorImplementation::Glorot::Mode::UNIFORM).build();
//
//     constexpr uint64_t NUM_INPUTS = 1024;
//     constexpr uint64_t NUM_OUTPUTS = 500;
//     constexpr uint64_t NUM_WEIGHTS = NUM_INPUTS * NUM_OUTPUTS;
//     constexpr uint64_t NUM_BIASES = NUM_OUTPUTS;
//
//     NetworkInput networkInput =
//         NetworkInput::Builder().network(network).name("input").dimensions({NUM_INPUTS}).dataType(DataType::FP16).build();
//     latestOutputTensor = networkInput.getFeatureOutput();
//
//     FullyConnected fullyConnected = FullyConnected::Builder()
//                                         .network(network)
//                                         .featureInput(latestOutputTensor)
//                                         .numOutputFeatures(NUM_OUTPUTS)
//                                         .hasBias(true)
//                                         .weightsInitializer(glorot)
//                                         .biasInitializer(glorot)
//                                         .noActivation()
//                                         .build();
//     latestOutputTensor = fullyConnected.getFeatureOutput();
//
//     GradientRivet gradientRivet = GradientRivet::Builder().network(network).tensor(latestOutputTensor).build();
//     latestOutputTensor = gradientRivet.getFeatureOutput();
//
//     NetworkOutput networkOutput =
//         NetworkOutput::Builder().network(network).name("output").inputTensor(latestOutputTensor).dataType(DataType::FP16).build();
//     Tensor networkOutputTensor = networkOutput.getFeatureOutput();
//
//     shared_ptr<Sgd> sgd = Sgd::Builder().network(network).initialLearningRate(0.1).decay(0.1).build();
//
//     ThorImplementation::StampedNetwork stampedNetwork;
//     int gpuNum = 0;
//     int batchSize = 32;
//     vector<Event> initDoneEvents;
//     shared_ptr<PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, false, {gpuNum}, 1);
//     ASSERT_TRUE(placedNetwork != nullptr);
//     ASSERT_EQ(placedNetwork->getNumStamps(), 1UL);
//     stampedNetwork = placedNetwork->getStampedNetwork(0);
//
//     // Check network structure
//     ASSERT_EQ(stampedNetwork.getInputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(),
//               stampedNetwork.getApiLayerToPhysicalLayer()[networkInput.getId()]->getFeatureOutput().value());
//     ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1u);
//     shared_ptr<ThorImplementation::FullyConnected> fc =
//         dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getApiLayerToPhysicalLayer()[fullyConnected.getId()]);
//     shared_ptr<ThorImplementation::GradientRivet> rivet =
//         dynamic_pointer_cast<ThorImplementation::GradientRivet>(stampedNetwork.getApiLayerToPhysicalLayer()[gradientRivet.getId()]);
//     ASSERT_NE(fc, nullptr);
//     ASSERT_NE(rivet, nullptr);
//     ASSERT_EQ(stampedNetwork.getInputs()[0]->getName(), "input");
//     ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(), fc->getFeatureInputs()[0].value());
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureInputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureInputs()[0].value(), fc->getFeatureInputs()[0].value());
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureOutputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureOutputs()[0].value(), fc->getFeatureOutputs()[0].value());
//     ASSERT_EQ(stampedNetwork.getOutputs().size(), 1u);
//     ASSERT_EQ(fc->getFeatureOutputs()[0].value(), rivet->getFeatureInput().value());
//     ASSERT_EQ(rivet->getFeatureOutput().value(), stampedNetwork.getOutputs()[0]->getFeatureInput().value());
//     ASSERT_EQ(stampedNetwork.getOutputs()[0]->getName(), "output");
//     ASSERT_EQ(stampedNetwork.getOutputs()[0]->getFeatureInput().value(),
//               stampedNetwork.getApiLayerToPhysicalLayer()[networkOutput.getId()]->getFeatureInput().value());
//     ASSERT_EQ(stampedNetwork.getOtherLayers().size(), 1u);
//
//     // Check weights initialization
//     // First check if fanIn and fanOut is correct
//     uint64_t fanIn = fc->getFanIn();
//     uint64_t fanOut = fc->getFanOut();
//     ASSERT_EQ(fanIn, NUM_INPUTS);
//     ASSERT_EQ(fanOut, NUM_BIASES);
//     half maxValue = sqrt(6.0 / (fanIn + fanOut)) * 1.01;
//     half minValue = (half)-1.0f * maxValue;
//     ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
//     ThorImplementation::Tensor fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *weightsMem = (half *)fcWeights.getMemPtr();
//     uint32_t numZero = 0;
//     uint32_t numNonZero = 0;
//     half zero(0);
//     for (uint32_t i = 0; i < NUM_WEIGHTS; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= minValue && weightsMem[i] <= maxValue);
//         if (weightsMem[i] == zero)
//             ++numZero;
//         else
//             ++numNonZero;
//     }
//     ASSERT_LT(numZero, numNonZero * 0.1);
//     ThorImplementation::Tensor fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *biasesMem = (half *)fcBiases.getMemPtr();
//     numZero = 0;
//     numNonZero = 0;
//     for (uint32_t i = 0; i < NUM_BIASES; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= minValue && biasesMem[i] <= maxValue);
//         if (biasesMem[i] == zero)
//             ++numZero;
//         else
//             ++numNonZero;
//     }
//     ASSERT_LT(numZero, numNonZero * 0.1);
// }
//
// TEST(Network, SimplestNetworkWithGlorotNormalProperlyFormed) {
//     Network network("testNetwork");
//     Tensor latestOutputTensor;
//     shared_ptr<Initializer> glorot = Glorot::Builder().mode(ThorImplementation::Glorot::Mode::UNIFORM).build();
//
//     constexpr uint64_t NUM_INPUTS = 1024;
//     constexpr uint64_t NUM_OUTPUTS = 500;
//     constexpr uint64_t NUM_WEIGHTS = NUM_INPUTS * NUM_OUTPUTS;
//     constexpr uint64_t NUM_BIASES = NUM_OUTPUTS;
//
//     NetworkInput networkInput =
//         NetworkInput::Builder().network(network).name("input").dimensions({NUM_INPUTS}).dataType(DataType::FP16).build();
//     latestOutputTensor = networkInput.getFeatureOutput();
//
//     FullyConnected fullyConnected = FullyConnected::Builder()
//                                         .network(network)
//                                         .featureInput(latestOutputTensor)
//                                         .numOutputFeatures(NUM_OUTPUTS)
//                                         .hasBias(true)
//                                         .weightsInitializer(glorot)
//                                         .biasInitializer(glorot)
//                                         .noActivation()
//                                         .build();
//     latestOutputTensor = fullyConnected.getFeatureOutput();
//
//     GradientRivet gradientRivet = GradientRivet::Builder().network(network).tensor(latestOutputTensor).build();
//     latestOutputTensor = gradientRivet.getFeatureOutput();
//
//     NetworkOutput networkOutput =
//         NetworkOutput::Builder().network(network).name("output").inputTensor(latestOutputTensor).dataType(DataType::FP16).build();
//     Tensor networkOutputTensor = networkOutput.getFeatureOutput();
//
//     shared_ptr<Sgd> sgd = Sgd::Builder().network(network).initialLearningRate(0.1).decay(0.1).build();
//     ThorImplementation::StampedNetwork stampedNetwork;
//     int gpuNum = 0;
//     int batchSize = 32;
//     uint32_t stampsPerGpu = 1;
//     vector<Event> initDoneEvents;
//     shared_ptr<PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, false, {gpuNum}, stampsPerGpu);
//     ASSERT_TRUE(placedNetwork != nullptr);
//     ASSERT_EQ(placedNetwork->getNumStamps(), 1UL);
//     stampedNetwork = placedNetwork->getStampedNetwork(0);
//
//     // Check network structure
//     ASSERT_EQ(stampedNetwork.getInputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(),
//               stampedNetwork.getApiLayerToPhysicalLayer()[networkInput.getId()]->getFeatureOutput().value());
//     ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1u);
//     shared_ptr<ThorImplementation::FullyConnected> fc =
//         dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getApiLayerToPhysicalLayer()[fullyConnected.getId()]);
//     shared_ptr<ThorImplementation::GradientRivet> rivet =
//         dynamic_pointer_cast<ThorImplementation::GradientRivet>(stampedNetwork.getApiLayerToPhysicalLayer()[gradientRivet.getId()]);
//     ASSERT_NE(fc, nullptr);
//     ASSERT_NE(rivet, nullptr);
//     ASSERT_EQ(stampedNetwork.getInputs()[0]->getName(), "input");
//     ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(), fc->getFeatureInputs()[0].value());
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureInputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureInputs()[0].value(), fc->getFeatureInputs()[0].value());
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureOutputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getTrainableLayer(0)->getFeatureOutputs()[0].value(), fc->getFeatureOutputs()[0].value());
//     ASSERT_EQ(stampedNetwork.getOutputs().size(), 1u);
//     ASSERT_EQ(fc->getFeatureOutputs()[0].value(), rivet->getFeatureInput().value());
//     ASSERT_EQ(rivet->getFeatureOutput().value(), stampedNetwork.getOutputs()[0]->getFeatureInput().value());
//     ASSERT_EQ(stampedNetwork.getOutputs()[0]->getName(), "output");
//     ASSERT_EQ(stampedNetwork.getOutputs()[0]->getFeatureInput().value(),
//               stampedNetwork.getApiLayerToPhysicalLayer()[networkOutput.getId()]->getFeatureInput().value());
//     ASSERT_EQ(stampedNetwork.getOtherLayers().size(), 1u);
//
//     // Check weights initialization
//     // First check if fanIn and fanOut is correct
//     uint64_t fanIn = fc->getFanIn();
//     uint64_t fanOut = fc->getFanOut();
//     ASSERT_EQ(fanIn, NUM_INPUTS);
//     ASSERT_EQ(fanOut, NUM_OUTPUTS);
//
//     ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
//     ThorImplementation::Tensor fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *weightsMem = (half *)fcWeights.getMemPtr();
//     uint32_t numZero = 0;
//     uint32_t numNonZero = 0;
//     half zero(0);
//     double totalWeights = 0.0;
//     for (uint32_t i = 0; i < NUM_WEIGHTS; ++i) {
//         totalWeights += (double)weightsMem[i];
//         if (weightsMem[i] == zero)
//             ++numZero;
//         else
//             ++numNonZero;
//     }
//     ASSERT_LT(numZero, numNonZero * 0.1);
//     double mean = totalWeights / double(NUM_WEIGHTS);
//     double totalVariance = 0;
//     for (uint32_t i = 0; i < NUM_WEIGHTS; ++i) {
//         double val = (double)weightsMem[i] - mean;
//         totalVariance += val * val;
//     }
//     double avgVariance = totalVariance / double(NUM_WEIGHTS);
//     double stdDev = sqrt(avgVariance);
//
//     double expectedMean = 0.0;
//     double expectedStdDev = sqrt(2.0 / (fanIn + fanOut));
//
//     ASSERT_LT(abs(mean - expectedMean), 5e-4);
//     ASSERT_LT(abs(stdDev - expectedStdDev), 0.10);
//
//     ThorImplementation::Tensor fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *biasesMem = (half *)fcBiases.getMemPtr();
//     double totalBias = 0.0;
//     numZero = 0;
//     numNonZero = 0;
//     for (uint32_t i = 0; i < NUM_BIASES; ++i) {
//         totalBias += (double)biasesMem[i];
//         if (biasesMem[i] == zero)
//             ++numZero;
//         else
//             ++numNonZero;
//     }
//     ASSERT_LT(numZero, numNonZero * 0.1);
//     mean = totalBias / NUM_BIASES;
//     totalVariance = 0;
//     for (uint32_t i = 0; i < NUM_BIASES; ++i) {
//         double val = (double)biasesMem[i] - mean;
//         totalVariance += val * val;
//     }
//     avgVariance = totalVariance / NUM_BIASES;
//     stdDev = sqrt(avgVariance);
//
//     expectedMean = 0.0;
//     expectedStdDev = sqrt(2.0 / (fanIn + fanOut));
//
//     ASSERT_LT(abs(mean - expectedMean), 5e-3);
//     ASSERT_LT(abs(stdDev - expectedStdDev), 0.25);
// }
//
// TEST(Network, SimpleNetworkWithCompoundLayerProperlyFormed) {
//     Network network("testNetwork");
//     Tensor latestOutputTensor;
//     shared_ptr<Activation> reluAct = Relu::Builder().build();
//     shared_ptr<Initializer> uniformRandomInitializer = UniformRandom::Builder().minValue(2).maxValue(3).build();
//
//     Tensor networkInputTensor = NetworkInput::Builder()
//                                     .network(network)
//                                     .name("features")
//                                     .dimensions({500})
//                                     .dataType(DataType::UINT8)
//                                     .build()
//                                     .getFeatureOutput();
//     latestOutputTensor = networkInputTensor;
//     latestOutputTensor = FullyConnected::Builder()
//                              .network(network)
//                              .featureInput(latestOutputTensor)
//                              .numOutputFeatures(800)
//                              .hasBias(false)
//                              .activation(reluAct)
//                              .dropOut(0.5)
//                              .batchNormalization()
//                              .weightsInitializer(uniformRandomInitializer)
//                              .build()
//                              .getFeatureOutput();
//     latestOutputTensor =
//         DropOut::Builder().network(network).featureInput(latestOutputTensor).dropProportion(0.25).build().getFeatureOutput();
//
//     GradientRivet gradientRivet = GradientRivet::Builder().network(network).tensor(latestOutputTensor).build();
//     latestOutputTensor = gradientRivet.getFeatureOutput();
//
//     Tensor networkOutputTensor = NetworkOutput::Builder()
//                                      .network(network)
//                                      .name("output")
//                                      .inputTensor(latestOutputTensor)
//                                      .dataType(DataType::FP32)
//                                      .build()
//                                      .getFeatureOutput();
//
//     shared_ptr<Sgd> sgd = Sgd::Builder().network(network).initialLearningRate(0.1).decay(0.1).build();
//
//     ASSERT_EQ(networkInputTensor.getDataType(), DataType::UINT8);
//     ASSERT_EQ(networkOutputTensor.getDataType(), DataType::FP32);
//
//     ThorImplementation::StampedNetwork stampedNetwork;
//     int gpuNum = 0;
//     int batchSize = 32;
//     uint32_t stampsPerGpu = 1;
//     vector<Event> initDoneEvents;
//     shared_ptr<PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, false, {gpuNum}, stampsPerGpu);
//     ASSERT_TRUE(placedNetwork != nullptr);
//     ASSERT_EQ(placedNetwork->getNumStamps(), 1UL);
//     stampedNetwork = placedNetwork->getStampedNetwork(0);
//
//     // Check network structure
//     ASSERT_EQ(stampedNetwork.getInputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getOutputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 2u);
//     ASSERT_EQ(stampedNetwork.getOtherLayers().size(), 6u);
//
//     shared_ptr<ThorImplementation::NetworkInput> input = stampedNetwork.getInputs()[0];
//     ASSERT_EQ(input->getName(), "features");
//
//     shared_ptr<ThorImplementation::NetworkOutput> output = stampedNetwork.getOutputs()[0];
//     ASSERT_EQ(output->getName(), "output");
//
//     shared_ptr<ThorImplementation::BatchNormalization> bn =
//         dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getTrainableLayer(0));
//     assert(bn != nullptr);
//     shared_ptr<ThorImplementation::FullyConnected> fc =
//         dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(1));
//     assert(fc != nullptr);
//
//     shared_ptr<ThorImplementation::TypeConversion> tc8_16 =
//         dynamic_pointer_cast<ThorImplementation::TypeConversion>(stampedNetwork.getOtherLayers()[0]);
//     ASSERT_NE(tc8_16, nullptr);
//     shared_ptr<ThorImplementation::DropOut> dropout =
//     dynamic_pointer_cast<ThorImplementation::DropOut>(stampedNetwork.getOtherLayers()[1]); ASSERT_NE(dropout, nullptr);
//     shared_ptr<ThorImplementation::Relu> relu = dynamic_pointer_cast<ThorImplementation::Relu>(stampedNetwork.getOtherLayers()[2]);
//     ASSERT_NE(relu, nullptr);
//     shared_ptr<ThorImplementation::DropOut> dropout2 =
//         dynamic_pointer_cast<ThorImplementation::DropOut>(stampedNetwork.getOtherLayers()[3]);
//     ASSERT_NE(dropout2, nullptr);
//     shared_ptr<ThorImplementation::GradientRivet> rivet =
//         dynamic_pointer_cast<ThorImplementation::GradientRivet>(stampedNetwork.getOtherLayers()[4]);
//     ASSERT_NE(rivet, nullptr);
//     shared_ptr<ThorImplementation::TypeConversion> tc16_32 =
//         dynamic_pointer_cast<ThorImplementation::TypeConversion>(stampedNetwork.getOtherLayers()[5]);
//     ASSERT_NE(tc16_32, nullptr);
//
//     ASSERT_EQ(input->getFeatureOutput().value(), tc8_16->getFeatureInput().value());
//     ASSERT_EQ(tc8_16->getFeatureOutput().value(), bn->getFeatureInputs()[0].value());
//     ASSERT_EQ(bn->getFeatureOutputs()[0].value(), dropout->getFeatureInput().value());
//     ASSERT_EQ(dropout->getFeatureOutput().value(), fc->getFeatureInputs()[0].value());
//     ASSERT_EQ(fc->getFeatureOutputs()[0].value(), relu->getFeatureInput().value());
//     ASSERT_EQ(relu->getFeatureOutput().value(), dropout2->getFeatureInput().value());
//     ASSERT_EQ(dropout2->getFeatureOutput().value(), rivet->getFeatureInput().value());
//     ASSERT_EQ(rivet->getFeatureOutput().value(), tc16_32->getFeatureInput().value());
//     ASSERT_EQ(tc16_32->getFeatureOutput().value(), output->getFeatureInput().value());
//
//     ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
//     ThorImplementation::Tensor fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *weightsMem = (half *)fcWeights.getMemPtr();
//     for (uint32_t i = 0; i < 500 * 800; ++i) {
//         if (!(weightsMem[i] >= (half)2 && weightsMem[i] <= (half)3)) {
//             printf("weightsMem[%d] %f\n", i, (float)weightsMem[i]);
//         }
//         ASSERT_TRUE(weightsMem[i] >= (half)2 && weightsMem[i] <= (half)3);
//     }
// }
//
// TEST(Network, BranchedNetworkProperlyFormed) {
//     Network network("testNetwork");
//     Tensor latestOutputTensor;
//     shared_ptr<Activation> relu = Relu::Builder().build();
//     shared_ptr<Initializer> uniformRandomInitializer = UniformRandom::Builder().minValue(-0.1).maxValue(0.1).build();
//
//     NetworkInput networkInput =
//         NetworkInput::Builder().network(network).name("input").dimensions({1024}).dataType(DataType::FP16).build();
//
//     FullyConnected::Builder fc0Builder = FullyConnected::Builder()
//                                              .network(network)
//                                              .featureInput(networkInput.getFeatureOutput())
//                                              .numOutputFeatures(800)
//                                              .hasBias(true)
//                                              .activation(relu)
//                                              .dropOut(0.5)
//                                              .batchNormalization()
//                                              .weightsInitializer(uniformRandomInitializer);
//     FullyConnected fc0 = fc0Builder.build();
//     FullyConnected fc1 = fc0Builder.build();
//
//     FullyConnected fc2 = FullyConnected::Builder()
//                              .network(network)
//                              .featureInput(fc0.getFeatureOutput())
//                              .featureInput(fc1.getFeatureOutput())
//                              .numOutputFeatures(200)
//                              .hasBias(true)
//                              .activation(relu)
//                              .dropOut(0.5)
//                              .batchNormalization()
//                              .weightsInitializer(uniformRandomInitializer)
//                              .build();
//
//     GradientRivet gradientRivet = GradientRivet::Builder().network(network).tensor(fc2.getFeatureOutputs()[0]).build();
//     latestOutputTensor = gradientRivet.getFeatureOutput();
//
//     NetworkOutput networkOutput = NetworkOutput::Builder()
//                                       .network(network)
//                                       .name("output")
//                                       .inputTensor(gradientRivet.getFeatureOutput())
//                                       .dataType(DataType::FP16)
//                                       .build();
//     Stub stub = Stub::Builder().network(network).inputTensor(fc2.getFeatureOutputs()[1]).build();
//
//     shared_ptr<Sgd> sgd = Sgd::Builder().network(network).initialLearningRate(0.1).decay(0.1).build();
//
//     ThorImplementation::StampedNetwork stampedNetwork;
//     int gpuNum = 0;
//     int batchSize = 32;
//     uint32_t stampsPerGpu = 1;
//     vector<Event> initDoneEvents;
//     shared_ptr<PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, false, {gpuNum}, stampsPerGpu);
//     ASSERT_TRUE(placedNetwork != nullptr);
//     ASSERT_EQ(placedNetwork->getNumStamps(), 1UL);
//     stampedNetwork = placedNetwork->getStampedNetwork(0);
//
//     // Check network structure
//     ASSERT_EQ(stampedNetwork.getInputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getOutputs().size(), 1u);
//     ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 6u);
//     ASSERT_EQ(stampedNetwork.getOtherLayers().size(), 10u);
//
//     ASSERT_EQ(stampedNetwork.getInputs()[0]->getName(), "input");
//     ASSERT_EQ(stampedNetwork.getOutputs()[0]->getName(), "output");
//
//     vector<shared_ptr<ThorImplementation::FullyConnected>> fcv;
//     vector<shared_ptr<ThorImplementation::BatchNormalization>> bn;
//     vector<shared_ptr<ThorImplementation::Relu>> r;
//     vector<shared_ptr<ThorImplementation::TensorFanout>> f;
//     vector<shared_ptr<ThorImplementation::DropOut>> d;
//
//     for (int i = 0; i < stampedNetwork.getNumTrainableLayers(); ++i) {
//         if (dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(i)) != nullptr)
//             fcv.push_back(dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(i)));
//         else if (dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getTrainableLayer(i)) != nullptr)
//             bn.push_back(dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getTrainableLayer(i)));
//         else {
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::Convolution2d>(stampedNetwork.getTrainableLayer(i)), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(stampedNetwork.getTrainableLayer(i)),
//             nullptr); ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::MultiConnectionLayer>(stampedNetwork.getTrainableLayer(i)),
//             nullptr); ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::Layer>(stampedNetwork.getTrainableLayer(i)), nullptr);
//             ASSERT_TRUE(false);
//         }
//     }
//
//     shared_ptr<ThorImplementation::GradientRivet> rivet;
//     for (int i = 0; i < stampedNetwork.getOtherLayers().size(); ++i) {
//         if (dynamic_pointer_cast<ThorImplementation::Relu>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             r.push_back(dynamic_pointer_cast<ThorImplementation::Relu>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             f.push_back(dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::DropOut>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             d.push_back(dynamic_pointer_cast<ThorImplementation::DropOut>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::GradientRivet>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             rivet = dynamic_pointer_cast<ThorImplementation::GradientRivet>(stampedNetwork.getOtherLayers()[i]);
//         else {
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::Tanh>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::MultiConnectionLayer>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::Layer>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_TRUE(false);
//         }
//     }
//
//     ASSERT_EQ(f.size(), 1u);
//     ASSERT_EQ(bn.size(), 3u);
//     ASSERT_EQ(d.size(), 4u);
//     ASSERT_EQ(r.size(), 4u);
//     ASSERT_EQ(fcv.size(), 3u);
//     ASSERT_TRUE(rivet != nullptr);
//
//     ASSERT_EQ(stampedNetwork.getInputs()[0]->getFeatureOutput().value(), f[0]->getFeatureInputs()[0].value());
//
//     ASSERT_EQ(f[0]->getFeatureOutputs()[0].value(), bn[0]->getFeatureInputs()[0].value());
//     ASSERT_EQ(bn[0]->getFeatureOutputs()[0].value(), d[0]->getFeatureInput().value());
//     ASSERT_EQ(d[0]->getFeatureOutput().value(), fcv[0]->getFeatureInputs()[0].value());
//     ASSERT_EQ(fcv[0]->getFeatureOutputs()[0].value(), r[0]->getFeatureInput().value());
//     ASSERT_EQ(r[0]->getFeatureOutput().value(), bn[2]->getFeatureInputs()[0].value());
//
//     ASSERT_EQ(bn[1]->getFeatureOutputs()[0].value(), d[1]->getFeatureInput().value());
//     ASSERT_EQ(d[1]->getFeatureOutput().value(), fcv[1]->getFeatureInputs()[0].value());
//     ASSERT_EQ(fcv[1]->getFeatureOutputs()[0].value(), r[1]->getFeatureInput().value());
//
//     //    for (uint32_t i = 0; i < 4; ++i) {
//     //        for (uint32_t j = 0; j < 3; ++j) {
//     //            for (uint32_t k = 0; k < fcv[j]->getFeatureInputs().size(); ++k) {
//     //                if (d[i]->!getFeatureOutput().has_value() || fcv[j]->!getFeatureInputs()[k].has_value())
//     //                    continue;
//     //                if (d[i]->getFeatureOutput().value() == fcv[j]->getFeatureInputs()[k].value()) {
//     //                    printf("ZZZZZZZZZZZZZ  %d == %d[%d]\n", i, j, k);
//     //                }
//     //            }
//     //        }
//     //    }
//
//     ASSERT_EQ(bn[2]->getFeatureOutputs()[0].value(), d[2]->getFeatureInput().value());
//     ASSERT_EQ(d[2]->getFeatureOutput().value(), fcv[2]->getFeatureInputs()[0].value());
//     ASSERT_EQ(fcv[2]->getFeatureOutputs()[0].value(), r[2]->getFeatureInput().value());
//
//     ASSERT_EQ(r[1]->getFeatureOutput().value(), bn[2]->getFeatureInputs()[1].value());
//     ASSERT_EQ(bn[2]->getFeatureOutputs()[1].value(), d[3]->getFeatureInput().value());
//     ASSERT_EQ(d[3]->getFeatureOutput().value(), fcv[2]->getFeatureInputs()[1].value());
//     ASSERT_EQ(fcv[2]->getFeatureOutputs()[1].value(), r[3]->getFeatureInput().value());
//
//     //    for (uint32_t i = 0; i < 4; ++i) {
//     //        if (r[i]->!getFeatureOutput().has_value()) {
//     //            printf("ZZZZZZZZZZZZZ  %d isEmpty\n", i);
//     //            continue;
//     //        }
//     //        if (r[i]->getFeatureOutput().value() == stampedNetwork.getOutputs()[0]->getFeatureInput().value()) {
//     //            printf("ZZZZZZZZZZZZZ  %d == 0\n", i);
//     //        }
//     //    }
//
//     if (r[3]->!getFeatureOutput().has_value()) {
//         EXPECT_EQ(r[2]->getFeatureOutput().value(), rivet->getFeatureInput().value());
//         EXPECT_EQ(rivet->getFeatureOutput().value(), stampedNetwork.getOutputs()[0]->getFeatureInput().value());
//     } else {
//         EXPECT_TRUE(r[2]->!getFeatureOutput().has_value());
//         EXPECT_EQ(r[3]->getFeatureOutput().value(), rivet->getFeatureInput().value());
//         EXPECT_EQ(rivet->getFeatureOutput().value(), stampedNetwork.getOutputs()[0]->getFeatureInput().value());
//     }
//
//     // Check weights initialization
//     ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
//     shared_ptr<ThorImplementation::FullyConnected> fc = fcv[0];
//     ThorImplementation::Tensor fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *weightsMem = (half *)fcWeights.getMemPtr();
//     for (uint32_t i = 0; i < 1024 * 800; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     ThorImplementation::Tensor fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     half *biasesMem = (half *)fcBiases.getMemPtr();
//     for (uint32_t i = 0; i < 800; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     fc = fcv[2];
//     fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     weightsMem = (half *)fcWeights.getMemPtr();
//     for (uint32_t i = 0; i < 800 * 200; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     biasesMem = (half *)fcBiases.getMemPtr();
//     for (uint32_t i = 0; i < 200; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     fc = fcv[1];
//     fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     weightsMem = (half *)fcWeights.getMemPtr();
//     for (uint32_t i = 0; i < 1024 * 800; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     biasesMem = (half *)fcBiases.getMemPtr();
//     for (uint32_t i = 0; i < 800; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
// }
//
// unsigned int numPresentTensors(std::vector<std::optional<ThorImplementation::Tensor>> tensors) {
//     unsigned int numPresent = 0;
//     for (auto it = tensors.rbegin(); it != tensors.rend(); ++it) {
//         if (it->has_value())
//             numPresent += 1;
//     }
//     return numPresent;
// }
//
// // FIXME: I updated the network stamping to do it in order, now need to update alexnet test
// TEST(Network, DISABLED_AlexnetIsProperlyFormed) {
//     ThorImplementation::StampedNetwork stampedNetwork;
//
//     Network alexNet = buildAlexNet();
//     int gpuNum = 0;
//     uint64_t batchSize = 128;
//     uint32_t stampsPerGpu = 1;
//     vector<Event> initDoneEvents;
//     shared_ptr<PlacedNetwork> placedNetwork = alexNet.place(batchSize, initDoneEvents, false, {gpuNum}, stampsPerGpu);
//     ASSERT_TRUE(placedNetwork != nullptr);
//     ASSERT_EQ(placedNetwork->getNumStamps(), 1UL);
//     stampedNetwork = placedNetwork->getStampedNetwork(0);
//
//     // Check network structure
//     EXPECT_EQ(stampedNetwork.getInputs().size(), 2u);
//     EXPECT_EQ(stampedNetwork.getOutputs().size(), 3u);
//     EXPECT_EQ(stampedNetwork.getNumTrainableLayers(), 13u);
//     EXPECT_EQ(stampedNetwork.getOtherLayers().size(), 30u);
//
//     shared_ptr<ThorImplementation::NetworkInput> images;
//     shared_ptr<ThorImplementation::NetworkInput> labels;
//     if (stampedNetwork.getInputs()[0]->getName() == "examples") {
//         images = stampedNetwork.getInputs()[0];
//         ASSERT_EQ(images->getName(), "examples");
//         labels = stampedNetwork.getInputs()[1];
//         ASSERT_EQ(labels->getName(), "labels");
//     } else {
//         labels = stampedNetwork.getInputs()[0];
//         ASSERT_EQ(labels->getName(), "labels");
//         images = stampedNetwork.getInputs()[1];
//         ASSERT_EQ(images->getName(), "examples");
//     }
//
//     set<string> outputs;
//     outputs.insert(stampedNetwork.getOutputs()[0]->getName());
//     outputs.insert(stampedNetwork.getOutputs()[1]->getName());
//     outputs.insert(stampedNetwork.getOutputs()[2]->getName());
//     ASSERT_EQ(outputs.size(), 3u);
//     ASSERT_EQ(outputs.count("predictions"), 1u);
//     ASSERT_EQ(outputs.count("loss"), 1u);
//     ASSERT_EQ(outputs.count("accuracy"), 1u);
//
//     shared_ptr<ThorImplementation::NetworkOutput> predictions = nullptr;
//     shared_ptr<ThorImplementation::NetworkOutput> loss = nullptr;
//     shared_ptr<ThorImplementation::NetworkOutput> accuracy = nullptr;
//
//     if (stampedNetwork.getOutputs()[0]->getName() == "predictions")
//         predictions = stampedNetwork.getOutputs()[0];
//     else if (stampedNetwork.getOutputs()[0]->getName() == "loss")
//         loss = stampedNetwork.getOutputs()[0];
//     else if (stampedNetwork.getOutputs()[0]->getName() == "accuracy")
//         accuracy = stampedNetwork.getOutputs()[0];
//
//     if (stampedNetwork.getOutputs()[1]->getName() == "predictions")
//         predictions = stampedNetwork.getOutputs()[1];
//     else if (stampedNetwork.getOutputs()[1]->getName() == "loss")
//         loss = stampedNetwork.getOutputs()[1];
//     else if (stampedNetwork.getOutputs()[1]->getName() == "accuracy")
//         accuracy = stampedNetwork.getOutputs()[1];
//
//     if (stampedNetwork.getOutputs()[2]->getName() == "predictions")
//         predictions = stampedNetwork.getOutputs()[2];
//     else if (stampedNetwork.getOutputs()[2]->getName() == "loss")
//         loss = stampedNetwork.getOutputs()[2];
//     else if (stampedNetwork.getOutputs()[2]->getName() == "accuracy")
//         accuracy = stampedNetwork.getOutputs()[2];
//
//     ASSERT_NE(predictions, nullptr);
//     ASSERT_NE(loss, nullptr);
//     ASSERT_NE(accuracy, nullptr);
//
//     vector<shared_ptr<ThorImplementation::Convolution2d>> cv;
//     for (int i = 0; i < 10; ++i) {
//         shared_ptr<ThorImplementation::Convolution2d> conv =
//             dynamic_pointer_cast<ThorImplementation::Convolution2d>(stampedNetwork.getTrainableLayer(i));
//         assert(conv != nullptr);
//         cv.push_back(conv);
//     }
//
//     shared_ptr<ThorImplementation::FullyConnected> fc0 =
//         dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(10));
//     assert(fc0 != nullptr);
//     shared_ptr<ThorImplementation::FullyConnected> fc1 =
//         dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(11));
//     assert(fc1 != nullptr);
//     shared_ptr<ThorImplementation::FullyConnected> fc2 =
//         dynamic_pointer_cast<ThorImplementation::FullyConnected>(stampedNetwork.getTrainableLayer(12));
//     assert(fc2 != nullptr);
//
//     vector<shared_ptr<ThorImplementation::TypeConversion>> tc;
//     vector<shared_ptr<ThorImplementation::Relu>> r;
//     vector<shared_ptr<ThorImplementation::Pooling>> p;
//     vector<shared_ptr<ThorImplementation::Flatten>> f;
//     vector<shared_ptr<ThorImplementation::DropOut>> d;
//     vector<shared_ptr<ThorImplementation::Softmax>> sm;
//     vector<shared_ptr<ThorImplementation::CrossEntropy>> ccl;
//     vector<shared_ptr<ThorImplementation::TensorFanout>> fo;
//     vector<shared_ptr<ThorImplementation::Concatenate>> cat;
//     vector<shared_ptr<ThorImplementation::CategoricalAccuracy>> acc;
//     vector<shared_ptr<ThorImplementation::Sigmoid>> sgm;
//     vector<shared_ptr<ThorImplementation::LossShaper>> ls;
//
//     for (uint64_t i = 0; i < stampedNetwork.getOtherLayers().size(); ++i) {
//         if (dynamic_pointer_cast<ThorImplementation::TypeConversion>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             tc.push_back(dynamic_pointer_cast<ThorImplementation::TypeConversion>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::Relu>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             r.push_back(dynamic_pointer_cast<ThorImplementation::Relu>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::Pooling>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             p.push_back(dynamic_pointer_cast<ThorImplementation::Pooling>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::Flatten>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             f.push_back(dynamic_pointer_cast<ThorImplementation::Flatten>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::DropOut>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             d.push_back(dynamic_pointer_cast<ThorImplementation::DropOut>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::CrossEntropy>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             ccl.push_back(dynamic_pointer_cast<ThorImplementation::CrossEntropy>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             fo.push_back(dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::Concatenate>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             cat.push_back(dynamic_pointer_cast<ThorImplementation::Concatenate>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::CategoricalAccuracy>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             acc.push_back(dynamic_pointer_cast<ThorImplementation::CategoricalAccuracy>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::Softmax>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             sm.push_back(dynamic_pointer_cast<ThorImplementation::Softmax>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::Sigmoid>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             sgm.push_back(dynamic_pointer_cast<ThorImplementation::Sigmoid>(stampedNetwork.getOtherLayers()[i]));
//         else if (dynamic_pointer_cast<ThorImplementation::LossShaper>(stampedNetwork.getOtherLayers()[i]) != nullptr)
//             ls.push_back(dynamic_pointer_cast<ThorImplementation::LossShaper>(stampedNetwork.getOtherLayers()[i]));
//         else {
//             printf("other layer id %ld type %s\n",
//                    stampedNetwork.getOtherLayers()[i]->getId(),
//                    stampedNetwork.getOtherLayers()[i]->getType().c_str());
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::BatchNormalization>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::Tanh>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::MultiConnectionLayer>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_EQ(dynamic_pointer_cast<ThorImplementation::Layer>(stampedNetwork.getOtherLayers()[i]), nullptr);
//             ASSERT_TRUE(false);
//         }
//     }
//
//     shared_ptr<ThorImplementation::TensorFanout> imagesFO = nullptr;
//     for (uint64_t i = 0; i < stampedNetwork.getOtherLayers().size(); ++i) {
//         if (dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i]) != nullptr) {
//             if (images->getFeatureOutput().value() ==
//                 dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i])->getFeatureInputs()[0].value())
//                 { imagesFO = dynamic_pointer_cast<ThorImplementation::TensorFanout>(stampedNetwork.getOtherLayers()[i]);
//             }
//         }
//     }
//     ASSERT_NE(imagesFO, nullptr);
//
//     ASSERT_EQ(tc.size(), 1u);
//     ASSERT_EQ(fo.size(), 3u);
//     ASSERT_EQ(r.size(), 12u);
//     ASSERT_EQ(p.size(), 6u);
//     ASSERT_EQ(cat.size(), 1u);
//     ASSERT_EQ(f.size(), 1u);
//     ASSERT_EQ(d.size(), 2u);
//     ASSERT_EQ(ccl.size(), 1u);
//     ASSERT_EQ(acc.size(), 1u);
//     ASSERT_EQ(sm.size(), 1u);
//     ASSERT_EQ(sgm.size(), 0u);
//     ASSERT_EQ(ls.size(), 1u);
//
//     // Forward
//     shared_ptr<ThorImplementation::TensorFanout> fo0;
//     shared_ptr<ThorImplementation::TensorFanout> fo1;
//     shared_ptr<ThorImplementation::TensorFanout> fo2;
//     if (labels->getFeatureOutput().value() == fo[0]->getFeatureInputs()[0].value()) {
//         fo0 = fo[0];
//         if (images->getFeatureOutput().value() == fo[1]->getFeatureInputs()[0].value()) {
//             fo1 = fo[1];
//             fo2 = fo[2];
//         } else {
//             fo1 = fo[2];
//             fo2 = fo[1];
//         }
//     } else if (labels->getFeatureOutput().value() == fo[1]->getFeatureInputs()[0].value()) {
//         fo0 = fo[1];
//         if (images->getFeatureOutput().value() == fo[0]->getFeatureInputs()[0].value()) {
//             fo1 = fo[0];
//             fo2 = fo[2];
//         } else {
//             fo1 = fo[2];
//             fo2 = fo[0];
//         }
//         fo1 = fo[0];
//     } else {
//         fo0 = fo[2];
//         if (images->getFeatureOutput().value() == fo[0]->getFeatureInputs()[0].value()) {
//             fo1 = fo[0];
//             fo2 = fo[1];
//         } else {
//             fo1 = fo[1];
//             fo2 = fo[0];
//         }
//         fo1 = fo[0];
//     }
//
//     // Fanouts
//     // Note: All tensor fanouts must be optimized away since they each have 0 or 1 populated error inputs
//     //       Optimization applies by forwarding the errorInput from the previous layer rather than instatiating one.
//     if (labels->getFeatureOutput().value() != fo0->getFeatureInputs()[0].value()) {
//         printf("labels = fo0 ? %i fo1 %i fo2 %i fo[0] %i fo[1] %i fo[2] %i\n",
//                labels->getFeatureOutput().value() == fo0->getFeatureInputs()[0].value(),
//                labels->getFeatureOutput().value() == fo1->getFeatureInputs()[0].value(),
//                labels->getFeatureOutput().value() == fo2->getFeatureInputs()[0].value(),
//                labels->getFeatureOutput().value() == fo[0]->getFeatureInputs()[0].value(),
//                labels->getFeatureOutput().value() == fo[1]->getFeatureInputs()[0].value(),
//                labels->getFeatureOutput().value() == fo[2]->getFeatureInputs()[0].value());
//         fflush(stdout);
//     }
//     ASSERT_EQ(labels->getFeatureOutput().value(), fo0->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo0->getFeatureOutputs()[0].value(), fo0->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo0->getStreams().size(), 2U);
//     ASSERT_EQ(fo0->getFeatureOutputs().size(), 1U);
//     ASSERT_EQ(images->getFeatureOutput().value(), fo1->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo1->getFeatureOutputs()[0].value(), fo1->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo1->getStreams().size(), 2U);
//     ASSERT_EQ(fo1->getFeatureOutputs().size(), 1U);
//     ASSERT_EQ(sm[0]->getFeatureOutput().value(), fo2->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo2->getFeatureOutputs()[0].value(), fo2->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo2->getStreams().size(), 3U);
//     ASSERT_EQ(fo2->getFeatureOutputs().size(), 1U);
//
//     // Conv top
//     ASSERT_EQ(images->getFeatureOutput().value(), fo1->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo1->getFeatureOutputs()[0].value(), cv[0]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[0]->getFeatureOutputs()[0].value(), r[0]->getFeatureInput().value());
//     ASSERT_EQ(r[0]->getFeatureOutput().value(), cv[1]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[1]->getFeatureOutputs()[0].value(), r[1]->getFeatureInput().value());
//     ASSERT_EQ(r[1]->getFeatureOutput().value(), p[0]->getFeatureInput().value());
//     ASSERT_EQ(p[0]->getFeatureOutput().value(), cv[2]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[2]->getFeatureOutputs()[0].value(), r[2]->getFeatureInput().value());
//     ASSERT_EQ(r[2]->getFeatureOutput().value(), p[1]->getFeatureInput().value());
//     ASSERT_EQ(p[1]->getFeatureOutput().value(), cv[3]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[3]->getFeatureOutputs()[0].value(), r[3]->getFeatureInput().value());
//     ASSERT_EQ(r[3]->getFeatureOutput().value(), cv[4]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[4]->getFeatureOutputs()[0].value(), r[4]->getFeatureInput().value());
//     ASSERT_EQ(r[4]->getFeatureOutput().value(), p[2]->getFeatureInput().value());
//
//     // Conv bottom
//     ASSERT_EQ(images->getFeatureOutput().value(), fo1->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo1->getFeatureOutputs()[0].value(), cv[5]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[5]->getFeatureOutputs()[0].value(), r[5]->getFeatureInput().value());
//     ASSERT_EQ(r[5]->getFeatureOutput().value(), cv[6]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[6]->getFeatureOutputs()[0].value(), r[6]->getFeatureInput().value());
//     ASSERT_EQ(r[6]->getFeatureOutput().value(), p[3]->getFeatureInput().value());
//     ASSERT_EQ(p[3]->getFeatureOutput().value(), cv[7]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[7]->getFeatureOutputs()[0].value(), r[7]->getFeatureInput().value());
//     ASSERT_EQ(r[7]->getFeatureOutput().value(), p[4]->getFeatureInput().value());
//     ASSERT_EQ(p[4]->getFeatureOutput().value(), cv[8]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[8]->getFeatureOutputs()[0].value(), r[8]->getFeatureInput().value());
//     ASSERT_EQ(r[8]->getFeatureOutput().value(), cv[9]->getFeatureInputs()[0].value());
//     ASSERT_EQ(cv[9]->getFeatureOutputs()[0].value(), r[9]->getFeatureInput().value());
//     ASSERT_EQ(r[9]->getFeatureOutput().value(), p[5]->getFeatureInput().value());
//
//     // Concatenate
//     ASSERT_EQ(p[2]->getFeatureOutput().value(), cat[0]->getFeatureInputs()[0].value());
//     ASSERT_EQ(p[5]->getFeatureOutput().value(), cat[0]->getFeatureInputs()[1].value());
//
//     // Fully Connected
//     ASSERT_EQ(cat[0]->getFeatureOutputs()[0].value(), d[0]->getFeatureInput().value());
//     ASSERT_EQ(d[0]->getFeatureOutput().value(), fc0->getFeatureInputs()[0].value());
//     ASSERT_EQ(fc0->getFeatureOutputs()[0].value(), r[10]->getFeatureInput().value());
//     ASSERT_EQ(r[10]->getFeatureOutput().value(), d[1]->getFeatureInput().value());
//     ASSERT_EQ(d[1]->getFeatureOutput().value(), fc1->getFeatureInputs()[0].value());
//     ASSERT_EQ(fc1->getFeatureOutputs()[0].value(), r[11]->getFeatureInput().value());
//     ASSERT_EQ(r[11]->getFeatureOutput().value(), fc2->getFeatureInputs()[0].value());
//
//     // Categorical Cross Entropy
//     ASSERT_EQ(fc2->getFeatureOutputs()[0].value(), sm[0]->getFeatureInput().value());
//     ASSERT_EQ(sm[0]->getFeatureOutput().value(), fo2->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo2->getFeatureOutputs()[0].value(), ccl[0]->getFeatureInput().value());
//     ASSERT_EQ(ccl[0]->getFeatureOutput().value(), ls[0]->getFeatureInput().value());
//     ASSERT_EQ(labels->getFeatureOutput().value(), fo0->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo0->getFeatureOutputs()[0].value(), ccl[0]->getLabelsInput().value());
//
//     // Categorical accuracy
//     ASSERT_EQ(sm[0]->getFeatureOutput().value(), fo2->getFeatureInputs()[0].value());
//     ASSERT_EQ(fo2->getFeatureInputs()[0].value(), acc[0]->getFeatureInput().value());
//     ASSERT_EQ(labels->getFeatureOutput().value(), fo0->getFeatureOutputs()[0].value());
//     ASSERT_EQ(fo0->getFeatureOutputs()[0].value(), ccl[0]->getLabelsInput().value());
//
//     // Loss Output
//     ASSERT_EQ(ls[0]->getFeatureOutput().value(), loss->getFeatureInput().value());
//
//     // Predictions output
//     ASSERT_EQ(fo2->getFeatureOutputs()[0].value(), tc[0]->getFeatureInput().value());
//     ASSERT_EQ(tc[0]->getFeatureOutput().value(), predictions->getFeatureInput().value());
//
//     // Accuracy Output
//     ASSERT_EQ(acc[0]->getFeatureOutput().value(), accuracy->getFeatureInput().value());
//
//     // Backward
//
//     // Back Prop stubs - conv top
//     ASSERT_TRUE(cv[0]->!getErrorOutputs()[0].has_value());
//     ASSERT_TRUE(cv[5]->!getErrorOutputs()[0].has_value());
//     ASSERT_EQ(fo0->getErrorInputs().size(), 2U);
//     ASSERT_EQ(numPresentTensors(fo0->getErrorInputs()), 0U);
//     ASSERT_EQ(fo0->getErrorOutputs().size(), 1U);
//     ASSERT_TRUE(fo0->!getErrorOutputs()[0].has_value());
//     ASSERT_EQ(fo1->getErrorInputs().size(), 2U);
//     ASSERT_EQ(numPresentTensors(fo1->getErrorInputs()), 0U);
//     ASSERT_EQ(fo1->getErrorOutputs().size(), 1U);
//     ASSERT_TRUE(fo1->!getErrorOutputs()[0].has_value());
//     ASSERT_EQ(fo2->getErrorInputs().size(), 3U);
//     ASSERT_EQ(numPresentTensors(fo2->getErrorInputs()), 1U);
//     ASSERT_EQ(fo2->getErrorOutputs().size(), 1U);
//     ASSERT_TRUE(fo2->getErrorOutputs()[0].has_value());
//
//     ASSERT_EQ(cv[0]->getErrorOutputs().size(), 1U);
//     ASSERT_EQ(cv[0]->getErrorInputs()[0].value(), r[0]->getErrorOutput().value());
//     ASSERT_EQ(r[0]->getErrorInput().value(), cv[1]->getErrorOutputs()[0].value());
//     ASSERT_EQ(cv[1]->getErrorInputs()[0].value(), r[1]->getErrorOutput().value());
//     ASSERT_EQ(r[1]->getErrorInput().value(), p[0]->getErrorOutput().value());
//     ASSERT_EQ(p[0]->getErrorInput().value(), cv[2]->getErrorOutputs()[0].value());
//     ASSERT_EQ(cv[2]->getErrorInputs()[0].value(), r[2]->getErrorOutput().value());
//     ASSERT_EQ(r[2]->getErrorInput().value(), p[1]->getErrorOutput().value());
//     ASSERT_EQ(p[1]->getErrorInput().value(), cv[3]->getErrorOutputs()[0].value());
//     ASSERT_EQ(cv[3]->getErrorInputs()[0].value(), r[3]->getErrorOutput().value());
//     ASSERT_EQ(r[3]->getErrorInput().value(), cv[4]->getErrorOutputs()[0].value());
//     ASSERT_EQ(cv[4]->getErrorInputs()[0].value(), r[4]->getErrorOutput().value());
//     ASSERT_EQ(r[4]->getErrorInput().value(), p[2]->getErrorOutput().value());
//     ASSERT_EQ(p[2]->getErrorInput().value(), cat[0]->getErrorOutputs()[0].value());
//
//     ASSERT_EQ(cv[5]->getErrorOutputs().size(), 1U);
//     ASSERT_EQ(cv[5]->getErrorInputs()[0].value(), r[5]->getErrorOutput().value());
//     ASSERT_EQ(r[5]->getErrorInput().value(), cv[6]->getErrorOutputs()[0].value());
//     ASSERT_EQ(cv[6]->getErrorInputs()[0].value(), r[6]->getErrorOutput().value());
//     ASSERT_EQ(r[6]->getErrorInput().value(), p[3]->getErrorOutput().value());
//     ASSERT_EQ(p[3]->getErrorInput().value(), cv[7]->getErrorOutputs()[0].value());
//     ASSERT_EQ(cv[7]->getErrorInputs()[0].value(), r[7]->getErrorOutput().value());
//     ASSERT_EQ(r[7]->getErrorInput().value(), p[4]->getErrorOutput().value());
//     ASSERT_EQ(p[4]->getErrorInput().value(), cv[8]->getErrorOutputs()[0].value());
//     ASSERT_EQ(cv[8]->getErrorInputs()[0].value(), r[8]->getErrorOutput().value());
//     ASSERT_EQ(r[8]->getErrorInput().value(), cv[9]->getErrorOutputs()[0].value());
//     ASSERT_EQ(cv[9]->getErrorInputs()[0].value(), r[9]->getErrorOutput().value());
//     ASSERT_EQ(r[9]->getErrorInput().value(), p[5]->getErrorOutput().value());
//     ASSERT_EQ(p[5]->getErrorInput().value(), cat[0]->getErrorOutputs()[1].value());
//
//     ASSERT_EQ(cat[0]->getErrorInputs().size(), 1U);
//     ASSERT_EQ(cat[0]->getErrorInputs()[0].value(), f[0]->getErrorOutput().value());
//     ASSERT_EQ(f[0]->getErrorInput().value(), d[0]->getErrorOutput().value());
//     ASSERT_EQ(d[0]->getErrorInput().value(), fc0->getErrorOutputs()[0].value());
//     ASSERT_EQ(fc0->getErrorInputs()[0].value(), r[10]->getErrorOutput().value());
//     ASSERT_EQ(r[10]->getErrorInput().value(), d[1]->getErrorOutput().value());
//     ASSERT_EQ(d[1]->getErrorInput().value(), fc1->getErrorOutputs()[0].value());
//     ASSERT_EQ(fc1->getErrorInputs()[0].value(), r[11]->getErrorOutput().value());
//     ASSERT_EQ(r[11]->getErrorInput().value(), fc2->getErrorOutputs()[0].value());
//     ASSERT_EQ(fc2->getErrorInputs()[0].value(), ccl[0]->getErrorOutput().value());
//
//     // Loss
//     ASSERT_TRUE(ccl[0]->!getErrorInput().has_value());
//     ASSERT_TRUE(ccl[0]->getErrorOutput().has_value());
//     if (fo2->getErrorInputs()[0].has_value()) {
//         ASSERT_EQ(ccl[0]->getErrorOutput().value(), fo2->getErrorInputs()[0].value());
//     }
//     if (fo2->getErrorInputs()[1].has_value()) {
//         ASSERT_EQ(ccl[0]->getErrorOutput().value(), fo2->getErrorInputs()[1].value());
//     }
//     if (fo2->getErrorInputs()[2].has_value()) {
//         ASSERT_EQ(ccl[0]->getErrorOutput().value(), fo2->getErrorInputs()[2].value());
//     }
//     ASSERT_EQ(fo2->getErrorOutputs()[0].value(), sm[0]->getErrorInput().value());
//     ASSERT_EQ(ccl[0]->getErrorOutput().value(), sm[0]->getErrorInput().value());
//     ASSERT_TRUE(sm[0]->isBackwardComputedExternally());
//     ASSERT_EQ(ccl[0]->getErrorOutput().value(), fc2->getErrorInputs()[0].value());
//
//     // Outputs
//     ASSERT_TRUE(acc[0]->!getErrorInput().has_value());
//     ASSERT_TRUE(acc[0]->!getErrorOutput().has_value());
//     ASSERT_TRUE(labels->!getErrorInput().has_value());
//     ASSERT_TRUE(labels->!getErrorOutput().has_value());
//     ASSERT_TRUE(predictions->!getErrorInput().has_value());
//     ASSERT_TRUE(predictions->!getErrorOutput().has_value());
//     ASSERT_TRUE(images->!getErrorInput().has_value());
//     ASSERT_TRUE(images->!getErrorOutput().has_value());
//
//     // Check tensor dimensions
//     // Forward
//     vector<uint64_t> expectedDimensions;
//     expectedDimensions = {batchSize, 48, 55, 55};
//     ASSERT_EQ(cv[0]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 55, 55};
//     ASSERT_EQ(cv[1]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 27, 27};
//     ASSERT_EQ(cv[2]->getFeatureInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 27, 27};
//     ASSERT_EQ(cv[2]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 13, 13};
//     ASSERT_EQ(cv[3]->getFeatureInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 13, 13};
//     ASSERT_EQ(cv[3]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 13, 13};
//     ASSERT_EQ(cv[4]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//
//     expectedDimensions = {batchSize, 48, 55, 55};
//     ASSERT_EQ(cv[5]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 55, 55};
//     ASSERT_EQ(cv[6]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 27, 27};
//     ASSERT_EQ(cv[7]->getFeatureInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 27, 27};
//     ASSERT_EQ(cv[7]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 13, 13};
//     ASSERT_EQ(cv[8]->getFeatureInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 13, 13};
//     ASSERT_EQ(cv[8]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 13, 13};
//     ASSERT_EQ(cv[9]->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//
//     expectedDimensions = {batchSize, 9216};
//     ASSERT_EQ(fc0->getFeatureInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 4096};
//     ASSERT_EQ(fc0->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 4096};
//     ASSERT_EQ(fc1->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 1000};
//     ASSERT_EQ(fc2->getFeatureOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//
//     // Backward
//     ASSERT_TRUE(cv[0]->!getErrorOutputs()[0].has_value());
//     expectedDimensions = {batchSize, 48, 55, 55};
//     ASSERT_EQ(cv[1]->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 55, 55};
//     ASSERT_EQ(cv[1]->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 27, 27};
//     ASSERT_EQ(cv[2]->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 27, 27};
//     ASSERT_EQ(cv[2]->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 13, 13};
//     ASSERT_EQ(cv[3]->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 13, 13};
//     ASSERT_EQ(cv[4]->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 13, 13};
//     ASSERT_EQ(cv[4]->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//
//     ASSERT_TRUE(cv[5]->!getErrorOutputs()[0].has_value());
//     expectedDimensions = {batchSize, 48, 55, 55};
//     ASSERT_EQ(cv[6]->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 55, 55};
//     ASSERT_EQ(cv[6]->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 27, 27};
//     ASSERT_EQ(cv[7]->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 27, 27};
//     ASSERT_EQ(cv[7]->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 13, 13};
//     ASSERT_EQ(cv[8]->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 192, 13, 13};
//     ASSERT_EQ(cv[9]->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 128, 13, 13};
//     ASSERT_EQ(cv[9]->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//
//     expectedDimensions = {batchSize, 9216};
//     ASSERT_EQ(fc0->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 4096};
//     ASSERT_EQ(fc0->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 4096};
//     ASSERT_EQ(fc1->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 4096};
//     ASSERT_EQ(fc1->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 4096};
//     ASSERT_EQ(fc2->getErrorOutputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//     expectedDimensions = {batchSize, 1000};
//     ASSERT_EQ(fc2->getErrorInputs()[0].value().getDescriptor().getDimensions(), expectedDimensions);
//
//     // Check weights initialization
//     ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
//     shared_ptr<ThorImplementation::Convolution2d> conv = cv[0];
//     ThorImplementation::Tensor convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     half *weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 11 * 11 * 3 * 48; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     ThorImplementation::Tensor convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     half *biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 48; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[1];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 5 * 5 * 48 * 128; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 128; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[2];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 3 * 3 * 128 * 192; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 192; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[3];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 3 * 3 * 192 * 192; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 192; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[4];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 3 * 3 * 192 * 128; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 128; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[5];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 11 * 11 * 3 * 48; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 48; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[6];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 5 * 5 * 48 * 128; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 128; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[7];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 3 * 3 * 128 * 192; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 192; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[8];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 3 * 3 * 192 * 192; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 192; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     conv = cv[9];
//     convWeights = conv->getWeights().clone(cpuPlacement);
//     convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     weightsMem = (half *)convWeights.getMemPtr();
//     for (uint32_t i = 0; i < 3 * 3 * 192 * 128; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     convBiases = conv->getBiases().value().clone(cpuPlacement);
//     convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
//     conv->getStreams()[0].synchronize();
//     biasesMem = (half *)convBiases.getMemPtr();
//     for (uint32_t i = 0; i < 128; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     shared_ptr<ThorImplementation::FullyConnected> fc = fc0;
//     ThorImplementation::Tensor fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     weightsMem = (half *)fcWeights.getMemPtr();
//     for (uint32_t i = 0; i < 9216 * 4096; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     ThorImplementation::Tensor fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     biasesMem = (half *)fcBiases.getMemPtr();
//     for (uint32_t i = 0; i < 4096; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     fc = fc1;
//     fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     weightsMem = (half *)fcWeights.getMemPtr();
//     for (uint32_t i = 0; i < 4096 * 4096; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     biasesMem = (half *)fcBiases.getMemPtr();
//     for (uint32_t i = 0; i < 4096; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
//
//     fc = fc2;
//     fcWeights = fc->getWeights().clone(cpuPlacement);
//     fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     weightsMem = (half *)fcWeights.getMemPtr();
//     for (uint32_t i = 0; i < 4096 * 1000; ++i) {
//         ASSERT_TRUE(weightsMem[i] >= (half)-0.1 && weightsMem[i] <= (half)0.1);
//     }
//     fcBiases = fc->getBiases().value().clone(cpuPlacement);
//     fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
//     fc->getStreams()[0].synchronize();
//     biasesMem = (half *)fcBiases.getMemPtr();
//     for (uint32_t i = 0; i < 1000; ++i) {
//         ASSERT_TRUE(biasesMem[i] >= (half)-0.1 && biasesMem[i] <= (half)0.1);
//     }
// }
#include <optional>
