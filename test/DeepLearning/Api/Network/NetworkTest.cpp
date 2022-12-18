#include "Thor.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <math.h>
#include <set>
#include <vector>

using std::set;
using std::vector;

using namespace Thor;

TEST(Network, SimplestNetworkProperlyFormed) {
    Network network;
    Tensor latestOutputTensor;
    UniformRandom::Builder uniformRandomInitializerBuilder = UniformRandom::Builder().minValue(-0.1).maxValue(0.1);

    NetworkInput networkInput =
        NetworkInput::Builder().network(network).name("input").dimensions({1024}).dataType(Tensor::DataType::FP16).build();
    latestOutputTensor = networkInput.getFeatureOutput();

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(network)
                                        .featureInput(latestOutputTensor)
                                        .numOutputFeatures(500)
                                        .hasBias(true)
                                        .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                                        .biasInitializerBuilder(uniformRandomInitializerBuilder)
                                        .noActivation()
                                        .build();
    latestOutputTensor = fullyConnected.getFeatureOutput();

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(network).name("output").inputTensor(latestOutputTensor).dataType(Tensor::DataType::FP16).build();
    Tensor networkOutputTensor = networkOutput.getFeatureOutput();

    ThorImplementation::StampedNetwork stampedNetwork;
    int gpuNum = 0;
    int batchSize = 32;
    Network::StatusCode statusCode = network.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    // Check network structure
    ASSERT_EQ(stampedNetwork.inputs.size(), 1u);
    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(),
              stampedNetwork.apiLayerToPhysicalLayer[networkInput.getId()]->getFeatureOutput().get());
    ASSERT_EQ(stampedNetwork.trainableLayers.size(), 1u);
    ThorImplementation::FullyConnected *fc =
        dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.apiLayerToPhysicalLayer[fullyConnected.getId()]);
    ASSERT_NE(fc, nullptr);
    ASSERT_EQ(stampedNetwork.inputs[0]->getName(), "input");
    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureInputs().size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureInputs()[0].get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureOutputs().size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureOutputs()[0].get(), fc->getFeatureOutputs()[0].get());
    ASSERT_EQ(stampedNetwork.outputs.size(), 1u);
    ASSERT_EQ(fc->getFeatureOutputs()[0].get(), stampedNetwork.outputs[0]->getFeatureInput().get());
    ASSERT_EQ(stampedNetwork.outputs[0]->getName(), "output");
    ASSERT_EQ(stampedNetwork.outputs[0]->getFeatureInput().get(),
              stampedNetwork.apiLayerToPhysicalLayer[networkOutput.getId()]->getFeatureInput().get());
    ASSERT_EQ(stampedNetwork.otherLayers.size(), 0u);

    // Check weights initialization
    ThorImplementation::Tensor fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 1024 * 500; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    ThorImplementation::Tensor fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *biasesMem = (half *)fcBiases.getMemPtr();
    for (uint32_t i = 0; i < 500; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    stampedNetwork.clear();
}

TEST(Network, SimplestNetworkWithGlorotUniformProperlyFormed) {
    Network network;
    Tensor latestOutputTensor;
    Glorot::Builder glorotBuilder = Glorot::Builder().mode(ThorImplementation::Glorot::Mode::UNIFORM);

    NetworkInput networkInput =
        NetworkInput::Builder().network(network).name("input").dimensions({1024}).dataType(Tensor::DataType::FP16).build();
    latestOutputTensor = networkInput.getFeatureOutput();

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(network)
                                        .featureInput(latestOutputTensor)
                                        .numOutputFeatures(500)
                                        .hasBias(true)
                                        .weightsInitializerBuilder(glorotBuilder)
                                        .biasInitializerBuilder(glorotBuilder)
                                        .noActivation()
                                        .build();
    latestOutputTensor = fullyConnected.getFeatureOutput();

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(network).name("output").inputTensor(latestOutputTensor).dataType(Tensor::DataType::FP16).build();
    Tensor networkOutputTensor = networkOutput.getFeatureOutput();

    ThorImplementation::StampedNetwork stampedNetwork;
    int gpuNum = 0;
    int batchSize = 32;
    Network::StatusCode statusCode = network.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    // Check network structure
    ASSERT_EQ(stampedNetwork.inputs.size(), 1u);
    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(),
              stampedNetwork.apiLayerToPhysicalLayer[networkInput.getId()]->getFeatureOutput().get());
    ASSERT_EQ(stampedNetwork.trainableLayers.size(), 1u);
    ThorImplementation::FullyConnected *fc =
        dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.apiLayerToPhysicalLayer[fullyConnected.getId()]);
    ASSERT_NE(fc, nullptr);
    ASSERT_EQ(stampedNetwork.inputs[0]->getName(), "input");
    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureInputs().size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureInputs()[0].get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureOutputs().size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureOutputs()[0].get(), fc->getFeatureOutputs()[0].get());
    ASSERT_EQ(stampedNetwork.outputs.size(), 1u);
    ASSERT_EQ(fc->getFeatureOutputs()[0].get(), stampedNetwork.outputs[0]->getFeatureInput().get());
    ASSERT_EQ(stampedNetwork.outputs[0]->getName(), "output");
    ASSERT_EQ(stampedNetwork.outputs[0]->getFeatureInput().get(),
              stampedNetwork.apiLayerToPhysicalLayer[networkOutput.getId()]->getFeatureInput().get());
    ASSERT_EQ(stampedNetwork.otherLayers.size(), 0u);

    // Check weights initialization
    // First check if fanIn and fanOut is correct
    uint64_t fanIn = fc->getFanIn();
    uint64_t fanOut = fc->getFanOut();
    ASSERT_EQ(fanIn, 1024U);
    ASSERT_EQ(fanOut, 500U);
    half maxValue = sqrt(6.0 / (fanIn + fanOut)) * 1.0001;
    half minValue = -1.0f * maxValue;
    ThorImplementation::Tensor fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 1024 * 500; ++i) {
        ASSERT_TRUE(weightsMem[i] >= minValue && weightsMem[i] <= maxValue);
    }
    ThorImplementation::Tensor fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *biasesMem = (half *)fcBiases.getMemPtr();
    for (uint32_t i = 0; i < 500; ++i) {
        ASSERT_TRUE(biasesMem[i] >= minValue && biasesMem[i] <= maxValue);
    }

    stampedNetwork.clear();
}

TEST(Network, SimplestNetworkWithGlorotNormalProperlyFormed) {
    Network network;
    Tensor latestOutputTensor;
    Glorot::Builder glorotBuilder = Glorot::Builder().mode(ThorImplementation::Glorot::Mode::UNIFORM);

    NetworkInput networkInput =
        NetworkInput::Builder().network(network).name("input").dimensions({1024}).dataType(Tensor::DataType::FP16).build();
    latestOutputTensor = networkInput.getFeatureOutput();

    FullyConnected fullyConnected = FullyConnected::Builder()
                                        .network(network)
                                        .featureInput(latestOutputTensor)
                                        .numOutputFeatures(500)
                                        .hasBias(true)
                                        .weightsInitializerBuilder(glorotBuilder)
                                        .biasInitializerBuilder(glorotBuilder)
                                        .noActivation()
                                        .build();
    latestOutputTensor = fullyConnected.getFeatureOutput();

    NetworkOutput networkOutput =
        NetworkOutput::Builder().network(network).name("output").inputTensor(latestOutputTensor).dataType(Tensor::DataType::FP16).build();
    Tensor networkOutputTensor = networkOutput.getFeatureOutput();

    ThorImplementation::StampedNetwork stampedNetwork;
    int gpuNum = 0;
    int batchSize = 32;
    Network::StatusCode statusCode = network.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    // Check network structure
    ASSERT_EQ(stampedNetwork.inputs.size(), 1u);
    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(),
              stampedNetwork.apiLayerToPhysicalLayer[networkInput.getId()]->getFeatureOutput().get());
    ASSERT_EQ(stampedNetwork.trainableLayers.size(), 1u);
    ThorImplementation::FullyConnected *fc =
        dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.apiLayerToPhysicalLayer[fullyConnected.getId()]);
    ASSERT_NE(fc, nullptr);
    ASSERT_EQ(stampedNetwork.inputs[0]->getName(), "input");
    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureInputs().size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureInputs()[0].get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureOutputs().size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers[0]->getFeatureOutputs()[0].get(), fc->getFeatureOutputs()[0].get());
    ASSERT_EQ(stampedNetwork.outputs.size(), 1u);
    ASSERT_EQ(fc->getFeatureOutputs()[0].get(), stampedNetwork.outputs[0]->getFeatureInput().get());
    ASSERT_EQ(stampedNetwork.outputs[0]->getName(), "output");
    ASSERT_EQ(stampedNetwork.outputs[0]->getFeatureInput().get(),
              stampedNetwork.apiLayerToPhysicalLayer[networkOutput.getId()]->getFeatureInput().get());
    ASSERT_EQ(stampedNetwork.otherLayers.size(), 0u);

    // Check weights initialization
    // First check if fanIn and fanOut is correct
    uint64_t fanIn = fc->getFanIn();
    uint64_t fanOut = fc->getFanOut();
    ASSERT_EQ(fanIn, 1024U);
    ASSERT_EQ(fanOut, 500U);
    half maxValue = sqrt(6.0 / (fanIn + fanOut)) * 1.0001;
    half minValue = -1.0f * maxValue;
    ThorImplementation::Tensor fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 1024 * 500; ++i) {
        ASSERT_TRUE(weightsMem[i] >= minValue && weightsMem[i] <= maxValue);
    }
    ThorImplementation::Tensor fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *biasesMem = (half *)fcBiases.getMemPtr();
    double totalBias = 0.0;
    for (uint32_t i = 0; i < 500; ++i) {
        totalBias += biasesMem[i];
    }
    double mean = totalBias / 500;
    double totalVariance = 0;
    for (uint32_t i = 0; i < 500; ++i) {
        double val = biasesMem[i] - mean;
        totalVariance += val * val;
    }
    double avgVariance = totalVariance / 500;
    double stdDev = sqrt(avgVariance);

    double expectedMean = 0.0;
    double expectedStdDev = sqrt(2.0 / (fanIn + fanOut));

    ASSERT_LT(abs(mean - expectedMean), 0.01);
    ASSERT_LT(abs(stdDev - expectedStdDev), 0.01);

    stampedNetwork.clear();
}

TEST(Network, SimpleNetworkWithCompoundLayerProperlyFormed) {
    Network network;
    Tensor latestOutputTensor;
    UniformRandom::Builder uniformRandomInitializerBuilder = UniformRandom::Builder().minValue(2).maxValue(3);

    Tensor networkInputTensor = NetworkInput::Builder()
                                    .network(network)
                                    .name("features")
                                    .dimensions({500})
                                    .dataType(Tensor::DataType::UINT8)
                                    .build()
                                    .getFeatureOutput();
    latestOutputTensor = networkInputTensor;
    latestOutputTensor = FullyConnected::Builder()
                             .network(network)
                             .featureInput(latestOutputTensor)
                             .numOutputFeatures(800)
                             .hasBias(false)
                             .activationBuilder(Relu::Builder())
                             .dropOut(0.5)
                             .batchNormalization()
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .build()
                             .getFeatureOutput();
    latestOutputTensor =
        DropOut::Builder().network(network).featureInput(latestOutputTensor).dropProportion(0.25).build().getFeatureOutput();
    Tensor networkOutputTensor = NetworkOutput::Builder()
                                     .network(network)
                                     .name("output")
                                     .inputTensor(latestOutputTensor)
                                     .dataType(Tensor::DataType::FP32)
                                     .build()
                                     .getFeatureOutput();

    ASSERT_EQ(networkInputTensor.getDataType(), Tensor::DataType::UINT8);
    ASSERT_EQ(networkOutputTensor.getDataType(), Tensor::DataType::FP32);

    ThorImplementation::StampedNetwork stampedNetwork;
    int gpuNum = 0;
    int batchSize = 32;
    Network::StatusCode statusCode = network.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    // Check network structure
    ASSERT_EQ(stampedNetwork.inputs.size(), 1u);
    ASSERT_EQ(stampedNetwork.outputs.size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers.size(), 2u);
    ASSERT_EQ(stampedNetwork.otherLayers.size(), 5u);

    ThorImplementation::NetworkInput *input = stampedNetwork.inputs[0];
    ASSERT_EQ(input->getName(), "features");

    ThorImplementation::NetworkOutput *output = stampedNetwork.outputs[0];
    ASSERT_EQ(output->getName(), "output");

    ThorImplementation::BatchNormalization *bn = dynamic_cast<ThorImplementation::BatchNormalization *>(stampedNetwork.trainableLayers[0]);
    assert(bn != nullptr);
    ThorImplementation::FullyConnected *fc = dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.trainableLayers[1]);
    assert(fc != nullptr);

    ThorImplementation::TypeConversion *tc8_16 = dynamic_cast<ThorImplementation::TypeConversion *>(stampedNetwork.otherLayers[0]);
    ASSERT_NE(tc8_16, nullptr);
    ThorImplementation::DropOut *dropout = dynamic_cast<ThorImplementation::DropOut *>(stampedNetwork.otherLayers[1]);
    ASSERT_NE(dropout, nullptr);
    ThorImplementation::Relu *relu = dynamic_cast<ThorImplementation::Relu *>(stampedNetwork.otherLayers[2]);
    ASSERT_NE(relu, nullptr);
    ThorImplementation::DropOut *dropout2 = dynamic_cast<ThorImplementation::DropOut *>(stampedNetwork.otherLayers[3]);
    ASSERT_NE(dropout2, nullptr);
    ThorImplementation::TypeConversion *tc16_32 = dynamic_cast<ThorImplementation::TypeConversion *>(stampedNetwork.otherLayers[4]);
    ASSERT_NE(tc16_32, nullptr);

    ASSERT_EQ(input->getFeatureOutput().get(), tc8_16->getFeatureInput().get());
    ASSERT_EQ(tc8_16->getFeatureOutput().get(), bn->getFeatureInputs()[0].get());
    ASSERT_EQ(bn->getFeatureOutputs()[0].get(), dropout->getFeatureInput().get());
    ASSERT_EQ(dropout->getFeatureOutput().get(), fc->getFeatureInputs()[0].get());
    ASSERT_EQ(fc->getFeatureOutputs()[0].get(), relu->getFeatureInput().get());
    ASSERT_EQ(relu->getFeatureOutput().get(), dropout2->getFeatureInput().get());
    ASSERT_EQ(dropout2->getFeatureOutput().get(), tc16_32->getFeatureInput().get());
    ASSERT_EQ(tc16_32->getFeatureOutput().get(), output->getFeatureInput().get());

    ThorImplementation::Tensor fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 500 * 800; ++i) {
        ASSERT_TRUE(weightsMem[i] >= 2 && weightsMem[i] <= 3);
    }

    stampedNetwork.clear();
}

TEST(Network, BranchedNetworkProperlyFormed) {
    Network network;
    Tensor latestOutputTensor;
    UniformRandom::Builder uniformRandomInitializerBuilder = UniformRandom::Builder().minValue(-0.1).maxValue(0.1);

    NetworkInput networkInput =
        NetworkInput::Builder().network(network).name("input").dimensions({1024}).dataType(Tensor::DataType::FP16).build();

    FullyConnected::Builder fc0Builder = FullyConnected::Builder()
                                             .network(network)
                                             .featureInput(networkInput.getFeatureOutput())
                                             .numOutputFeatures(800)
                                             .hasBias(true)
                                             .activationBuilder(Relu::Builder())
                                             .dropOut(0.5)
                                             .batchNormalization()
                                             .weightsInitializerBuilder(uniformRandomInitializerBuilder);
    FullyConnected fc0 = fc0Builder.build();
    FullyConnected fc1 = fc0Builder.build();

    FullyConnected fc2 = FullyConnected::Builder()
                             .network(network)
                             .featureInput(fc0.getFeatureOutput())
                             .featureInput(fc1.getFeatureOutput())
                             .numOutputFeatures(200)
                             .hasBias(true)
                             .activationBuilder(Relu::Builder())
                             .dropOut(0.5)
                             .batchNormalization()
                             .weightsInitializerBuilder(uniformRandomInitializerBuilder)
                             .build();

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(network)
                                      .name("output")
                                      .inputTensor(fc2.getFeatureOutputs()[0])
                                      .dataType(Tensor::DataType::FP16)
                                      .build();
    Stub stub = Stub::Builder().network(network).inputTensor(fc2.getFeatureOutputs()[1]).build();

    ThorImplementation::StampedNetwork stampedNetwork;
    int gpuNum = 0;
    int batchSize = 32;
    Network::StatusCode statusCode = network.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    // Check network structure
    ASSERT_EQ(stampedNetwork.inputs.size(), 1u);
    ASSERT_EQ(stampedNetwork.outputs.size(), 1u);
    ASSERT_EQ(stampedNetwork.trainableLayers.size(), 6u);
    ASSERT_EQ(stampedNetwork.otherLayers.size(), 9u);

    ASSERT_EQ(stampedNetwork.inputs[0]->getName(), "input");
    ASSERT_EQ(stampedNetwork.outputs[0]->getName(), "output");

    vector<ThorImplementation::FullyConnected *> fcv;
    vector<ThorImplementation::BatchNormalization *> bn;
    vector<ThorImplementation::Relu *> r;
    vector<ThorImplementation::TensorFanout *> f;
    vector<ThorImplementation::DropOut *> d;

    for (int i = 0; i < 6; ++i) {
        if (dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.trainableLayers[i]) != nullptr)
            fcv.push_back(dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.trainableLayers[i]));
        else if (dynamic_cast<ThorImplementation::BatchNormalization *>(stampedNetwork.trainableLayers[i]) != nullptr)
            bn.push_back(dynamic_cast<ThorImplementation::BatchNormalization *>(stampedNetwork.trainableLayers[i]));
        else {
            ASSERT_EQ(dynamic_cast<ThorImplementation::Convolution2d *>(stampedNetwork.trainableLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::TrainableWeightsBiasesLayer *>(stampedNetwork.trainableLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::MultiConnectionLayer *>(stampedNetwork.trainableLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::Layer *>(stampedNetwork.trainableLayers[i]), nullptr);
            ASSERT_TRUE(false);
        }
    }

    for (int i = 0; i < 9; ++i) {
        if (dynamic_cast<ThorImplementation::Relu *>(stampedNetwork.otherLayers[i]) != nullptr)
            r.push_back(dynamic_cast<ThorImplementation::Relu *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i]) != nullptr)
            f.push_back(dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::DropOut *>(stampedNetwork.otherLayers[i]) != nullptr)
            d.push_back(dynamic_cast<ThorImplementation::DropOut *>(stampedNetwork.otherLayers[i]));
        else {
            ASSERT_EQ(dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::BatchNormalization *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::Tanh *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::MultiConnectionLayer *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::Layer *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_TRUE(false);
        }
    }

    ASSERT_EQ(f.size(), 1u);
    ASSERT_EQ(bn.size(), 3u);
    ASSERT_EQ(d.size(), 4u);
    ASSERT_EQ(r.size(), 4u);
    ASSERT_EQ(fcv.size(), 3u);

    ASSERT_EQ(stampedNetwork.inputs[0]->getFeatureOutput().get(), f[0]->getFeatureInputs()[0].get());

    ASSERT_EQ(f[0]->getFeatureOutputs()[0].get(), bn[0]->getFeatureInputs()[0].get());
    ASSERT_EQ(bn[0]->getFeatureOutputs()[0].get(), d[0]->getFeatureInput().get());
    ASSERT_EQ(d[0]->getFeatureOutput().get(), fcv[0]->getFeatureInputs()[0].get());
    ASSERT_EQ(fcv[0]->getFeatureOutputs()[0].get(), r[0]->getFeatureInput().get());
    ASSERT_EQ(r[0]->getFeatureOutput().get(), bn[1]->getFeatureInputs()[0].get());

    ASSERT_EQ(bn[1]->getFeatureOutputs()[0].get(), d[1]->getFeatureInput().get());
    ASSERT_EQ(d[1]->getFeatureOutput().get(), fcv[1]->getFeatureInputs()[0].get());
    ASSERT_EQ(fcv[1]->getFeatureOutputs()[0].get(), r[1]->getFeatureInput().get());

    ASSERT_EQ(bn[2]->getFeatureOutputs()[0].get(), d[2]->getFeatureInput().get());
    ASSERT_EQ(d[2]->getFeatureOutput().get(), fcv[2]->getFeatureInputs()[0].get());
    ASSERT_EQ(fcv[2]->getFeatureOutputs()[0].get(), r[2]->getFeatureInput().get());

    ASSERT_EQ(r[2]->getFeatureOutput().get(), bn[1]->getFeatureInputs()[1].get());
    ASSERT_EQ(bn[1]->getFeatureOutputs()[1].get(), d[3]->getFeatureInput().get());
    ASSERT_EQ(d[3]->getFeatureOutput().get(), fcv[1]->getFeatureInputs()[1].get());
    ASSERT_EQ(fcv[1]->getFeatureOutputs()[1].get(), r[3]->getFeatureInput().get());

    if (r[3]->getFeatureOutput().isEmpty()) {
        ASSERT_EQ(r[1]->getFeatureOutput().get(), stampedNetwork.outputs[0]->getFeatureInput().get());
    } else {
        ASSERT_TRUE(r[1]->getFeatureOutput().isEmpty());
        ASSERT_EQ(r[3]->getFeatureOutput().get(), stampedNetwork.outputs[0]->getFeatureInput().get());
    }

    // Check weights initialization
    ThorImplementation::FullyConnected *fc = fcv[0];
    ThorImplementation::Tensor fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 1024 * 800; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    ThorImplementation::Tensor fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    half *biasesMem = (half *)fcBiases.getMemPtr();
    for (uint32_t i = 0; i < 800; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    fc = fcv[1];
    fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 800 * 200; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    biasesMem = (half *)fcBiases.getMemPtr();
    for (uint32_t i = 0; i < 200; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    fc = fcv[2];
    fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 1024 * 800; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    biasesMem = (half *)fcBiases.getMemPtr();
    for (uint32_t i = 0; i < 800; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    stampedNetwork.clear();
}

TEST(Network, AlexnetIsProperlyFormed) {
    ThorImplementation::StampedNetwork stampedNetwork;

    Network alexNet = buildAlexNet();
    int gpuNum = 0;
    uint64_t batchSize = 128;
    Network::StatusCode statusCode = alexNet.stampNetwork(gpuNum, batchSize, stampedNetwork);
    ASSERT_EQ(statusCode, Network::StatusCode::SUCCESS);
    stampedNetwork.initialize();

    // Check network structure
    EXPECT_EQ(stampedNetwork.inputs.size(), 2u);
    EXPECT_EQ(stampedNetwork.outputs.size(), 3u);
    EXPECT_EQ(stampedNetwork.trainableLayers.size(), 13u);
    EXPECT_EQ(stampedNetwork.otherLayers.size(), 30u);

    ThorImplementation::NetworkInput *images;
    ThorImplementation::NetworkInput *labels;
    if (stampedNetwork.inputs[0]->getName() == "examples") {
        images = stampedNetwork.inputs[0];
        ASSERT_EQ(images->getName(), "examples");
        labels = stampedNetwork.inputs[1];
        ASSERT_EQ(labels->getName(), "labels");
    } else {
        labels = stampedNetwork.inputs[0];
        ASSERT_EQ(labels->getName(), "labels");
        images = stampedNetwork.inputs[1];
        ASSERT_EQ(images->getName(), "examples");
    }

    set<string> outputs;
    outputs.insert(stampedNetwork.outputs[0]->getName());
    outputs.insert(stampedNetwork.outputs[1]->getName());
    outputs.insert(stampedNetwork.outputs[2]->getName());
    ASSERT_EQ(outputs.size(), 3u);
    ASSERT_EQ(outputs.count("predictions"), 1u);
    ASSERT_EQ(outputs.count("loss"), 1u);
    ASSERT_EQ(outputs.count("accuracy"), 1u);

    ThorImplementation::NetworkOutput *predictions = nullptr;
    ThorImplementation::NetworkOutput *loss = nullptr;
    ThorImplementation::NetworkOutput *accuracy = nullptr;

    if (stampedNetwork.outputs[0]->getName() == "predictions")
        predictions = stampedNetwork.outputs[0];
    else if (stampedNetwork.outputs[0]->getName() == "loss")
        loss = stampedNetwork.outputs[0];
    else if (stampedNetwork.outputs[0]->getName() == "accuracy")
        accuracy = stampedNetwork.outputs[0];

    if (stampedNetwork.outputs[1]->getName() == "predictions")
        predictions = stampedNetwork.outputs[1];
    else if (stampedNetwork.outputs[1]->getName() == "loss")
        loss = stampedNetwork.outputs[1];
    else if (stampedNetwork.outputs[1]->getName() == "accuracy")
        accuracy = stampedNetwork.outputs[1];

    if (stampedNetwork.outputs[2]->getName() == "predictions")
        predictions = stampedNetwork.outputs[2];
    else if (stampedNetwork.outputs[2]->getName() == "loss")
        loss = stampedNetwork.outputs[2];
    else if (stampedNetwork.outputs[2]->getName() == "accuracy")
        accuracy = stampedNetwork.outputs[2];

    ASSERT_NE(predictions, nullptr);
    ASSERT_NE(loss, nullptr);
    ASSERT_NE(accuracy, nullptr);

    vector<ThorImplementation::Convolution2d *> cv;
    for (int i = 0; i < 10; ++i) {
        ThorImplementation::Convolution2d *conv = dynamic_cast<ThorImplementation::Convolution2d *>(stampedNetwork.trainableLayers[i]);
        assert(conv != nullptr);
        cv.push_back(conv);
    }

    ThorImplementation::FullyConnected *fc0 = dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.trainableLayers[10]);
    assert(fc0 != nullptr);
    ThorImplementation::FullyConnected *fc1 = dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.trainableLayers[11]);
    assert(fc1 != nullptr);
    ThorImplementation::FullyConnected *fc2 = dynamic_cast<ThorImplementation::FullyConnected *>(stampedNetwork.trainableLayers[12]);
    assert(fc2 != nullptr);

    vector<ThorImplementation::TypeConversion *> tc;
    vector<ThorImplementation::Relu *> r;
    vector<ThorImplementation::Pooling *> p;
    vector<ThorImplementation::Flatten *> f;
    vector<ThorImplementation::DropOut *> d;
    vector<ThorImplementation::Softmax *> sm;
    vector<ThorImplementation::CrossEntropy *> ccl;
    vector<ThorImplementation::TensorFanout *> fo;
    vector<ThorImplementation::Concatenate *> cat;
    vector<ThorImplementation::CategoricalAccuracy *> acc;

    for (uint64_t i = 0; i < stampedNetwork.otherLayers.size(); ++i) {
        if (dynamic_cast<ThorImplementation::TypeConversion *>(stampedNetwork.otherLayers[i]) != nullptr)
            tc.push_back(dynamic_cast<ThorImplementation::TypeConversion *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::Relu *>(stampedNetwork.otherLayers[i]) != nullptr)
            r.push_back(dynamic_cast<ThorImplementation::Relu *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::Pooling *>(stampedNetwork.otherLayers[i]) != nullptr)
            p.push_back(dynamic_cast<ThorImplementation::Pooling *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::Flatten *>(stampedNetwork.otherLayers[i]) != nullptr)
            f.push_back(dynamic_cast<ThorImplementation::Flatten *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::DropOut *>(stampedNetwork.otherLayers[i]) != nullptr)
            d.push_back(dynamic_cast<ThorImplementation::DropOut *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::CrossEntropy *>(stampedNetwork.otherLayers[i]) != nullptr)
            ccl.push_back(dynamic_cast<ThorImplementation::CrossEntropy *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i]) != nullptr)
            fo.push_back(dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::Concatenate *>(stampedNetwork.otherLayers[i]) != nullptr)
            cat.push_back(dynamic_cast<ThorImplementation::Concatenate *>(stampedNetwork.otherLayers[i]));
        else if (dynamic_cast<ThorImplementation::CategoricalAccuracy *>(stampedNetwork.otherLayers[i]) != nullptr)
            acc.push_back(dynamic_cast<ThorImplementation::CategoricalAccuracy *>(stampedNetwork.otherLayers[i]));
        else {
            ASSERT_EQ(dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::BatchNormalization *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::Tanh *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::MultiConnectionLayer *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_EQ(dynamic_cast<ThorImplementation::Layer *>(stampedNetwork.otherLayers[i]), nullptr);
            ASSERT_TRUE(false);
        }
    }

    ThorImplementation::TensorFanout *imagesFO = nullptr;
    for (uint64_t i = 0; i < stampedNetwork.otherLayers.size(); ++i) {
        if (dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i]) != nullptr) {
            if (images->getFeatureOutput().get() ==
                dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i])->getFeatureInputs()[0].get()) {
                imagesFO = dynamic_cast<ThorImplementation::TensorFanout *>(stampedNetwork.otherLayers[i]);
            }
        }
    }
    ASSERT_NE(imagesFO, nullptr);

    ASSERT_EQ(tc.size(), 0u);
    ASSERT_EQ(fo.size(), 3u);
    ASSERT_EQ(r.size(), 12u);
    ASSERT_EQ(p.size(), 6u);
    ASSERT_EQ(cat.size(), 1u);
    ASSERT_EQ(f.size(), 1u);
    ASSERT_EQ(d.size(), 2u);
    ASSERT_EQ(ccl.size(), 1u);
    ASSERT_EQ(acc.size(), 1u);

    // Forward
    ASSERT_EQ(images->getFeatureOutput().get(), imagesFO->getFeatureInputs()[0].get());
    ASSERT_EQ(imagesFO->getFeatureOutputs()[0].get(), cv[0]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[0]->getFeatureOutputs()[0].get(), r[0]->getFeatureInput().get());
    ASSERT_EQ(r[0]->getFeatureOutput().get(), cv[1]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[1]->getFeatureOutputs()[0].get(), r[1]->getFeatureInput().get());
    ASSERT_EQ(r[1]->getFeatureOutput().get(), p[0]->getFeatureInput().get());
    ASSERT_EQ(p[0]->getFeatureOutput().get(), cv[2]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[2]->getFeatureOutputs()[0].get(), r[2]->getFeatureInput().get());
    ASSERT_EQ(r[2]->getFeatureOutput().get(), p[1]->getFeatureInput().get());
    ASSERT_EQ(p[1]->getFeatureOutput().get(), cv[3]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[3]->getFeatureOutputs()[0].get(), r[3]->getFeatureInput().get());
    ASSERT_EQ(r[3]->getFeatureOutput().get(), cv[4]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[4]->getFeatureOutputs()[0].get(), r[4]->getFeatureInput().get());
    ASSERT_EQ(r[4]->getFeatureOutput().get(), p[2]->getFeatureInput().get());
    ASSERT_EQ(p[2]->getFeatureOutput().get(), cat[0]->getFeatureInputs()[0].get());

    ASSERT_EQ(imagesFO->getFeatureInputs()[0].get(), cv[5]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[5]->getFeatureOutputs()[0].get(), r[5]->getFeatureInput().get());
    ASSERT_EQ(r[5]->getFeatureOutput().get(), cv[6]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[6]->getFeatureOutputs()[0].get(), r[6]->getFeatureInput().get());
    ASSERT_EQ(r[6]->getFeatureOutput().get(), p[3]->getFeatureInput().get());
    ASSERT_EQ(p[3]->getFeatureOutput().get(), cv[7]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[7]->getFeatureOutputs()[0].get(), r[7]->getFeatureInput().get());
    ASSERT_EQ(r[7]->getFeatureOutput().get(), p[4]->getFeatureInput().get());
    ASSERT_EQ(p[4]->getFeatureOutput().get(), cv[8]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[8]->getFeatureOutputs()[0].get(), r[8]->getFeatureInput().get());
    ASSERT_EQ(r[8]->getFeatureOutput().get(), cv[9]->getFeatureInputs()[0].get());
    ASSERT_EQ(cv[9]->getFeatureOutputs()[0].get(), r[9]->getFeatureInput().get());
    ASSERT_EQ(r[9]->getFeatureOutput().get(), p[5]->getFeatureInput().get());
    ASSERT_EQ(p[5]->getFeatureOutput().get(), cat[0]->getFeatureInputs()[1].get());

    ASSERT_EQ(cat[0]->getFeatureOutputs()[0].get(), d[0]->getFeatureInput().get());
    ASSERT_EQ(d[0]->getFeatureOutput().get(), fc0->getFeatureInputs()[0].get());
    ASSERT_EQ(fc0->getFeatureOutputs()[0].get(), r[10]->getFeatureInput().get());
    ASSERT_EQ(r[10]->getFeatureOutput().get(), d[1]->getFeatureInput().get());
    ASSERT_EQ(d[1]->getFeatureOutput().get(), fc1->getFeatureInputs()[0].get());
    ASSERT_EQ(fc1->getFeatureOutputs()[0].get(), r[11]->getFeatureInput().get());
    ASSERT_EQ(r[11]->getFeatureOutput().get(), fc2->getFeatureInputs()[0].get());
    ASSERT_EQ(fc2->getFeatureOutputs()[0].get(), ccl[0]->getFeatureInput().get());

    ASSERT_EQ(labels->getFeatureOutput().get(), ccl[0]->getLabelsInput().get());
    ASSERT_EQ(ccl[0]->getFeatureOutput().get(), predictions->getFeatureInput().get());
    ASSERT_EQ(ccl[0]->getLossOutput().get(), loss->getFeatureInput().get());

    ASSERT_EQ(ccl[0]->getFeatureOutput().get(), acc[0]->getFeatureInput().get());
    ASSERT_EQ(labels->getFeatureOutput().get(), acc[0]->getLabelsInput().get());
    ASSERT_EQ(acc[0]->getFeatureOutput().get(), accuracy->getFeatureInput().get());

    // Backward
    ASSERT_TRUE(fo[0]->getErrorOutputs()[0].isEmpty());
    ASSERT_TRUE(fo[0]->getErrorInputs()[0].isEmpty());
    ASSERT_TRUE(cv[0]->getErrorOutputs()[0].isEmpty());
    ASSERT_EQ(cv[0]->getErrorOutputs().size(), 1U);
    ASSERT_EQ(cv[0]->getErrorInputs()[0].get(), r[0]->getErrorOutput().get());
    ASSERT_EQ(r[0]->getErrorInput().get(), cv[1]->getErrorOutputs()[0].get());
    ASSERT_EQ(cv[1]->getErrorInputs()[0].get(), r[1]->getErrorOutput().get());
    ASSERT_EQ(r[1]->getErrorInput().get(), p[0]->getErrorOutput().get());
    ASSERT_EQ(p[0]->getErrorInput().get(), cv[2]->getErrorOutputs()[0].get());
    ASSERT_EQ(cv[2]->getErrorInputs()[0].get(), r[2]->getErrorOutput().get());
    ASSERT_EQ(r[2]->getErrorInput().get(), p[1]->getErrorOutput().get());
    ASSERT_EQ(p[1]->getErrorInput().get(), cv[3]->getErrorOutputs()[0].get());
    ASSERT_EQ(cv[3]->getErrorInputs()[0].get(), r[3]->getErrorOutput().get());
    ASSERT_EQ(r[3]->getErrorInput().get(), cv[4]->getErrorOutputs()[0].get());
    ASSERT_EQ(cv[4]->getErrorInputs()[0].get(), r[4]->getErrorOutput().get());
    ASSERT_EQ(r[4]->getErrorInput().get(), p[2]->getErrorOutput().get());
    ASSERT_EQ(p[2]->getErrorInput().get(), cat[0]->getErrorOutputs()[0].get());

    ASSERT_TRUE(fo[0]->getErrorInputs()[1].isEmpty());
    ASSERT_TRUE(cv[5]->getErrorOutputs()[0].isEmpty());
    ASSERT_EQ(cv[5]->getErrorOutputs().size(), 1U);
    ASSERT_EQ(cv[5]->getErrorInputs()[0].get(), r[5]->getErrorOutput().get());
    ASSERT_EQ(r[5]->getErrorInput().get(), cv[6]->getErrorOutputs()[0].get());
    ASSERT_EQ(cv[6]->getErrorInputs()[0].get(), r[6]->getErrorOutput().get());
    ASSERT_EQ(r[6]->getErrorInput().get(), p[3]->getErrorOutput().get());
    ASSERT_EQ(p[3]->getErrorInput().get(), cv[7]->getErrorOutputs()[0].get());
    ASSERT_EQ(cv[7]->getErrorInputs()[0].get(), r[7]->getErrorOutput().get());
    ASSERT_EQ(r[7]->getErrorInput().get(), p[4]->getErrorOutput().get());
    ASSERT_EQ(p[4]->getErrorInput().get(), cv[8]->getErrorOutputs()[0].get());
    ASSERT_EQ(cv[8]->getErrorInputs()[0].get(), r[8]->getErrorOutput().get());
    ASSERT_EQ(r[8]->getErrorInput().get(), cv[9]->getErrorOutputs()[0].get());
    ASSERT_EQ(cv[9]->getErrorInputs()[0].get(), r[9]->getErrorOutput().get());
    ASSERT_EQ(r[9]->getErrorInput().get(), p[5]->getErrorOutput().get());
    ASSERT_EQ(p[5]->getErrorInput().get(), cat[0]->getErrorOutputs()[1].get());

    ASSERT_EQ(cat[0]->getErrorInputs().size(), 1U);
    ASSERT_EQ(cat[0]->getErrorInputs()[0].get(), f[0]->getErrorOutput().get());
    ASSERT_EQ(f[0]->getErrorInput().get(), d[0]->getErrorOutput().get());
    ASSERT_EQ(d[0]->getErrorInput().get(), fc0->getErrorOutputs()[0].get());
    ASSERT_EQ(fc0->getErrorInputs()[0].get(), r[10]->getErrorOutput().get());
    ASSERT_EQ(r[10]->getErrorInput().get(), d[1]->getErrorOutput().get());
    ASSERT_EQ(d[1]->getErrorInput().get(), fc1->getErrorOutputs()[0].get());
    ASSERT_EQ(fc1->getErrorInputs()[0].get(), r[11]->getErrorOutput().get());
    ASSERT_EQ(r[11]->getErrorInput().get(), fc2->getErrorOutputs()[0].get());
    ASSERT_EQ(fc2->getErrorInputs()[0].get(), ccl[0]->getErrorOutput().get());

    ASSERT_TRUE(ccl[0]->getErrorInput().isEmpty());
    ASSERT_TRUE(labels->getErrorInput().isEmpty());
    ASSERT_TRUE(predictions->getErrorInput().isEmpty());

    // Check tensor dimensions
    // Forward
    vector<uint64_t> expectedDimensions;
    expectedDimensions = {batchSize, 48, 55, 55};
    ASSERT_EQ(cv[0]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 55, 55};
    ASSERT_EQ(cv[1]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 27, 27};
    ASSERT_EQ(cv[2]->getFeatureInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 27, 27};
    ASSERT_EQ(cv[2]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 13, 13};
    ASSERT_EQ(cv[3]->getFeatureInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 13, 13};
    ASSERT_EQ(cv[3]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 13, 13};
    ASSERT_EQ(cv[4]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);

    expectedDimensions = {batchSize, 48, 55, 55};
    ASSERT_EQ(cv[5]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 55, 55};
    ASSERT_EQ(cv[6]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 27, 27};
    ASSERT_EQ(cv[7]->getFeatureInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 27, 27};
    ASSERT_EQ(cv[7]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 13, 13};
    ASSERT_EQ(cv[8]->getFeatureInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 13, 13};
    ASSERT_EQ(cv[8]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 13, 13};
    ASSERT_EQ(cv[9]->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);

    expectedDimensions = {batchSize, 9216};
    ASSERT_EQ(fc0->getFeatureInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 4096};
    ASSERT_EQ(fc0->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 4096};
    ASSERT_EQ(fc1->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 1000};
    ASSERT_EQ(fc2->getFeatureOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);

    // Backward
    ASSERT_TRUE(cv[0]->getErrorOutputs()[0].isEmpty());
    expectedDimensions = {batchSize, 48, 55, 55};
    ASSERT_EQ(cv[1]->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 55, 55};
    ASSERT_EQ(cv[1]->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 27, 27};
    ASSERT_EQ(cv[2]->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 27, 27};
    ASSERT_EQ(cv[2]->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 13, 13};
    ASSERT_EQ(cv[3]->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 13, 13};
    ASSERT_EQ(cv[4]->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 13, 13};
    ASSERT_EQ(cv[4]->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);

    ASSERT_TRUE(cv[5]->getErrorOutputs()[0].isEmpty());
    expectedDimensions = {batchSize, 48, 55, 55};
    ASSERT_EQ(cv[6]->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 55, 55};
    ASSERT_EQ(cv[6]->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 27, 27};
    ASSERT_EQ(cv[7]->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 27, 27};
    ASSERT_EQ(cv[7]->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 13, 13};
    ASSERT_EQ(cv[8]->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 192, 13, 13};
    ASSERT_EQ(cv[9]->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 128, 13, 13};
    ASSERT_EQ(cv[9]->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);

    expectedDimensions = {batchSize, 9216};
    ASSERT_EQ(fc0->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 4096};
    ASSERT_EQ(fc0->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 4096};
    ASSERT_EQ(fc1->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 4096};
    ASSERT_EQ(fc1->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 4096};
    ASSERT_EQ(fc2->getErrorOutputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);
    expectedDimensions = {batchSize, 1000};
    ASSERT_EQ(fc2->getErrorInputs()[0].get().getDescriptor().getDimensions(), expectedDimensions);

    // Check weights initialization
    ThorImplementation::Convolution2d *conv = cv[0];
    ThorImplementation::Tensor convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    half *weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 11 * 11 * 3 * 48; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    ThorImplementation::Tensor convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    half *biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 48; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[1];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 5 * 5 * 48 * 128; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 128; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[2];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 3 * 3 * 128 * 192; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 192; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[3];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 3 * 3 * 192 * 192; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 192; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[4];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 3 * 3 * 192 * 128; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 128; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[5];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 11 * 11 * 3 * 48; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 48; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[6];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 5 * 5 * 48 * 128; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 128; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[7];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 3 * 3 * 128 * 192; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 192; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[8];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 3 * 3 * 192 * 192; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 192; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    conv = cv[9];
    convWeights = conv->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convWeights.copyFromAsync(conv->getWeights(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    weightsMem = (half *)convWeights.getMemPtr();
    for (uint32_t i = 0; i < 3 * 3 * 192 * 128; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    convBiases = conv->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    convBiases.copyFromAsync(conv->getBiases(), conv->getStreams()[0]);
    conv->getStreams()[0].synchronize();
    biasesMem = (half *)convBiases.getMemPtr();
    for (uint32_t i = 0; i < 128; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    ThorImplementation::FullyConnected *fc = fc0;
    ThorImplementation::Tensor fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 9216 * 4096; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    ThorImplementation::Tensor fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    biasesMem = (half *)fcBiases.getMemPtr();
    for (uint32_t i = 0; i < 4096; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    fc = fc1;
    fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 4096 * 4096; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    biasesMem = (half *)fcBiases.getMemPtr();
    for (uint32_t i = 0; i < 4096; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    fc = fc2;
    fcWeights = fc->getWeights().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcWeights.copyFromAsync(fc->getWeights(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    weightsMem = (half *)fcWeights.getMemPtr();
    for (uint32_t i = 0; i < 4096 * 1000; ++i) {
        ASSERT_TRUE(weightsMem[i] >= -0.1 && weightsMem[i] <= 0.1);
    }
    fcBiases = fc->getBiases().get().clone(ThorImplementation::TensorPlacement::MemDevices::CPU);
    fcBiases.copyFromAsync(fc->getBiases(), fc->getStreams()[0]);
    fc->getStreams()[0].synchronize();
    biasesMem = (half *)fcBiases.getMemPtr();
    for (uint32_t i = 0; i < 1000; ++i) {
        ASSERT_TRUE(biasesMem[i] >= -0.1 && biasesMem[i] <= 0.1);
    }

    stampedNetwork.clear();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
