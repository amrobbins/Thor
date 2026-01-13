#include "DeepLearning/Api/ExampleNetworks/AlexNet.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/Convolution2d.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Loss/MeanSquaredError.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Activation/Sigmoid.h"

#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <DeepLearning/Api/Layers/Loss/MeanAbsoluteError.h>
#include <math.h>

#include <set>
#include <vector>

using namespace std;
using namespace Thor;
using json = nlohmann::json;

inline float randFloat(float lo = 0.0f, float hi = 1.0f) {
    static uint32_t state = static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^ 0x9E3779B9u;

    state = state * 1664525u + 1013904223u;
    const float t = (state >> 8) * (1.0f / 16777216.0f);
    return lo + (hi - lo) * t;
}

Tensor verifyNetworkInputApiLevel(shared_ptr<NetworkInput> layer,
                                  vector<uint64_t> inputDimensions,
                                  Tensor::DataType dataType,
                                  const string &networkInputName) {
    EXPECT_EQ(layer->getName(), networkInputName);
    EXPECT_EQ(layer->getDimensions(), inputDimensions);
    EXPECT_EQ(layer->getDataType(), dataType);
    return layer->getFeatureOutput();
}

void verifySgdApiLevel(shared_ptr<Sgd> originalSgd, shared_ptr<Sgd> attachedSgd) {
    ASSERT_EQ(attachedSgd->getDecay(), originalSgd->getDecay());
    ASSERT_EQ(attachedSgd->getEpoch(), originalSgd->getEpoch());
    ASSERT_EQ(attachedSgd->getInitialLearningRate(), originalSgd->getInitialLearningRate());
    ASSERT_EQ(attachedSgd->getMomentum(), originalSgd->getMomentum());
    ASSERT_EQ(attachedSgd->getUseNesterovMomentum(), originalSgd->getUseNesterovMomentum());
}

Tensor verifyBatchNormApiLevel(
    shared_ptr<BatchNormalization> layer, Tensor inputTensor, double epsilon, double exponentialRunningAverageFactor, shared_ptr<Sgd> sgd) {
    EXPECT_EQ(layer->getFeatureInput().get(), inputTensor);
    float layerEpsilon = layer->getEpsilon();
    EXPECT_EQ(layerEpsilon, epsilon);
    float layerExponentialRunningAverageFactor = layer->getExponentialRunningAverageFactor();
    EXPECT_EQ(layerExponentialRunningAverageFactor, exponentialRunningAverageFactor);
    verifySgdApiLevel(sgd, dynamic_pointer_cast<Sgd>(layer->getOptimizer()));

    // ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    // ThorImplementation::TensorDescriptor descriptor(ThorImplementation::TensorDescriptor::DataType::FP16, inputTensor.getDimensions());
    // Stream stream(0);

    return layer->getFeatureOutput();
}

Tensor verifyDropOutApiLevel(shared_ptr<DropOut> layer, Tensor inputTensor, float dropProportion) {
    EXPECT_EQ(layer->getFeatureInput().get(), inputTensor);
    EXPECT_EQ(layer->getDropProportion(), dropProportion);
    return layer->getFeatureOutput();
}

Tensor verifyFullyConnectedApiLevel(shared_ptr<FullyConnected> layer, Tensor inputTensor, uint64_t numOutputFeatures) {
    EXPECT_EQ(layer->getFeatureInput().get(), inputTensor);
    vector<uint64_t> outputDimensions = {numOutputFeatures};
    EXPECT_EQ(layer->getFeatureOutput().get().getDimensions(), outputDimensions);
    return layer->getFeatureOutput();
}

Tensor verifyReluApiLevel(shared_ptr<Relu> layer, Tensor inputTensor) {
    EXPECT_EQ(layer->getFeatureInput().get(), inputTensor);
    return layer->getFeatureOutput();
}

Tensor verifyMeanAbsoluteErrorApiLevel(shared_ptr<MeanAbsoluteError> layer, Tensor predictionsInput, Tensor labelsInput) {
    EXPECT_EQ(layer->getPredictions(), predictionsInput);
    EXPECT_EQ(layer->getLabels(), labelsInput);
    return layer->getLoss();
}

void verifyNetworkOutputApiLevel(shared_ptr<NetworkOutput> layer, Tensor inputTensor) {
    EXPECT_EQ(layer->getFeatureInput().get(), inputTensor);
}

TEST(Network, SaveLoadRoundTripUnstamped) {
    srand(time(nullptr));

    Network initialNetwork;

    Tensor::DataType dataType = Tensor::DataType::FP16;

    vector<uint64_t> inputDimensions = {1UL + (rand() % 16)};

    uint32_t numOutputFeatures = 1 + (rand() % 1000);
    bool hasBias = rand() % 2;
    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
    bool useBatchNorm = rand() % 2;
    bool useRelu = rand() % 2;

    double epsilon = randFloat(0.1, 0.3);
    double exponentialRunningAverageFactor = randFloat(0.4, 0.5);

    const string networkInputName = "testInput";
    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name(networkInputName).dimensions(inputDimensions).dataType(dataType).build();

    shared_ptr<Initializer> uniformRandomPositive = UniformRandom::Builder().minValue(0.2).maxValue(3.0).build();
    shared_ptr<Initializer> uniformRandomNegative = UniformRandom::Builder().minValue(-3.0).maxValue(-1.0).build();

    FullyConnected::Builder fullyConnectedBuilder = FullyConnected::Builder()
                                                        .network(initialNetwork)
                                                        .featureInput(networkInput.getFeatureOutput())
                                                        .numOutputFeatures(numOutputFeatures)
                                                        .weightsInitializer(uniformRandomPositive)
                                                        .biasInitializer(uniformRandomNegative)
                                                        .hasBias(hasBias)
                                                        .dropOut(dropProportion);
    if (useBatchNorm) {
        fullyConnectedBuilder.batchNormalization(exponentialRunningAverageFactor, epsilon);
    }
    if (useRelu) {
        shared_ptr<Activation> relu = Relu::Builder().build();
        fullyConnectedBuilder.activation(relu);
    } else {
        fullyConnectedBuilder.noActivation();
    }

    FullyConnected fullyConnected = fullyConnectedBuilder.build();

    Tensor logits = fullyConnected.getFeatureOutputs()[0];
    uint32_t numClasses = logits.getDimensions()[0];
    const string labelsInputName = "labelsInput";
    NetworkInput labelsInput =
        NetworkInput::Builder().network(initialNetwork).name(labelsInputName).dimensions({numClasses}).dataType(dataType).build();

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

    bool saveOptimizerState = rand() % 2;
    initialNetwork.save("TestModel", "/tmp", true, saveOptimizerState);

    ///////////////////////////////
    // Load
    ///////////////////////////////
    Network newNetwork;
    newNetwork.load("TestModel", "/tmp");

    // Ensure all the expected layers are in the network and properly connected
    uint32_t expectedNumLayers = 5;
    if (useBatchNorm)
        expectedNumLayers += 1;
    if (dropProportion != 0.0f)
        expectedNumLayers += 1;
    if (useRelu)
        expectedNumLayers += 1;
    uint32_t actualNumLayers = newNetwork.getNumLayers();
    ASSERT_EQ(actualNumLayers, expectedNumLayers);

    uint32_t featureInputCount = 0;
    uint32_t labelsInputCount = 0;
    uint32_t batchNormCount = 0;
    uint32_t dropOutCount = 0;
    uint32_t fullyConnectedCount = 0;
    uint32_t reluCount = 0;
    uint32_t meanAbsoluteErrorCount = 0;
    uint32_t outputCount = 0;

    Tensor featureInputOutput;
    Tensor labelsInputOutput;
    Tensor batchNormOutput;
    Tensor dropOutOutput;
    Tensor fullyConnectedOutput;
    Tensor reluOutput;
    Tensor lossOutput;
    for (uint32_t i = 0; i < actualNumLayers; ++i) {
        shared_ptr<Layer> layer = newNetwork.getLayer(i);
        shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        shared_ptr<BatchNormalization> batchNorm = dynamic_pointer_cast<BatchNormalization>(layer);
        shared_ptr<DropOut> dropOut = dynamic_pointer_cast<DropOut>(layer);
        shared_ptr<FullyConnected> fullyConnected = dynamic_pointer_cast<FullyConnected>(layer);
        shared_ptr<Relu> relu = dynamic_pointer_cast<Relu>(layer);
        shared_ptr<MeanAbsoluteError> meanAbsoluteError = dynamic_pointer_cast<MeanAbsoluteError>(layer);
        shared_ptr<NetworkOutput> networkOutput = dynamic_pointer_cast<NetworkOutput>(layer);
        if (networkInput) {
            if (networkInput->getName() == networkInputName) {
                featureInputCount += 1;
                featureInputOutput = verifyNetworkInputApiLevel(networkInput, inputDimensions, dataType, networkInputName);
            } else {
                ASSERT_EQ(networkInput->getName(), labelsInputName);
                labelsInputCount += 1;
                labelsInputOutput = verifyNetworkInputApiLevel(networkInput, {numClasses}, dataType, labelsInputName);
            }
        } else if (batchNorm) {
            batchNormCount += 1;
            batchNormOutput = verifyBatchNormApiLevel(batchNorm, featureInputOutput, epsilon, exponentialRunningAverageFactor, sgd);
        } else if (dropOut) {
            dropOutCount += 1;
            Tensor dropOutInput = useBatchNorm ? batchNormOutput : featureInputOutput;
            dropOutOutput = verifyDropOutApiLevel(dropOut, dropOutInput, dropProportion);
        } else if (fullyConnected) {
            fullyConnectedCount += 1;
            Tensor fullyConnectedInput;
            if (dropProportion != 0.0f)
                fullyConnectedInput = dropOutOutput;
            else if (useBatchNorm)
                fullyConnectedInput = batchNormOutput;
            else
                fullyConnectedInput = featureInputOutput;
            fullyConnectedOutput = verifyFullyConnectedApiLevel(fullyConnected, fullyConnectedInput, numOutputFeatures);
        } else if (relu) {
            reluCount += 1;
            reluOutput = verifyReluApiLevel(relu, fullyConnectedOutput);
        } else if (meanAbsoluteError) {
            meanAbsoluteErrorCount += 1;
            lossOutput = verifyMeanAbsoluteErrorApiLevel(meanAbsoluteError, useRelu ? reluOutput : fullyConnectedOutput, labelsInputOutput);
        } else if (networkOutput) {
            outputCount += 1;
            verifyNetworkOutputApiLevel(networkOutput, lossOutput);
        } else {
            // Unexpected layer type
            ASSERT_FALSE(true);
        }
    }
    ASSERT_EQ(featureInputCount, 1);
    ASSERT_EQ(labelsInputCount, 1);
    ASSERT_EQ(batchNormCount, useBatchNorm);
    ASSERT_EQ(dropOutCount, dropProportion != 0.0f);
    ASSERT_EQ(fullyConnectedCount, 1);
    ASSERT_EQ(reluCount, useRelu);
    ASSERT_EQ(outputCount, 1);

    // Stamp the loaded network for the first time, ensure that it is initialized using the provided initializers
    uint32_t batchSize = 1 + rand() % 10;
    Stream stream(0);
    vector<Event> initDoneEvents;
    newNetwork.place(batchSize, initDoneEvents);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNorm;
    shared_ptr<ThorImplementation::FullyConnected> physicalFullyConnected;
    ASSERT_EQ(newNetwork.getNumStamps(), 1);
    ThorImplementation::StampedNetwork &stampedNetwork = newNetwork.getStampedNetwork(0);
    for (uint32_t i = 0; i < stampedNetwork.getNumTrainableLayers(); ++i) {
        shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> twb = stampedNetwork.getTrainableLayer(i);
        if (dynamic_pointer_cast<ThorImplementation::BatchNormalization>(twb) != nullptr)
            physicalBatchNorm = dynamic_pointer_cast<ThorImplementation::BatchNormalization>(twb);
        if (dynamic_pointer_cast<ThorImplementation::FullyConnected>(twb) != nullptr)
            physicalFullyConnected = dynamic_pointer_cast<ThorImplementation::FullyConnected>(twb);
    }
    ASSERT_EQ(physicalBatchNorm != nullptr, useBatchNorm);
    ASSERT_TRUE(physicalFullyConnected != nullptr);

    // Upon stamping, both layers should have been initialized using their initializers, verify that.
    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    if (physicalBatchNorm != nullptr) {
        ThorImplementation::Tensor weights = physicalBatchNorm->getWeights();
        ThorImplementation::Tensor biases = physicalBatchNorm->getBiases();
        ThorImplementation::Tensor means = physicalBatchNorm->getResultRunningMean();
        ThorImplementation::Tensor variances = physicalBatchNorm->getResultRunningVariance();
        vector<uint64_t> weightsDimensions = inputDimensions;
        ASSERT_EQ(weights.getDimensions(), weightsDimensions);
        ASSERT_EQ(biases.getDimensions(), weightsDimensions);
        ASSERT_EQ(means.getDimensions(), weightsDimensions);
        ASSERT_EQ(variances.getDimensions(), weightsDimensions);

        ThorImplementation::Tensor weightsCpu = weights.clone(cpuPlacement);
        ThorImplementation::Tensor biasesCpu = biases.clone(cpuPlacement);
        ThorImplementation::Tensor meansCpu = means.clone(cpuPlacement);
        ThorImplementation::Tensor variancesCpu = variances.clone(cpuPlacement);
        weightsCpu.copyFromAsync(weights, stream);
        biasesCpu.copyFromAsync(biases, stream);
        meansCpu.copyFromAsync(means, stream);
        variancesCpu.copyFromAsync(variances, stream);
        float *weightsMem = weightsCpu.getMemPtr<float>();
        float *biasesMem = biasesCpu.getMemPtr<float>();
        float *meansMem = meansCpu.getMemPtr<float>();
        float *variancesMem = variancesCpu.getMemPtr<float>();

        stream.synchronize();
        for (uint32_t i = 0; i < weightsDimensions[0]; ++i) {
            EXPECT_EQ(weightsMem[i], 1.0f);
            EXPECT_EQ(variancesMem[i], 1.0f);
            EXPECT_EQ(biasesMem[i], 0.0f);
            EXPECT_EQ(meansMem[i], 0.0f);
        }
    }

    // FullyConnected
    ThorImplementation::Tensor weights = physicalFullyConnected->getWeights();
    vector<uint64_t> weightsDimensions = {inputDimensions[0], numOutputFeatures};
    ASSERT_EQ(weights.getDimensions(), weightsDimensions);
    ThorImplementation::Tensor weightsCpu = weights.clone(cpuPlacement);
    weightsCpu.copyFromAsync(weights, stream);
    stream.synchronize();
    half *weightsMem = weightsCpu.getMemPtr<half>();
    uint32_t numWeights = weightsDimensions[0] * weightsDimensions[1];
    for (uint32_t i = 0; i < numWeights; ++i) {
        ASSERT_LE(float(weightsMem[i]), 3.01);
        ASSERT_GE(float(weightsMem[i]), 0.19);
    }
    if (hasBias) {
        ThorImplementation::Tensor biases = physicalFullyConnected->getBiases();
        ThorImplementation::Tensor biasesCpu = biases.clone(cpuPlacement);
        vector<uint64_t> biasesDimensions = {numOutputFeatures};
        ASSERT_EQ(biases.getDimensions(), biasesDimensions);
        biasesCpu.copyFromAsync(biases, stream);
        stream.synchronize();
        half *biasesMem = biasesCpu.getMemPtr<half>();
        for (uint32_t i = 0; i < biasesDimensions[0]; ++i) {
            ASSERT_LE(float(biasesMem[i]), -0.99);
            ASSERT_GE(float(biasesMem[i]), -3.01);
        }
    } else {
        ASSERT_TRUE(physicalFullyConnected->getBiases().isEmpty());
    }
}

TEST(Network, SaveLoadRoundTripStamped) {
    srand(time(nullptr));

    Network initialNetwork;

    Tensor::DataType dataType = Tensor::DataType::FP16;

    vector<uint64_t> inputDimensions = {1UL + (rand() % 16)};

    uint32_t numOutputFeatures = 1 + (rand() % 1000);
    bool hasBias = rand() % 2;
    float dropProportion = rand() % 3 == 0 ? 0.0f : (rand() % 1000) / 1000.0f;
    bool useBatchNorm = true;  // rand() % 2;
    bool useRelu = rand() % 2;

    double epsilon = randFloat(0.1, 0.3);
    double exponentialRunningAverageFactor = randFloat(0.4, 0.5);

    const string networkInputName = "testInput";
    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name(networkInputName).dimensions(inputDimensions).dataType(dataType).build();

    shared_ptr<Initializer> uniformRandomPositive = UniformRandom::Builder().minValue(0.2).maxValue(3.0).build();
    shared_ptr<Initializer> uniformRandomNegative = UniformRandom::Builder().minValue(-3.0).maxValue(-1.0).build();

    FullyConnected::Builder fullyConnectedBuilder = FullyConnected::Builder()
                                                        .network(initialNetwork)
                                                        .featureInput(networkInput.getFeatureOutput())
                                                        .numOutputFeatures(numOutputFeatures)
                                                        .weightsInitializer(uniformRandomPositive)
                                                        .biasInitializer(uniformRandomNegative)
                                                        .hasBias(hasBias)
                                                        .dropOut(dropProportion);
    if (useBatchNorm) {
        fullyConnectedBuilder.batchNormalization(exponentialRunningAverageFactor, epsilon);
    }
    if (useRelu) {
        shared_ptr<Activation> relu = Relu::Builder().build();
        fullyConnectedBuilder.activation(relu);
    } else {
        fullyConnectedBuilder.noActivation();
    }

    FullyConnected fullyConnected = fullyConnectedBuilder.build();

    Tensor logits = fullyConnected.getFeatureOutputs()[0];
    uint32_t numClasses = logits.getDimensions()[0];
    const string labelsInputName = "labelsInput";
    NetworkInput labelsInput =
        NetworkInput::Builder().network(initialNetwork).name(labelsInputName).dimensions({numClasses}).dataType(dataType).build();

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

    // Stamp the network
    uint32_t batchSize = 1 + rand() % 10;
    Stream stream(0);
    vector<Event> initDoneEvents;
    initialNetwork.place(batchSize, initDoneEvents);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Ensure the stamped network's state has been initialized
    shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNorm;
    shared_ptr<ThorImplementation::FullyConnected> physicalFullyConnected;
    ASSERT_EQ(initialNetwork.getNumStamps(), 1);
    ThorImplementation::StampedNetwork &stampedNetwork = initialNetwork.getStampedNetwork(0);
    for (uint32_t i = 0; i < stampedNetwork.getNumTrainableLayers(); ++i) {
        shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> twb = stampedNetwork.getTrainableLayer(i);
        if (dynamic_pointer_cast<ThorImplementation::BatchNormalization>(twb) != nullptr)
            physicalBatchNorm = dynamic_pointer_cast<ThorImplementation::BatchNormalization>(twb);
        if (dynamic_pointer_cast<ThorImplementation::FullyConnected>(twb) != nullptr)
            physicalFullyConnected = dynamic_pointer_cast<ThorImplementation::FullyConnected>(twb);
    }
    ASSERT_EQ(physicalBatchNorm != nullptr, useBatchNorm);
    ASSERT_TRUE(physicalFullyConnected != nullptr);

    ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor bnWeights;
    ThorImplementation::Tensor bnBiases;
    ThorImplementation::Tensor means;
    ThorImplementation::Tensor variances;
    ThorImplementation::Tensor bnWeightsCpu;
    ThorImplementation::Tensor bnBiasesCpu;
    ThorImplementation::Tensor meansCpu;
    ThorImplementation::Tensor variancesCpu;
    if (physicalBatchNorm != nullptr) {
        bnWeights = physicalBatchNorm->getWeights();
        bnBiases = physicalBatchNorm->getBiases();
        means = physicalBatchNorm->getResultRunningMean();
        variances = physicalBatchNorm->getResultRunningVariance();
        vector<uint64_t> bnWeightsDimensions = inputDimensions;
        ASSERT_EQ(bnWeights.getDimensions(), bnWeightsDimensions);
        ASSERT_EQ(bnBiases.getDimensions(), bnWeightsDimensions);
        ASSERT_EQ(means.getDimensions(), bnWeightsDimensions);
        ASSERT_EQ(variances.getDimensions(), bnWeightsDimensions);

        bnWeightsCpu = bnWeights.clone(cpuPlacement);
        bnBiasesCpu = bnBiases.clone(cpuPlacement);
        meansCpu = means.clone(cpuPlacement);
        variancesCpu = variances.clone(cpuPlacement);
        bnWeightsCpu.copyFromAsync(bnWeights, stream);
        bnBiasesCpu.copyFromAsync(bnBiases, stream);
        meansCpu.copyFromAsync(means, stream);
        variancesCpu.copyFromAsync(variances, stream);
        float *weightsMem = bnWeightsCpu.getMemPtr<float>();
        float *biasesMem = bnBiasesCpu.getMemPtr<float>();
        float *meansMem = meansCpu.getMemPtr<float>();
        float *variancesMem = variancesCpu.getMemPtr<float>();

        stream.synchronize();
        for (uint32_t i = 0; i < bnWeightsDimensions[0]; ++i) {
            EXPECT_EQ(weightsMem[i], 1.0f);
            EXPECT_EQ(variancesMem[i], 1.0f);
            EXPECT_EQ(biasesMem[i], 0.0f);
            EXPECT_EQ(meansMem[i], 0.0f);
        }
    }

    ThorImplementation::Tensor weights = physicalFullyConnected->getWeights();
    vector<uint64_t> weightsDimensions = {inputDimensions[0], numOutputFeatures};
    ASSERT_EQ(weights.getDimensions(), weightsDimensions);
    ThorImplementation::Tensor weightsCpu = weights.clone(cpuPlacement);
    weightsCpu.copyFromAsync(weights, stream);
    stream.synchronize();
    half *weightsMem = weightsCpu.getMemPtr<half>();
    uint32_t numWeights = weightsDimensions[0] * weightsDimensions[1];
    for (uint32_t i = 0; i < numWeights; ++i) {
        ASSERT_LE(float(weightsMem[i]), 3.01);
        ASSERT_GE(float(weightsMem[i]), 0.19);
    }
    ThorImplementation::Tensor biases;
    ThorImplementation::Tensor biasesCpu;
    if (hasBias) {
        biases = physicalFullyConnected->getBiases();
        biasesCpu = biases.clone(cpuPlacement);
        vector<uint64_t> biasesDimensions = {numOutputFeatures};
        ASSERT_EQ(biases.getDimensions(), biasesDimensions);
        biasesCpu.copyFromAsync(biases, stream);
        stream.synchronize();
        half *biasesMem = biasesCpu.getMemPtr<half>();
        for (uint32_t i = 0; i < biasesDimensions[0]; ++i) {
            ASSERT_LE(float(biasesMem[i]), -0.99);
            ASSERT_GE(float(biasesMem[i]), -3.01);
        }
    } else {
        ASSERT_TRUE(physicalFullyConnected->getBiases().isEmpty());
    }

    // Now let's write some values into the state to ensure that they survive a save/load cycle
    uint32_t numElements = weightsCpu.getTotalNumElements();
    half *hMemPtr = weightsCpu.getMemPtr<half>();
    for (uint32_t i = 0; i < numElements; ++i) {
        hMemPtr[i] = randFloat(-2.0f, 2.0f);
    }
    weights.copyFromAsync(weightsCpu, stream);
    if (hasBias) {
        numElements = biasesCpu.getTotalNumElements();
        hMemPtr = biasesCpu.getMemPtr<half>();
        for (uint32_t i = 0; i < numElements; ++i) {
            hMemPtr[i] = randFloat(-2.0f, 2.0f);
        }
        biases.copyFromAsync(biasesCpu, stream);
    }
    if (useBatchNorm) {
        float *memPtr;
        numElements = bnWeightsCpu.getTotalNumElements();
        memPtr = bnWeightsCpu.getMemPtr<float>();
        for (uint32_t i = 0; i < numElements; ++i) {
            memPtr[i] = randFloat(-2.0f, 2.0f);
        }
        bnWeights.copyFromAsync(bnWeightsCpu, stream);

        numElements = bnBiasesCpu.getTotalNumElements();
        memPtr = bnBiasesCpu.getMemPtr<float>();
        for (uint32_t i = 0; i < numElements; ++i) {
            memPtr[i] = randFloat(-2.0f, 2.0f);
        }
        bnBiases.copyFromAsync(bnBiasesCpu, stream);

        numElements = meansCpu.getTotalNumElements();
        memPtr = meansCpu.getMemPtr<float>();
        for (uint32_t i = 0; i < numElements; ++i) {
            memPtr[i] = randFloat(-2.0f, 2.0f);
        }
        means.copyFromAsync(meansCpu, stream);

        numElements = variancesCpu.getTotalNumElements();
        memPtr = variancesCpu.getMemPtr<float>();
        for (uint32_t i = 0; i < numElements; ++i) {
            memPtr[i] = randFloat(-2.0f, 2.0f);
        }
        variances.copyFromAsync(variancesCpu, stream);
    }

    // Save the network
    bool saveOptimizerState = rand() % 2;
    initialNetwork.save("TestModel", "/tmp", true, saveOptimizerState);

    ///////////////////////////////
    // Load
    ///////////////////////////////
    Network newNetwork;
    newNetwork.load("TestModel", "/tmp");

    // Ensure all the expected layers are in the network and properly connected
    uint32_t expectedNumLayers = 5;
    if (useBatchNorm)
        expectedNumLayers += 1;
    if (dropProportion != 0.0f)
        expectedNumLayers += 1;
    if (useRelu)
        expectedNumLayers += 1;
    uint32_t actualNumLayers = newNetwork.getNumLayers();
    ASSERT_EQ(actualNumLayers, expectedNumLayers);

    uint32_t featureInputCount = 0;
    uint32_t labelsInputCount = 0;
    uint32_t batchNormCount = 0;
    uint32_t dropOutCount = 0;
    uint32_t fullyConnectedCount = 0;
    uint32_t reluCount = 0;
    uint32_t meanAbsoluteErrorCount = 0;
    uint32_t outputCount = 0;

    Tensor featureInputOutput;
    Tensor labelsInputOutput;
    Tensor batchNormOutput;
    Tensor dropOutOutput;
    Tensor fullyConnectedOutput;
    Tensor reluOutput;
    Tensor lossOutput;
    for (uint32_t i = 0; i < actualNumLayers; ++i) {
        shared_ptr<Layer> layer = newNetwork.getLayer(i);
        shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        shared_ptr<BatchNormalization> batchNorm = dynamic_pointer_cast<BatchNormalization>(layer);
        shared_ptr<DropOut> dropOut = dynamic_pointer_cast<DropOut>(layer);
        shared_ptr<FullyConnected> fullyConnected = dynamic_pointer_cast<FullyConnected>(layer);
        shared_ptr<Relu> relu = dynamic_pointer_cast<Relu>(layer);
        shared_ptr<MeanAbsoluteError> meanAbsoluteError = dynamic_pointer_cast<MeanAbsoluteError>(layer);
        shared_ptr<NetworkOutput> networkOutput = dynamic_pointer_cast<NetworkOutput>(layer);
        if (networkInput) {
            if (networkInput->getName() == networkInputName) {
                featureInputCount += 1;
                featureInputOutput = verifyNetworkInputApiLevel(networkInput, inputDimensions, dataType, networkInputName);
            } else {
                ASSERT_EQ(networkInput->getName(), labelsInputName);
                labelsInputCount += 1;
                labelsInputOutput = verifyNetworkInputApiLevel(networkInput, {numClasses}, dataType, labelsInputName);
            }
        } else if (batchNorm) {
            batchNormCount += 1;
            batchNormOutput = verifyBatchNormApiLevel(batchNorm, featureInputOutput, epsilon, exponentialRunningAverageFactor, sgd);
        } else if (dropOut) {
            dropOutCount += 1;
            Tensor dropOutInput = useBatchNorm ? batchNormOutput : featureInputOutput;
            dropOutOutput = verifyDropOutApiLevel(dropOut, dropOutInput, dropProportion);
        } else if (fullyConnected) {
            fullyConnectedCount += 1;
            Tensor fullyConnectedInput;
            if (dropProportion != 0.0f)
                fullyConnectedInput = dropOutOutput;
            else if (useBatchNorm)
                fullyConnectedInput = batchNormOutput;
            else
                fullyConnectedInput = featureInputOutput;
            fullyConnectedOutput = verifyFullyConnectedApiLevel(fullyConnected, fullyConnectedInput, numOutputFeatures);
        } else if (relu) {
            reluCount += 1;
            reluOutput = verifyReluApiLevel(relu, fullyConnectedOutput);
        } else if (meanAbsoluteError) {
            meanAbsoluteErrorCount += 1;
            lossOutput = verifyMeanAbsoluteErrorApiLevel(meanAbsoluteError, useRelu ? reluOutput : fullyConnectedOutput, labelsInputOutput);
        } else if (networkOutput) {
            outputCount += 1;
            verifyNetworkOutputApiLevel(networkOutput, lossOutput);
        } else {
            // Unexpected layer type
            ASSERT_FALSE(true);
        }
    }
    ASSERT_EQ(featureInputCount, 1);
    ASSERT_EQ(labelsInputCount, 1);
    ASSERT_EQ(batchNormCount, useBatchNorm);
    ASSERT_EQ(dropOutCount, dropProportion != 0.0f);
    ASSERT_EQ(fullyConnectedCount, 1);
    ASSERT_EQ(reluCount, useRelu);
    ASSERT_EQ(outputCount, 1);

    // Stamp the loaded network
    newNetwork.place(batchSize, initDoneEvents);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    shared_ptr<ThorImplementation::BatchNormalization> newPhysicalBatchNorm;
    shared_ptr<ThorImplementation::FullyConnected> newPhysicalFullyConnected;
    ASSERT_EQ(newNetwork.getNumStamps(), 1);
    ThorImplementation::StampedNetwork &newStampedNetwork = newNetwork.getStampedNetwork(0);
    for (uint32_t i = 0; i < newStampedNetwork.getNumTrainableLayers(); ++i) {
        shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> twb = newStampedNetwork.getTrainableLayer(i);
        if (dynamic_pointer_cast<ThorImplementation::BatchNormalization>(twb) != nullptr)
            newPhysicalBatchNorm = dynamic_pointer_cast<ThorImplementation::BatchNormalization>(twb);
        if (dynamic_pointer_cast<ThorImplementation::FullyConnected>(twb) != nullptr)
            newPhysicalFullyConnected = dynamic_pointer_cast<ThorImplementation::FullyConnected>(twb);
    }
    ASSERT_EQ(newPhysicalBatchNorm != nullptr, useBatchNorm);
    ASSERT_TRUE(newPhysicalFullyConnected != nullptr);

    // Since they network was already stamped before it was saved, its state in the loaded network must match that of the saved network.
    if (newPhysicalBatchNorm != nullptr) {
        ThorImplementation::Tensor newBnWeights = newPhysicalBatchNorm->getWeights();
        ThorImplementation::Tensor newBnBiases = newPhysicalBatchNorm->getBiases();
        ThorImplementation::Tensor newBnMeans = newPhysicalBatchNorm->getResultRunningMean();
        ThorImplementation::Tensor newBnVariances = newPhysicalBatchNorm->getResultRunningVariance();
        vector<uint64_t> newBnWeightsDimensions = inputDimensions;
        ASSERT_EQ(newBnWeights.getDimensions(), newBnWeightsDimensions);
        ASSERT_EQ(newBnBiases.getDimensions(), newBnWeightsDimensions);
        ASSERT_EQ(newBnMeans.getDimensions(), newBnWeightsDimensions);
        ASSERT_EQ(newBnVariances.getDimensions(), newBnWeightsDimensions);

        ThorImplementation::Tensor newBnWeightsCpu = newBnWeights.clone(cpuPlacement);
        ThorImplementation::Tensor newBnBiasesCpu = newBnBiases.clone(cpuPlacement);
        ThorImplementation::Tensor newBnMeansCpu = newBnMeans.clone(cpuPlacement);
        ThorImplementation::Tensor newBnVariancesCpu = newBnVariances.clone(cpuPlacement);
        newBnWeightsCpu.copyFromAsync(newBnWeights, stream);
        newBnBiasesCpu.copyFromAsync(newBnBiases, stream);
        newBnMeansCpu.copyFromAsync(newBnMeans, stream);
        newBnVariancesCpu.copyFromAsync(newBnVariances, stream);
        float *newBnWeightsMem = newBnWeightsCpu.getMemPtr<float>();
        float *newBnBiasesMem = newBnBiasesCpu.getMemPtr<float>();
        float *newBnMeansMem = newBnMeansCpu.getMemPtr<float>();
        float *newBnVariancesMem = newBnVariancesCpu.getMemPtr<float>();

        float *expectedBnWeightsMem = bnWeightsCpu.getMemPtr<float>();
        float *expectedBnBiasesMem = bnBiasesCpu.getMemPtr<float>();
        float *expectedBnMeansMem = meansCpu.getMemPtr<float>();
        float *expectedBnVariancesMem = variancesCpu.getMemPtr<float>();

        stream.synchronize();
        for (uint32_t i = 0; i < newBnWeightsDimensions[0]; ++i) {
            EXPECT_EQ(newBnWeightsMem[i], expectedBnWeightsMem[i]);
            EXPECT_EQ(newBnVariancesMem[i], expectedBnVariancesMem[i]);
            EXPECT_EQ(newBnBiasesMem[i], expectedBnBiasesMem[i]);
            EXPECT_EQ(newBnMeansMem[i], expectedBnMeansMem[i]);
        }
    }

    // FullyConnected
    ThorImplementation::Tensor newWeights = newPhysicalFullyConnected->getWeights();
    ASSERT_EQ(newWeights.getDimensions(), weightsDimensions);
    ThorImplementation::Tensor newWeightsCpu = newWeights.clone(cpuPlacement);
    newWeightsCpu.copyFromAsync(newWeights, stream);
    stream.synchronize();
    half *newWeightsMem = newWeightsCpu.getMemPtr<half>();
    half *expectedWeights = weightsCpu.getMemPtr<half>();
    numWeights = weightsDimensions[0] * weightsDimensions[1];
    for (uint32_t i = 0; i < numWeights; ++i) {
        ASSERT_EQ(float(newWeightsMem[i]), float(expectedWeights[i]));
    }
    if (hasBias) {
        ThorImplementation::Tensor newBiases = newPhysicalFullyConnected->getBiases();
        ThorImplementation::Tensor newBiasesCpu = newBiases.clone(cpuPlacement);
        vector<uint64_t> biasesDimensions = {numOutputFeatures};
        ASSERT_EQ(biases.getDimensions(), biasesDimensions);
        newBiasesCpu.copyFromAsync(newBiases, stream);
        stream.synchronize();
        half *newBiasesMem = newBiasesCpu.getMemPtr<half>();
        half *expectedBiasesMem = biasesCpu.getMemPtr<half>();
        for (uint32_t i = 0; i < biasesDimensions[0]; ++i) {
            ASSERT_EQ(float(newBiasesMem[i]), float(expectedBiasesMem[i]));
        }
    } else {
        ASSERT_TRUE(newPhysicalFullyConnected->getBiases().isEmpty());
    }
}
