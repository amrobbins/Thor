#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

#include "DeepLearning/Implementation/Initializers/UniformRandom.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <memory>
#include <vector>

using namespace std;

using namespace ThorImplementation;

// FIXME: make a test with accumulate
// FIXME: make a test for multiple connections

static void backwardPass(shared_ptr<FullyConnected> fullyConnectedLayer, bool hasBiases, bool accumulate);

TEST(FullyConnected, FullyConnectedWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 5; ++test) {
        uint64_t numInputFeatures = (rand() % 300) + 1;
        uint64_t numOutputFeatures = (rand() % 300) + 1;
        uint64_t batchSize = (rand() % 300) + 1;
        bool inferenceOnly = (rand() % 4) == 0;
        bool hasBiases = (rand() % 4) != 0;

        Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));
        Tensor weights = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numInputFeatures, numOutputFeatures}));
        Tensor featureOut = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numOutputFeatures}));
        Tensor featureOutGpu_h;
        Tensor biases = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {numOutputFeatures}));

        const int featureInSize = batchSize * numInputFeatures;
        half *featureInMem = (half *)featureIn.getMemPtr();
        for (int i = 0; i < featureInSize; ++i) {
            featureInMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        const int weightsSize = numInputFeatures * numOutputFeatures;
        half *weightsMem = (half *)weights.getMemPtr();
        for (int i = 0; i < weightsSize; ++i) {
            weightsMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }

        half *biasesMem = (half *)biases.getMemPtr();
        if (hasBiases) {
            for (unsigned int i = 0; i < numOutputFeatures; ++i) {
                biasesMem[i] = ((rand() % 100) / 2.0f) - 5.0f;
            }
        }

        vector<shared_ptr<Layer>> layers;

        layers.push_back(
            make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions()));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<FullyConnected> fullyConnectedLayer = make_shared<FullyConnected>(numOutputFeatures, hasBiases);
        fullyConnectedLayer->setConstructForInferenceOnly(inferenceOnly);
        layers.push_back(fullyConnectedLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

        Stream dataStream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        float learningRate;
        if (!inferenceOnly) {
            ThorImplementation::Tensor anErrorInput =
                ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
            ThorImplementation::Tensor anErrorOutput =
                ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
            learningRate = (10.0f * batchSize * Loss::getLossScalingFactor()) / ((rand() % 10) + 3);
            shared_ptr<Optimizer> sgd =
                make_shared<ThorImplementation::Sgd>(fullyConnectedLayer, learningRate, 0, 0, false, anErrorInput, anErrorOutput);
            fullyConnectedLayer->setOptimizer(sgd);
        }

        // Backward tensors must not be created, since they would be unused and would waist memory.
        if (inferenceOnly) {
            ASSERT_TRUE(fullyConnectedLayer->getErrorOutputs()[0].isEmpty());
            ASSERT_TRUE(fullyConnectedLayer->getOptimizer().isEmpty());
        }

        if (!hasBiases) {
            ASSERT_TRUE(fullyConnectedLayer->getBiases().isEmpty());
        }

        featureOutGpu_h = layers.back()->getFeatureOutput();

        Event weightsUpdatedEvent = fullyConnectedLayer->setWeightsAndBiases(weights, biases, dataStream.putEvent());
        dataStream.waitEvent(weightsUpdatedEvent);
        if (!inferenceOnly)
            fullyConnectedLayer->getOptimizer().get()->getGradientUpdateStream().waitEvent(weightsUpdatedEvent);

        // Network is runnable here
        layers[0]->forward(featureIn, false);
        dataStream.waitEvent(dynamic_pointer_cast<NetworkOutput>(layers.back())->getOutputReadyEvent());

        matrixMultiplyCpuHalf((half *)featureIn.getMemPtr(),
                              (half *)weights.getMemPtr(),
                              (half *)featureOut.getMemPtr(),
                              batchSize,
                              numInputFeatures,
                              numInputFeatures,
                              numOutputFeatures,
                              numInputFeatures,
                              numOutputFeatures,
                              numOutputFeatures,
                              false,
                              false,
                              false,
                              false);

        if (hasBiases) {
            half *featureOutMem = (half *)featureOut.getMemPtr();
            for (unsigned int batch = 0; batch < batchSize; ++batch) {
                for (unsigned int outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                    featureOutMem[batch * numOutputFeatures + outputFeature] =
                        featureOutMem[batch * numOutputFeatures + outputFeature] + biasesMem[outputFeature];
                }
            }
        }

        dataStream.synchronize();

        half *cpuFeatureOut = (half *)featureOut.getMemPtr();
        half *gpuFeatureOut = (half *)featureOutGpu_h.getMemPtr();
        int numOutputElements = batchSize * numOutputFeatures;
        float maxDiff = numInputFeatures * 0.0125;
        for (int i = 0; i < numOutputElements; ++i) {
            if (abs((float)(cpuFeatureOut[i]) - (float)(gpuFeatureOut[i])) >= maxDiff) {
                int batch = i / numOutputFeatures;
                int outputFeature = i - (batch * numOutputFeatures);
                printf("%f %f   bias %f\n",
                       (float)(cpuFeatureOut[i]),
                       (float)(gpuFeatureOut[i]),
                       hasBiases ? (float)biasesMem[outputFeature] : 0.0f);
            }
            float expected = (float)(cpuFeatureOut[i]);
            maxDiff = max(abs((float)expected * 0.02f), maxDiff);
            ASSERT_LT(abs(expected - (float)(gpuFeatureOut[i])), max(maxDiff, expected * 0.01f));
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        bool accumulate = false;  // FIXME

        backwardPass(fullyConnectedLayer, hasBiases, accumulate);

        LayerTestHelper::tearDownNetwork(layers);
    }
}

void backwardPass(shared_ptr<FullyConnected> fullyConnectedLayer, bool hasBiases, bool accumulate) {
    assert(accumulate == false);  // FIXME

    assert(fullyConnectedLayer->getOptimizer().isPresent());
    shared_ptr<Sgd> sgd = dynamic_pointer_cast<Sgd>(fullyConnectedLayer->getOptimizer().get());
    assert(sgd != nullptr);
    float learningRate = sgd->getInitialLearningRate();
    assert(sgd->getDecay() == 0.0f);
    assert(sgd->getMomentum() == 0.0f);
    sgd->updateHyperParameters(0, 0, 1);

    Stream dataStream = fullyConnectedLayer->getStreams()[0];
    Stream gradientUpdateStream = fullyConnectedLayer->getOptimizer().get()->getGradientUpdateStream();

    Tensor errorInput = fullyConnectedLayer->getErrorInputs().front().get().clone(TensorPlacement::MemDevices::CPU);

    Tensor featureInput = fullyConnectedLayer->getFeatureInputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor errorOutput = fullyConnectedLayer->getErrorOutputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor weights = fullyConnectedLayer->getWeights().clone(TensorPlacement::MemDevices::CPU);
    Tensor weightsGradient = fullyConnectedLayer->getOptimizer().get()->getWeightsGradient().clone(TensorPlacement::MemDevices::CPU);
    Tensor biases;
    Tensor biasesGradient;
    Tensor biasesGradientFloat;
    if (hasBiases) {
        biases = fullyConnectedLayer->getBiases().get().clone(TensorPlacement::MemDevices::CPU);
        biasesGradient = fullyConnectedLayer->getOptimizer().get()->getBiasesGradient().get().clone(TensorPlacement::MemDevices::CPU);
        biasesGradientFloat = biasesGradient.clone(TensorDescriptor::DataType::FP32);
    }

    int batchSize = featureInput.getDescriptor().getDimensions()[0];
    int numInputFeatures = featureInput.getDescriptor().getDimensions()[1];
    int numOutputFeatures = errorInput.getDescriptor().getDimensions()[1];

    // featureIn, weights and biases are populated still by the forward pass
    // just need to populate errorIn
    featureInput.copyFromAsync(fullyConnectedLayer->getFeatureInputs().front(), dataStream);
    weights.copyFromAsync(fullyConnectedLayer->getWeights(), dataStream);
    if (hasBiases) {
        biases.copyFromAsync(fullyConnectedLayer->getBiases(), dataStream);
        biasesGradient.copyFromAsync(fullyConnectedLayer->getOptimizer().get()->getBiasesGradient(), dataStream);
        biasesGradientFloat.copyFromAsync(biasesGradient, dataStream);
    }
    dataStream.synchronize();

    const int errorInputSize = batchSize * numOutputFeatures;
    half *errorInputMem = (half *)errorInput.getMemPtr();
    for (int i = 0; i < errorInputSize; ++i) {
        errorInputMem[i] = ((rand() % 100) / 10.0f) - 5.0f;
    }
    fullyConnectedLayer->getErrorInputs().front().get().copyFromAsync(errorInput, dataStream);

    fullyConnectedLayer->backward(fullyConnectedLayer->getErrorInputs().front());

    matrixMultiplyCpuHalf((half *)errorInput.getMemPtr(),
                          (half *)weights.getMemPtr(),
                          (half *)errorOutput.getMemPtr(),
                          batchSize,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          numInputFeatures,
                          false,
                          true,
                          false,
                          false);

    matrixMultiplyCpuHalf((half *)featureInput.getMemPtr(),
                          (half *)errorInput.getMemPtr(),
                          (half *)weightsGradient.getMemPtr(),
                          batchSize,
                          numInputFeatures,
                          batchSize,
                          numOutputFeatures,
                          numInputFeatures,
                          numOutputFeatures,
                          numOutputFeatures,
                          true,
                          false,
                          false,
                          false);

    if (hasBiases) {
        // Reduce errorIn_batchSizexnumOutputFeatures to biasesGradient_numOutputFeatures by summing the elements of each batch item.
        float *biasesGradientMem = (float *)biasesGradientFloat.getMemPtr();
        half *errorInMem = (half *)errorInput.getMemPtr();
        for (int batch = 0; batch < batchSize; ++batch) {
            for (int outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                if (batch == 0 && !accumulate)
                    biasesGradientMem[outputFeature] = 0.0f;
                biasesGradientMem[outputFeature] =
                    biasesGradientMem[outputFeature] + (float)errorInMem[batch * numOutputFeatures + outputFeature];
            }
        }
        biasesGradient.copyFromAsync(biasesGradientFloat, gradientUpdateStream);
    }

    Tensor errorOutputGpu_h = fullyConnectedLayer->getErrorOutputs().front().get().clone(TensorPlacement::MemDevices::CPU);
    Tensor weightsGradientGpu_h = fullyConnectedLayer->getOptimizer().get()->getWeightsGradient().clone(TensorPlacement::MemDevices::CPU);
    Tensor biasesGradientGpu_h;
    if (hasBiases)
        biasesGradientGpu_h = fullyConnectedLayer->getOptimizer().get()->getBiasesGradient().get().clone(TensorPlacement::MemDevices::CPU);

    errorOutputGpu_h.copyFromAsync(fullyConnectedLayer->getErrorOutputs().front(), dataStream);
    weightsGradientGpu_h.copyFromAsync(fullyConnectedLayer->getOptimizer().get()->getWeightsGradient(), gradientUpdateStream);
    if (hasBiases)
        biasesGradientGpu_h.copyFromAsync(fullyConnectedLayer->getOptimizer().get()->getBiasesGradient(), gradientUpdateStream);

    dataStream.synchronize();
    gradientUpdateStream.synchronize();

    half *cpuErrorOutput = (half *)errorOutput.getMemPtr();
    half *gpuErrorOutput = (half *)errorOutputGpu_h.getMemPtr();
    int numOutputElements = batchSize * numInputFeatures;
    float maxDiff = numOutputFeatures * 0.01;
    for (int i = 0; i < numOutputElements; ++i) {
        float expected = (float)(cpuErrorOutput[i]);
        if (abs(expected - (float)(gpuErrorOutput[i])) >= maxDiff)
            printf("%f %f\n", (float)(cpuErrorOutput[i]), (float)(gpuErrorOutput[i]));
        ASSERT_LT(abs((float)(cpuErrorOutput[i]) - (float)(gpuErrorOutput[i])), max(maxDiff, expected * 0.01f));
    }

    half *cpuWeightsGradient = (half *)weightsGradient.getMemPtr();
    half *gpuWeightsGradient = (half *)weightsGradientGpu_h.getMemPtr();
    int numWeights = numInputFeatures * numOutputFeatures;
    maxDiff = batchSize * 0.01;
    for (int i = 0; i < numWeights; ++i) {
        if (abs((float)(cpuWeightsGradient[i]) - (float)(gpuWeightsGradient[i])) >= maxDiff)
            printf("%f %f\n", (float)(cpuWeightsGradient[i]), (float)(gpuWeightsGradient[i]));
        float expected = (float)(cpuWeightsGradient[i]);
        ASSERT_LT(abs(expected - (float)(gpuWeightsGradient[i])), max(maxDiff, expected * 0.01f));
    }

    if (hasBiases) {
        half *cpuBiasesGradient = (half *)biasesGradient.getMemPtr();
        half *gpuBiasesGradient = (half *)biasesGradientGpu_h.getMemPtr();

        for (int i = 0; i < numOutputFeatures; ++i) {
            if (abs((float)(cpuBiasesGradient[i]) - (float)(gpuBiasesGradient[i])) < maxDiff) {
            } else {
                printf("%d of %d   %f %f\n", i + 1, numOutputFeatures, (float)(cpuBiasesGradient[i]), (float)(gpuBiasesGradient[i]));
            }
            ASSERT_LT(abs((float)(cpuBiasesGradient[i]) - (float)(gpuBiasesGradient[i])), maxDiff);
        }
    }

    // Test apply gradients
    Tensor weightsGpu_h = fullyConnectedLayer->getWeights().clone(TensorPlacement::MemDevices::CPU);
    Tensor biasesGpu_h;
    if (hasBiases)
        biasesGpu_h = fullyConnectedLayer->getBiases().get().clone(TensorPlacement::MemDevices::CPU);

    gradientUpdateStream.synchronize();
    weightsGpu_h.copyFromAsync(fullyConnectedLayer->getWeights(), dataStream);
    if (hasBiases)
        biasesGpu_h.copyFromAsync(fullyConnectedLayer->getBiases(), dataStream);
    dataStream.synchronize();

    half *weightsMem = (half *)weights.getMemPtr();
    half *weightsGpuMem = (half *)weightsGpu_h.getMemPtr();
    half *weightsGradientMem = (half *)weightsGradientGpu_h.getMemPtr();
    for (int i = 0; i < numWeights; ++i) {
        half expected = (float)weightsMem[i] - (learningRate * (float)weightsGradientMem[i]) / (Loss::getLossScalingFactor() * batchSize);
        maxDiff = abs((float)expected / 1000.0f);
        if (maxDiff < 0.016)
            maxDiff = 0.016;
        if (abs((float)expected - (float)weightsGpuMem[i]) >= maxDiff) {
            printf("%d  cpu %f  gpu %f     weight %f   gradient %f  learningRate %f maxDiff %f\n",
                   i,
                   (float)expected,
                   (float)weightsGpuMem[i],
                   (float)weightsMem[i],
                   (float)weightsGradientMem[i],
                   learningRate,
                   maxDiff);
        }
        ASSERT_LT(abs((float)expected - (float)weightsGpuMem[i]), maxDiff);
    }

    if (hasBiases) {
        int numBiases = fullyConnectedLayer->getBiases().get().getDescriptor().getTotalNumElements();
        half *biasesMem = (half *)biases.getMemPtr();
        half *biasesGpuMem = (half *)biasesGpu_h.getMemPtr();
        half *biasesGradientMem = (half *)biasesGradientGpu_h.getMemPtr();
        for (int i = 0; i < numBiases; ++i) {
            half expected = (float)biasesMem[i] - (learningRate * (float)biasesGradientMem[i]) / (Loss::getLossScalingFactor() * batchSize);
            maxDiff = abs((float)expected / 1000.0f);
            if (maxDiff < 0.01)
                maxDiff = 0.01;
            ASSERT_LT(abs((float)expected - (float)biasesGpuMem[i]), maxDiff);
            if (abs((float)expected - (float)biasesGpuMem[i]) >= maxDiff) {
                printf("%d  cpu %f  gpu %f     bias %f   gradient %f  learningRate %f\n",
                       i,
                       (float)expected,
                       (float)biasesGpuMem[i],
                       (float)biasesMem[i],
                       (float)biasesGradientMem[i],
                       learningRate);
            }
        }
    }
}

TEST(FullyConnectedInitializers, UniformRandomWorks) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 5; ++test) {
        uint64_t numInputFeatures = (rand() % 1024) + 1;
        uint64_t numOutputFeatures = (rand() % 2048) + 1;
        uint64_t batchSize = (rand() % 300) + 1;
        bool hasBiases = (rand() % 4) != 0;

        Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));

        vector<shared_ptr<Layer>> layers;
        layers.push_back(
            make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions()));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<FullyConnected> fullyConnectedLayer = make_shared<FullyConnected>(numOutputFeatures, hasBiases);
        layers.push_back(fullyConnectedLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

        Stream stream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        UniformRandom initializer(0.1, -0.1);
        initializer.initialize(fullyConnectedLayer.get(), fullyConnectedLayer->getWeights());
        if (hasBiases) {
            initializer.initialize(fullyConnectedLayer.get(), fullyConnectedLayer->getBiases());
        }

        Tensor weights = fullyConnectedLayer->getWeights().clone(cpuPlacement);
        weights.copyFromAsync(fullyConnectedLayer->getWeights(), stream);
        Tensor biases;
        if (hasBiases) {
            biases = fullyConnectedLayer->getBiases().get().clone(cpuPlacement);
            biases.copyFromAsync(fullyConnectedLayer->getBiases(), stream);
        }

        stream.synchronize();

        int totalNumWeights = fullyConnectedLayer->getWeights().getDescriptor().getTotalNumElements();
        half *weightsMem = (half *)weights.getMemPtr();
        for (int i = 0; i < totalNumWeights; ++i) {
            ASSERT_LT(abs((float)weightsMem[i]), 0.1);
        }

        if (hasBiases) {
            int totalNumBiases = fullyConnectedLayer->getBiases().get().getDescriptor().getTotalNumElements();
            half *biasesMem = (half *)biases.getMemPtr();
            for (int i = 0; i < totalNumBiases; ++i) {
                ASSERT_LT(abs((float)biasesMem[i]), 0.1);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}
