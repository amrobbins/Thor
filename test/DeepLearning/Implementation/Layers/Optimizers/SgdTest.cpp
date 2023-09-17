#include "Thor.h"
#include "gtest/gtest.h"

#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

using namespace ThorImplementation;
using namespace std;

float computeCurrentLearningRate(float initialLearningRate, float decay, float epoch) {
    float currentLearningRate = initialLearningRate * pow(1.0 - (double)decay, (double)epoch);
    return currentLearningRate;
}

void verifyMatricesMatch(
    half *expected, half *actual, uint32_t rows, uint32_t cols, bool print = false, float staticThresh = 0.1, float threshScale = 0.002) {
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            float expectedValue = expected[row * cols + col];
            float actualValue = actual[row * cols + col];
            float diff = abs(expectedValue - actualValue);
            float scaledThresh = max(staticThresh, fabsf(expectedValue * threshScale));
            if (print || diff > scaledThresh) {
                printf("[%d,%d] GPU %f vs %f CPU\n", row, col, actualValue, expectedValue);
            }
            ASSERT_LE(diff, scaledThresh);
        }
    }
}

void reduceBatch(half *original, half *reduced, uint32_t batchSize, uint32_t featureOutSize, bool accumulate) {
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t o = 0; o < featureOutSize; ++o) {
            if (!accumulate && b == 0)
                reduced[o] = original[b * featureOutSize + o];
            else
                reduced[o] += original[b * featureOutSize + o];
        }
    }
}

TEST(SgdTest, TestConstrutorSettersGetters) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    uint32_t batchSize = (rand() % 300) + 1;
    uint32_t numInputFeatures = (rand() % 300) + 1;
    uint32_t numOutputFeatures = (rand() % 300) + 1;
    bool hasBias = rand() % 2;

    Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));

    float initialLearningRate = 10.0 / (1 + rand() % 100);
    float decay = 1.0f / (1.0f + (rand() % 10));
    float momentum = rand() % 2 ? 0.0f : 1.0f / (1.0f + (rand() % 10));
    bool useNesterovMomentum = rand() % 2;

    vector<shared_ptr<Layer>> layers;

    layers.push_back(make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions()));
    layers.push_back(make_shared<NoOpLayer>());
    shared_ptr<FullyConnected> fullyConnectedLayer = make_shared<FullyConnected>(numOutputFeatures, hasBias);
    layers.push_back(fullyConnectedLayer);
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

    Stream dataStream = layers.front()->getStream();

    LayerTestHelper::connectAndInitializeNetwork(layers);

    ThorImplementation::Tensor anErrorInput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
    ThorImplementation::Tensor anErrorOutput =
        ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
    shared_ptr<Sgd> sgd = make_shared<ThorImplementation::Sgd>(
        fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, anErrorInput, anErrorOutput);
    fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(sgd));

    ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
    ASSERT_EQ(sgd->getDecay(), decay);
    ASSERT_EQ(sgd->getMomentum(), momentum);
    ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesterovMomentum);

    initialLearningRate = 10.0 / (1 + rand() % 100);
    decay = 1.0f / (1.0f + (rand() % 10));
    momentum = rand() % 2 ? 0.0f : 1.0f / (1.0f + (rand() % 10));
    ;
    useNesterovMomentum = rand() % 2;

    sgd->setInitialLearningRate(initialLearningRate);
    sgd->setDecay(decay);
    sgd->setMomentum(momentum);
    sgd->setUseNesterovMomentum(useNesterovMomentum);

    ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
    ASSERT_EQ(sgd->getDecay(), decay);
    ASSERT_EQ(sgd->getMomentum(), momentum);
    ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesterovMomentum);

    unordered_map<std::string, float> hyperParameters;

    uint64_t epoch = rand() % 10;
    uint64_t batchesPerEpoch = 1 + (rand() % 3000);
    uint64_t batch = rand() % batchesPerEpoch;
    uint8_t flags = 0b00000000;
    const uint8_t CLR = 0b00000001;
    const uint8_t ILR = 0b00000010;
    const uint8_t D = 0b00000100;
    const uint8_t M = 0b00001000;
    const uint8_t UNM = 0b00010000;
    hyperParameters = sgd->getAllHyperParameters(epoch, batch, batchesPerEpoch);
    for (auto it = hyperParameters.begin(); it != hyperParameters.end(); ++it) {
        string parameter = it->first;
        float value = it->second;

        if (parameter == "currentLearningRate") {
            ASSERT_EQ(value, computeCurrentLearningRate(initialLearningRate, decay, epoch));
            ASSERT_EQ(flags & CLR, 0);
            flags |= CLR;
        } else if (parameter == "initialLearningRate") {
            ASSERT_EQ(value, initialLearningRate);
            ASSERT_EQ(flags & ILR, 0);
            flags |= ILR;
        } else if (parameter == "decay") {
            ASSERT_EQ(value, decay);
            ASSERT_EQ(flags & D, 0);
            flags |= D;
        } else if (parameter == "momentum") {
            ASSERT_EQ(value, momentum);
            ASSERT_EQ(flags & M, 0);
            flags |= M;
        } else if (parameter == "useNesterovMomentum") {
            ASSERT_EQ(value, useNesterovMomentum);
            ASSERT_EQ(flags & UNM, 0);
            flags |= UNM;
        } else {
            ASSERT_EQ(true, false);
        }
    }

    epoch = rand() % 10;
    batchesPerEpoch = 1 + (rand() % 3000);
    batch = rand() % batchesPerEpoch;
    flags = 0b00000000;
    hyperParameters = sgd->getAllHyperParameters(epoch, batch, batchesPerEpoch);
    for (auto it = hyperParameters.begin(); it != hyperParameters.end(); ++it) {
        string parameter = it->first;
        float value = it->second;

        if (parameter == "currentLearningRate") {
            ASSERT_EQ(value, computeCurrentLearningRate(initialLearningRate, decay, epoch));
            ASSERT_EQ(flags & CLR, 0);
            flags |= CLR;
        } else if (parameter == "initialLearningRate") {
            ASSERT_EQ(value, initialLearningRate);
            ASSERT_EQ(flags & ILR, 0);
            flags |= ILR;
        } else if (parameter == "decay") {
            ASSERT_EQ(value, decay);
            ASSERT_EQ(flags & D, 0);
            flags |= D;
        } else if (parameter == "momentum") {
            ASSERT_EQ(value, momentum);
            ASSERT_EQ(flags & M, 0);
            flags |= M;
        } else if (parameter == "useNesterovMomentum") {
            ASSERT_EQ(value, useNesterovMomentum);
            ASSERT_EQ(flags & UNM, 0);
            flags |= UNM;
        } else {
            ASSERT_EQ(true, false);
        }
    }
}

TEST(SgdTest, TestWeightsUpdate) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 3; ++t) {
        uint32_t batchSize = (rand() % 300) + 1;
        uint32_t numInputFeatures = (rand() % 300) + 1;
        uint32_t numOutputFeatures = (rand() % 300) + 1;
        bool hasBias = rand() % 2;

        Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));

        float initialLearningRate = 10000.0 / (1 + rand() % 33);
        float decay = 1.0f / (1.0f + (rand() % 10));
        float momentum = 0.0f;  // FIXME
        // float momentum = rand() % 2 ? 0.0f : 1.0f / (1.0f + (rand() % 10));
        bool useNesterovMomentum = rand() % 2;
        uint64_t epoch = rand() % 10;

        vector<shared_ptr<Layer>> layers;

        layers.push_back(
            make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, featureIn.getDescriptor().getDimensions()));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<FullyConnected> fullyConnectedLayer = make_shared<FullyConnected>(numOutputFeatures, hasBias);
        layers.push_back(fullyConnectedLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

        Stream dataStream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        ThorImplementation::Tensor featureInput =
            ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs());
        ThorImplementation::Tensor errorInput =
            ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
        ThorImplementation::Tensor errorOutput =
            ThorImplementation::MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
        shared_ptr<Sgd> sgd = make_shared<ThorImplementation::Sgd>(
            fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, errorInput, errorOutput);
        fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(sgd));

        ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
        ASSERT_EQ(sgd->getDecay(), decay);
        ASSERT_EQ(sgd->getMomentum(), momentum);
        ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesterovMomentum);

        Tensor featureInput_h = featureInput.clone(cpuPlacement);
        Tensor errorInput_h = errorInput.clone(cpuPlacement);
        Tensor weightsGradient = sgd->getWeightsGradient();
        Tensor weightsGradient_h = weightsGradient.clone(cpuPlacement);
        Tensor weightsGradientGpu_h = weightsGradient_h.clone();

        Optional<Tensor> biases = fullyConnectedLayer->getBiases();
        Tensor biases_h;
        Tensor biasesGpu_h;
        half *biasesMem_h;
        half *biasesGpuMem_h;
        Optional<Tensor> biasesGradient = sgd->getBiasesGradient();
        Tensor biasesGradient_h;
        Tensor biasesGradientGpu_h;
        half *biasesGradientMem_h;
        half *biasesGradientGpuMem_h;
        if (hasBias) {
            ASSERT_TRUE(biases.isPresent());
            ASSERT_TRUE(biasesGradient.isPresent());
            biases_h = biases.get().clone(cpuPlacement);
            biasesGpu_h = biases_h.clone();
            biasesMem_h = biases_h.getMemPtr<half>();
            biasesGpuMem_h = biasesGpu_h.getMemPtr<half>();
            biasesGradient_h = biasesGradient.get().clone(cpuPlacement);
            biasesGradientGpu_h = biasesGradient_h.clone();
            biasesGradientMem_h = biasesGradient_h.getMemPtr<half>();
            biasesGradientGpuMem_h = biasesGradientGpu_h.getMemPtr<half>();
        }

        Stream gradientUpdateStream = sgd->getGradientUpdateStream();

        // Set fill featureIn and errorIn and check that weights are updated properly
        half *featureInMem_h = featureInput_h.getMemPtr<half>();
        half *errorInMem_h = errorInput_h.getMemPtr<half>();
        half *weightsGradientMem_h = weightsGradient_h.getMemPtr<half>();
        half *weightsGradientGpuMem_h = weightsGradientGpu_h.getMemPtr<half>();
        for (uint32_t i = 0; i < numInputFeatures * batchSize; ++i) {
            float val = 10.0f / (1 + (rand() % 100));
            if (rand() % 2)
                val = -val;
            featureInMem_h[i] = val;
        }
        for (uint32_t i = 0; i < numOutputFeatures * batchSize; ++i) {
            float val = 10.0f / (1 + (rand() % 100));
            if (rand() % 2)
                val = -val;
            errorInMem_h[i] = val;
        }
        featureInput.copyFromAsync(featureInput_h, dataStream);
        errorInput.copyFromAsync(errorInput_h, dataStream);
        dataStream.synchronize();

        uint32_t batchesPerEpoch = 1 + rand() % 10000;
        sgd->updateHyperParameters(epoch, rand() % batchesPerEpoch, batchesPerEpoch);

        sgd->computeWeightsUpdate(featureInput, errorInput, false);
        matrixMultiplyCpuHalf(featureInMem_h,
                              errorInMem_h,
                              weightsGradientMem_h,
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
        weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
        gradientUpdateStream.synchronize();
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures);

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.2f, 0.02f);
        }

        sgd->computeWeightsUpdate(featureInput, errorInput, false);
        matrixMultiplyCpuHalf(featureInMem_h,
                              errorInMem_h,
                              weightsGradientMem_h,
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
        weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
        gradientUpdateStream.synchronize();
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures);

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.2f, 0.02f);
        }

        sgd->computeWeightsUpdate(featureInput, errorInput, true);
        matrixMultiplyCpuHalf(featureInMem_h,
                              errorInMem_h,
                              weightsGradientMem_h,
                              batchSize,
                              numInputFeatures,
                              batchSize,
                              numOutputFeatures,
                              numInputFeatures,
                              numOutputFeatures,
                              numOutputFeatures,
                              true,
                              false,
                              true,
                              false);
        weightsGradientGpu_h.copyFromAsync(weightsGradient, gradientUpdateStream);
        gradientUpdateStream.synchronize();
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures);

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, true);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.2f, 0.02f);
        }

        Tensor weights = fullyConnectedLayer->getWeights();
        Tensor weights_h = weights.clone(cpuPlacement);
        Tensor weightsGpu_h = weights_h.clone();
        half *weightsMem_h = weights_h.getMemPtr<half>();
        half *weightsMemGpu_h = weightsGpu_h.getMemPtr<half>();
        weights_h.clear();
        weights.copyFromAsync(weights_h, gradientUpdateStream);
        if (hasBias) {
            biases_h.clear();
            biases.get().copyFromAsync(biases_h, gradientUpdateStream);
        }
        gradientUpdateStream.synchronize();

        sgd->updateWeights(weights, fullyConnectedLayer->getBiases(), batchSize);

        weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (momentum == 0.0f) {
            float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, epoch);
            float weightUpdateScalingFactor = (-1.0f * currentLearningRate) / batchSize;
            weightUpdateScalingFactor *= 1.0f / Loss::getLossScalingFactor();
            for (uint32_t row = 0; row < numInputFeatures; ++row) {
                for (uint32_t col = 0; col < numOutputFeatures; ++col) {
                    weightsMem_h[row * numOutputFeatures + col] =
                        weightsGradientMem_h[row * numOutputFeatures + col] * (half)weightUpdateScalingFactor;
                }
            }
            verifyMatricesMatch(weightsMem_h, weightsMemGpu_h, numInputFeatures, numOutputFeatures);

            if (hasBias) {
                biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
                gradientUpdateStream.synchronize();

                for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                    biasesMem_h[outputFeature] = biasesGradientMem_h[outputFeature] * (half)weightUpdateScalingFactor;
                }
                verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.2f, 0.02f);
            }
        } else if (useNesterovMomentum) {
            // FIXME: implement
        } else {
            // Standard momentum
            // FIXME: implement
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
