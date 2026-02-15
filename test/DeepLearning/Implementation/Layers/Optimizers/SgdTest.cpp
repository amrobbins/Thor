// FIXME: Check over this to ensure it would properly catch failures
// FIXME: Get Nesterov momentum test working

#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

#include "gtest/gtest.h"

#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

using namespace ThorImplementation;
using namespace std;

static float computeCurrentLearningRate(float initialLearningRate, float decay, float epoch) {
    float currentLearningRate = initialLearningRate * pow(1.0 - (double)decay, (double)epoch);
    return currentLearningRate;
}

static bool verifyMatricesMatch(
    half *expected, half *actual, uint32_t rows, uint32_t cols, bool print = false, float staticThresh = 0.1, float dynamicThresh = 0.004) {
    double meanAbsoluteError = 0.0;
    bool foundError = false;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < cols; ++col) {
            float expectedValue = expected[row * cols + col];
            float actualValue = actual[row * cols + col];
            float diff = abs(expectedValue - actualValue);
            float scaledThresh = max(staticThresh, fabsf(expectedValue * dynamicThresh));
            if (print || diff > scaledThresh) {
                printf("[%d,%d] GPU %f vs %f CPU\n", row, col, actualValue, expectedValue);
            }
            fflush(stdout);
            EXPECT_LE(diff, scaledThresh);
            if (!(diff <= scaledThresh))
                foundError = true;
            meanAbsoluteError += diff;
        }
    }
    meanAbsoluteError /= (double)(rows * cols);
    // printf("MeanAbsoluteError: %lf\n", meanAbsoluteError);
    // fflush(stdout);

    //    if (print == false && meanAbsoluteError > 0.75) {
    //        verifyMatricesMatch(expected, actual, rows, cols, true, staticThresh, dynamicThresh);
    //        printf("rows %d cols %d\n", rows, cols);
    //        fflush(stdout);
    //    }

    return foundError;
}

static void reduceBatch(half *original, half *reduced, uint32_t batchSize, uint32_t featureOutSize, bool accumulate) {
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
    srand(time(nullptr));
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
    fullyConnectedLayer->setConstructForInferenceOnly(true);
    layers.push_back(fullyConnectedLayer);
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

    Stream dataStream = layers.front()->getStream();

    LayerTestHelper::connectAndInitializeNetwork(layers);

    shared_ptr<Sgd> sgd = make_shared<Sgd>(fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, 0);
    sgd->compile();
    fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(sgd));

    ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
    ASSERT_EQ(sgd->getDecay(), decay);
    ASSERT_EQ(sgd->getMomentum(), momentum);
    ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesterovMomentum);

    initialLearningRate = 10.0 / (1 + rand() % 100);
    decay = 1.0f / (1.0f + (rand() % 10));
    momentum = rand() % 2 ? 0.0f : 1.0f / (1.0f + (rand() % 10));
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
    sgd->updateHyperParameters(epoch, batch, batchesPerEpoch);
    hyperParameters = sgd->getAllHyperParameters();
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
    // Ensure all params reported
    ASSERT_EQ(flags, CLR | ILR | D | M | UNM);

    epoch = rand() % 10;
    batchesPerEpoch = 1 + (rand() % 3000);
    batch = rand() % batchesPerEpoch;
    flags = 0b00000000;
    sgd->updateHyperParameters(epoch, batch, batchesPerEpoch);
    hyperParameters = sgd->getAllHyperParameters();
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
    ASSERT_EQ(flags, CLR | ILR | D | M | UNM);
}

TEST(SgdTest, TestWeightsUpdateNoMomentum) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 3; ++t) {
        uint32_t batchSize = (rand() % 300) + 1;
        uint32_t numInputFeatures = (rand() % 300) + 1;
        uint32_t numOutputFeatures = (rand() % 300) + 1;
        bool hasBias = rand() % 2;

        Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));

        float initialLearningRate = 10000.0 / (1 + rand() % 33);
        float decay = ((rand() % 5) / 10.0f);
        float momentum = 0.0f;
        bool useNesterovMomentum = false;
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

        LayerTestHelper::connectNetwork(layers);

        Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs());
        Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
        shared_ptr<Sgd> sgd = make_shared<Sgd>(fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, 0);
        sgd->compile();
        fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(sgd));
        LayerTestHelper::initializeNetwork(layers);

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

        // Fill featureIn and errorIn and check that weights are updated properly
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
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weightsGradient_h.copyFromAsync(weightsGradientGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biasesGradient_h.copyFromAsync(biasesGradientGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
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
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weightsGradient_h.copyFromAsync(weightsGradientGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biasesGradient_h.copyFromAsync(biasesGradientGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
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
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weightsGradient_h.copyFromAsync(weightsGradientGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, true);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biasesGradient_h.copyFromAsync(biasesGradientGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
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

        float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, epoch);
        assert(momentum == 0.0f);
        assert(useNesterovMomentum == false);
        float weightUpdateScalingFactor = (-1.0f * currentLearningRate) / batchSize;
        weightUpdateScalingFactor *= 1.0f / Loss::getLossScalingFactor();
        for (uint32_t row = 0; row < numInputFeatures; ++row) {
            for (uint32_t col = 0; col < numOutputFeatures; ++col) {
                weightsMem_h[row * numOutputFeatures + col] +=
                    weightsGradientMem_h[row * numOutputFeatures + col] * (half)weightUpdateScalingFactor;
            }
        }
        verifyMatricesMatch(weightsMem_h, weightsMemGpu_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
            gradientUpdateStream.synchronize();

            for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                biasesMem_h[outputFeature] += biasesGradientMem_h[outputFeature] * (half)weightUpdateScalingFactor;
            }
            verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
        }

        sgd->updateWeights(weights, fullyConnectedLayer->getBiases(), batchSize);

        weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, epoch);
        weightUpdateScalingFactor = (-1.0f * currentLearningRate) / batchSize;
        weightUpdateScalingFactor *= 1.0f / Loss::getLossScalingFactor();
        for (uint32_t row = 0; row < numInputFeatures; ++row) {
            for (uint32_t col = 0; col < numOutputFeatures; ++col) {
                weightsMem_h[row * numOutputFeatures + col] +=
                    weightsGradientMem_h[row * numOutputFeatures + col] * (half)weightUpdateScalingFactor;
            }
        }
        verifyMatricesMatch(weightsMem_h, weightsMemGpu_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
            gradientUpdateStream.synchronize();

            for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                biasesMem_h[outputFeature] += biasesGradientMem_h[outputFeature] * (half)weightUpdateScalingFactor;
            }
            verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
        }
    }
}

TEST(SgdTest, TestWeightsUpdateWithMomentum) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 3; ++t) {
        uint32_t batchSize = (rand() % 300) + 1;
        uint32_t numInputFeatures = (rand() % 300) + 1;
        uint32_t numOutputFeatures = (rand() % 300) + 1;
        bool hasBias = rand() % 2;

        Tensor featureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));

        float initialLearningRate = 10000.0 / (1 + rand() % 33);
        float decay = ((rand() % 5) / 10.0f);
        float momentum = (1.0f + (rand() % 100)) / 100.0f;
        bool useNesterovMomentum = false;
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

        LayerTestHelper::connectNetwork(layers);

        Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs());
        Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
        // Tensor errorOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
        shared_ptr<Sgd> sgd = make_shared<Sgd>(fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, 0);
        sgd->compile();
        fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(sgd));
        LayerTestHelper::initializeNetwork(layers);

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
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weightsGradient_h.copyFromAsync(weightsGradientGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biasesGradient_h.copyFromAsync(biasesGradientGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
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
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weightsGradient_h.copyFromAsync(weightsGradientGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biasesGradient_h.copyFromAsync(biasesGradientGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
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
        verifyMatricesMatch(weightsGradientMem_h, weightsGradientGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weightsGradient_h.copyFromAsync(weightsGradientGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, true);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biasesGradient_h.copyFromAsync(biasesGradientGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
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
        Tensor previousWeightsUpdate_h = weights_h.clone();
        half *previousWeightsUpdateMem_h = previousWeightsUpdate_h.getMemPtr<half>();
        previousWeightsUpdate_h.clearAsync(gradientUpdateStream);
        Tensor previousBiasesUpdate_h;
        half *previousBiasesUpdateMem_h;
        if (hasBias) {
            previousBiasesUpdate_h = biases_h.clone();
            previousBiasesUpdateMem_h = previousBiasesUpdate_h.getMemPtr<half>();
            previousBiasesUpdate_h.clearAsync(gradientUpdateStream);
        }
        gradientUpdateStream.synchronize();

        sgd->updateWeights(weights, fullyConnectedLayer->getBiases(), batchSize);

        weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        // Standard momentum
        // WeightUpdate_t = WeightUpdate_t-1 * momentum - (lr * gradient_t) / batchSize
        assert(momentum > 0.0f);
        assert(useNesterovMomentum == false);
        float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, epoch);
        for (uint32_t row = 0; row < numInputFeatures; ++row) {
            for (uint32_t col = 0; col < numOutputFeatures; ++col) {
                uint32_t index = row * numOutputFeatures + col;
                previousWeightsUpdateMem_h[index] = (float)previousWeightsUpdateMem_h[index] * momentum -
                                                    ((float)weightsGradientMem_h[index] * currentLearningRate) / batchSize;
                weightsMem_h[index] += (float)previousWeightsUpdateMem_h[index] / Loss::getLossScalingFactor();
            }
        }
        verifyMatricesMatch(weightsMem_h, weightsMemGpu_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
            gradientUpdateStream.synchronize();

            for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                previousBiasesUpdateMem_h[outputFeature] = (float)previousBiasesUpdateMem_h[outputFeature] * momentum -
                                                           ((float)biasesGradientMem_h[outputFeature] * currentLearningRate) / batchSize;
                biasesMem_h[outputFeature] += (float)previousBiasesUpdateMem_h[outputFeature] / Loss::getLossScalingFactor();
            }
            verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
        }

        sgd->updateWeights(weights, fullyConnectedLayer->getBiases(), batchSize);

        weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, epoch);
        for (uint32_t row = 0; row < numInputFeatures; ++row) {
            for (uint32_t col = 0; col < numOutputFeatures; ++col) {
                uint32_t index = row * numOutputFeatures + col;
                previousWeightsUpdateMem_h[index] = (float)previousWeightsUpdateMem_h[index] * momentum -
                                                    ((float)weightsGradientMem_h[index] * currentLearningRate) / batchSize;
                weightsMem_h[index] += (float)previousWeightsUpdateMem_h[index] / Loss::getLossScalingFactor();
            }
        }
        verifyMatricesMatch(weightsMem_h, weightsMemGpu_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
        gradientUpdateStream.synchronize();

        if (hasBias) {
            biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
            gradientUpdateStream.synchronize();

            for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                previousBiasesUpdateMem_h[outputFeature] = (float)previousBiasesUpdateMem_h[outputFeature] * momentum -
                                                           ((float)biasesGradientMem_h[outputFeature] * currentLearningRate) / batchSize;
                biasesMem_h[outputFeature] += (float)previousBiasesUpdateMem_h[outputFeature] / Loss::getLossScalingFactor();
            }
            verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
            gradientUpdateStream.synchronize();
        }
    }
}

TEST(FullyConnectedTest, BackwardProducesCorrectErrorOutputAndWeightGradient) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());

    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);

    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpuPlacement));

    LayerTestHelper::connectNetwork(layers);

    // Attach SGD only if needed to expose weightsGradient.
    // Hyperparams don't matter since we won't call update.
    auto sgd = std::make_shared<Sgd>(fc,
                                     /*lr*/ 0.1f,
                                     /*decay*/ 0.0f,
                                     /*momentum*/ 0.0f,
                                     /*useNesterov*/ false,
                                     /*resumeEpoch*/ 0);
    sgd->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(sgd));

    LayerTestHelper::initializeNetwork(layers);

    Stream stream = layers.front()->getStream();

    Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
    Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorInputs());    // dY
    Tensor errorOutput = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorOutputs());  // dX
    Tensor weights = fc->getWeights();

    Tensor weightsGradient = sgd->getWeightsGradient();

    // ----------------------------
    // Set deterministic X, dY, W
    // ----------------------------
    // X = [[1,2],[3,4]]
    Tensor x_h = featureInput.clone(cpuPlacement);
    half *x = x_h.getMemPtr<half>();
    x[0] = (half)1;
    x[1] = (half)2;
    x[2] = (half)3;
    x[3] = (half)4;
    featureInput.copyFromAsync(x_h, stream);

    // dY = [[1,0],[0,1]] (identity)
    Tensor dy_h = errorInput.clone(cpuPlacement);
    half *dy = dy_h.getMemPtr<half>();
    dy[0] = (half)1;
    dy[1] = (half)0;
    dy[2] = (half)0;
    dy[3] = (half)1;
    errorInput.copyFromAsync(dy_h, stream);

    // W = [[1,2],[3,4]]
    // (not symmetric, so transpose mistakes show up)
    Tensor w_h = weights.clone(cpuPlacement);
    half *w = w_h.getMemPtr<half>();
    w[0] = (half)1;
    w[1] = (half)2;
    w[2] = (half)3;
    w[3] = (half)4;
    weights.copyFromAsync(w_h, stream);

    stream.synchronize();

    // Forward once (some implementations require forward before backward)
    fc->forward(featureInput, /*inference=*/true);
    stream.synchronize();

    // Backward
    fc->backward(errorInput);

    // ----------------------------
    // Check dX = dY * W^T
    // With dY=I, dX should equal W^T:
    // dX = [[1,3],[2,4]]
    // ----------------------------
    Tensor dx_gpu_h = errorOutput.clone(cpuPlacement);
    dx_gpu_h.copyFromAsync(errorOutput, stream);
    stream.synchronize();

    half *dxg = dx_gpu_h.getMemPtr<half>();
    EXPECT_NEAR((float)dxg[0], 1.0f, 1e-3f);
    EXPECT_NEAR((float)dxg[1], 3.0f, 1e-3f);
    EXPECT_NEAR((float)dxg[2], 2.0f, 1e-3f);
    EXPECT_NEAR((float)dxg[3], 4.0f, 1e-3f);

    // ----------------------------
    // Check dW = X^T * dY
    // With dY=I, dW should equal X^T:
    // dW = [[1,3],[2,4]]
    // ----------------------------
    // Important: gradients might be produced on a different stream; your long test used sgd->getGradientUpdateStream()
    Stream gradStream = sgd->getGradientUpdateStream();
    gradStream.synchronize();

    Tensor dw_gpu_h = weightsGradient.clone(cpuPlacement);
    dw_gpu_h.copyFromAsync(weightsGradient, gradStream);
    gradStream.synchronize();

    half *dwg = dw_gpu_h.getMemPtr<half>();
    EXPECT_NEAR((float)dwg[0], 1.0f, 1e-3f);
    EXPECT_NEAR((float)dwg[1], 3.0f, 1e-3f);
    EXPECT_NEAR((float)dwg[2], 2.0f, 1e-3f);
    EXPECT_NEAR((float)dwg[3], 4.0f, 1e-3f);
}

// Forward-only test:
//  - No backward
//  - No SGD step
//  - Deterministic constants (no random)
//  - Verifies:
//      * forward(inference=false) uses projected weights (if present)
//      * forward(inference=true)  uses real weights
TEST(FullyConnectedTest, ForwardUsesProjectedWeightsOnlyInTrainingMode) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    // Tiny deterministic shapes
    const uint32_t batchSize = 2;
    const uint32_t numInputFeatures = 2;
    const uint32_t numOutputFeatures = 2;
    const bool hasBias = false;

    // Build a minimal network: Input -> FC -> Output
    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(
        std::make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, numInputFeatures}));
    layers.push_back(std::make_shared<NoOpLayer>());

    shared_ptr<FullyConnected> fc = std::make_shared<FullyConnected>(numOutputFeatures, hasBias);
    layers.push_back(fc);

    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpuPlacement));

    LayerTestHelper::connectNetwork(layers);

    float initialLearningRate = 0.1f;
    float decay = 0.2;
    float momentum = 0.3;
    bool useNesterovMomentum = true;
    shared_ptr<Sgd> sgd = make_shared<Sgd>(fc, initialLearningRate, decay, momentum, useNesterovMomentum, 0);
    sgd->compile();
    fc->setOptimizer(dynamic_pointer_cast<Optimizer>(sgd));

    LayerTestHelper::initializeNetwork(layers);

    Stream dataStream = layers.front()->getStream();
    // Stream gradientUpdateStream = sgd->getGradientUpdateStream();

    // Grab tensors
    Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
    Tensor featureOutput = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureOutputs());
    Tensor weights = fc->getWeights();

    // ---- Fill INPUT deterministically ----
    // X = [[1, 2],
    //      [3, 4]]
    Tensor featureInput_h = featureInput.clone(cpuPlacement);
    half *x = featureInput_h.getMemPtr<half>();
    x[0] = (half)1.0f;
    x[1] = (half)2.0f;
    x[2] = (half)3.0f;
    x[3] = (half)4.0f;
    featureInput.copyFromAsync(featureInput_h, dataStream);

    // ---- Fill WEIGHTS deterministically ----
    // W = I = [[1, 0],
    //          [0, 1]]
    Tensor weights_h = weights.clone(cpuPlacement);
    half *w = weights_h.getMemPtr<half>();
    w[0] = (half)1.0f;
    w[1] = (half)0.0f;
    w[2] = (half)0.0f;
    w[3] = (half)1.0f;
    weights.copyFromAsync(weights_h, dataStream);
    dataStream.synchronize();

    // Helper CPU output buffers
    Tensor featureOutput_h = featureOutput.clone(cpuPlacement);
    half *y_expected = featureOutput_h.getMemPtr<half>();

    Tensor featureOutputGpu_h = featureOutput_h.clone();
    half *y_gpu = featureOutputGpu_h.getMemPtr<half>();

    // -------------------------------------------------------
    // (2) Forward in INFERENCE mode: should use REAL weights
    // -------------------------------------------------------
    fc->forward(featureInput, /*inference=*/true);

    featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
    dataStream.synchronize();

    // Expected: Y = X * W (identity) = X
    y_expected[0] = (half)1.0f;
    y_expected[1] = (half)2.0f;
    y_expected[2] = (half)3.0f;
    y_expected[3] = (half)4.0f;

    EXPECT_EQ(y_expected[0], y_gpu[0]);
    EXPECT_EQ(y_expected[1], y_gpu[1]);
    EXPECT_EQ(y_expected[2], y_gpu[2]);
    EXPECT_EQ(y_expected[3], y_gpu[3]);

    // Ensure that projected weights were initialized to match the regular weights
    ASSERT_TRUE(fc->hasProjectedWeights());
    Tensor projectedWeights = fc->getProjectedWeightsTensor().get();
    Tensor projectedWeights_h = projectedWeights.clone(cpuPlacement);
    half *wp = projectedWeights_h.getMemPtr<half>();
    projectedWeights_h.copyFromAsync(projectedWeights, dataStream);
    dataStream.synchronize();
    // gradientUpdateStream.synchronize();

    EXPECT_EQ(w[0], wp[0]);
    EXPECT_EQ(w[1], wp[1]);
    EXPECT_EQ(w[2], wp[2]);
    EXPECT_EQ(w[3], wp[3]);

    // ---- Force PROJECTED WEIGHTS to be different ----
    // WP = [[2, 0],
    //       [0, 2]]
    //
    // You may need to adjust this depending on how your layer exposes projected weights.
    // In your earlier snippet you used:
    //   fullyConnectedLayer->getProjectedWeightsTensor()
    // so we try that first.
    wp[0] = (half)2.0f;
    wp[1] = (half)0.0f;
    wp[2] = (half)0.0f;
    wp[3] = (half)2.0f;
    projectedWeights.copyFromAsync(projectedWeights_h, dataStream);

    dataStream.synchronize();

    // Read back the projected weights immediately after writing them.
    Tensor projectedWeightsReadback0 = projectedWeights.clone(cpuPlacement);
    projectedWeightsReadback0.copyFromAsync(projectedWeights, dataStream);
    dataStream.synchronize();
    half *pr = projectedWeightsReadback0.getMemPtr<half>();
    EXPECT_EQ((float)pr[0], 2.0f);
    EXPECT_EQ((float)pr[1], 0.0f);
    EXPECT_EQ((float)pr[2], 0.0f);
    EXPECT_EQ((float)pr[3], 2.0f);

    // -------------------------------------------------------
    // (1) Forward in TRAINING mode: should use PROJECTED weights
    // -------------------------------------------------------
    fc->forward(featureInput, /*inference=*/false);

    featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
    dataStream.synchronize();

    // Expected: Y = X * WP
    // With X as above and WP=2I:
    // Y = [[2,4],
    //      [6,8]]
    y_expected[0] = (half)2.0f;
    y_expected[1] = (half)4.0f;
    y_expected[2] = (half)6.0f;
    y_expected[3] = (half)8.0f;

    EXPECT_EQ(y_expected[0], y_gpu[0]);
    EXPECT_EQ(y_expected[1], y_gpu[1]);
    EXPECT_EQ(y_expected[2], y_gpu[2]);
    EXPECT_EQ(y_expected[3], y_gpu[3]);

    // -------------------------------------------------------
    // Optional: prove projected weights weren't overwritten during inference forward.
    // (Only keep this if your implementation guarantees that.)
    // -------------------------------------------------------
    Tensor projectedWeightsReadback = projectedWeights.clone(cpuPlacement);
    projectedWeightsReadback.copyFromAsync(projectedWeights, dataStream);
    dataStream.synchronize();
    half *wp2 = projectedWeightsReadback.getMemPtr<half>();
    EXPECT_EQ((float)wp2[0], 2.0f);
    EXPECT_EQ((float)wp2[1], 0.0f);
    EXPECT_EQ((float)wp2[2], 0.0f);
    EXPECT_EQ((float)wp2[3], 2.0f);
}

TEST(SgdTest, NesterovMomentum_SingleStep_UpdateAndProjectionAreCorrect) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    // Build minimal net so FC + SGD are fully constructed
    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpuPlacement));
    LayerTestHelper::connectNetwork(layers);

    // Hyperparams: deterministic and simple
    const float lr0 = 0.1f;
    const float decay = 0.0f;
    const float mu = 0.5f;
    const bool useNesterov = true;

    auto sgd = std::make_shared<Sgd>(fc, lr0, decay, mu, useNesterov);
    sgd->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(sgd));

    LayerTestHelper::initializeNetwork(layers);
    Stream stream = layers.front()->getStream();

    // Ensure lr used is exactly lr0
    sgd->updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/1);

    Tensor weights = fc->getWeights();
    Tensor weights_h = weights.clone(cpuPlacement);
    half *w0 = weights_h.getMemPtr<half>();

    // W0 = [[ 1,  2],
    //       [ 3,  4]]
    w0[0] = (half)1.0f;
    w0[1] = (half)2.0f;
    w0[2] = (half)3.0f;
    w0[3] = (half)4.0f;
    weights.copyFromAsync(weights_h, stream);

    // Get internal tensors
    Tensor grad = sgd->getWeightsGradient();
    Tensor vel = sgd->getWeightsUpdate();
    Tensor proj = sgd->getProjectedWeights();

    // Initialize velocity to zero (important!)
    // If there is no clearAsync, fill CPU zeros and copy.
    Tensor vel_h = vel.clone(cpuPlacement);
    std::fill_n(vel_h.getMemPtr<half>(), 4, (half)0.0f);
    vel.copyFromAsync(vel_h, stream);

    // Inject a known gradient G (FP16 exact values)
    // G = [[10, 20],
    //      [30, 40]]
    Tensor grad_h = grad.clone(cpuPlacement);
    half *g = grad_h.getMemPtr<half>();
    g[0] = (half)10.0f;
    g[1] = (half)20.0f;
    g[2] = (half)30.0f;
    g[3] = (half)40.0f;
    grad.copyFromAsync(grad_h, stream);

    stream.synchronize();

    // ---- Act: call the update ----
    // You want specifically Sgd::updateWeights(weights, biases, batchSize).
    // If updateWeights is not public, call the public method that triggers it
    // (e.g., fc->updateWeights(batchSize) or sgd->step(batchSize), etc.)
    //
    sgd->updateWeights(weights, Optional<Tensor>(), batchSize);

    // Ensure update stream finished
    sgd->getGradientUpdateStream().synchronize();

    // ---- Compute CPU expected using the *implementation's* math ----
    const float lr = lr0;  // if hyperparams update modifies this, use sgd->getCurrentLearningRate()

    const float LS = Loss::getLossScalingFactor();
    const float invLS = 1.0f / LS;

    // u = mu*u - (lr/batch)*g          (NO /LS here)
    // w = w + u/LS
    // proj = w + mu*u/LS
    float u_exp[4];
    float w_exp[4];
    float p_exp[4];

    for (int i = 0; i < 4; ++i) {
        float w0f = (float)w0[i];
        float gf = (float)g[i];
        float u0f = 0.0f;

        float u1 = mu * u0f + (-lr / batchSize) * gf;
        float w1 = w0f + u1 * invLS;
        float p1 = w1 + mu * u1 * invLS;

        u_exp[i] = u1;
        w_exp[i] = w1;
        p_exp[i] = p1;
    }

    // ---- Read back GPU results ----
    Tensor weights_gpu_h = weights.clone(cpuPlacement);
    Tensor proj_gpu_h = proj.clone(cpuPlacement);
    weights_gpu_h.copyFromAsync(weights, sgd->getGradientUpdateStream());
    proj_gpu_h.copyFromAsync(proj, sgd->getGradientUpdateStream());
    sgd->getGradientUpdateStream().synchronize();

    half *w_gpu = weights_gpu_h.getMemPtr<half>();
    half *p_gpu = proj_gpu_h.getMemPtr<half>();

    // verify velocity too
    Tensor vel_gpu_h = vel.clone(cpuPlacement);
    vel_gpu_h.copyFromAsync(vel, sgd->getGradientUpdateStream());
    sgd->getGradientUpdateStream().synchronize();
    half *u_gpu = vel_gpu_h.getMemPtr<half>();

    auto expectNear = [](float a, float b, float eps, const char *msg) {
        EXPECT_LE(std::abs(a - b), eps) << msg << " expected=" << a << " got=" << b;
    };

    // EPS: FP16 rounding.
    const float eps = 5e-3f;

    for (int i = 0; i < 4; ++i) {
        expectNear(u_exp[i], (float)u_gpu[i], eps, "velocity");
        expectNear(w_exp[i], (float)w_gpu[i], eps, "weights");
        expectNear(p_exp[i], (float)p_gpu[i], eps, "projected");
    }
}

TEST(SgdTest, NesterovMomentum_TwoStep_UpdateCarriesMomentum) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpuPlacement));
    LayerTestHelper::connectNetwork(layers);

    const float lr0 = 0.1f;
    const float decay = 0.0f;
    const float mu = 0.5f;
    const bool useNesterov = true;

    auto sgd = std::make_shared<Sgd>(fc, lr0, decay, mu, useNesterov);
    sgd->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(sgd));

    LayerTestHelper::initializeNetwork(layers);
    Stream stream = layers.front()->getStream();

    sgd->updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/1);

    Tensor weights = fc->getWeights();
    Tensor grad = sgd->getWeightsGradient();
    Tensor vel = sgd->getWeightsUpdate();
    Tensor proj = sgd->getProjectedWeights();

    // W0 = [[ 1,  2],
    //       [ 3,  4]]
    Tensor weights_h = weights.clone(cpuPlacement);
    half *w0 = weights_h.getMemPtr<half>();
    w0[0] = (half)1.0f;
    w0[1] = (half)2.0f;
    w0[2] = (half)3.0f;
    w0[3] = (half)4.0f;
    weights.copyFromAsync(weights_h, stream);

    // u0 = 0
    Tensor vel_h = vel.clone(cpuPlacement);
    std::fill_n(vel_h.getMemPtr<half>(), 4, (half)0.0f);
    vel.copyFromAsync(vel_h, stream);

    stream.synchronize();

    const float lr = lr0;
    const float LS = Loss::getLossScalingFactor();
    const float invLS = 1.0f / LS;

    auto expectNear = [](float a, float b, float eps, const char *msg) {
        EXPECT_LE(std::abs(a - b), eps) << msg << " expected=" << a << " got=" << b;
    };
    const float eps = 5e-3f;

    // Helper to write a 2x2 gradient
    auto writeGrad = [&](half g00, half g01, half g10, half g11) {
        Tensor grad_h = grad.clone(cpuPlacement);
        half *g = grad_h.getMemPtr<half>();
        g[0] = g00;
        g[1] = g01;
        g[2] = g10;
        g[3] = g11;
        grad.copyFromAsync(grad_h, stream);
        stream.synchronize();
    };

    // ------------------------
    // Step 1 with G1
    // G1 = [[10, 20],
    //       [30, 40]]
    // ------------------------
    writeGrad((half)10.0f, (half)20.0f, (half)30.0f, (half)40.0f);

    sgd->updateWeights(weights, Optional<Tensor>(), batchSize);
    sgd->getGradientUpdateStream().synchronize();

    // CPU expected after step 1
    float u1[4], w1[4], p1[4];
    for (int i = 0; i < 4; ++i) {
        float w0f = (float)w0[i];
        float g1f;
        switch (i) {
            case 0:
                g1f = 10.0f;
                break;
            case 1:
                g1f = 20.0f;
                break;
            case 2:
                g1f = 30.0f;
                break;
            default:
                g1f = 40.0f;
                break;
        }
        float u0f = 0.0f;
        float u = mu * u0f + (-lr / batchSize) * g1f;  // no /LS
        float w = w0f + u * invLS;
        float p = w + mu * u * invLS;
        u1[i] = u;
        w1[i] = w;
        p1[i] = p;
    }

    // Read back GPU after step 1
    Tensor w_gpu_h1 = weights.clone(cpuPlacement);
    Tensor p_gpu_h1 = proj.clone(cpuPlacement);
    Tensor u_gpu_h1 = vel.clone(cpuPlacement);
    w_gpu_h1.copyFromAsync(weights, sgd->getGradientUpdateStream());
    p_gpu_h1.copyFromAsync(proj, sgd->getGradientUpdateStream());
    u_gpu_h1.copyFromAsync(vel, sgd->getGradientUpdateStream());
    sgd->getGradientUpdateStream().synchronize();

    half *w_gpu1 = w_gpu_h1.getMemPtr<half>();
    half *p_gpu1 = p_gpu_h1.getMemPtr<half>();
    half *u_gpu1 = u_gpu_h1.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        expectNear(u1[i], (float)u_gpu1[i], eps, "step1 velocity");
        expectNear(w1[i], (float)w_gpu1[i], eps, "step1 weights");
        expectNear(p1[i], (float)p_gpu1[i], eps, "step1 projected");
    }

    // ------------------------
    // Step 2 with a DIFFERENT gradient G2
    // Pick something FP16-exact and different to ensure momentum matters.
    // G2 = [[ 1,  2],
    //       [ 3,  4]]
    // ------------------------
    writeGrad((half)1.0f, (half)2.0f, (half)3.0f, (half)4.0f);

    sgd->updateWeights(weights, Optional<Tensor>(), batchSize);
    sgd->getGradientUpdateStream().synchronize();

    // CPU expected after step 2
    float u2[4], w2[4], p2[4];
    for (int i = 0; i < 4; ++i) {
        float g2f;
        switch (i) {
            case 0:
                g2f = 1.0f;
                break;
            case 1:
                g2f = 2.0f;
                break;
            case 2:
                g2f = 3.0f;
                break;
            default:
                g2f = 4.0f;
                break;
        }

        float u = mu * u1[i] + (-lr / batchSize) * g2f;  // carry momentum
        float w = w1[i] + u * invLS;
        float p = w + mu * u * invLS;
        u2[i] = u;
        w2[i] = w;
        p2[i] = p;
    }

    // Read back GPU after step 2
    Tensor w_gpu_h2 = weights.clone(cpuPlacement);
    Tensor p_gpu_h2 = proj.clone(cpuPlacement);
    Tensor u_gpu_h2 = vel.clone(cpuPlacement);
    w_gpu_h2.copyFromAsync(weights, sgd->getGradientUpdateStream());
    p_gpu_h2.copyFromAsync(proj, sgd->getGradientUpdateStream());
    u_gpu_h2.copyFromAsync(vel, sgd->getGradientUpdateStream());
    sgd->getGradientUpdateStream().synchronize();

    half *w_gpu2 = w_gpu_h2.getMemPtr<half>();
    half *p_gpu2 = p_gpu_h2.getMemPtr<half>();
    half *u_gpu2 = u_gpu_h2.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        expectNear(u2[i], (float)u_gpu2[i], eps, "step2 velocity");
        expectNear(w2[i], (float)w_gpu2[i], eps, "step2 weights");
        expectNear(p2[i], (float)p_gpu2[i], eps, "step2 projected");
    }
}

TEST(SgdTest, NesterovMomentum_Integrated_ForwardBackwardUpdate_ProjectedUsedInTraining) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    // Build minimal net: Input -> FC -> Output
    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpuPlacement));
    LayerTestHelper::connectNetwork(layers);

    // Deterministic hyperparams
    const float lr0 = 0.1f;
    const float decay = 0.0f;
    const float mu = 0.5f;
    const bool useNesterov = true;

    auto sgd = std::make_shared<Sgd>(fc, lr0, decay, mu, useNesterov);
    sgd->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(sgd));

    LayerTestHelper::initializeNetwork(layers);

    Stream stream = layers.front()->getStream();
    sgd->updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/1);

    // Grab tensors
    Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
    Tensor featureOutput = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureOutputs());
    Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorInputs());

    Tensor weights = fc->getWeights();

    // These exist only because Nesterov+optimizer is attached
    Tensor vel = sgd->getWeightsUpdate();
    Tensor grad = sgd->getWeightsGradient();
    Tensor proj = sgd->getProjectedWeights();

    // ----------------------------
    // Set deterministic X, dY, W0
    // ----------------------------
    // X = [[1, 2],
    //      [3, 4]]
    Tensor x_h = featureInput.clone(cpuPlacement);
    half *x = x_h.getMemPtr<half>();
    x[0] = (half)1.0f;
    x[1] = (half)2.0f;
    x[2] = (half)3.0f;
    x[3] = (half)4.0f;
    featureInput.copyFromAsync(x_h, stream);

    // dY = [[1, 0],
    //       [0, 1]]  (identity-ish)
    Tensor dy_h = errorInput.clone(cpuPlacement);
    half *dy = dy_h.getMemPtr<half>();
    dy[0] = (half)1.0f;
    dy[1] = (half)0.0f;
    dy[2] = (half)0.0f;
    dy[3] = (half)1.0f;
    errorInput.copyFromAsync(dy_h, stream);

    // W0 = [[1, 2],
    //       [3, 4]]
    Tensor w0_h = weights.clone(cpuPlacement);
    half *w0 = w0_h.getMemPtr<half>();
    w0[0] = (half)1.0f;
    w0[1] = (half)2.0f;
    w0[2] = (half)3.0f;
    w0[3] = (half)4.0f;
    weights.copyFromAsync(w0_h, stream);

    // Force velocity to 0 so the step is unambiguous
    Tensor vel0_h = vel.clone(cpuPlacement);
    std::fill_n(vel0_h.getMemPtr<half>(), 4, (half)0.0f);
    vel.copyFromAsync(vel0_h, stream);

    stream.synchronize();

    // ---------------------------------------------------
    // Forward (training=true) sanity check
    // ---------------------------------------------------
    // Inference should use real weights (W0)
    {
        fc->forward(featureInput, /*inference=*/false);
        Tensor y_gpu_h = featureOutput.clone(cpuPlacement);
        y_gpu_h.copyFromAsync(featureOutput, stream);
        stream.synchronize();
        half *y = y_gpu_h.getMemPtr<half>();

        // Y = X * W0
        // [ [1,2] * [[1,2],[3,4]] ] = [1*1+2*3, 1*2+2*4] = [7, 10]
        // [ [3,4] * [[1,2],[3,4]] ] = [3*1+4*3, 3*2+4*4] = [15, 22]
        EXPECT_EQ(y[0], (half)7.0f);
        EXPECT_EQ(y[1], (half)10.0f);
        EXPECT_EQ(y[2], (half)15.0f);
        EXPECT_EQ(y[3], (half)22.0f);
    }

    // -------------------------------------------
    // Sanity: projected weights start == weights
    //         Copied on first forward pass
    // -------------------------------------------
    {
        Tensor w_rb = weights.clone(cpuPlacement);
        Tensor p_rb = proj.clone(cpuPlacement);
        w_rb.copyFromAsync(weights, stream);
        p_rb.copyFromAsync(proj, stream);
        stream.synchronize();

        half *ww = w_rb.getMemPtr<half>();
        half *pp = p_rb.getMemPtr<half>();
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(ww[i], pp[i]) << "Projected weights should initialize equal to weights.";
        }
    }

    // ----------------------------
    // Backward: produces dW and updates weights/proj via SGD
    // ----------------------------
    fc->backward(errorInput);

    sgd->getGradientUpdateStream().synchronize();

    // ---------------------------------------
    // Compute expected dW and expected updates
    // ---------------------------------------
    // For FC: dW = X^T * dY
    // X^T = [[1,3],[2,4]]
    // dY  = [[1,0],[0,1]]
    // dW = [[1,3],[2,4]]  (because dY is identity-ish)
    const float dW_exp[4] = {1.0f, 3.0f, 2.0f, 4.0f};

    const float lr = lr0;
    const float LS = Loss::getLossScalingFactor();
    const float invLS = 1.0f / LS;

    // u1 = mu*u0 - (lr/batch)*dW       (u0=0)
    // w1 = w0 + u1/LS
    // p1 = w1 + mu*u1/LS
    float u1_exp[4], w1_exp[4], p1_exp[4];
    for (int i = 0; i < 4; ++i) {
        float u1 = (mu * 0.0f) + (-lr / batchSize) * dW_exp[i];  // NO /LS here
        float w1 = (float)w0[i] + u1 * invLS;
        float p1 = w1 + mu * u1 * invLS;
        u1_exp[i] = u1;
        w1_exp[i] = w1;
        p1_exp[i] = p1;
    }

    // Read back GPU tensors after backward/update
    Tensor w1_gpu_h = weights.clone(cpuPlacement);
    Tensor p1_gpu_h = proj.clone(cpuPlacement);
    Tensor u1_gpu_h = vel.clone(cpuPlacement);
    Tensor dW_gpu_h = grad.clone(cpuPlacement);

    w1_gpu_h.copyFromAsync(weights, sgd->getGradientUpdateStream());
    p1_gpu_h.copyFromAsync(proj, sgd->getGradientUpdateStream());
    u1_gpu_h.copyFromAsync(vel, sgd->getGradientUpdateStream());
    dW_gpu_h.copyFromAsync(grad, sgd->getGradientUpdateStream());
    sgd->getGradientUpdateStream().synchronize();

    half *w1_gpu = w1_gpu_h.getMemPtr<half>();
    half *p1_gpu = p1_gpu_h.getMemPtr<half>();
    half *u1_gpu = u1_gpu_h.getMemPtr<half>();
    half *dW_gpu = dW_gpu_h.getMemPtr<half>();

    auto expectNear = [](float a, float b, float eps, const char *msg) {
        EXPECT_LE(std::abs(a - b), eps) << msg << " expected=" << a << " got=" << b;
    };
    const float eps = 5e-3f;

    // Optional gradient check (you said you trust backward; keep it anyway because it localizes failures)
    for (int i = 0; i < 4; ++i) {
        expectNear(dW_exp[i], (float)dW_gpu[i], eps, "dW");
    }

    for (int i = 0; i < 4; ++i) {
        expectNear(u1_exp[i], (float)u1_gpu[i], eps, "velocity u1");
        expectNear(w1_exp[i], (float)w1_gpu[i], eps, "weights w1");
        expectNear(p1_exp[i], (float)p1_gpu[i], eps, "projected p1");
    }

    // ---------------------------------------------------------
    // Now verify forward uses projected weights in training mode,
    // and real weights in inference mode (post-update).
    // ---------------------------------------------------------
    // Use a new X2 so we’re not accidentally matching the old output.
    // X2 = [[ 2, 1],
    //       [ 0, 1]]
    Tensor x2_h = featureInput.clone(cpuPlacement);
    half *x2 = x2_h.getMemPtr<half>();
    x2[0] = (half)2.0f;
    x2[1] = (half)1.0f;
    x2[2] = (half)0.0f;
    x2[3] = (half)1.0f;
    featureInput.copyFromAsync(x2_h, stream);
    stream.synchronize();

    auto matmul2x2_rowmajor = [](const float X[4], const float W[4], float Y[4]) {
        // X: (2x2), W: (2x2), Y = X*W
        Y[0] = X[0] * W[0] + X[1] * W[2];
        Y[1] = X[0] * W[1] + X[1] * W[3];
        Y[2] = X[2] * W[0] + X[3] * W[2];
        Y[3] = X[2] * W[1] + X[3] * W[3];
    };

    const float X2f[4] = {2.0f, 1.0f, 0.0f, 1.0f};

    float Y_train_exp[4];
    float Y_inf_exp[4];
    matmul2x2_rowmajor(X2f, p1_exp, Y_train_exp);  // training should use projected
    matmul2x2_rowmajor(X2f, w1_exp, Y_inf_exp);    // inference should use real weights

    // Training forward (inference=false) -> projected
    fc->forward(featureInput, /*inference=*/false);
    Tensor y_train_gpu_h = featureOutput.clone(cpuPlacement);
    y_train_gpu_h.copyFromAsync(featureOutput, stream);
    stream.synchronize();
    half *y_train = y_train_gpu_h.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        expectNear(Y_train_exp[i], (float)y_train[i], 1e-2f, "Y training (projected)");
    }

    // Inference forward (inference=true) -> real weights
    fc->forward(featureInput, /*inference=*/true);
    Tensor y_inf_gpu_h = featureOutput.clone(cpuPlacement);
    y_inf_gpu_h.copyFromAsync(featureOutput, stream);
    stream.synchronize();
    half *y_inf = y_inf_gpu_h.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        expectNear(Y_inf_exp[i], (float)y_inf[i], 1e-2f, "Y inference (weights)");
    }
}

TEST(SgdTest, NesterovMomentum_Integrated_TwoIterations_CarriesMomentumAndUsesProjection) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = false;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpuPlacement));
    LayerTestHelper::connectNetwork(layers);

    const float lr0 = 0.1f;
    const float decay = 0.6f;
    const float mu = 0.5f;
    const bool useNesterov = true;

    auto sgd = std::make_shared<Sgd>(fc, lr0, decay, mu, useNesterov);
    sgd->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(sgd));

    LayerTestHelper::initializeNetwork(layers);

    Stream stream = layers.front()->getStream();
    sgd->updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/1);

    Tensor x = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
    Tensor y = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureOutputs());
    Tensor dy = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorInputs());

    Tensor w = fc->getWeights();
    Tensor u = sgd->getWeightsUpdate();
    Tensor g = sgd->getWeightsGradient();
    Tensor p = sgd->getProjectedWeights();

    // W0 = [[1,2],[3,4]]
    Tensor w0_h = w.clone(cpuPlacement);
    half *w0 = w0_h.getMemPtr<half>();
    w0[0] = (half)1.0f;
    w0[1] = (half)2.0f;
    w0[2] = (half)3.0f;
    w0[3] = (half)4.0f;
    w.copyFromAsync(w0_h, stream);

    // u0 = 0
    Tensor u0_h = u.clone(cpuPlacement);
    std::fill_n(u0_h.getMemPtr<half>(), 4, (half)0.0f);
    u.copyFromAsync(u0_h, stream);

    stream.synchronize();

    float lr = lr0;
    const float LS = Loss::getLossScalingFactor();
    const float invLS = 1.0f / LS;

    auto expectNear = [](float a, float b, float eps, const char *msg) {
        EXPECT_LE(std::abs(a - b), eps) << msg << " expected=" << a << " got=" << b;
    };
    const float eps = 5e-3f;

    auto matmul2x2_rowmajor = [](const float A[4], const float B[4], float C[4]) {
        C[0] = A[0] * B[0] + A[1] * B[2];
        C[1] = A[0] * B[1] + A[1] * B[3];
        C[2] = A[2] * B[0] + A[3] * B[2];
        C[3] = A[2] * B[1] + A[3] * B[3];
    };

    auto transpose2x2 = [](const float M[4], float Mt[4]) {
        Mt[0] = M[0];
        Mt[1] = M[2];
        Mt[2] = M[1];
        Mt[3] = M[3];
    };

    // Helper: set X and dY on GPU from CPU floats
    auto set2x2 = [&](Tensor t, float a00, float a01, float a10, float a11) {
        Tensor th = t.clone(cpuPlacement);
        half *m = th.getMemPtr<half>();
        m[0] = (half)a00;
        m[1] = (half)a01;
        m[2] = (half)a10;
        m[3] = (half)a11;
        t.copyFromAsync(th, stream);
    };

    // CPU state holders
    float w_exp[4] = {1, 2, 3, 4};
    float u_exp[4] = {0, 0, 0, 0};
    float p_exp[4] = {0, 0, 0, 0};

    // =========================
    // Iteration 1
    // X1 = [[1,2],[3,4]]
    // dY1 = [[1,0],[0,1]]
    // =========================
    set2x2(x, 1, 2, 3, 4);
    set2x2(dy, 1, 0, 0, 1);
    stream.synchronize();

    // Training forward first (forces projected init/copy)
    fc->forward(x, /*inference=*/false);
    // Backward triggers update internally
    fc->backward(dy);
    sgd->getGradientUpdateStream().synchronize();

    // CPU expected gradient: dW1 = X1^T * dY1
    const float X1[4] = {1, 2, 3, 4};
    const float dY1[4] = {1, 0, 0, 1};
    float X1T[4];
    transpose2x2(X1, X1T);
    float dW1[4];
    matmul2x2_rowmajor(X1T, dY1, dW1);  // -> [[1,3],[2,4]]

    // CPU expected update (implementation-matching):
    for (int i = 0; i < 4; ++i) {
        u_exp[i] = mu * u_exp[i] + (-lr / batchSize) * dW1[i];
        w_exp[i] = w_exp[i] + u_exp[i] * invLS;
        p_exp[i] = w_exp[i] + mu * u_exp[i] * invLS;
    }

    // Read back GPU u,w,p and optionally gradient
    Tensor w1_h = w.clone(cpuPlacement);
    Tensor p1_h = p.clone(cpuPlacement);
    Tensor u1_h = u.clone(cpuPlacement);
    Tensor g1_h = g.clone(cpuPlacement);
    w1_h.copyFromAsync(w, sgd->getGradientUpdateStream());
    p1_h.copyFromAsync(p, sgd->getGradientUpdateStream());
    u1_h.copyFromAsync(u, sgd->getGradientUpdateStream());
    g1_h.copyFromAsync(g, sgd->getGradientUpdateStream());
    sgd->getGradientUpdateStream().synchronize();

    half *w1 = w1_h.getMemPtr<half>();
    half *p1 = p1_h.getMemPtr<half>();
    half *u1 = u1_h.getMemPtr<half>();
    half *g1 = g1_h.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        expectNear(dW1[i], (float)g1[i], eps, "iter1 dW");
        expectNear(u_exp[i], (float)u1[i], eps, "iter1 u");
        expectNear(w_exp[i], (float)w1[i], eps, "iter1 w");
        expectNear(p_exp[i], (float)p1[i], eps, "iter1 p");
    }

    // Also verify that TRAINING forward uses projected weights p1:
    // Pick Xcheck (small) and compare output against CPU Xcheck*p1.
    set2x2(x, 2, 1, 0, 1);
    stream.synchronize();

    fc->forward(x, /*inference=*/false);
    Tensor ytrain_h = y.clone(cpuPlacement);
    ytrain_h.copyFromAsync(y, stream);
    stream.synchronize();
    half *ytrain = ytrain_h.getMemPtr<half>();

    float Xcheck[4] = {2, 1, 0, 1};
    float Ycheck[4];
    matmul2x2_rowmajor(Xcheck, p_exp, Ycheck);
    for (int i = 0; i < 4; ++i) {
        expectNear(Ycheck[i], (float)ytrain[i], 1e-2f, "iter1 training forward uses projected");
    }

    // =========================
    // Iteration 2
    // Use DIFFERENT X2 and dY2 so momentum effect is observable
    // X2  = [[ 2, 1],
    //        [ 0, 1]]
    // dY2 = [[ 1, 1],
    //        [ 0, 1]]
    // =========================
    set2x2(x, 2, 1, 0, 1);
    set2x2(dy, 1, 1, 0, 1);
    stream.synchronize();

    sgd->updateHyperParameters(/*epoch=*/1, /*batch=*/0, /*batchesPerEpoch=*/1);
    lr *= 1.0f - decay;

    fc->forward(x, /*inference=*/false);
    fc->backward(dy);
    sgd->getGradientUpdateStream().synchronize();

    const float X2[4] = {2, 1, 0, 1};
    const float dY2[4] = {1, 1, 0, 1};
    float X2T[4];
    transpose2x2(X2, X2T);
    float dW2[4];
    matmul2x2_rowmajor(X2T, dY2, dW2);

    for (int i = 0; i < 4; ++i) {
        u_exp[i] = mu * u_exp[i] + (-lr / batchSize) * dW2[i];
        w_exp[i] = w_exp[i] + u_exp[i] * invLS;
        p_exp[i] = w_exp[i] + mu * u_exp[i] * invLS;
    }

    Tensor w2_h = w.clone(cpuPlacement);
    Tensor p2_h = p.clone(cpuPlacement);
    Tensor u2_h = u.clone(cpuPlacement);
    Tensor g2_h = g.clone(cpuPlacement);
    w2_h.copyFromAsync(w, sgd->getGradientUpdateStream());
    p2_h.copyFromAsync(p, sgd->getGradientUpdateStream());
    u2_h.copyFromAsync(u, sgd->getGradientUpdateStream());
    g2_h.copyFromAsync(g, sgd->getGradientUpdateStream());
    sgd->getGradientUpdateStream().synchronize();

    half *w2 = w2_h.getMemPtr<half>();
    half *p2 = p2_h.getMemPtr<half>();
    half *u2 = u2_h.getMemPtr<half>();
    half *g2 = g2_h.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        expectNear(dW2[i], (float)g2[i], eps, "iter2 dW");
        expectNear(u_exp[i], (float)u2[i], eps, "iter2 u");
        expectNear(w_exp[i], (float)w2[i], eps, "iter2 w");
        expectNear(p_exp[i], (float)p2[i], eps, "iter2 p");
    }

    // Finally: inference forward should use REAL weights w2, not projected p2
    fc->forward(x, /*inference=*/true);
    Tensor yinf_h = y.clone(cpuPlacement);
    yinf_h.copyFromAsync(y, stream);
    stream.synchronize();
    half *yinf = yinf_h.getMemPtr<half>();

    float Yinf[4];
    matmul2x2_rowmajor(X2, w_exp, Yinf);
    for (int i = 0; i < 4; ++i) {
        expectNear(Yinf[i], (float)yinf[i], 1e-2f, "iter2 inference forward uses weights");
    }
}

TEST(SgdTest, NesterovMomentum_Integrated_TwoIterations_WithBias) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    const uint32_t batchSize = 2;
    const uint32_t inF = 2;
    const uint32_t outF = 2;
    const bool hasBias = true;

    std::vector<std::shared_ptr<Layer>> layers;
    layers.push_back(std::make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, std::vector<uint64_t>{batchSize, inF}));
    layers.push_back(std::make_shared<NoOpLayer>());
    auto fc = std::make_shared<FullyConnected>(outF, hasBias);
    layers.push_back(fc);
    layers.push_back(std::make_shared<NoOpLayer>());
    layers.push_back(std::make_shared<NetworkOutput>(cpuPlacement));
    LayerTestHelper::connectNetwork(layers);

    const float lr0 = 0.1f;
    const float decay = 0.25f;
    const float mu = 0.5f;
    const bool useNesterov = true;

    auto sgd = std::make_shared<Sgd>(fc, lr0, decay, mu, useNesterov);
    sgd->compile();
    fc->setOptimizer(std::dynamic_pointer_cast<Optimizer>(sgd));

    LayerTestHelper::initializeNetwork(layers);

    Stream stream = layers.front()->getStream();
    sgd->updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/1);

    Tensor x = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureInputs());
    Tensor y = MultiConnectionLayer::getFirstPresentTensor(fc->getFeatureOutputs());
    Tensor dy = MultiConnectionLayer::getFirstPresentTensor(fc->getErrorInputs());

    Tensor w = fc->getWeights();
    Tensor uw = sgd->getWeightsUpdate();
    Tensor gw = sgd->getWeightsGradient();
    Tensor pw = sgd->getProjectedWeights();

    Optional<Tensor> bOpt = fc->getBiases();
    ASSERT_TRUE(bOpt.isPresent());
    Tensor b = bOpt.get();

    Optional<Tensor> ubOpt = sgd->getBiasesUpdate();
    Optional<Tensor> gbOpt = sgd->getBiasesGradient();
    Optional<Tensor> pbOpt = sgd->getProjectedBiases();
    ASSERT_TRUE(ubOpt.isPresent());
    ASSERT_TRUE(gbOpt.isPresent());
    ASSERT_TRUE(pbOpt.isPresent());

    Tensor ub = ubOpt.get();
    Tensor gb = gbOpt.get();
    Tensor pb = pbOpt.get();

    // W0 = [[1,2],[3,4]]
    Tensor w0_h = w.clone(cpuPlacement);
    half *w0 = w0_h.getMemPtr<half>();
    w0[0] = (half)1.0f;
    w0[1] = (half)2.0f;
    w0[2] = (half)3.0f;
    w0[3] = (half)4.0f;
    w.copyFromAsync(w0_h, stream);

    // b0 = [5, 6]
    Tensor b0_h = b.clone(cpuPlacement);
    half *b0 = b0_h.getMemPtr<half>();
    b0[0] = (half)5.0f;
    b0[1] = (half)6.0f;
    b.copyFromAsync(b0_h, stream);

    // uw0 = 0, ub0 = 0
    Tensor uw0_h = uw.clone(cpuPlacement);
    std::fill_n(uw0_h.getMemPtr<half>(), 4, (half)0.0f);
    uw.copyFromAsync(uw0_h, stream);

    Tensor ub0_h = ub.clone(cpuPlacement);
    std::fill_n(ub0_h.getMemPtr<half>(), outF, (half)0.0f);
    ub.copyFromAsync(ub0_h, stream);

    stream.synchronize();

    float lr = lr0;
    const float LS = Loss::getLossScalingFactor();
    const float invLS = 1.0f / LS;

    auto expectNear = [](float a, float b, float eps, const char *msg) {
        EXPECT_LE(std::abs(a - b), eps) << msg << " expected=" << a << " got=" << b;
    };
    const float eps = 5e-3f;

    auto matmul2x2_rowmajor = [](const float A[4], const float B[4], float C[4]) {
        C[0] = A[0] * B[0] + A[1] * B[2];
        C[1] = A[0] * B[1] + A[1] * B[3];
        C[2] = A[2] * B[0] + A[3] * B[2];
        C[3] = A[2] * B[1] + A[3] * B[3];
    };
    auto transpose2x2 = [](const float M[4], float Mt[4]) {
        Mt[0] = M[0];
        Mt[1] = M[2];
        Mt[2] = M[1];
        Mt[3] = M[3];
    };

    auto set2x2 = [&](Tensor t, float a00, float a01, float a10, float a11) {
        Tensor th = t.clone(cpuPlacement);
        half *m = th.getMemPtr<half>();
        m[0] = (half)a00;
        m[1] = (half)a01;
        m[2] = (half)a10;
        m[3] = (half)a11;
        t.copyFromAsync(th, stream);
    };
    auto setBias = [&](Tensor t, float v0, float v1) {
        Tensor th = t.clone(cpuPlacement);
        half *m = th.getMemPtr<half>();
        m[0] = (half)v0;
        m[1] = (half)v1;
        t.copyFromAsync(th, stream);
    };

    // CPU expected state holders
    float w_exp[4] = {1, 2, 3, 4};
    float b_exp[2] = {5, 6};
    float uw_exp[4] = {0, 0, 0, 0};
    float ub_exp[2] = {0, 0};
    float pw_exp[4] = {0, 0, 0, 0};
    float pb_exp[2] = {0, 0};

    // =========================
    // Iteration 1
    // =========================
    // X1 = [[1,2],[3,4]]
    // dY1 = [[1,0],[0,1]]
    set2x2(x, 1, 2, 3, 4);
    set2x2(dy, 1, 0, 0, 1);
    stream.synchronize();

    // Training forward first: forces projected init/copy
    fc->forward(x, /*inference=*/false);

    // Sanity: projected should equal weights/biases after first training forward
    {
        Tensor w_rb = w.clone(cpuPlacement);
        Tensor pw_rb = pw.clone(cpuPlacement);
        Tensor b_rb = b.clone(cpuPlacement);
        Tensor pb_rb = pb.clone(cpuPlacement);
        w_rb.copyFromAsync(w, stream);
        pw_rb.copyFromAsync(pw, stream);
        b_rb.copyFromAsync(b, stream);
        pb_rb.copyFromAsync(pb, stream);
        stream.synchronize();

        half *ww = w_rb.getMemPtr<half>();
        half *pp = pw_rb.getMemPtr<half>();
        for (int i = 0; i < 4; ++i)
            EXPECT_EQ(ww[i], pp[i]);

        half *bb = b_rb.getMemPtr<half>();
        half *bp = pb_rb.getMemPtr<half>();
        for (int i = 0; i < 2; ++i)
            EXPECT_EQ(bb[i], bp[i]);
    }

    // Backward triggers update internally
    fc->backward(dy);
    sgd->getGradientUpdateStream().synchronize();

    // CPU expected gradients:
    const float X1[4] = {1, 2, 3, 4};
    const float dY1[4] = {1, 0, 0, 1};
    float X1T[4];
    transpose2x2(X1, X1T);
    float dW1[4];
    matmul2x2_rowmajor(X1T, dY1, dW1);  // [[1,3],[2,4]]
    float db1[2] = {dY1[0] + dY1[2],    // sum over batch for col0
                    dY1[1] + dY1[3]};   // sum over batch for col1
    // db1 = [1, 1]

    // CPU expected update step 1
    for (int i = 0; i < 4; ++i) {
        uw_exp[i] = mu * uw_exp[i] + (-lr / batchSize) * dW1[i];
        w_exp[i] = w_exp[i] + uw_exp[i] * invLS;
        pw_exp[i] = w_exp[i] + mu * uw_exp[i] * invLS;
    }
    for (int i = 0; i < 2; ++i) {
        ub_exp[i] = mu * ub_exp[i] + (-lr / batchSize) * db1[i];
        b_exp[i] = b_exp[i] + ub_exp[i] * invLS;
        pb_exp[i] = b_exp[i] + mu * ub_exp[i] * invLS;
    }

    // Read back GPU after iter 1
    Tensor w1_h = w.clone(cpuPlacement);
    Tensor pw1_h = pw.clone(cpuPlacement);
    Tensor uw1_h = uw.clone(cpuPlacement);
    Tensor gw1_h = gw.clone(cpuPlacement);

    Tensor b1_h = b.clone(cpuPlacement);
    Tensor pb1_h = pb.clone(cpuPlacement);
    Tensor ub1_h = ub.clone(cpuPlacement);
    Tensor gb1_h = gb.clone(cpuPlacement);

    w1_h.copyFromAsync(w, sgd->getGradientUpdateStream());
    pw1_h.copyFromAsync(pw, sgd->getGradientUpdateStream());
    uw1_h.copyFromAsync(uw, sgd->getGradientUpdateStream());
    gw1_h.copyFromAsync(gw, sgd->getGradientUpdateStream());

    b1_h.copyFromAsync(b, sgd->getGradientUpdateStream());
    pb1_h.copyFromAsync(pb, sgd->getGradientUpdateStream());
    ub1_h.copyFromAsync(ub, sgd->getGradientUpdateStream());
    gb1_h.copyFromAsync(gb, sgd->getGradientUpdateStream());

    sgd->getGradientUpdateStream().synchronize();

    half *w1 = w1_h.getMemPtr<half>();
    half *pw1 = pw1_h.getMemPtr<half>();
    half *uw1 = uw1_h.getMemPtr<half>();
    half *gw1 = gw1_h.getMemPtr<half>();

    half *b1 = b1_h.getMemPtr<half>();
    half *pb1 = pb1_h.getMemPtr<half>();
    half *ub1 = ub1_h.getMemPtr<half>();
    half *gb1 = gb1_h.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        expectNear(dW1[i], (float)gw1[i], eps, "iter1 dW");
        expectNear(uw_exp[i], (float)uw1[i], eps, "iter1 uw");
        expectNear(w_exp[i], (float)w1[i], eps, "iter1 w");
        expectNear(pw_exp[i], (float)pw1[i], eps, "iter1 pw");
    }
    for (int i = 0; i < 2; ++i) {
        expectNear(db1[i], (float)gb1[i], eps, "iter1 db");
        expectNear(ub_exp[i], (float)ub1[i], eps, "iter1 ub");
        expectNear(b_exp[i], (float)b1[i], eps, "iter1 b");
        expectNear(pb_exp[i], (float)pb1[i], eps, "iter1 pb");
    }

    // Verify training forward uses projected weights + projected bias (post-iter1)
    // Use Xcheck = [[2,1],[0,1]]
    set2x2(x, 2, 1, 0, 1);
    stream.synchronize();

    fc->forward(x, /*inference=*/false);
    Tensor ytrain1_h = y.clone(cpuPlacement);
    ytrain1_h.copyFromAsync(y, stream);
    stream.synchronize();
    half *ytrain1 = ytrain1_h.getMemPtr<half>();

    const float Xcheck[4] = {2, 1, 0, 1};
    float Ycheck[4];
    matmul2x2_rowmajor(Xcheck, pw_exp, Ycheck);
    // add bias (broadcast to each row)
    Ycheck[0] += pb_exp[0];
    Ycheck[1] += pb_exp[1];
    Ycheck[2] += pb_exp[0];
    Ycheck[3] += pb_exp[1];

    for (int i = 0; i < 4; ++i) {
        expectNear(Ycheck[i], (float)ytrain1[i], 1e-2f, "iter1 training forward uses projected (w+b)");
    }

    // =========================
    // Iteration 2
    // =========================
    // X2  = [[ 2, 1],
    //        [ 0, 1]]
    // dY2 = [[ 1, 1],
    //        [ 0, 1]]
    set2x2(x, 2, 1, 0, 1);
    set2x2(dy, 1, 1, 0, 1);
    stream.synchronize();

    sgd->updateHyperParameters(/*epoch=*/1, /*batch=*/0, /*batchesPerEpoch=*/1);
    lr *= 1.0f - decay;

    fc->forward(x, /*inference=*/false);
    fc->backward(dy);
    sgd->getGradientUpdateStream().synchronize();

    const float X2[4] = {2, 1, 0, 1};
    const float dY2[4] = {1, 1, 0, 1};
    float X2T[4];
    transpose2x2(X2, X2T);
    float dW2[4];
    matmul2x2_rowmajor(X2T, dY2, dW2);
    float db2[2] = {dY2[0] + dY2[2],   // [1 + 0] = 1
                    dY2[1] + dY2[3]};  // [1 + 1] = 2

    for (int i = 0; i < 4; ++i) {
        uw_exp[i] = mu * uw_exp[i] + (-lr / batchSize) * dW2[i];
        w_exp[i] = w_exp[i] + uw_exp[i] * invLS;
        pw_exp[i] = w_exp[i] + mu * uw_exp[i] * invLS;
    }
    for (int i = 0; i < 2; ++i) {
        ub_exp[i] = mu * ub_exp[i] + (-lr / batchSize) * db2[i];
        b_exp[i] = b_exp[i] + ub_exp[i] * invLS;
        pb_exp[i] = b_exp[i] + mu * ub_exp[i] * invLS;
    }

    Tensor w2_h = w.clone(cpuPlacement);
    Tensor pw2_h = pw.clone(cpuPlacement);
    Tensor uw2_h = uw.clone(cpuPlacement);
    Tensor gw2_h = gw.clone(cpuPlacement);

    Tensor b2_h = b.clone(cpuPlacement);
    Tensor pb2_h = pb.clone(cpuPlacement);
    Tensor ub2_h = ub.clone(cpuPlacement);
    Tensor gb2_h = gb.clone(cpuPlacement);

    w2_h.copyFromAsync(w, sgd->getGradientUpdateStream());
    pw2_h.copyFromAsync(pw, sgd->getGradientUpdateStream());
    uw2_h.copyFromAsync(uw, sgd->getGradientUpdateStream());
    gw2_h.copyFromAsync(gw, sgd->getGradientUpdateStream());

    b2_h.copyFromAsync(b, sgd->getGradientUpdateStream());
    pb2_h.copyFromAsync(pb, sgd->getGradientUpdateStream());
    ub2_h.copyFromAsync(ub, sgd->getGradientUpdateStream());
    gb2_h.copyFromAsync(gb, sgd->getGradientUpdateStream());

    sgd->getGradientUpdateStream().synchronize();

    half *w2 = w2_h.getMemPtr<half>();
    half *pw2 = pw2_h.getMemPtr<half>();
    half *uw2 = uw2_h.getMemPtr<half>();
    half *gw2 = gw2_h.getMemPtr<half>();

    half *b2 = b2_h.getMemPtr<half>();
    half *pb2 = pb2_h.getMemPtr<half>();
    half *ub2 = ub2_h.getMemPtr<half>();
    half *gb2 = gb2_h.getMemPtr<half>();

    for (int i = 0; i < 4; ++i) {
        expectNear(dW2[i], (float)gw2[i], eps, "iter2 dW");
        expectNear(uw_exp[i], (float)uw2[i], eps, "iter2 uw");
        expectNear(w_exp[i], (float)w2[i], eps, "iter2 w");
        expectNear(pw_exp[i], (float)pw2[i], eps, "iter2 pw");
    }
    for (int i = 0; i < 2; ++i) {
        expectNear(db2[i], (float)gb2[i], eps, "iter2 db");
        expectNear(ub_exp[i], (float)ub2[i], eps, "iter2 ub");
        expectNear(b_exp[i], (float)b2[i], eps, "iter2 b");
        expectNear(pb_exp[i], (float)pb2[i], eps, "iter2 pb");
    }

    // Final: inference forward uses real weights + real bias (w2,b2), NOT projected
    fc->forward(x, /*inference=*/true);
    Tensor yinf_h = y.clone(cpuPlacement);
    yinf_h.copyFromAsync(y, stream);
    stream.synchronize();
    half *yinf = yinf_h.getMemPtr<half>();

    float Yinf[4];
    matmul2x2_rowmajor(X2, w_exp, Yinf);
    Yinf[0] += b_exp[0];
    Yinf[1] += b_exp[1];
    Yinf[2] += b_exp[0];
    Yinf[3] += b_exp[1];

    for (int i = 0; i < 4; ++i) {
        expectNear(Yinf[i], (float)yinf[i], 1e-2f, "iter2 inference forward uses real (w+b)");
    }
}
