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
    half *expected, half *actual, uint32_t rows, uint32_t cols, bool print = false, float staticThresh = 0.1, float dynamicThresh = 0.004) {
    double meanAbsoluteError = 0.0;
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
            assert(diff <= scaledThresh);
            ASSERT_LE(diff, scaledThresh);
            meanAbsoluteError += diff;
        }
    }
    meanAbsoluteError /= (double)(rows * cols);
    printf("MeanAbsoluteError: %lf\n", meanAbsoluteError);
    fflush(stdout);

    //    if (print == false && meanAbsoluteError > 0.75) {
    //        verifyMatricesMatch(expected, actual, rows, cols, true, staticThresh, dynamicThresh);
    //        printf("rows %d cols %d\n", rows, cols);
    //        fflush(stdout);
    //    }
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
    layers.push_back(fullyConnectedLayer);
    layers.push_back(make_shared<NoOpLayer>());
    layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

    Stream dataStream = layers.front()->getStream();

    LayerTestHelper::connectAndInitializeNetwork(layers);

    Tensor anErrorInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
    Tensor anErrorOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
    shared_ptr<Sgd> sgd =
        make_shared<Sgd>(fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, anErrorInput, anErrorOutput);
    sgd->initialize();
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
    ASSERT_EQ(flags, CLR | ILR | D | M | UNM);

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

        LayerTestHelper::connectAndInitializeNetwork(layers);

        Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs());
        Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
        Tensor errorOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
        shared_ptr<Sgd> sgd =
            make_shared<Sgd>(fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, errorInput, errorOutput);
        sgd->initialize();
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

        LayerTestHelper::connectAndInitializeNetwork(layers);

        Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs());
        Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
        Tensor errorOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
        shared_ptr<Sgd> sgd =
            make_shared<Sgd>(fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, errorInput, errorOutput);
        sgd->initialize();
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

// For this test I will actually need to call forward and backward to verify that projected weights are being used.
// 1. Fill feature in random
// 2. fill errorIn random
// 3. fill weights in random
// 4. call forward
// 5. check inference is correct: this time there is no previous weight update so momentum so update with just gradient
// 6. call backward
// 7. Check weights values
// 8. call forward
// 9. check inference is correct: this time there is a previous weight update so using projected weights
// 10. fill errorIn random
// 11. call backward
// 12. Check weights values
// 13. Call forward for a third time, this time in inference mode, toensure that projected weights are not used to create featureOutput
// 14. Verify featureOutput
TEST(SgdTest, TestWeightsUpdateWithNesterovMomentum) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 50; ++t) {
        uint32_t batchSize = (rand() % 300) + 1;
        uint32_t numInputFeatures = (rand() % 300) + 1;
        uint32_t numOutputFeatures = (rand() % 300) + 1;
        bool hasBias = rand() % 2;

        Tensor networkFeatureIn = Tensor(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numInputFeatures}));

        float initialLearningRate = 10000.0 / (1 + rand() % 33);
        float decay = ((rand() % 5) / 10.0f);
        float momentum = (1.0f + (rand() % 100)) / 100.0f;
        bool useNesterovMomentum = true;
        uint64_t epoch = rand() % 10;

        vector<shared_ptr<Layer>> layers;

        layers.push_back(
            make_shared<NetworkInput>(gpuPlacement, TensorDescriptor::DataType::FP16, networkFeatureIn.getDescriptor().getDimensions()));
        layers.push_back(make_shared<NoOpLayer>());
        shared_ptr<FullyConnected> fullyConnectedLayer = make_shared<FullyConnected>(numOutputFeatures, hasBias);
        layers.push_back(fullyConnectedLayer);
        layers.push_back(make_shared<NoOpLayer>());
        layers.push_back(make_shared<NetworkOutput>(cpuPlacement));

        Stream dataStream = layers.front()->getStream();

        LayerTestHelper::connectAndInitializeNetwork(layers);

        Tensor featureInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs());
        Tensor featureOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureOutputs());
        Tensor errorInput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorInputs());
        Tensor errorOutput = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getErrorOutputs());
        Tensor weights = fullyConnectedLayer->getWeights();
        featureInput.fillRandom(-2.0, 2.0, dataStream);
        errorInput.fillRandom(-2.0, 2.0, dataStream);
        weights.fillRandom(-2.0, 2.0, dataStream);
        dataStream.synchronize();
        shared_ptr<Sgd> sgd =
            make_shared<Sgd>(fullyConnectedLayer, initialLearningRate, decay, momentum, useNesterovMomentum, errorInput, errorOutput);
        sgd->initialize();
        fullyConnectedLayer->setOptimizer(dynamic_pointer_cast<Optimizer>(sgd));
        Tensor weightsGradient = sgd->getWeightsGradient();

        ASSERT_EQ(sgd->getInitialLearningRate(), initialLearningRate);
        ASSERT_EQ(sgd->getDecay(), decay);
        ASSERT_EQ(sgd->getMomentum(), momentum);
        ASSERT_EQ(sgd->getUseNesterovMomentum(), useNesterovMomentum);

        Tensor featureInput_h = featureInput.clone(cpuPlacement);
        Tensor featureOutput_h = featureOutput.clone(cpuPlacement);
        Tensor featureOutputGpu_h = featureOutput_h.clone();
        Tensor errorInput_h = errorInput.clone(cpuPlacement);
        Tensor errorOutput_h = errorOutput.clone(cpuPlacement);
        Tensor errorOutputGpu_h = errorOutput_h.clone();
        Tensor weights_h = weights.clone(cpuPlacement);
        Tensor projectedWeights_h = weights_h.clone();
        Tensor projectedWeightsGpu_h = weights_h.clone();
        Tensor weightsGpu_h = weights_h.clone();
        Tensor weightsGradient_h = weightsGradient.clone(cpuPlacement);
        Tensor weightsGradientGpu_h = weightsGradient_h.clone();
        half *featureInMem_h = featureInput_h.getMemPtr<half>();
        half *featureOutMem_h = featureOutput_h.getMemPtr<half>();
        half *featureOutGpuMem_h = featureOutputGpu_h.getMemPtr<half>();
        half *errorInMem_h = errorInput_h.getMemPtr<half>();
        half *errorOutMem_h = errorOutput_h.getMemPtr<half>();
        half *errorOutGpuMem_h = errorOutputGpu_h.getMemPtr<half>();
        half *weightsMem_h = weights_h.getMemPtr<half>();
        half *weightsGpuMem_h = weightsGpu_h.getMemPtr<half>();
        half *projectedWeightsMem_h = projectedWeights_h.getMemPtr<half>();
        half *projectedWeightsGpuMem_h = projectedWeightsGpu_h.getMemPtr<half>();
        half *weightsGradientMem_h = weightsGradient_h.getMemPtr<half>();
        half *weightsGradientGpuMem_h = weightsGradientGpu_h.getMemPtr<half>();
        featureInput_h.copyFromAsync(featureInput, dataStream);
        errorInput_h.copyFromAsync(errorInput, dataStream);
        weights_h.copyFromAsync(weights, dataStream);

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
        Tensor projectedBiases_h;
        Tensor projectedBiasesGpu_h;
        half *projectedBiasesMem_h;
        half *projectedBiasesGpuMem_h;
        if (hasBias) {
            ASSERT_TRUE(biases.isPresent());
            ASSERT_TRUE(biasesGradient.isPresent());
            biases.get().fillRandom(-2.0, 2.0, dataStream);
            biases_h = biases.get().clone(cpuPlacement);
            biases_h.copyFromAsync(biases, dataStream);
            biasesGpu_h = biases_h.clone();
            biasesMem_h = biases_h.getMemPtr<half>();
            biasesGpuMem_h = biasesGpu_h.getMemPtr<half>();
            biasesGradient_h = biasesGradient.get().clone(cpuPlacement);
            biasesGradientGpu_h = biasesGradient_h.clone();
            biasesGradientMem_h = biasesGradient_h.getMemPtr<half>();
            biasesGradientGpuMem_h = biasesGradientGpu_h.getMemPtr<half>();
            projectedBiases_h = biases_h.clone();
            projectedBiasesGpu_h = biases_h.clone();
            projectedBiasesMem_h = projectedBiases_h.getMemPtr<half>();
            projectedBiasesGpuMem_h = projectedBiasesGpu_h.getMemPtr<half>();
        }

        Tensor weightsUpdate_h = weights_h.clone();
        half *weightsUpdateMem_h = weightsUpdate_h.getMemPtr<half>();
        weightsUpdate_h.clearAsync(dataStream);
        Tensor biasesUpdate_h;
        half *biasesUpdateMem_h;
        if (hasBias) {
            biasesUpdate_h = biases_h.clone();
            biasesUpdateMem_h = biasesUpdate_h.getMemPtr<half>();
            biasesUpdate_h.clearAsync(dataStream);
        }

        dataStream.synchronize();
        Stream gradientUpdateStream = sgd->getGradientUpdateStream();

        uint32_t batchesPerEpoch = 1 + rand() % 10000;
        assert(momentum > 0.0f);
        assert(useNesterovMomentum == true);
        float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, epoch);
        sgd->updateHyperParameters(epoch, rand() % batchesPerEpoch, batchesPerEpoch);

        // Call forward
        fullyConnectedLayer->forward(featureInput, false);
        featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
        dataStream.synchronize();

        // Verify featureOutput
        matrixMultiplyCpuHalf(featureInMem_h,
                              weightsMem_h,
                              featureOutMem_h,
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
        printf("Num input features %d\n", numInputFeatures);
        verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.03, 0.03f);

        // Call backward
        fullyConnectedLayer->backward(errorInput);

        // Verify errorOutput
        errorOutputGpu_h.copyFromAsync(errorOutput, dataStream);
        dataStream.synchronize();
        matrixMultiplyCpuHalf(errorInMem_h,
                              weightsMem_h,
                              errorOutMem_h,
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
        printf("Num output features %d\n", numOutputFeatures);
        verifyMatricesMatch(errorOutMem_h, errorOutGpuMem_h, batchSize, numInputFeatures, false, numOutputFeatures * 0.03, 0.03f);

        // Verify weights
        gradientUpdateStream.synchronize();
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

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.02f);
        }

        weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
        projectedWeightsGpu_h.copyFromAsync(sgd->getProjectedWeights(), gradientUpdateStream);
        gradientUpdateStream.synchronize();

        // printf("%f * %f - (%f * %f) / %d\n",
        //        (float)weightsUpdateMem_h[0],
        //        momentum,
        //        currentLearningRate,
        //        (float)weightsGradientGpuMem_h[0],
        //        batchSize);
        for (uint32_t row = 0; row < numInputFeatures; ++row) {
            for (uint32_t col = 0; col < numOutputFeatures; ++col) {
                uint32_t index = row * numOutputFeatures + col;
                weightsUpdateMem_h[index] =
                    (float)weightsUpdateMem_h[index] * momentum -
                    (((float)weightsGradientGpuMem_h[index] * currentLearningRate) / batchSize) / Loss::getLossScalingFactor();
                weightsMem_h[index] += (float)weightsUpdateMem_h[index];
                projectedWeightsMem_h[index] = weightsMem_h[index] + (half)momentum * weightsUpdateMem_h[index];
            }
        }
        verifyMatricesMatch(weightsMem_h, weightsGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        verifyMatricesMatch(projectedWeightsMem_h, projectedWeightsGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        // Don't let error increase as test gets longer:
        weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
        projectedWeights_h.copyFromAsync(projectedWeightsGpu_h, gradientUpdateStream);

        if (hasBias) {
            biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
            projectedBiasesGpu_h.copyFromAsync(sgd->getProjectedBiases(), gradientUpdateStream);
            gradientUpdateStream.synchronize();

            for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                biasesUpdateMem_h[outputFeature] =
                    (float)biasesUpdateMem_h[outputFeature] * momentum -
                    (((float)biasesGradientGpuMem_h[outputFeature] * currentLearningRate) / batchSize) / Loss::getLossScalingFactor();
                biasesMem_h[outputFeature] += (float)biasesUpdateMem_h[outputFeature];
                projectedBiasesMem_h[outputFeature] = biasesMem_h[outputFeature] + (half)momentum * biasesUpdateMem_h[outputFeature];
            }
            verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            verifyMatricesMatch(projectedBiasesMem_h, projectedBiasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            // Don't let error increase as test gets longer:
            biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
            projectedBiases_h.copyFromAsync(projectedBiasesGpu_h, gradientUpdateStream);
        }

        // Re-randomize featureInput and errorInput
        featureInput.fillRandom(-2.0, 2.0, dataStream);
        errorInput.fillRandom(-2.0, 2.0, dataStream);
        featureInput_h.copyFromAsync(featureInput, dataStream);
        errorInput_h.copyFromAsync(errorInput, dataStream);
        dataStream.synchronize();

        // Call forward for second time, now there is a previous weights update
        fullyConnectedLayer->forward(featureInput, false);

        // Verify featureOutput
        featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
        dataStream.synchronize();

        // FIXME: TEMP
        Tensor inputFromGpu = MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs())
                                  .get()
                                  .clone(TensorPlacement::MemDevices::CPU);
        Tensor weightsFromGpu = fullyConnectedLayer->getProjectedWeightsTensor().get().clone(TensorPlacement::MemDevices::CPU);
        inputFromGpu.copyFromAsync(MultiConnectionLayer::getFirstPresentTensor(fullyConnectedLayer->getFeatureInputs()).get(), dataStream);
        weightsFromGpu.copyFromAsync(fullyConnectedLayer->getProjectedWeightsTensor(), dataStream);
        dataStream.synchronize();
        for (int i = 0; i < (int)inputFromGpu.getTotalNumElements(); i++) {
            if (!(featureInMem_h[i] == inputFromGpu.getMemPtr<half>()[i])) {
                printf("CPU featureIn[%d] %f, Gpu featureIn[%d] %f\n",
                       i,
                       (float)featureInMem_h[i],
                       i,
                       (float)inputFromGpu.getMemPtr<half>()[i]);
                fflush(stdout);
            }
            assert(featureInMem_h[i] == inputFromGpu.getMemPtr<half>()[i]);
        }
        for (int i = 0; i < (int)weightsFromGpu.getTotalNumElements(); i++) {
            if (!(projectedWeightsMem_h[i] == weightsFromGpu.getMemPtr<half>()[i])) {
                printf("CPU weights[%d] %f, Gpu weights[%d] %f\n",
                       i,
                       (float)projectedWeightsMem_h[i],
                       i,
                       (float)weightsFromGpu.getMemPtr<half>()[i]);
                fflush(stdout);
            }
            assert(projectedWeightsMem_h[i] == weightsFromGpu.getMemPtr<half>()[i]);
        }

        matrixMultiplyCpuHalf(featureInMem_h,
                              projectedWeightsMem_h,
                              featureOutMem_h,
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
        verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.05, 0.03f);

        // Call backward
        fullyConnectedLayer->backward(errorInput);

        // Verify errorOutput
        errorOutputGpu_h.copyFromAsync(errorOutput, dataStream);
        dataStream.synchronize();
        matrixMultiplyCpuHalf(errorInMem_h,
                              weightsMem_h,
                              errorOutMem_h,
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
        verifyMatricesMatch(errorOutMem_h, errorOutGpuMem_h, batchSize, numInputFeatures, false, numOutputFeatures * 0.03, 0.03f);

        // Verify weights
        gradientUpdateStream.synchronize();
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

        if (hasBias) {
            reduceBatch(errorInMem_h, biasesGradientMem_h, batchSize, numOutputFeatures, false);
            biasesGradientGpu_h.copyFromAsync(biasesGradient, gradientUpdateStream);
            gradientUpdateStream.synchronize();
            verifyMatricesMatch(biasesGradientMem_h, biasesGradientGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
        }

        weightsGpu_h.copyFromAsync(weights, gradientUpdateStream);
        projectedWeightsGpu_h.copyFromAsync(sgd->getProjectedWeights(), gradientUpdateStream);
        gradientUpdateStream.synchronize();

        // printf("%f * %f - (%f * %f) / %d\n",
        //        (float)weightsUpdateMem_h[0],
        //        momentum,
        //        currentLearningRate,
        //        (float)weightsGradientGpuMem_h[0],
        //        batchSize);
        for (uint32_t row = 0; row < numInputFeatures; ++row) {
            for (uint32_t col = 0; col < numOutputFeatures; ++col) {
                uint32_t index = row * numOutputFeatures + col;
                weightsUpdateMem_h[index] =
                    (float)weightsUpdateMem_h[index] * momentum -
                    (((float)weightsGradientGpuMem_h[index] * currentLearningRate) / batchSize) / Loss::getLossScalingFactor();
                weightsMem_h[index] += (float)weightsUpdateMem_h[index];
                projectedWeightsMem_h[index] = weightsMem_h[index] + (half)momentum * weightsUpdateMem_h[index];
            }
        }
        verifyMatricesMatch(weightsMem_h, weightsGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        verifyMatricesMatch(projectedWeightsMem_h, projectedWeightsGpuMem_h, numInputFeatures, numOutputFeatures, false, 0.4f, 0.03f);
        // Don't let error increase as test gets longer:
        weights_h.copyFromAsync(weightsGpu_h, gradientUpdateStream);
        projectedWeights_h.copyFromAsync(projectedWeightsGpu_h, gradientUpdateStream);

        if (hasBias) {
            biasesGpu_h.copyFromAsync(biases, gradientUpdateStream);
            projectedBiasesGpu_h.copyFromAsync(sgd->getProjectedBiases(), gradientUpdateStream);
            gradientUpdateStream.synchronize();

            for (uint32_t outputFeature = 0; outputFeature < numOutputFeatures; ++outputFeature) {
                biasesUpdateMem_h[outputFeature] =
                    (float)biasesUpdateMem_h[outputFeature] * momentum -
                    (((float)biasesGradientGpuMem_h[outputFeature] * currentLearningRate) / batchSize) / Loss::getLossScalingFactor();
                biasesMem_h[outputFeature] += (float)biasesUpdateMem_h[outputFeature];
                projectedBiasesMem_h[outputFeature] = biasesMem_h[outputFeature] + (half)momentum * biasesUpdateMem_h[outputFeature];
            }
            verifyMatricesMatch(biasesMem_h, biasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            verifyMatricesMatch(projectedBiasesMem_h, projectedBiasesGpuMem_h, 1, numOutputFeatures, false, 0.4f, 0.03f);
            // Don't let error increase as test gets longer:
            biases_h.copyFromAsync(biasesGpu_h, gradientUpdateStream);
            projectedBiases_h.copyFromAsync(projectedBiasesGpu_h, gradientUpdateStream);
        }

        // Call forward for a third time, this time in inference mode, ensure that projected weights are not used to create featureOutput
        fullyConnectedLayer->forward(featureInput, true);

        // Verify featureOutput
        featureOutputGpu_h.copyFromAsync(featureOutput, dataStream);
        dataStream.synchronize();
        matrixMultiplyCpuHalf(featureInMem_h,
                              weightsMem_h,
                              featureOutMem_h,
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
        verifyMatricesMatch(featureOutMem_h, featureOutGpuMem_h, batchSize, numOutputFeatures, false, numInputFeatures * 0.03, 0.03f);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
