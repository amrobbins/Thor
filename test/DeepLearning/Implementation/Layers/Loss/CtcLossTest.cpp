#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace ThorImplementation;

namespace {

uint64_t ctcIndex(uint64_t t, uint64_t b, uint64_t c, uint64_t maxTimeSteps, uint64_t numClasses) {
    return (b * maxTimeSteps + t) * numClasses + c;
}

vector<float> softmaxNormalizedActivations(const vector<float>& activations,
                                           uint64_t maxTimeSteps,
                                           uint64_t batchSize,
                                           uint64_t numClasses) {
    vector<float> probabilities(activations.size());
    for (uint64_t t = 0; t < maxTimeSteps; ++t) {
        for (uint64_t b = 0; b < batchSize; ++b) {
            float maxActivation = activations[ctcIndex(t, b, 0, maxTimeSteps, numClasses)];
            for (uint64_t c = 1; c < numClasses; ++c) {
                maxActivation = std::max(maxActivation, activations[ctcIndex(t, b, c, maxTimeSteps, numClasses)]);
            }

            double sum = 0.0;
            for (uint64_t c = 0; c < numClasses; ++c) {
                const double value = exp(static_cast<double>(activations[ctcIndex(t, b, c, maxTimeSteps, numClasses)] -
                                                               maxActivation));
                probabilities[ctcIndex(t, b, c, maxTimeSteps, numClasses)] = static_cast<float>(value);
                sum += value;
            }
            for (uint64_t c = 0; c < numClasses; ++c) {
                probabilities[ctcIndex(t, b, c, maxTimeSteps, numClasses)] =
                    static_cast<float>(static_cast<double>(probabilities[ctcIndex(t, b, c, maxTimeSteps, numClasses)]) / sum);
            }
        }
    }
    return probabilities;
}

double cpuCtcLossForSample(const vector<float>& probabilities,
                           uint64_t maxTimeSteps,
                           uint64_t batchSize,
                           uint64_t numClasses,
                           uint64_t batch,
                           const vector<int>& concatenatedLabels,
                           const vector<int>& labelLengths,
                           const vector<int>& inputLengths,
                           int blankLabel) {
    const uint64_t T = static_cast<uint64_t>(inputLengths[batch]);
    const uint64_t L = static_cast<uint64_t>(labelLengths[batch]);
    EXPECT_GT(T, 0u);
    EXPECT_LE(T, maxTimeSteps);
    EXPECT_GT(L, 0u);

    uint64_t labelOffset = 0;
    for (uint64_t i = 0; i < batch; ++i)
        labelOffset += static_cast<uint64_t>(labelLengths[i]);

    vector<int> extendedLabels(2 * L + 1, blankLabel);
    for (uint64_t i = 0; i < L; ++i)
        extendedLabels[2 * i + 1] = concatenatedLabels[labelOffset + i];

    const uint64_t S = extendedLabels.size();
    vector<double> prev(S, 0.0);
    vector<double> curr(S, 0.0);

    prev[0] = probabilities[ctcIndex(0, batch, static_cast<uint64_t>(extendedLabels[0]), maxTimeSteps, numClasses)];
    if (S > 1)
        prev[1] = probabilities[ctcIndex(0, batch, static_cast<uint64_t>(extendedLabels[1]), maxTimeSteps, numClasses)];

    for (uint64_t t = 1; t < T; ++t) {
        fill(curr.begin(), curr.end(), 0.0);
        for (uint64_t s = 0; s < S; ++s) {
            double total = prev[s];
            if (s > 0)
                total += prev[s - 1];
            if (s > 1 && extendedLabels[s] != blankLabel && extendedLabels[s] != extendedLabels[s - 2])
                total += prev[s - 2];
            curr[s] = total * probabilities[ctcIndex(t, batch, static_cast<uint64_t>(extendedLabels[s]), maxTimeSteps, numClasses)];
        }
        prev.swap(curr);
    }

    double sequenceProbability = prev[S - 1];
    if (S > 1)
        sequenceProbability += prev[S - 2];
    EXPECT_GT(sequenceProbability, 0.0);
    return -log(sequenceProbability);
}

vector<double> cpuCtcLoss(const vector<float>& probabilities,
                          uint64_t maxTimeSteps,
                          uint64_t batchSize,
                          uint64_t numClasses,
                          const vector<int>& concatenatedLabels,
                          const vector<int>& labelLengths,
                          const vector<int>& inputLengths,
                          int blankLabel) {
    vector<double> losses(batchSize);
    for (uint64_t b = 0; b < batchSize; ++b) {
        losses[b] = cpuCtcLossForSample(probabilities,
                                        maxTimeSteps,
                                        batchSize,
                                        numClasses,
                                        b,
                                        concatenatedLabels,
                                        labelLengths,
                                        inputLengths,
                                        blankLabel);
    }
    return losses;
}


double cpuCtcLossForActivationsSample(const vector<float>& activations,
                                      uint64_t maxTimeSteps,
                                      uint64_t batchSize,
                                      uint64_t numClasses,
                                      uint64_t batch,
                                      const vector<int>& concatenatedLabels,
                                      const vector<int>& labelLengths,
                                      const vector<int>& inputLengths,
                                      int blankLabel) {
    const vector<float> probabilities = softmaxNormalizedActivations(activations, maxTimeSteps, batchSize, numClasses);
    return cpuCtcLossForSample(probabilities,
                               maxTimeSteps,
                               batchSize,
                               numClasses,
                               batch,
                               concatenatedLabels,
                               labelLengths,
                               inputLengths,
                               blankLabel);
}

vector<double> cpuCtcLossForActivations(const vector<float>& activations,
                                        uint64_t maxTimeSteps,
                                        uint64_t batchSize,
                                        uint64_t numClasses,
                                        const vector<int>& concatenatedLabels,
                                        const vector<int>& labelLengths,
                                        const vector<int>& inputLengths,
                                        int blankLabel) {
    const vector<float> probabilities = softmaxNormalizedActivations(activations, maxTimeSteps, batchSize, numClasses);
    return cpuCtcLoss(probabilities, maxTimeSteps, batchSize, numClasses, concatenatedLabels, labelLengths, inputLengths, blankLabel);
}

vector<double> finiteDifferenceCtcLogitGradient(const vector<float>& activations,
                                                uint64_t maxTimeSteps,
                                                uint64_t batchSize,
                                                uint64_t numClasses,
                                                const vector<int>& concatenatedLabels,
                                                const vector<int>& labelLengths,
                                                const vector<int>& inputLengths,
                                                int blankLabel,
                                                double epsilon) {
    vector<double> gradient(activations.size(), 0.0);
    vector<float> perturbed = activations;
    for (uint64_t b = 0; b < batchSize; ++b) {
        for (uint64_t t = 0; t < maxTimeSteps; ++t) {
            for (uint64_t c = 0; c < numClasses; ++c) {
                const uint64_t index = ctcIndex(t, b, c, maxTimeSteps, numClasses);
                perturbed[index] = static_cast<float>(static_cast<double>(activations[index]) + epsilon);
                const double plus = cpuCtcLossForActivationsSample(perturbed,
                                                                   maxTimeSteps,
                                                                   batchSize,
                                                                   numClasses,
                                                                   b,
                                                                   concatenatedLabels,
                                                                   labelLengths,
                                                                   inputLengths,
                                                                   blankLabel);
                perturbed[index] = static_cast<float>(static_cast<double>(activations[index]) - epsilon);
                const double minus = cpuCtcLossForActivationsSample(perturbed,
                                                                    maxTimeSteps,
                                                                    batchSize,
                                                                    numClasses,
                                                                    b,
                                                                    concatenatedLabels,
                                                                    labelLengths,
                                                                    inputLengths,
                                                                    blankLabel);
                perturbed[index] = activations[index];
                gradient[index] = (plus - minus) / (2.0 * epsilon);
            }
        }
    }
    return gradient;
}

struct CtcNumericalCase {
    vector<float> activations;
    vector<int> paddedLabels;
    vector<int> packedLabels;
    vector<int> labelLengths;
    vector<int> inputLengths;
};

CtcNumericalCase makeRepeatedAndVariableLengthCase() {
    return CtcNumericalCase{
        {
            // b=0, repeated target [1, 1], input length 4.
            0.25f, -0.10f, 0.40f,
            0.85f, 0.20f, -0.35f,
            -0.45f, 0.90f, 0.10f,
            0.30f, 0.55f, -0.25f,
            // b=1, target [2], input length 3; t=3 is ignored by inputLengths.
            -0.15f, 0.50f, 0.35f,
            0.45f, -0.20f, 0.70f,
            0.10f, 0.15f, 0.60f,
            0.95f, -0.75f, 0.05f,
        },
        // The trailing 99 is intentionally outside the class range. It must be ignored because
        // labelLengths[1] == 1 and the implementation compacts only the valid prefix.
        {1, 1, 2, 99},
        {1, 1, 2},
        {2, 1},
        {4, 3},
    };
}


struct CtcLayerNetwork {
    TensorPlacement cpuPlacement{TensorPlacement::MemDevices::CPU};
    TensorPlacement gpuPlacement{TensorPlacement::MemDevices::GPU, 0};

    Tensor probabilitiesCpu;
    Tensor labelsCpu;
    Tensor labelLengthsCpu;
    Tensor inputLengthsCpu;

    shared_ptr<NetworkInput> probabilitiesInput;
    shared_ptr<NetworkInput> labelsInput;
    shared_ptr<NetworkInput> labelLengthsInput;
    shared_ptr<NetworkInput> inputLengthsInput;
    shared_ptr<CtcLoss> ctcLoss;
    shared_ptr<NetworkOutput> lossOutput;
    vector<shared_ptr<Layer>> layers;
};

CtcLayerNetwork makeTinyCtcNetwork(bool inferenceOnly,
                                    DataType probabilitiesType = DataType::FP32,
                                    DataType lengthType = DataType::INT32,
                                    optional<float> lossWeight = nullopt) {
    CtcLayerNetwork network;
    constexpr uint64_t T = 4;
    constexpr uint64_t B = 2;
    constexpr uint64_t C = 3;

    network.probabilitiesCpu = Tensor(network.cpuPlacement, TensorDescriptor(probabilitiesType, {B, T, C}));
    network.labelsCpu = Tensor(network.cpuPlacement, TensorDescriptor(DataType::INT32, {B, 2}));
    network.labelLengthsCpu = Tensor(network.cpuPlacement, TensorDescriptor(lengthType, {B, 1}));
    network.inputLengthsCpu = Tensor(network.cpuPlacement, TensorDescriptor(DataType::INT32, {B, 1}));

    network.probabilitiesInput = make_shared<NetworkInput>(network.gpuPlacement, probabilitiesType, vector<unsigned long>{B, T, C});
    network.labelsInput = make_shared<NetworkInput>(network.gpuPlacement, DataType::INT32, vector<unsigned long>{B, 2});
    network.labelLengthsInput = make_shared<NetworkInput>(network.gpuPlacement, lengthType, vector<unsigned long>{B, 1});
    network.inputLengthsInput = make_shared<NetworkInput>(network.gpuPlacement, DataType::INT32, vector<unsigned long>{B, 1});
    network.ctcLoss = make_shared<CtcLoss>(2, CtcLossOobGradientMode::ZERO, lossWeight);
    network.ctcLoss->setConstructForInferenceOnly(inferenceOnly);
    network.lossOutput = make_shared<NetworkOutput>(network.cpuPlacement);

    network.layers = {network.probabilitiesInput,
                      network.labelsInput,
                      network.labelLengthsInput,
                      network.inputLengthsInput,
                      network.ctcLoss,
                      network.lossOutput};

    LayerTestHelper::connectTwoLayers(network.probabilitiesInput, network.ctcLoss, 0, static_cast<int>(Loss::ConnectionType::FORWARD_BACKWARD));
    LayerTestHelper::connectTwoLayers(network.labelsInput, network.ctcLoss, 0, static_cast<int>(Loss::ConnectionType::LABELS));
    LayerTestHelper::connectTwoLayers(network.labelLengthsInput, network.ctcLoss, 0, CtcLoss::LABEL_LENGTHS_CONNECTION_TYPE);
    LayerTestHelper::connectTwoLayers(network.inputLengthsInput, network.ctcLoss, 0, CtcLoss::INPUT_LENGTHS_CONNECTION_TYPE);
    LayerTestHelper::connectTwoLayers(network.ctcLoss, network.lossOutput);

    return network;
}

void populateTinyCtcNetwork(CtcLayerNetwork& network,
                            const vector<float>& activations,
                            const vector<int>& paddedLabels,
                            const vector<int>& labelLengths,
                            const vector<int>& inputLengths) {
    std::copy(activations.begin(), activations.end(), network.probabilitiesCpu.getMemPtr<float>());
    std::copy(paddedLabels.begin(), paddedLabels.end(), network.labelsCpu.getMemPtr<int>());
    std::copy(labelLengths.begin(), labelLengths.end(), network.labelLengthsCpu.getMemPtr<int>());
    std::copy(inputLengths.begin(), inputLengths.end(), network.inputLengthsCpu.getMemPtr<int>());
}

void runTinyCtcNetwork(CtcLayerNetwork& network, vector<float>* gradient = nullptr) {
    LayerTestHelper::initializeNetwork(network.layers);

    network.probabilitiesInput->forward(network.probabilitiesCpu, false);
    network.labelsInput->forward(network.labelsCpu, false);
    network.labelLengthsInput->forward(network.labelLengthsCpu, false);
    network.inputLengthsInput->forward(network.inputLengthsCpu, false);

    Stream syncStream = network.probabilitiesInput->getStream();
    syncStream.waitEvent(network.lossOutput->getOutputReadyEvent());
    syncStream.synchronize();

    if (gradient != nullptr) {
        ASSERT_TRUE(network.ctcLoss->getErrorOutput().has_value());
        Tensor gradientCpu = network.ctcLoss->getErrorOutput().value().clone(network.cpuPlacement);
        gradientCpu.copyFromAsync(network.ctcLoss->getErrorOutput().value(), syncStream);
        syncStream.synchronize();
        const float* gradientMem = gradientCpu.getMemPtr<float>();
        gradient->assign(gradientMem, gradientMem + gradientCpu.getTotalNumElements());
    }
}
}  // namespace

TEST(CtcLossImplementationLayer, CreatesPerSampleLossAndPredictionGradientDescriptors) {
    CtcLayerNetwork network = makeTinyCtcNetwork(false);
    LayerTestHelper::initializeNetwork(network.layers);

    ASSERT_TRUE(network.ctcLoss->getLossOutput().has_value());
    EXPECT_EQ(network.ctcLoss->getLossOutput().value().getDescriptor(), TensorDescriptor(DataType::FP32, {2, 1}));
    ASSERT_TRUE(network.ctcLoss->getErrorOutput().has_value());
    EXPECT_EQ(network.ctcLoss->getErrorOutput().value().getDescriptor(), TensorDescriptor(DataType::FP32, {2, 4, 3}));

    LayerTestHelper::tearDownNetwork(network.layers);
}

TEST(CtcLossImplementationLayer, InferenceOnlyUsesPerSampleLossWithoutTrainingErrorOutput) {
    CtcLayerNetwork network = makeTinyCtcNetwork(true);
    LayerTestHelper::initializeNetwork(network.layers);

    ASSERT_TRUE(network.ctcLoss->getLossOutput().has_value());
    EXPECT_EQ(network.ctcLoss->getLossOutput().value().getDescriptor(), TensorDescriptor(DataType::FP32, {2, 1}));
    EXPECT_FALSE(network.ctcLoss->getErrorOutput().has_value());

    LayerTestHelper::tearDownNetwork(network.layers);
}

TEST(CtcLossImplementationLayer, RejectsNonFp32Probabilities) {
    CtcLayerNetwork network = makeTinyCtcNetwork(true, DataType::FP16);
    EXPECT_THROW(LayerTestHelper::initializeNetwork(network.layers), std::logic_error);
    LayerTestHelper::tearDownNetwork(network.layers);
}

TEST(CtcLossImplementationLayer, RejectsNonInt32Lengths) {
    CtcLayerNetwork network = makeTinyCtcNetwork(true, DataType::FP32, DataType::UINT32);
    EXPECT_THROW(LayerTestHelper::initializeNetwork(network.layers), std::logic_error);
    LayerTestHelper::tearDownNetwork(network.layers);
}

TEST(CtcLossImplementationLayer, InferenceForwardMatchesCpuReferenceForSmallBatchWithSoftmaxNormalization) {
    constexpr uint64_t T = 4;
    constexpr uint64_t B = 2;
    constexpr uint64_t C = 3;
    constexpr int blankLabel = 0;

    CtcLayerNetwork network = makeTinyCtcNetwork(true);

    const vector<float> activations = {
        // b=0, t=0..3
        0.60f, 0.30f, 0.10f,
        0.20f, 0.70f, 0.10f,
        0.25f, 0.65f, 0.10f,
        0.70f, 0.20f, 0.10f,
        // b=1, t=0..3
        0.50f, 0.20f, 0.30f,
        0.20f, 0.20f, 0.60f,
        0.30f, 0.40f, 0.30f,
        0.60f, 0.30f, 0.10f,
    };
    const vector<int> paddedLabels = {1, 0, 2, 1};
    const vector<int> packedLabels = {1, 2, 1};
    const vector<int> labelLengths = {1, 2};
    const vector<int> inputLengths = {4, 4};

    std::copy(activations.begin(), activations.end(), network.probabilitiesCpu.getMemPtr<float>());
    std::copy(paddedLabels.begin(), paddedLabels.end(), network.labelsCpu.getMemPtr<int>());
    std::copy(labelLengths.begin(), labelLengths.end(), network.labelLengthsCpu.getMemPtr<int>());
    std::copy(inputLengths.begin(), inputLengths.end(), network.inputLengthsCpu.getMemPtr<int>());

    const vector<float> normalizedProbabilities = softmaxNormalizedActivations(activations, T, B, C);
    const vector<double> expected = cpuCtcLoss(normalizedProbabilities, T, B, C, packedLabels, labelLengths, inputLengths, blankLabel);

    LayerTestHelper::initializeNetwork(network.layers);

    network.probabilitiesInput->forward(network.probabilitiesCpu, false);
    network.labelsInput->forward(network.labelsCpu, false);
    network.labelLengthsInput->forward(network.labelLengthsCpu, false);
    network.inputLengthsInput->forward(network.inputLengthsCpu, false);

    Stream syncStream = network.probabilitiesInput->getStream();
    syncStream.waitEvent(network.lossOutput->getOutputReadyEvent());
    syncStream.synchronize();

    Tensor actualLossCpu = network.lossOutput->getFeatureOutput().value();
    const float* actual = actualLossCpu.getMemPtr<float>();
    for (uint64_t i = 0; i < B; ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]));
        EXPECT_NEAR(actual[i], expected[i], 1.0e-4);
    }

    LayerTestHelper::tearDownNetwork(network.layers);
}


TEST(CtcLossImplementationLayer, ForwardMatchesCpuReferenceForRepeatedLabelsVariableLengthsAndIgnoredPadding) {
    constexpr uint64_t T = 4;
    constexpr uint64_t B = 2;
    constexpr uint64_t C = 3;
    constexpr int blankLabel = 0;

    CtcLayerNetwork network = makeTinyCtcNetwork(true);
    const CtcNumericalCase testCase = makeRepeatedAndVariableLengthCase();
    populateTinyCtcNetwork(network, testCase.activations, testCase.paddedLabels, testCase.labelLengths, testCase.inputLengths);

    const vector<double> expected = cpuCtcLossForActivations(testCase.activations,
                                                            T,
                                                            B,
                                                            C,
                                                            testCase.packedLabels,
                                                            testCase.labelLengths,
                                                            testCase.inputLengths,
                                                            blankLabel);

    runTinyCtcNetwork(network);

    Tensor actualLossCpu = network.lossOutput->getFeatureOutput().value();
    const float* actual = actualLossCpu.getMemPtr<float>();
    for (uint64_t i = 0; i < B; ++i) {
        ASSERT_TRUE(std::isfinite(actual[i]));
        EXPECT_NEAR(actual[i], expected[i], 1.0e-4f);
    }

    LayerTestHelper::tearDownNetwork(network.layers);
}

TEST(CtcLossImplementationLayer, BackwardMatchesFiniteDifferenceCpuReferenceForSmallBatch) {
    constexpr uint64_t T = 4;
    constexpr uint64_t B = 2;
    constexpr uint64_t C = 3;
    constexpr int blankLabel = 0;

    CtcLayerNetwork network = makeTinyCtcNetwork(false);
    const CtcNumericalCase testCase = makeRepeatedAndVariableLengthCase();
    populateTinyCtcNetwork(network, testCase.activations, testCase.paddedLabels, testCase.labelLengths, testCase.inputLengths);

    vector<float> actualGradient;
    runTinyCtcNetwork(network, &actualGradient);

    const vector<double> referenceGradient = finiteDifferenceCtcLogitGradient(testCase.activations,
                                                                             T,
                                                                             B,
                                                                             C,
                                                                             testCase.packedLabels,
                                                                             testCase.labelLengths,
                                                                             testCase.inputLengths,
                                                                             blankLabel,
                                                                             1.0e-3);

    ASSERT_EQ(actualGradient.size(), referenceGradient.size());
    const double gradientScale = static_cast<double>(Loss::getLossScalingFactor());
    for (uint64_t i = 0; i < actualGradient.size(); ++i) {
        const double expected = referenceGradient[i] * gradientScale;
        const double tolerance = std::abs(expected) < 1.0e-10 ? 1.0e-5 : 8.0e-2;
        ASSERT_TRUE(std::isfinite(actualGradient[i]));
        EXPECT_NEAR(static_cast<double>(actualGradient[i]), expected, tolerance) << "gradient index " << i;
    }

    LayerTestHelper::tearDownNetwork(network.layers);
}

TEST(CtcLossImplementationLayer, LossWeightScalesForwardAndBackwardNumerically) {
    constexpr uint64_t T = 4;
    constexpr uint64_t B = 2;
    constexpr uint64_t C = 3;
    constexpr int blankLabel = 0;
    constexpr float lossWeight = 0.25f;

    CtcLayerNetwork network = makeTinyCtcNetwork(false, DataType::FP32, DataType::INT32, lossWeight);

    const CtcNumericalCase testCase = makeRepeatedAndVariableLengthCase();
    populateTinyCtcNetwork(network, testCase.activations, testCase.paddedLabels, testCase.labelLengths, testCase.inputLengths);

    vector<float> actualGradient;
    runTinyCtcNetwork(network, &actualGradient);

    const vector<double> expectedLoss = cpuCtcLossForActivations(testCase.activations,
                                                                T,
                                                                B,
                                                                C,
                                                                testCase.packedLabels,
                                                                testCase.labelLengths,
                                                                testCase.inputLengths,
                                                                blankLabel);
    const vector<double> referenceGradient = finiteDifferenceCtcLogitGradient(testCase.activations,
                                                                             T,
                                                                             B,
                                                                             C,
                                                                             testCase.packedLabels,
                                                                             testCase.labelLengths,
                                                                             testCase.inputLengths,
                                                                             blankLabel,
                                                                             1.0e-3);

    Tensor actualLossCpu = network.lossOutput->getFeatureOutput().value();
    const float* actualLoss = actualLossCpu.getMemPtr<float>();
    for (uint64_t i = 0; i < B; ++i) {
        ASSERT_TRUE(std::isfinite(actualLoss[i]));
        EXPECT_NEAR(actualLoss[i], expectedLoss[i] * lossWeight, 1.0e-4f);
    }

    ASSERT_EQ(actualGradient.size(), referenceGradient.size());
    const double gradientScale = static_cast<double>(Loss::getLossScalingFactor()) * static_cast<double>(lossWeight);
    for (uint64_t i = 0; i < actualGradient.size(); ++i) {
        const double expected = referenceGradient[i] * gradientScale;
        const double tolerance = std::abs(expected) < 1.0e-10 ? 1.0e-5 : 8.0e-2;
        ASSERT_TRUE(std::isfinite(actualGradient[i]));
        EXPECT_NEAR(static_cast<double>(actualGradient[i]), expected, tolerance) << "weighted gradient index " << i;
    }

    LayerTestHelper::tearDownNetwork(network.layers);
}
