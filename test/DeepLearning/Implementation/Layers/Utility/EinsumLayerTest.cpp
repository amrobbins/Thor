#include "DeepLearning/Implementation/Layers/Utility/EinsumLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

class EinsumBatchCardinalityCaptureLayer final : public NoOpLayer {
   public:
    void forward(std::optional<Tensor> featureInput, bool validationPass, uint32_t validExampleCount = 0) override {
        observedValidExampleCounts.push_back(validExampleCount);
        NoOpLayer::forward(featureInput, validationPass, validExampleCount);
    }

    std::vector<uint32_t> observedValidExampleCounts;
};

void waitForCpuOutput(const std::shared_ptr<NetworkOutput>& output, Stream stream) {
    stream.waitEvent(output->getOutputReadyEvent());
    stream.synchronize();
}

}  // namespace

TEST(EinsumLayer, ForwardMatrixContractionPreservesImplicitBatchAndPartialBatchCardinality) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    constexpr uint32_t batchCapacity = 3;
    Tensor lhsCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {batchCapacity, 2, 3}));
    Tensor rhsCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {batchCapacity, 3, 2}));

    float* lhs = lhsCpu.getMemPtr<float>();
    float* rhs = rhsCpu.getMemPtr<float>();
    for (uint32_t b = 0; b < batchCapacity; ++b) {
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t k = 0; k < 3; ++k) {
                lhs[(b * 2 + i) * 3 + k] = static_cast<float>(1 + 10 * b + 3 * i + k);
            }
        }
        for (uint32_t k = 0; k < 3; ++k) {
            for (uint32_t j = 0; j < 2; ++j) {
                rhs[(b * 3 + k) * 2 + j] = static_cast<float>(1 + 7 * b + 2 * k + j);
            }
        }
    }

    auto lhsInput = std::make_shared<NetworkInput>(gpuPlacement, DataType::FP32, std::vector<unsigned long>{batchCapacity, 2, 3});
    auto rhsInput = std::make_shared<NetworkInput>(gpuPlacement, DataType::FP32, std::vector<unsigned long>{batchCapacity, 3, 2});
    auto einsum = std::make_shared<EinsumLayer>("ik,kj->ij");
    auto cardinality = std::make_shared<EinsumBatchCardinalityCaptureLayer>();
    auto output = std::make_shared<NetworkOutput>(cpuPlacement);
    std::vector<std::shared_ptr<Layer>> layers{lhsInput, rhsInput, einsum, cardinality, output};

    lhsInput->connectToNextLayer(einsum.get(), 0, 0);
    rhsInput->connectToNextLayer(einsum.get(), 0, 1);
    einsum->connectToNextLayer(cardinality.get());
    cardinality->connectToNextLayer(output.get());

    ASSERT_TRUE(einsum->getFeatureOutput().has_value());
    EXPECT_EQ(einsum->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{batchCapacity, 2, 2}));

    LayerTestHelper::initializeNetwork(layers);
    ASSERT_NE(einsum->getStampedForwardExecution(), nullptr);
    ASSERT_EQ(einsum->getBackwardPlan().feature_equation.output_dimensions, (std::vector<uint64_t>{2, 2}));

    // Submit in reverse operand order to ensure execution waits for every input.
    rhsInput->forward(rhsCpu, false, 2);
    EXPECT_TRUE(cardinality->observedValidExampleCounts.empty());
    lhsInput->forward(lhsCpu, false, 2);
    ASSERT_EQ(cardinality->observedValidExampleCounts, (std::vector<uint32_t>{2}));

    waitForCpuOutput(output, einsum->getStream());
    ASSERT_TRUE(output->getFeatureOutput().has_value());
    const Tensor outputCpu = output->getFeatureOutput().value();
    const float* observed = outputCpu.getMemPtr<float>();

    for (uint32_t b = 0; b < 2; ++b) {
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 2; ++j) {
                float expected = 0.0f;
                for (uint32_t k = 0; k < 3; ++k) {
                    expected += lhs[(b * 2 + i) * 3 + k] * rhs[(b * 3 + k) * 2 + j];
                }
                EXPECT_NEAR(observed[(b * 2 + i) * 2 + j], expected, 1.0e-3f);
            }
        }
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(EinsumLayer, ResolvedPhysicalBatchDoesNotDisturbFeatureEllipsisPosition) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    constexpr uint32_t batchCapacity = 2;
    Tensor lhsCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {batchCapacity, 2, 3, 4}));
    Tensor rhsCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {batchCapacity, 4, 2}));

    float* lhs = lhsCpu.getMemPtr<float>();
    float* rhs = rhsCpu.getMemPtr<float>();
    for (uint32_t b = 0; b < batchCapacity; ++b) {
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t e = 0; e < 3; ++e) {
                for (uint32_t j = 0; j < 4; ++j) {
                    lhs[((b * 2 + i) * 3 + e) * 4 + j] =
                        static_cast<float>(1 + 20 * b + 7 * i + 2 * e + j);
                }
            }
        }
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t k = 0; k < 2; ++k) {
                rhs[(b * 4 + j) * 2 + k] = static_cast<float>(1 + 11 * b + 2 * j + k);
            }
        }
    }

    auto lhsInput = std::make_shared<NetworkInput>(gpuPlacement, DataType::FP32, std::vector<unsigned long>{batchCapacity, 2, 3, 4});
    auto rhsInput = std::make_shared<NetworkInput>(gpuPlacement, DataType::FP32, std::vector<unsigned long>{batchCapacity, 4, 2});
    auto einsum = std::make_shared<EinsumLayer>("i...j,jk->i...k");
    auto output = std::make_shared<NetworkOutput>(cpuPlacement);
    std::vector<std::shared_ptr<Layer>> layers{lhsInput, rhsInput, einsum, output};

    lhsInput->connectToNextLayer(einsum.get(), 0, 0);
    rhsInput->connectToNextLayer(einsum.get(), 0, 1);
    einsum->connectToNextLayer(output.get());
    ASSERT_EQ(einsum->getFeatureOutput()->getDimensions(), (std::vector<uint64_t>{batchCapacity, 2, 3, 2}));

    LayerTestHelper::initializeNetwork(layers);
    lhsInput->forward(lhsCpu, false);
    rhsInput->forward(rhsCpu, false);
    waitForCpuOutput(output, einsum->getStream());

    const float* observed = output->getFeatureOutput()->getMemPtr<float>();
    for (uint32_t b = 0; b < batchCapacity; ++b) {
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t e = 0; e < 3; ++e) {
                for (uint32_t k = 0; k < 2; ++k) {
                    float expected = 0.0f;
                    for (uint32_t j = 0; j < 4; ++j) {
                        expected += lhs[((b * 2 + i) * 3 + e) * 4 + j] * rhs[(b * 4 + j) * 2 + k];
                    }
                    EXPECT_NEAR(observed[((b * 2 + i) * 3 + e) * 2 + k], expected, 1.0e-3f);
                }
            }
        }
    }

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(EinsumLayer, RejectsMismatchedRuntimeValidExampleCountsAcrossOperands) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Tensor lhsCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {4, 2, 3}));
    Tensor rhsCpu(cpuPlacement, TensorDescriptor(DataType::FP32, {4, 3, 2}));

    auto lhsInput = std::make_shared<NetworkInput>(gpuPlacement, DataType::FP32, std::vector<unsigned long>{4, 2, 3});
    auto rhsInput = std::make_shared<NetworkInput>(gpuPlacement, DataType::FP32, std::vector<unsigned long>{4, 3, 2});
    auto einsum = std::make_shared<EinsumLayer>("ik,kj->ij");
    auto output = std::make_shared<NetworkOutput>(gpuPlacement);
    std::vector<std::shared_ptr<Layer>> layers{lhsInput, rhsInput, einsum, output};

    lhsInput->connectToNextLayer(einsum.get(), 0, 0);
    rhsInput->connectToNextLayer(einsum.get(), 0, 1);
    einsum->connectToNextLayer(output.get());
    LayerTestHelper::initializeNetwork(layers);

    lhsInput->forward(lhsCpu, false, 2);
    EXPECT_THROW(rhsInput->forward(rhsCpu, false, 3), std::logic_error);

    LayerTestHelper::tearDownNetwork(layers);
}

TEST(EinsumLayer, RejectsDifferentPhysicalBatchCapacitiesAtConnectionTime) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    auto lhsInput = std::make_shared<NetworkInput>(gpuPlacement, DataType::FP32, std::vector<unsigned long>{4, 2, 3});
    auto rhsInput = std::make_shared<NetworkInput>(gpuPlacement, DataType::FP32, std::vector<unsigned long>{5, 3, 2});
    auto einsum = std::make_shared<EinsumLayer>("ik,kj->ij");

    lhsInput->connectToNextLayer(einsum.get(), 0, 0);
    EXPECT_THROW(rhsInput->connectToNextLayer(einsum.get(), 0, 1), std::logic_error);
}

namespace {

class EinsumGradientCaptureLayer final : public NoOpLayer {
   public:
    void backward(std::optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        capturedErrors.push_back(errorInput);
        capturedBatchSizes.push_back(batchSize);
    }

    std::vector<std::optional<Tensor>> capturedErrors;
    std::vector<uint32_t> capturedBatchSizes;
};

struct EinsumBackwardRunResult {
    std::vector<Tensor> gradientsCpu;
    std::vector<uint32_t> capturedBatchSizes;
    std::vector<bool> hasPostprocess;
};

EinsumBackwardRunResult runEinsumBackward(const std::string& equation,
                                          const std::vector<Tensor>& inputCpu,
                                          const Tensor& upstreamGradientCpu,
                                          uint32_t validExampleCount = 0) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    std::vector<Stream> streams;
    std::vector<Tensor> inputGpu;
    std::vector<std::unique_ptr<EinsumGradientCaptureLayer>> captures;
    streams.reserve(inputCpu.size());
    inputGpu.reserve(inputCpu.size());
    captures.reserve(inputCpu.size());

    auto einsum = std::make_unique<EinsumLayer>(equation);
    for (size_t i = 0; i < inputCpu.size(); ++i) {
        streams.emplace_back(0);
        inputGpu.emplace_back(gpuPlacement, inputCpu[i].getDescriptor());
        inputGpu.back().copyFromAsync(inputCpu[i], streams.back());
        captures.push_back(std::make_unique<EinsumGradientCaptureLayer>());
        einsum->connectToPreviousLayer(
            captures.back().get(), inputGpu.back(), streams.back(), true, static_cast<int>(i));
    }

    // Reproduce the production forward rendezvous before invoking backward in
    // isolation.  A real EinsumLayer forward has already made stream 0 wait for
    // every preserved operand producer before its output can reach downstream
    // backward.  Keep that transitive dependency here so the fixture does not
    // race an operand setup copy against a reverse contraction.
    for (size_t i = 1; i < streams.size(); ++i) {
        streams[0].waitEvent(streams[i].putEvent());
    }

    NoOpLayer sink;
    einsum->connectToNextLayer(&sink);
    einsum->compile();
    einsum->initialize();

    std::vector<std::optional<Tensor>> errorInputs = einsum->getErrorInputs();
    if (errorInputs.size() != 1 || !errorInputs[0].has_value()) {
        throw std::runtime_error("Einsum backward test expected one connected downstream gradient tensor.");
    }
    errorInputs[0]->copyFromAsync(upstreamGradientCpu, streams[0]);

    std::vector<bool> hasPostprocess;
    hasPostprocess.reserve(inputCpu.size());
    for (size_t i = 0; i < inputCpu.size(); ++i) {
        const uint32_t operandIndex = static_cast<uint32_t>(i);
        if (einsum->getStampedBackwardContraction(operandIndex) == nullptr) {
            throw std::runtime_error("Einsum backward test expected a stamped contraction for every live operand.");
        }
        hasPostprocess.push_back(einsum->backwardOperandHasPostprocess(operandIndex));
    }

    einsum->backward(errorInputs[0], validExampleCount);

    const std::vector<std::optional<Tensor>> errorOutputs = einsum->getErrorOutputs();
    if (errorOutputs.size() != inputCpu.size()) {
        throw std::runtime_error("Einsum backward test received an unexpected gradient-output count.");
    }
    std::vector<Tensor> gradientsCpu;
    gradientsCpu.reserve(errorOutputs.size());
    for (size_t i = 0; i < errorOutputs.size(); ++i) {
        if (!errorOutputs[i].has_value()) {
            throw std::runtime_error("Einsum backward test expected every operand gradient output to be live.");
        }
        gradientsCpu.emplace_back(cpuPlacement, errorOutputs[i]->getDescriptor());
        gradientsCpu.back().copyFromAsync(errorOutputs[i].value(), streams[i]);
    }
    for (Stream& stream : streams) {
        stream.synchronize();
    }

    std::vector<uint32_t> capturedBatchSizes;
    capturedBatchSizes.reserve(captures.size());
    for (const auto& capture : captures) {
        if (capture->capturedErrors.size() != 1 || capture->capturedBatchSizes.size() != 1) {
            throw std::runtime_error("Einsum backward test expected exactly one upstream callback per operand.");
        }
        capturedBatchSizes.push_back(capture->capturedBatchSizes[0]);
    }

    einsum->cleanup();
    return EinsumBackwardRunResult{std::move(gradientsCpu), std::move(capturedBatchSizes), std::move(hasPostprocess)};
}

}  // namespace

TEST(EinsumLayer, BackwardMatrixContractionMatchesAnalyticAndFiniteDifferenceGradients) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    constexpr uint32_t batch = 1;
    Tensor lhs(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 2, 3}));
    Tensor rhs(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 3, 2}));
    Tensor dOutput(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 2, 2}));

    const std::vector<float> lhsValues{0.25f, -0.5f, 1.0f, 1.5f, 0.75f, -0.25f};
    const std::vector<float> rhsValues{0.5f, -1.0f, 1.25f, 0.75f, -0.5f, 2.0f};
    const std::vector<float> gradValues{1.0f, -0.75f, 0.5f, 1.25f};
    std::copy(lhsValues.begin(), lhsValues.end(), lhs.getMemPtr<float>());
    std::copy(rhsValues.begin(), rhsValues.end(), rhs.getMemPtr<float>());
    std::copy(gradValues.begin(), gradValues.end(), dOutput.getMemPtr<float>());

    EinsumBackwardRunResult result = runEinsumBackward("ik,kj->ij", {lhs, rhs}, dOutput, 1);
    ASSERT_EQ(result.gradientsCpu.size(), 2u);
    EXPECT_FALSE(result.hasPostprocess[0]);
    EXPECT_FALSE(result.hasPostprocess[1]);
    EXPECT_EQ(result.capturedBatchSizes, (std::vector<uint32_t>{1, 1}));

    const float* dLhs = result.gradientsCpu[0].getMemPtr<float>();
    const float* dRhs = result.gradientsCpu[1].getMemPtr<float>();
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t k = 0; k < 3; ++k) {
            float expected = 0.0f;
            for (uint32_t j = 0; j < 2; ++j) {
                expected += gradValues[i * 2 + j] * rhsValues[k * 2 + j];
            }
            EXPECT_NEAR(dLhs[i * 3 + k], expected, 2.0e-3f);
        }
    }
    for (uint32_t k = 0; k < 3; ++k) {
        for (uint32_t j = 0; j < 2; ++j) {
            float expected = 0.0f;
            for (uint32_t i = 0; i < 2; ++i) {
                expected += lhsValues[i * 3 + k] * gradValues[i * 2 + j];
            }
            EXPECT_NEAR(dRhs[k * 2 + j], expected, 2.0e-3f);
        }
    }

    auto loss = [&](const std::vector<float>& lhsCandidate, const std::vector<float>& rhsCandidate) -> double {
        double total = 0.0;
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 2; ++j) {
                double y = 0.0;
                for (uint32_t k = 0; k < 3; ++k) {
                    y += static_cast<double>(lhsCandidate[i * 3 + k]) * rhsCandidate[k * 2 + j];
                }
                total += y * gradValues[i * 2 + j];
            }
        }
        return total;
    };
    constexpr float epsilon = 1.0e-3f;
    for (size_t index = 0; index < lhsValues.size(); ++index) {
        std::vector<float> plus = lhsValues;
        std::vector<float> minus = lhsValues;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const double finiteDifference = (loss(plus, rhsValues) - loss(minus, rhsValues)) / (2.0 * epsilon);
        EXPECT_NEAR(dLhs[index], finiteDifference, 2.0e-2);
    }
    for (size_t index = 0; index < rhsValues.size(); ++index) {
        std::vector<float> plus = rhsValues;
        std::vector<float> minus = rhsValues;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        const double finiteDifference = (loss(lhsValues, plus) - loss(lhsValues, minus)) / (2.0 * epsilon);
        EXPECT_NEAR(dRhs[index], finiteDifference, 2.0e-2);
    }
}

TEST(EinsumLayer, BackwardReducesEquationBroadcastBackToSingletonOperandExtent) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    constexpr uint32_t batch = 2;
    Tensor lhs(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 1, 2, 3}));
    Tensor rhs(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 2, 3, 2}));
    Tensor dOutput(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 2, 2, 2}));

    for (uint64_t i = 0; i < lhs.getTotalNumElements(); ++i) lhs.getMemPtr<float>()[i] = 0.25f + 0.1f * i;
    for (uint64_t i = 0; i < rhs.getTotalNumElements(); ++i) rhs.getMemPtr<float>()[i] = -0.5f + 0.05f * i;
    for (uint64_t i = 0; i < dOutput.getTotalNumElements(); ++i) dOutput.getMemPtr<float>()[i] = 0.2f + 0.03f * i;

    EinsumBackwardRunResult result = runEinsumBackward("qij,qjk->qik", {lhs, rhs}, dOutput, 2);
    ASSERT_EQ(result.gradientsCpu.size(), 2u);
    EXPECT_TRUE(result.hasPostprocess[0]);

    const float* lhsValues = lhs.getMemPtr<float>();
    const float* rhsValues = rhs.getMemPtr<float>();
    const float* gradValues = dOutput.getMemPtr<float>();
    const float* dLhs = result.gradientsCpu[0].getMemPtr<float>();
    const float* dRhs = result.gradientsCpu[1].getMemPtr<float>();

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 3; ++j) {
                float expected = 0.0f;
                for (uint32_t q = 0; q < 2; ++q) {
                    for (uint32_t k = 0; k < 2; ++k) {
                        const uint64_t gradIndex = (((b * 2 + q) * 2 + i) * 2 + k);
                        const uint64_t rhsIndex = (((b * 2 + q) * 3 + j) * 2 + k);
                        expected += gradValues[gradIndex] * rhsValues[rhsIndex];
                    }
                }
                EXPECT_NEAR(dLhs[(b * 2 + i) * 3 + j], expected, 3.0e-3f);
            }
        }
        for (uint32_t q = 0; q < 2; ++q) {
            for (uint32_t j = 0; j < 3; ++j) {
                for (uint32_t k = 0; k < 2; ++k) {
                    float expected = 0.0f;
                    for (uint32_t i = 0; i < 2; ++i) {
                        const uint64_t lhsIndex = (b * 2 + i) * 3 + j;
                        const uint64_t gradIndex = (((b * 2 + q) * 2 + i) * 2 + k);
                        expected += lhsValues[lhsIndex] * gradValues[gradIndex];
                    }
                    const uint64_t rhsGradIndex = (((b * 2 + q) * 3 + j) * 2 + k);
                    EXPECT_NEAR(dRhs[rhsGradIndex], expected, 3.0e-3f);
                }
            }
        }
    }
}

TEST(EinsumLayer, BackwardExpandsSingletonAndTargetOnlyAxes) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    constexpr uint32_t batch = 2;
    Tensor matrix(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 2, 3}));
    Tensor vector(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 1}));
    Tensor dOutput(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 2}));

    for (uint64_t i = 0; i < matrix.getTotalNumElements(); ++i) matrix.getMemPtr<float>()[i] = 1.0f + 0.2f * i;
    vector.getMemPtr<float>()[0] = 0.75f;
    vector.getMemPtr<float>()[1] = -1.25f;
    for (uint64_t i = 0; i < dOutput.getTotalNumElements(); ++i) dOutput.getMemPtr<float>()[i] = 0.5f + 0.15f * i;

    EinsumBackwardRunResult result = runEinsumBackward("ij,j->i", {matrix, vector}, dOutput, 2);
    ASSERT_EQ(result.gradientsCpu.size(), 2u);
    EXPECT_TRUE(result.hasPostprocess[0]);
    EXPECT_TRUE(result.hasPostprocess[1]);

    const float* x = matrix.getMemPtr<float>();
    const float* v = vector.getMemPtr<float>();
    const float* dy = dOutput.getMemPtr<float>();
    const float* dx = result.gradientsCpu[0].getMemPtr<float>();
    const float* dv = result.gradientsCpu[1].getMemPtr<float>();
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 3; ++j) {
                EXPECT_NEAR(dx[(b * 2 + i) * 3 + j], dy[b * 2 + i] * v[b], 2.0e-3f);
            }
        }
        float expectedVectorGrad = 0.0f;
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 3; ++j) {
                expectedVectorGrad += dy[b * 2 + i] * x[(b * 2 + i) * 3 + j];
            }
        }
        EXPECT_NEAR(dv[b], expectedVectorGrad, 3.0e-3f);
    }
}

TEST(EinsumLayer, BackwardRestoresForwardReducedTargetOnlyAxis) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    constexpr uint32_t batch = 2;
    Tensor input(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 2, 3}));
    Tensor dOutput(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, 2}));

    for (uint64_t i = 0; i < input.getTotalNumElements(); ++i) {
        input.getMemPtr<float>()[i] = 0.1f * static_cast<float>(i + 1);
    }
    dOutput.getMemPtr<float>()[0] = 0.5f;
    dOutput.getMemPtr<float>()[1] = -0.75f;
    dOutput.getMemPtr<float>()[2] = 1.25f;
    dOutput.getMemPtr<float>()[3] = -1.5f;

    EinsumBackwardRunResult result = runEinsumBackward("ij->i", {input}, dOutput, batch);
    ASSERT_EQ(result.gradientsCpu.size(), 1u);
    EXPECT_TRUE(result.hasPostprocess[0]);

    const float* gradient = result.gradientsCpu[0].getMemPtr<float>();
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < 2; ++i) {
            for (uint32_t j = 0; j < 3; ++j) {
                EXPECT_NEAR(gradient[(b * 2 + i) * 3 + j], dOutput.getMemPtr<float>()[b * 2 + i], 2.0e-3f);
            }
        }
    }
}

TEST(EinsumLayer, BackwardDiagonalScatterRestoresDenseInputWithZeroOffDiagonal) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    constexpr uint32_t batch = 2;
    constexpr uint32_t dimension = 3;
    Tensor matrix(cpuPlacement, TensorDescriptor(DataType::FP32, {batch, dimension, dimension}));
    Tensor dOutput(cpuPlacement, TensorDescriptor(DataType::FP32, {batch}));
    for (uint64_t i = 0; i < matrix.getTotalNumElements(); ++i) matrix.getMemPtr<float>()[i] = 0.1f * static_cast<float>(i + 1);
    dOutput.getMemPtr<float>()[0] = 1.5f;
    dOutput.getMemPtr<float>()[1] = -0.75f;

    EinsumBackwardRunResult result = runEinsumBackward("ii->", {matrix}, dOutput, 2);
    ASSERT_EQ(result.gradientsCpu.size(), 1u);
    EXPECT_TRUE(result.hasPostprocess[0]);
    const float* gradient = result.gradientsCpu[0].getMemPtr<float>();
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < dimension; ++i) {
            for (uint32_t j = 0; j < dimension; ++j) {
                const float expected = i == j ? dOutput.getMemPtr<float>()[b] : 0.0f;
                EXPECT_NEAR(gradient[(b * dimension + i) * dimension + j], expected, 2.0e-3f);
            }
        }
    }
}

TEST(EinsumLayer, BackwardStampsOnlyLiveOperandGradientPaths) {
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Tensor lhs(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2, 3}));
    Tensor rhs(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, 2}));
    Stream lhsStream(0);
    Stream rhsStream(0);
    EinsumGradientCaptureLayer lhsCapture;
    EinsumGradientCaptureLayer rhsCapture;
    NoOpLayer sink;
    EinsumLayer einsum("ik,kj->ij");

    einsum.connectToPreviousLayer(&lhsCapture, lhs, lhsStream, false, 0);
    einsum.connectToPreviousLayer(&rhsCapture, rhs, rhsStream, true, 1);
    einsum.connectToNextLayer(&sink);
    einsum.compile();
    einsum.initialize();

    EXPECT_EQ(einsum.getStampedBackwardContraction(0), nullptr);
    EXPECT_NE(einsum.getStampedBackwardContraction(1), nullptr);
    EXPECT_FALSE(einsum.getErrorOutputs()[0].has_value());
    EXPECT_TRUE(einsum.getErrorOutputs()[1].has_value());

    einsum.cleanup();
}

TEST(EinsumLayer, BackwardPreservesPartialBatchCardinality) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Tensor input(cpuPlacement, TensorDescriptor(DataType::FP32, {3, 2}));
    Tensor dOutput(cpuPlacement, TensorDescriptor(DataType::FP32, {3, 2}));
    for (uint64_t i = 0; i < input.getTotalNumElements(); ++i) {
        input.getMemPtr<float>()[i] = 0.1f * static_cast<float>(i + 1);
        dOutput.getMemPtr<float>()[i] = -0.2f + 0.05f * static_cast<float>(i);
    }

    EinsumBackwardRunResult result = runEinsumBackward("i->i", {input}, dOutput, 2);
    ASSERT_EQ(result.gradientsCpu.size(), 1u);
    ASSERT_EQ(result.capturedBatchSizes, (std::vector<uint32_t>{2}));
    const float* observed = result.gradientsCpu[0].getMemPtr<float>();
    for (uint64_t i = 0; i < dOutput.getTotalNumElements(); ++i) {
        EXPECT_NEAR(observed[i], dOutput.getMemPtr<float>()[i], 1.0e-6f);
    }
}
