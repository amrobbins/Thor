#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;
using DataType = TensorDescriptor::DataType;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

void expectMapHasValue(const std::unordered_map<std::string, float>& values, const std::string& name, float expected) {
    ASSERT_TRUE(values.contains(name)) << "Missing hyperparameter: " << name;
    EXPECT_FLOAT_EQ(values.at(name), expected) << "Hyperparameter mismatch for " << name;
}

float computeCurrentLearningRate(float initialLearningRate, float decay, uint64_t epoch) {
    return static_cast<float>(static_cast<double>(initialLearningRate) * std::pow(1.0 - static_cast<double>(decay), epoch));
}

void writeCpuFp32Tensor(Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(tensor.getTotalNumElements(), values.size());

    float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

std::vector<float> readCpuFp32Tensor(const Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);

    std::vector<float> values(tensor.getTotalNumElements());
    const float* ptr = tensor.getMemPtr<float>();
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];

    return values;
}

void copyValuesToGpuFp32Tensor(Tensor& gpuTensor, const std::vector<float>& values, Stream& stream) {
    ASSERT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    ASSERT_EQ(gpuTensor.getDataType(), DataType::FP32);
    ASSERT_EQ(gpuTensor.getTotalNumElements(), values.size());

    Tensor host(cpuPlacement, gpuTensor.getDescriptor());
    writeCpuFp32Tensor(host, values);

    gpuTensor.copyFromAsync(host, stream);
    stream.synchronize();
}

std::vector<float> copyGpuFp32TensorToValues(const Tensor& gpuTensor, Stream& stream) {
    EXPECT_EQ(gpuTensor.getPlacement(), gpuPlacement);
    EXPECT_EQ(gpuTensor.getDataType(), DataType::FP32);

    Tensor host = gpuTensor.clone(cpuPlacement);
    host.copyFromAsync(gpuTensor, stream);
    stream.synchronize();

    return readCpuFp32Tensor(host);
}

void expectAllClose(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-5f, float rtol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());

    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(diff, tol) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

struct SgdReferenceState {
    std::vector<float> weights;
    std::vector<float> velocity;
};

float computeStep(float currentLearningRate, uint32_t batchSize) {
    EXPECT_GT(batchSize, 0u);
    return currentLearningRate / (static_cast<float>(batchSize) * Loss::getLossScalingFactor());
}

void applyPlainSgdReferenceStep(SgdReferenceState& state,
                                const std::vector<float>& rawGradient,
                                uint32_t batchSize,
                                float currentLearningRate) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());

    const float step = computeStep(currentLearningRate, batchSize);
    for (uint64_t i = 0; i < rawGradient.size(); ++i)
        state.weights[i] -= step * rawGradient[i];
}

void applyMomentumSgdReferenceStep(SgdReferenceState& state,
                                   const std::vector<float>& rawGradient,
                                   uint32_t batchSize,
                                   float currentLearningRate,
                                   float momentum,
                                   bool useNesterovMomentum) {
    ASSERT_EQ(state.weights.size(), rawGradient.size());
    ASSERT_EQ(state.velocity.size(), rawGradient.size());

    const float step = computeStep(currentLearningRate, batchSize);
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float vNext = momentum * state.velocity[i] - step * rawGradient[i];

        if (useNesterovMomentum) {
            state.weights[i] += momentum * vNext - step * rawGradient[i];
        } else {
            state.weights[i] += vNext;
        }

        state.velocity[i] = vNext;
    }
}

void runSgdStep(Sgd& sgd, const std::vector<float>& rawGradient, uint32_t batchSize, Stream& stream) {
    Optional<Tensor> gradientOpt = sgd.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.isPresent());

    Tensor gradient = gradientOpt.get();
    copyValuesToGpuFp32Tensor(gradient, rawGradient, stream);

    sgd.updateWeights(batchSize);
    stream.synchronize();
}

Tensor getMomentumStorage(Sgd& sgd) {
    EXPECT_TRUE(sgd.hasParameter("momentum"));

    Optional<Tensor> storage = sgd.getParameter("momentum")->getStorage();
    EXPECT_TRUE(storage.isPresent());

    return storage.get();
}

}  // namespace

TEST(SgdTest, ConstructorGettersSettersAndHyperParameters) {
    constexpr uint64_t id = 17;
    Sgd sgd(id,
            /*initialLearningRate=*/0.5f,
            /*decay=*/0.2f,
            /*momentum=*/0.3f,
            /*useNesterovMomentum=*/true,
            /*startResumeEpoch=*/5);

    EXPECT_EQ(sgd.getId(), id);
    EXPECT_FLOAT_EQ(sgd.getInitialLearningRate(), 0.5f);
    EXPECT_FLOAT_EQ(sgd.getDecay(), 0.2f);
    EXPECT_FLOAT_EQ(sgd.getMomentum(), 0.3f);
    EXPECT_TRUE(sgd.getUseNesterovMomentum());
    EXPECT_EQ(sgd.getEpoch(), 5u);

    sgd.setInitialLearningRate(0.25f);
    sgd.setDecay(0.1f);
    sgd.setMomentum(0.7f);
    sgd.setUseNesterovMomentum(false);

    EXPECT_FLOAT_EQ(sgd.getInitialLearningRate(), 0.25f);
    EXPECT_FLOAT_EQ(sgd.getDecay(), 0.1f);
    EXPECT_FLOAT_EQ(sgd.getMomentum(), 0.7f);
    EXPECT_FALSE(sgd.getUseNesterovMomentum());

    constexpr uint64_t epoch = 3;
    const float expectedLearningRate = computeCurrentLearningRate(0.25f, 0.1f, epoch);

    std::unordered_map<std::string, float> updated = sgd.updateHyperParameters(epoch, /*batch=*/4, /*batchesPerEpoch=*/10);

    ASSERT_EQ(updated.size(), 1u);
    expectMapHasValue(updated, "currentLearningRate", expectedLearningRate);
    EXPECT_EQ(sgd.getEpoch(), epoch);

    std::unordered_map<std::string, float> all = sgd.getAllHyperParameters();
    ASSERT_EQ(all.size(), 5u);
    expectMapHasValue(all, "currentLearningRate", expectedLearningRate);
    expectMapHasValue(all, "initialLearningRate", 0.25f);
    expectMapHasValue(all, "decay", 0.1f);
    expectMapHasValue(all, "momentum", 0.7f);
    expectMapHasValue(all, "useNesterovMomentum", 0.0f);
}

TEST(SgdTest, CompileWithoutMomentumCreatesGradientAndWeightsOutputOnly) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP16, {2, 3}));

    Sgd sgd(1,
            /*initialLearningRate=*/0.01f,
            /*decay=*/0.0f,
            /*momentum=*/0.0f,
            /*useNesterovMomentum=*/false);

    EXPECT_FALSE(sgd.isCompiled());

    sgd.compile(weights, stream);
    stream.synchronize();

    EXPECT_TRUE(sgd.isCompiled());
    EXPECT_FALSE(sgd.hasParameter("momentum"));

    Optional<Tensor> gradientOpt = sgd.getWeightsGradient();
    ASSERT_TRUE(gradientOpt.isPresent());

    Tensor gradient = gradientOpt.get();
    EXPECT_EQ(gradient.getPlacement(), gpuPlacement);
    EXPECT_EQ(gradient.getDataType(), DataType::FP16);
    EXPECT_EQ(gradient.getDimensions(), weights.getDimensions());

    Tensor weightsOut = sgd.getOptimizerParameterTensor("weights");
    EXPECT_EQ(weightsOut, weights);

    std::vector<std::string> outputNames = sgd.getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights"}));

    EXPECT_THROW((void)sgd.getOptimizerParameterTensor("velocity"), std::runtime_error);
    EXPECT_THROW((void)sgd.getOptimizerParameterTensor("missing"), std::runtime_error);
}

TEST(SgdTest, CompileWithMomentumCreatesVelocityParameterAndNamedOutputs) {
    Stream stream(gpuPlacement);

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, stream);

    Sgd sgd(2,
            /*initialLearningRate=*/0.01f,
            /*decay=*/0.0f,
            /*momentum=*/0.8f,
            /*useNesterovMomentum=*/false);

    sgd.compile(weights, stream);
    stream.synchronize();

    EXPECT_TRUE(sgd.isCompiled());
    EXPECT_TRUE(sgd.hasParameter("momentum"));

    Tensor weightsOut = sgd.getOptimizerParameterTensor("weights");
    Tensor velocityOut = sgd.getOptimizerParameterTensor("velocity");
    Tensor momentumStorage = getMomentumStorage(sgd);

    EXPECT_EQ(weightsOut, weights);
    EXPECT_EQ(velocityOut, momentumStorage);
    EXPECT_EQ(momentumStorage.getPlacement(), gpuPlacement);
    EXPECT_EQ(momentumStorage.getDataType(), DataType::FP32);
    EXPECT_EQ(momentumStorage.getDimensions(), weights.getDimensions());

    std::vector<std::string> outputNames = sgd.getOptimizerParameterNames();
    std::set<std::string> names(outputNames.begin(), outputNames.end());
    EXPECT_EQ(names, (std::set<std::string>{"weights", "velocity"}));

    expectAllClose(copyGpuFp32TensorToValues(momentumStorage, stream), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(SgdTest, PlainSgdSingleStepMatchesCpuReference) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gradient{8.0f, -12.0f, 0.5f, -2.0f};

    constexpr uint32_t batchSize = 2;
    constexpr float initialLearningRate = 0.1f;
    constexpr float decay = 0.0f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Sgd sgd(3,
            initialLearningRate,
            decay,
            /*momentum=*/0.0f,
            /*useNesterovMomentum=*/false);

    sgd.compile(weights, stream);
    stream.synchronize();

    SgdReferenceState expected;
    expected.weights = initialWeights;

    const float currentLearningRate = initialLearningRate;
    applyPlainSgdReferenceStep(expected, gradient, batchSize, currentLearningRate);
    runSgdStep(sgd, gradient, batchSize, stream);

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 2e-5f, 2e-5f);

    std::unordered_map<std::string, float> all = sgd.getAllHyperParameters();
    expectMapHasValue(all, "currentLearningRate", currentLearningRate);
    expectMapHasValue(all, "momentum", 0.0f);
    expectMapHasValue(all, "useNesterovMomentum", 0.0f);
}

TEST(SgdTest, PlainSgdTwoStepsUseUpdatedLearningRate) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<float> gradient1{10.0f, -20.0f, 0.5f, -4.0f, 8.0f, -16.0f};
    const std::vector<float> gradient2{-3.0f, 5.0f, 7.0f, -11.0f, 13.0f, -17.0f};

    constexpr float initialLearningRate = 0.2f;
    constexpr float decay = 0.25f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Sgd sgd(4,
            initialLearningRate,
            decay,
            /*momentum=*/0.0f,
            /*useNesterovMomentum=*/false);

    sgd.compile(weights, stream);
    stream.synchronize();

    SgdReferenceState expected;
    expected.weights = initialWeights;

    sgd.updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/10);
    float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, 0);
    applyPlainSgdReferenceStep(expected, gradient1, /*batchSize=*/2, currentLearningRate);
    runSgdStep(sgd, gradient1, /*batchSize=*/2, stream);

    sgd.updateHyperParameters(/*epoch=*/2, /*batch=*/3, /*batchesPerEpoch=*/10);
    currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, 2);
    applyPlainSgdReferenceStep(expected, gradient2, /*batchSize=*/4, currentLearningRate);
    runSgdStep(sgd, gradient2, /*batchSize=*/4, stream);

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);

    std::unordered_map<std::string, float> updated = sgd.updateHyperParameters(/*epoch=*/2, /*batch=*/4, /*batchesPerEpoch=*/10);
    ASSERT_EQ(updated.size(), 1u);
    expectMapHasValue(updated, "currentLearningRate", currentLearningRate);

    std::unordered_map<std::string, float> all = sgd.getAllHyperParameters();
    expectMapHasValue(all, "currentLearningRate", currentLearningRate);
    expectMapHasValue(all, "initialLearningRate", initialLearningRate);
    expectMapHasValue(all, "decay", decay);
}

TEST(SgdTest, ClassicalMomentumTwoStepsCarryVelocity) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    const std::vector<float> gradient1{10.0f, -20.0f, 0.5f, -4.0f, 8.0f, -16.0f};
    const std::vector<float> gradient2{-3.0f, 5.0f, 7.0f, -11.0f, 13.0f, -17.0f};

    constexpr float initialLearningRate = 0.12f;
    constexpr float decay = 0.1f;
    constexpr float momentum = 0.8f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 3}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Sgd sgd(5,
            initialLearningRate,
            decay,
            momentum,
            /*useNesterovMomentum=*/false);

    sgd.compile(weights, stream);
    stream.synchronize();

    Tensor velocity = sgd.getOptimizerParameterTensor("velocity");

    SgdReferenceState expected;
    expected.weights = initialWeights;
    expected.velocity.assign(initialWeights.size(), 0.0f);

    sgd.updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/10);
    float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, 0);
    applyMomentumSgdReferenceStep(expected,
                                  gradient1,
                                  /*batchSize=*/2,
                                  currentLearningRate,
                                  momentum,
                                  /*useNesterovMomentum=*/false);
    runSgdStep(sgd, gradient1, /*batchSize=*/2, stream);

    sgd.updateHyperParameters(/*epoch=*/3, /*batch=*/1, /*batchesPerEpoch=*/10);
    currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, 3);
    applyMomentumSgdReferenceStep(expected,
                                  gradient2,
                                  /*batchSize=*/4,
                                  currentLearningRate,
                                  momentum,
                                  /*useNesterovMomentum=*/false);
    runSgdStep(sgd, gradient2, /*batchSize=*/4, stream);

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(velocity, stream), expected.velocity, 3e-5f, 3e-5f);

    std::unordered_map<std::string, float> all = sgd.getAllHyperParameters();
    expectMapHasValue(all, "currentLearningRate", currentLearningRate);
    expectMapHasValue(all, "momentum", momentum);
    expectMapHasValue(all, "useNesterovMomentum", 0.0f);
}

TEST(SgdTest, NesterovMomentumTwoStepsCarryVelocity) {
    Stream stream(gpuPlacement);

    const std::vector<float> initialWeights{2.0f, -1.0f, 3.0f, -4.0f};
    const std::vector<float> gradient1{4.0f, -8.0f, 1.0f, -2.0f};
    const std::vector<float> gradient2{-5.0f, 6.0f, -7.0f, 8.0f};

    constexpr float initialLearningRate = 0.08f;
    constexpr float decay = 0.2f;
    constexpr float momentum = 0.6f;

    Tensor weights(gpuPlacement, TensorDescriptor(DataType::FP32, {2, 2}));
    copyValuesToGpuFp32Tensor(weights, initialWeights, stream);

    Sgd sgd(6,
            initialLearningRate,
            decay,
            momentum,
            /*useNesterovMomentum=*/true);

    sgd.compile(weights, stream);
    stream.synchronize();

    Tensor velocity = sgd.getOptimizerParameterTensor("velocity");

    SgdReferenceState expected;
    expected.weights = initialWeights;
    expected.velocity.assign(initialWeights.size(), 0.0f);

    sgd.updateHyperParameters(/*epoch=*/0, /*batch=*/0, /*batchesPerEpoch=*/10);
    float currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, 0);
    applyMomentumSgdReferenceStep(expected,
                                  gradient1,
                                  /*batchSize=*/1,
                                  currentLearningRate,
                                  momentum,
                                  /*useNesterovMomentum=*/true);
    runSgdStep(sgd, gradient1, /*batchSize=*/1, stream);

    sgd.updateHyperParameters(/*epoch=*/2, /*batch=*/5, /*batchesPerEpoch=*/10);
    currentLearningRate = computeCurrentLearningRate(initialLearningRate, decay, 2);
    applyMomentumSgdReferenceStep(expected,
                                  gradient2,
                                  /*batchSize=*/2,
                                  currentLearningRate,
                                  momentum,
                                  /*useNesterovMomentum=*/true);
    runSgdStep(sgd, gradient2, /*batchSize=*/2, stream);

    expectAllClose(copyGpuFp32TensorToValues(weights, stream), expected.weights, 3e-5f, 3e-5f);
    expectAllClose(copyGpuFp32TensorToValues(velocity, stream), expected.velocity, 3e-5f, 3e-5f);

    std::unordered_map<std::string, float> all = sgd.getAllHyperParameters();
    expectMapHasValue(all, "currentLearningRate", currentLearningRate);
    expectMapHasValue(all, "momentum", momentum);
    expectMapHasValue(all, "useNesterovMomentum", 1.0f);
}
