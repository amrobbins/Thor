#include "Utilities/TensorOperations/Loss/CtcLoss.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/CudaDriver/CudaGraph.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for CTC CUDA graph tests.";                                        \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

CudnnCtcLossConfig validConfig() {
    CudnnCtcLossConfig config;
    config.maxTimeSteps = 16;
    config.batchSize = 4;
    config.numClasses = 8;
    config.maxLabelLength = 15;
    config.dataType = DataType::FP32;
    return config;
}

template <typename T>
DataType dataTypeFor();

template <>
DataType dataTypeFor<float>() {
    return DataType::FP32;
}

template <>
DataType dataTypeFor<int32_t>() {
    return DataType::INT32;
}

template <>
DataType dataTypeFor<uint32_t>() {
    return DataType::UINT32;
}

template <>
DataType dataTypeFor<uint64_t>() {
    return DataType::UINT64;
}

template <typename T>
Tensor makeGpuVector(const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dataTypeFor<T>(), {static_cast<uint64_t>(values.size())}));
    std::copy(values.begin(), values.end(), cpu.getMemPtr<T>());
    Tensor gpu(gpuPlacement, cpu.getDescriptor());
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
void replaceGpuVector(Tensor& gpu, const std::vector<T>& values, Stream& stream) {
    ASSERT_EQ(gpu.getDataType(), dataTypeFor<T>());
    ASSERT_EQ(gpu.getTotalNumElements(), values.size());
    Tensor cpu(cpuPlacement, gpu.getDescriptor());
    std::copy(values.begin(), values.end(), cpu.getMemPtr<T>());
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
}

template <typename T>
std::vector<T> copyGpuVector(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const T* values = cpu.getMemPtr<T>();
    return std::vector<T>(values, values + cpu.getTotalNumElements());
}

struct CapturableCtcBuffers {
    Tensor activations;
    Tensor labels;
    Tensor offsets;
    Tensor inputLengths;
    Tensor generatedLabelLengths;
    Tensor validationErrorBits;
    Tensor costs;
    Tensor gradients;
    std::unique_ptr<Tensor> workspace;
};

CapturableCtcBuffers makeCapturableCtcBuffers(const CudnnCtcLossPlan& plan,
                                               const std::vector<float>& activations,
                                               const std::vector<int32_t>& labels,
                                               const std::vector<uint64_t>& offsets,
                                               const std::vector<int32_t>& inputLengths,
                                               Stream& stream) {
    const CudnnCtcLossConfig& config = plan.getConfig();
    EXPECT_EQ(activations.size(), static_cast<size_t>(config.batchSize) * config.maxTimeSteps * config.numClasses);
    EXPECT_EQ(offsets.size(), static_cast<size_t>(config.batchSize) + 1U);
    EXPECT_EQ(inputLengths.size(), config.batchSize);

    CapturableCtcBuffers buffers{
        makeGpuVector<float>(activations, stream),
        makeGpuVector<int32_t>(labels, stream),
        makeGpuVector<uint64_t>(offsets, stream),
        makeGpuVector<int32_t>(inputLengths, stream),
        Tensor(gpuPlacement, TensorDescriptor(DataType::INT32, {config.batchSize})),
        Tensor(gpuPlacement, TensorDescriptor(DataType::UINT32, {1})),
        Tensor(gpuPlacement, TensorDescriptor(DataType::FP32, {config.batchSize})),
        Tensor(gpuPlacement,
               TensorDescriptor(DataType::FP32,
                                {config.batchSize, config.maxTimeSteps, config.numClasses})),
        nullptr,
    };
    if (plan.getWorkspaceSizeInBytes() > 0) {
        buffers.workspace = std::make_unique<Tensor>(
            gpuPlacement, TensorDescriptor(DataType::UINT8, {plan.getWorkspaceSizeInBytes()}));
    }
    return buffers;
}

void runCapturableCtcPipeline(const CudnnCtcLossPlan& plan,
                              CapturableCtcBuffers& buffers,
                              uint64_t maxTotalLabelValues,
                              float lossScale,
                              float gradientScale,
                              Stream& stream) {
    const CudnnCtcLossConfig& config = plan.getConfig();
    rowPartitionOffsetsToInt32LengthsChecked(buffers.offsets,
                                             buffers.generatedLabelLengths,
                                             buffers.validationErrorBits,
                                             config.batchSize,
                                             maxTotalLabelValues,
                                             config.maxLabelLength,
                                             stream);

    plan.run(buffers.activations.getMemPtr(),
             buffers.labels.getMemPtr<int>(),
             buffers.generatedLabelLengths.getMemPtr<int>(),
             buffers.inputLengths.getMemPtr<int>(),
             buffers.costs.getMemPtr(),
             buffers.gradients.getMemPtr(),
             buffers.workspace ? buffers.workspace->getMemPtr() : nullptr,
             plan.getWorkspaceSizeInBytes(),
             stream);

    launchCorrectCtcEmptyTargetRows(buffers.activations.getMemPtr<float>(),
                                    buffers.generatedLabelLengths.getMemPtr<int>(),
                                    buffers.inputLengths.getMemPtr<int>(),
                                    buffers.costs.getMemPtr<float>(),
                                    buffers.gradients.getMemPtr<float>(),
                                    config.batchSize,
                                    config.maxTimeSteps,
                                    config.numClasses,
                                    stream);

    launchScaleCtcLossOutputs(buffers.costs.getMemPtr<float>(),
                              buffers.gradients.getMemPtr<float>(),
                              buffers.inputLengths.getMemPtr<int>(),
                              config.batchSize,
                              config.maxTimeSteps,
                              config.numClasses,
                              buffers.costs.getTotalNumElements(),
                              true,
                              lossScale,
                              gradientScale,
                              stream);
}

void expectFloatVectorsNear(const std::vector<float>& actual,
                            const std::vector<float>& expected,
                            float tolerance = 1.0e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "mismatch at index " << i;
    }
}

}  // namespace

TEST(CudnnCtcLossPlan, AcceptsNarrowDeterministicFp32Config) {
    CudnnCtcLossPlan::validateConfig(validConfig());
}

TEST(CudnnCtcLossPlan, RejectsNonFp32DataType) {
    CudnnCtcLossConfig config = validConfig();
    config.dataType = DataType::FP16;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);
}

TEST(CudnnCtcLossPlan, RejectsDegenerateShape) {
    CudnnCtcLossConfig config = validConfig();
    config.maxTimeSteps = 0;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);

    config = validConfig();
    config.batchSize = 0;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);

    config = validConfig();
    config.numClasses = 1;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);
}

TEST(CudnnCtcLossPlan, RejectsDeterministicMaxLabelLengthAtCudnnLimit) {
    CudnnCtcLossConfig config = validConfig();
    config.maxLabelLength = 256;
    EXPECT_THROW(CudnnCtcLossPlan::validateConfig(config), std::logic_error);
}

TEST(CudnnCtcLossPlan, CapturesCanonicalRaggedPipelineAndReplaysWithChangedRuntimeInputs) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    CudnnCtcLossConfig config;
    config.maxTimeSteps = 4;
    config.batchSize = 2;
    config.numClasses = 3;
    config.maxLabelLength = 4;
    config.dataType = DataType::FP32;
    config.algorithm = CtcLossAlgorithm::DETERMINISTIC;
    config.normalization = CtcLossNormalization::SOFTMAX;
    config.oobGradientMode = CtcLossOobGradientMode::ZERO;
    CudnnCtcLossPlan plan(config, stream);

    constexpr uint64_t maxTotalLabelValues = 4;
    constexpr float lossScale = 0.75f;
    constexpr float gradientScale = 1.25f;

    const std::vector<float> activationsA = {
        0.60f, 0.30f, 0.10f, 0.20f, 0.70f, 0.10f, 0.25f, 0.65f, 0.10f, 0.70f, 0.20f, 0.10f,
        0.50f, 0.20f, 0.30f, 0.20f, 0.20f, 0.60f, 0.30f, 0.40f, 0.30f, 0.60f, 0.30f, 0.10f,
    };
    const std::vector<int32_t> labelsA = {1, 2, 1, 99};
    const std::vector<uint64_t> offsetsA = {0, 1, 3};
    const std::vector<int32_t> inputLengthsA = {4, 4};

    // Replay changes every runtime input and includes an active empty target row,
    // which proves the captured graph also reuses the row-partition conversion and
    // Thor's post-cuDNN empty-target correction rather than just replaying cuDNN.
    const std::vector<float> activationsB = {
        0.15f, 0.75f, 0.10f, 0.65f, 0.20f, 0.15f, 0.40f, 0.35f, 0.25f, 0.55f, 0.15f, 0.30f,
        0.30f, 0.25f, 0.45f, 0.10f, 0.15f, 0.75f, 0.25f, 0.50f, 0.25f, 0.45f, 0.40f, 0.15f,
    };
    const std::vector<int32_t> labelsB = {2, 77, 78, 79};
    const std::vector<uint64_t> offsetsB = {0, 0, 1};
    const std::vector<int32_t> inputLengthsB = {3, 4};

    CapturableCtcBuffers captured =
        makeCapturableCtcBuffers(plan, activationsA, labelsA, offsetsA, inputLengthsA, stream);
    CapturableCtcBuffers reference =
        makeCapturableCtcBuffers(plan, activationsA, labelsA, offsetsA, inputLengthsA, stream);

    runCapturableCtcPipeline(plan, reference, maxTotalLabelValues, lossScale, gradientScale, stream);
    stream.synchronize();
    const std::vector<float> expectedCostsA = copyGpuVector<float>(reference.costs, stream);
    const std::vector<float> expectedGradientsA = copyGpuVector<float>(reference.gradients, stream);

    CudaGraphCaptureBuilder builder(stream);
    runCapturableCtcPipeline(plan, captured, maxTotalLabelValues, lossScale, gradientScale, stream);
    CudaGraphExecutable executable = builder.endCaptureAndInstantiate(stream);

    executable.launch(stream);
    stream.synchronize();
    expectFloatVectorsNear(copyGpuVector<float>(captured.costs, stream), expectedCostsA);
    expectFloatVectorsNear(copyGpuVector<float>(captured.gradients, stream), expectedGradientsA);
    EXPECT_EQ(copyGpuVector<int32_t>(captured.generatedLabelLengths, stream),
              (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(copyGpuVector<uint32_t>(captured.validationErrorBits, stream),
              (std::vector<uint32_t>{0U}));

    replaceGpuVector(captured.activations, activationsB, stream);
    replaceGpuVector(captured.labels, labelsB, stream);
    replaceGpuVector(captured.offsets, offsetsB, stream);
    replaceGpuVector(captured.inputLengths, inputLengthsB, stream);
    replaceGpuVector(reference.activations, activationsB, stream);
    replaceGpuVector(reference.labels, labelsB, stream);
    replaceGpuVector(reference.offsets, offsetsB, stream);
    replaceGpuVector(reference.inputLengths, inputLengthsB, stream);

    runCapturableCtcPipeline(plan, reference, maxTotalLabelValues, lossScale, gradientScale, stream);
    stream.synchronize();
    const std::vector<float> expectedCostsB = copyGpuVector<float>(reference.costs, stream);
    const std::vector<float> expectedGradientsB = copyGpuVector<float>(reference.gradients, stream);

    executable.launch(stream);
    stream.synchronize();
    expectFloatVectorsNear(copyGpuVector<float>(captured.costs, stream), expectedCostsB);
    expectFloatVectorsNear(copyGpuVector<float>(captured.gradients, stream), expectedGradientsB);
    EXPECT_EQ(copyGpuVector<int32_t>(captured.generatedLabelLengths, stream),
              (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(copyGpuVector<uint32_t>(captured.validationErrorBits, stream),
              (std::vector<uint32_t>{0U}));
}
