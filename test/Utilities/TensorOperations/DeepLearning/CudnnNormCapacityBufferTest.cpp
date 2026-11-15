#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

// A capacity bucket runs a plan for the leading rows of full-capacity tensors directly, without a
// prefix view: the plan reads and writes only its rows, and rows past them are left untouched.

namespace {

bool haveCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

constexpr uint64_t PLAN_ROWS = 3;
constexpr uint64_t CAPACITY_ROWS = 5;
constexpr uint64_t HIDDEN = 16;
constexpr float SENTINEL = 7.0f;
constexpr float EPSILON = 1.0e-5f;

Tensor gpuTensor(const std::vector<float>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, {values.size()}));
    float* data = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
    Tensor gpu(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), cpu.getDescriptor());
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();  // cpu is freed on return
    return gpu;
}

std::vector<float> hostValues(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const float* data = cpu.getMemPtr<float>();
    return std::vector<float>(data, data + gpu.getTotalNumElements());
}

std::vector<float> rowValues(uint64_t rows) {
    std::vector<float> values(rows * HIDDEN);
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = 0.25f * static_cast<float>(static_cast<int64_t>((i * 7) % 13) - 6);
    }
    return values;
}

std::vector<float> parameterValues(float base) {
    std::vector<float> values(HIDDEN);
    for (uint64_t i = 0; i < HIDDEN; ++i) {
        values[i] = base + 0.125f * static_cast<float>(i % 5);
    }
    return values;
}

template <typename Call>
std::string invalidArgumentMessage(Call call) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return "";
}

void expectRowsPastThePlanUntouched(const std::vector<float>& values, uint64_t rowWidth, const char* name) {
    for (uint64_t i = PLAN_ROWS * rowWidth; i < values.size(); ++i) {
        EXPECT_EQ(values[i], SENTINEL) << name << " element " << i << " is past the plan's rows";
    }
}

}  // namespace

TEST(CudnnNormCapacityBuffer, LayerNormRunsLeadingRowsOfCapacityTensors) {
    if (!haveCudaDevice())
        GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    CudnnLayerNormDescriptor descriptor;
    descriptor.outerSize = PLAN_ROWS;
    descriptor.normalizedFeatureCount = HIDDEN;
    descriptor.inputDataType = DataType::FP32;
    descriptor.outputDataType = DataType::FP32;
    descriptor.parameterDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = EPSILON;
    descriptor.training = true;
    descriptor.debugName = "layer_norm_capacity_buffer";

    CudnnLayerNorm& layerNorm = CudnnLayerNorm::instance();
    CudnnLayerNormExecutablePlan plan = layerNorm.prepareForward(descriptor, stream);
    std::optional<Tensor> workspace;
    if (plan.workspaceBytes() > 0) {
        workspace = Tensor(
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::UINT8, {plan.workspaceBytes()}), 256);
    }

    const std::vector<float> x = rowValues(CAPACITY_ROWS);
    const std::vector<float> scale = parameterValues(0.75f);
    const std::vector<float> bias = parameterValues(-0.5f);
    CudnnLayerNormForwardArgs args;
    args.x = gpuTensor(x, stream);
    args.scale = gpuTensor(scale, stream);
    args.bias = gpuTensor(bias, stream);
    args.y = gpuTensor(std::vector<float>(CAPACITY_ROWS * HIDDEN, SENTINEL), stream);
    args.mean = gpuTensor(std::vector<float>(CAPACITY_ROWS, SENTINEL), stream);
    args.invVariance = gpuTensor(std::vector<float>(CAPACITY_ROWS, SENTINEL), stream);
    layerNorm.forward(plan, args, workspace, stream);

    const std::vector<float> y = hostValues(args.y, stream);
    const std::vector<float> mean = hostValues(args.mean.value(), stream);
    const std::vector<float> invVariance = hostValues(args.invVariance.value(), stream);
    for (uint64_t row = 0; row < PLAN_ROWS; ++row) {
        double sum = 0.0;
        for (uint64_t col = 0; col < HIDDEN; ++col) sum += x[row * HIDDEN + col];
        const double rowMean = sum / HIDDEN;
        double squares = 0.0;
        for (uint64_t col = 0; col < HIDDEN; ++col) squares += (x[row * HIDDEN + col] - rowMean) * (x[row * HIDDEN + col] - rowMean);
        const double rowInvStd = 1.0 / std::sqrt(squares / HIDDEN + EPSILON);
        EXPECT_NEAR(mean[row], rowMean, 1.0e-5) << "row " << row;
        EXPECT_NEAR(invVariance[row], rowInvStd, 1.0e-3 * rowInvStd) << "row " << row;
        for (uint64_t col = 0; col < HIDDEN; ++col) {
            const double expected = (x[row * HIDDEN + col] - rowMean) * rowInvStd * scale[col] + bias[col];
            EXPECT_NEAR(y[row * HIDDEN + col], expected, 1.0e-4) << "row " << row << " col " << col;
        }
    }
    expectRowsPastThePlanUntouched(y, HIDDEN, "y");
    expectRowsPastThePlanUntouched(mean, 1, "mean");
    expectRowsPastThePlanUntouched(invVariance, 1, "invVariance");

    // A tensor with fewer rows than the plan is still rejected.
    CudnnLayerNormForwardArgs tooSmall = args;
    tooSmall.x = gpuTensor(rowValues(PLAN_ROWS - 1), stream);
    const std::string message = invalidArgumentMessage([&]() { layerNorm.forward(plan, tooSmall, workspace, stream); });
    EXPECT_NE(message.find("cuDNN LayerNorm tensor 'x' is too small for the plan"), std::string::npos) << message;
    stream.synchronize();
}

TEST(CudnnNormCapacityBuffer, RmsNormRunsLeadingRowsOfCapacityTensors) {
    if (!haveCudaDevice())
        GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = PLAN_ROWS;
    descriptor.normalizedFeatureCount = HIDDEN;
    descriptor.inputDataType = DataType::FP32;
    descriptor.outputDataType = DataType::FP32;
    descriptor.parameterDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = EPSILON;
    descriptor.training = true;
    descriptor.debugName = "rms_norm_capacity_buffer";

    CudnnRmsNorm& rmsNorm = CudnnRmsNorm::instance();
    CudnnRmsNormExecutablePlan plan = rmsNorm.prepareForward(descriptor, stream);
    std::optional<Tensor> workspace;
    if (plan.workspaceBytes() > 0) {
        workspace = Tensor(
            TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::UINT8, {plan.workspaceBytes()}), 256);
    }

    const std::vector<float> x = rowValues(CAPACITY_ROWS);
    const std::vector<float> scale = parameterValues(0.75f);
    CudnnRmsNormForwardArgs args;
    args.x = gpuTensor(x, stream);
    args.scale = gpuTensor(scale, stream);
    args.y = gpuTensor(std::vector<float>(CAPACITY_ROWS * HIDDEN, SENTINEL), stream);
    args.invVariance = gpuTensor(std::vector<float>(CAPACITY_ROWS, SENTINEL), stream);
    rmsNorm.forward(plan, args, workspace, stream);

    const std::vector<float> y = hostValues(args.y, stream);
    const std::vector<float> invRms = hostValues(args.invVariance.value(), stream);
    for (uint64_t row = 0; row < PLAN_ROWS; ++row) {
        double squares = 0.0;
        for (uint64_t col = 0; col < HIDDEN; ++col) squares += x[row * HIDDEN + col] * x[row * HIDDEN + col];
        const double rowInvRms = 1.0 / std::sqrt(squares / HIDDEN + EPSILON);
        EXPECT_NEAR(invRms[row], rowInvRms, 1.0e-3 * rowInvRms) << "row " << row;
        for (uint64_t col = 0; col < HIDDEN; ++col) {
            EXPECT_NEAR(y[row * HIDDEN + col], x[row * HIDDEN + col] * rowInvRms * scale[col], 1.0e-4) << "row " << row << " col " << col;
        }
    }
    expectRowsPastThePlanUntouched(y, HIDDEN, "y");
    expectRowsPastThePlanUntouched(invRms, 1, "invVariance");

    CudnnRmsNormForwardArgs tooSmall = args;
    tooSmall.y = gpuTensor(std::vector<float>((PLAN_ROWS - 1) * HIDDEN, SENTINEL), stream);
    const std::string message = invalidArgumentMessage([&]() { rmsNorm.forward(plan, tooSmall, workspace, stream); });
    EXPECT_NE(message.find("cuDNN RMSNorm tensor 'y' is too small for the plan"), std::string::npos) << message;
    stream.synchronize();
}

TEST(CudnnNormCapacityBuffer, StampedNormDescriptorsCoverTheRequestedLeadingRows) {
    CompiledLayerNorm layerNorm;
    layerNorm.normalized_feature_count = HIDDEN;
    layerNorm.input_dtype = DataType::FP32;
    layerNorm.scale_dtype = DataType::FP32;
    layerNorm.bias_dtype = DataType::FP32;
    layerNorm.output_dtype = DataType::FP32;

    const Tensor io(cpuPlacement, TensorDescriptor(DataType::FP32, {CAPACITY_ROWS, HIDDEN}));
    const Tensor parameter(cpuPlacement, TensorDescriptor(DataType::FP32, {HIDDEN}));
    EXPECT_EQ(layerNorm.descriptorFor(io, parameter, parameter, io).outerSize, CAPACITY_ROWS);
    EXPECT_EQ(layerNorm.descriptorFor(io, parameter, parameter, io, PLAN_ROWS).outerSize, PLAN_ROWS);
    EXPECT_THROW((void)layerNorm.descriptorFor(io, parameter, parameter, io, CAPACITY_ROWS + 1), std::runtime_error);
    EXPECT_THROW((void)layerNorm.descriptorFor(io, parameter, parameter, io, 0), std::runtime_error);

    CompiledRmsNorm rmsNorm;
    rmsNorm.normalized_feature_count = HIDDEN;
    rmsNorm.input_dtype = DataType::FP32;
    rmsNorm.scale_dtype = DataType::FP32;
    rmsNorm.output_dtype = DataType::FP32;
    EXPECT_EQ(rmsNorm.descriptorFor(io, parameter, io, PLAN_ROWS).outerSize, PLAN_ROWS);
    EXPECT_THROW((void)rmsNorm.descriptorFor(io, parameter, io, CAPACITY_ROWS + 1), std::runtime_error);

    CompiledRmsNormBackward rmsNormBackward;
    rmsNormBackward.normalized_feature_count = HIDDEN;
    rmsNormBackward.input_dtype = DataType::FP32;
    rmsNormBackward.scale_dtype = DataType::FP32;
    rmsNormBackward.dy_dtype = DataType::FP32;
    rmsNormBackward.dx_dtype = DataType::FP32;
    rmsNormBackward.dscale_dtype = DataType::FP32;
    EXPECT_EQ(rmsNormBackward.descriptorFor(io, parameter, io, io, parameter, PLAN_ROWS).outerSize, PLAN_ROWS);
}
