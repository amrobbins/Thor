#include "Utilities/TensorOperations/DeepLearning/ExpressionDenseSoftmax.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

int cudaDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess ? count : 0;
}

ExpressionDenseSoftmaxDescriptor descriptor(ExpressionDenseSoftmaxKind kind,
                                             uint64_t outer,
                                             uint64_t channels,
                                             DataType inputDtype,
                                             optional<DataType> outputDtype = nullopt) {
    ExpressionDenseSoftmaxDescriptor d;
    d.outerSize = outer;
    d.channelCount = channels;
    d.inputDataType = inputDtype;
    d.outputDataType = outputDtype;
    d.computeDataType = DataType::FP32;
    d.kind = kind;
    d.debugName = "d5_1_expression_dense_softmax";
    return d;
}

Tensor makeGpuFromFp32(const vector<float>& values,
                       const vector<uint64_t>& dims,
                       DataType dtype,
                       Stream& stream) {
    const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, stream.getGpuNum());

    Tensor host(cpu, TensorDescriptor(DataType::FP32, dims));
    float* hostValues = host.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) hostValues[i] = values[i];

    Tensor fp32Gpu(gpu, TensorDescriptor(DataType::FP32, dims));
    fp32Gpu.copyFromAsync(host, stream);
    if (dtype == DataType::FP32) {
        stream.synchronize();
        return fp32Gpu;
    }

    Tensor converted(gpu, TensorDescriptor(dtype, dims));
    TypeConverter::convertType(fp32Gpu.getMemPtr<void>(),
                               converted.getMemPtr<void>(),
                               DataType::FP32,
                               dtype,
                               static_cast<long>(values.size()),
                               stream,
                               stream.getGpuNum());
    stream.synchronize();
    return converted;
}

vector<float> copyAsFp32(Tensor gpu, Stream& stream) {
    const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    Tensor fp32Gpu(gpuPlacement, TensorDescriptor(DataType::FP32, gpu.getDimensions()));
    if (gpu.getDataType() == DataType::FP32) {
        fp32Gpu.copyFromAsync(gpu, stream);
    } else {
        TypeConverter::convertType(gpu.getMemPtr<void>(),
                                   fp32Gpu.getMemPtr<void>(),
                                   gpu.getDataType(),
                                   DataType::FP32,
                                   static_cast<long>(gpu.getTotalNumElements()),
                                   stream,
                                   stream.getGpuNum());
    }

    Tensor host(cpu, TensorDescriptor(DataType::FP32, gpu.getDimensions()));
    host.copyFromAsync(fp32Gpu, stream);
    stream.synchronize();
    const float* values = host.getMemPtr<float>();
    return vector<float>(values, values + gpu.getTotalNumElements());
}

vector<float> forwardReference(const vector<float>& x, uint64_t outer, uint64_t channels, bool logSoftmax) {
    vector<float> result(x.size(), 0.0f);
    for (uint64_t row = 0; row < outer; ++row) {
        const uint64_t base = row * channels;
        float maximum = x[base];
        for (uint64_t c = 1; c < channels; ++c) maximum = max(maximum, x[base + c]);
        float sum = 0.0f;
        for (uint64_t c = 0; c < channels; ++c) sum += exp(x[base + c] - maximum);
        const float logSum = log(sum);
        for (uint64_t c = 0; c < channels; ++c) {
            const float centered = x[base + c] - maximum;
            result[base + c] = logSoftmax ? centered - logSum : exp(centered) / sum;
        }
    }
    return result;
}

vector<float> backwardReference(const vector<float>& y,
                                const vector<float>& dy,
                                uint64_t outer,
                                uint64_t channels,
                                bool logSoftmax) {
    vector<float> result(y.size(), 0.0f);
    for (uint64_t row = 0; row < outer; ++row) {
        const uint64_t base = row * channels;
        if (logSoftmax) {
            float dySum = 0.0f;
            for (uint64_t c = 0; c < channels; ++c) dySum += dy[base + c];
            for (uint64_t c = 0; c < channels; ++c) result[base + c] = dy[base + c] - exp(y[base + c]) * dySum;
        } else {
            float dot = 0.0f;
            for (uint64_t c = 0; c < channels; ++c) dot += y[base + c] * dy[base + c];
            for (uint64_t c = 0; c < channels; ++c) result[base + c] = y[base + c] * (dy[base + c] - dot);
        }
    }
    return result;
}

float toleranceFor(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return 5.0e-5f;
        case DataType::FP16:
            return 3.0e-3f;
        case DataType::BF16:
            return 2.0e-2f;
        case DataType::FP8_E4M3:
            return 6.0e-2f;
        case DataType::FP8_E5M2:
            return 1.2e-1f;
        default:
            return 0.0f;
    }
}

void expectNearVector(const vector<float>& actual, const vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(isfinite(actual[i])) << "index " << i;
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
    }
}

vector<float> inputValues(uint64_t outer, uint64_t channels) {
    vector<float> values;
    values.reserve(outer * channels);
    for (uint64_t row = 0; row < outer; ++row) {
        for (uint64_t c = 0; c < channels; ++c) {
            values.push_back(static_cast<float>(static_cast<int>(c) - 3) * 0.625f + static_cast<float>(row) * 0.17f);
        }
    }
    return values;
}

vector<float> gradientValues(uint64_t outer, uint64_t channels) {
    vector<float> values;
    values.reserve(outer * channels);
    for (uint64_t row = 0; row < outer; ++row) {
        for (uint64_t c = 0; c < channels; ++c) {
            values.push_back(static_cast<float>((static_cast<int>(row + 2 * c) % 7) - 3) * 0.2f);
        }
    }
    return values;
}

bool containsOp(const PhysicalOutputs& outputs, ExprOp op) {
    if (!outputs.expr) return false;
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op == op) return true;
    }
    return false;
}

size_t countOp(const PhysicalOutputs& outputs, ExprOp op) {
    if (!outputs.expr) return 0;
    size_t count = 0;
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op == op) ++count;
    }
    return count;
}

bool isSoftmaxPointwiseOp(ExprOp op) {
    return op == ExprOp::SUB || op == ExprOp::MUL || op == ExprOp::DIV || op == ExprOp::EXP || op == ExprOp::LN;
}

void expectPointwiseComputeIsFp32(const PhysicalOutputs& outputs) {
    ASSERT_NE(outputs.expr, nullptr);
    for (const ExprNode& node : outputs.expr->nodes) {
        if (!isSoftmaxPointwiseOp(node.op)) continue;
        ASSERT_TRUE(node.compute_dtype.has_value());
        EXPECT_EQ(node.compute_dtype.value(), DataType::FP32);
    }
}

void runForwardCase(DataType inputDtype,
                    optional<DataType> outputDtype,
                    ExpressionDenseSoftmaxKind kind,
                    uint64_t outer = 5,
                    uint64_t channels = 7) {
    Stream stream(0);
    const ExpressionDenseSoftmaxDescriptor d = descriptor(kind, outer, channels, inputDtype, outputDtype);
    const DataType resolvedOutput = d.resolvedOutputDataType();
    ExpressionDenseSoftmax& softmax = ExpressionDenseSoftmax::instance();
    ExpressionDenseSoftmaxPlan plan = softmax.prepareForward(d, 0);

    const vector<uint64_t> dims{outer, channels};
    Tensor x = makeGpuFromFp32(inputValues(outer, channels), dims, inputDtype, stream);
    const vector<float> quantizedInput = copyAsFp32(x, stream);
    Tensor y(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(resolvedOutput, dims));
    softmax.forward(plan, {.x = x, .y = y}, stream);
    const vector<float> actual = copyAsFp32(y, stream);
    const vector<float> expected =
        forwardReference(quantizedInput, outer, channels, kind == ExpressionDenseSoftmaxKind::LogSoftmax);
    expectNearVector(actual, expected, toleranceFor(resolvedOutput));
}

void runForwardBackwardCase(DataType inputDtype,
                            DataType outputDtype,
                            ExpressionDenseSoftmaxKind kind,
                            uint64_t outer = 4,
                            uint64_t channels = 8) {
    Stream stream(0);
    const ExpressionDenseSoftmaxDescriptor d = descriptor(kind, outer, channels, inputDtype, outputDtype);
    ExpressionDenseSoftmax& softmax = ExpressionDenseSoftmax::instance();
    ExpressionDenseSoftmaxPlan forwardPlan = softmax.prepareForward(d, 0);
    ExpressionDenseSoftmaxPlan backwardPlan = softmax.prepareBackward(d, 0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    const vector<uint64_t> dims{outer, channels};

    Tensor x = makeGpuFromFp32(inputValues(outer, channels), dims, inputDtype, stream);
    Tensor y(gpu, TensorDescriptor(outputDtype, dims));
    softmax.forward(forwardPlan, {.x = x, .y = y}, stream);
    const vector<float> materializedY = copyAsFp32(y, stream);

    Tensor dy = makeGpuFromFp32(gradientValues(outer, channels), dims, outputDtype, stream);
    const vector<float> materializedDy = copyAsFp32(dy, stream);
    Tensor dx(gpu, TensorDescriptor(inputDtype, dims));
    softmax.backward(backwardPlan, {.y = y, .dy = dy, .dx = dx}, stream);
    const vector<float> actualDx = copyAsFp32(dx, stream);
    const vector<float> expectedDx =
        backwardReference(materializedY, materializedDy, outer, channels, kind == ExpressionDenseSoftmaxKind::LogSoftmax);
    expectNearVector(actualDx, expectedDx, toleranceFor(inputDtype));
}

}  // namespace

TEST(ExpressionDenseSoftmaxDescriptor, OutputDefaultsToInputAndExplicitOutputIsIndependent) {
    for (DataType input : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        ExpressionDenseSoftmaxDescriptor d = descriptor(ExpressionDenseSoftmaxKind::Softmax, 8, 16, input);
        EXPECT_EQ(d.resolvedOutputDataType(), input);
        EXPECT_NO_THROW(d.validateForward());
        EXPECT_NO_THROW(d.validateBackward());

        for (DataType output : {DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16, DataType::FP32}) {
            d.outputDataType = output;
            EXPECT_EQ(d.resolvedOutputDataType(), output);
            EXPECT_NO_THROW(d.validateForward());
        }
    }
}

TEST(ExpressionDenseSoftmaxDescriptor, RequiresFinalAxisShapeFloatingIoAndFp32Compute) {
    ExpressionDenseSoftmaxDescriptor d = descriptor(ExpressionDenseSoftmaxKind::Softmax, 0, 16, DataType::FP16);
    EXPECT_THROW(d.validateForward(), invalid_argument);

    d = descriptor(ExpressionDenseSoftmaxKind::Softmax, 8, 0, DataType::FP16);
    EXPECT_THROW(d.validateForward(), invalid_argument);

    d = descriptor(ExpressionDenseSoftmaxKind::Softmax, 8, 16, DataType::INT32);
    EXPECT_THROW(d.validateForward(), invalid_argument);

    d = descriptor(ExpressionDenseSoftmaxKind::Softmax, 8, 16, DataType::FP8_E4M3);
    EXPECT_THROW(d.validateForward(), invalid_argument);
    d = descriptor(ExpressionDenseSoftmaxKind::LogSoftmax, 8, 16, DataType::FP8_E5M2, DataType::BF16);
    EXPECT_THROW(d.validateForward(), invalid_argument);

    d = descriptor(ExpressionDenseSoftmaxKind::Softmax, 8, 16, DataType::FP16, DataType::UINT32);
    EXPECT_THROW(d.validateForward(), invalid_argument);

    d = descriptor(ExpressionDenseSoftmaxKind::Softmax, 8, 16, DataType::FP16);
    d.computeDataType = DataType::BF16;
    EXPECT_THROW(d.validateForward(), invalid_argument);
}

TEST(ExpressionDenseSoftmaxD51, ForwardGraphUsesExpressionReductionsAndNeverLegacySoftmax) {
    ExpressionDenseSoftmax& softmax = ExpressionDenseSoftmax::instance();

    auto verifyForwardPlan = [](const ExpressionDenseSoftmaxPlan& plan, bool logSoftmax) {
        ASSERT_EQ(plan.equationCount(), 4U);
        size_t reduceMaxCount = 0;
        size_t reduceSumCount = 0;
        bool sawExp = false;
        bool sawDiv = false;
        bool sawLn = false;
        for (size_t i = 0; i < plan.equationCount(); ++i) {
            const PhysicalOutputs& graph = plan.equation(i).physicalOutputs();
            reduceMaxCount += countOp(graph, ExprOp::REDUCE_MAX);
            reduceSumCount += countOp(graph, ExprOp::REDUCE_SUM);
            sawExp = sawExp || containsOp(graph, ExprOp::EXP);
            sawDiv = sawDiv || containsOp(graph, ExprOp::DIV);
            sawLn = sawLn || containsOp(graph, ExprOp::LN);
            EXPECT_FALSE(containsOp(graph, ExprOp::SOFTMAX));
            expectPointwiseComputeIsFp32(graph);
        }
        EXPECT_EQ(reduceMaxCount, 1U);
        EXPECT_EQ(reduceSumCount, 1U);
        EXPECT_TRUE(sawExp);
        if (logSoftmax) {
            EXPECT_TRUE(sawLn);
        } else {
            EXPECT_TRUE(sawDiv);
        }
    };

    ExpressionDenseSoftmaxPlan softmaxPlan =
        softmax.prepareForward(descriptor(ExpressionDenseSoftmaxKind::Softmax, 8, 16, DataType::BF16, DataType::BF16), 0);
    verifyForwardPlan(softmaxPlan, false);

    ExpressionDenseSoftmaxPlan logPlan =
        softmax.prepareForward(descriptor(ExpressionDenseSoftmaxKind::LogSoftmax, 8, 16, DataType::FP32, DataType::BF16), 0);
    verifyForwardPlan(logPlan, true);
}

TEST(ExpressionDenseSoftmaxD51, ReductionsAndMaterializedIntermediatesResolveToFp32) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    const uint64_t outer = 3;
    const uint64_t channels = 5;
    const vector<uint64_t> dims{outer, channels};
    ExpressionDenseSoftmax& softmax = ExpressionDenseSoftmax::instance();
    ExpressionDenseSoftmaxPlan plan =
        softmax.prepareForward(descriptor(ExpressionDenseSoftmaxKind::Softmax, outer, channels, DataType::BF16), 0);
    Tensor x = makeGpuFromFp32(inputValues(outer, channels), dims, DataType::BF16, stream);

    auto expectSingleFp32Reduction = [](const shared_ptr<CompiledOutputs>& compiled) {
        size_t reductionCount = 0;
        for (const CompiledExecutionStage& stage : compiled->stages) {
            if (stage.kind != CompiledExecutionStage::Kind::Reduction) continue;
            ++reductionCount;
            ASSERT_NE(stage.reduction, nullptr);
            EXPECT_EQ(stage.reduction->compute_dtype, DataType::FP32);
            EXPECT_EQ(stage.reduction->output_dtype, DataType::FP32);
        }
        EXPECT_EQ(reductionCount, 1U);
    };

    const shared_ptr<CompiledOutputs> maxCompiled = plan.equation(0).compileForInputs({{"x", x}});
    expectSingleFp32Reduction(maxCompiled);

    Tensor expCentered(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    const shared_ptr<CompiledOutputs> sumCompiled =
        plan.equation(2).compileForInputs({{"exp_centered", expCentered}});
    expectSingleFp32Reduction(sumCompiled);
}

TEST(ExpressionDenseSoftmaxD51, DefaultOutputStorageMatchesInputAcrossQualifiedDtypes) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        SCOPED_TRACE(TensorDescriptor::getElementTypeName(dtype));
        runForwardCase(dtype, nullopt, ExpressionDenseSoftmaxKind::Softmax);
    }
}

TEST(ExpressionDenseSoftmaxD51, ExplicitOutputStorageIsIndependentOfQualifiedInput) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    runForwardCase(DataType::FP16, DataType::BF16, ExpressionDenseSoftmaxKind::Softmax);
    runForwardCase(DataType::BF16, DataType::FP32, ExpressionDenseSoftmaxKind::Softmax);
    runForwardCase(DataType::BF16, DataType::FP8_E4M3, ExpressionDenseSoftmaxKind::Softmax);
}

TEST(ExpressionDenseSoftmaxD51, QualifiedInputSelectedOutputSoftmaxForwardBackwardIsNumerical) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    runForwardBackwardCase(DataType::FP16, DataType::BF16, ExpressionDenseSoftmaxKind::Softmax);
    runForwardBackwardCase(DataType::BF16, DataType::FP32, ExpressionDenseSoftmaxKind::Softmax);
}

TEST(ExpressionDenseSoftmaxD51, StableLogSoftmaxForwardBackwardSupportsQualifiedInputAndSelectedOutput) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    runForwardCase(DataType::FP16, DataType::BF16, ExpressionDenseSoftmaxKind::LogSoftmax);
    runForwardBackwardCase(DataType::BF16, DataType::FP32, ExpressionDenseSoftmaxKind::LogSoftmax);
}

TEST(ExpressionDenseSoftmaxD51, Bf16LogSoftmaxUsesFp32RangeBeforeSelectedBf16Output) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    const ExpressionDenseSoftmaxDescriptor d =
        descriptor(ExpressionDenseSoftmaxKind::LogSoftmax, 1, 2, DataType::BF16, DataType::BF16);
    ExpressionDenseSoftmax& softmax = ExpressionDenseSoftmax::instance();
    ExpressionDenseSoftmaxPlan plan = softmax.prepareForward(d, 0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);

    Tensor x = makeGpuFromFp32({57344.0f, -57344.0f}, {1, 2}, DataType::BF16, stream);
    Tensor y(gpu, TensorDescriptor(DataType::BF16, {1, 2}));
    softmax.forward(plan, {.x = x, .y = y}, stream);

    const vector<float> actual = copyAsFp32(y, stream);
    ASSERT_EQ(actual.size(), 2U);
    EXPECT_TRUE(isfinite(actual[0]));
    EXPECT_TRUE(isfinite(actual[1]));
    EXPECT_NEAR(actual[0], 0.0f, 1.0e-3f);
    EXPECT_NEAR(actual[1], -114688.0f, 256.0f);
}

TEST(ExpressionDenseSoftmaxD51, PreparedPlanIsReusableAcrossRuns) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    const uint64_t outer = 6;
    const uint64_t channels = 9;
    const vector<uint64_t> dims{outer, channels};
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    const ExpressionDenseSoftmaxDescriptor d =
        descriptor(ExpressionDenseSoftmaxKind::Softmax, outer, channels, DataType::BF16, DataType::BF16);
    ExpressionDenseSoftmax& softmax = ExpressionDenseSoftmax::instance();
    ExpressionDenseSoftmaxPlan plan = softmax.prepareForward(d, 0);

    Tensor xA = makeGpuFromFp32(inputValues(outer, channels), dims, DataType::BF16, stream);
    vector<float> secondInput = inputValues(outer, channels);
    for (float& value : secondInput) value *= -0.7f;
    Tensor xB = makeGpuFromFp32(secondInput, dims, DataType::BF16, stream);
    Tensor yA(gpu, TensorDescriptor(DataType::BF16, dims));
    Tensor yB(gpu, TensorDescriptor(DataType::BF16, dims));

    softmax.forward(plan, {.x = xA, .y = yA}, stream);
    softmax.forward(plan, {.x = xB, .y = yB}, stream);
    const vector<float> a = copyAsFp32(yA, stream);
    const vector<float> b = copyAsFp32(yB, stream);
    EXPECT_NE(a, b);
}
