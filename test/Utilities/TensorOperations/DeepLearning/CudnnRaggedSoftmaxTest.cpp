#include "Utilities/TensorOperations/DeepLearning/CudnnRaggedSoftmax.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;
using namespace std;

namespace {

int cudaDeviceCount() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess ? count : 0;
}

uint64_t numel(const vector<uint64_t>& dims) {
    uint64_t result = 1;
    for (uint64_t dim : dims) result *= dim;
    return result;
}

Tensor makeGpuFp32(const vector<uint64_t>& dims, const vector<float>& values, Stream& stream) {
    if (numel(dims) != values.size()) {
        throw runtime_error("makeGpuFp32 value-count mismatch");
    }
    const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    Tensor host(cpu, TensorDescriptor(DataType::FP32, dims));
    float* ptr = host.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) ptr[i] = values[i];
    Tensor device(gpu, TensorDescriptor(DataType::FP32, dims));
    device.copyFromAsync(host, stream);
    return device;
}

vector<float> copyFp32ToHost(const Tensor& gpu, Stream& stream) {
    const TensorPlacement cpu(TensorPlacement::MemDevices::CPU);
    Tensor host(cpu, TensorDescriptor(DataType::FP32, gpu.getDimensions()));
    host.copyFromAsync(gpu, stream);
    stream.synchronize();
    vector<float> result(gpu.getTotalNumElements());
    const float* ptr = host.getMemPtr<float>();
    for (size_t i = 0; i < result.size(); ++i) result[i] = ptr[i];
    return result;
}

vector<float> softmaxReference(const vector<float>& input, uint64_t rows, uint64_t channels, bool logSoftmax) {
    vector<float> output(input.size(), 0.0f);
    for (uint64_t row = 0; row < rows; ++row) {
        const uint64_t base = row * channels;
        float maximum = input[base];
        for (uint64_t channel = 1; channel < channels; ++channel) maximum = max(maximum, input[base + channel]);
        float sum = 0.0f;
        for (uint64_t channel = 0; channel < channels; ++channel) sum += exp(input[base + channel] - maximum);
        const float logDenominator = maximum + log(sum);
        for (uint64_t channel = 0; channel < channels; ++channel) {
            output[base + channel] =
                logSoftmax ? input[base + channel] - logDenominator : exp(input[base + channel] - logDenominator);
        }
    }
    return output;
}

vector<float> softmaxBackwardReference(const vector<float>& y,
                                       const vector<float>& dy,
                                       uint64_t rows,
                                       uint64_t channels,
                                       bool logSoftmax) {
    vector<float> dx(y.size(), 0.0f);
    for (uint64_t row = 0; row < rows; ++row) {
        const uint64_t base = row * channels;
        if (logSoftmax) {
            float dySum = 0.0f;
            for (uint64_t channel = 0; channel < channels; ++channel) dySum += dy[base + channel];
            for (uint64_t channel = 0; channel < channels; ++channel) {
                dx[base + channel] = dy[base + channel] - exp(y[base + channel]) * dySum;
            }
        } else {
            float dot = 0.0f;
            for (uint64_t channel = 0; channel < channels; ++channel) dot += y[base + channel] * dy[base + channel];
            for (uint64_t channel = 0; channel < channels; ++channel) {
                dx[base + channel] = y[base + channel] * (dy[base + channel] - dot);
            }
        }
    }
    return dx;
}

void expectNearPrefixAndSentinel(const vector<float>& actual,
                                 const vector<float>& expectedPrefix,
                                 uint64_t activeElements,
                                 float sentinel,
                                 float tolerance = 2.0e-5f) {
    ASSERT_GE(static_cast<uint64_t>(actual.size()), activeElements);
    ASSERT_EQ(expectedPrefix.size(), activeElements);
    for (uint64_t i = 0; i < activeElements; ++i) {
        EXPECT_NEAR(actual[i], expectedPrefix[i], tolerance) << "active element " << i;
    }
    for (uint64_t i = activeElements; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i], sentinel) << "inactive element " << i << " was modified";
    }
}

CudnnRaggedSoftmaxDescriptor descriptor(CudnnRaggedSoftmaxKind kind,
                                        uint64_t capacity,
                                        vector<uint64_t> trailing,
                                        DataType dtype = DataType::FP32) {
    CudnnRaggedSoftmaxDescriptor d;
    d.maxTotalValues = capacity;
    d.trailingDimensions = std::move(trailing);
    d.dataType = dtype;
    d.kind = kind;
    d.debugName = "r11a1_fixed_function";
    return d;
}

}  // namespace

TEST(CudnnRaggedSoftmaxDescriptor, SupportsArbitraryTrailingRankAndQualifiedDtypes) {
    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        EXPECT_NO_THROW(descriptor(CudnnRaggedSoftmaxKind::Softmax, 17, {5}, dtype).validate());
        EXPECT_NO_THROW(descriptor(CudnnRaggedSoftmaxKind::LogSoftmax, 17, {2, 3, 5}, dtype).validate());
    }

    const CudnnRaggedSoftmaxDescriptor d = descriptor(CudnnRaggedSoftmaxKind::Softmax, 17, {2, 3, 5});
    EXPECT_EQ(d.outerPerValue(), 6U);
    EXPECT_EQ(d.channelCount(), 5U);
    EXPECT_EQ(d.capacityElementCount(), 17U * 2U * 3U * 5U);
}

TEST(CudnnRaggedSoftmaxDescriptor, RejectsMissingTrailingAxisUnsupportedDtypeAndCudnnDimensionOverflow) {
    EXPECT_THROW(descriptor(CudnnRaggedSoftmaxKind::Softmax, 8, {}).validate(), invalid_argument);
    EXPECT_THROW(descriptor(CudnnRaggedSoftmaxKind::Softmax, 8, {0, 4}).validate(), invalid_argument);
    EXPECT_THROW(descriptor(CudnnRaggedSoftmaxKind::Softmax, 8, {4}, DataType::FP64).validate(), invalid_argument);

    CudnnRaggedSoftmaxDescriptor tooLarge = descriptor(CudnnRaggedSoftmaxKind::Softmax, 2, {2, 3});
    tooLarge.maxTotalValues = static_cast<uint64_t>(numeric_limits<int>::max());
    EXPECT_THROW(tooLarge.validate(), invalid_argument);
}

TEST(CudnnRaggedSoftmaxR11A1, ExactPrefixSoftmaxForwardBackwardLeavesInactiveCapacityUntouched) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    constexpr uint64_t capacity = 4;
    const vector<uint64_t> dims{capacity, 2, 3};
    CudnnRaggedSoftmaxDescriptor d = descriptor(CudnnRaggedSoftmaxKind::Softmax, capacity, {2, 3});
    CudnnRaggedSoftmaxExecutionState state = CudnnRaggedSoftmax::instance().prepare(d, stream);

    vector<float> input{
        1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 0.0f, -2.0f, -1.0f, 1.0f, 0.0f,
    };
    input.resize(numel(dims), numeric_limits<float>::quiet_NaN());
    Tensor x = makeGpuFp32(dims, input, stream);
    Tensor y(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    constexpr float ySentinel = -77.0f;
    y.fill(ySentinel, stream);

    CudnnRaggedSoftmax::instance().forward(state, {.x = x, .y = y, .activeValueCount = 2}, stream);
    const vector<float> actualY = copyFp32ToHost(y, stream);
    const vector<float> activeInput(input.begin(), input.begin() + 12);
    const vector<float> expectedY = softmaxReference(activeInput, 4, 3, false);
    expectNearPrefixAndSentinel(actualY, expectedY, 12, ySentinel);

    const vector<float> activeDy{
        0.5f, -1.0f, 2.0f, 1.0f, 3.0f, -2.0f,
        -0.25f, 0.75f, 1.5f, 2.0f, -1.0f, 0.5f,
    };
    vector<float> dyValues = activeDy;
    dyValues.resize(numel(dims), numeric_limits<float>::quiet_NaN());
    Tensor dy = makeGpuFp32(dims, dyValues, stream);
    Tensor dx(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    constexpr float dxSentinel = -88.0f;
    dx.fill(dxSentinel, stream);

    CudnnRaggedSoftmax::instance().backward(state, {.y = y, .dy = dy, .dx = dx, .activeValueCount = 2}, stream);
    const vector<float> actualDx = copyFp32ToHost(dx, stream);
    const vector<float> expectedDx = softmaxBackwardReference(expectedY, activeDy, 4, 3, false);
    expectNearPrefixAndSentinel(actualDx, expectedDx, 12, dxSentinel, 5.0e-5f);
}

TEST(CudnnRaggedSoftmaxR11A1, LogSoftmaxHigherTrailingRankUsesFinalAxisOnly) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    constexpr uint64_t capacity = 3;
    const vector<uint64_t> dims{capacity, 2, 2, 3};
    CudnnRaggedSoftmaxDescriptor d = descriptor(CudnnRaggedSoftmaxKind::LogSoftmax, capacity, {2, 2, 3});
    CudnnRaggedSoftmaxExecutionState state = CudnnRaggedSoftmax::instance().prepare(d, stream);

    vector<float> values;
    values.reserve(numel(dims));
    for (uint64_t i = 0; i < 24; ++i) values.push_back(static_cast<float>((static_cast<int>(i) % 7) - 3) * 0.4f);
    values.resize(numel(dims), numeric_limits<float>::quiet_NaN());
    Tensor x = makeGpuFp32(dims, values, stream);
    Tensor y(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    y.fill(-51.0f, stream);

    CudnnRaggedSoftmax::instance().forward(state, {.x = x, .y = y, .activeValueCount = 2}, stream);
    const vector<float> actual = copyFp32ToHost(y, stream);
    const vector<float> active(values.begin(), values.begin() + 24);
    const vector<float> expected = softmaxReference(active, 8, 3, true);
    expectNearPrefixAndSentinel(actual, expected, 24, -51.0f, 5.0e-5f);

    vector<float> activeDy;
    activeDy.reserve(24);
    for (uint64_t i = 0; i < 24; ++i) activeDy.push_back(static_cast<float>((static_cast<int>(i) % 5) - 2) * 0.25f);
    vector<float> dyValues = activeDy;
    dyValues.resize(numel(dims), numeric_limits<float>::quiet_NaN());
    Tensor dy = makeGpuFp32(dims, dyValues, stream);
    Tensor dx(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    dx.fill(-61.0f, stream);
    CudnnRaggedSoftmax::instance().backward(state, {.y = y, .dy = dy, .dx = dx, .activeValueCount = 2}, stream);
    const vector<float> actualDx = copyFp32ToHost(dx, stream);
    const vector<float> expectedDx = softmaxBackwardReference(expected, activeDy, 8, 3, true);
    expectNearPrefixAndSentinel(actualDx, expectedDx, 24, -61.0f, 7.5e-5f);
}

TEST(CudnnRaggedSoftmaxR11A1, SamePreparedStateSupportsShortLongShortWithoutRuntimePlanConstruction) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    constexpr uint64_t capacity = 4;
    const vector<uint64_t> dims{capacity, 3};
    CudnnRaggedSoftmaxDescriptor d = descriptor(CudnnRaggedSoftmaxKind::Softmax, capacity, {3});
    CudnnRaggedSoftmaxExecutionState state = CudnnRaggedSoftmax::instance().prepare(d, stream);

    vector<float> values{
        1.0f, 2.0f, 3.0f,
        4.0f, 2.0f, 0.0f,
        -1.0f, -2.0f, -3.0f,
        1.0f, 1.0f, 1.0f,
    };
    Tensor x = makeGpuFp32(dims, values, stream);
    Tensor y(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));

    for (uint64_t active : {1U, 4U, 1U}) {
        y.fill(-42.0f, stream);
        CudnnRaggedSoftmax::instance().forward(state, {.x = x, .y = y, .activeValueCount = active}, stream);
        const vector<float> actual = copyFp32ToHost(y, stream);
        const uint64_t activeElements = active * 3;
        const vector<float> prefix(values.begin(), values.begin() + activeElements);
        const vector<float> expected = softmaxReference(prefix, active, 3, false);
        expectNearPrefixAndSentinel(actual, expected, activeElements, -42.0f);
    }
}

TEST(CudnnRaggedSoftmaxR11A1, AllEmptyIsTrueNoOpAndOversizedActiveCountIsRejected) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    constexpr uint64_t capacity = 3;
    const vector<uint64_t> dims{capacity, 4};
    CudnnRaggedSoftmaxDescriptor d = descriptor(CudnnRaggedSoftmaxKind::Softmax, capacity, {4});
    CudnnRaggedSoftmaxExecutionState state = CudnnRaggedSoftmax::instance().prepare(d, stream);

    Tensor x(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    Tensor y(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    Tensor dy(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    Tensor dx(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    x.fill(numeric_limits<float>::quiet_NaN(), stream);
    dy.fill(numeric_limits<float>::quiet_NaN(), stream);
    y.fill(-7.0f, stream);
    dx.fill(-9.0f, stream);

    CudnnRaggedSoftmax::instance().forward(state, {.x = x, .y = y, .activeValueCount = 0}, stream);
    CudnnRaggedSoftmax::instance().backward(state, {.y = y, .dy = dy, .dx = dx, .activeValueCount = 0}, stream);
    const vector<float> yValues = copyFp32ToHost(y, stream);
    const vector<float> dxValues = copyFp32ToHost(dx, stream);
    for (float value : yValues) EXPECT_EQ(value, -7.0f);
    for (float value : dxValues) EXPECT_EQ(value, -9.0f);

    EXPECT_THROW(CudnnRaggedSoftmax::instance().forward(state, {.x = x, .y = y, .activeValueCount = capacity + 1}, stream),
                 invalid_argument);
    EXPECT_THROW(CudnnRaggedSoftmax::instance().backward(
                     state, {.y = y, .dy = dy, .dx = dx, .activeValueCount = capacity + 1}, stream),
                 invalid_argument);
}

TEST(CudnnRaggedSoftmaxR11A1, Fp16Bf16AndFp32ForwardBackwardAreQualified) {
    if (cudaDeviceCount() < 1) GTEST_SKIP() << "CUDA device is required.";

    Stream stream(0);
    constexpr uint64_t capacity = 2;
    const vector<uint64_t> dims{capacity, 4};
    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        CudnnRaggedSoftmaxDescriptor d = descriptor(CudnnRaggedSoftmaxKind::Softmax, capacity, {4}, dtype);
        CudnnRaggedSoftmaxExecutionState state = CudnnRaggedSoftmax::instance().prepare(d, stream);
        const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
        Tensor x(gpu, TensorDescriptor(dtype, dims));
        Tensor y(gpu, TensorDescriptor(dtype, dims));
        Tensor dy(gpu, TensorDescriptor(dtype, dims));
        Tensor dx(gpu, TensorDescriptor(dtype, dims));
        x.fill(0.5, stream);
        y.fill(-3.0, stream);
        dy.fill(1.0, stream);
        dx.fill(-4.0, stream);

        EXPECT_NO_THROW(CudnnRaggedSoftmax::instance().forward(state, {.x = x, .y = y, .activeValueCount = capacity}, stream));
        EXPECT_NO_THROW(
            CudnnRaggedSoftmax::instance().backward(state, {.y = y, .dy = dy, .dx = dx, .activeValueCount = capacity}, stream));
        EXPECT_NO_THROW(stream.synchronize());
    }
}
