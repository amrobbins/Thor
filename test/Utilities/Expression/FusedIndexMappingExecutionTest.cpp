// GPU execution checks for index-aware fused kernels against host references:
//  - RoPE whose input subgraph has a broadcast operand (x + bias[D]) or a non-dense runtime view
//    (q sliced out of a fused qkv row): the partner element must come from the same input row.
//  - A strided view broadcast by its parent, and an unsqueezed operand next to a logical transpose.
//  - A batched FP8 transposed broadcast (each batch reads its own matrix).
// FP32 storage is used where the reference involves transcendental math, so the comparison only
// has to absorb device-vs-host libm differences.

#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                     \
    do {                                                                                          \
        int cuda_device_count_for_test = 0;                                                       \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test); \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {             \
            GTEST_SKIP() << "CUDA device is required for fused index-mapping tests.";            \
        }                                                                                         \
    } while (false)

TensorPlacement cpu_placement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpu_placement(TensorPlacement::MemDevices::GPU, 0);

Tensor toGpu(const std::vector<float>& values, const std::vector<uint64_t>& dims, DataType dtype, Stream& stream) {
    Tensor cpu(cpu_placement, TensorDescriptor(DataType::FP32, dims));
    float* p = cpu.getMemPtr<float>();
    for (size_t i = 0; i < values.size(); ++i) p[i] = values[i];
    Tensor fp32_gpu(gpu_placement, TensorDescriptor(DataType::FP32, dims));
    fp32_gpu.copyFromAsync(cpu, stream);
    if (dtype == DataType::FP32) {
        stream.synchronize();
        return fp32_gpu;
    }
    Tensor converted(gpu_placement, TensorDescriptor(dtype, dims));
    TypeConverter::convertType(fp32_gpu.getMemPtr<void>(),
                               converted.getMemPtr<void>(),
                               DataType::FP32,
                               dtype,
                               static_cast<long>(values.size()),
                               stream,
                               gpu_placement.getDeviceNum());
    stream.synchronize();
    return converted;
}

std::vector<float> toHost(const Tensor& tensor, Stream& stream) {
    Tensor fp32_gpu(gpu_placement, TensorDescriptor(DataType::FP32, tensor.getDimensions()));
    if (tensor.getDataType() == DataType::FP32) {
        fp32_gpu.copyFromAsync(tensor, stream);
    } else {
        TypeConverter::convertType(const_cast<Tensor&>(tensor).getMemPtr<void>(),
                                   fp32_gpu.getMemPtr<void>(),
                                   tensor.getDataType(),
                                   DataType::FP32,
                                   static_cast<long>(tensor.getTotalNumElements()),
                                   stream,
                                   gpu_placement.getDeviceNum());
    }
    Tensor cpu(cpu_placement, TensorDescriptor(DataType::FP32, tensor.getDimensions()));
    cpu.copyFromAsync(fp32_gpu, stream);
    stream.synchronize();
    const float* p = cpu.getMemPtr<float>();
    return std::vector<float>(p, p + tensor.getTotalNumElements());
}

std::vector<float> ramp(size_t count, float scale, int modulus) {
    std::vector<float> v(count);
    for (size_t i = 0; i < count; ++i) v[i] = scale * static_cast<float>(static_cast<int>((i * 37) % modulus) - modulus / 2);
    return v;
}

// Rotate-half or interleaved RoPE over the last axis, position = coordinate on axis `rank - 2`.
std::vector<float> ropeReference(const std::vector<float>& v, const std::vector<uint64_t>& dims, bool interleaved) {
    const uint64_t d = dims.back();
    const uint64_t s = dims[dims.size() - 2];
    const uint64_t half = d / 2;
    std::vector<float> out(v.size());
    for (size_t row = 0; row < v.size() / d; ++row) {
        const double position = static_cast<double>(row % s);
        for (uint64_t k = 0; k < d; ++k) {
            const uint64_t pair = interleaved ? k / 2 : (k < half ? k : k - half);
            const bool first = interleaved ? (k % 2 == 0) : (k < half);
            const uint64_t partner = interleaved ? (first ? k + 1 : k - 1) : (first ? k + half : k - half);
            const double theta = position * std::pow(10000.0, -2.0 * static_cast<double>(pair) / static_cast<double>(d));
            const double c = std::cos(theta), sn = std::sin(theta);
            const double cur = v[row * d + k], peer = v[row * d + partner];
            out[row * d + k] = static_cast<float>(first ? cur * c - peer * sn : peer * sn + cur * c);
        }
    }
    return out;
}

void expectNear(const std::vector<float>& got, const std::vector<float>& want, float rel, const std::string& label) {
    ASSERT_EQ(got.size(), want.size()) << label;
    for (size_t i = 0; i < got.size(); ++i) {
        ASSERT_NEAR(got[i], want[i], rel * (1.0f + std::fabs(want[i]))) << label << " element " << i;
    }
}

}  // namespace

TEST(FusedIndexMappingExecution, RopeOfBroadcastSumReadsPartnerFromTheSameRow) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    const std::vector<uint64_t> dims{2, 3, 8, 16};
    const size_t numel = 2 * 3 * 8 * 16;
    for (const bool interleaved : {false, true}) {
        const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
        const Expression bias = Expression::input("bias", DataType::FP32, DataType::FP32);
        RotaryPositionEmbeddingOptions options;
        options.interleaved = interleaved;
        FusedEquation eq = FusedEquation::compile(Expression::outputs({{"y", (x + bias).rope(options)}}).physicalOutputs(), 0);

        const std::vector<float> xv = ramp(numel, 0.125f, 23);
        const std::vector<float> bv = ramp(16, 0.25f, 7);
        Tensor xs = toGpu(xv, dims, DataType::FP32, stream);
        Tensor bs = toGpu(bv, {16}, DataType::FP32, stream);
        Tensor ys(gpu_placement, TensorDescriptor(DataType::FP32, dims));
        eq.run({{"x", xs}, {"bias", bs}}, ys, stream);

        std::vector<float> sum(numel);
        for (size_t i = 0; i < numel; ++i) sum[i] = xv[i] + bv[i % 16];
        expectNear(toHost(ys, stream), ropeReference(sum, dims, interleaved), 2e-5f, interleaved ? "interleaved" : "rotate-half");
    }
}

TEST(FusedIndexMappingExecution, RopeOfStridedRuntimeViewReadsPartnerFromTheSameRow) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // q = qkv[..., 0:16] of a [2,3,8,48] buffer, starting at element 5: row pitch 48, not 16.
    const std::vector<uint64_t> storage_dims{2 * 3 * 8 * 48 + 8};
    const std::vector<uint64_t> view_dims{2, 3, 8, 16};
    const std::vector<float> storage = ramp(storage_dims[0], 0.0625f, 29);
    Tensor buffer = toGpu(storage, storage_dims, DataType::FP32, stream);
    Tensor q = buffer.aliasView(view_dims, {3 * 8 * 48, 8 * 48, 48, 1}, 5);

    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    FusedEquation eq = FusedEquation::compile(Expression::outputs({{"y", x.rope()}}).physicalOutputs(), 0);
    Tensor ys(gpu_placement, TensorDescriptor(DataType::FP32, view_dims));
    eq.run({{"x", q}}, ys, stream);

    std::vector<float> dense(2 * 3 * 8 * 16);
    for (size_t row = 0; row < 2 * 3 * 8; ++row)
        for (size_t k = 0; k < 16; ++k) dense[row * 16 + k] = storage[5 + row * 48 + k];
    expectNear(toHost(ys, stream), ropeReference(dense, view_dims, false), 2e-5f, "strided q view");
}

TEST(FusedIndexMappingExecution, StridedViewBroadcastByParentUsesItsOwnCoordinates) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // view(x, [3,1], strides [4,1], offset 1) + y[3,4]: element (r, c) reads x[1 + 4r].
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression y = Expression::input("y", DataType::FP32, DataType::FP32);
    FusedEquation eq =
        FusedEquation::compile(Expression::outputs({{"z", x.stridedView({3, 1}, {4, 1}, 1) + y}}).physicalOutputs(), 0);
    const std::vector<float> xv = ramp(12, 1.0f, 12);
    const std::vector<float> yv = ramp(12, 100.0f, 12);
    Tensor xs = toGpu(xv, {3, 4}, DataType::FP32, stream);
    Tensor ys = toGpu(yv, {3, 4}, DataType::FP32, stream);
    Tensor zs(gpu_placement, TensorDescriptor(DataType::FP32, {3, 4}));
    eq.run({{"x", xs}, {"y", ys}}, zs, stream);
    std::vector<float> want(12);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 4; ++c) want[r * 4 + c] = xv[1 + 4 * r] + yv[r * 4 + c];
    expectNear(toHost(zs, stream), want, 0.0f, "broadcast strided view");
}

TEST(FusedIndexMappingExecution, TransposedOperandPlusUnsqueezedBias) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // x[40,72]^T + b[72].unsqueeze(1): out[r, c] = x[c, r] + b[r], shape [72, 40].
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression b = Expression::input("b", DataType::FP32, DataType::FP32);
    FusedEquation eq = FusedEquation::compile(Expression::outputs({{"z", x.transpose() + b.unsqueeze({1})}}).physicalOutputs(), 0);
    const std::vector<float> xv = ramp(40 * 72, 0.5f, 31);
    const std::vector<float> bv = ramp(72, 10.0f, 17);
    Tensor xs = toGpu(xv, {40, 72}, DataType::FP32, stream);
    Tensor bs = toGpu(bv, {72}, DataType::FP32, stream);
    Tensor zs(gpu_placement, TensorDescriptor(DataType::FP32, {72, 40}));
    eq.run({{"x", xs}, {"b", bs}}, zs, stream);
    std::vector<float> want(72 * 40);
    for (size_t r = 0; r < 72; ++r)
        for (size_t c = 0; c < 40; ++c) want[r * 40 + c] = xv[c * 72 + r] + bv[r];
    expectNear(toHost(zs, stream), want, 0.0f, "transpose + unsqueezed bias");
}

TEST(FusedIndexMappingExecution, BatchedFp8TransposedBroadcastReadsEachBatch) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    // (x[3,40,72] + bias[72])^T over the last two axes, FP8 storage: batches 1 and 2 must not
    // reuse batch 0's inputs.
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        const Expression x = Expression::input("x", dtype, dtype);
        const Expression bias = Expression::input("bias", dtype, dtype);
        FusedEquation eq = FusedEquation::compile(Expression::outputs({{"z", (x + bias).transpose()}}).physicalOutputs(), 0);
        // Values on a 0.5 grid in [-2, 1.5]; sums stay exactly representable in E4M3 and E5M2.
        std::vector<float> xv(3 * 40 * 72), bv(72);
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = 0.5f * static_cast<float>(static_cast<int>((i * 7 + i / 2880) % 8) - 4);
        for (size_t i = 0; i < bv.size(); ++i) bv[i] = 0.5f * static_cast<float>(static_cast<int>(i % 4) - 2);
        Tensor xs = toGpu(xv, {3, 40, 72}, dtype, stream);
        Tensor bs = toGpu(bv, {72}, dtype, stream);
        Tensor zs(gpu_placement, TensorDescriptor(dtype, {3, 72, 40}));
        eq.run({{"x", xs}, {"bias", bs}}, zs, stream);
        std::vector<float> want(3 * 72 * 40);
        for (size_t n = 0; n < 3; ++n)
            for (size_t r = 0; r < 40; ++r)
                for (size_t c = 0; c < 72; ++c) want[(n * 72 + c) * 40 + r] = xv[(n * 40 + r) * 72 + c] + bv[c];
        expectNear(toHost(zs, stream), want, 0.0f, TensorDescriptor::getElementTypeName(dtype));
    }
}
