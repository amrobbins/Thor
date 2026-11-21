// Expression scans' dtypes: FP8 inputs, a float sum or product written to another float (withOutputDType), the pairs
// the scan kernel refuses, and the backward of a widened sum. Values are small multiples of 1/4, so every FP32 prefix
// is exact and the expected outputs are the exact prefixes rounded once into the output dtype.

#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                     \
    do {                                                                                          \
        int cuda_device_count_for_test = 0;                                                       \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test); \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {             \
            GTEST_SKIP() << "CUDA device is required for Expression scan dtype tests.";           \
        }                                                                                         \
    } while (false)

// One element's bytes in a float dtype, rounded from `value` (FP8 saturating, as the scan kernel writes it).
std::vector<uint8_t> floatBytes(DataType dtype, float value) {
    std::vector<uint8_t> bytes(TensorDescriptor::getElementSizeInBytes(dtype));
    if (dtype == DataType::FP32) {
        std::memcpy(bytes.data(), &value, 4);
    } else if (dtype == DataType::FP16) {
        const __half h = __float2half(value);
        std::memcpy(bytes.data(), &h, 2);
    } else if (dtype == DataType::BF16) {
        const __nv_bfloat16 b = __float2bfloat16(value);
        std::memcpy(bytes.data(), &b, 2);
    } else if (dtype == DataType::FP8_E4M3) {
        const __nv_fp8_e4m3 q(value);
        std::memcpy(bytes.data(), &q, 1);
    } else if (dtype == DataType::FP8_E5M2) {
        const __nv_fp8_e5m2 q(value);
        std::memcpy(bytes.data(), &q, 1);
    } else {
        throw std::runtime_error("floatBytes: not a float dtype");
    }
    return bytes;
}

float floatValue(DataType dtype, const uint8_t* bytes) {
    if (dtype == DataType::FP32) {
        float f;
        std::memcpy(&f, bytes, 4);
        return f;
    }
    if (dtype == DataType::FP16) {
        __half h;
        std::memcpy(&h, bytes, 2);
        return __half2float(h);
    }
    if (dtype == DataType::BF16) {
        __nv_bfloat16 b;
        std::memcpy(&b, bytes, 2);
        return __bfloat162float(b);
    }
    if (dtype == DataType::FP8_E4M3) {
        __nv_fp8_e4m3 q;
        std::memcpy(&q, bytes, 1);
        return static_cast<float>(q);
    }
    if (dtype == DataType::FP8_E5M2) {
        __nv_fp8_e5m2 q;
        std::memcpy(&q, bytes, 1);
        return static_cast<float>(q);
    }
    throw std::runtime_error("floatValue: not a float dtype");
}

Tensor makeGpuTensor(DataType dtype, const std::vector<uint64_t>& dims, const std::vector<uint8_t>& bytes, Stream& stream) {
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(dtype, dims));
    if (cpu.getArraySizeInBytes() < bytes.size()) {
        throw std::runtime_error("makeGpuTensor: too many bytes for the dimensions");
    }
    std::memcpy(cpu.getMemPtr(), bytes.data(), bytes.size());
    Tensor gpu(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(dtype, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

Tensor makeFloatTensor(DataType dtype, const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    std::vector<uint8_t> bytes;
    for (const float value : values) {
        const std::vector<uint8_t> element = floatBytes(dtype, value);
        bytes.insert(bytes.end(), element.begin(), element.end());
    }
    return makeGpuTensor(dtype, dims, bytes, stream);
}

Tensor makeUint32Tensor(const std::vector<uint64_t>& dims, const std::vector<uint32_t>& values, Stream& stream) {
    std::vector<uint8_t> bytes(values.size() * 4);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return makeGpuTensor(DataType::UINT32, dims, bytes, stream);
}

std::vector<uint8_t> copyBytes(const Tensor& gpu, Stream& stream) {
    Tensor cpu(TensorPlacement(TensorPlacement::MemDevices::CPU), gpu.getDescriptor());
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const uint64_t bytes = gpu.getTotalNumElements() * TensorDescriptor::getElementSizeInBytes(gpu.getDataType());
    const auto* ptr = static_cast<const uint8_t*>(cpu.getMemPtr());
    return std::vector<uint8_t>(ptr, ptr + bytes);
}

// Each output element against the exact prefix of its row (rows of `cols`), rounded into the output's dtype.
void expectFloatPrefixes(const Tensor& output,
                         DataType dtype,
                         const std::vector<float>& input,
                         uint64_t cols,
                         bool product,
                         Stream& stream) {
    ASSERT_EQ(output.getDataType(), dtype);
    const std::vector<uint8_t> got = copyBytes(output, stream);
    const uint64_t element_bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    ASSERT_EQ(got.size(), input.size() * element_bytes);
    float prefix = 0.0f;
    for (uint64_t i = 0; i < input.size(); ++i) {
        prefix = i % cols == 0 ? input[i] : (product ? prefix * input[i] : prefix + input[i]);
        const std::vector<uint8_t> want = floatBytes(dtype, prefix);
        EXPECT_EQ(std::memcmp(got.data() + i * element_bytes, want.data(), element_bytes), 0)
            << "element " << i << ": got " << floatValue(dtype, got.data() + i * element_bytes) << ", want " << prefix;
    }
}

const std::vector<float> kValues{0.5f, -1.0f, 2.0f, 1.5f, -0.25f, 2.0f, -2.0f, 0.75f,
                                 1.0f, 1.0f, -0.5f, 3.0f, 0.25f, -1.5f, 3.0f, 0.5f};

}  // namespace

TEST(ExpressionScanDType, Fp8ScansWriteFp8OrAWiderFloat) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t rows = 2;
    constexpr uint64_t cols = 8;
    for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        SCOPED_TRACE(TensorDescriptor::getElementTypeName(dtype));
        const Expression x = Expression::input("x");
        FusedEquation equation = FusedEquation::compile(
            Expression::outputs({{"sum", x.scan(ScanOp::Sum)},
                                 {"wide", x.scan(ScanOp::Sum).withOutputDType(DataType::FP32)},
                                 {"max", x.scan(ScanOp::Max)},
                                 {"argmax", x.scan(ScanOp::ArgMax)}})
                .physicalOutputs(),
            0);
        Tensor input = makeFloatTensor(dtype, {rows, cols}, kValues, stream);
        StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);
        plan.run();
        stream.synchronize();

        // The input as stored, so the references see what the kernel reads.
        const std::vector<uint8_t> stored = copyBytes(input, stream);
        std::vector<float> values(kValues.size());
        for (uint64_t i = 0; i < values.size(); ++i) values[i] = floatValue(dtype, stored.data() + i);

        expectFloatPrefixes(plan.output("sum"), dtype, values, cols, false, stream);
        expectFloatPrefixes(plan.output("wide"), DataType::FP32, values, cols, false, stream);

        const std::vector<uint8_t> max = copyBytes(plan.output("max"), stream);
        const std::vector<uint8_t> argmax = copyBytes(plan.output("argmax"), stream);
        ASSERT_EQ(plan.output("max").getDataType(), dtype);
        ASSERT_EQ(plan.output("argmax").getDataType(), DataType::UINT32);
        uint32_t best = 0;
        for (uint32_t i = 0; i < values.size(); ++i) {
            if (i % cols == 0 || values[i] > values[best]) best = i;  // the first of equal values wins
            uint32_t index;
            std::memcpy(&index, argmax.data() + 4 * i, 4);
            EXPECT_EQ(index, best) << "element " << i;
            EXPECT_EQ(max[i], stored[best]) << "element " << i;
        }
    }
}

TEST(ExpressionScanDType, SumsAndProductsWriteAnotherFloat) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // Dense rows: FP16 sums and products written as FP32, E5M2 sums written as FP16, and FP32 sums and products written
    // as FP16 and E4M3 (narrower: the FP32 accumulator rounded once).
    {
        const Expression x = Expression::input("x");
        FusedEquation equation = FusedEquation::compile(
            Expression::outputs({{"sum", x.scan(ScanOp::Sum).withOutputDType(DataType::FP32)},
                                 {"product", x.scan(ScanOp::Product).withOutputDType(DataType::FP32)}})
                .physicalOutputs(),
            0);
        Tensor input = makeFloatTensor(DataType::FP16, {4, 4}, kValues, stream);
        StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);
        plan.run();
        stream.synchronize();
        expectFloatPrefixes(plan.output("sum"), DataType::FP32, kValues, 4, false, stream);
        expectFloatPrefixes(plan.output("product"), DataType::FP32, kValues, 4, true, stream);
    }
    {
        const Expression x = Expression::input("x");
        FusedEquation equation = FusedEquation::compile(
            Expression::outputs({{"sum", x.scan(ScanOp::Sum).withOutputDType(DataType::FP16)}}).physicalOutputs(), 0);
        Tensor input = makeFloatTensor(DataType::FP8_E5M2, {2, 8}, kValues, stream);
        StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);
        plan.run();
        stream.synchronize();
        expectFloatPrefixes(plan.output("sum"), DataType::FP16, kValues, 8, false, stream);
    }
    {
        const Expression x = Expression::input("x");
        FusedEquation equation = FusedEquation::compile(
            Expression::outputs({{"sum16", x.scan(ScanOp::Sum).withOutputDType(DataType::FP16)},
                                 {"sum8", x.scan(ScanOp::Sum).withOutputDType(DataType::FP8_E4M3)},
                                 {"product8", x.scan(ScanOp::Product).withOutputDType(DataType::FP8_E4M3)}})
                .physicalOutputs(),
            0);
        Tensor input = makeFloatTensor(DataType::FP32, {2, 8}, kValues, stream);
        StampedExecutionPlan plan = equation.stamp({{"x", input}}, stream);
        plan.run();
        stream.synchronize();
        expectFloatPrefixes(plan.output("sum16"), DataType::FP16, kValues, 8, false, stream);
        expectFloatPrefixes(plan.output("sum8"), DataType::FP8_E4M3, kValues, 8, false, stream);
        expectFloatPrefixes(plan.output("product8"), DataType::FP8_E4M3, kValues, 8, true, stream);
    }

    // Ragged rows: a BF16 sum written as FP32, rows [0, 3), [3, 8), [8, 16).
    {
        const Expression x = Expression::input("x");
        const Expression offsets = Expression::input("offsets");
        FusedEquation equation = FusedEquation::compile(
            Expression::outputs({{"sum", x.segmentedScan(offsets, ScanOp::Sum).withOutputDType(DataType::FP32)}})
                .physicalOutputs(),
            0);
        Tensor input = makeFloatTensor(DataType::BF16, {16}, kValues, stream);
        Tensor row_offsets = makeUint32Tensor({4}, {0U, 3U, 8U, 16U}, stream);
        StampedExecutionPlan plan = equation.stamp({{"x", input}, {"offsets", row_offsets}}, stream);
        plan.run();
        stream.synchronize();
        const Tensor sum = plan.output("sum");
        ASSERT_EQ(sum.getDataType(), DataType::FP32);
        const std::vector<uint8_t> got = copyBytes(sum, stream);
        float prefix = 0.0f;
        for (uint32_t i = 0; i < kValues.size(); ++i) {
            const bool head = i == 0 || i == 3 || i == 8;
            prefix = head ? kValues[i] : prefix + kValues[i];
            EXPECT_EQ(floatValue(DataType::FP32, got.data() + 4 * i), prefix) << "element " << i;
        }
    }
}

TEST(ExpressionScanDType, RefusesPairsTheScanKernelDoesNotTake) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor uint32 = makeUint32Tensor({2, 8}, std::vector<uint32_t>(16, 1U), stream);
    Tensor fp16 = makeFloatTensor(DataType::FP16, {2, 8}, kValues, stream);
    struct Case {
        const char* what;
        Expression scan;
        Tensor input;
    };
    const Expression x = Expression::input("x");
    const std::vector<Case> cases{
        {"a UINT32 sum writes UINT32", x.scan(ScanOp::Sum).withOutputDType(DataType::FP32), uint32},
        {"a max writes its input's dtype", x.scan(ScanOp::Max).withOutputDType(DataType::FP32), fp16},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.what);
        FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", c.scan}}).physicalOutputs(), 0);
        try {
            const StampedExecutionPlan plan = equation.stamp({{"x", c.input}}, stream);
            static_cast<void>(plan);
            FAIL() << "the scan was stamped";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string(error.what()).find("Expression scan from"), std::string::npos) << error.what();
        }
    }
}

// The gradient of a sum written as FP32 is FP32; the backward scan reads it and writes the FP16 input's gradient
// directly, so the backward runs the same stages as for a sum written as FP16 (no cast pass).
TEST(ExpressionScanDType, WidenedSumBackwardWritesTheInputsGradient) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    constexpr uint64_t cols = 4;
    Tensor input = makeFloatTensor(DataType::FP16, {2, cols}, std::vector<float>(2 * cols, 1.0f), stream);
    std::vector<std::vector<std::string>> stage_kinds;
    for (const bool widened : {false, true}) {
        SCOPED_TRACE(widened ? "FP16 sum written as FP32" : "FP16 sum written as FP16");
        const Expression x = Expression::input("x");
        const Expression y = widened ? x.scan(ScanOp::Sum).withOutputDType(DataType::FP32) : x.scan(ScanOp::Sum);
        FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0);
        FusedEquation backward = forward.compileBackward({"x"}, "dy");
        const DataType dy_dtype = widened ? DataType::FP32 : DataType::FP16;
        Tensor dy = makeFloatTensor(dy_dtype, {2, cols}, std::vector<float>(2 * cols, 1.0f), stream);
        StampedExecutionPlan plan = backward.stamp({{"x", input}, {"dy", dy}}, stream);
        stage_kinds.push_back(plan.stageKindNames());
        plan.run();
        stream.synchronize();

        // dx[i] = the sum of dy over the elements at or after i in its row: cols - (i % cols).
        const Tensor dx = plan.output("x_grad");
        ASSERT_EQ(dx.getDataType(), DataType::FP16);
        const std::vector<uint8_t> got = copyBytes(dx, stream);
        for (uint64_t i = 0; i < 2 * cols; ++i) {
            EXPECT_EQ(floatValue(DataType::FP16, got.data() + 2 * i), static_cast<float>(cols - i % cols))
                << "element " << i;
        }
    }
    ASSERT_EQ(stage_kinds.size(), 2U);
    EXPECT_EQ(stage_kinds[1], stage_kinds[0]);
}
