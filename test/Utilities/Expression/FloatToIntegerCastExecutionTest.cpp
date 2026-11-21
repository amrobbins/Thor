#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

const std::vector<DataType> integerTypes = {DataType::INT8,
                                            DataType::INT16,
                                            DataType::INT32,
                                            DataType::INT64,
                                            DataType::UINT8,
                                            DataType::UINT16,
                                            DataType::UINT32,
                                            DataType::UINT64};

std::string outputName(DataType dtype) { return "y_" + TensorDescriptor::getElementTypeName(dtype); }

// NaN, infinities, signed zeros, the smallest subnormal, the largest values, fractions, and every power of two from
// 2^7 to 2^64 with its neighbours, both signs.
std::vector<uint8_t> fp32Specials() {
    std::vector<float> values = {std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::denorm_min(),
                                 std::numeric_limits<float>::max(),
                                 0.0f,
                                 1e30f};
    for (const float fraction : {0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.5f, 99.9f})
        values.push_back(fraction);
    for (const int exponent : {7, 8, 15, 16, 31, 32, 63, 64}) {
        const float power = std::ldexp(1.0f, exponent);
        for (const float offset : {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f})
            values.push_back(power + offset);
        values.push_back(std::nextafter(power, 0.0f));
        values.push_back(std::nextafter(power, std::numeric_limits<float>::infinity()));
    }
    const size_t positives = values.size();
    for (size_t i = 0; i < positives; ++i)
        values.push_back(-values[i]);
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<uint8_t> everyBitPattern(size_t bytesPerElement) {
    const size_t count = size_t(1) << (8 * bytesPerElement);
    std::vector<uint8_t> bytes(count * bytesPerElement);
    for (size_t i = 0; i < count; ++i)
        memcpy(bytes.data() + i * bytesPerElement, &i, bytesPerElement);
    return bytes;
}

// Casts every element of source to every integer type in one fused equation on the GPU, and expects the bytes that
// TypeConverter's CPU path gives (Thor's float -> integer rule, which TypeConverterTest checks against an independent
// reference): truncate toward zero, saturate, and NaN as PTX documents for cvt.rzi.
void expectEmittedCastsMatchTypeConverter(DataType sourceType, const std::vector<uint8_t>& source) {
    Stream stream(0);
    const size_t sourceBytes = static_cast<size_t>(TensorDescriptor::getElementSizeInBytes(sourceType));
    const uint64_t count = source.size() / sourceBytes;
    const std::vector<uint64_t> dims{count};

    Tensor sourceCpu(cpuPlacement, TensorDescriptor(sourceType, dims));
    memcpy(sourceCpu.getMemPtr<void>(), source.data(), source.size());
    Tensor sourceGpu(gpuPlacement, TensorDescriptor(sourceType, dims));
    sourceGpu.copyFromAsync(sourceCpu, stream);

    const Expression x = Expression::input("x", sourceType, sourceType);
    std::vector<std::pair<std::string, Expression>> outputs;
    for (const DataType integerType : integerTypes)
        outputs.emplace_back(outputName(integerType), x.cast(integerType));
    FusedEquation equation = FusedEquation::compile(Expression::outputs(outputs).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"x", sourceGpu}}, stream);
    plan.run();

    for (const DataType integerType : integerTypes) {
        const size_t destBytes = static_cast<size_t>(TensorDescriptor::getElementSizeInBytes(integerType));
        Tensor emitted(cpuPlacement, TensorDescriptor(integerType, dims));
        emitted.copyFromAsync(plan.output(outputName(integerType)), stream);
        std::vector<uint8_t> expected(count * destBytes);
        TypeConverter::convertType(const_cast<uint8_t*>(source.data()), expected.data(), sourceType, integerType, count, stream, -1);
        stream.synchronize();

        const auto* got = static_cast<const uint8_t*>(emitted.getMemPtr<void>());
        size_t mismatches = 0;
        for (uint64_t i = 0; i < count; ++i) {
            if (memcmp(got + i * destBytes, expected.data() + i * destBytes, destBytes) == 0)
                continue;
            if (mismatches++ < 8) {
                uint64_t sourceBits = 0;
                uint64_t gotBits = 0;
                uint64_t expectedBits = 0;
                memcpy(&sourceBits, source.data() + i * sourceBytes, sourceBytes);
                memcpy(&gotBits, got + i * destBytes, destBytes);
                memcpy(&expectedBits, expected.data() + i * destBytes, destBytes);
                ADD_FAILURE() << TensorDescriptor::getElementTypeName(sourceType) << " -> "
                              << TensorDescriptor::getElementTypeName(integerType) << " of bits 0x" << std::hex << sourceBits
                              << ": emitted 0x" << gotBits << ", expected 0x" << expectedBits << std::dec;
            }
        }
        EXPECT_EQ(mismatches, 0u) << TensorDescriptor::getElementTypeName(sourceType) << " -> "
                                  << TensorDescriptor::getElementTypeName(integerType);
    }
}

}  // namespace

// Expression's emitted float -> integer casts follow the same rule as TypeConverter. FP16, BF16 and both FP8 kinds
// take every bit pattern; FP32 takes the special values.
TEST(FloatToIntegerCastExecution, EmittedCastsMatchTypeConverterForEverySourceAndIntegerType) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0)
        GTEST_SKIP() << "CUDA device is required.";
    expectEmittedCastsMatchTypeConverter(DataType::FP32, fp32Specials());
    expectEmittedCastsMatchTypeConverter(DataType::FP16, everyBitPattern(2));
    expectEmittedCastsMatchTypeConverter(DataType::BF16, everyBitPattern(2));
    expectEmittedCastsMatchTypeConverter(DataType::FP8_E4M3, everyBitPattern(1));
    expectEmittedCastsMatchTypeConverter(DataType::FP8_E5M2, everyBitPattern(1));
}
