#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for RaggedConcatenate kernel tests.";                              \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<float>() { return DataType::FP32; }
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <>
DataType dtypeFor<uint32_t>() { return DataType::UINT32; }
template <>
DataType dtypeFor<uint64_t>() { return DataType::UINT64; }

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t> &dimensions, const std::vector<T> &values, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size())
        throw std::runtime_error("RaggedConcatenate test tensor value count mismatch.");
    T *hostValues = host.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) hostValues[i] = values[i];
    Tensor device(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

template <typename T>
std::vector<T> copyGpuTensor(const Tensor &device, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), device.getDimensions()));
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const T *hostValues = host.getMemPtr<T>();
    return std::vector<T>(hostValues, hostValues + host.getTotalNumElements());
}

class DeviceAllocation {
   public:
    DeviceAllocation() = default;
    explicit DeviceAllocation(std::size_t bytes) { allocate(bytes); }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    DeviceAllocation(DeviceAllocation &&other) noexcept : pointer(other.pointer) { other.pointer = nullptr; }
    DeviceAllocation &operator=(DeviceAllocation &&other) noexcept {
        if (this == &other) return *this;
        if (pointer != nullptr) cudaFree(pointer);
        pointer = other.pointer;
        other.pointer = nullptr;
        return *this;
    }
    ~DeviceAllocation() {
        if (pointer != nullptr) cudaFree(pointer);
    }

    void allocate(std::size_t bytes) {
        if (pointer != nullptr) throw std::logic_error("DeviceAllocation may only be allocated once.");
        CUDA_CHECK(cudaMalloc(&pointer, bytes));
    }
    void *get() const { return pointer; }

   private:
    void *pointer = nullptr;
};

template <typename T>
DeviceAllocation uploadArray(const std::vector<T> &values, Stream &stream) {
    DeviceAllocation allocation(values.size() * sizeof(T));
    CUDA_CHECK(cudaMemcpyAsync(allocation.get(), values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream.getStream()));
    stream.synchronize();
    return allocation;
}

// DeviceAllocation is intentionally move-only so uploaded metadata has simple
// ownership even when a test assertion exits early.
DeviceAllocation makePointerTable(const std::vector<void *> &pointers, Stream &stream) {
    DeviceAllocation allocation(pointers.size() * sizeof(void *));
    CUDA_CHECK(cudaMemcpyAsync(allocation.get(), pointers.data(), pointers.size() * sizeof(void *), cudaMemcpyHostToDevice, stream.getStream()));
    stream.synchronize();
    return allocation;
}

std::vector<float> sourceValues(uint64_t rows, uint64_t axis, uint64_t inner, float base, float inactivePoison) {
    std::vector<float> values(rows * axis * inner, inactivePoison);
    constexpr uint64_t activeRows = 5;
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t d1 = 0; d1 < axis; ++d1) {
            for (uint64_t d2 = 0; d2 < inner; ++d2) {
                values[(row * axis + d1) * inner + d2] = base + static_cast<float>(100 * row + 10 * d1 + d2);
            }
        }
    }
    return values;
}

void runForwardAndSplitCase() {
    constexpr uint64_t rows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t inner = 2;
    constexpr uint64_t leftAxis = 2;
    constexpr uint64_t rightAxis = 3;
    constexpr uint64_t joinedAxis = leftAxis + rightAxis;
    constexpr float inactiveInputPoison = std::numeric_limits<float>::quiet_NaN();
    constexpr float outputSentinel = -7777.0F;
    constexpr float splitSentinel = -8888.0F;

    Stream stream(0);
    std::vector<float> leftValues = sourceValues(rows, leftAxis, inner, 1000.0F, inactiveInputPoison);
    std::vector<float> rightValues = sourceValues(rows, rightAxis, inner, 2000.0F, inactiveInputPoison);
    Tensor left = makeGpuTensor<float>({rows, leftAxis, inner}, leftValues, stream);
    Tensor right = makeGpuTensor<float>({rows, rightAxis, inner}, rightValues, stream);
    Tensor joined = makeGpuTensor<float>({rows, joinedAxis, inner},
                                         std::vector<float>(rows * joinedAxis * inner, outputSentinel),
                                         stream);

    DeviceAllocation sourceTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    const auto spanGeometryHost = buildRaggedConcatenateSpanGeometry(
        sizeof(float), 1, inner, {0, leftAxis, joinedAxis});
    DeviceAllocation spanGeometry = uploadArray(spanGeometryHost, stream);

    launchRaggedConcatenate(joined.getMemPtr(),
                            reinterpret_cast<void **>(sourceTable.get()),
                            rows,
                            joinedAxis * inner * sizeof(float),
                            1,
                            2,
                            static_cast<RaggedConcatenateSpanGeometry *>(spanGeometry.get()),
                            activeRows,
                            stream);
    stream.synchronize();

    const std::vector<float> actualJoined = copyGpuTensor<float>(joined, stream);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t d1 = 0; d1 < joinedAxis; ++d1) {
            for (uint64_t d2 = 0; d2 < inner; ++d2) {
                const uint64_t joinedIndex = (row * joinedAxis + d1) * inner + d2;
                const float expected = d1 < leftAxis
                    ? leftValues[(row * leftAxis + d1) * inner + d2]
                    : rightValues[(row * rightAxis + (d1 - leftAxis)) * inner + d2];
                EXPECT_EQ(actualJoined[joinedIndex], expected) << "row=" << row << " d1=" << d1 << " d2=" << d2;
            }
        }
    }
    for (uint64_t i = activeRows * joinedAxis * inner; i < actualJoined.size(); ++i)
        EXPECT_EQ(actualJoined[i], outputSentinel) << "inactive joined element " << i;

    std::vector<float> upstream(rows * joinedAxis * inner, 3333.0F);
    for (uint64_t i = 0; i < activeRows * joinedAxis * inner; ++i) upstream[i] = static_cast<float>(i + 1);
    Tensor upstreamGpu = makeGpuTensor<float>({rows, joinedAxis, inner}, upstream, stream);
    Tensor leftGradient = makeGpuTensor<float>({rows, leftAxis, inner},
                                               std::vector<float>(rows * leftAxis * inner, splitSentinel),
                                               stream);
    Tensor rightGradient = makeGpuTensor<float>({rows, rightAxis, inner},
                                                std::vector<float>(rows * rightAxis * inner, splitSentinel),
                                                stream);
    DeviceAllocation destinationTable = makePointerTable({leftGradient.getMemPtr(), rightGradient.getMemPtr()}, stream);

    launchRaggedSplit(reinterpret_cast<void **>(destinationTable.get()),
                      upstreamGpu.getMemPtr(),
                      rows,
                      joinedAxis * inner * sizeof(float),
                      1,
                      2,
                      static_cast<RaggedConcatenateSpanGeometry *>(spanGeometry.get()),
                      activeRows,
                      stream);
    stream.synchronize();

    const std::vector<float> actualLeftGradient = copyGpuTensor<float>(leftGradient, stream);
    const std::vector<float> actualRightGradient = copyGpuTensor<float>(rightGradient, stream);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t d1 = 0; d1 < joinedAxis; ++d1) {
            for (uint64_t d2 = 0; d2 < inner; ++d2) {
                const float expected = upstream[(row * joinedAxis + d1) * inner + d2];
                if (d1 < leftAxis) {
                    EXPECT_EQ(actualLeftGradient[(row * leftAxis + d1) * inner + d2], expected);
                } else {
                    EXPECT_EQ(actualRightGradient[(row * rightAxis + (d1 - leftAxis)) * inner + d2], expected);
                }
            }
        }
    }
    for (uint64_t i = activeRows * leftAxis * inner; i < actualLeftGradient.size(); ++i)
        EXPECT_EQ(actualLeftGradient[i], splitSentinel) << "inactive left gradient element " << i;
    for (uint64_t i = activeRows * rightAxis * inner; i < actualRightGradient.size(); ++i)
        EXPECT_EQ(actualRightGradient[i], splitSentinel) << "inactive right gradient element " << i;
}

void runAllEmptyCase() {
    constexpr uint64_t rows = 4;
    constexpr uint64_t leftAxis = 1;
    constexpr uint64_t rightAxis = 2;
    constexpr uint64_t joinedAxis = 3;
    constexpr float sentinel = -9191.0F;

    Stream stream(0);
    Tensor left = makeGpuTensor<float>({rows, leftAxis}, std::vector<float>(rows * leftAxis, 1.0F), stream);
    Tensor right = makeGpuTensor<float>({rows, rightAxis}, std::vector<float>(rows * rightAxis, 2.0F), stream);
    Tensor joined = makeGpuTensor<float>({rows, joinedAxis}, std::vector<float>(rows * joinedAxis, sentinel), stream);

    DeviceAllocation sourceTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    const auto spanGeometryHost = buildRaggedConcatenateSpanGeometry(
        sizeof(float), 1, 1, {0, leftAxis, joinedAxis});
    DeviceAllocation spanGeometry = uploadArray(spanGeometryHost, stream);

    launchRaggedConcatenate(joined.getMemPtr(),
                            reinterpret_cast<void **>(sourceTable.get()),
                            rows,
                            joinedAxis * sizeof(float),
                            1,
                            2,
                            static_cast<RaggedConcatenateSpanGeometry *>(spanGeometry.get()),
                            0,
                            stream);
    stream.synchronize();
    for (float value : copyGpuTensor<float>(joined, stream)) EXPECT_EQ(value, sentinel);

    Tensor leftGradient = makeGpuTensor<float>({rows, leftAxis}, std::vector<float>(rows * leftAxis, sentinel), stream);
    Tensor rightGradient = makeGpuTensor<float>({rows, rightAxis}, std::vector<float>(rows * rightAxis, sentinel), stream);
    DeviceAllocation destinationTable = makePointerTable({leftGradient.getMemPtr(), rightGradient.getMemPtr()}, stream);
    launchRaggedSplit(reinterpret_cast<void **>(destinationTable.get()),
                      joined.getMemPtr(),
                      rows,
                      joinedAxis * sizeof(float),
                      1,
                      2,
                      static_cast<RaggedConcatenateSpanGeometry *>(spanGeometry.get()),
                      0,
                      stream);
    stream.synchronize();
    for (float value : copyGpuTensor<float>(leftGradient, stream)) EXPECT_EQ(value, sentinel);
    for (float value : copyGpuTensor<float>(rightGradient, stream)) EXPECT_EQ(value, sentinel);
}

void runOuterSliceGroupingCase(uint64_t rows, uint64_t outer, uint64_t activeRows) {
    constexpr uint64_t inner = 3;
    constexpr uint64_t leftAxis = 1;
    constexpr uint64_t rightAxis = 2;
    constexpr uint64_t joinedAxis = leftAxis + rightAxis;
    constexpr float inputPoison = -12345.0F;
    constexpr float outputSentinel = -23456.0F;
    constexpr float splitSentinel = -34567.0F;

    ASSERT_GE(rows, activeRows);
    ASSERT_GT(activeRows, 0U);

    Stream stream(0);
    std::vector<float> leftValues(rows * outer * leftAxis * inner, inputPoison);
    std::vector<float> rightValues(rows * outer * rightAxis * inner, inputPoison);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t outerIndex = 0; outerIndex < outer; ++outerIndex) {
            for (uint64_t innerIndex = 0; innerIndex < inner; ++innerIndex) {
                leftValues[((row * outer + outerIndex) * leftAxis) * inner + innerIndex] =
                    1000.0F + static_cast<float>(100 * row + 10 * outerIndex + innerIndex);
            }
            for (uint64_t axisIndex = 0; axisIndex < rightAxis; ++axisIndex) {
                for (uint64_t innerIndex = 0; innerIndex < inner; ++innerIndex) {
                    rightValues[((row * outer + outerIndex) * rightAxis + axisIndex) * inner + innerIndex] =
                        2000.0F + static_cast<float>(100 * row + 10 * outerIndex + 3 * axisIndex + innerIndex);
                }
            }
        }
    }

    Tensor left = makeGpuTensor<float>({rows, outer, leftAxis, inner}, leftValues, stream);
    Tensor right = makeGpuTensor<float>({rows, outer, rightAxis, inner}, rightValues, stream);
    Tensor joined = makeGpuTensor<float>({rows, outer, joinedAxis, inner},
                                         std::vector<float>(rows * outer * joinedAxis * inner, outputSentinel),
                                         stream);
    DeviceAllocation sourceTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    const auto spanGeometryHost = buildRaggedConcatenateSpanGeometry(
        sizeof(float), outer, inner, {0, leftAxis, joinedAxis});
    DeviceAllocation spanGeometry = uploadArray(spanGeometryHost, stream);

    launchRaggedConcatenate(joined.getMemPtr(),
                            reinterpret_cast<void **>(sourceTable.get()),
                            rows,
                            outer * joinedAxis * inner * sizeof(float),
                            outer,
                            2,
                            static_cast<RaggedConcatenateSpanGeometry *>(spanGeometry.get()),
                            activeRows,
                            stream);
    stream.synchronize();

    const std::vector<float> actualJoined = copyGpuTensor<float>(joined, stream);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t outerIndex = 0; outerIndex < outer; ++outerIndex) {
            for (uint64_t axisIndex = 0; axisIndex < joinedAxis; ++axisIndex) {
                for (uint64_t innerIndex = 0; innerIndex < inner; ++innerIndex) {
                    const uint64_t joinedIndex =
                        ((row * outer + outerIndex) * joinedAxis + axisIndex) * inner + innerIndex;
                    const float expected = axisIndex < leftAxis
                        ? leftValues[((row * outer + outerIndex) * leftAxis + axisIndex) * inner + innerIndex]
                        : rightValues[((row * outer + outerIndex) * rightAxis + (axisIndex - leftAxis)) * inner + innerIndex];
                    EXPECT_EQ(actualJoined[joinedIndex], expected)
                        << "row=" << row << " outer=" << outerIndex << " axis=" << axisIndex
                        << " inner=" << innerIndex;
                }
            }
        }
    }
    for (uint64_t i = activeRows * outer * joinedAxis * inner; i < actualJoined.size(); ++i)
        EXPECT_EQ(actualJoined[i], outputSentinel) << "inactive joined element " << i;

    std::vector<float> upstream(rows * outer * joinedAxis * inner, 4444.0F);
    for (uint64_t i = 0; i < activeRows * outer * joinedAxis * inner; ++i)
        upstream[i] = static_cast<float>(i + 1);
    Tensor upstreamGpu = makeGpuTensor<float>({rows, outer, joinedAxis, inner}, upstream, stream);
    Tensor leftGradient = makeGpuTensor<float>({rows, outer, leftAxis, inner},
                                               std::vector<float>(rows * outer * leftAxis * inner, splitSentinel),
                                               stream);
    Tensor rightGradient = makeGpuTensor<float>({rows, outer, rightAxis, inner},
                                                std::vector<float>(rows * outer * rightAxis * inner, splitSentinel),
                                                stream);
    DeviceAllocation destinationTable = makePointerTable({leftGradient.getMemPtr(), rightGradient.getMemPtr()}, stream);

    launchRaggedSplit(reinterpret_cast<void **>(destinationTable.get()),
                      upstreamGpu.getMemPtr(),
                      rows,
                      outer * joinedAxis * inner * sizeof(float),
                      outer,
                      2,
                      static_cast<RaggedConcatenateSpanGeometry *>(spanGeometry.get()),
                      activeRows,
                      stream);
    stream.synchronize();

    const std::vector<float> actualLeft = copyGpuTensor<float>(leftGradient, stream);
    const std::vector<float> actualRight = copyGpuTensor<float>(rightGradient, stream);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t outerIndex = 0; outerIndex < outer; ++outerIndex) {
            for (uint64_t axisIndex = 0; axisIndex < joinedAxis; ++axisIndex) {
                for (uint64_t innerIndex = 0; innerIndex < inner; ++innerIndex) {
                    const float expected = upstream[
                        ((row * outer + outerIndex) * joinedAxis + axisIndex) * inner + innerIndex];
                    if (axisIndex < leftAxis) {
                        EXPECT_EQ(actualLeft[
                            ((row * outer + outerIndex) * leftAxis + axisIndex) * inner + innerIndex], expected);
                    } else {
                        EXPECT_EQ(actualRight[
                            ((row * outer + outerIndex) * rightAxis + (axisIndex - leftAxis)) * inner + innerIndex],
                            expected);
                    }
                }
            }
        }
    }
    for (uint64_t i = activeRows * outer * leftAxis * inner; i < actualLeft.size(); ++i)
        EXPECT_EQ(actualLeft[i], splitSentinel) << "inactive left gradient element " << i;
    for (uint64_t i = activeRows * outer * rightAxis * inner; i < actualRight.size(); ++i)
        EXPECT_EQ(actualRight[i], splitSentinel) << "inactive right gradient element " << i;
}

void runBytePayloadCase(uint64_t activeRows, uint64_t spanBytes) {
    ASSERT_GT(activeRows, 0U);
    ASSERT_GT(spanBytes, 0U);
    const uint64_t rows = activeRows + 1;
    constexpr uint8_t leftValue = 0x31;
    constexpr uint8_t rightValue = 0xA7;
    constexpr uint8_t sentinel = 0xDC;

    Stream stream(0);
    Tensor left = makeGpuTensor<uint8_t>(
        {rows, spanBytes}, std::vector<uint8_t>(rows * spanBytes, leftValue), stream);
    Tensor right = makeGpuTensor<uint8_t>(
        {rows, spanBytes}, std::vector<uint8_t>(rows * spanBytes, rightValue), stream);
    Tensor joined = makeGpuTensor<uint8_t>(
        {rows, 2 * spanBytes}, std::vector<uint8_t>(rows * 2 * spanBytes, sentinel), stream);
    DeviceAllocation sourceTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    const auto geometryHost = buildRaggedConcatenateSpanGeometry(
        sizeof(uint8_t), 1, 1, {0, spanBytes, 2 * spanBytes});
    DeviceAllocation geometry = uploadArray(geometryHost, stream);

    launchRaggedConcatenate(joined.getMemPtr(),
                            reinterpret_cast<void **>(sourceTable.get()),
                            rows,
                            2 * spanBytes,
                            1,
                            2,
                            static_cast<RaggedConcatenateSpanGeometry *>(geometry.get()),
                            activeRows,
                            stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(joined, stream);
    for (uint64_t row = 0; row < activeRows; ++row) {
        const uint64_t base = row * 2 * spanBytes;
        EXPECT_EQ(actual[base], leftValue);
        EXPECT_EQ(actual[base + spanBytes - 1], leftValue);
        EXPECT_EQ(actual[base + spanBytes], rightValue);
        EXPECT_EQ(actual[base + 2 * spanBytes - 1], rightValue);
    }
    const uint64_t inactiveBase = activeRows * 2 * spanBytes;
    EXPECT_EQ(actual[inactiveBase], sentinel);
    EXPECT_EQ(actual[inactiveBase + 2 * spanBytes - 1], sentinel);
}

void runExactTailCase(uint32_t copyWidth, uint32_t tailBytes) {
    ASSERT_GT(copyWidth, 1U);
    ASSERT_GT(tailBytes, 0U);
    ASSERT_LT(tailBytes, copyWidth);

    uint64_t alignmentOffset = 0;
    switch (copyWidth) {
        case 32: alignmentOffset = 0; break;
        case 16: alignmentOffset = 16; break;
        case 8: alignmentOffset = 8; break;
        case 4: alignmentOffset = 4; break;
        case 2: alignmentOffset = 2; break;
        default: FAIL() << "unsupported copy width"; return;
    }

    const uint64_t spanBytes = copyWidth + tailBytes;
    const uint64_t sourceLogicalBytes = alignmentOffset + spanBytes;
    const uint64_t destinationValueBytes = spanBytes + copyWidth;
    constexpr uint8_t sentinel = 0xD7;

    std::vector<uint8_t> sourceValues(sourceLogicalBytes);
    for (uint64_t i = 0; i < sourceLogicalBytes; ++i)
        sourceValues[i] = static_cast<uint8_t>((37U * i + 11U) & 0xFFU);

    Stream stream(0);
    Tensor source = makeGpuTensor<uint8_t>({sourceLogicalBytes}, sourceValues, stream);
    Tensor forwardDestination = makeGpuTensor<uint8_t>(
        {alignmentOffset + destinationValueBytes},
        std::vector<uint8_t>(alignmentOffset + destinationValueBytes, sentinel),
        stream);

    auto *sourceStart = static_cast<uint8_t *>(source.getMemPtr()) + alignmentOffset;
    auto *forwardStart = static_cast<uint8_t *>(forwardDestination.getMemPtr()) + alignmentOffset;
    DeviceAllocation sourceTable = makePointerTable({sourceStart, sourceStart}, stream);
    const std::vector<RaggedConcatenateSpanGeometry> geometryHost = {
        {spanBytes, spanBytes, 0},
        {0, 0, spanBytes},
    };
    DeviceAllocation geometry = uploadArray(geometryHost, stream);

    launchRaggedConcatenate(forwardStart,
                            reinterpret_cast<void **>(sourceTable.get()),
                            1,
                            destinationValueBytes,
                            1,
                            2,
                            static_cast<RaggedConcatenateSpanGeometry *>(geometry.get()),
                            1,
                            stream);
    stream.synchronize();

    const std::vector<uint8_t> forwardValues = copyGpuTensor<uint8_t>(forwardDestination, stream);
    for (uint64_t i = 0; i < spanBytes; ++i)
        EXPECT_EQ(forwardValues[alignmentOffset + i], sourceValues[alignmentOffset + i]);
    for (uint64_t i = spanBytes; i < destinationValueBytes; ++i)
        EXPECT_EQ(forwardValues[alignmentOffset + i], sentinel) << "forward tail over-store byte=" << i;

    Tensor splitDestination = makeGpuTensor<uint8_t>(
        {alignmentOffset + destinationValueBytes},
        std::vector<uint8_t>(alignmentOffset + destinationValueBytes, sentinel),
        stream);
    auto *splitStart = static_cast<uint8_t *>(splitDestination.getMemPtr()) + alignmentOffset;
    DeviceAllocation destinationTable = makePointerTable({splitStart, nullptr}, stream);
    launchRaggedSplit(reinterpret_cast<void **>(destinationTable.get()),
                      sourceStart,
                      1,
                      destinationValueBytes,
                      1,
                      2,
                      static_cast<RaggedConcatenateSpanGeometry *>(geometry.get()),
                      1,
                      stream);
    stream.synchronize();

    const std::vector<uint8_t> splitValues = copyGpuTensor<uint8_t>(splitDestination, stream);
    for (uint64_t i = 0; i < spanBytes; ++i)
        EXPECT_EQ(splitValues[alignmentOffset + i], sourceValues[alignmentOffset + i]);
    for (uint64_t i = spanBytes; i < destinationValueBytes; ++i)
        EXPECT_EQ(splitValues[alignmentOffset + i], sentinel) << "split tail over-store byte=" << i;
}

}  // namespace

TEST(RaggedConcatenate, ForwardAndSplitRespectHostActivePrefixAndNonLastTrailingAxisStrides) {
    REQUIRE_CUDA_DEVICE();
    runForwardAndSplitCase();
}

TEST(RaggedConcatenate, AllEmptyPartitionLeavesForwardAndBackwardCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    runAllEmptyCase();
}


TEST(RaggedConcatenate, PayloadAwareSpanGroupingCoversFullOneThrough256SpecializationLadder) {
    REQUIRE_CUDA_DEVICE();

    // These spans are tiny, so payload geometry permits 256 spans/CTA. Increase
    // only the authoritative active prefix to force every parallelism-floor
    // specialization in the 1/2/4/.../256 span ladder.
    for (const uint64_t activeRows : std::vector<uint64_t>{
             32,    // 64 active spans:     1 span / CTA
             64,    // 128 active spans:    2 spans / CTA
             128,   // 256 active spans:    4 spans / CTA
             256,   // 512 active spans:    8 spans / CTA
             512,   // 1024 active spans:  16 spans / CTA
             1024,  // 2048 active spans:  32 spans / CTA
             2048,  // 4096 active spans:  64 spans / CTA
             4096,  // 8192 active spans: 128 spans / CTA
             8192,  // 16384 active spans:256 spans / CTA
         }) {
        runOuterSliceGroupingCase(activeRows + 1, 1, activeRows);
    }
}


TEST(RaggedConcatenate, AlignedBulkCopiesUseExactTailForEverySupportedWidthAndTailSize) {
    REQUIRE_CUDA_DEVICE();
    for (const uint32_t copyWidth : {32U, 16U, 8U, 4U, 2U}) {
        for (uint32_t tailBytes = 1; tailBytes < copyWidth; ++tailBytes)
            runExactTailCase(copyWidth, tailBytes);
    }
}


TEST(RaggedConcatenate, PayloadThresholdTransitionsUseNextWiderSpanGrouping) {
    REQUIRE_CUDA_DEVICE();

    struct LaunchCase { uint64_t spanBytes; uint64_t activeRows; };
    // The first byte above each 32-bytes/lane boundary halves spans/CTA. The
    // active-span count is chosen so the block-parallelism floor is never the
    // limiting selector for the expected result.
    for (const LaunchCase launchCase : std::vector<LaunchCase>{
             {33, 8192},
             {65, 4096},
             {129, 2048},
             {257, 1024},
             {513, 512},
             {1025, 256},
             {2049, 128},
             {4097, 64},
         }) {
        runBytePayloadCase(launchCase.activeRows, launchCase.spanBytes);
    }
}
