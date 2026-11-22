#include "DeepLearning/Implementation/Data/Residency/DeviceResidentWindowMaterializationKernel.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorSpan.h"
#include "Utilities/Common/LowPrecisionFloat.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for device-resident window materialization tests.";                \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <>
DataType dtypeFor<uint16_t>() { return DataType::UINT16; }

template <typename T>
Tensor makeGpuTensor(
    const std::vector<uint64_t> &dimensions,
    const std::vector<T> &values,
    Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error(
            "DeviceResidentWindowMaterializationKernel test tensor value count mismatch.");
    }
    std::memcpy(host.getMemPtr(), values.data(), values.size() * sizeof(T));
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

template <typename T>
std::vector<T> copyGpuTensor(const Tensor &device, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), device.getDimensions()));
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const T *values = host.getMemPtr<T>();
    return std::vector<T>(values, values + host.getTotalNumElements());
}

Tensor makeGpuPlans(
    const std::vector<DeviceResidentWindowRowPlan32> &plans,
    Stream &stream) {
    const uint64_t bytes = plans.size() * sizeof(DeviceResidentWindowRowPlan32);
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT8, {bytes}));
    std::memcpy(host.getMemPtr(), plans.data(), static_cast<size_t>(bytes));
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

Tensor makeGpuPlansAsWords(
    const std::vector<DeviceResidentWindowRowPlan32> &plans,
    Stream &stream) {
    const uint64_t bytes = plans.size() * sizeof(DeviceResidentWindowRowPlan32);
    if (bytes % sizeof(uint64_t) != 0) {
        throw std::runtime_error(
            "DeviceResidentWindowMaterializationKernel plan-word tensor must be UINT64 aligned.");
    }
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT64, {bytes / sizeof(uint64_t)}));
    std::memcpy(host.getMemPtr(), plans.data(), static_cast<size_t>(bytes));
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

TEST(DeviceResidentWindowMaterializationKernelTest,
     ResolvedPlanCopiesContiguousMiddleAndFillsExactPadding) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchCapacity = 3;
    constexpr uint64_t windowLength = 5;
    constexpr uint64_t stepBytes = sizeof(uint16_t);
    constexpr uint16_t pad = 0x1234;
    std::vector<uint16_t> source{10, 11, 12, 13, 14, 15, 16, 17};
    std::vector<DeviceResidentWindowRowPlan32> plans{
        {.sourceOffsetBytes = 2 * stepBytes, .validStepBegin = 1, .validStepCount = 3},
        {.sourceOffsetBytes = 0, .validStepBegin = 0, .validStepCount = 0},
        {.sourceOffsetBytes = 5 * stepBytes, .validStepBegin = 0, .validStepCount = 2},
    };
    const std::vector<uint16_t> expected{
        pad, 12, 13, 14, pad,
        pad, pad, pad, pad, pad,
        15, 16, pad, pad, pad,
    };

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint16_t>({source.size()}, source, stream);
    // File-backed sessions expose plan sections as UINT64 alias views into the
    // consolidated selection upload allocation.
    Tensor planDevice = makeGpuPlansAsWords(plans, stream);
    Tensor destination = makeGpuTensor<uint16_t>(
        {batchCapacity, windowLength},
        std::vector<uint16_t>(batchCapacity * windowLength, 0xffffU),
        stream);

    DeviceResidentWindowMaterializationSpec spec;
    spec.dataType = DataType::UINT16;
    spec.windowLength = windowLength;
    spec.sourceStepBytes = stepBytes;
    spec.padValue = static_cast<double>(pad);
    launchDeviceResidentWindowMaterializationKernel(
        sourceDevice, planDevice, batchCapacity, spec, destination, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<uint16_t>(destination, stream), expected);
}

TEST(DeviceResidentWindowMaterializationKernelTest,
     MaskUsesSameResolvedPlanAndLeavesInactiveCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchCapacity = 4;
    constexpr uint64_t logicalRows = 2;
    constexpr uint64_t windowLength = 7;
    constexpr uint8_t sentinel = 0xcdU;
    // Only live plan rows are transferred by the file-backed session; inactive
    // destination capacity is intentionally not represented in the plan table.
    std::vector<DeviceResidentWindowRowPlan32> plans{
        {.sourceOffsetBytes = 0, .validStepBegin = 2, .validStepCount = 3},
        {.sourceOffsetBytes = 0, .validStepBegin = 0, .validStepCount = 5},
    };
    std::vector<uint8_t> expected(batchCapacity * windowLength, sentinel);
    for (uint64_t step = 0; step < windowLength; ++step) {
        expected[step] = step >= 2 && step < 5 ? 1 : 0;
        expected[windowLength + step] = step < 5 ? 1 : 0;
    }

    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint8_t>({1}, {0}, stream);
    Tensor planDevice = makeGpuPlans(plans, stream);
    Tensor destination = makeGpuTensor<uint8_t>(
        {batchCapacity, windowLength},
        std::vector<uint8_t>(batchCapacity * windowLength, sentinel),
        stream);

    DeviceResidentWindowMaterializationSpec spec;
    spec.dataType = DataType::UINT8;
    spec.windowLength = windowLength;
    spec.sourceStepBytes = 1;
    spec.materializeMask = true;
    launchDeviceResidentWindowMaterializationKernel(
        sourceDevice, planDevice, logicalRows, spec, destination, stream);
    stream.synchronize();

    EXPECT_EQ(copyGpuTensor<uint8_t>(destination, stream), expected);
}

Tensor makeGpuBytes(DataType dataType,
                    const std::vector<uint64_t> &dimensions,
                    const std::vector<uint8_t> &bytes,
                    Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dataType, dimensions));
    if (host.getArraySizeInBytes() != bytes.size()) {
        throw std::runtime_error("DeviceResidentWindowMaterializationKernel test tensor byte count mismatch.");
    }
    std::memcpy(host.getMemPtr(), bytes.data(), bytes.size());
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

std::vector<uint8_t> copyGpuBytes(const Tensor &device, Stream &stream) {
    Tensor host(cpuPlacement, device.getDescriptor());
    host.copyFromAsync(device, stream);
    stream.synchronize();
    // Untyped: the tensor's own element type need not be a byte.
    const auto *bytes = static_cast<const uint8_t *>(host.getMemPtr());
    return std::vector<uint8_t>(bytes, bytes + host.getArraySizeInBytes());
}

uint64_t splitMix64(uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

void expectSameBytes(const std::vector<uint8_t> &actual, const std::vector<uint8_t> &expected, const std::string &what) {
    ASSERT_EQ(actual.size(), expected.size()) << what;
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end()) {
        ADD_FAILURE() << what << ": first wrong byte at " << (mismatch.first - actual.begin()) << " of " << actual.size()
                      << ", got " << static_cast<uint32_t>(*mismatch.first) << ", expected "
                      << static_cast<uint32_t>(*mismatch.second);
    }
}

// Row plans that between them start, end and skip the valid steps everywhere a window can: one row in five is all
// valid, one in five all padding, and the others take a pseudo-random run of steps (no more than the source holds).
// A plan's offset is a multiple of offsetAlignmentBytes that leaves its valid steps inside the source.
std::vector<DeviceResidentWindowRowPlan32> mixedPlans(uint64_t logicalRows,
                                                      uint64_t windowLength,
                                                      uint64_t stepBytes,
                                                      uint64_t sourceBytes,
                                                      uint64_t offsetAlignmentBytes) {
    std::vector<DeviceResidentWindowRowPlan32> plans(logicalRows);
    for (uint64_t row = 0; row < logicalRows; ++row) {
        const uint64_t kind = splitMix64(row * 3 + windowLength) % 5;
        uint64_t begin = kind == 0 ? 0 : splitMix64(row * 5 + 1) % (windowLength + 1);
        uint64_t count = kind == 0 ? windowLength : kind == 1 ? 0 : splitMix64(row * 7 + 2) % (windowLength - begin + 1);
        count = std::min(count, sourceBytes / stepBytes);
        const uint64_t room = sourceBytes - count * stepBytes;
        plans[row].sourceOffsetBytes = splitMix64(row * 11 + 3) % (room / offsetAlignmentBytes + 1) * offsetAlignmentBytes;
        plans[row].validStepBegin = static_cast<uint32_t>(begin);
        plans[row].validStepCount = static_cast<uint32_t>(count);
    }
    return plans;
}

struct PayloadCase {
    DataType dataType = DataType::UINT8;
    // The bytes of one padding element, as the data type stores padValue.
    std::vector<uint8_t> padElement{0};
    double padValue = 0.0;
    uint64_t stepBytes = 1;
    uint64_t windowLength = 1;
    uint64_t logicalRows = 1;
    uint64_t sourceOffsetAlignmentBytes = 1;
    // 0: three windows' worth.
    uint64_t sourceBytes = 0;
};

// Materialize mixed plans and compare every byte with the definition: a valid step is the source's, the others are
// the padding element, and the batch row past the logical rows is untouched.
void runPayloadCase(const PayloadCase &testCase) {
    constexpr uint8_t sentinel = 0xcdU;
    const uint64_t elementBytes = testCase.padElement.size();
    ASSERT_EQ(testCase.stepBytes % elementBytes, 0U);
    const uint64_t rowBytes = testCase.windowLength * testCase.stepBytes;
    const uint64_t batchCapacity = testCase.logicalRows + 1;
    const uint64_t sourceBytes = testCase.sourceBytes != 0 ? testCase.sourceBytes : 3 * rowBytes;
    std::vector<uint8_t> source(sourceBytes);
    for (uint64_t byte = 0; byte < sourceBytes; ++byte) source[byte] = static_cast<uint8_t>(splitMix64(byte) >> 24);
    const std::vector<DeviceResidentWindowRowPlan32> plans = mixedPlans(
        testCase.logicalRows, testCase.windowLength, testCase.stepBytes, sourceBytes, testCase.sourceOffsetAlignmentBytes);

    std::vector<uint8_t> expected(batchCapacity * rowBytes, sentinel);
    for (uint64_t row = 0; row < testCase.logicalRows; ++row) {
        const uint64_t validBegin = plans[row].validStepBegin * testCase.stepBytes;
        const uint64_t validEnd = validBegin + plans[row].validStepCount * testCase.stepBytes;
        for (uint64_t byte = 0; byte < rowBytes; ++byte) {
            expected[row * rowBytes + byte] = byte >= validBegin && byte < validEnd
                                                  ? source[plans[row].sourceOffsetBytes + byte - validBegin]
                                                  : testCase.padElement[byte % elementBytes];
        }
    }

    Stream stream(0);
    Tensor sourceDevice = makeGpuBytes(DataType::UINT8, {sourceBytes}, source, stream);
    Tensor planDevice = makeGpuPlansAsWords(plans, stream);
    Tensor destination = makeGpuBytes(
        testCase.dataType, {batchCapacity, rowBytes / elementBytes}, std::vector<uint8_t>(batchCapacity * rowBytes, sentinel), stream);

    DeviceResidentWindowMaterializationSpec spec;
    spec.dataType = testCase.dataType;
    spec.windowLength = testCase.windowLength;
    spec.sourceStepBytes = testCase.stepBytes;
    spec.padValue = testCase.padValue;
    spec.sourceOffsetAlignmentBytes = testCase.sourceOffsetAlignmentBytes;
    launchDeviceResidentWindowMaterializationKernel(sourceDevice, planDevice, testCase.logicalRows, spec, destination, stream);
    stream.synchronize();

    expectSameBytes(copyGpuBytes(destination, stream), expected,
                    "payload stepBytes=" + std::to_string(testCase.stepBytes) + " windowLength=" +
                        std::to_string(testCase.windowLength) + " rows=" + std::to_string(testCase.logicalRows) +
                        " elementBytes=" + std::to_string(elementBytes) + " offsetAlignment=" +
                        std::to_string(testCase.sourceOffsetAlignmentBytes) + " sourceBytes=" + std::to_string(sourceBytes));
}

void runMaskCase(uint64_t windowLength, uint64_t logicalRows) {
    constexpr uint8_t sentinel = 0xcdU;
    const uint64_t batchCapacity = logicalRows + 1;
    const std::vector<DeviceResidentWindowRowPlan32> plans =
        mixedPlans(logicalRows, windowLength, /*stepBytes=*/1, /*sourceBytes=*/windowLength, /*offsetAlignmentBytes=*/1);
    std::vector<uint8_t> expected(batchCapacity * windowLength, sentinel);
    for (uint64_t row = 0; row < logicalRows; ++row) {
        for (uint64_t step = 0; step < windowLength; ++step) {
            expected[row * windowLength + step] =
                step >= plans[row].validStepBegin && step < plans[row].validStepBegin + plans[row].validStepCount ? 1 : 0;
        }
    }

    Stream stream(0);
    Tensor planDevice = makeGpuPlansAsWords(plans, stream);
    Tensor destination = makeGpuBytes(
        DataType::UINT8, {batchCapacity, windowLength}, std::vector<uint8_t>(batchCapacity * windowLength, sentinel), stream);
    DeviceResidentWindowMaterializationSpec spec;
    spec.dataType = DataType::UINT8;
    spec.windowLength = windowLength;
    spec.sourceStepBytes = 1;
    spec.materializeMask = true;
    launchDeviceResidentWindowMaterializationKernel(Tensor(), planDevice, logicalRows, spec, destination, stream);
    stream.synchronize();

    expectSameBytes(copyGpuBytes(destination, stream), expected,
                    "mask windowLength=" + std::to_string(windowLength) + " rows=" + std::to_string(logicalRows));
}

template <typename T>
std::vector<uint8_t> bytesOf(T value) {
    std::vector<uint8_t> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

// The kernel moves 1, 2 or 4 packets per thread by the batch's packet count (steps of 196608), so these packet counts
// put a packet width on each of its launches.
constexpr std::array<uint64_t, 3> packetCountsOfEveryDepth{{1551, 250000, 500000}};

TEST(DeviceResidentWindowMaterializationKernelTest, EveryPacketWidthMaterializesPayloadExactlyAtEveryDepth) {
    REQUIRE_CUDA_DEVICE();

    struct Layout {
        uint64_t stepBytes;
        uint64_t sourceOffsetAlignmentBytes;
        uint64_t packetBytes;
    };
    // The packet is the largest power of two up to 16 that divides the step bytes and the plans' offset alignment
    // (the tensors are allocation aligned). The 3-byte steps make 15-byte rows, shorter than the 16-byte packets that
    // byte-granular windows otherwise move, so those are 1-byte packets.
    constexpr std::array<Layout, 5> layouts{{{32, 16, 16}, {24, 8, 8}, {12, 4, 4}, {6, 2, 2}, {3, 1, 1}}};
    constexpr uint64_t windowLength = 5;
    for (const Layout layout : layouts) {
        for (const uint64_t packets : packetCountsOfEveryDepth) {
            PayloadCase testCase;
            testCase.padElement = {0xa7};
            testCase.padValue = 0xa7;
            testCase.stepBytes = layout.stepBytes;
            testCase.windowLength = windowLength;
            testCase.logicalRows = packets / (windowLength * layout.stepBytes / layout.packetBytes);
            testCase.sourceOffsetAlignmentBytes = layout.sourceOffsetAlignmentBytes;
            runPayloadCase(testCase);
        }
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, EveryStepLengthUpToThirtyThreeBytesMaterializesExactly) {
    REQUIRE_CUDA_DEVICE();

    for (uint64_t stepBytes = 1; stepBytes <= 33; ++stepBytes) {
        // Plans at any byte of the source, and plans on the widest boundary the step allows.
        const uint64_t stepBits = stepBytes | 16;
        for (const uint64_t offsetAlignmentBytes : {uint64_t{1}, stepBits & (~stepBits + 1)}) {
            PayloadCase testCase;
            testCase.padElement = {0x5e};
            testCase.padValue = 0x5e;
            testCase.stepBytes = stepBytes;
            testCase.windowLength = 7;
            testCase.logicalRows = 300;
            testCase.sourceOffsetAlignmentBytes = offsetAlignmentBytes;
            runPayloadCase(testCase);
        }
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, ByteStepWindowsMaterializeExactlyAtEveryRowLength) {
    REQUIRE_CUDA_DEVICE();

    // One-byte steps (UINT8 and FP8_E5M2 sequences) start at any byte of the source, and two-byte steps (FP16) at any
    // even byte. A batch of up to 256 KiB moves in packets of the step's width; a larger one moves 16 bytes a packet:
    // rows shorter than a packet, a whole number of packets, and neither; batches that end on a packet and in single
    // bytes; one row, and enough rows for many blocks, on both sides of that bound for every row length.
    struct Element {
        DataType dataType;
        double padValue;
        std::vector<uint8_t> bytes;
    };
    const std::vector<Element> elements{
        {DataType::UINT8, 167.0, bytesOf<uint8_t>(167)},
        {DataType::FP8_E5M2, 1.5, bytesOf(ThorLowPrecision::doubleToFp8E5M2Bits(1.5))},
        {DataType::FP16, 1.5, bytesOf(ThorLowPrecision::doubleToFp16Bits(1.5))},
    };
    for (const Element &element : elements) {
        const uint64_t stepBytes = element.bytes.size();
        for (const uint64_t windowLength : {8ULL, 9ULL, 15ULL, 16ULL, 17ULL, 31ULL, 32ULL, 33ULL, 100ULL, 255ULL, 256ULL, 257ULL, 1024ULL, 1025ULL}) {
            if (windowLength * stepBytes < 16) continue;
            for (const uint64_t logicalRows : {1ULL, 2ULL, 3ULL, 299ULL, 300ULL, 5000ULL, 20000ULL}) {
                // 20000 rows for the rows that 5000 do not take past the bound.
                if (logicalRows == 20000 && 5000 * windowLength * stepBytes > 256 * 1024) continue;
                PayloadCase testCase;
                testCase.dataType = element.dataType;
                testCase.padElement = element.bytes;
                testCase.padValue = element.padValue;
                testCase.stepBytes = stepBytes;
                testCase.windowLength = windowLength;
                testCase.logicalRows = logicalRows;
                testCase.sourceOffsetAlignmentBytes = stepBytes;
                runPayloadCase(testCase);
            }
        }
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, PaddingRepeatsTheElementForEveryElementWidthAndPacketWidth) {
    REQUIRE_CUDA_DEVICE();

    struct Element {
        DataType dataType;
        double padValue;
        std::vector<uint8_t> bytes;
    };
    const std::vector<Element> elements{
        {DataType::UINT8, 167.0, bytesOf<uint8_t>(167)},
        {DataType::FP8_E5M2, 1.5, bytesOf(ThorLowPrecision::doubleToFp8E5M2Bits(1.5))},
        {DataType::UINT16, 4660.0, bytesOf<uint16_t>(4660)},
        {DataType::INT16, -2.0, bytesOf<int16_t>(-2)},
        {DataType::FP16, 1.5, bytesOf(ThorLowPrecision::doubleToFp16Bits(1.5))},
        {DataType::UINT32, 2309737967.0, bytesOf<uint32_t>(2309737967U)},
        {DataType::FP32, -1.5, bytesOf<float>(-1.5F)},
        {DataType::UINT64, 8987183256397123.0, bytesOf<uint64_t>(8987183256397123ULL)},
        {DataType::FP64, 2.75, bytesOf<double>(2.75)},
    };
    // A 16-byte step holds whole elements of every width, and the plans' offset alignment is then the packet width:
    // packets narrower than the element, as wide and wider. 200 rows are a small batch and 2000 a large one, which
    // moves in 16-byte packets where the plans would give 1 or 2 bytes.
    for (const Element &element : elements) {
        for (const uint64_t packetBytes : {1ULL, 2ULL, 4ULL, 8ULL, 16ULL}) {
            for (const uint64_t logicalRows : {200ULL, 2000ULL}) {
                PayloadCase testCase;
                testCase.dataType = element.dataType;
                testCase.padElement = element.bytes;
                testCase.padValue = element.padValue;
                testCase.stepBytes = 16;
                testCase.windowLength = 9;
                testCase.logicalRows = logicalRows;
                testCase.sourceOffsetAlignmentBytes = packetBytes;
                runPayloadCase(testCase);
            }
        }
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, LongWindowsMaterializeExactly) {
    REQUIRE_CUDA_DEVICE();

    struct LaunchCase {
        uint64_t stepBytes;
        uint64_t windowLength;
        uint64_t sourceOffsetAlignmentBytes;
    };
    constexpr std::array<LaunchCase, 6> cases{{{1, 4097, 1}, {1, 4096, 1}, {16, 256, 16}, {4, 1025, 4}, {2, 513, 2}, {32, 33, 8}}};
    for (const LaunchCase launchCase : cases) {
        PayloadCase testCase;
        testCase.padElement = {0x3b};
        testCase.padValue = 0x3b;
        testCase.stepBytes = launchCase.stepBytes;
        testCase.windowLength = launchCase.windowLength;
        testCase.logicalRows = 96;
        testCase.sourceOffsetAlignmentBytes = launchCase.sourceOffsetAlignmentBytes;
        runPayloadCase(testCase);
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, SourceSmallerThanAStepPadsEveryRow) {
    REQUIRE_CUDA_DEVICE();

    // No plan can take a step of such a source, so every row is padding; the source is still a tensor of a few bytes.
    for (const uint64_t sourceBytes : {1ULL, 2ULL, 5ULL, 15ULL}) {
        PayloadCase testCase;
        testCase.padElement = {0x91};
        testCase.padValue = 0x91;
        testCase.stepBytes = 16;
        testCase.windowLength = 6;
        testCase.logicalRows = 300;
        testCase.sourceOffsetAlignmentBytes = 16;
        testCase.sourceBytes = sourceBytes;
        runPayloadCase(testCase);
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, EveryPacketWidthMaterializesMaskExactlyAtEveryDepth) {
    REQUIRE_CUDA_DEVICE();

    struct Layout {
        uint64_t windowLength;
        uint64_t packetBytes;
    };
    // A mask packet is the largest power of two up to 16 that is no longer than the window; it need not divide it, so
    // the packets of the 13-, 7- and 3-step windows cross rows and their batches end in a few single bytes.
    constexpr std::array<Layout, 8> layouts{{{48, 16}, {17, 16}, {13, 8}, {8, 8}, {7, 4}, {3, 2}, {2, 2}, {1, 1}}};
    for (const Layout layout : layouts) {
        for (const uint64_t packets : packetCountsOfEveryDepth) {
            runMaskCase(layout.windowLength, packets * layout.packetBytes / layout.windowLength + 1);
        }
    }
}

TEST(DeviceResidentWindowMaterializationKernelTest, EveryWindowLengthUpToSixtyFiveStepsMasksExactly) {
    REQUIRE_CUDA_DEVICE();

    // 300 rows end on a packet boundary for more window lengths than 299 rows, which mostly leave single bytes.
    for (uint64_t windowLength = 1; windowLength <= 65; ++windowLength) {
        runMaskCase(windowLength, 299);
        runMaskCase(windowLength, 300);
    }
    for (const uint64_t windowLength : {255ULL, 256ULL, 1000ULL, 1025ULL, 4097ULL}) runMaskCase(windowLength, 64);
}

TEST(DeviceResidentWindowMaterializationKernelTest, ZeroLogicalRowsWriteNothing) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batchCapacity = 3;
    constexpr uint64_t windowLength = 4;
    constexpr uint8_t sentinel = 0xcdU;
    Stream stream(0);
    Tensor sourceDevice = makeGpuTensor<uint8_t>({windowLength}, {1, 2, 3, 4}, stream);
    Tensor planDevice = makeGpuPlansAsWords({DeviceResidentWindowRowPlan32{}}, stream);
    const std::vector<uint8_t> untouched(batchCapacity * windowLength, sentinel);
    for (const bool materializeMask : {false, true}) {
        Tensor destination = makeGpuTensor<uint8_t>({batchCapacity, windowLength}, untouched, stream);
        DeviceResidentWindowMaterializationSpec spec;
        spec.dataType = DataType::UINT8;
        spec.windowLength = windowLength;
        spec.sourceStepBytes = 1;
        spec.materializeMask = materializeMask;
        launchDeviceResidentWindowMaterializationKernel(
            sourceDevice, TensorSpan(planDevice, 0, 0), /*logicalRows=*/0, spec, destination, stream);
        stream.synchronize();
        EXPECT_EQ(copyGpuTensor<uint8_t>(destination, stream), untouched);
    }
}

}  // namespace
