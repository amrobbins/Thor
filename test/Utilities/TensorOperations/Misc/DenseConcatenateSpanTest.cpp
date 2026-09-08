#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cstdint>
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
            GTEST_SKIP() << "CUDA device is required for dense Concatenate/Split kernel tests.";                        \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t> &dimensions, const std::vector<T> &values, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size())
        throw std::runtime_error("Dense concatenate test tensor value count mismatch.");
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
    explicit DeviceAllocation(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&pointer, bytes)); }
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
    void *get() const { return pointer; }

   private:
    void *pointer = nullptr;
};

template <typename T>
DeviceAllocation uploadArray(const std::vector<T> &values, Stream &stream) {
    DeviceAllocation allocation(values.size() * sizeof(T));
    CUDA_CHECK(cudaMemcpyAsync(allocation.get(), values.data(), values.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream.getStream()));
    stream.synchronize();
    return allocation;
}

DeviceAllocation makePointerTable(const std::vector<void *> &pointers, Stream &stream) {
    DeviceAllocation allocation(pointers.size() * sizeof(void *));
    CUDA_CHECK(cudaMemcpyAsync(allocation.get(), pointers.data(), pointers.size() * sizeof(void *),
                               cudaMemcpyHostToDevice, stream.getStream()));
    stream.synchronize();
    return allocation;
}

void runBytePayloadCase(uint64_t outerSlices, uint64_t spanBytes) {
    ASSERT_GT(outerSlices, 0U);
    ASSERT_GT(spanBytes, 0U);
    constexpr uint8_t leftValue = 0x31;
    constexpr uint8_t rightValue = 0xA7;
    constexpr uint8_t sentinel = 0xDC;

    Stream stream(0);
    Tensor left = makeGpuTensor<uint8_t>(
        {outerSlices * spanBytes}, std::vector<uint8_t>(outerSlices * spanBytes, leftValue), stream);
    Tensor right = makeGpuTensor<uint8_t>(
        {outerSlices * spanBytes}, std::vector<uint8_t>(outerSlices * spanBytes, rightValue), stream);
    Tensor joined = makeGpuTensor<uint8_t>(
        {outerSlices * 2 * spanBytes}, std::vector<uint8_t>(outerSlices * 2 * spanBytes, sentinel), stream);
    DeviceAllocation sourceTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    const std::vector<ConcatenateSpanGeometry> geometryHost = buildConcatenateSpanGeometry(
        sizeof(uint8_t), 1, {spanBytes, spanBytes});
    DeviceAllocation geometry = uploadArray(geometryHost, stream);

    launchConcatenate(joined.getMemPtr(),
                      reinterpret_cast<void **>(sourceTable.get()),
                      outerSlices,
                      2,
                      2 * spanBytes,
                      static_cast<ConcatenateSpanGeometry *>(geometry.get()),
                      stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(joined, stream);
    EXPECT_EQ(actual.front(), leftValue);
    EXPECT_EQ(actual[spanBytes - 1], leftValue);
    EXPECT_EQ(actual[spanBytes], rightValue);
    EXPECT_EQ(actual[2 * spanBytes - 1], rightValue);
    const uint64_t finalBase = (outerSlices - 1) * 2 * spanBytes;
    EXPECT_EQ(actual[finalBase], leftValue);
    EXPECT_EQ(actual[finalBase + spanBytes - 1], leftValue);
    EXPECT_EQ(actual[finalBase + spanBytes], rightValue);
    EXPECT_EQ(actual[finalBase + 2 * spanBytes - 1], rightValue);
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
    const uint64_t logicalBytes = alignmentOffset + spanBytes;
    const uint64_t destinationBytes = spanBytes + copyWidth;
    constexpr uint8_t sentinel = 0xD7;

    std::vector<uint8_t> sourceValues(logicalBytes);
    for (uint64_t i = 0; i < logicalBytes; ++i)
        sourceValues[i] = static_cast<uint8_t>((37U * i + 11U) & 0xFFU);

    Stream stream(0);
    Tensor source = makeGpuTensor<uint8_t>({logicalBytes}, sourceValues, stream);
    Tensor forwardDestination = makeGpuTensor<uint8_t>(
        {alignmentOffset + destinationBytes},
        std::vector<uint8_t>(alignmentOffset + destinationBytes, sentinel),
        stream);

    auto *sourceStart = static_cast<uint8_t *>(source.getMemPtr()) + alignmentOffset;
    auto *forwardStart = static_cast<uint8_t *>(forwardDestination.getMemPtr()) + alignmentOffset;
    DeviceAllocation sourceTable = makePointerTable({sourceStart, sourceStart}, stream);
    const std::vector<ConcatenateSpanGeometry> geometryHost = {
        {spanBytes, 0},
        {0, spanBytes},
    };
    DeviceAllocation geometry = uploadArray(geometryHost, stream);

    launchConcatenate(forwardStart,
                      reinterpret_cast<void **>(sourceTable.get()),
                      1,
                      2,
                      spanBytes,
                      static_cast<ConcatenateSpanGeometry *>(geometry.get()),
                      stream);
    stream.synchronize();

    const std::vector<uint8_t> forwardValues = copyGpuTensor<uint8_t>(forwardDestination, stream);
    for (uint64_t i = 0; i < spanBytes; ++i)
        EXPECT_EQ(forwardValues[alignmentOffset + i], sourceValues[alignmentOffset + i]);
    for (uint64_t i = spanBytes; i < destinationBytes; ++i)
        EXPECT_EQ(forwardValues[alignmentOffset + i], sentinel) << "forward tail over-store byte=" << i;

    Tensor splitDestination = makeGpuTensor<uint8_t>(
        {alignmentOffset + destinationBytes},
        std::vector<uint8_t>(alignmentOffset + destinationBytes, sentinel),
        stream);
    auto *splitStart = static_cast<uint8_t *>(splitDestination.getMemPtr()) + alignmentOffset;
    DeviceAllocation destinationTable = makePointerTable({splitStart, nullptr}, stream);
    launchSplit(reinterpret_cast<void **>(destinationTable.get()),
                sourceStart,
                1,
                2,
                spanBytes,
                static_cast<ConcatenateSpanGeometry *>(geometry.get()),
                stream);
    stream.synchronize();

    const std::vector<uint8_t> splitValues = copyGpuTensor<uint8_t>(splitDestination, stream);
    for (uint64_t i = 0; i < spanBytes; ++i)
        EXPECT_EQ(splitValues[alignmentOffset + i], sourceValues[alignmentOffset + i]);
    for (uint64_t i = spanBytes; i < destinationBytes; ++i)
        EXPECT_EQ(splitValues[alignmentOffset + i], sentinel) << "split tail over-store byte=" << i;
}

}  // namespace

TEST(DenseConcatenateSpan, StaticGeometryRepresentsContiguousAxisSpans) {
    const std::vector<ConcatenateSpanGeometry> geometry =
        buildConcatenateSpanGeometry(/*elementSizeBytes=*/2, /*innerElements=*/3, {2, 5, 1});
    ASSERT_EQ(geometry.size(), 3U);
    EXPECT_EQ(geometry[0].spanBytes, 12U);
    EXPECT_EQ(geometry[0].packedOffsetBytes, 0U);
    EXPECT_EQ(geometry[1].spanBytes, 30U);
    EXPECT_EQ(geometry[1].packedOffsetBytes, 12U);
    EXPECT_EQ(geometry[2].spanBytes, 6U);
    EXPECT_EQ(geometry[2].packedOffsetBytes, 42U);
}

TEST(DenseConcatenateSpan, ForwardAndSplitTouchOnlyRequestedOuterPrefix) {
    REQUIRE_CUDA_DEVICE();
    constexpr uint64_t outerCapacity = 24;
    constexpr uint64_t activeOuterSlices = 15;
    constexpr uint64_t leftBytes = 5;
    constexpr uint64_t rightBytes = 7;
    constexpr uint64_t packedBytes = leftBytes + rightBytes;
    constexpr uint8_t inactivePoison = 0xEE;
    constexpr uint8_t outputSentinel = 0x91;
    constexpr uint8_t splitSentinel = 0xA3;

    std::vector<uint8_t> leftValues(outerCapacity * leftBytes, inactivePoison);
    std::vector<uint8_t> rightValues(outerCapacity * rightBytes, inactivePoison);
    for (uint64_t outer = 0; outer < activeOuterSlices; ++outer) {
        for (uint64_t byte = 0; byte < leftBytes; ++byte)
            leftValues[outer * leftBytes + byte] = static_cast<uint8_t>(outer + byte + 1);
        for (uint64_t byte = 0; byte < rightBytes; ++byte)
            rightValues[outer * rightBytes + byte] = static_cast<uint8_t>(100 + outer + byte);
    }

    Stream stream(0);
    Tensor left = makeGpuTensor<uint8_t>({outerCapacity * leftBytes}, leftValues, stream);
    Tensor right = makeGpuTensor<uint8_t>({outerCapacity * rightBytes}, rightValues, stream);
    Tensor joined = makeGpuTensor<uint8_t>(
        {outerCapacity * packedBytes}, std::vector<uint8_t>(outerCapacity * packedBytes, outputSentinel), stream);
    DeviceAllocation sourceTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    const auto geometryHost = buildConcatenateSpanGeometry(sizeof(uint8_t), 1, {leftBytes, rightBytes});
    DeviceAllocation geometry = uploadArray(geometryHost, stream);

    launchConcatenate(joined.getMemPtr(), reinterpret_cast<void **>(sourceTable.get()),
                      activeOuterSlices, 2, packedBytes,
                      static_cast<ConcatenateSpanGeometry *>(geometry.get()), stream);
    stream.synchronize();

    const std::vector<uint8_t> joinedValues = copyGpuTensor<uint8_t>(joined, stream);
    for (uint64_t outer = 0; outer < activeOuterSlices; ++outer) {
        for (uint64_t byte = 0; byte < leftBytes; ++byte)
            EXPECT_EQ(joinedValues[outer * packedBytes + byte], leftValues[outer * leftBytes + byte]);
        for (uint64_t byte = 0; byte < rightBytes; ++byte)
            EXPECT_EQ(joinedValues[outer * packedBytes + leftBytes + byte], rightValues[outer * rightBytes + byte]);
    }
    for (uint64_t i = activeOuterSlices * packedBytes; i < joinedValues.size(); ++i)
        EXPECT_EQ(joinedValues[i], outputSentinel) << "inactive output byte=" << i;

    Tensor leftSplit = makeGpuTensor<uint8_t>(
        {outerCapacity * leftBytes}, std::vector<uint8_t>(outerCapacity * leftBytes, splitSentinel), stream);
    Tensor rightSplit = makeGpuTensor<uint8_t>(
        {outerCapacity * rightBytes}, std::vector<uint8_t>(outerCapacity * rightBytes, splitSentinel), stream);
    DeviceAllocation destinationTable = makePointerTable({leftSplit.getMemPtr(), rightSplit.getMemPtr()}, stream);
    launchSplit(reinterpret_cast<void **>(destinationTable.get()), joined.getMemPtr(),
                activeOuterSlices, 2, packedBytes,
                static_cast<ConcatenateSpanGeometry *>(geometry.get()), stream);
    stream.synchronize();

    const std::vector<uint8_t> actualLeft = copyGpuTensor<uint8_t>(leftSplit, stream);
    const std::vector<uint8_t> actualRight = copyGpuTensor<uint8_t>(rightSplit, stream);
    for (uint64_t i = 0; i < activeOuterSlices * leftBytes; ++i) EXPECT_EQ(actualLeft[i], leftValues[i]);
    for (uint64_t i = activeOuterSlices * leftBytes; i < actualLeft.size(); ++i) EXPECT_EQ(actualLeft[i], splitSentinel);
    for (uint64_t i = 0; i < activeOuterSlices * rightBytes; ++i) EXPECT_EQ(actualRight[i], rightValues[i]);
    for (uint64_t i = activeOuterSlices * rightBytes; i < actualRight.size(); ++i) EXPECT_EQ(actualRight[i], splitSentinel);
}

TEST(DenseConcatenateSpan, PayloadAwareGroupingCoversFullOneThrough256SpecializationLadder) {
    REQUIRE_CUDA_DEVICE();
    // Two spans per outer slice. Tiny spans permit the full 256-spans/CTA
    // payload choice; these outer counts force each parallelism-floor rung.
    for (const uint64_t outerSlices : std::vector<uint64_t>{
             32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
         }) {
        runBytePayloadCase(outerSlices, 1);
    }
}

TEST(DenseConcatenateSpan, PayloadThresholdTransitionsUseNextWiderSpanGrouping) {
    REQUIRE_CUDA_DEVICE();
    struct LaunchCase {
        uint64_t spanBytes;
        uint64_t outerSlices;
    };
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
        runBytePayloadCase(launchCase.outerSlices, launchCase.spanBytes);
    }
}

TEST(DenseConcatenateSpan, AlignedBulkCopiesUseExactTailForEverySupportedWidthAndTailSize) {
    REQUIRE_CUDA_DEVICE();
    for (const uint32_t copyWidth : {32U, 16U, 8U, 4U, 2U}) {
        for (uint32_t tailBytes = 1; tailBytes < copyWidth; ++tailBytes)
            runExactTailCase(copyWidth, tailBytes);
    }
}
