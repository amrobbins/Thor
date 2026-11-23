#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/DataTypeConversions/TypeConverter.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
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

Tensor makeGpuBytes(const std::vector<uint8_t> &values, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(values.size())}));
    uint8_t *hostValues = host.getMemPtr<uint8_t>();
    for (size_t i = 0; i < values.size(); ++i) hostValues[i] = values[i];
    Tensor device(gpuPlacement, host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

std::vector<uint8_t> copyGpuBytes(const Tensor &device, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT8, device.getDimensions()));
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const uint8_t *hostValues = host.getMemPtr<uint8_t>();
    return std::vector<uint8_t>(hostValues, hostValues + host.getTotalNumElements());
}

uint8_t patternByte(uint64_t array, uint64_t byte) {
    return static_cast<uint8_t>((37U * byte + 11U * array + 5U) ^ (byte >> 7));
}

// One geometry both ways, every byte checked against the definition: the arrays' active slices concatenated into a
// packed tensor of outerCapacity slices (the rest untouched; an array with a null source is zeros), then the packed
// tensor split into fresh arrays (those with a null destination, and the inactive slices, untouched). Every store
// starts offsetBytes past a tensor's 256-byte boundary, so unaligned bases are covered too.
struct ReferenceCase {
    std::string name;
    uint64_t outerSlices = 1;
    std::vector<uint64_t> spanBytes;
    uint64_t packedOffsetBytes = 0;
    std::vector<uint64_t> splitOffsetBytes;   // empty: all zero
    std::vector<bool> nullSplitDestination;   // empty: none
    uint64_t inactiveSlices = 2;
    std::vector<bool> nullSource;             // concatenate: empty, none
};

void runReferenceCase(const ReferenceCase &testCase) {
    SCOPED_TRACE(testCase.name);
    constexpr uint8_t sentinel = 0xDC;
    const size_t arrays = testCase.spanBytes.size();
    const uint64_t outerCapacity = testCase.outerSlices + testCase.inactiveSlices;
    // Bytes as UINT8 elements: every array has the packed tensor's type, so the launches are the byte kernels.
    const std::vector<DataType> types(arrays, DataType::UINT8);
    std::vector<uint64_t> packedOffsetBytes(arrays, 0);
    uint64_t packedSliceBytes = 0;
    for (size_t a = 0; a < arrays; ++a) {
        packedOffsetBytes[a] = packedSliceBytes;
        packedSliceBytes += testCase.spanBytes[a];
    }
    std::vector<uint64_t> splitOffsets = testCase.splitOffsetBytes;
    splitOffsets.resize(arrays, 0);
    std::vector<bool> nullDestination = testCase.nullSplitDestination;
    nullDestination.resize(arrays, false);
    std::vector<bool> nullSource = testCase.nullSource;
    nullSource.resize(arrays, false);

    Stream stream(0);
    std::vector<Tensor> sources;
    std::vector<void *> sourcePointers;
    std::vector<std::vector<uint8_t>> sourceValues(arrays);
    for (size_t a = 0; a < arrays; ++a) {
        sourceValues[a].resize(outerCapacity * testCase.spanBytes[a]);
        for (uint64_t byte = 0; byte < sourceValues[a].size(); ++byte) sourceValues[a][byte] = patternByte(a, byte);
        std::vector<uint8_t> withOffset(splitOffsets[a], sentinel);
        withOffset.insert(withOffset.end(), sourceValues[a].begin(), sourceValues[a].end());
        withOffset.resize(std::max<size_t>(withOffset.size(), 16), sentinel);  // a tensor of no bytes is not allocated
        sources.push_back(makeGpuBytes(withOffset, stream));
        sourcePointers.push_back(testCase.spanBytes[a] == 0 || nullSource[a] ? nullptr : sources.back().getMemPtr<uint8_t>() + splitOffsets[a]);
    }
    Tensor packed = makeGpuBytes(std::vector<uint8_t>(testCase.packedOffsetBytes + outerCapacity * packedSliceBytes, sentinel), stream);
    uint8_t *const packedStart = packed.getMemPtr<uint8_t>() + testCase.packedOffsetBytes;

    launchConcatenate(packedStart, DataType::UINT8, sourcePointers, types, testCase.outerSlices, testCase.spanBytes, stream);
    stream.synchronize();
    const std::vector<uint8_t> packedValues = copyGpuBytes(packed, stream);
    uint64_t wrongBytes = 0;
    for (uint64_t byte = 0; byte < testCase.packedOffsetBytes; ++byte) wrongBytes += packedValues[byte] != sentinel;
    for (uint64_t slice = 0; slice < outerCapacity; ++slice) {
        for (size_t a = 0; a < arrays; ++a) {
            for (uint64_t byte = 0; byte < testCase.spanBytes[a]; ++byte) {
                const uint8_t active = nullSource[a] ? 0 : sourceValues[a][slice * testCase.spanBytes[a] + byte];
                const uint8_t expected = slice < testCase.outerSlices ? active : sentinel;
                wrongBytes += packedValues[testCase.packedOffsetBytes + slice * packedSliceBytes + packedOffsetBytes[a] + byte] != expected;
            }
        }
    }
    EXPECT_EQ(wrongBytes, 0U) << "concatenate";

    std::vector<Tensor> destinations;
    std::vector<void *> destinationPointers;
    for (size_t a = 0; a < arrays; ++a) {
        destinations.push_back(makeGpuBytes(
            std::vector<uint8_t>(std::max<uint64_t>(16, splitOffsets[a] + outerCapacity * testCase.spanBytes[a]), sentinel), stream));
        destinationPointers.push_back(nullDestination[a] ? nullptr : destinations.back().getMemPtr<uint8_t>() + splitOffsets[a]);
    }
    launchSplit(destinationPointers, types, packedStart, DataType::UINT8, testCase.outerSlices, testCase.spanBytes, stream);
    stream.synchronize();
    wrongBytes = 0;
    for (size_t a = 0; a < arrays; ++a) {
        const std::vector<uint8_t> destinationValues = copyGpuBytes(destinations[a], stream);
        const uint64_t activeBytes = nullDestination[a] ? 0 : testCase.outerSlices * testCase.spanBytes[a];
        for (uint64_t byte = 0; byte < destinationValues.size(); ++byte) {
            const bool active = byte >= splitOffsets[a] && byte - splitOffsets[a] < activeBytes;
            const uint8_t expected = active ? (nullSource[a] ? 0 : sourceValues[a][byte - splitOffsets[a]]) : sentinel;
            wrongBytes += destinationValues[byte] != expected;
        }
    }
    EXPECT_EQ(wrongBytes, 0U) << "split";
}

// The typed concatenate and split, whose arrays are of any element types, against what they replace: every array of
// another type than the packed tensor's converted by TypeConverter to (or from) a scratch array of the packed type,
// around the byte concatenate (or split). The typed launch must give those bytes, and leave the inactive slices, the
// bytes ahead of every store and the arrays with a null destination untouched. The values are random bits (0 and 1
// for BOOLEAN, whose other bytes are not values), so every float format's specials and both signs are among them.
const std::vector<DataType> kElementTypes = {DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::BF16,   DataType::FP16,   DataType::FP32,
                                             DataType::FP64,     DataType::INT8,     DataType::INT16,  DataType::INT32,  DataType::INT64,
                                             DataType::UINT8,    DataType::UINT16,   DataType::UINT32, DataType::UINT64, DataType::BOOLEAN};

uint64_t elementBytes(DataType dataType) { return static_cast<uint64_t>(TensorDescriptor::getElementSizeInBytes(dataType)); }

std::vector<uint8_t> randomElements(DataType dataType, uint64_t elements, uint64_t seed) {
    std::vector<uint8_t> bytes(elements * elementBytes(dataType));
    for (uint64_t byte = 0; byte < bytes.size(); ++byte) {
        uint64_t z = (seed * 0x100000001b3ULL + byte) * 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        bytes[byte] = static_cast<uint8_t>(z >> 40U);
        if (dataType == DataType::BOOLEAN) bytes[byte] &= 1U;
    }
    return bytes;
}

// values after `ahead` sentinel bytes, at least 16 bytes in all (a tensor of no bytes is not allocated).
Tensor makeGpuStore(uint64_t ahead, const std::vector<uint8_t> &values, uint8_t sentinel, Stream &stream) {
    std::vector<uint8_t> bytes(ahead, sentinel);
    bytes.insert(bytes.end(), values.begin(), values.end());
    bytes.resize(std::max<size_t>(bytes.size(), 16), sentinel);
    return makeGpuBytes(bytes, stream);
}

struct TypedCase {
    std::string name;
    uint64_t outerSlices = 1;
    DataType packedType = DataType::FP32;
    std::vector<DataType> types;
    std::vector<uint64_t> spanElements;
    // Every store starts this many of its own elements past a tensor's boundary.
    uint64_t packedOffsetElements = 0;
    std::vector<uint64_t> splitOffsetElements;   // empty: all zero
    std::vector<bool> nullSplitDestination;      // empty: none
    uint64_t inactiveSlices = 2;
    std::vector<bool> nullSource;                // concatenate: empty, none (a null source is zeros of the packed type)
};

void runTypedCase(const TypedCase &testCase) {
    SCOPED_TRACE(testCase.name);
    constexpr uint8_t sentinel = 0xDC;
    const size_t arrays = testCase.spanElements.size();
    ASSERT_EQ(testCase.types.size(), arrays);
    const uint64_t outerCapacity = testCase.outerSlices + testCase.inactiveSlices;
    const uint64_t packedElementBytes = elementBytes(testCase.packedType);
    uint64_t packedSliceElements = 0;
    for (const uint64_t span : testCase.spanElements) packedSliceElements += span;
    const uint64_t packedSliceBytes = packedSliceElements * packedElementBytes;
    // The reference's concatenate and split: every array of the packed type, so the byte kernels.
    const std::vector<DataType> packedTypes(arrays, testCase.packedType);
    const uint64_t packedAhead = testCase.packedOffsetElements * packedElementBytes;
    std::vector<uint64_t> splitOffsets = testCase.splitOffsetElements;
    splitOffsets.resize(arrays, 0);
    std::vector<bool> nullDestination = testCase.nullSplitDestination;
    nullDestination.resize(arrays, false);
    std::vector<bool> nullSource = testCase.nullSource;
    nullSource.resize(arrays, false);
    Stream stream(0);

    {
        std::vector<Tensor> stores;
        std::vector<void *> typedSources;
        std::vector<void *> referenceSources;
        for (size_t a = 0; a < arrays; ++a) {
            const uint64_t arrayElementBytes = elementBytes(testCase.types[a]);
            const uint64_t span = testCase.spanElements[a];
            if (nullSource[a] && span != 0) {
                // The reference reads zeros of the packed type for it.
                typedSources.push_back(nullptr);
                stores.push_back(makeGpuStore(0, std::vector<uint8_t>(outerCapacity * span * packedElementBytes, 0), sentinel, stream));
                referenceSources.push_back(stores.back().getMemPtr<uint8_t>());
                continue;
            }
            const std::vector<uint8_t> values = randomElements(testCase.types[a], outerCapacity * span, 1000 + a);
            stores.push_back(makeGpuStore(splitOffsets[a] * arrayElementBytes, values, sentinel, stream));
            typedSources.push_back(span == 0 ? nullptr : stores.back().getMemPtr<uint8_t>() + splitOffsets[a] * arrayElementBytes);
            if (span == 0) {
                referenceSources.push_back(nullptr);
                continue;
            }
            // TypeConverter takes a tensor's alignment, so the reference converts a copy of the values at a tensor's start.
            stores.push_back(makeGpuStore(0, values, sentinel, stream));
            void *const alignedValues = stores.back().getMemPtr<uint8_t>();
            if (testCase.types[a] == testCase.packedType) {
                referenceSources.push_back(alignedValues);
                continue;
            }
            stores.push_back(makeGpuStore(0, std::vector<uint8_t>(testCase.outerSlices * span * packedElementBytes, sentinel), sentinel, stream));
            TypeConverter::convertType(alignedValues, stores.back().getMemPtr<uint8_t>(), testCase.types[a], testCase.packedType,
                                       static_cast<long>(testCase.outerSlices * span), stream, 0);
            referenceSources.push_back(stores.back().getMemPtr<uint8_t>());
        }
        const std::vector<uint8_t> untouched(outerCapacity * packedSliceBytes, sentinel);
        Tensor typedPacked = makeGpuStore(packedAhead, untouched, sentinel, stream);
        Tensor referencePacked = makeGpuStore(0, untouched, sentinel, stream);
        launchConcatenate(typedPacked.getMemPtr<uint8_t>() + packedAhead, testCase.packedType, typedSources, testCase.types, testCase.outerSlices,
                          testCase.spanElements, stream);
        launchConcatenate(referencePacked.getMemPtr<uint8_t>(), testCase.packedType, referenceSources, packedTypes, testCase.outerSlices,
                          testCase.spanElements, stream);
        stream.synchronize();
        const std::vector<uint8_t> typedValues = copyGpuBytes(typedPacked, stream);
        const std::vector<uint8_t> referenceValues = copyGpuBytes(referencePacked, stream);
        uint64_t wrongBytes = 0;
        for (uint64_t byte = 0; byte < packedAhead; ++byte) wrongBytes += typedValues[byte] != sentinel;
        for (uint64_t byte = 0; byte < untouched.size(); ++byte) wrongBytes += typedValues[packedAhead + byte] != referenceValues[byte];
        EXPECT_EQ(wrongBytes, 0U) << "concatenate";
    }

    {
        const std::vector<uint8_t> packedValues = randomElements(testCase.packedType, outerCapacity * packedSliceBytes / packedElementBytes, 77);
        Tensor typedPacked = makeGpuStore(packedAhead, packedValues, sentinel, stream);
        Tensor referencePacked = makeGpuStore(0, packedValues, sentinel, stream);
        std::vector<Tensor> typedStores;
        std::vector<Tensor> scratchStores;
        std::vector<Tensor> referenceStores;
        std::vector<void *> typedDestinations;
        std::vector<void *> scratchDestinations;
        for (size_t a = 0; a < arrays; ++a) {
            const uint64_t arrayElementBytes = elementBytes(testCase.types[a]);
            const uint64_t capacityElements = outerCapacity * testCase.spanElements[a];
            typedStores.push_back(
                makeGpuStore(splitOffsets[a] * arrayElementBytes, std::vector<uint8_t>(capacityElements * arrayElementBytes, sentinel), sentinel, stream));
            typedDestinations.push_back(nullDestination[a] ? nullptr : typedStores.back().getMemPtr<uint8_t>() + splitOffsets[a] * arrayElementBytes);
            scratchStores.push_back(makeGpuStore(0, std::vector<uint8_t>(capacityElements * packedElementBytes, sentinel), sentinel, stream));
            scratchDestinations.push_back(nullDestination[a] ? nullptr : scratchStores.back().getMemPtr<uint8_t>());
            referenceStores.push_back(makeGpuStore(0, std::vector<uint8_t>(capacityElements * arrayElementBytes, sentinel), sentinel, stream));
        }
        launchSplit(typedDestinations, testCase.types, typedPacked.getMemPtr<uint8_t>() + packedAhead, testCase.packedType, testCase.outerSlices,
                    testCase.spanElements, stream);
        launchSplit(scratchDestinations, packedTypes, referencePacked.getMemPtr<uint8_t>(), testCase.packedType, testCase.outerSlices,
                    testCase.spanElements, stream);
        for (size_t a = 0; a < arrays; ++a) {
            const uint64_t activeElements = testCase.outerSlices * testCase.spanElements[a];
            if (nullDestination[a] || activeElements == 0) continue;
            if (testCase.types[a] == testCase.packedType) {
                referenceStores[a] = scratchStores[a];
            } else {
                TypeConverter::convertType(scratchStores[a].getMemPtr<uint8_t>(), referenceStores[a].getMemPtr<uint8_t>(), testCase.packedType,
                                           testCase.types[a], static_cast<long>(activeElements), stream, 0);
            }
        }
        stream.synchronize();
        uint64_t wrongBytes = 0;
        for (size_t a = 0; a < arrays; ++a) {
            const uint64_t arrayElementBytes = elementBytes(testCase.types[a]);
            const uint64_t ahead = splitOffsets[a] * arrayElementBytes;
            const uint64_t capacityBytes = outerCapacity * testCase.spanElements[a] * arrayElementBytes;
            const std::vector<uint8_t> typedValues = copyGpuBytes(typedStores[a], stream);
            const std::vector<uint8_t> referenceValues = copyGpuBytes(referenceStores[a], stream);
            for (uint64_t byte = 0; byte < ahead; ++byte) wrongBytes += typedValues[byte] != sentinel;
            for (uint64_t byte = 0; byte < capacityBytes; ++byte) wrongBytes += typedValues[ahead + byte] != referenceValues[byte];
        }
        EXPECT_EQ(wrongBytes, 0U) << "split";
    }
}

}  // namespace

TEST(DenseConcatenateSpan, ForwardAndSplitTouchOnlyRequestedOuterPrefix) {
    REQUIRE_CUDA_DEVICE();
    runReferenceCase({"5 and 7 bytes, 15 of 24 slices", 15, {5, 7}, 0, {}, {}, 9});
}

// Spans and stores on 4-, 8- and 16-byte boundaries move in packets of that width, 1 to 8 per thread by size.
TEST(DenseConcatenateSpan, AlignedSpansMoveInPacketsOfTheirWidth) {
    REQUIRE_CUDA_DEVICE();
    for (const ReferenceCase &testCase : std::vector<ReferenceCase>{
             {"4-byte", 1000, {4, 8}},
             {"8-byte", 1000, {8, 16, 24}},
             {"16-byte", 1000, {32, 64, 16}},
             {"16-byte, one slice", 1, {4096, 65536, 16}},
             {"16-byte, 3 MiB", 100000, {16, 16}},
             {"4-byte, 2.4 MiB", 200000, {4, 8}},
             {"8-byte, 4-byte stores", 2000, {8, 16}, 4, {4, 0}},
         }) {
        runReferenceCase(testCase);
    }
}

// A concatenate and a split take whole slices of the packed tensor per block while a slice is at most 256 packets, with
// idle threads where the slice does not divide 256, and a slice of more as several blocks of 256 packets.
TEST(DenseConcatenateSpan, WholeSlicesUpToABlockOfPackets) {
    REQUIRE_CUDA_DEVICE();
    for (const ReferenceCase &testCase : std::vector<ReferenceCase>{
             {"129 packets a slice: one slice a block, 127 idle threads", 700, {688, 688, 688}},
             {"255 packets a slice", 700, {1360, 1360, 1360}},
             {"256 packets a slice", 700, {2048, 2048}},
             {"258 packets a slice: per array", 700, {1376, 1376, 1376}},
             {"arrays of every width, 15 16-byte packets a slice, 2.2 MiB", 9000, {128, 64, 32, 16, 8, 4, 2, 2}},
             {"129 one-byte packets a slice", 3000, {100, 29}},
             {"256 one-byte packets a slice", 2500, {3, 253}},
             {"257 one-byte packets a slice: per array", 2500, {200, 57}},
             {"8 packets a slice, the last block part full, a null destination", 100001, {16, 16, 16, 16, 16, 16, 16, 16}, 0, {},
              {false, false, true, false, false, false, false, true}},
             {"1 packet a slice", 70000, {16}},
         }) {
        runReferenceCase(testCase);
    }
}

// A slice of more than 256 packets is ceil(packets / 256) blocks, the last part full; the array of a packet is a
// binary search over the arrays' first packets.
TEST(DenseConcatenateSpan, LongSlicesSpanSeveralBlocks) {
    REQUIRE_CUDA_DEVICE();
    for (const ReferenceCase &testCase : std::vector<ReferenceCase>{
             {"258 packets a slice", 700, {1376, 1376, 1376}},
             {"512 packets a slice, 128 MiB shape at 4 MiB", 512, {4096, 4096}},
             {"one array of 8192 bytes between two of 16", 300, {16, 8192, 16}},
             {"307 one-byte packets a slice", 2000, {300, 7}},
             {"16 arrays, 4112-byte ends: every binary-search step", 200,
              {4112, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 4112}},
             {"18 arrays: two launches, the first of 16", 200,
              {4112, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 4112}},
             {"one slice of 64 KiB spans", 1, {65536, 65536, 16}},
             {"long slices, a null destination", 1000, {2048, 2064, 2080}, 0, {}, {false, true, false}},
         }) {
        runReferenceCase(testCase);
    }
}

// Spans of odd lengths move in 1- and 2-byte packets while that spreads them, and in shifted 16-byte packets past that.
TEST(DenseConcatenateSpan, OddSpansMoveInNarrowPacketsThenShiftedPackets) {
    REQUIRE_CUDA_DEVICE();
    for (const ReferenceCase &testCase : std::vector<ReferenceCase>{
             {"1-byte spans", 1024, {1, 1}},
             {"3 and 5", 500, {3, 5}},
             {"33 and 33, 66 KiB", 1024, {33, 33}},
             {"33 and 17, 100 KiB", 2000, {33, 17}},
             {"34 and 18, 2-byte packets", 3000, {34, 18}},
             {"33 and 33, 1 MiB: shifted", 16384, {33, 33}},
             {"33 and 17, 1.4 MiB: shifted, packed tensor ends off a boundary", 30000, {33, 17}},
             {"34 and 34, 2.1 MiB: 2-byte packets then shifted", 32768, {34, 34}},
             {"1025 and 17, 2 MiB: shifted, long rows", 2048, {1025, 17}},
             {"17 and 1024, 2 MiB: shifted beside an aligned span", 2048, {17, 1024}},
             {"3 and 5, 2.4 MiB: spans under a packet stay narrow", 300000, {3, 5}},
             {"33 and 33, 1 MiB, unaligned stores stay narrow", 16384, {33, 33}, 8, {4, 0}},
             {"33 and 33, 1 MiB, one unaligned store stays narrow when concatenating", 16384, {33, 33}, 0, {0, 2}},
         }) {
        runReferenceCase(testCase);
    }
}

// Arrays of different widths are separate launches; concatenating in shifted packets takes every array along.
TEST(DenseConcatenateSpan, MixedWidthsAndManyArrays) {
    REQUIRE_CUDA_DEVICE();
    for (const ReferenceCase &testCase : std::vector<ReferenceCase>{
             {"16, 16, 1, 1 bytes wide", 200, {48, 16, 33, 31}},
             {"16, 16, 1, 1 bytes wide, 2.5 MiB", 20000, {48, 16, 33, 31}},
             {"20 arrays", 100, {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}},
             {"20 arrays, 1.5 MiB", 3000, {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36}},
             {"18 one-byte arrays", 5000, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
             {"zero-byte spans", 100, {0, 16, 0, 32}},
             {"zero-byte last span", 100, {16, 0}},
         }) {
        runReferenceCase(testCase);
    }
}

// A null split destination skips its array; the other arrays and its own bytes are untouched.
TEST(DenseConcatenateSpan, NullSplitDestinationsAreSkipped) {
    REQUIRE_CUDA_DEVICE();
    for (const ReferenceCase &testCase : std::vector<ReferenceCase>{
             {"middle array", 5000, {16, 32, 16}, 0, {}, {false, true, false}},
             {"first array, shifted", 20000, {33, 33, 33}, 0, {}, {true, false, false}},
             {"last two arrays", 3000, {5, 7, 9}, 0, {}, {false, true, true}},
         }) {
        runReferenceCase(testCase);
    }
}

// A null source is zeros: every byte kernel (packets of each width, narrow and shifted packets, long slices), the
// source before, between and after others, next to rows that straddle into it, and every source null.
TEST(DenseConcatenateSpan, NullSourcesAreZeros) {
    REQUIRE_CUDA_DEVICE();
    for (const ReferenceCase &testCase : std::vector<ReferenceCase>{
             {"16-byte packets, the middle source", 1000, {64, 64, 64}, 0, {}, {}, 2, {false, true, false}},
             {"4-byte packets, the first source", 1000, {4, 8}, 0, {}, {}, 2, {true, false}},
             {"narrow packets, odd spans", 300, {5, 7, 3}, 0, {}, {}, 2, {false, true, false}},
             {"narrow, a packed tensor off a boundary", 1000, {40, 24}, 3, {}, {}, 2, {true, false}},
             {"shifted, the middle source", 40000, {33, 35, 33}, 0, {}, {}, 2, {false, true, false}},
             {"shifted, the first source", 40000, {33, 35, 33}, 0, {}, {}, 2, {true, false, false}},
             {"shifted, the last source (the packed tensor's last row)", 40000, {33, 35, 33}, 0, {}, {}, 2, {false, false, true}},
             {"shifted, two sources", 30000, {33, 17, 1025}, 0, {}, {}, 2, {true, false, true}},
             {"long slices", 100, {2048, 4112, 2064}, 0, {}, {}, 2, {true, false, false}},
             {"every source null", 50, {16, 32}, 0, {}, {}, 2, {true, true}},
             {"every source null, shifted", 40000, {33, 35}, 0, {}, {}, 2, {true, true}},
             {"a slice's worth of 1-byte packets", 3000, {100, 29}, 0, {}, {}, 2, {false, true}},
         }) {
        runReferenceCase(testCase);
    }
}

// Every pair of element types, both ways: an array of each of the 15 types beside a packed tensor of each type. The
// spans are of 2 to 64 elements, so the arrays take packets of every size, and one launch holds every type.
TEST(DenseConcatenateTyped, EveryElementTypeConvertsAsTypeConverterDoes) {
    REQUIRE_CUDA_DEVICE();
    const std::vector<uint64_t> spans = {16, 8, 24, 4, 12, 32, 16, 6, 48, 2, 64, 16, 20, 8, 16};
    for (const DataType packedType : kElementTypes) {
        TypedCase testCase;
        testCase.name = "packed type " + std::to_string(static_cast<int>(packedType));
        testCase.outerSlices = 37;
        testCase.packedType = packedType;
        testCase.types = kElementTypes;
        testCase.spanElements = spans;
        runTypedCase(testCase);
    }
}

// The shapes of the grid: slices of a few packets (many slices a block), slices longer than a block (chunks, the
// binary search), one slice, many slices, and FP8 E5M2 on each side.
TEST(DenseConcatenateTyped, ShortAndLongSlices) {
    REQUIRE_CUDA_DEVICE();
    for (const TypedCase &testCase : std::vector<TypedCase>{
             {"FP32 and BF16 into BF16, short slices", 5000, DataType::BF16, {DataType::FP32, DataType::BF16}, {16, 16}},
             {"FP32 and BF16 by turns into BF16", 700, DataType::BF16,
              {DataType::FP32, DataType::BF16, DataType::FP32, DataType::BF16, DataType::FP32, DataType::BF16, DataType::FP32, DataType::BF16},
              {16, 16, 16, 16, 16, 16, 16, 16}},
             {"FP32 into E5M2", 700, DataType::FP8_E5M2, {DataType::FP32, DataType::FP32}, {128, 128}},
             {"E5M2 into FP32", 700, DataType::FP32, {DataType::FP8_E5M2, DataType::FP8_E5M2}, {128, 128}},
             {"E4M3 and BF16 into BF16, one slice", 1, DataType::BF16, {DataType::FP8_E4M3, DataType::BF16}, {16, 16}},
             {"one-byte types, packets of 16 elements", 400, DataType::FP8_E5M2, {DataType::FP8_E4M3, DataType::INT8, DataType::FP8_E5M2}, {32, 64, 16}},
             {"long slices of three types", 9, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E5M2}, {4096, 1000, 24}},
             {"a long slice, one array", 300, DataType::FP16, {DataType::FP64}, {1024}},
             {"UINT8 and FP32 into FP32, long", 64, DataType::FP32, {DataType::UINT8, DataType::FP32}, {1024, 1024}},
             {"a slice of exactly a block", 33, DataType::FP32, {DataType::BF16, DataType::INT32}, {512, 512}},
             {"a slice a packet over a block", 33, DataType::FP32, {DataType::BF16, DataType::INT32}, {512, 516}},
         }) {
        runTypedCase(testCase);
    }
}

// Launches past the spread bound (256 blocks of one packet a thread) take four packets per thread, in slices a
// block's columns apart: the last block's threads have fewer slices than packets, whole packets and smaller ones,
// short slices and long; and the launches on each side of the bound.
TEST(DenseConcatenateTyped, LargeLaunchesTakeFourPacketsPerThread) {
    REQUIRE_CUDA_DEVICE();
    for (const TypedCase &testCase : std::vector<TypedCase>{
             {"256 blocks of one packet", 8192, DataType::BF16, {DataType::FP32, DataType::FP32}, {16, 16}},
             {"257 blocks of one packet, 65 of four", 8193, DataType::BF16, {DataType::FP32, DataType::FP32}, {16, 16}},
             {"513 blocks of one packet, 129 of four", 16385, DataType::FP32, {DataType::INT32, DataType::FP8_E5M2}, {16, 16}},
             {"FP32 and BF16 into BF16", 60001, DataType::BF16, {DataType::FP32, DataType::BF16}, {16, 16}},
             {"FP32 into E5M2", 4001, DataType::FP8_E5M2, {DataType::FP32, DataType::FP32}, {128, 128}},
             {"E5M2 into FP32", 4002, DataType::FP32, {DataType::FP8_E5M2, DataType::FP8_E5M2}, {128, 128}},
             {"long slices of three types", 303, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E5M2}, {4096, 1000, 24}},
             {"odd spans", 50003, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E5M2}, {3, 5, 7}},
             {"one odd span among whole ones", 30001, DataType::FP32, {DataType::BF16, DataType::FP8_E5M2, DataType::BF16, DataType::FP32},
              {16, 3, 16, 13}},
             {"null destinations", 40001, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E4M3}, {16, 16, 16}, 0, {},
              {false, true, false}},
         }) {
        runTypedCase(testCase);
    }
}

// In a block of short slices each type's threads start on a warp, with fewer slices to the block where that needs
// the room and idle threads between the types; where not one slice fits that way the types follow each other.
TEST(DenseConcatenateTyped, TypesStartOnWarps) {
    REQUIRE_CUDA_DEVICE();
    const std::vector<DataType> nine = {DataType::BF16, DataType::INT32, DataType::FP8_E5M2, DataType::UINT16, DataType::FP16,
                                        DataType::INT8, DataType::UINT32, DataType::FP8_E4M3, DataType::BOOLEAN};
    for (const TypedCase &testCase : std::vector<TypedCase>{
             {"two types, 40 slices a block for 42", 123, DataType::BF16, {DataType::FP32, DataType::BF16}, {16, 16}},
             {"five types of a packet or two, 32 slices a block for 42", 1001, DataType::FP32,
              {DataType::BF16, DataType::INT32, DataType::FP8_E5M2, DataType::UINT16, DataType::FP64}, {4, 4, 4, 4, 4}},
             {"the same past the spread bound", 30001, DataType::FP32,
              {DataType::BF16, DataType::INT32, DataType::FP8_E5M2, DataType::UINT16, DataType::FP64}, {4, 4, 4, 4, 4}},
             {"two types that fill a block, one slice to it", 77, DataType::BF16, {DataType::FP32, DataType::INT32}, {400, 400}},
             {"three types of 70 packets, no room for warps", 77, DataType::BF16, {DataType::FP32, DataType::INT32, DataType::UINT32},
              {280, 280, 280}},
             {"nine types of a packet, no room for warps", 501, DataType::FP32, nine, {4, 4, 4, 4, 4, 4, 4, 4, 4}},
             {"a type between two arrays of another", 300, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP32}, {16, 48, 32}},
             {"null destinations", 300, DataType::BF16, {DataType::FP32, DataType::FP8_E5M2, DataType::BF16}, {16, 16, 16}, 0, {},
              {false, true, false}},
         }) {
        runTypedCase(testCase);
    }
}

// Spans whose element counts, places or stores do not hold a whole packet take smaller ones, down to an element.
TEST(DenseConcatenateTyped, OddSpansAndOffsetStoresTakeSmallerPackets) {
    REQUIRE_CUDA_DEVICE();
    for (const TypedCase &testCase : std::vector<TypedCase>{
             {"odd spans", 300, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E5M2}, {3, 5, 7}},
             {"one odd span among whole ones", 300, DataType::FP32, {DataType::BF16, DataType::FP8_E5M2, DataType::BF16, DataType::FP32},
              {16, 3, 16, 13}},
             {"single elements", 2000, DataType::FP8_E5M2, {DataType::FP32, DataType::INT8, DataType::FP64}, {1, 1, 1}},
             {"stores off a packet's boundary", 300, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E4M3}, {16, 16, 16}, 1,
              {1, 3, 5}},
             {"stores eight bytes off", 300, DataType::FP32, {DataType::BF16, DataType::FP32}, {64, 64}, 2, {4, 2}},
             {"an array of no elements", 50, DataType::FP32, {DataType::BF16, DataType::FP64, DataType::FP8_E5M2}, {16, 0, 16}},
         }) {
        runTypedCase(testCase);
    }
}

// A null source converts nothing: its span is zeros of the packed type, in the converting kernel when another array
// converts and in the byte kernels when every other array has the packed type, whatever type the null source names.
TEST(DenseConcatenateTyped, NullSourcesAreZerosOfThePackedType) {
    REQUIRE_CUDA_DEVICE();
    for (const TypedCase &testCase : std::vector<TypedCase>{
             {"between converting arrays", 1000, DataType::FP32, {DataType::BF16, DataType::FP32, DataType::FP8_E5M2}, {16, 16, 16}, 0, {}, {}, 2,
              {false, true, false}},
             {"named another type, beside one that converts", 600, DataType::BF16, {DataType::FP32, DataType::FP16}, {24, 8}, 0, {}, {}, 2,
              {true, false}},
             {"named another type, every other array the packed type: byte kernels", 600, DataType::BF16, {DataType::FP32, DataType::BF16},
              {24, 8}, 0, {}, {}, 2, {true, false}},
             {"long slices", 20, DataType::FP32, {DataType::FP32, DataType::BF16}, {1000, 600}, 0, {}, {}, 2, {true, false}},
             {"four packets per thread", 40001, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E4M3}, {16, 16, 16}, 0, {}, {}, 2,
              {false, true, false}},
             {"odd spans, smaller packets", 300, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E5M2}, {3, 5, 7}, 0, {}, {}, 2,
              {false, false, true}},
             {"stores off a packet's boundary", 300, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E4M3}, {16, 16, 16}, 1,
              {1, 3, 5}, {}, 2, {true, false, false}},
             {"every source null", 300, DataType::FP16, {DataType::FP32, DataType::BF16}, {16, 16}, 0, {}, {}, 2, {true, true}},
         }) {
        runTypedCase(testCase);
    }
}

// More arrays than a launch holds, null split destinations, and arrays that all have the packed type (the byte
// kernels, through the typed call).
TEST(DenseConcatenateTyped, ManyArraysNullDestinationsAndOneType) {
    REQUIRE_CUDA_DEVICE();
    TypedCase many;
    many.name = "40 arrays of every type";
    many.outerSlices = 200;
    many.packedType = DataType::FP32;
    for (uint32_t a = 0; a < 40; ++a) {
        many.types.push_back(kElementTypes[(a * 7) % kElementTypes.size()]);
        many.spanElements.push_back(8 + 4 * (a % 5));
    }
    runTypedCase(many);
    for (const TypedCase &testCase : std::vector<TypedCase>{
             {"null destinations", 500, DataType::BF16, {DataType::FP32, DataType::BF16, DataType::FP8_E5M2, DataType::FP32}, {16, 16, 16, 16}, 0, {},
              {false, true, true, false}},
             {"every destination but one null", 500, DataType::FP32, {DataType::BF16, DataType::FP8_E5M2}, {32, 32}, 0, {}, {true, false}},
             {"one type, typed call", 500, DataType::BF16, {DataType::BF16, DataType::BF16, DataType::BF16}, {16, 33, 8}},
             {"one type, E5M2", 3000, DataType::FP8_E5M2, {DataType::FP8_E5M2, DataType::FP8_E5M2}, {33, 31}, 0, {}, {false, true}},
         }) {
        runTypedCase(testCase);
    }
}

TEST(DenseConcatenateTyped, RejectsMalformedLaunches) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor packed = makeGpuBytes(std::vector<uint8_t>(256, 0), stream);
    Tensor source = makeGpuBytes(std::vector<uint8_t>(256, 0), stream);
    void *const p = packed.getMemPtr();
    void *const a = source.getMemPtr();
    void *const b = source.getMemPtr<uint8_t>() + 128;
    const std::vector<DataType> types = {DataType::FP32, DataType::BF16};
    EXPECT_THROW(launchConcatenate(p, DataType::FP32, {}, {}, 1, {}, stream), std::invalid_argument);
    EXPECT_THROW(launchConcatenate(p, DataType::FP32, {a, b}, {DataType::FP32}, 1, {4, 4}, stream), std::invalid_argument);
    EXPECT_THROW(launchConcatenate(p, DataType::FP32, {a, b}, types, 1, {4}, stream), std::invalid_argument);
    EXPECT_THROW(launchConcatenate(nullptr, DataType::FP32, {a, b}, types, 1, {4, 4}, stream), std::invalid_argument);
    EXPECT_THROW(launchConcatenate(p, DataType::FP32, {a, b}, types, 1, {0, 0}, stream), std::invalid_argument);
    EXPECT_THROW(launchConcatenate(p, DataType::TF32, {a, b}, types, 1, {4, 4}, stream), std::invalid_argument);
    EXPECT_THROW(launchConcatenate(p, DataType::FP32, {a, b}, {DataType::FP32, DataType::TF32}, 1, {4, 4}, stream), std::invalid_argument);
    // A store off a boundary of its element type.
    EXPECT_THROW(launchConcatenate(packed.getMemPtr<uint8_t>() + 2, DataType::FP32, {a, b}, types, 1, {4, 4}, stream), std::invalid_argument);
    EXPECT_THROW(launchConcatenate(p, DataType::FP32, {a, source.getMemPtr<uint8_t>() + 129}, types, 1, {4, 4}, stream), std::invalid_argument);
    EXPECT_THROW(launchSplit({a, b}, types, nullptr, DataType::FP32, 1, {4, 4}, stream), std::invalid_argument);
    // Nothing to do is not an error.
    launchConcatenate(p, DataType::FP32, {a, b}, types, 0, {4, 4}, stream);
    launchSplit({nullptr, nullptr}, types, p, DataType::FP32, 1, {4, 4}, stream);
    stream.synchronize();
}
