#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
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
            GTEST_SKIP() << "CUDA device is required for RaggedConcatenate kernel tests.";                              \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

Tensor makeGpuBytes(const std::vector<uint8_t> &values, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(std::max<size_t>(values.size(), 16))}));
    uint8_t *hostValues = host.getMemPtr<uint8_t>();
    for (size_t i = 0; i < host.getTotalNumElements(); ++i) hostValues[i] = i < values.size() ? values[i] : 0;
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

// The element type of a test's byte width: every input and the output share it, so these cases take the byte kernels.
DataType elementTypeOf(uint64_t elementBytes) {
    switch (elementBytes) {
        case 1:
            return DataType::UINT8;
        case 2:
            return DataType::UINT16;
        case 4:
            return DataType::UINT32;
        case 8:
            return DataType::UINT64;
        default:
            throw std::invalid_argument("No element type of that width.");
    }
}

uint64_t product(const std::vector<uint64_t> &dimensions, size_t first, size_t end) {
    uint64_t result = 1;
    for (size_t d = first; d < end; ++d) result *= dimensions[d];
    return result;
}

// Packed values of capacityRows rows with trailing dimensions `trailing`, concatenated on trailing axis `axis` (1 is
// the first trailing axis) from inputs of the given extents on it, for the first activeRows rows. Checked byte for byte
// against the index arithmetic of a trailing-axis concatenate, with every inactive row and every byte of a null split
// destination untouched.
struct RaggedCase {
    std::string name;
    uint64_t capacityRows = 8;
    uint64_t activeRows = 5;
    std::vector<uint64_t> trailing;
    uint32_t axis = 1;
    std::vector<uint64_t> axisExtents;
    uint64_t elementBytes = 4;
    std::vector<bool> nullSplitDestination;  // empty: none
};

void runRaggedCase(const RaggedCase &testCase) {
    SCOPED_TRACE(testCase.name);
    constexpr uint8_t sentinel = 0xDC;
    const size_t inputs = testCase.axisExtents.size();
    std::vector<uint64_t> outputDimensions{testCase.capacityRows};
    outputDimensions.insert(outputDimensions.end(), testCase.trailing.begin(), testCase.trailing.end());
    uint64_t outputAxis = 0;
    for (const uint64_t extent : testCase.axisExtents) outputAxis += extent;
    outputDimensions[testCase.axis] = outputAxis;
    const uint64_t outer = product(outputDimensions, 1, testCase.axis);
    const uint64_t inner = product(outputDimensions, testCase.axis + 1, outputDimensions.size()) * testCase.elementBytes;
    std::vector<bool> nullDestination = testCase.nullSplitDestination;
    nullDestination.resize(inputs, false);

    const RaggedTrailingConcatenateGeometry geometry =
        buildRaggedTrailingConcatenateGeometry(outputDimensions, testCase.axis, testCase.axisExtents);
    ASSERT_EQ(geometry.outerSlicesPerValue, outer);
    const DataType elementType = elementTypeOf(testCase.elementBytes);
    const std::vector<DataType> inputTypes(inputs, elementType);

    Stream stream(0);
    std::vector<std::vector<uint8_t>> inputValues(inputs);
    std::vector<Tensor> inputTensors;
    std::vector<void *> inputPointers;
    for (size_t a = 0; a < inputs; ++a) {
        inputValues[a].resize(testCase.capacityRows * outer * testCase.axisExtents[a] * inner);
        for (uint64_t byte = 0; byte < inputValues[a].size(); ++byte) inputValues[a][byte] = static_cast<uint8_t>(31 * byte + 7 * a + 3);
        inputTensors.push_back(makeGpuBytes(inputValues[a], stream));
        inputPointers.push_back(testCase.axisExtents[a] == 0 ? nullptr : inputTensors.back().getMemPtr());
    }
    const uint64_t outputValueBytes = outer * outputAxis * inner;
    Tensor output = makeGpuBytes(std::vector<uint8_t>(testCase.capacityRows * outputValueBytes, sentinel), stream);

    launchConcatenate(output.getMemPtr(), elementType, inputPointers, inputTypes, geometry.outerSlicesFor(testCase.activeRows),
                      geometry.spanElements, stream);
    stream.synchronize();
    const std::vector<uint8_t> outputValues = copyGpuBytes(output, stream);
    // The expected output: row r, outer slice o, axis position p (input a's position q), inner byte i.
    uint64_t wrong = 0;
    for (uint64_t r = 0; r < testCase.capacityRows; ++r) {
        for (uint64_t o = 0; o < outer; ++o) {
            uint64_t axisBase = 0;
            for (size_t a = 0; a < inputs; ++a) {
                for (uint64_t q = 0; q < testCase.axisExtents[a]; ++q) {
                    for (uint64_t i = 0; i < inner; ++i) {
                        const uint64_t outputByte = ((r * outer + o) * outputAxis + axisBase + q) * inner + i;
                        const uint64_t inputByte = ((r * outer + o) * testCase.axisExtents[a] + q) * inner + i;
                        const uint8_t expected = r < testCase.activeRows ? inputValues[a][inputByte] : sentinel;
                        wrong += outputValues[outputByte] != expected;
                    }
                }
                axisBase += testCase.axisExtents[a];
            }
        }
    }
    EXPECT_EQ(wrong, 0U) << "concatenate";

    std::vector<Tensor> splitTensors;
    std::vector<void *> splitPointers;
    for (size_t a = 0; a < inputs; ++a) {
        splitTensors.push_back(makeGpuBytes(std::vector<uint8_t>(inputValues[a].size(), sentinel), stream));
        splitPointers.push_back(nullDestination[a] || testCase.axisExtents[a] == 0 ? nullptr : splitTensors.back().getMemPtr());
    }
    launchSplit(splitPointers, inputTypes, output.getMemPtr(), elementType, geometry.outerSlicesFor(testCase.activeRows),
                geometry.spanElements, stream);
    stream.synchronize();
    wrong = 0;
    for (size_t a = 0; a < inputs; ++a) {
        const std::vector<uint8_t> splitValues = copyGpuBytes(splitTensors[a], stream);
        const uint64_t valueBytes = outer * testCase.axisExtents[a] * inner;
        for (uint64_t byte = 0; byte < inputValues[a].size(); ++byte) {
            const bool written = !nullDestination[a] && byte / std::max<uint64_t>(valueBytes, 1) < testCase.activeRows;
            wrong += splitValues[byte] != (written ? inputValues[a][byte] : sentinel);
        }
    }
    EXPECT_EQ(wrong, 0U) << "split";
}

}  // namespace

TEST(RaggedConcatenate, TrailingGeometryIsTheDenseConcatenateOfOuterSlices) {
    // Values [rows][3][5][2] concatenated on axis 2 from extents 2 and 3: spans of 2 x 2 and 3 x 2 elements.
    const RaggedTrailingConcatenateGeometry geometry = buildRaggedTrailingConcatenateGeometry({16, 3, 5, 2}, 2, {2, 3});
    EXPECT_EQ(geometry.outerSlicesPerValue, 3U);
    EXPECT_EQ(geometry.spanElements, (std::vector<uint64_t>{4, 6}));
    EXPECT_EQ(geometry.outerSlicesFor(7), 21U);
    EXPECT_THROW(buildRaggedTrailingConcatenateGeometry({16, 3, 5, 2}, 0, {2, 3}), std::invalid_argument);
    EXPECT_THROW(buildRaggedTrailingConcatenateGeometry({16, 3, 5, 2}, 2, {2, 2}), std::invalid_argument);
}

TEST(RaggedConcatenate, ForwardAndSplitRespectHostActivePrefixAndNonLastTrailingAxisStrides) {
    REQUIRE_CUDA_DEVICE();
    for (const RaggedCase &testCase : std::vector<RaggedCase>{
             {"first trailing axis, inner 2, FP32", 8, 5, {0, 2}, 1, {2, 3}, 4},
             {"second trailing axis: 3 outer slices a value", 8, 5, {3, 0, 2}, 2, {2, 3}, 4},
             {"last trailing axis, 4 outer slices, 1-byte", 9, 6, {2, 2, 0}, 3, {5, 7}, 1},
             {"three inputs, odd extents, 2-byte", 16, 11, {0, 3}, 1, {1, 4, 3}, 2},
             {"every row active", 6, 6, {0, 4}, 1, {3, 5}, 4},
             {"a null split destination", 8, 5, {2, 0, 2}, 2, {2, 3, 1}, 4, {false, true, false}},
             {"2000 rows: 16-byte packets", 2100, 2000, {0, 4}, 1, {2, 2}, 4},
             {"32768 of 33000 rows of 33-byte values, past the narrow-block limit", 33000, 32768, {0}, 1, {17, 16}, 1},
         }) {
        runRaggedCase(testCase);
    }
}

TEST(RaggedConcatenate, AllEmptyPartitionLeavesForwardAndBackwardCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    runRaggedCase({"no active rows", 8, 0, {2, 0}, 2, {2, 3}, 4});
}
