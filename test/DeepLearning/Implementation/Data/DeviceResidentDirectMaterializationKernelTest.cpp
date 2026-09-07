#include "DeepLearning/Implementation/Data/Residency/DeviceResidentDirectMaterializationKernel.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cudaDeviceCountForTest = 0;                                                                                  \
        const cudaError_t cudaStatusForTest = cudaGetDeviceCount(&cudaDeviceCountForTest);                               \
        if (cudaStatusForTest != cudaSuccess || cudaDeviceCountForTest <= 0) {                                            \
            GTEST_SKIP() << "CUDA device is required for device-resident direct materialization tests.";                \
        }                                                                                                                \
    } while (false)

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
const TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

template <typename T>
DataType dtypeFor();
template <>
DataType dtypeFor<uint8_t>() { return DataType::UINT8; }
template <>
DataType dtypeFor<uint64_t>() { return DataType::UINT64; }

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t> &dimensions, const std::vector<T> &values, Stream &stream) {
    Tensor host(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dimensions));
    if (host.getTotalNumElements() != values.size()) {
        throw std::runtime_error("DeviceResidentDirectMaterializationKernel test tensor value count mismatch.");
    }
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
    const T *values = host.getMemPtr<T>();
    return std::vector<T>(values, values + host.getTotalNumElements());
}

void runMaterializationCase(
    uint64_t numExamples,
    uint64_t batchSize,
    uint64_t recordSizeBytes,
    uint64_t fieldOffsetBytes,
    uint64_t fieldBytes) {
    constexpr uint8_t recordSentinel = 0x5aU;
    constexpr uint8_t destinationSentinel = 0xcdU;

    std::vector<uint8_t> records(numExamples * recordSizeBytes, recordSentinel);
    for (uint64_t sourceRow = 0; sourceRow < numExamples; ++sourceRow) {
        for (uint64_t byte = 0; byte < fieldBytes; ++byte) {
            records[sourceRow * recordSizeBytes + fieldOffsetBytes + byte] =
                static_cast<uint8_t>((sourceRow * 29 + byte * 17 + 11) & 0xffU);
        }
    }

    std::vector<uint64_t> rowIndices(batchSize, 0);
    for (uint64_t batchRow = 0; batchRow < batchSize; ++batchRow) {
        rowIndices[batchRow] = (batchRow * 137 + 19) % numExamples;
    }
    // Invalid row indices historically leave their destination row untouched.
    // Keep that behavior under the row/chunk mapping as well.
    if (batchSize > 7) rowIndices[7] = numExamples + 3;
    if (batchSize > 263) rowIndices[263] = numExamples;

    std::vector<uint8_t> expected(batchSize * fieldBytes, destinationSentinel);
    for (uint64_t batchRow = 0; batchRow < batchSize; ++batchRow) {
        const uint64_t sourceRow = rowIndices[batchRow];
        if (sourceRow >= numExamples) continue;
        for (uint64_t byte = 0; byte < fieldBytes; ++byte) {
            expected[batchRow * fieldBytes + byte] =
                records[sourceRow * recordSizeBytes + fieldOffsetBytes + byte];
        }
    }

    Stream stream(0);
    Tensor recordStorage = makeGpuTensor<uint8_t>({records.size()}, records, stream);
    Tensor rowIndicesDevice = makeGpuTensor<uint64_t>({batchSize}, rowIndices, stream);
    Tensor destination = makeGpuTensor<uint8_t>(
        {batchSize, fieldBytes},
        std::vector<uint8_t>(batchSize * fieldBytes, destinationSentinel),
        stream);

    launchDeviceResidentDirectMaterializationKernel(
        recordStorage,
        numExamples,
        recordSizeBytes,
        fieldOffsetBytes,
        fieldBytes,
        destination,
        rowIndicesDevice,
        stream);
    stream.synchronize();

    const std::vector<uint8_t> actual = copyGpuTensor<uint8_t>(destination, stream);
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t offset = 0; offset < expected.size(); ++offset) {
        EXPECT_EQ(actual[offset], expected[offset]) << "byte offset " << offset;
    }
}

TEST(DeviceResidentDirectMaterializationKernelTest, AlignedWideFieldsUseRowChunkMapping) {
    REQUIRE_CUDA_DEVICE();
    // All row starts are 16-byte aligned, permitting uint4 copies. Three uint4
    // chunks per row also exercises the multi-lane small-subgroup path.
    runMaterializationCase(/*numExamples=*/701,
                           /*batchSize=*/513,
                           /*recordSizeBytes=*/64,
                           /*fieldOffsetBytes=*/16,
                           /*fieldBytes=*/48);
}

TEST(DeviceResidentDirectMaterializationKernelTest, PackedUnalignedOddFieldsPreserveEveryByte) {
    REQUIRE_CUDA_DEVICE();
    // Neither source row starts nor the field size have useful typed alignment.
    // This forces the byte fallback and an incomplete final row subgroup.
    runMaterializationCase(/*numExamples=*/607,
                           /*batchSize=*/517,
                           /*recordSizeBytes=*/53,
                           /*fieldOffsetBytes=*/3,
                           /*fieldBytes=*/37);
}

TEST(DeviceResidentDirectMaterializationKernelTest, ScalarFieldsKeepManyRowsResidentPerBlock) {
    REQUIRE_CUDA_DEVICE();
    // A single aligned uint32 per row selects one thread per row, so one CTA can
    // service 256 independent rows instead of launching a mostly-idle CTA/row.
    runMaterializationCase(/*numExamples=*/1031,
                           /*batchSize=*/1025,
                           /*recordSizeBytes=*/16,
                           /*fieldOffsetBytes=*/4,
                           /*fieldBytes=*/4);
}

}  // namespace
