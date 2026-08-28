#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Expression/CudaHelpers.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
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

template <typename OffsetT>
void runForwardAndSplitCase(DataType offsetsDataType) {
    constexpr uint64_t rows = 8;
    constexpr uint64_t activeRows = 5;
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t inner = 2;
    constexpr uint64_t leftAxis = 2;
    constexpr uint64_t rightAxis = 3;
    constexpr uint64_t joinedAxis = leftAxis + rightAxis;
    constexpr float inactiveInputPoison = std::numeric_limits<float>::quiet_NaN();
    constexpr float outputSentinel = -7777.0F;
    constexpr float splitSentinel = -8888.0F;

    Stream stream(0);
    const std::vector<OffsetT> offsets{0, 2, 2, static_cast<OffsetT>(activeRows)};
    Tensor offsetsGpu = makeGpuTensor<OffsetT>({batchSize + 1}, offsets, stream);
    ASSERT_EQ(offsetsGpu.getDataType(), offsetsDataType);

    std::vector<float> leftValues = sourceValues(rows, leftAxis, inner, 1000.0F, inactiveInputPoison);
    std::vector<float> rightValues = sourceValues(rows, rightAxis, inner, 2000.0F, inactiveInputPoison);
    Tensor left = makeGpuTensor<float>({rows, leftAxis, inner}, leftValues, stream);
    Tensor right = makeGpuTensor<float>({rows, rightAxis, inner}, rightValues, stream);
    Tensor joined = makeGpuTensor<float>({rows, joinedAxis, inner},
                                         std::vector<float>(rows * joinedAxis * inner, outputSentinel),
                                         stream);

    DeviceAllocation sourceTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    DeviceAllocation axisElements = uploadArray<long>({static_cast<long>(leftAxis), static_cast<long>(rightAxis)}, stream);
    DeviceAllocation joinedStrides =
        uploadArray<long>({static_cast<long>(joinedAxis * inner), static_cast<long>(inner), 1L}, stream);
    DeviceAllocation sourceStrides = uploadArray<long>(
        {static_cast<long>(leftAxis * inner), static_cast<long>(inner), 1L,
         static_cast<long>(rightAxis * inner), static_cast<long>(inner), 1L},
        stream);

    launchRaggedConcatenate(joined.getMemPtr(),
                            reinterpret_cast<void **>(sourceTable.get()),
                            sizeof(float),
                            static_cast<long>(rows * joinedAxis * inner),
                            joinedAxis * inner,
                            3,
                            2,
                            1,
                            static_cast<long *>(axisElements.get()),
                            static_cast<long *>(joinedStrides.get()),
                            static_cast<long *>(sourceStrides.get()),
                            offsetsGpu.getMemPtr(),
                            sizeof(OffsetT),
                            batchSize,
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
                      sizeof(float),
                      static_cast<long>(rows * joinedAxis * inner),
                      joinedAxis * inner,
                      3,
                      2,
                      1,
                      static_cast<long *>(axisElements.get()),
                      static_cast<long *>(joinedStrides.get()),
                      static_cast<long *>(sourceStrides.get()),
                      offsetsGpu.getMemPtr(),
                      sizeof(OffsetT),
                      batchSize,
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

template <typename OffsetT>
void runAllEmptyCase(DataType offsetsDataType) {
    constexpr uint64_t rows = 4;
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t leftAxis = 1;
    constexpr uint64_t rightAxis = 2;
    constexpr uint64_t joinedAxis = 3;
    constexpr float sentinel = -9191.0F;

    Stream stream(0);
    Tensor offsets = makeGpuTensor<OffsetT>({batchSize + 1}, {0, 0, 0}, stream);
    ASSERT_EQ(offsets.getDataType(), offsetsDataType);
    Tensor left = makeGpuTensor<float>({rows, leftAxis}, std::vector<float>(rows * leftAxis, 1.0F), stream);
    Tensor right = makeGpuTensor<float>({rows, rightAxis}, std::vector<float>(rows * rightAxis, 2.0F), stream);
    Tensor joined = makeGpuTensor<float>({rows, joinedAxis}, std::vector<float>(rows * joinedAxis, sentinel), stream);

    DeviceAllocation sourceTable = makePointerTable({left.getMemPtr(), right.getMemPtr()}, stream);
    DeviceAllocation axisElements = uploadArray<long>({1L, 2L}, stream);
    DeviceAllocation joinedStrides = uploadArray<long>({3L, 1L}, stream);
    DeviceAllocation sourceStrides = uploadArray<long>({1L, 1L, 2L, 1L}, stream);

    launchRaggedConcatenate(joined.getMemPtr(),
                            reinterpret_cast<void **>(sourceTable.get()),
                            sizeof(float),
                            static_cast<long>(rows * joinedAxis),
                            joinedAxis,
                            2,
                            2,
                            1,
                            static_cast<long *>(axisElements.get()),
                            static_cast<long *>(joinedStrides.get()),
                            static_cast<long *>(sourceStrides.get()),
                            offsets.getMemPtr(),
                            sizeof(OffsetT),
                            batchSize,
                            stream);
    stream.synchronize();
    for (float value : copyGpuTensor<float>(joined, stream)) EXPECT_EQ(value, sentinel);

    Tensor leftGradient = makeGpuTensor<float>({rows, leftAxis}, std::vector<float>(rows * leftAxis, sentinel), stream);
    Tensor rightGradient = makeGpuTensor<float>({rows, rightAxis}, std::vector<float>(rows * rightAxis, sentinel), stream);
    DeviceAllocation destinationTable = makePointerTable({leftGradient.getMemPtr(), rightGradient.getMemPtr()}, stream);
    launchRaggedSplit(reinterpret_cast<void **>(destinationTable.get()),
                      joined.getMemPtr(),
                      sizeof(float),
                      static_cast<long>(rows * joinedAxis),
                      joinedAxis,
                      2,
                      2,
                      1,
                      static_cast<long *>(axisElements.get()),
                      static_cast<long *>(joinedStrides.get()),
                      static_cast<long *>(sourceStrides.get()),
                      offsets.getMemPtr(),
                      sizeof(OffsetT),
                      batchSize,
                      stream);
    stream.synchronize();
    for (float value : copyGpuTensor<float>(leftGradient, stream)) EXPECT_EQ(value, sentinel);
    for (float value : copyGpuTensor<float>(rightGradient, stream)) EXPECT_EQ(value, sentinel);
}

}  // namespace

TEST(RaggedConcatenate, ForwardAndSplitRespectUint32ActivePrefixAndNonLastTrailingAxisStrides) {
    REQUIRE_CUDA_DEVICE();
    runForwardAndSplitCase<uint32_t>(DataType::UINT32);
}

TEST(RaggedConcatenate, ForwardAndSplitRespectUint64ActivePrefixAndNonLastTrailingAxisStrides) {
    REQUIRE_CUDA_DEVICE();
    runForwardAndSplitCase<uint64_t>(DataType::UINT64);
}

TEST(RaggedConcatenate, AllEmptyPartitionLeavesForwardAndBackwardCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    runAllEmptyCase<uint32_t>(DataType::UINT32);
    runAllEmptyCase<uint64_t>(DataType::UINT64);
}
