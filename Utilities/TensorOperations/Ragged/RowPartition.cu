#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include "Utilities/Expression/CudaHelpers.h"
#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include <cub/cub.cuh>

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ThorImplementation {
namespace {
using namespace CubDevicePrimitiveSupport;

template <typename T>
size_t queryInclusiveSum(uint64_t batch_size) {
    if (batch_size == 0) {
        return 0;
    }
    if (batch_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("row partition batch_size exceeds CUB's int item-count limit.");
    }
    size_t queried_bytes = 0;
    const T* input = nullptr;
    T* output = nullptr;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, queried_bytes, input, output, static_cast<int>(batch_size)));
    return queried_bytes;
}

template <typename T>
void launchInclusiveSum(const RowPartitionLengthsToOffsetsPlan& plan,
                        const Tensor& temp_storage,
                        const Tensor& lengths,
                        Tensor& offsets,
                        Stream& stream) {
    if (plan.batch_size == 0) {
        return;
    }
    void* temp_ptr = const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
    size_t temp_bytes = plan.temp_storage_bytes;
    const T* lengths_ptr = lengths.getMemPtr<T>();
    T* offsets_ptr = offsets.getMemPtr<T>();
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp_ptr,
                                             temp_bytes,
                                             lengths_ptr,
                                             offsets_ptr + 1,
                                             static_cast<int>(plan.batch_size),
                                             stream.getStream()));
}

template <typename T>
__global__ void setFirstOffsetZeroKernel(T* offsets) {
    offsets[0] = T{0};
}

template <typename OffsetT, typename LengthT>
__global__ void offsetsToLengthsKernel(const OffsetT* offsets, LengthT* lengths, uint64_t batch_size) {
    const uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) {
        return;
    }
    lengths[row] = static_cast<LengthT>(offsets[row + 1] - offsets[row]);
}

template <typename OffsetT, typename RowIdT>
__global__ void offsetsToRowIdsKernel(const OffsetT* offsets, RowIdT* row_ids, uint64_t batch_size, uint64_t max_total_values) {
    const uint64_t row = blockIdx.x;
    if (row >= batch_size) {
        return;
    }
    const uint64_t begin = static_cast<uint64_t>(offsets[row]);
    const uint64_t end = static_cast<uint64_t>(offsets[row + 1]);
    const uint64_t bounded_end = end < max_total_values ? end : max_total_values;
    for (uint64_t idx = begin + threadIdx.x; idx < bounded_end; idx += blockDim.x) {
        row_ids[idx] = static_cast<RowIdT>(row);
    }
}

template <typename OffsetT>
__global__ void validateOffsetsKernel(const OffsetT* offsets,
                                      uint32_t* validation_error_bits,
                                      uint64_t batch_size,
                                      uint64_t max_total_values) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 && offsets[0] != OffsetT{0}) {
        atomicOr(validation_error_bits, static_cast<uint32_t>(ROW_PARTITION_OFFSETS_MUST_START_AT_ZERO));
    }
    if (idx < batch_size) {
        const uint64_t current = static_cast<uint64_t>(offsets[idx]);
        const uint64_t next = static_cast<uint64_t>(offsets[idx + 1]);
        if (current > next) {
            atomicOr(validation_error_bits, static_cast<uint32_t>(ROW_PARTITION_OFFSETS_MUST_BE_MONOTONIC));
        }
    }
    if (idx == 0 && static_cast<uint64_t>(offsets[batch_size]) > max_total_values) {
        atomicOr(validation_error_bits, static_cast<uint32_t>(ROW_PARTITION_OFFSETS_EXCEED_CAPACITY));
    }
}


uint64_t checkedOffsetsElements(uint64_t batch_size) {
    if (batch_size == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("row partition batch_size overflows offsets element count.");
    }
    return batch_size + 1;
}

void requireRowPartitionVectorGpuTensor(const Tensor& tensor, const char* name) {
    requireDenseContiguousGpuTensor(tensor, name);
    if (tensor.getNumDimensions() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a rank-1 tensor.");
    }
}

void requireRowPartitionOffsetDType(DataType dtype, const char* name) {
    if (!isRowPartitionOffsetDTypeSupported(dtype)) {
        throw std::invalid_argument(std::string(name) + " dtype must be UINT32 or UINT64.");
    }
}

void validateLengthsAndOffsets(const Tensor& lengths, const Tensor& offsets, uint64_t batch_size) {
    requireRowPartitionVectorGpuTensor(lengths, "row partition lengths");
    requireRowPartitionVectorGpuTensor(offsets, "row partition offsets");
    requireSameGpuPlacement(lengths, offsets, "row partition lengths", "row partition offsets");
    requireStorageForNumItems(lengths, "row partition lengths", batch_size);
    requireStorageForNumItems(offsets, "row partition offsets", checkedOffsetsElements(batch_size));
    requireRowPartitionOffsetDType(lengths.getDataType(), "row partition lengths");
    requireRowPartitionOffsetDType(offsets.getDataType(), "row partition offsets");
    if (lengths.getDataType() != offsets.getDataType()) {
        throw std::invalid_argument("row partition lengths and offsets must have the same dtype.");
    }
}

void validateOffsetsAndLengths(const Tensor& offsets, const Tensor& lengths, uint64_t batch_size) {
    requireRowPartitionVectorGpuTensor(offsets, "row partition offsets");
    requireRowPartitionVectorGpuTensor(lengths, "row partition lengths");
    requireSameGpuPlacement(offsets, lengths, "row partition offsets", "row partition lengths");
    requireStorageForNumItems(offsets, "row partition offsets", checkedOffsetsElements(batch_size));
    requireStorageForNumItems(lengths, "row partition lengths", batch_size);
    requireRowPartitionOffsetDType(offsets.getDataType(), "row partition offsets");
    requireRowPartitionOffsetDType(lengths.getDataType(), "row partition lengths");
    if (offsets.getDataType() != lengths.getDataType()) {
        throw std::invalid_argument("row partition offsets and lengths must have the same dtype.");
    }
}

void validateOffsetsAndRowIds(const Tensor& offsets, const Tensor& row_ids, uint64_t batch_size, uint64_t max_total_values) {
    requireRowPartitionVectorGpuTensor(offsets, "row partition offsets");
    requireRowPartitionVectorGpuTensor(row_ids, "row partition row_ids");
    requireSameGpuPlacement(offsets, row_ids, "row partition offsets", "row partition row_ids");
    requireStorageForNumItems(offsets, "row partition offsets", checkedOffsetsElements(batch_size));
    requireStorageForNumItems(row_ids, "row partition row_ids", max_total_values);
    requireRowPartitionOffsetDType(offsets.getDataType(), "row partition offsets");
    requireRowPartitionOffsetDType(row_ids.getDataType(), "row partition row_ids");
    if (row_ids.getDataType() == DataType::UINT32 && batch_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("row partition row_ids UINT32 output cannot represent batch_size.");
    }
}

void validateOffsetsAndErrorBits(const Tensor& offsets, const Tensor& validation_error_bits, uint64_t batch_size) {
    requireRowPartitionVectorGpuTensor(offsets, "row partition offsets");
    requireRowPartitionVectorGpuTensor(validation_error_bits, "row partition validation_error_bits");
    requireSameGpuPlacement(offsets, validation_error_bits, "row partition offsets", "row partition validation_error_bits");
    requireStorageForNumItems(offsets, "row partition offsets", checkedOffsetsElements(batch_size));
    requireStorageForNumItems(validation_error_bits, "row partition validation_error_bits", 1);
    requireRowPartitionOffsetDType(offsets.getDataType(), "row partition offsets");
    if (validation_error_bits.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("row partition validation_error_bits dtype must be UINT32.");
    }
}

template <typename Fn>
decltype(auto) dispatchOffsetDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT32:
            return fn.template operator()<uint32_t>();
        case DataType::UINT64:
            return fn.template operator()<uint64_t>();
        default:
            throw std::invalid_argument("Unsupported row partition offset dtype " + TensorDescriptor::getElementTypeName(dtype) + ".");
    }
}

template <typename OffsetT, typename Fn>
decltype(auto) dispatchRowIdDType(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::UINT32:
            return fn.template operator()<OffsetT, uint32_t>();
        case DataType::UINT64:
            return fn.template operator()<OffsetT, uint64_t>();
        default:
            throw std::invalid_argument("Unsupported row partition row_ids dtype " + TensorDescriptor::getElementTypeName(dtype) + ".");
    }
}

struct QueryInclusiveSumFn {
    uint64_t batch_size;
    template <typename T>
    size_t operator()() const {
        return queryInclusiveSum<T>(batch_size);
    }
};

struct LaunchInclusiveSumFn {
    const RowPartitionLengthsToOffsetsPlan& plan;
    const Tensor& temp_storage;
    const Tensor& lengths;
    Tensor& offsets;
    Stream& stream;
    template <typename T>
    void operator()() const {
        launchInclusiveSum<T>(plan, temp_storage, lengths, offsets, stream);
    }
};

struct SetFirstOffsetZeroFn {
    Tensor& offsets;
    Stream& stream;
    template <typename T>
    void operator()() const {
        setFirstOffsetZeroKernel<T><<<1, 1, 0, stream.getStream()>>>(offsets.getMemPtr<T>());
        CUDA_CHECK(cudaGetLastError());
    }
};

struct OffsetsToLengthsFn {
    const Tensor& offsets;
    Tensor& lengths;
    uint64_t batch_size;
    Stream& stream;
    template <typename T>
    void operator()() const {
        if (batch_size == 0) {
            return;
        }
        const uint32_t threads = 256;
        const uint64_t blocks64 = (batch_size + threads - 1U) / threads;
        if (blocks64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument("row partition offsetsToLengths grid exceeds CUDA 1D grid limit.");
        }
        offsetsToLengthsKernel<T, T><<<static_cast<uint32_t>(blocks64), threads, 0, stream.getStream()>>>(
            offsets.getMemPtr<T>(), lengths.getMemPtr<T>(), batch_size);
        CUDA_CHECK(cudaGetLastError());
    }
};

struct OffsetsToRowIdsFn {
    const Tensor& offsets;
    Tensor& row_ids;
    uint64_t batch_size;
    uint64_t max_total_values;
    Stream& stream;
    template <typename OffsetT, typename RowIdT>
    void operator()() const {
        if (batch_size == 0 || max_total_values == 0) {
            return;
        }
        if (batch_size > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument("row partition offsetsToRowIds grid exceeds CUDA 1D grid limit.");
        }
        const uint32_t threads = 256;
        offsetsToRowIdsKernel<OffsetT, RowIdT><<<static_cast<uint32_t>(batch_size), threads, 0, stream.getStream()>>>(
            offsets.getMemPtr<OffsetT>(), row_ids.getMemPtr<RowIdT>(), batch_size, max_total_values);
        CUDA_CHECK(cudaGetLastError());
    }
};


struct OffsetsToRowIdsOffsetFn {
    const Tensor& offsets;
    Tensor& row_ids;
    uint64_t batch_size;
    uint64_t max_total_values;
    Stream& stream;
    template <typename OffsetT>
    void operator()() const {
        dispatchRowIdDType<OffsetT>(row_ids.getDataType(), OffsetsToRowIdsFn{offsets, row_ids, batch_size, max_total_values, stream});
    }
};

struct ValidateOffsetsFn {
    const Tensor& offsets;
    Tensor& validation_error_bits;
    uint64_t batch_size;
    uint64_t max_total_values;
    Stream& stream;
    template <typename OffsetT>
    void operator()() const {
        const uint64_t items = checkedOffsetsElements(batch_size);
        const uint32_t threads = 256;
        const uint64_t blocks64 = (items + threads - 1U) / threads;
        if (blocks64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument("row partition validateOffsetsDebug grid exceeds CUDA 1D grid limit.");
        }
        validateOffsetsKernel<OffsetT><<<static_cast<uint32_t>(blocks64), threads, 0, stream.getStream()>>>(
            offsets.getMemPtr<OffsetT>(), validation_error_bits.getMemPtr<uint32_t>(), batch_size, max_total_values);
        CUDA_CHECK(cudaGetLastError());
    }
};

}  // namespace

bool isRowPartitionOffsetDTypeSupported(DataType dtype) { return dtype == DataType::UINT32 || dtype == DataType::UINT64; }

RowPartitionLengthsToOffsetsPlan prepareRowPartitionLengthsToOffsets(const Tensor& lengths,
                                                                    const Tensor& offsets,
                                                                    uint64_t batch_size) {
    validateLengthsAndOffsets(lengths, offsets, batch_size);
    RowPartitionLengthsToOffsetsPlan plan;
    plan.placement = lengths.getPlacement();
    plan.dtype = lengths.getDataType();
    plan.batch_size = batch_size;
    plan.temp_storage_bytes = dispatchOffsetDType(plan.dtype, QueryInclusiveSumFn{batch_size});
    return plan;
}

size_t rowPartitionLengthsToOffsetsTempBytes(const Tensor& lengths, const Tensor& offsets, uint64_t batch_size) {
    return prepareRowPartitionLengthsToOffsets(lengths, offsets, batch_size).temp_storage_bytes;
}

void rowPartitionLengthsToOffsets(const RowPartitionLengthsToOffsetsPlan& plan,
                                  const Tensor& temp_storage,
                                  const Tensor& lengths,
                                  Tensor& offsets,
                                  Stream& stream) {
    validateLengthsAndOffsets(lengths, offsets, plan.batch_size);
    if (lengths.getPlacement() != plan.placement || lengths.getDataType() != plan.dtype) {
        throw std::invalid_argument("row partition lengthsToOffsets tensors do not match prepared plan.");
    }
    if (plan.temp_storage_bytes > 0) {
        requireTempStorage(temp_storage, plan.placement, plan.temp_storage_bytes);
    }
    dispatchOffsetDType(plan.dtype, SetFirstOffsetZeroFn{offsets, stream});
    dispatchOffsetDType(plan.dtype, LaunchInclusiveSumFn{plan, temp_storage, lengths, offsets, stream});
}

void rowPartitionLengthsToOffsets(const Tensor& temp_storage,
                                  size_t temp_storage_bytes,
                                  const Tensor& lengths,
                                  Tensor& offsets,
                                  uint64_t batch_size,
                                  Stream& stream) {
    RowPartitionLengthsToOffsetsPlan plan = prepareRowPartitionLengthsToOffsets(lengths, offsets, batch_size);
    if (temp_storage_bytes < plan.temp_storage_bytes) {
        throw std::invalid_argument("row partition lengthsToOffsets temp_storage_bytes is smaller than the prepared requirement.");
    }
    plan.temp_storage_bytes = temp_storage_bytes;
    rowPartitionLengthsToOffsets(plan, temp_storage, lengths, offsets, stream);
}

void rowPartitionOffsetsToLengths(const Tensor& offsets, Tensor& lengths, uint64_t batch_size, Stream& stream) {
    validateOffsetsAndLengths(offsets, lengths, batch_size);
    dispatchOffsetDType(offsets.getDataType(), OffsetsToLengthsFn{offsets, lengths, batch_size, stream});
}

void rowPartitionOffsetsToRowIds(const Tensor& offsets,
                                 Tensor& row_ids,
                                 uint64_t batch_size,
                                 uint64_t max_total_values,
                                 Stream& stream) {
    validateOffsetsAndRowIds(offsets, row_ids, batch_size, max_total_values);
    dispatchOffsetDType(offsets.getDataType(), OffsetsToRowIdsOffsetFn{offsets, row_ids, batch_size, max_total_values, stream});
}

void rowPartitionValidateOffsetsDebug(const Tensor& offsets,
                                      Tensor& validation_error_bits,
                                      uint64_t batch_size,
                                      uint64_t max_total_values,
                                      Stream& stream) {
    validateOffsetsAndErrorBits(offsets, validation_error_bits, batch_size);
    validation_error_bits.fill(0.0, stream);
    dispatchOffsetDType(offsets.getDataType(), ValidateOffsetsFn{offsets, validation_error_bits, batch_size, max_total_values, stream});
}

}  // namespace ThorImplementation
