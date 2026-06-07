#include "Utilities/TensorOperations/Cub/CubDevicePrimitiveSupport.h"

#include <algorithm>
#include <limits>

namespace ThorImplementation::CubDevicePrimitiveSupport {

std::string dtypeName(DataType dtype) { return TensorDescriptor::getElementTypeName(dtype); }

void requireDenseContiguousGpuTensor(const Tensor& tensor, const char* name) {
    if (!tensor.isInitialized()) {
        throw std::invalid_argument(std::string(name) + " must be initialized.");
    }
    if (tensor.getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument(std::string(name) + " must be a GPU tensor.");
    }
    if (tensor.hasCustomStrides() || !tensor.isDenseContiguous()) {
        throw std::invalid_argument(std::string(name) + " must be dense contiguous.");
    }
}

void requireSameGpuPlacement(const Tensor& a, const Tensor& b, const char* a_name, const char* b_name) {
    if (a.getPlacement() != b.getPlacement()) {
        throw std::invalid_argument(std::string(a_name) + " and " + b_name + " must live on the same GPU placement.");
    }
}

void requireStorageForNumItems(const Tensor& tensor, const char* name, uint64_t num_items) {
    if (tensor.getTotalNumElements() < num_items) {
        throw std::invalid_argument(std::string(name) + " has fewer elements than num_items.");
    }
}

int checkedCubNumItems(uint64_t num_items) {
    if (num_items > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("CUB device primitive num_items exceeds CUB's int item-count limit.");
    }
    return static_cast<int>(num_items);
}

int64_t checkedCubInt64Count(uint64_t count, const char* name) {
    if (count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument(std::string(name) + " exceeds CUB's int64 count limit.");
    }
    return static_cast<int64_t>(count);
}

int fullEndBitFor(DataType dtype, int end_bit) {
    if (end_bit != 0) {
        return end_bit;
    }
    return static_cast<int>(TensorDescriptor::getElementSizeInBytes(dtype) * 8);
}

void validateBitRange(DataType dtype, int begin_bit, int end_bit) {
    const int full_bits = static_cast<int>(TensorDescriptor::getElementSizeInBytes(dtype) * 8);
    if (begin_bit < 0 || end_bit <= begin_bit || end_bit > full_bits) {
        throw std::invalid_argument("Invalid CUB radix-sort bit range for dtype " + dtypeName(dtype) + ".");
    }
}

void requireTempStorage(const Tensor& temp_storage, const TensorPlacement& placement, size_t temp_storage_bytes) {
    requireDenseContiguousGpuTensor(temp_storage, "temp_storage");
    if (temp_storage.getPlacement() != placement) {
        throw std::invalid_argument("temp_storage must live on the same GPU placement as the operation tensors.");
    }
    if (temp_storage.getDataType() != DataType::UINT8) {
        throw std::invalid_argument("temp_storage must have dtype uint8.");
    }
    if (temp_storage.getArraySizeInBytes() < temp_storage_bytes) {
        throw std::invalid_argument("temp_storage has fewer bytes than temp_storage_bytes.");
    }
}

void* mutableCubTempStoragePtr(const Tensor& temp_storage) {
    return const_cast<void*>(static_cast<const void*>(temp_storage.getMemPtr<void>()));
}

void validateSortKeys(const Tensor& keys_in, const Tensor& keys_out, uint64_t num_items, int begin_bit, int end_bit) {
    requireDenseContiguousGpuTensor(keys_in, "keys_in");
    requireDenseContiguousGpuTensor(keys_out, "keys_out");
    requireSameGpuPlacement(keys_in, keys_out, "keys_in", "keys_out");
    requireStorageForNumItems(keys_in, "keys_in", num_items);
    requireStorageForNumItems(keys_out, "keys_out", num_items);
    if (keys_in.getDataType() != keys_out.getDataType()) {
        throw std::invalid_argument("CUB radix-sort key input/output dtypes must match.");
    }
    if (!isCubRadixSortKeyDTypeSupported(keys_in.getDataType())) {
        throw std::invalid_argument("Unsupported CUB radix-sort key dtype " + dtypeName(keys_in.getDataType()) + ".");
    }
    validateBitRange(keys_in.getDataType(), begin_bit, end_bit);
}

void validateSortPairs(const Tensor& keys_in,
                       const Tensor& keys_out,
                       const Tensor& values_in,
                       const Tensor& values_out,
                       uint64_t num_items,
                       int begin_bit,
                       int end_bit) {
    validateSortKeys(keys_in, keys_out, num_items, begin_bit, end_bit);
    requireDenseContiguousGpuTensor(values_in, "values_in");
    requireDenseContiguousGpuTensor(values_out, "values_out");
    requireSameGpuPlacement(keys_in, values_in, "keys_in", "values_in");
    requireSameGpuPlacement(keys_in, values_out, "keys_in", "values_out");
    requireStorageForNumItems(values_in, "values_in", num_items);
    requireStorageForNumItems(values_out, "values_out", num_items);
    if (values_in.getDataType() != values_out.getDataType()) {
        throw std::invalid_argument("CUB radix-sort value input/output dtypes must match.");
    }
    if (!isCubRadixSortValueDTypeSupported(values_in.getDataType())) {
        throw std::invalid_argument("Unsupported CUB radix-sort index value dtype " + dtypeName(values_in.getDataType()) + ".");
    }
}

void validateSegmentOffsets(const Tensor& keys_in, const Tensor& segment_offsets, uint64_t num_items, uint64_t num_segments) {
    requireDenseContiguousGpuTensor(segment_offsets, "segment_offsets");
    requireSameGpuPlacement(keys_in, segment_offsets, "keys_in", "segment_offsets");
    if (num_segments == std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("num_segments is too large for a contiguous segment-offset tensor.");
    }
    requireStorageForNumItems(segment_offsets, "segment_offsets", num_segments + 1);
    if (!isCubSegmentOffsetDTypeSupported(segment_offsets.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segment-offset dtype " + dtypeName(segment_offsets.getDataType()) + ".");
    }
    static_cast<void>(checkedCubInt64Count(num_items, "num_items"));
    static_cast<void>(checkedCubInt64Count(num_segments, "num_segments"));
}

void validateSegmentedSortKeys(const Tensor& keys_in,
                               const Tensor& keys_out,
                               const Tensor& segment_offsets,
                               uint64_t num_items,
                               uint64_t num_segments,
                               int begin_bit,
                               int end_bit) {
    validateSortKeys(keys_in, keys_out, num_items, begin_bit, end_bit);
    validateSegmentOffsets(keys_in, segment_offsets, num_items, num_segments);
}

void validateSegmentedSortPairs(const Tensor& keys_in,
                                const Tensor& keys_out,
                                const Tensor& values_in,
                                const Tensor& values_out,
                                const Tensor& segment_offsets,
                                uint64_t num_items,
                                uint64_t num_segments,
                                int begin_bit,
                                int end_bit) {
    validateSortPairs(keys_in, keys_out, values_in, values_out, num_items, begin_bit, end_bit);
    validateSegmentOffsets(keys_in, segment_offsets, num_items, num_segments);
}

void validateRle(const Tensor& input, const Tensor& unique_out, const Tensor& counts_out, const Tensor& num_runs_out, uint64_t num_items) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(unique_out, "unique_out");
    requireDenseContiguousGpuTensor(counts_out, "counts_out");
    requireDenseContiguousGpuTensor(num_runs_out, "num_runs_out");
    requireSameGpuPlacement(input, unique_out, "input", "unique_out");
    requireSameGpuPlacement(input, counts_out, "input", "counts_out");
    requireSameGpuPlacement(input, num_runs_out, "input", "num_runs_out");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(unique_out, "unique_out", num_items);
    requireStorageForNumItems(counts_out, "counts_out", num_items);
    requireStorageForNumItems(num_runs_out, "num_runs_out", 1);
    if (input.getDataType() != unique_out.getDataType()) {
        throw std::invalid_argument("CUB RLE input and unique_out dtypes must match.");
    }
    if (!isCubRunLengthEncodeDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB RLE dtype " + dtypeName(input.getDataType()) + ".");
    }
    if (counts_out.getDataType() != DataType::UINT32 || num_runs_out.getDataType() != DataType::UINT32) {
        throw std::invalid_argument("CUB RLE counts_out and num_runs_out must have dtype uint32.");
    }
}

void validateExclusiveSum(const Tensor& input, const Tensor& output, uint64_t num_items) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(output, "output", num_items);
    if (input.getDataType() != output.getDataType()) {
        throw std::invalid_argument("CUB exclusive-sum input/output dtypes must match.");
    }
    if (!isCubExclusiveSumDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB exclusive-sum dtype " + dtypeName(input.getDataType()) + ".");
    }
}

}  // namespace ThorImplementation::CubDevicePrimitiveSupport

namespace ThorImplementation {

bool isCubRadixSortKeyDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FP64:
            return true;
#endif
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return true;
#endif
        default:
            return false;
    }
}

bool isCubRadixSortValueDTypeSupported(DataType dtype) {
    // SortPairs is Thor's internal argsort/segmented-argsort backend: the
    // carried payload is an index vector, not an arbitrary user value tensor.
    // Arbitrary value reordering belongs in take_along_axis after argsort.
    // Keeping pair values to unsigned index dtypes avoids a key-dtype x
    // arbitrary-value-dtype template-instantiation explosion in CUB.
    if (dtype == DataType::UINT32) {
        return true;
    }
#if THOR_CUB_ENABLE_64BIT_TYPES
    if (dtype == DataType::UINT64) {
        return true;
    }
#endif
    return false;
}

bool isCubRunLengthEncodeDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
        case DataType::INT64:
        case DataType::FP64:
            return true;
#endif
#if THOR_CUB_ENABLE_FP8_TYPES
        case DataType::FP8_E4M3:
        case DataType::FP8_E5M2:
            return true;
#endif
        default:
            return false;
    }
}

bool isCubExclusiveSumDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:
        case DataType::FP16:
        case DataType::BF16:
        case DataType::FP32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
        case DataType::FP64:
            return true;
#endif
        default:
            return false;
    }
}

bool isCubSegmentOffsetDTypeSupported(DataType dtype) {
    switch (dtype) {
        case DataType::UINT32:
            return true;
#if THOR_CUB_ENABLE_64BIT_TYPES
        case DataType::UINT64:
            return true;
#endif
        default:
            return false;
    }
}

CubTemporaryStoragePlan cubTemporaryStoragePlan(const TensorPlacement& placement, size_t bytes) {
    if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("CUB temporary storage must be planned on GPU placement.");
    }
    CubTemporaryStoragePlan plan;
    plan.placement = placement;
    plan.bytes = std::max<size_t>(bytes, 1);
    return plan;
}

CubTemporaryStoragePlan cubMaxTemporaryStoragePlan(std::initializer_list<CubTemporaryStoragePlan> plans) {
    if (plans.size() == 0) {
        throw std::invalid_argument("At least one CUB temporary-storage plan is required.");
    }

    auto it = plans.begin();
    CubTemporaryStoragePlan combined = cubTemporaryStoragePlan(it->placement, it->bytes);
    ++it;
    for (; it != plans.end(); ++it) {
        if (it->placement != combined.placement) {
            throw std::invalid_argument("All CUB temporary-storage plans must use the same GPU placement.");
        }
        combined.bytes = std::max(combined.bytes, std::max<size_t>(it->bytes, 1));
    }
    return combined;
}

Tensor allocateCubTemporaryStorage(const TensorPlacement& placement, size_t bytes) {
    const CubTemporaryStoragePlan plan = cubTemporaryStoragePlan(placement, bytes);
    return allocateCubTemporaryStorage(plan);
}

Tensor allocateCubTemporaryStorage(const CubTemporaryStoragePlan& plan) {
    if (plan.placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("CUB temporary storage must be allocated on GPU.");
    }
    return Tensor(plan.placement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(std::max<size_t>(plan.bytes, 1))}));
}

}  // namespace ThorImplementation
