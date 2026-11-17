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
    if (!tensor.isDenseContiguous()) {
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

void validateSegmentOffsets(const Tensor& reference,
                            const Tensor& segment_offsets,
                            uint64_t num_items,
                            uint64_t num_segments,
                            const char* reference_name) {
    requireDenseContiguousGpuTensor(segment_offsets, "segment_offsets");
    requireSameGpuPlacement(reference, segment_offsets, reference_name, "segment_offsets");
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

void validateScan(const Tensor& input, const Tensor& output, uint64_t num_items) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(output, "output", num_items);
    if (input.getDataType() != output.getDataType()) {
        throw std::invalid_argument("CUB scan input/output dtypes must match.");
    }
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB scan dtype " + dtypeName(input.getDataType()) + ".");
    }
}

void validateSegmentedScan(const Tensor& input,
                           const Tensor& output,
                           const Tensor& segment_offsets,
                           uint64_t num_items,
                           uint64_t num_segments) {
    requireDenseContiguousGpuTensor(input, "input");
    requireDenseContiguousGpuTensor(output, "output");
    requireSameGpuPlacement(input, output, "input", "output");
    requireStorageForNumItems(input, "input", num_items);
    requireStorageForNumItems(output, "output", num_items);
    if (input.getDataType() != output.getDataType()) {
        throw std::invalid_argument("CUB segmented scan input/output dtypes must match.");
    }
    if (!isCubScanDTypeSupported(input.getDataType())) {
        throw std::invalid_argument("Unsupported CUB segmented scan dtype " + dtypeName(input.getDataType()) + ".");
    }
    validateSegmentOffsets(input, segment_offsets, num_items, num_segments, "input");
}

}  // namespace ThorImplementation::CubDevicePrimitiveSupport

namespace ThorImplementation {

bool isCubScanDTypeSupported(DataType dtype) {
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
#if THOR_CUB_ENABLE_64BIT_SEGMENT_OFFSETS
        case DataType::UINT64:
            return true;
#endif
        default:
            return false;
    }
}

Tensor allocateCubTemporaryStorage(const TensorPlacement& placement, size_t bytes) {
    if (placement.getMemDevice() != TensorPlacement::MemDevices::GPU) {
        throw std::invalid_argument("CUB temporary storage must be allocated on GPU.");
    }
    return Tensor(placement, TensorDescriptor(DataType::UINT8, {static_cast<uint64_t>(std::max<size_t>(bytes, 1))}));
}

}  // namespace ThorImplementation
