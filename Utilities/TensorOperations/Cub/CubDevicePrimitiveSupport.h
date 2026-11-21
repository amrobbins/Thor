#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ThorImplementation::CubDevicePrimitiveSupport {

[[nodiscard]] std::string dtypeName(DataType dtype);

void requireDenseContiguousGpuTensor(const Tensor& tensor, const char* name);
void requireSameGpuPlacement(const Tensor& a, const Tensor& b, const char* a_name, const char* b_name);
void requireStorageForNumItems(const Tensor& tensor, const char* name, uint64_t num_items);
void requireTempStorage(const Tensor& temp_storage, const TensorPlacement& placement, size_t temp_storage_bytes);

// The end bit of a radix sort whose keys are all at most largest_key. The bits above it are zero in every key, so
// sorting them would only add passes that move nothing. At least 1, so a sort whose only key is 0 still has a range.
// The temp-storage query and the sort must use the same end bit.
[[nodiscard]] inline int radixSortEndBit(uint64_t largest_key) {
    return std::max(1, static_cast<int>(std::bit_width(largest_key)));
}

}  // namespace ThorImplementation::CubDevicePrimitiveSupport
