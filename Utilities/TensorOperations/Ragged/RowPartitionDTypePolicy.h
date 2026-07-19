#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <cstdint>
#include <limits>

namespace ThorImplementation {

// Canonical first-class row-partition storage uses unsigned offsets. UINT32 is
// preferred unless a model's maximum packed value capacity exceeds UINT32.
inline constexpr DataType kDefaultRowPartitionOffsetDataType = DataType::UINT32;

[[nodiscard]] inline constexpr bool isCanonicalRowPartitionOffsetDataType(DataType dtype) {
    return dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

[[nodiscard]] inline constexpr uint64_t maxCanonicalRowPartitionOffsetValue(DataType dtype) {
    return dtype == DataType::UINT32 ? static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
                                     : (dtype == DataType::UINT64 ? std::numeric_limits<uint64_t>::max() : 0ULL);
}

[[nodiscard]] inline constexpr bool canonicalRowPartitionOffsetCanRepresent(DataType dtype, uint64_t value) {
    return isCanonicalRowPartitionOffsetDataType(dtype) && value <= maxCanonicalRowPartitionOffsetValue(dtype);
}

// cuDNN ragged-attention descriptors use signed 32-bit offsets. These are a
// backend ABI representation, not Thor's canonical RaggedTensor offset type.
[[nodiscard]] inline constexpr bool isCudnnRaggedOffsetDataType(DataType dtype) { return dtype == DataType::INT32; }

// cuDNN CTC consumes signed 32-bit lengths. Canonical offsets/lengths must be
// checked before an explicit device-side conversion to this backend ABI type.
[[nodiscard]] inline constexpr bool isCudnnCtcLengthDataType(DataType dtype) { return dtype == DataType::INT32; }

}  // namespace ThorImplementation
