#pragma once

#include <cstdint>

namespace ThorImplementation {

// Physical row-partition information consumed by an executing ragged operator.
//
// These flags deliberately describe information, not the current Tensor used
// to transport it. In particular, DEVICE_ACTIVE_COUNT means that the GPU
// consumer needs only one scalar logical packed-prefix extent; it does not mean
// that the full device offsets array is semantically required. HOST_EXTENT is
// likewise a host-side dispatch/planning requirement derived from the
// authoritative host partition. Requirements are independent and may be
// combined when one physical operation needs more than one form.
enum class RaggedPartitionRequirement : uint8_t {
    NONE = 0,
    HOST_EXTENT = 1U << 0U,
    DEVICE_ACTIVE_COUNT = 1U << 1U,
    DEVICE_OFFSETS = 1U << 2U,
};

[[nodiscard]] constexpr RaggedPartitionRequirement operator|(RaggedPartitionRequirement lhs,
                                                              RaggedPartitionRequirement rhs) noexcept {
    return static_cast<RaggedPartitionRequirement>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

[[nodiscard]] constexpr RaggedPartitionRequirement operator&(RaggedPartitionRequirement lhs,
                                                              RaggedPartitionRequirement rhs) noexcept {
    return static_cast<RaggedPartitionRequirement>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

constexpr RaggedPartitionRequirement& operator|=(RaggedPartitionRequirement& lhs, RaggedPartitionRequirement rhs) noexcept {
    lhs = lhs | rhs;
    return lhs;
}

[[nodiscard]] constexpr bool hasRaggedPartitionRequirement(RaggedPartitionRequirement requirements,
                                                            RaggedPartitionRequirement requirement) noexcept {
    return requirement == RaggedPartitionRequirement::NONE
               ? requirements == RaggedPartitionRequirement::NONE
               : (requirements & requirement) == requirement;
}

[[nodiscard]] constexpr bool consumesAnyRaggedPartitionInformation(RaggedPartitionRequirement requirements) noexcept {
    return requirements != RaggedPartitionRequirement::NONE;
}

}  // namespace ThorImplementation
