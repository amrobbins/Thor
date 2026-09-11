#pragma once

#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>

namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kPaddedRaggedSequenceKernelPartitionRequirement = RaggedPartitionRequirement::DEVICE_OFFSETS;

// Immutable launch geometry for the active-value packed <-> padded adapter.
// C==1 is a contiguous span copy; wider channel counts use the tiled transpose.
// Build one plan for each placement-time width choice. Runtime execution only
// consumes current canonical offsets and launches the preselected path/geometry
// and index-width specialization. The inverse adapter uses the same plan so the
// two directions cannot silently drift apart.
struct PaddedRaggedPackLaunchPlan {
    DataType valuesDataType = DataType::FP32;
    DataType offsetsDataType = DataType::UINT32;
    uint64_t batchSize = 0;
    uint64_t maxTotalValues = 0;
    uint64_t channels = 0;
    uint64_t widthCapacity = 0;
    uint64_t selectedValueElements = 0;
    uint64_t packedValueElements = 0;
    uint64_t selectedValueBytes = 0;
    uint64_t packedValueBytes = 0;
    uint32_t channelTileBlocks = 0;
    uint32_t timestepTileBlocks = 0;
    uint32_t rowBlocks = 0;
    // Coordinate/control arithmetic can stay 32-bit even when a linear
    // element address requires 64 bits. Keep these policies separate so large
    // tensors do not promote row/channel/timestep loops unnecessarily.
    bool use32BitCoordinateIndexing = false;
    // Linear packed/padded element addressing policy. Byte-address formation
    // may still widen at the final pointer arithmetic boundary.
    bool use32BitIndexing = false;

    // PRS9: when C==1 there is no transpose. Copy each active row as one
    // contiguous byte span using the same payload-aware row grouping policy as
    // Thor's optimized materialization kernels. These fields are zero/false for
    // the general tiled-transpose path.
    bool useChannelOneDirectCopy = false;
    uint32_t directCopyElementBytes = 0;
    uint32_t directCopyElementShift = 0;
    uint64_t directCopyMaxRowBytes = 0;
    uint32_t directCopyRowsPerBlock = 0;
    uint32_t directCopyRowBlocks = 0;
    bool directCopyUse32BitRowIndexing = false;
    bool directCopyUse32BitSpanIndexing = false;

    [[nodiscard]] bool empty() const { return widthCapacity == 0; }

    bool operator==(const PaddedRaggedPackLaunchPlan& other) const = default;
};

[[nodiscard]] PaddedRaggedPackLaunchPlan preparePaddedRaggedPackLaunchPlan(
    uint64_t batchSize,
    uint64_t maxTotalValues,
    uint64_t channels,
    uint64_t widthCapacity,
    DataType valuesDataType,
    DataType offsetsDataType);

using PaddedRaggedUnpackLaunchPlan = PaddedRaggedPackLaunchPlan;

[[nodiscard]] PaddedRaggedUnpackLaunchPlan preparePaddedRaggedUnpackLaunchPlan(
    uint64_t batchSize,
    uint64_t maxTotalValues,
    uint64_t channels,
    uint64_t widthCapacity,
    DataType valuesDataType,
    DataType offsetsDataType);

// Immutable launch geometry for zero-canonicalizing the inactive tail of one
// compact padded ragged representation. Build this once for each stamped width
// choice. Runtime execution only consumes canonical device offsets and dispatches
// the preselected lane/channel-group and 32/64-bit index specializations.
struct PaddedRaggedTailZeroLaunchPlan {
    DataType valuesDataType = DataType::FP32;
    DataType offsetsDataType = DataType::UINT32;
    uint64_t batchSize = 0;
    uint64_t channels = 0;
    uint64_t widthCapacity = 0;
    uint64_t selectedValueElements = 0;
    uint64_t selectedValueBytes = 0;
    uint64_t maxTailBytes = 0;
    uint32_t elementBytes = 0;
    uint32_t elementShift = 0;
    uint32_t lanesPerChannel = 0;
    uint32_t laneShift = 0;
    uint32_t channelsPerBlock = 0;
    uint32_t channelBlocks = 0;
    uint32_t rowBlocks = 0;
    uint32_t threadsPerBlock = 0;
    // Row/channel/width loop/control arithmetic policy.
    bool use32BitCoordinateIndexing = false;
    // Linear padded element addressing policy.
    bool use32BitIndexing = false;
    bool use32BitSpanIndexing = false;

    [[nodiscard]] bool empty() const { return widthCapacity == 0; }

    bool operator==(const PaddedRaggedTailZeroLaunchPlan& other) const = default;
};

[[nodiscard]] PaddedRaggedTailZeroLaunchPlan preparePaddedRaggedTailZeroLaunchPlan(
    uint64_t batchSize,
    uint64_t channels,
    uint64_t widthCapacity,
    DataType valuesDataType,
    DataType offsetsDataType);

// Canonicalize only the inactive selected-width tail in place:
//     padded[b,c,L_b:W] = all-bits-zero
// Active [0,L_b) storage and the reserved suffix beyond the compact [B,C,W]
// selected representation are not read or written. Width zero is a no-op.
void launchZeroPaddedRaggedSequenceTail(Tensor& paddedValues,
                                        const Tensor& rowOffsets,
                                        const PaddedRaggedTailZeroLaunchPlan& plan,
                                        Stream& stream);

// Transpose only logical active positions from packed [sum(L), C] into the
// selected padded [B, C, W] prefix. Inactive [L_b, W) storage and the reserved
// suffix beyond selected W are not touched.
void launchPackedToPaddedRaggedSequence(const Tensor& packedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& paddedValues,
                                        const PaddedRaggedPackLaunchPlan& plan,
                                        Stream& stream);

// Convenience path for non-stamped/direct callers. Stamped training code should
// prebuild PaddedRaggedPackLaunchPlan instances instead of reconstructing launch
// geometry on each execution.
void launchPackedToPaddedRaggedSequence(const Tensor& packedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& paddedValues,
                                        uint64_t batchSize,
                                        uint64_t channels,
                                        uint64_t widthCapacity,
                                        Stream& stream);

// Inverse tiled transpose. Only logical active positions are written to the
// packed destination; packed spare capacity is untouched.
void launchPaddedToPackedRaggedSequence(const Tensor& paddedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& packedValues,
                                        const PaddedRaggedUnpackLaunchPlan& plan,
                                        Stream& stream);

// Convenience path for direct/non-stamped callers.
void launchPaddedToPackedRaggedSequence(const Tensor& paddedValues,
                                        const Tensor& rowOffsets,
                                        Tensor& packedValues,
                                        uint64_t batchSize,
                                        uint64_t channels,
                                        uint64_t widthCapacity,
                                        Stream& stream);

}  // namespace ThorImplementation
