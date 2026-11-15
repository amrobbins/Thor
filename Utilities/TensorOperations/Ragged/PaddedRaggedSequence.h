#pragma once

#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "DeepLearning/Implementation/Tensor/TensorSpan.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/Ragged/RaggedPartitionRequirement.h"

#include <cstdint>
#include <vector>

namespace ThorImplementation {

inline constexpr RaggedPartitionRequirement kPaddedRaggedSequencePlanningPartitionRequirement =
    RaggedPartitionRequirement::HOST_EXTENT;

// One dense physical representation of an entire ragged batch. Logical values
// remain packed [sum(L_i), C] plus canonical offsets; the padded representation
// is a compact NCHW tensor [B, C, 1, W] where W is the selected physical width
// for the current batch. Every row keeps its original batch index -- there is no
// grouping or reordering by sequence length.
struct PaddedRaggedSequencePlan {
    DataType valuesDataType = DataType::FP32;
    DataType offsetsDataType = DataType::UINT32;
    uint64_t batchSize = 0;
    uint64_t maxTotalValues = 0;
    uint64_t maxValuesPerRow = 0;
    uint64_t channels = 0;
    uint64_t activeValues = 0;
    uint64_t widthCapacity = 0;
    uint64_t valueElements = 0;
    uint64_t valueBytes = 0;

    [[nodiscard]] uint64_t logicalValueElements() const;
    [[nodiscard]] uint64_t denseCapacityValues() const;
    [[nodiscard]] uint64_t paddingValueCapacity() const;
    [[nodiscard]] uint64_t totalWorkspaceBytes() const { return valueBytes; }
    [[nodiscard]] bool empty() const { return activeValues == 0; }

    bool operator==(const PaddedRaggedSequencePlan& other) const = default;
};

// Prepare one compact [B,C,1,W] physical batch. widthCapacity is selected by the
// caller from the finite placement-time Conv1D width-capacity family.
// This function consumes only already-published scalar row-partition metadata;
// it never requires a full host offsets mirror and never reads device offsets.
[[nodiscard]] PaddedRaggedSequencePlan preparePaddedRaggedSequencePlan(
    const RowPartitionRuntime& rowPartition,
    uint64_t channels,
    DataType valuesDataType,
    uint64_t widthCapacity);

// T8A runtime owner for one compiler-level padded ragged physical value. One
// maximum-sized allocation is retained; the dense [B,C,1,W] batch at a width W
// is its first B*C*W elements (paddedValuesForWidth / paddedDimensionsForWidth),
// so consumers address it without a view. The active prefix of each row is semantically
// valid; the inactive tail is undefined by default. Consumers that require a
// canonical zero tail must establish it explicitly for the exact physical
// generation before use.
struct PaddedRaggedPackLaunchPlan;
using PaddedRaggedUnpackLaunchPlan = PaddedRaggedPackLaunchPlan;

class PaddedRaggedSequence {
   public:
    PaddedRaggedSequence(PaddedRaggedSequencePlan plan,
                         Tensor rowOffsets,
                         TensorPlacement placement,
                         uint64_t reservedWidthCapacity);

    [[nodiscard]] const PaddedRaggedSequencePlan& getPlan() const { return plan; }
    [[nodiscard]] Tensor getRowOffsets() const { return rowOffsets; }
    [[nodiscard]] Tensor getPaddedValuesStorage() const { return paddedValues; }
    // The [B, C, 1, W] shape of the dense batch at width W. Retained width-generic operators use it with
    // paddedValuesForWidth to pre-stamp one invocation for every allowed W without changing the current plan.
    // Throws unless 0 < W <= the reserved width and the storage exists.
    [[nodiscard]] std::vector<uint64_t> paddedDimensionsForWidth(uint64_t widthCapacity) const;
    // The first B*C*W elements of the storage: the dense batch at width W.
    [[nodiscard]] TensorSpan paddedValuesForWidth(uint64_t widthCapacity) const;
    [[nodiscard]] bool hasValueStorage() const { return paddedValues.isInitialized(); }
    [[nodiscard]] uint64_t valueWorkspaceBytes() const { return plan.valueBytes; }
    [[nodiscard]] uint64_t totalWorkspaceBytes() const { return plan.totalWorkspaceBytes(); }
    [[nodiscard]] uint64_t allocatedValueElements() const {
        return paddedValues.isInitialized() ? paddedValues.getTotalNumElements() : 0;
    }
    [[nodiscard]] uint64_t allocatedValueBytes() const {
        return paddedValues.isInitialized() ? paddedValues.getArraySizeInBytes() : 0;
    }

    // Change only the runtime-selected width/active population. The backing
    // allocation and all structural row-partition geometry remain fixed.
    void reconfigure(PaddedRaggedSequencePlan newPlan);

    // Copy only logical positions into the selected dense prefix. Inactive
    // selected-width tail storage remains untouched and semantically undefined.
    void packFrom(const Tensor& packedValues, Stream& stream);
    // Stamped path: launch geometry/index-width policy was prepared once for
    // this selected W during stamping.
    void packFrom(const Tensor& packedValues, const PaddedRaggedPackLaunchPlan& launchPlan, Stream& stream);

    // Exit adapter writes only logical positions back to packed storage; packed
    // spare capacity is left untouched.
    void unpackTo(Tensor& packedValues, Stream& stream) const;
    // Stamped path: inverse-transpose geometry/index-width policy was prepared
    // once for this selected W during stamping.
    void unpackTo(Tensor& packedValues, const PaddedRaggedUnpackLaunchPlan& launchPlan, Stream& stream) const;

   private:
    PaddedRaggedSequencePlan plan;
    Tensor rowOffsets;
    Tensor paddedValues;
    uint64_t reservedWidthCapacity = 0;
};

}  // namespace ThorImplementation
