#pragma once

#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"

#include <cstdint>

namespace ThorImplementation {

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

// Prepare one compact [B,C,1,W] physical view. widthCapacity is selected by the
// caller from the finite placement-time Conv1D width-capacity family.
// This function consumes only already-published scalar row-partition metadata;
// it never requires a full host offsets mirror and never reads device offsets.
[[nodiscard]] PaddedRaggedSequencePlan preparePaddedRaggedSequencePlan(
    const RowPartitionRuntime& rowPartition,
    uint64_t channels,
    DataType valuesDataType,
    uint64_t widthCapacity);

// T8A runtime owner for one compiler-level padded ragged physical value. One
// maximum-sized allocation is retained, while paddedTensor() exposes only the
// selected [B,C,1,W] prefix. The active prefix of each row is semantically
// valid; the inactive tail is undefined after arbitrary compatible producers.
// Entry packing canonicalizes that tail only because convolution consumers may
// require zeros at the representation boundary.
class PaddedRaggedSequence {
   public:
    PaddedRaggedSequence(PaddedRaggedSequencePlan plan,
                         Tensor rowOffsets,
                         TensorPlacement placement,
                         uint64_t reservedWidthCapacity);

    [[nodiscard]] const PaddedRaggedSequencePlan& getPlan() const { return plan; }
    [[nodiscard]] Tensor getRowOffsets() const { return rowOffsets; }
    [[nodiscard]] Tensor getPaddedValuesStorage() const { return paddedValues; }
    [[nodiscard]] Tensor paddedTensor() const;
    // Placement-time view used by retained width-generic operators to pre-stamp
    // one invocation for every allowed runtime W without mutating the current
    // runtime plan.
    [[nodiscard]] Tensor paddedTensorForWidth(uint64_t widthCapacity) const;
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

    // One fused direct loader copies logical positions and writes zero to each
    // inactive tail position within the selected dense prefix.
    void packFrom(const Tensor& packedValues, Stream& stream);

    // Consumer-owned copy from another retained representation with the same
    // structural/runtime plan. Active values are preserved and this object's
    // inactive selected-width tail is zeroed. The source is never modified.
    void sanitizedCopyFrom(const PaddedRaggedSequence& source, Stream& stream);

    // Exit adapter writes only logical positions back to packed storage; packed
    // spare capacity is left untouched.
    void unpackTo(Tensor& packedValues, Stream& stream) const;

   private:
    PaddedRaggedSequencePlan plan;
    Tensor rowOffsets;
    Tensor paddedValues;
    uint64_t reservedWidthCapacity = 0;
};

}  // namespace ThorImplementation
