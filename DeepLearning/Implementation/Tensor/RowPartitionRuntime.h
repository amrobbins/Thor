#pragma once

#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation {

using RowPartitionId = uint64_t;

// Runtime state shared by tensors that preserve one ragged row partition.
//
// Complete host offsets are the authoritative semantic partition. For batch size B,
// hostOffsets[B] is the exclusive logical end of packed values and row boundaries
// are hostOffsets[i:i+2]. activeValueCount and maxActiveRowLength are derived from
// that one publication and can never be updated independently.
//
// The offsets Tensor is an execution representation only. RP1 intentionally keeps
// that representation present everywhere for compatibility; later migration steps
// may materialize it only for consumers that need row boundaries on device. Generic
// Tensor payload mutations do not redefine or invalidate the host partition.
class RowPartitionRuntime {
   public:
    RowPartitionRuntime() = default;
    RowPartitionRuntime(Tensor offsets, RowPartitionDescriptor descriptor);

    // Build a semantic runtime view from authoritative host state published on
    // an arbitrary physical carrier (for example packed values). The carrier is
    // not treated as a device offsets representation.
    [[nodiscard]] static RowPartitionRuntime fromHostStateCarrier(
        Tensor carrier, RowPartitionDescriptor descriptor);
    [[nodiscard]] static RowPartitionRuntime fromHostStateCarrier(
        Tensor carrier, uint64_t expectedBatchSize, uint64_t maxTotalValues);
    [[nodiscard]] static RowPartitionRuntime fromHostStateCarrier(
        Tensor carrier, uint64_t maxTotalValues);

    // Placement-time diagnostics may probe whether a batch has published its
    // authoritative host partition yet. This does not validate or interpret
    // the carrier payload and is intentionally safe before the first batch.
    [[nodiscard]] static bool hasPublishedHostState(const Tensor& carrier);

    // Publish authoritative host partition state with an explicit logical id on
    // any physical carrier. The carrier payload is untouched.
    static void publishHostState(Tensor carrier,
                                 RowPartitionDescriptor descriptor,
                                 RowPartitionId rowPartitionId,
                                 std::vector<uint64_t> hostOffsets);

    // Copy an already-authoritative host publication from one physical carrier
    // to another without interpreting either tensor payload. This is the
    // partition-preserving-layer primitive used when the structural input is a
    // HOST_EXTENT carrier rather than a device offsets tensor.
    static void propagateHostState(Tensor sourceCarrier, Tensor destinationCarrier);

    bool isInitialized() const { return initialized; }

    Tensor getOffsets() const;
    [[nodiscard]] bool hasDeviceOffsetsRepresentation() const {
        return initialized && offsets.isInitialized();
    }

    // Publish this authoritative logical partition on another physical tensor.
    // This changes only the carrier's host structural metadata; it does not copy
    // or reinterpret the carrier payload.
    void publishHostStateTo(Tensor carrier) const;
    RowPartitionId getRowPartitionId() const;
    RowPartitionDescriptor getDescriptor() const;
    uint64_t getBatchSize() const;
    uint64_t getMaxTotalValues() const;
    bool hasMaxValuesPerRow() const;
    uint64_t getMaxValuesPerRow() const;
    DataType getOffsetsDataType() const;
    TensorPlacement getPlacement() const;

    // Publish the complete authoritative partition. This is the only operation
    // that may change logical row boundaries for an existing runtime state.
    // activeValueCount and maxActiveRowLength are derived atomically from it.
    void setHostOffsets(std::vector<uint64_t> hostOffsets);
    [[nodiscard]] bool hasHostOffsets() const;
    [[nodiscard]] std::optional<std::vector<uint64_t>> getHostOffsetsIfAvailable() const;
    [[nodiscard]] std::vector<uint64_t> requireHostOffsets() const;

    [[nodiscard]] std::optional<uint64_t> getHostActiveValueCountIfAvailable() const;
    [[nodiscard]] uint64_t requireHostActiveValueCount() const;
    [[nodiscard]] std::optional<uint64_t> getHostMaxActiveRowLengthIfAvailable() const;
    [[nodiscard]] uint64_t requireHostMaxActiveRowLength() const;

    RaggedRuntimeExtent getRuntimeExtent(uint64_t elementsPerValue) const;

    bool describesSamePartition(const RowPartitionRuntime &rhs) const;
    bool sharesPartitionWith(const RowPartitionRuntime &rhs) const;
    bool sharesRuntimeStateWith(const RowPartitionRuntime &rhs) const;

   private:
    [[nodiscard]] uint64_t maxActiveRowLengthFromHostOffsets(const std::vector<uint64_t>& hostOffsets) const;
    void validateHostOffsets(const std::vector<uint64_t> &hostOffsets) const;

    Tensor offsets;
    Tensor hostStateCarrier;
    RowPartitionId rowPartitionId = 0;
    RowPartitionDescriptor descriptor;
    bool initialized = false;
};

}  // namespace ThorImplementation
