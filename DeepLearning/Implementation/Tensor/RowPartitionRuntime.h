#pragma once

#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation {

// Runtime state shared by tensors that preserve one ragged row partition.
//
// The offsets tensor is the semantic source of truth. For batch size B,
// offsets[B] is the exclusive logical end of packed values; physical capacity
// beyond that point is undefined storage. Consumers that intentionally launch
// over a larger physical extent must sanitize precisely their own over-read
// region before consuming it.
//
// hostActiveValueCount and hostMaxActiveRowLength are host-side caches derived
// from canonical offsets, used by execution paths that must choose capacity-dependent
// work without synchronously reading device offsets. Generic Tensor mutations
// invalidate these caches; inspectable CPU offsets are checked for agreement whenever
// cached values are published or consumed.
class RowPartitionRuntime {
   public:
    RowPartitionRuntime() = default;
    RowPartitionRuntime(Tensor offsets, RowPartitionDescriptor descriptor);

    bool isInitialized() const { return initialized; }

    Tensor getOffsets() const;
    RowPartitionDescriptor getDescriptor() const;
    uint64_t getBatchSize() const;
    uint64_t getMaxTotalValues() const;
    bool hasMaxValuesPerRow() const;
    uint64_t getMaxValuesPerRow() const;
    DataType getOffsetsDataType() const;
    TensorPlacement getPlacement() const;

    void setHostActiveValueCount(uint64_t activeValueCount);
    void clearHostActiveValueCount();
    [[nodiscard]] std::optional<uint64_t> getHostActiveValueCountIfAvailable() const;
    [[nodiscard]] uint64_t requireHostActiveValueCount() const;

    // Host dispatch scalar for consumers whose physical shape depends on the
    // longest logical row (for example ragged Conv1D). This is independent of
    // the optional full host offsets mirror and never triggers a device read.
    void setHostMaxActiveRowLength(uint64_t maxActiveRowLength);
    void clearHostMaxActiveRowLength();
    [[nodiscard]] std::optional<uint64_t> getHostMaxActiveRowLengthIfAvailable() const;
    [[nodiscard]] uint64_t requireHostMaxActiveRowLength() const;

    // Optional host mirror of the complete canonical offsets vector. This is
    // structural metadata only: GPU offsets remain the semantic source of truth.
    // Producers that already know row lengths/offsets on the host may publish
    // them here for diagnostics or consumers that truly need every boundary, without
    // introducing a device-to-host synchronization. Generic
    // tensor mutations invalidate this cache automatically.
    void setHostOffsets(std::vector<uint64_t> hostOffsets);
    void clearHostOffsets();
    [[nodiscard]] std::optional<std::vector<uint64_t>> getHostOffsetsIfAvailable() const;
    [[nodiscard]] std::vector<uint64_t> requireHostOffsets() const;

    RaggedRuntimeExtent getRuntimeExtent(uint64_t elementsPerValue) const;

    bool describesSamePartition(const RowPartitionRuntime &rhs) const;
    bool sharesRuntimeStateWith(const RowPartitionRuntime &rhs) const;

   private:
    [[nodiscard]] uint64_t readCpuActiveValueCount() const;
    [[nodiscard]] uint64_t readCpuMaxActiveRowLength() const;
    [[nodiscard]] std::vector<uint64_t> readCpuOffsets() const;
    [[nodiscard]] uint64_t maxActiveRowLengthFromHostOffsets(const std::vector<uint64_t>& hostOffsets) const;
    void validateHostOffsets(const std::vector<uint64_t> &hostOffsets) const;

    Tensor offsets;
    RowPartitionDescriptor descriptor;
    bool initialized = false;
};

}  // namespace ThorImplementation
