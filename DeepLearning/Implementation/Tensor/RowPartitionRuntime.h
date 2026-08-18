#pragma once

#include "DeepLearning/Implementation/Tensor/RowPartitionDescriptor.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"

#include <cstdint>
#include <optional>

namespace ThorImplementation {

// Runtime state shared by tensors that preserve one ragged row partition.
//
// The offsets tensor is the semantic source of truth. For batch size B,
// offsets[B] is the exclusive logical end of packed values; physical capacity
// beyond that point is undefined storage. Consumers that intentionally launch
// over a larger physical extent must sanitize precisely their own over-read
// region before consuming it.
//
// hostActiveValueCount is only
// a host-side cache of offsets[batchSize], used by execution paths that must choose
// capacity-dependent work without synchronously reading device offsets. Generic
// Tensor mutations invalidate the cache; inspectable CPU offsets are checked for
// agreement whenever the cache is published or consumed.
class RowPartitionRuntime {
   public:
    RowPartitionRuntime() = default;
    RowPartitionRuntime(Tensor offsets, RowPartitionDescriptor descriptor);

    bool isInitialized() const { return initialized; }

    Tensor getOffsets() const;
    RowPartitionDescriptor getDescriptor() const;
    uint64_t getBatchSize() const;
    uint64_t getMaxTotalValues() const;
    DataType getOffsetsDataType() const;
    TensorPlacement getPlacement() const;

    void setHostActiveValueCount(uint64_t activeValueCount);
    void clearHostActiveValueCount();
    [[nodiscard]] std::optional<uint64_t> getHostActiveValueCountIfAvailable() const;
    [[nodiscard]] uint64_t requireHostActiveValueCount() const;

    RaggedRuntimeExtent getRuntimeExtent(uint64_t elementsPerValue) const;

    bool describesSamePartition(const RowPartitionRuntime &rhs) const;
    bool sharesRuntimeStateWith(const RowPartitionRuntime &rhs) const;

   private:
    [[nodiscard]] uint64_t readCpuActiveValueCount() const;

    Tensor offsets;
    RowPartitionDescriptor descriptor;
    bool initialized = false;
};

}  // namespace ThorImplementation
