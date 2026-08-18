#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <stdexcept>
#include <utility>

namespace ThorImplementation {

RowPartitionRuntime::RowPartitionRuntime(Tensor offsets, RowPartitionDescriptor descriptor)
    : offsets(std::move(offsets)), descriptor(descriptor) {
    THOR_THROW_IF_FALSE(this->offsets.isInitialized());
    THOR_THROW_IF_FALSE(this->offsets.getDescriptor() == descriptor.getOffsetsDescriptor());
    // A row partition is represented by one canonical dense offsets tensor. Keeping
    // the runtime cache on that allocation is safe only if views cannot reinterpret it.
    THOR_THROW_IF_FALSE(this->offsets.isDenseContiguous());
    THOR_THROW_IF_FALSE(this->offsets.getStorageElementOffset() == 0);
    THOR_THROW_IF_FALSE(!this->offsets.hasCustomStrides());
    initialized = true;
}

Tensor RowPartitionRuntime::getOffsets() const {
    THOR_THROW_IF_FALSE(initialized);
    return offsets;
}

RowPartitionDescriptor RowPartitionRuntime::getDescriptor() const {
    THOR_THROW_IF_FALSE(initialized);
    return descriptor;
}

uint64_t RowPartitionRuntime::getBatchSize() const { return getDescriptor().getBatchSize(); }

uint64_t RowPartitionRuntime::getMaxTotalValues() const { return getDescriptor().getMaxTotalValues(); }

DataType RowPartitionRuntime::getOffsetsDataType() const { return getDescriptor().getOffsetsDataType(); }

TensorPlacement RowPartitionRuntime::getPlacement() const {
    THOR_THROW_IF_FALSE(initialized);
    return offsets.getPlacement();
}

void RowPartitionRuntime::setHostActiveValueCount(uint64_t activeValueCount) {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(activeValueCount <= getMaxTotalValues());
    if (offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        // CPU offsets are immediately inspectable, so never permit the cache to
        // become a competing source of truth.
        THOR_THROW_IF_FALSE(activeValueCount == readCpuActiveValueCount());
    }
    offsets.setRowPartitionHostActiveValueCount(activeValueCount);
}

void RowPartitionRuntime::clearHostActiveValueCount() {
    THOR_THROW_IF_FALSE(initialized);
    offsets.clearRowPartitionHostActiveValueCount();
}

std::optional<uint64_t> RowPartitionRuntime::getHostActiveValueCountIfAvailable() const {
    THOR_THROW_IF_FALSE(initialized);
    if (const std::optional<uint64_t> cached = offsets.getRowPartitionHostActiveValueCount(); cached.has_value()) {
        THOR_THROW_IF_FALSE(cached.value() <= getMaxTotalValues());
        if (offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
            // Mutable CPU access can bypass Tensor's mutation hooks. Detect that
            // case here rather than silently returning stale structural state.
            THOR_THROW_IF_FALSE(cached.value() == readCpuActiveValueCount());
        }
        return cached;
    }

    if (offsets.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU) {
        return std::nullopt;
    }

    return readCpuActiveValueCount();
}

uint64_t RowPartitionRuntime::readCpuActiveValueCount() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);

    uint64_t activeValueCount = 0;
    switch (getOffsetsDataType()) {
        case DataType::UINT32:
            activeValueCount = static_cast<uint64_t>(offsets.getMemPtr<uint32_t>()[getBatchSize()]);
            break;
        case DataType::UINT64:
            activeValueCount = offsets.getMemPtr<uint64_t>()[getBatchSize()];
            break;
        default:
            THOR_UNREACHABLE();
    }

    THOR_THROW_IF_FALSE(activeValueCount <= getMaxTotalValues());
    return activeValueCount;
}

uint64_t RowPartitionRuntime::requireHostActiveValueCount() const {
    const std::optional<uint64_t> activeValueCount = getHostActiveValueCountIfAvailable();
    if (!activeValueCount.has_value()) {
        throw std::runtime_error(
            "RowPartitionRuntime requires a host-known active-value count for this host-dispatched operation.");
    }
    return activeValueCount.value();
}

RaggedRuntimeExtent RowPartitionRuntime::getRuntimeExtent(uint64_t elementsPerValue) const {
    THOR_THROW_IF_FALSE(initialized);
    return raggedRuntimeExtentFromOffsets(offsets, getBatchSize(), getMaxTotalValues(), elementsPerValue);
}

bool RowPartitionRuntime::describesSamePartition(const RowPartitionRuntime &rhs) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(rhs.initialized);
    return offsets == rhs.offsets && descriptor == rhs.descriptor;
}

bool RowPartitionRuntime::sharesRuntimeStateWith(const RowPartitionRuntime &rhs) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(rhs.initialized);
    return offsets.backingMemory == rhs.offsets.backingMemory;
}

}  // namespace ThorImplementation
