#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

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

bool RowPartitionRuntime::hasMaxValuesPerRow() const { return getDescriptor().hasMaxValuesPerRow(); }

uint64_t RowPartitionRuntime::getMaxValuesPerRow() const { return getDescriptor().getMaxValuesPerRow(); }

DataType RowPartitionRuntime::getOffsetsDataType() const { return getDescriptor().getOffsetsDataType(); }

TensorPlacement RowPartitionRuntime::getPlacement() const {
    THOR_THROW_IF_FALSE(initialized);
    return offsets.getPlacement();
}

void RowPartitionRuntime::setHostActiveValueCount(uint64_t activeValueCount) {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(activeValueCount <= getMaxTotalValues());
    if (const std::optional<std::vector<uint64_t>> hostOffsets = offsets.getRowPartitionHostOffsets(); hostOffsets.has_value()) {
        validateHostOffsets(hostOffsets.value());
        THOR_THROW_IF_FALSE(hostOffsets->back() == activeValueCount);
    }
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

    if (const std::optional<std::vector<uint64_t>> hostOffsets = offsets.getRowPartitionHostOffsets(); hostOffsets.has_value()) {
        validateHostOffsets(hostOffsets.value());
        if (offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
            THOR_THROW_IF_FALSE(hostOffsets.value() == readCpuOffsets());
        }
        return hostOffsets->back();
    }

    if (offsets.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU) {
        return std::nullopt;
    }

    return readCpuActiveValueCount();
}


void RowPartitionRuntime::setHostMaxActiveRowLength(uint64_t maxActiveRowLength) {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(maxActiveRowLength <= getMaxTotalValues());
    if (hasMaxValuesPerRow()) {
        THOR_THROW_IF_FALSE(maxActiveRowLength <= getMaxValuesPerRow());
    }
    if (const std::optional<std::vector<uint64_t>> hostOffsets = offsets.getRowPartitionHostOffsets(); hostOffsets.has_value()) {
        validateHostOffsets(hostOffsets.value());
        THOR_THROW_IF_FALSE(maxActiveRowLengthFromHostOffsets(hostOffsets.value()) == maxActiveRowLength);
    }
    if (offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        THOR_THROW_IF_FALSE(maxActiveRowLength == readCpuMaxActiveRowLength());
    }
    offsets.setRowPartitionHostMaxActiveRowLength(maxActiveRowLength);
}

void RowPartitionRuntime::clearHostMaxActiveRowLength() {
    THOR_THROW_IF_FALSE(initialized);
    offsets.clearRowPartitionHostMaxActiveRowLength();
}

std::optional<uint64_t> RowPartitionRuntime::getHostMaxActiveRowLengthIfAvailable() const {
    THOR_THROW_IF_FALSE(initialized);
    if (const std::optional<uint64_t> cached = offsets.getRowPartitionHostMaxActiveRowLength(); cached.has_value()) {
        THOR_THROW_IF_FALSE(cached.value() <= getMaxTotalValues());
        if (hasMaxValuesPerRow()) {
            THOR_THROW_IF_FALSE(cached.value() <= getMaxValuesPerRow());
        }
        if (offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
            THOR_THROW_IF_FALSE(cached.value() == readCpuMaxActiveRowLength());
        }
        return cached;
    }
    if (const std::optional<std::vector<uint64_t>> hostOffsets = offsets.getRowPartitionHostOffsets(); hostOffsets.has_value()) {
        validateHostOffsets(hostOffsets.value());
        return maxActiveRowLengthFromHostOffsets(hostOffsets.value());
    }
    if (offsets.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU) {
        return std::nullopt;
    }
    return readCpuMaxActiveRowLength();
}

uint64_t RowPartitionRuntime::requireHostMaxActiveRowLength() const {
    const std::optional<uint64_t> maxActiveRowLength = getHostMaxActiveRowLengthIfAvailable();
    if (!maxActiveRowLength.has_value()) {
        throw std::runtime_error(
            "Ragged Conv1D requires host-resolved maximum row length metadata. "
            "The row-partition producer did not publish max_active_row_length, and Thor will not "
            "introduce an implicit device-to-host synchronization to recover it.");
    }
    return maxActiveRowLength.value();
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

uint64_t RowPartitionRuntime::readCpuMaxActiveRowLength() const {
    return maxActiveRowLengthFromHostOffsets(readCpuOffsets());
}

uint64_t RowPartitionRuntime::maxActiveRowLengthFromHostOffsets(const std::vector<uint64_t>& hostOffsets) const {
    validateHostOffsets(hostOffsets);
    uint64_t maxActiveRowLength = 0;
    for (uint64_t row = 0; row < getBatchSize(); ++row) {
        maxActiveRowLength = std::max(maxActiveRowLength, hostOffsets[row + 1] - hostOffsets[row]);
    }
    if (hasMaxValuesPerRow()) {
        THOR_THROW_IF_FALSE(maxActiveRowLength <= getMaxValuesPerRow());
    }
    return maxActiveRowLength;
}

std::vector<uint64_t> RowPartitionRuntime::readCpuOffsets() const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU);

    std::vector<uint64_t> hostOffsets(getBatchSize() + 1);
    switch (getOffsetsDataType()) {
        case DataType::UINT32: {
            const uint32_t *rawOffsets = offsets.getMemPtr<uint32_t>();
            for (uint64_t i = 0; i <= getBatchSize(); ++i) {
                hostOffsets[i] = static_cast<uint64_t>(rawOffsets[i]);
            }
            break;
        }
        case DataType::UINT64: {
            const uint64_t *rawOffsets = offsets.getMemPtr<uint64_t>();
            for (uint64_t i = 0; i <= getBatchSize(); ++i) {
                hostOffsets[i] = rawOffsets[i];
            }
            break;
        }
        default:
            THOR_UNREACHABLE();
    }
    validateHostOffsets(hostOffsets);
    return hostOffsets;
}

void RowPartitionRuntime::validateHostOffsets(const std::vector<uint64_t> &hostOffsets) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(hostOffsets.size() == getBatchSize() + 1);
    THOR_THROW_IF_FALSE(!hostOffsets.empty());
    THOR_THROW_IF_FALSE(hostOffsets.front() == 0);
    for (uint64_t row = 0; row < getBatchSize(); ++row) {
        THOR_THROW_IF_FALSE(hostOffsets[row] <= hostOffsets[row + 1]);
        if (hasMaxValuesPerRow()) {
            THOR_THROW_IF_FALSE(hostOffsets[row + 1] - hostOffsets[row] <= getMaxValuesPerRow());
        }
    }
    THOR_THROW_IF_FALSE(hostOffsets.back() <= getMaxTotalValues());
}

void RowPartitionRuntime::setHostOffsets(std::vector<uint64_t> hostOffsets) {
    THOR_THROW_IF_FALSE(initialized);
    validateHostOffsets(hostOffsets);
    if (offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
        // CPU offsets are inspectable. An explicitly published mirror must be an
        // exact copy, never a competing source of row-partition truth.
        THOR_THROW_IF_FALSE(hostOffsets == readCpuOffsets());
    }
    const uint64_t activeValueCount = hostOffsets.back();
    const uint64_t maxActiveRowLength = maxActiveRowLengthFromHostOffsets(hostOffsets);
    offsets.setRowPartitionHostOffsets(std::move(hostOffsets));
    offsets.setRowPartitionHostActiveValueCount(activeValueCount);
    offsets.setRowPartitionHostMaxActiveRowLength(maxActiveRowLength);
}

void RowPartitionRuntime::clearHostOffsets() {
    THOR_THROW_IF_FALSE(initialized);
    offsets.clearRowPartitionHostOffsets();
}

std::optional<std::vector<uint64_t>> RowPartitionRuntime::getHostOffsetsIfAvailable() const {
    THOR_THROW_IF_FALSE(initialized);
    if (const std::optional<std::vector<uint64_t>> cached = offsets.getRowPartitionHostOffsets(); cached.has_value()) {
        validateHostOffsets(cached.value());
        if (offsets.getPlacement().getMemDevice() == TensorPlacement::MemDevices::CPU) {
            THOR_THROW_IF_FALSE(cached.value() == readCpuOffsets());
        }
        return cached;
    }

    if (offsets.getPlacement().getMemDevice() != TensorPlacement::MemDevices::CPU) {
        return std::nullopt;
    }
    return readCpuOffsets();
}

std::vector<uint64_t> RowPartitionRuntime::requireHostOffsets() const {
    const std::optional<std::vector<uint64_t>> hostOffsets = getHostOffsetsIfAvailable();
    if (!hostOffsets.has_value()) {
        throw std::runtime_error(
            "RowPartitionRuntime requires host-known row offsets for this host-dispatched operation.");
    }
    return hostOffsets.value();
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
