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
    hostStateCarrier = this->offsets;
    THOR_THROW_IF_FALSE(this->offsets.getDescriptor() == descriptor.getOffsetsDescriptor());
    // The offsets allocation remains the transitional storage anchor for the
    // authoritative host publication, but it is no longer the semantic row-
    // partition identity. Views are rejected so one execution representation
    // still has exactly one host publication during this migration.
    THOR_THROW_IF_FALSE(this->offsets.isDenseContiguous());
    THOR_THROW_IF_FALSE(this->offsets.getStorageElementOffset() == 0);
    THOR_THROW_IF_FALSE(!this->offsets.hasCustomStrides());
    rowPartitionId = this->offsets.getTensorId();
    THOR_THROW_IF_FALSE(rowPartitionId != 0);
    initialized = true;
}


bool RowPartitionRuntime::hasPublishedHostState(const Tensor& carrier) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    return carrier.getRowPartitionHostId().has_value();
}

void RowPartitionRuntime::publishHostState(Tensor carrier,
                                           RowPartitionDescriptor descriptor,
                                           RowPartitionId rowPartitionId,
                                           std::vector<uint64_t> hostOffsets) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    THOR_THROW_IF_FALSE(rowPartitionId != 0);
    THOR_THROW_IF_FALSE(hostOffsets.size() == descriptor.getBatchSize() + 1);
    THOR_THROW_IF_FALSE(!hostOffsets.empty());
    THOR_THROW_IF_FALSE(hostOffsets.front() == 0);

    uint64_t maxActiveRowLength = 0;
    for (uint64_t row = 0; row < descriptor.getBatchSize(); ++row) {
        THOR_THROW_IF_FALSE(hostOffsets[row] <= hostOffsets[row + 1]);
        const uint64_t rowLength = hostOffsets[row + 1] - hostOffsets[row];
        if (descriptor.hasMaxValuesPerRow()) {
            THOR_THROW_IF_FALSE(rowLength <= descriptor.getMaxValuesPerRow());
        }
        maxActiveRowLength = std::max(maxActiveRowLength, rowLength);
    }
    THOR_THROW_IF_FALSE(hostOffsets.back() <= descriptor.getMaxTotalValues());

    const uint64_t activeValueCount = hostOffsets.back();
    carrier.setRowPartitionHostOffsets(
        rowPartitionId, std::move(hostOffsets), activeValueCount, maxActiveRowLength);
}

void RowPartitionRuntime::propagateHostState(Tensor sourceCarrier, Tensor destinationCarrier) {
    THOR_THROW_IF_FALSE(sourceCarrier.isInitialized());
    THOR_THROW_IF_FALSE(destinationCarrier.isInitialized());
    const std::optional<uint64_t> rowPartitionId = sourceCarrier.getRowPartitionHostId();
    const std::optional<std::vector<uint64_t>> hostOffsets = sourceCarrier.getRowPartitionHostOffsets();
    const std::optional<uint64_t> activeValueCount = sourceCarrier.getRowPartitionHostActiveValueCount();
    const std::optional<uint64_t> maxActiveRowLength = sourceCarrier.getRowPartitionHostMaxActiveRowLength();
    const bool hasAnyHostPublication = rowPartitionId.has_value() || hostOffsets.has_value() ||
                                       activeValueCount.has_value() || maxActiveRowLength.has_value();
    if (!hasAnyHostPublication) {
        // Transitional compatibility for direct implementation tests and legacy
        // internally-produced device-offset carriers.  External RP6B managed
        // inputs always publish authoritative host state before notification.
        return;
    }
    if (!rowPartitionId.has_value() || !hostOffsets.has_value() || !activeValueCount.has_value() ||
        !maxActiveRowLength.has_value()) {
        throw std::runtime_error(
            "RowPartitionRuntime cannot propagate an incomplete authoritative host partition publication.");
    }
    THOR_THROW_IF_FALSE(rowPartitionId.value() != 0);
    THOR_THROW_IF_FALSE(!hostOffsets->empty());
    THOR_THROW_IF_FALSE(hostOffsets->front() == 0);
    THOR_THROW_IF_FALSE(hostOffsets->back() == activeValueCount.value());
    destinationCarrier.setRowPartitionHostOffsets(rowPartitionId.value(),
                                                   hostOffsets.value(),
                                                   activeValueCount.value(),
                                                   maxActiveRowLength.value());
}

RowPartitionRuntime RowPartitionRuntime::fromHostStateCarrier(
    Tensor carrier, RowPartitionDescriptor descriptor) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    const std::optional<uint64_t> publishedId = carrier.getRowPartitionHostId();
    if (!publishedId.has_value()) {
        throw std::runtime_error(
            "RowPartitionRuntime host-state carrier has no authoritative host row partition publication.");
    }

    RowPartitionRuntime runtime;
    runtime.hostStateCarrier = std::move(carrier);
    runtime.rowPartitionId = publishedId.value();
    runtime.descriptor = descriptor;
    runtime.initialized = true;
    // Validate the complete publication against the descriptor immediately.
    (void)runtime.requireHostOffsets();
    return runtime;
}

RowPartitionRuntime RowPartitionRuntime::fromHostStateCarrier(
    Tensor carrier, uint64_t expectedBatchSize, uint64_t maxTotalValues) {
    // HOST_EXTENT consumers never inspect a device offsets payload. UINT64 is
    // used only as a canonical host-only descriptor dtype; the logical
    // partition id and authoritative offsets come from the carrier publication.
    RowPartitionRuntime runtime = fromHostStateCarrier(
        std::move(carrier), RowPartitionDescriptor(expectedBatchSize, maxTotalValues, DataType::UINT64));
    THOR_THROW_IF_FALSE(runtime.getBatchSize() == expectedBatchSize);
    return runtime;
}

RowPartitionRuntime RowPartitionRuntime::fromHostStateCarrier(
    Tensor carrier, uint64_t maxTotalValues) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    const std::optional<std::vector<uint64_t>> hostOffsets = carrier.getRowPartitionHostOffsets();
    if (!hostOffsets.has_value() || hostOffsets->empty()) {
        throw std::runtime_error(
            "RowPartitionRuntime host-state carrier has no authoritative offsets from which to infer batch size.");
    }
    return fromHostStateCarrier(
        std::move(carrier), hostOffsets->size() - 1, maxTotalValues);
}

Tensor RowPartitionRuntime::getOffsets() const {
    THOR_THROW_IF_FALSE(initialized);
    if (!offsets.isInitialized()) {
        throw std::runtime_error("RowPartitionRuntime has no device offsets representation.");
    }
    return offsets;
}

void RowPartitionRuntime::publishHostStateTo(Tensor carrier) const {
    THOR_THROW_IF_FALSE(initialized);
    publishHostState(std::move(carrier), descriptor, rowPartitionId, requireHostOffsets());
}

RowPartitionId RowPartitionRuntime::getRowPartitionId() const {
    THOR_THROW_IF_FALSE(initialized);
    return rowPartitionId;
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
    THOR_THROW_IF_FALSE(hostStateCarrier.isInitialized());
    return hostStateCarrier.getPlacement();
}

std::optional<uint64_t> RowPartitionRuntime::getHostActiveValueCountIfAvailable() const {
    THOR_THROW_IF_FALSE(initialized);
    const std::optional<uint64_t> activeValueCount = hostStateCarrier.getRowPartitionHostActiveValueCount();
    if (activeValueCount.has_value()) THOR_THROW_IF_FALSE(activeValueCount.value() <= getMaxTotalValues());
    return activeValueCount;
}

std::optional<uint64_t> RowPartitionRuntime::getHostMaxActiveRowLengthIfAvailable() const {
    THOR_THROW_IF_FALSE(initialized);
    const std::optional<uint64_t> maxActiveRowLength = hostStateCarrier.getRowPartitionHostMaxActiveRowLength();
    if (maxActiveRowLength.has_value()) {
        THOR_THROW_IF_FALSE(maxActiveRowLength.value() <= getMaxTotalValues());
        if (hasMaxValuesPerRow()) THOR_THROW_IF_FALSE(maxActiveRowLength.value() <= getMaxValuesPerRow());
    }
    return maxActiveRowLength;
}

uint64_t RowPartitionRuntime::requireHostMaxActiveRowLength() const {
    const std::optional<uint64_t> maxActiveRowLength = getHostMaxActiveRowLengthIfAvailable();
    if (!maxActiveRowLength.has_value()) {
        throw std::runtime_error(
            "RowPartitionRuntime has no authoritative host row partition bound for this batch.");
    }
    return maxActiveRowLength.value();
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
    const uint64_t activeValueCount = hostOffsets.back();
    const uint64_t maxActiveRowLength = maxActiveRowLengthFromHostOffsets(hostOffsets);
    hostStateCarrier.setRowPartitionHostOffsets(
        rowPartitionId, std::move(hostOffsets), activeValueCount, maxActiveRowLength);
}

bool RowPartitionRuntime::hasHostOffsets() const {
    THOR_THROW_IF_FALSE(initialized);
    return hostStateCarrier.getRowPartitionHostOffsets().has_value();
}

std::optional<std::vector<uint64_t>> RowPartitionRuntime::getHostOffsetsIfAvailable() const {
    THOR_THROW_IF_FALSE(initialized);
    const std::optional<std::vector<uint64_t>> hostOffsets = hostStateCarrier.getRowPartitionHostOffsets();
    if (hostOffsets.has_value()) validateHostOffsets(hostOffsets.value());
    return hostOffsets;
}

std::vector<uint64_t> RowPartitionRuntime::requireHostOffsets() const {
    const std::optional<std::vector<uint64_t>> hostOffsets = getHostOffsetsIfAvailable();
    if (!hostOffsets.has_value()) {
        throw std::runtime_error(
            "RowPartitionRuntime has no authoritative host row partition bound for this batch.");
    }
    return hostOffsets.value();
}

uint64_t RowPartitionRuntime::requireHostActiveValueCount() const {
    const std::optional<uint64_t> activeValueCount = getHostActiveValueCountIfAvailable();
    if (!activeValueCount.has_value()) {
        throw std::runtime_error(
            "RowPartitionRuntime has no authoritative host row partition bound for this batch.");
    }
    return activeValueCount.value();
}

RaggedRuntimeExtent RowPartitionRuntime::getRuntimeExtent(uint64_t elementsPerValue) const {
    THOR_THROW_IF_FALSE(initialized);
    // A device-side active-count view is an execution representation, not a
    // substitute for the semantic row partition. RP2 requires the authoritative
    // host partition to be bound before an executing RaggedTensor exposes a
    // runtime extent, even though the eventual CUDA consumer reads offsets[B].
    (void)requireHostOffsets();
    if (!offsets.isInitialized()) {
        throw std::runtime_error(
            "RowPartitionRuntime has no device active-count/offsets representation for runtime extent execution.");
    }
    return raggedRuntimeExtentFromOffsets(offsets, getBatchSize(), getMaxTotalValues(), elementsPerValue);
}

bool RowPartitionRuntime::describesSamePartition(const RowPartitionRuntime &rhs) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(rhs.initialized);
    return rowPartitionId == rhs.rowPartitionId && descriptor == rhs.descriptor;
}

bool RowPartitionRuntime::sharesPartitionWith(const RowPartitionRuntime &rhs) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(rhs.initialized);
    return rowPartitionId == rhs.rowPartitionId;
}

bool RowPartitionRuntime::sharesRuntimeStateWith(const RowPartitionRuntime &rhs) const {
    THOR_THROW_IF_FALSE(initialized);
    THOR_THROW_IF_FALSE(rhs.initialized);
    return hostStateCarrier.backingMemory == rhs.hostStateCarrier.backingMemory;
}

}  // namespace ThorImplementation
