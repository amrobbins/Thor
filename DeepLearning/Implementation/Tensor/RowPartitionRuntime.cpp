#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "DeepLearning/Implementation/ThorError.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation {

namespace {

bool tryAccumulateProduct(uint64_t lhs, uint64_t rhs, uint64_t& accumulator) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    const uint64_t product = lhs * rhs;
    if (product > std::numeric_limits<uint64_t>::max() - accumulator) {
        return false;
    }
    accumulator += product;
    return true;
}

}  // namespace

RowPartitionRuntime::RowPartitionRuntime(Tensor offsets, RowPartitionDescriptor descriptor)
    : offsets(std::move(offsets)), descriptor(descriptor) {
    THOR_THROW_IF_FALSE(this->offsets.isInitialized());
    hostStateCarrier = this->offsets;
    THOR_THROW_IF_FALSE(this->offsets.getDescriptor() == descriptor.getOffsetsDescriptor());
    // When an offsets representation exists it also serves as a host-state carrier,
    // but it is not the semantic row-partition identity. HOST_EXTENT-only runtimes
    // instead carry the same publication on another tensor. Views are rejected so
    // one offsets representation still has exactly one host publication.
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

std::optional<uint64_t> RowPartitionRuntime::getPublishedHostActiveValueCountIfAvailable(
    const Tensor& carrier) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    return carrier.getRowPartitionHostActiveValueCount();
}

std::optional<uint64_t> RowPartitionRuntime::getPublishedHostOffsetIfAvailable(
    const Tensor& carrier, uint64_t row) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    return carrier.getRowPartitionHostOffset(row);
}

std::optional<uint64_t> RowPartitionRuntime::getPublishedHostNonEmptyRowCountIfAvailable(
    const Tensor& carrier) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    return carrier.getRowPartitionHostNonEmptyRowCount();
}

std::optional<uint64_t> RowPartitionRuntime::getPublishedHostNonEmptyRowCountForPrefixIfAvailable(
    const Tensor& carrier, uint64_t validRowCount) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    const auto memory = carrier.backingMemory;
    const auto hostState = memory->rowPartitionHostState;
    const auto* logicalWorkState = memory->rowPartitionLogicalWorkState.get();
    if (hostState == nullptr || logicalWorkState == nullptr ||
        logicalWorkState->nonEmptyRowCountPrefixPublicationGeneration != hostState->publicationGeneration ||
        validRowCount >= logicalWorkState->nonEmptyRowCountPrefix.size()) {
        return std::nullopt;
    }
    return logicalWorkState->nonEmptyRowCountPrefix[validRowCount];
}

std::optional<uint64_t> RowPartitionRuntime::getPublishedHostSumSquaredRowLengthsIfAvailable(
    const Tensor& carrier) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    return carrier.getRowPartitionHostSumSquaredRowLengths();
}

void RowPartitionRuntime::registerLogicalWorkNonEmptyRowCount(const Tensor& carrier) noexcept {
    try {
        if (!carrier.isInitialized()) {
            return;
        }
        const auto memory = carrier.backingMemory;
        if (memory->rowPartitionLogicalWorkState == nullptr) {
            memory->rowPartitionLogicalWorkState =
                std::make_unique<Tensor::BackingMemory::RowPartitionLogicalWorkState>();
        }
        auto& logicalWorkState = *memory->rowPartitionLogicalWorkState;
        logicalWorkState.needsNonEmptyRowCount = true;

        // Allocate reusable prefix storage only while stamping/registering.
        // Per-batch publication must not allocate memory on behalf of telemetry.
        const auto state = memory->rowPartitionHostState;
        size_t prefixSize = 0;
        if (state != nullptr) {
            prefixSize = state->offsets.size();
        } else {
            const std::vector<uint64_t> dims = carrier.getDimensions();
            if (dims.size() == 1) {
                prefixSize = static_cast<size_t>(dims[0]);
            }
        }
        if (prefixSize < 2) {
            return;
        }
        if (state != nullptr &&
            logicalWorkState.nonEmptyRowCountPrefix.size() == prefixSize &&
            logicalWorkState.nonEmptyRowCountPrefixPublicationGeneration == state->publicationGeneration) {
            return;
        }
        if (logicalWorkState.nonEmptyRowCountPrefix.size() != prefixSize) {
            logicalWorkState.nonEmptyRowCountPrefix.assign(prefixSize, 0);
        }
        logicalWorkState.nonEmptyRowCountPrefixPublicationGeneration = 0;

        if (state == nullptr) {
            return;
        }
        auto& prefix = logicalWorkState.nonEmptyRowCountPrefix;
        prefix[0] = 0;
        for (size_t row = 0; row + 1 < state->offsets.size(); ++row) {
            prefix[row + 1] = prefix[row] +
                static_cast<uint64_t>(state->offsets[row + 1] != state->offsets[row]);
        }
        state->nonEmptyRowCount = prefix.back();
        logicalWorkState.nonEmptyRowCountPrefixPublicationGeneration = state->publicationGeneration;
    } catch (...) {
        // Logical-work metadata is best-effort and must never make stamping fail.
    }
}

void RowPartitionRuntime::registerLogicalWorkRowPartitionPair(const Tensor& firstCarrier,
                                                              const Tensor& secondCarrier) noexcept {
    try {
        if (!firstCarrier.isInitialized() || !secondCarrier.isInitialized()) {
            return;
        }

        if (firstCarrier.backingMemory == secondCarrier.backingMemory) {
            const auto memory = firstCarrier.backingMemory;
            if (memory->rowPartitionLogicalWorkState == nullptr) {
                memory->rowPartitionLogicalWorkState =
                    std::make_unique<Tensor::BackingMemory::RowPartitionLogicalWorkState>();
            }
            auto& logicalWorkState = *memory->rowPartitionLogicalWorkState;
            logicalWorkState.needsSquaredRowLengthSum = true;

            size_t prefixSize = 0;
            const auto state = memory->rowPartitionHostState;
            if (state != nullptr) {
                prefixSize = state->offsets.size();
            } else {
                const std::vector<uint64_t> dims = firstCarrier.getDimensions();
                if (dims.size() == 1) {
                    prefixSize = static_cast<size_t>(dims[0]);
                }
            }
            if (prefixSize >= 2 && logicalWorkState.squaredRowLengthPrefix.size() != prefixSize) {
                logicalWorkState.squaredRowLengthPrefix.assign(prefixSize, 0);
                logicalWorkState.squaredRowLengthPrefixPublicationGeneration = 0;
            }
            if (state == nullptr || logicalWorkState.squaredRowLengthPrefix.size() != state->offsets.size()) {
                return;
            }
            if (logicalWorkState.squaredRowLengthPrefixPublicationGeneration == state->publicationGeneration &&
                state->sumSquaredRowLengths.has_value()) {
                return;
            }

            auto& prefix = logicalWorkState.squaredRowLengthPrefix;
            prefix[0] = 0;
            uint64_t sumSquaredRowLengths = 0;
            bool available = true;
            for (size_t row = 0; row + 1 < state->offsets.size(); ++row) {
                const uint64_t rowLength = state->offsets[row + 1] - state->offsets[row];
                if (!tryAccumulateProduct(rowLength, rowLength, sumSquaredRowLengths)) {
                    available = false;
                    break;
                }
                prefix[row + 1] = sumSquaredRowLengths;
            }
            state->sumSquaredRowLengths =
                available ? std::optional<uint64_t>(sumSquaredRowLengths) : std::nullopt;
            logicalWorkState.squaredRowLengthPrefixPublicationGeneration =
                available ? state->publicationGeneration : 0;
            return;
        }

        const auto firstMemory = firstCarrier.backingMemory;
        const auto secondMemory = secondCarrier.backingMemory;
        if (firstMemory->rowPartitionLogicalWorkState == nullptr) {
            firstMemory->rowPartitionLogicalWorkState =
                std::make_unique<Tensor::BackingMemory::RowPartitionLogicalWorkState>();
        }
        if (secondMemory->rowPartitionLogicalWorkState == nullptr) {
            secondMemory->rowPartitionLogicalWorkState =
                std::make_unique<Tensor::BackingMemory::RowPartitionLogicalWorkState>();
        }

        auto ensurePartner = [](auto& entries, const auto& partnerMemory) -> size_t {
            entries.erase(
                std::remove_if(entries.begin(), entries.end(), [](const auto& entry) { return entry.partner.expired(); }),
                entries.end());
            for (size_t i = 0; i < entries.size(); ++i) {
                const auto partner = entries[i].partner.lock();
                if (partner != nullptr && partner.get() == partnerMemory.get()) {
                    return i;
                }
            }
            entries.emplace_back();
            entries.back().partner = partnerMemory;
            return entries.size() - 1;
        };

        auto& firstPairs = firstMemory->rowPartitionLogicalWorkState->pairs;
        auto& secondPairs = secondMemory->rowPartitionLogicalWorkState->pairs;
        const size_t firstIndex = ensurePartner(firstPairs, secondMemory);
        const size_t secondIndex = ensurePartner(secondPairs, firstMemory);
        auto& firstEntry = firstPairs[firstIndex];
        auto& secondEntry = secondPairs[secondIndex];

        size_t prefixSize = 0;
        const std::vector<uint64_t> firstDims = firstCarrier.getDimensions();
        const std::vector<uint64_t> secondDims = secondCarrier.getDimensions();
        if (firstDims.size() == 1 && firstDims == secondDims) {
            prefixSize = static_cast<size_t>(firstDims[0]);
        }
        std::shared_ptr<std::vector<uint64_t>> prefix = firstEntry.rowLengthProductPrefix;
        if (prefix == nullptr) {
            prefix = secondEntry.rowLengthProductPrefix;
        }
        if (prefixSize >= 2 && (prefix == nullptr || prefix->size() != prefixSize)) {
            prefix = std::make_shared<std::vector<uint64_t>>(prefixSize, 0);
        }
        firstEntry.rowLengthProductPrefix = prefix;
        secondEntry.rowLengthProductPrefix = prefix;

        const auto firstState = firstMemory->rowPartitionHostState;
        const auto secondState = secondMemory->rowPartitionHostState;
        if (firstState == nullptr || secondState == nullptr || prefix == nullptr ||
            firstState->offsets.size() != secondState->offsets.size() ||
            prefix->size() != firstState->offsets.size()) {
            return;
        }

        if (firstEntry.localPublicationGeneration == firstState->publicationGeneration &&
            firstEntry.partnerPublicationGeneration == secondState->publicationGeneration &&
            firstEntry.sumRowLengthProducts.has_value()) {
            return;
        }

        (*prefix)[0] = 0;
        uint64_t sumRowLengthProducts = 0;
        bool available = true;
        for (size_t row = 0; row + 1 < firstState->offsets.size(); ++row) {
            const uint64_t firstLength = firstState->offsets[row + 1] - firstState->offsets[row];
            const uint64_t secondLength = secondState->offsets[row + 1] - secondState->offsets[row];
            if (!tryAccumulateProduct(firstLength, secondLength, sumRowLengthProducts)) {
                available = false;
                break;
            }
            (*prefix)[row + 1] = sumRowLengthProducts;
        }

        const std::optional<uint64_t> summary =
            available ? std::optional<uint64_t>(sumRowLengthProducts) : std::nullopt;
        firstEntry.localPublicationGeneration = firstState->publicationGeneration;
        firstEntry.partnerPublicationGeneration = secondState->publicationGeneration;
        firstEntry.sumRowLengthProducts = summary;
        secondEntry.localPublicationGeneration = secondState->publicationGeneration;
        secondEntry.partnerPublicationGeneration = firstState->publicationGeneration;
        secondEntry.sumRowLengthProducts = summary;
    } catch (...) {
        // Logical-work metadata is best-effort and must never make stamping fail.
    }
}

std::optional<uint64_t> RowPartitionRuntime::getPublishedHostRowLengthProductSumIfAvailable(
    const Tensor& firstCarrier, const Tensor& secondCarrier) noexcept {
    try {
        if (!firstCarrier.isInitialized() || !secondCarrier.isInitialized()) {
            return std::nullopt;
        }
        const auto firstMemory = firstCarrier.backingMemory;
        const auto secondMemory = secondCarrier.backingMemory;
        const auto firstState = firstMemory->rowPartitionHostState;
        const auto secondState = secondMemory->rowPartitionHostState;
        if (firstState == nullptr || secondState == nullptr) {
            return std::nullopt;
        }

        if (firstMemory == secondMemory) {
            return firstState->sumSquaredRowLengths;
        }

        auto findValidSummary = [](const auto& entries,
                                   const auto& localState,
                                   const auto& partnerMemory,
                                   const auto& partnerState) -> std::optional<uint64_t> {
            for (const auto& entry : entries) {
                const auto partner = entry.partner.lock();
                if (partner == nullptr || partner.get() != partnerMemory.get()) {
                    continue;
                }
                if (entry.localPublicationGeneration == localState->publicationGeneration &&
                    entry.partnerPublicationGeneration == partnerState->publicationGeneration) {
                    return entry.sumRowLengthProducts;
                }
                return std::nullopt;
            }
            return std::nullopt;
        };

        if (firstMemory->rowPartitionLogicalWorkState != nullptr) {
            if (const std::optional<uint64_t> summary =
                    findValidSummary(firstMemory->rowPartitionLogicalWorkState->pairs,
                                     firstState,
                                     secondMemory,
                                     secondState);
                summary.has_value()) {
                return summary;
            }
        }
        if (secondMemory->rowPartitionLogicalWorkState == nullptr) {
            return std::nullopt;
        }
        return findValidSummary(secondMemory->rowPartitionLogicalWorkState->pairs,
                                secondState,
                                firstMemory,
                                firstState);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<uint64_t> RowPartitionRuntime::getPublishedHostRowLengthProductSumForPrefixIfAvailable(
    const Tensor& firstCarrier, const Tensor& secondCarrier, uint64_t validRowCount) noexcept {
    try {
        if (!firstCarrier.isInitialized() || !secondCarrier.isInitialized()) {
            return std::nullopt;
        }
        const auto firstMemory = firstCarrier.backingMemory;
        const auto secondMemory = secondCarrier.backingMemory;
        const auto firstState = firstMemory->rowPartitionHostState;
        const auto secondState = secondMemory->rowPartitionHostState;
        if (firstState == nullptr || secondState == nullptr ||
            firstState->offsets.size() != secondState->offsets.size() ||
            validRowCount >= firstState->offsets.size()) {
            return std::nullopt;
        }

        if (firstMemory == secondMemory) {
            const auto* logicalWorkState = firstMemory->rowPartitionLogicalWorkState.get();
            if (logicalWorkState == nullptr ||
                logicalWorkState->squaredRowLengthPrefixPublicationGeneration != firstState->publicationGeneration ||
                logicalWorkState->squaredRowLengthPrefix.size() != firstState->offsets.size()) {
                return std::nullopt;
            }
            return logicalWorkState->squaredRowLengthPrefix[validRowCount];
        }

        auto findValidPrefix = [validRowCount](const auto& entries,
                                               const auto& localState,
                                               const auto& partnerMemory,
                                               const auto& partnerState) -> std::optional<uint64_t> {
            for (const auto& entry : entries) {
                const auto partner = entry.partner.lock();
                if (partner == nullptr || partner.get() != partnerMemory.get()) {
                    continue;
                }
                if (entry.localPublicationGeneration != localState->publicationGeneration ||
                    entry.partnerPublicationGeneration != partnerState->publicationGeneration ||
                    !entry.sumRowLengthProducts.has_value() ||
                    entry.rowLengthProductPrefix == nullptr ||
                    validRowCount >= entry.rowLengthProductPrefix->size()) {
                    return std::nullopt;
                }
                return entry.rowLengthProductPrefix->at(validRowCount);
            }
            return std::nullopt;
        };

        if (firstMemory->rowPartitionLogicalWorkState != nullptr) {
            if (const std::optional<uint64_t> summary =
                    findValidPrefix(firstMemory->rowPartitionLogicalWorkState->pairs,
                                    firstState,
                                    secondMemory,
                                    secondState);
                summary.has_value()) {
                return summary;
            }
        }
        if (secondMemory->rowPartitionLogicalWorkState == nullptr) {
            return std::nullopt;
        }
        return findValidPrefix(secondMemory->rowPartitionLogicalWorkState->pairs,
                               secondState,
                               firstMemory,
                               firstState);
    } catch (...) {
        return std::nullopt;
    }
}

void RowPartitionRuntime::publishValidatedHostState(Tensor carrier,
                                                    RowPartitionDescriptor descriptor,
                                                    RowPartitionId rowPartitionId,
                                                    std::vector<uint64_t> hostOffsets) {
    THOR_THROW_IF_FALSE(carrier.isInitialized());
    THOR_THROW_IF_FALSE(rowPartitionId != 0);
    THOR_THROW_IF_FALSE(hostOffsets.size() == descriptor.getBatchSize() + 1);
    THOR_THROW_IF_FALSE(!hostOffsets.empty());
    THOR_THROW_IF_FALSE(hostOffsets.front() == 0);

    auto& memory = carrier.backingMemory;
    auto* logicalWorkState = memory->rowPartitionLogicalWorkState.get();
    if (logicalWorkState != nullptr) {
        auto& pairEntries = logicalWorkState->pairs;
        pairEntries.erase(
            std::remove_if(pairEntries.begin(),
                           pairEntries.end(),
                           [](const auto& entry) { return entry.partner.expired(); }),
            pairEntries.end());

        // Prepare pair accumulators without heap allocation. Each live
        // registration keeps only one scalar accumulator plus a non-owning
        // snapshot pointer. The snapshot is validated again before publication.
        for (auto& entry : pairEntries) {
            entry.pendingPartnerState = nullptr;
            entry.pendingPartnerGeneration = 0;
            entry.pendingSumRowLengthProducts = 0;
            entry.pendingSummaryAvailable = false;
            const auto partnerMemory = entry.partner.lock();
            if (partnerMemory == nullptr || partnerMemory->rowPartitionHostState == nullptr ||
                partnerMemory->rowPartitionHostState->offsets.size() != hostOffsets.size()) {
                continue;
            }
            entry.pendingPartnerState = partnerMemory->rowPartitionHostState.get();
            entry.pendingPartnerGeneration = partnerMemory->rowPartitionHostState->publicationGeneration;
            entry.pendingSummaryAvailable = true;
            if (entry.rowLengthProductPrefix != nullptr &&
                entry.rowLengthProductPrefix->size() == hostOffsets.size()) {
                (*entry.rowLengthProductPrefix)[0] = 0;
            }
        }
    }

    uint64_t maxActiveRowLength = 0;
    std::optional<uint64_t> nonEmptyRowCount =
        logicalWorkState != nullptr && logicalWorkState->needsNonEmptyRowCount
            ? std::optional<uint64_t>(0)
            : std::nullopt;
    std::vector<uint64_t>* nonEmptyRowCountPrefix = nullptr;
    if (nonEmptyRowCount.has_value() && logicalWorkState != nullptr &&
        logicalWorkState->nonEmptyRowCountPrefix.size() == descriptor.getBatchSize() + 1) {
        nonEmptyRowCountPrefix = &logicalWorkState->nonEmptyRowCountPrefix;
        (*nonEmptyRowCountPrefix)[0] = 0;
    }
    std::optional<uint64_t> sumSquaredRowLengths =
        logicalWorkState != nullptr && logicalWorkState->needsSquaredRowLengthSum
            ? std::optional<uint64_t>(0)
            : std::nullopt;
    std::vector<uint64_t>* squaredRowLengthPrefix = nullptr;
    if (sumSquaredRowLengths.has_value() && logicalWorkState != nullptr &&
        logicalWorkState->squaredRowLengthPrefix.size() == descriptor.getBatchSize() + 1) {
        squaredRowLengthPrefix = &logicalWorkState->squaredRowLengthPrefix;
        (*squaredRowLengthPrefix)[0] = 0;
    }

    for (uint64_t row = 0; row < descriptor.getBatchSize(); ++row) {
        THOR_THROW_IF_FALSE(hostOffsets[row] <= hostOffsets[row + 1]);
        const uint64_t rowLength = hostOffsets[row + 1] - hostOffsets[row];
        if (descriptor.hasMaxValuesPerRow()) {
            THOR_THROW_IF_FALSE(rowLength <= descriptor.getMaxValuesPerRow());
        }
        maxActiveRowLength = std::max(maxActiveRowLength, rowLength);

        if (nonEmptyRowCount.has_value()) {
            nonEmptyRowCount.value() += rowLength != 0;
            if (nonEmptyRowCountPrefix != nullptr) {
                (*nonEmptyRowCountPrefix)[row + 1] = nonEmptyRowCount.value();
            }
        }
        if (sumSquaredRowLengths.has_value()) {
            if (!tryAccumulateProduct(rowLength, rowLength, sumSquaredRowLengths.value())) {
                sumSquaredRowLengths = std::nullopt;
            } else if (squaredRowLengthPrefix != nullptr) {
                (*squaredRowLengthPrefix)[row + 1] = sumSquaredRowLengths.value();
            }
        }

        if (logicalWorkState != nullptr) {
            for (auto& entry : logicalWorkState->pairs) {
                if (!entry.pendingSummaryAvailable || entry.pendingPartnerState == nullptr) {
                    continue;
                }
                const uint64_t partnerLength =
                    entry.pendingPartnerState->offsets[row + 1] - entry.pendingPartnerState->offsets[row];
                if (!tryAccumulateProduct(rowLength, partnerLength, entry.pendingSumRowLengthProducts)) {
                    entry.pendingSummaryAvailable = false;
                } else if (entry.rowLengthProductPrefix != nullptr &&
                           entry.rowLengthProductPrefix->size() == hostOffsets.size()) {
                    (*entry.rowLengthProductPrefix)[row + 1] = entry.pendingSumRowLengthProducts;
                }
            }
        }
    }
    THOR_THROW_IF_FALSE(hostOffsets.back() <= descriptor.getMaxTotalValues());

    const uint64_t activeValueCount = hostOffsets.back();
    const uint64_t publicationGeneration = carrier.setRowPartitionHostOffsets(rowPartitionId,
                                                                                std::move(hostOffsets),
                                                                                activeValueCount,
                                                                                maxActiveRowLength,
                                                                                nonEmptyRowCount,
                                                                                sumSquaredRowLengths);

    if (logicalWorkState != nullptr) {
        logicalWorkState->nonEmptyRowCountPrefixPublicationGeneration =
            nonEmptyRowCountPrefix != nullptr ? publicationGeneration : 0;
        logicalWorkState->squaredRowLengthPrefixPublicationGeneration =
            squaredRowLengthPrefix != nullptr && sumSquaredRowLengths.has_value() ? publicationGeneration : 0;
        for (auto& entry : logicalWorkState->pairs) {
            const auto partnerMemory = entry.partner.lock();
            const bool partnerSnapshotStillCurrent =
                partnerMemory != nullptr && entry.pendingPartnerState != nullptr &&
                partnerMemory->rowPartitionHostState != nullptr &&
                partnerMemory->rowPartitionHostState.get() == entry.pendingPartnerState &&
                partnerMemory->rowPartitionHostState->publicationGeneration == entry.pendingPartnerGeneration;
            if (!partnerSnapshotStillCurrent) {
                entry.localPublicationGeneration = 0;
                entry.partnerPublicationGeneration = 0;
                entry.sumRowLengthProducts = std::nullopt;
            } else {
                entry.localPublicationGeneration = publicationGeneration;
                entry.partnerPublicationGeneration = entry.pendingPartnerGeneration;
                entry.sumRowLengthProducts = entry.pendingSummaryAvailable
                    ? std::optional<uint64_t>(entry.pendingSumRowLengthProducts)
                    : std::nullopt;
            }
            entry.pendingPartnerState = nullptr;
            entry.pendingPartnerGeneration = 0;
            entry.pendingSumRowLengthProducts = 0;
            entry.pendingSummaryAvailable = false;
        }
    }
}

void RowPartitionRuntime::publishHostState(Tensor carrier,
                                           RowPartitionDescriptor descriptor,
                                           RowPartitionId rowPartitionId,
                                           std::vector<uint64_t> hostOffsets) {
    publishValidatedHostState(std::move(carrier), descriptor, rowPartitionId, std::move(hostOffsets));
}

void RowPartitionRuntime::propagateHostState(Tensor sourceCarrier, Tensor destinationCarrier) {
    THOR_THROW_IF_FALSE(sourceCarrier.isInitialized());
    THOR_THROW_IF_FALSE(destinationCarrier.isInitialized());
    const auto sourceState = sourceCarrier.backingMemory->rowPartitionHostState;
    if (sourceState == nullptr) {
        // Low-level implementation fixtures may connect structural carriers
        // before publishing a logical partition. Supported network execution
        // always publishes authoritative host state before notification.
        return;
    }
    THOR_THROW_IF_FALSE(sourceState->rowPartitionId != 0);
    THOR_THROW_IF_FALSE(!sourceState->offsets.empty());
    THOR_THROW_IF_FALSE(sourceState->offsets.front() == 0);
    THOR_THROW_IF_FALSE(sourceState->offsets.back() == sourceState->activeValueCount);
    (void)destinationCarrier.setRowPartitionHostOffsets(sourceState->rowPartitionId,
                                                        sourceState->offsets,
                                                        sourceState->activeValueCount,
                                                        sourceState->maxActiveRowLength,
                                                        sourceState->nonEmptyRowCount,
                                                        sourceState->sumSquaredRowLengths);
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

RowPartitionRuntime RowPartitionRuntime::fromOffsetsAndHostState(
    Tensor offsets,
    RowPartitionDescriptor descriptor,
    RowPartitionId rowPartitionId,
    std::vector<uint64_t> hostOffsets) {
    THOR_THROW_IF_FALSE(rowPartitionId != 0);
    RowPartitionRuntime runtime(std::move(offsets), descriptor);
    runtime.rowPartitionId = rowPartitionId;
    runtime.setHostOffsets(std::move(hostOffsets));
    return runtime;
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
    publishValidatedHostState(hostStateCarrier, descriptor, rowPartitionId, std::move(hostOffsets));
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

uint64_t RowPartitionRuntime::requireHostOffset(uint64_t row) const {
    THOR_THROW_IF_FALSE(initialized);
    if (row > getBatchSize())
        throw std::out_of_range("RowPartitionRuntime host offset row is outside [0, batch_size].");

    const std::optional<uint64_t> hostOffset = hostStateCarrier.getRowPartitionHostOffset(row);
    if (!hostOffset.has_value()) {
        throw std::runtime_error(
            "RowPartitionRuntime has no authoritative host row partition bound for this batch.");
    }
    return hostOffset.value();
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
    // runtime extent, even though the eventual CUDA consumer reads only the
    // explicitly selected device execution representation.
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
