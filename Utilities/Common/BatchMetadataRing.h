#pragma once

#include "DeepLearning/Implementation/ThorError.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/ScopedGpu.h"
#include "Utilities/Common/Stream.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {

/**
 * Per-batch metadata that a stamp's operations derive on the host and read on the device: a ring of slots that follows
 * the stamp's batch queue, as its NetworkInputs' staging slots do.
 *
 * An operation declares a table and its capacity in bytes once. Every batch then takes the slot of its queue slot:
 * the host fills the batch's tables into the slot's pinned arena, one after another (beginBatch, openTable /
 * closeTable per table, each table at most once); commitBatch copies the bytes they used to the slot's device arena
 * in one copy, on a pooled upload stream; each reading operation's stream waits on uploadedEvent before it reads its
 * table at deviceTable; and releaseBatch, on the stream that joins the whole batch's processing (its backward and
 * updates included), marks the slot free for its next batch. The device address of a table holds for the whole
 * batch, so an operation's forward and backward read the same table.
 *
 * Every wait is a queue depth of batches back: the host rewrites a slot's pinned arena once that slot's previous copy
 * is done (beginBatch), and the upload stream rewrites its device arena once that slot's previous batch is released.
 * The upload stream never waits on a batch's own work.
 *
 * A ring with no tables does nothing.
 */
class BatchMetadataRing {
   public:
    using TableId = uint32_t;
    static constexpr uint64_t kTableAlignmentBytes = 16;

    BatchMetadataRing() = default;
    BatchMetadataRing(const BatchMetadataRing &) = delete;
    BatchMetadataRing &operator=(const BatchMetadataRing &) = delete;
    ~BatchMetadataRing() { clearNoThrow(); }

    // Declare a table of at most capacityBytes a batch. Only before the first reserveSlots.
    TableId declareTable(uint64_t capacityBytes) {
        if (!slots.empty()) throw std::logic_error("BatchMetadataRing tables must be declared before its slots are allocated.");
        if (capacityBytes == 0) throw std::invalid_argument("BatchMetadataRing table capacity must be positive.");
        const uint64_t rounded = roundUp(capacityBytes);
        if (slotBytes > std::numeric_limits<uint64_t>::max() - rounded) throw std::overflow_error("BatchMetadataRing slot size overflow.");
        slotBytes += rounded;
        tables.push_back(Table{capacityBytes});
        return static_cast<TableId>(tables.size() - 1);
    }

    [[nodiscard]] uint32_t getNumTables() const { return static_cast<uint32_t>(tables.size()); }
    [[nodiscard]] uint32_t getDepth() const { return depth; }
    [[nodiscard]] uint64_t getSlotBytes() const { return slotBytes; }
    // The GPU of the slots (-1 before the first reserveSlots).
    [[nodiscard]] int32_t getGpu() const { return gpu; }

    // At least `slotCount` slots on GPU gpuNum (the queue depth). Slots are only ever added.
    void reserveSlots(int32_t gpuNum, uint32_t slotCount) {
        if (slotCount == 0) throw std::invalid_argument("BatchMetadataRing needs at least one slot.");
        if (depth != 0 && gpuNum != gpu) throw std::logic_error("BatchMetadataRing slots must stay on one GPU.");
        gpu = gpuNum;
        if (slotCount <= depth) return;
        depth = slotCount;
        if (tables.empty()) return;
        ScopedGpu scopedGpu(gpu);
        if (!uploadStream.has_value()) uploadStream = Stream::getNextUploadStream(static_cast<uint32_t>(gpu));
        while (slots.size() < depth) {
            Slot slot;
            check(cudaHostAlloc(reinterpret_cast<void **>(&slot.host), slotBytes, cudaHostAllocPortable), "cudaHostAlloc");
            const cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&slot.device), slotBytes);
            if (status != cudaSuccess) {
                (void)cudaFreeHost(slot.host);
                check(status, "cudaMalloc");
            }
            // Record both events once, so a slot's first waits are on events already reached.
            uploadStream->putEvent(slot.uploaded, /*enableTiming=*/false, /*expectingHostToWaitOnThisOne=*/true);
            uploadStream->putEvent(slot.released);
            slots.push_back(slot);
        }
    }

    // Start the batch of queue slot `slot`: its pinned arena is the host's to fill once its previous copy is done.
    void beginBatch(uint32_t slot) {
        if (!tables.empty()) {
            if (depth == 0) throw std::logic_error("BatchMetadataRing slots were not allocated.");
            if (slot >= depth) throw std::out_of_range("BatchMetadataRing queue slot is past its depth.");
        }
        // A submission that failed after committing its batch left no release; that slot keeps its earlier release.
        phase = Phase::Idle;
        fillBytes = 0;
        openTableId.reset();
        for (Table &table : tables) table.written = false;
        if (!tables.empty()) {
            currentSlot = slot;
            Slot &current = slots[currentSlot];
            if (current.uploadSubmitted) current.uploaded.synchronize();
        }
        phase = Phase::Filling;
    }

    // The pinned bytes of table `id` for this batch, its capacity long. Close it before opening another.
    [[nodiscard]] uint8_t *openTable(TableId id) {
        requireTable(id);
        if (phase != Phase::Filling) throw std::logic_error("BatchMetadataRing table opened outside a batch.");
        if (openTableId.has_value()) throw std::logic_error("BatchMetadataRing opens one table at a time.");
        if (tables[id].written) throw std::logic_error("BatchMetadataRing table written twice in one batch.");
        tables[id].offset = roundUp(fillBytes);
        openTableId = id;
        return slots[currentSlot].host + tables[id].offset;
    }

    // The table holds usedBytes this batch.
    void closeTable(TableId id, uint64_t usedBytes) {
        requireTable(id);
        if (!openTableId.has_value() || openTableId.value() != id) throw std::logic_error("BatchMetadataRing closed a table that is not open.");
        if (usedBytes > tables[id].capacityBytes) throw std::length_error("BatchMetadataRing table exceeds its capacity.");
        tables[id].usedBytes = usedBytes;
        tables[id].written = true;
        fillBytes = tables[id].offset + usedBytes;
        openTableId.reset();
    }

    // Where table `id` is on the device for this batch (valid for the whole batch, once uploadedEvent is reached).
    [[nodiscard]] const void *deviceTable(TableId id) const {
        requireTable(id);
        if (phase == Phase::Idle || !tables[id].written) throw std::logic_error("BatchMetadataRing table was not written this batch.");
        return slots[currentSlot].device + tables[id].offset;
    }

    // Reached once this batch's tables are on the device. A copy of the slot's event: it stays valid for the batch.
    [[nodiscard]] Event uploadedEvent() const {
        if (phase == Phase::Idle || tables.empty()) throw std::logic_error("BatchMetadataRing has no batch in progress.");
        return slots[currentSlot].uploaded;
    }

    // Copy the bytes this batch's tables used to the slot's device arena, once the slot's previous batch is released.
    void commitBatch() {
        if (phase != Phase::Filling) throw std::logic_error("BatchMetadataRing committed outside a batch.");
        if (openTableId.has_value()) throw std::logic_error("BatchMetadataRing committed with a table still open.");
        phase = Phase::Committed;
        if (tables.empty()) return;
        Slot &current = slots[currentSlot];
        ScopedGpu scopedGpu(gpu);
        uploadStream->waitEvent(current.released);
        if (fillBytes != 0) {
            check(cudaMemcpyAsync(current.device, current.host, fillBytes, cudaMemcpyHostToDevice, uploadStream->getStream()),
                  "cudaMemcpyAsync");
        }
        uploadStream->putEvent(current.uploaded, /*enableTiming=*/false, /*expectingHostToWaitOnThisOne=*/true);
        current.uploadSubmitted = true;
        lastCommittedBytes = fillBytes;
    }

    // The batch's processing is done once `joinedStream` reaches this point: its slot may take its next batch.
    void releaseBatch(const Stream &joinedStream) {
        if (phase != Phase::Committed) return;
        phase = Phase::Idle;
        if (tables.empty()) return;
        joinedStream.putEvent(slots[currentSlot].released);
    }

    // An event reached once every copy enqueued so far is done (nullopt when nothing is ever copied).
    [[nodiscard]] std::optional<Event> synchronizeEvent() const {
        if (!uploadStream.has_value()) return std::nullopt;
        return uploadStream->putEvent(false, true);
    }

    // The bytes the last committed batch copied.
    [[nodiscard]] uint64_t getLastCommittedBytes() const { return lastCommittedBytes; }

    // Free every slot once its copies are done; the tables go too.
    void clear() {
        std::optional<std::runtime_error> failure;
        for (Slot &slot : slots) {
            try {
                if (slot.uploadSubmitted) slot.uploaded.synchronize();
            } catch (const std::runtime_error &error) {
                if (!failure.has_value()) failure = error;
            }
        }
        releaseStorage();
        if (failure.has_value()) throw failure.value();
    }

   private:
    struct Table {
        uint64_t capacityBytes = 0;
        uint64_t offset = 0;
        uint64_t usedBytes = 0;
        bool written = false;
    };
    struct Slot {
        uint8_t *host = nullptr;
        uint8_t *device = nullptr;
        Event uploaded;
        Event released;
        bool uploadSubmitted = false;
    };
    enum class Phase { Idle, Filling, Committed };

    static void check(cudaError_t status, const char *call) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("BatchMetadataRing: ") + call + " failed: " + cudaGetErrorString(status));
        }
    }

    static uint64_t roundUp(uint64_t bytes) {
        if (bytes > std::numeric_limits<uint64_t>::max() - (kTableAlignmentBytes - 1)) {
            throw std::overflow_error("BatchMetadataRing table size overflow.");
        }
        return (bytes + kTableAlignmentBytes - 1) & ~(kTableAlignmentBytes - 1);
    }

    void requireTable(TableId id) const {
        if (id >= tables.size()) throw std::out_of_range("BatchMetadataRing table id is not declared.");
    }

    void releaseStorage() {
        if (!slots.empty()) {
            ScopedGpu scopedGpu(gpu);
            for (Slot &slot : slots) {
                if (slot.device != nullptr) (void)cudaFree(slot.device);
                if (slot.host != nullptr) (void)cudaFreeHost(slot.host);
            }
        }
        slots.clear();
        tables.clear();
        slotBytes = 0;
        depth = 0;
        gpu = -1;
        uploadStream.reset();
        phase = Phase::Idle;
        openTableId.reset();
        fillBytes = 0;
        lastCommittedBytes = 0;
    }

    void clearNoThrow() noexcept {
        try {
            clear();
        } catch (...) {
            // Teardown continues; the storage is released either way.
        }
    }

    std::vector<Table> tables;
    uint64_t slotBytes = 0;
    std::vector<Slot> slots;
    uint32_t depth = 0;
    int32_t gpu = -1;
    std::optional<Stream> uploadStream;
    Phase phase = Phase::Idle;
    uint32_t currentSlot = 0;
    uint64_t fillBytes = 0;
    std::optional<TableId> openTableId;
    uint64_t lastCommittedBytes = 0;
};

}  // namespace ThorImplementation
