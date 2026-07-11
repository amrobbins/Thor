#pragma once

#include "DeepLearning/Api/Loaders/Loader.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "Utilities/Common/Event.h"

#include <memory>
#include <optional>
#include <string>
#include <set>
#include <utility>
#include <vector>

namespace Thor {

class BatchSession;

/**
 * RAII ownership for one batch borrowed from a BatchSession.
 *
 * The batch buffers are returned to the originating session automatically on
 * destruction.  release() transfers the raw Batch to legacy code that still
 * performs an explicit returnBatchBuffers() call.
 */
class BatchLease {
   public:
    BatchLease() = default;
    BatchLease(std::shared_ptr<BatchSession> session, ExampleType exampleType, Batch batch);
    ~BatchLease();

    BatchLease(const BatchLease &) = delete;
    BatchLease &operator=(const BatchLease &) = delete;
    BatchLease(BatchLease &&other) noexcept;
    BatchLease &operator=(BatchLease &&other) noexcept;

    [[nodiscard]] Batch &get();
    [[nodiscard]] const Batch &get() const;
    [[nodiscard]] bool empty() const { return session == nullptr; }
    Batch release();
    void reset() noexcept;

   private:
    std::shared_ptr<BatchSession> session;
    ExampleType exampleType = ExampleType::TRAIN;
    Batch batch;
};

/**
 * Mutable, per-run data iteration state.
 *
 * A BatchSession owns cursors, randomization state, queues, worker threads and
 * reusable batch buffers.  It does not own the immutable dataset or split
 * definition that created it.
 */
class BatchSession : public Loader, public std::enable_shared_from_this<BatchSession> {
   public:
    ~BatchSession() override = default;

    BatchLease leaseBatch(ExampleType exampleType, uint64_t &batchNum) {
        return BatchLease(shared_from_this(), exampleType, getBatch(exampleType, batchNum));
    }

    [[nodiscard]] virtual std::vector<Event> getSynchronizeEvents() const { return {}; }
    [[nodiscard]] virtual const std::set<DatasetFieldId>& getRequiredDatasetFieldIds() const {
        static const std::set<DatasetFieldId> none;
        return none;
    }

    /**
     * Placement of a named tensor produced by this session, when known before
     * the first batch is requested. Device sessions use this to bypass host
     * staging rings; ordinary host sessions return std::nullopt.
     */
    [[nodiscard]] virtual std::optional<ThorImplementation::TensorPlacement> getBatchTensorPlacement(
        const std::string& tensorName) const {
        (void)tensorName;
        return std::nullopt;
    }

    virtual void cancel() {}
};

}  // namespace Thor
