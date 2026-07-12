#pragma once

#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"
#include "DeepLearning/Api/Data/ExampleType.h"
#include "Utilities/Common/Event.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Thor {

class BatchSession;
class TrainingData;

/**
 * RAII ownership for one batch borrowed from a BatchSession.
 *
 * The batch buffers are returned to the originating session automatically on
 * destruction. Batch ownership cannot be detached from the session.
 */
class BatchLease {
   public:
    BatchLease() = default;
    ~BatchLease();

    BatchLease(const BatchLease &) = delete;
    BatchLease &operator=(const BatchLease &) = delete;
    BatchLease(BatchLease &&other) noexcept;
    BatchLease &operator=(BatchLease &&other) noexcept;

    [[nodiscard]] const Batch &get() const;
    [[nodiscard]] bool empty() const { return session == nullptr; }
    void reset() noexcept;

   private:
    BatchLease(std::shared_ptr<BatchSession> session,
               ExampleType exampleType,
               Batch batch);

    std::shared_ptr<BatchSession> session;
    ExampleType exampleType = ExampleType::TRAIN;
    Batch batch;

    friend class BatchSession;
};

/**
 * Mutable, per-run data iteration state.
 *
 * A BatchSession owns cursors, randomization state, queues, worker threads and
 * reusable batch buffers. It does not own the immutable dataset or split
 * definition that created it. Public callers acquire batches only as leases;
 * buffer recycling remains an implementation detail shared with BatchLease.
 */
class BatchSession : public std::enable_shared_from_this<BatchSession> {
   public:
    virtual ~BatchSession() = default;

    [[nodiscard]] BatchLease leaseBatch(ExampleType exampleType, uint64_t &batchNum) {
        std::shared_ptr<BatchSession> owner;
        try {
            owner = shared_from_this();
        } catch (const std::bad_weak_ptr&) {
            throw std::runtime_error(
                "BatchSession must be owned by std::shared_ptr before acquiring a BatchLease.");
        }
        return BatchLease(std::move(owner), exampleType, acquireBatch(exampleType, batchNum));
    }

    [[nodiscard]] virtual uint64_t getBatchSize() const { return batchSize; }
    [[nodiscard]] virtual uint64_t getNumBatchesPerEpoch(ExampleType exampleType) = 0;
    [[nodiscard]] virtual uint64_t getNumExamples(ExampleType exampleType) = 0;
    [[nodiscard]] virtual uint64_t getNextBatchNum(ExampleType exampleType) = 0;
    [[nodiscard]] virtual std::string getDatasetName() const { return datasetName; }

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

   protected:
    BatchSession() = default;
    explicit BatchSession(std::string datasetName) : datasetName(std::move(datasetName)) {}

    uint64_t batchSize = 0;

   private:
    virtual Batch acquireBatch(ExampleType exampleType, uint64_t &batchNum) = 0;
    virtual void recycleBatch(ExampleType exampleType, Batch &&batch) = 0;

    void setDatasetName(std::string name) { datasetName = std::move(name); }

    std::string datasetName;

    friend class BatchLease;
    friend class TrainingData;
};

}  // namespace Thor
