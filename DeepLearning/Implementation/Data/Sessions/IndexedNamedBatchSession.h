#pragma once

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/FileDataset.h"
#include "Utilities/Data/Assembly/IndexedBatchAssembler.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

/**
 * Per-run indexed named batch session backed by one immutable named dataset.
 *
 * The dataset reader exposes a canonical indexed row space.  train/validate/test are
 * logical views over that row space, supplied as immutable index sets.
 * For the FileDataset backend, fold-specific sessions share one
 * physical dataset instead of duplicating records per fold.
 */
class IndexedNamedBatchSession : public Thor::BatchSession {
   public:
    IndexedNamedBatchSession(std::shared_ptr<const Thor::FileDataset> dataset,
                             Thor::DatasetSplitManifest splits,
                             Thor::BatchPolicy batching,
                             uint64_t batchQueueDepth = 32,
                             std::set<Thor::DatasetFieldId> requiredFieldIds = {});

    ~IndexedNamedBatchSession() override;

    IndexedNamedBatchSession(const IndexedNamedBatchSession &) = delete;
    IndexedNamedBatchSession &operator=(const IndexedNamedBatchSession &) = delete;
    IndexedNamedBatchSession(IndexedNamedBatchSession &&) = delete;
    IndexedNamedBatchSession &operator=(IndexedNamedBatchSession &&) = delete;


    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;

    [[nodiscard]] const DatasetLayout &getLayout() const;
    [[nodiscard]] const std::shared_ptr<const Thor::FileDataset> &getDataset() const { return dataset; }
    [[nodiscard]] const Thor::DatasetSplitManifest &getSplitManifest() const { return splitManifest; }
    [[nodiscard]] const std::set<Thor::DatasetFieldId>& getRequiredDatasetFieldIds() const override {
        return requiredFieldIds;
    }
    [[nodiscard]] uint64_t getNumDatasetExamples() const;
    [[nodiscard]] uint64_t getBatchQueueDepth() const;
    [[nodiscard]] bool getRandomizeTrain() const;
    [[nodiscard]] std::optional<uint64_t> getRandomSeed() const;
    [[nodiscard]] const Thor::ExampleIndexSet &getSplitIndices(ExampleType exampleType) const;
    [[nodiscard]] bool hasExplicitTestSplit() const;
    void cancel() override;
    [[nodiscard]] IndexedBatchAssemblerStats getStatsSnapshot(ExampleType exampleType);

#ifdef THOR_GTEST
    void recycleBatchForTesting(ExampleType exampleType, Batch&& batch) {
        recycleBatch(exampleType, std::move(batch));
    }

    uint64_t getReadyBatchCountForTesting(ExampleType exampleType) {
        IndexedBatchAssembler *assembler = assemblerFor(exampleType);
        return assembler == nullptr ? 0 : assembler->getReadyBatchCountForTesting();
    }
#endif

   private:
    Batch acquireBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void recycleBatch(ExampleType exampleType, Batch &&batch) override;
    std::shared_ptr<const Thor::FileDataset> dataset;
    Thor::DatasetSplitManifest splitManifest;
    std::set<Thor::DatasetFieldId> requiredFieldIds;

    std::unique_ptr<IndexedBatchAssembler> trainAssembler;
    std::unique_ptr<IndexedBatchAssembler> validateAssembler;
    std::unique_ptr<IndexedBatchAssembler> testAssembler;

    uint64_t numDatasetExamples = 0;
    uint64_t batchQueueDepth = 32;
    bool randomizeTrain = true;
    std::optional<uint64_t> seed;
    std::atomic<bool> cancelled{false};

    struct PendingReturnedBuffers {
        ExampleType exampleType = ExampleType::TRAIN;
        std::shared_ptr<std::map<std::string, ThorImplementation::Tensor>> tensors;
        std::vector<Event> consumedEvents;
    };

    mutable std::mutex recyclerMutex;
    std::condition_variable recyclerNotEmpty;
    std::deque<PendingReturnedBuffers> pendingReturnedBuffers;
    std::thread recyclerThread;
    bool recyclerStopping = false;
    std::exception_ptr recyclerFailure;

    void validateIndex(uint64_t index, const char *splitName) const;
    void validateIndices(const Thor::ExampleIndexSet &indices, const char *splitName) const;
    std::unique_ptr<IndexedBatchAssembler> createAssembler(
        std::shared_ptr<const Thor::ExampleIndexSet> indices,
        const char *splitName,
        bool randomized,
        std::optional<uint64_t> splitSeed) const;
    IndexedBatchAssembler *assemblerFor(ExampleType exampleType);
    const IndexedBatchAssembler *assemblerFor(ExampleType exampleType) const;
    void enqueueReturnedBuffers(
        ExampleType exampleType,
        std::shared_ptr<std::map<std::string, ThorImplementation::Tensor>> tensors,
        std::vector<Event> consumedEvents) noexcept;
    void recyclerMain() noexcept;
    void stopRecycler() noexcept;
    void throwIfRecyclerFailed() const;
};
