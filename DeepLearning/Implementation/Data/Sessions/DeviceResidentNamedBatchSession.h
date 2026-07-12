#pragma once

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetResidency.h"
#include "Utilities/Common/Stream.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedDataset.h"
#include "Utilities/Random/FullPeriodRandom.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

struct DeviceResidentNamedBatchSessionStats {
    std::string splitName;
    uint64_t batchesGathered = 0;
    uint64_t batchesReturned = 0;
    uint64_t currentAvailableBatches = 0;
    uint64_t batchQueueDepth = 0;
    uint64_t residentExamples = 0;
    uint64_t residentBytes = 0;
};

/**
 * Mutable batch session over one canonical DeviceResidentNamedDataset.
 *
 * Split membership, cursors, randomization, row-index tensors and reusable
 * batch buffers are session-owned. Persistent field storage remains shared and
 * split-independent in DeviceResidentNamedDataset.
 */
class DeviceResidentNamedBatchSession : public Thor::BatchSession {
   public:
    DeviceResidentNamedBatchSession(
        Thor::DeviceDatasetLease dataset,
        Thor::DatasetSplitManifest splits,
        Thor::BatchPolicy batching,
        uint64_t batchQueueDepth = 2,
        std::string datasetName = {});
    DeviceResidentNamedBatchSession(
        Thor::DeviceDatasetLease dataset,
        Thor::DeviceDatasetSessionDescription session,
        uint64_t batchQueueDepth = 2,
        std::string datasetName = {});
    ~DeviceResidentNamedBatchSession() override;

    DeviceResidentNamedBatchSession(const DeviceResidentNamedBatchSession &) = delete;
    DeviceResidentNamedBatchSession &operator=(const DeviceResidentNamedBatchSession &) = delete;
    DeviceResidentNamedBatchSession(DeviceResidentNamedBatchSession &&) = delete;
    DeviceResidentNamedBatchSession &operator=(DeviceResidentNamedBatchSession &&) = delete;


    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;
    [[nodiscard]] std::optional<ThorImplementation::TensorPlacement> getBatchTensorPlacement(
        const std::string &tensorName) const override;
    [[nodiscard]] std::vector<Event> getSynchronizeEvents() const override;
    [[nodiscard]] const std::set<Thor::DatasetFieldId>& getRequiredDatasetFieldIds() const override {
        return requiredFieldIds;
    }
    void cancel() override;

    [[nodiscard]] const std::shared_ptr<const DeviceResidentNamedDataset> &getDeviceDataset() const { return dataset.getShared(); }
    [[nodiscard]] const Thor::DatasetSplitManifest &getSplitManifest() const { return splits; }
    [[nodiscard]] const Thor::BatchPolicy &getBatchPolicy() const { return batching; }
    [[nodiscard]] uint64_t getBatchQueueDepth() const { return batchQueueDepth; }
    [[nodiscard]] bool isCancelled() const { return cancelled.load(std::memory_order_acquire); }
    [[nodiscard]] DeviceResidentNamedBatchSessionStats getStatsSnapshot(ExampleType exampleType) const;

   private:
    Batch acquireBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void recycleBatch(ExampleType exampleType, Batch &&batch) override;
    struct SplitRuntime {
        ExampleType exampleType = ExampleType::TRAIN;
        std::string splitName;
        std::shared_ptr<const Thor::ExampleIndexSet> sourceIndices;
        bool randomized = false;
        std::optional<uint64_t> seed;
        uint64_t batchesPerEpoch = 0;
        std::deque<std::map<std::string, ThorImplementation::Tensor>> availableBatches;
        ThorImplementation::Tensor rowIndicesHost;
        ThorImplementation::Tensor rowIndicesDevice;
        std::unique_ptr<FullPeriodRandom> randomizer;
        uint64_t nextBatchNum = 0;
        uint64_t nextLogicalPosition = 0;
        uint64_t batchesGathered = 0;
        uint64_t batchesReturned = 0;
        mutable std::mutex mutex;
        std::condition_variable notEmpty;
        Stream gatherStream;

        [[nodiscard]] uint64_t numExamples() const {
            return sourceIndices == nullptr ? 0 : static_cast<uint64_t>(sourceIndices->size());
        }
    };

    Thor::DeviceDatasetLease dataset;
    Thor::DatasetSplitManifest splits;
    Thor::BatchPolicy batching;
    std::set<Thor::DatasetFieldId> requiredFieldIds;
    uint64_t batchQueueDepth = 0;
    std::map<ExampleType, std::unique_ptr<SplitRuntime>> splitRuntimes;
    std::atomic<bool> cancelled{false};

    void initializeSplit(ExampleType exampleType,
                         std::shared_ptr<const Thor::ExampleIndexSet> sourceIndices,
                         bool randomized,
                         std::optional<uint64_t> seed);
    [[nodiscard]] SplitRuntime &runtimeFor(ExampleType exampleType);
    [[nodiscard]] const SplitRuntime &runtimeFor(ExampleType exampleType) const;
    [[nodiscard]] std::map<std::string, ThorImplementation::Tensor> allocateBatchTensorSet() const;
    void validateReturnedBatch(const std::map<std::string, ThorImplementation::Tensor> &tensors) const;
    void fillRowIndexTensor(SplitRuntime &runtime);
};
