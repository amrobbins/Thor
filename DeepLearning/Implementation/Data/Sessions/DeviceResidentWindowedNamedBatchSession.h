#pragma once

#include "DeepLearning/Implementation/Data/Materialization/DeviceDatasetMaterialization.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceDatasetResidency.h"
#include "Utilities/Common/Stream.h"
#include "DeepLearning/Implementation/Data/Residency/DeviceResidentNamedDataset.h"
#include "Utilities/Data/Readers/IndexedDatasetReader.h"
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

/**
 * Hybrid session used when only canonical windowed fields fit on the device.
 * Direct fields are read from the source dataset while windowed fields are
 * gathered by canonical dataset row from shared device storage.
 */
class DeviceResidentWindowedNamedBatchSession : public Thor::BatchSession {
   public:
    DeviceResidentWindowedNamedBatchSession(
        Thor::DatasetMaterializationDescription datasetDescription,
        Thor::DeviceDatasetSessionDescription sessionDescription,
        Thor::DeviceDatasetLease windowedDataset,
        uint64_t batchQueueDepth = 2,
        uint64_t readerQueueDepth = 32,
        std::string datasetName = {});
    ~DeviceResidentWindowedNamedBatchSession() override;

    DeviceResidentWindowedNamedBatchSession(const DeviceResidentWindowedNamedBatchSession &) = delete;
    DeviceResidentWindowedNamedBatchSession &operator=(const DeviceResidentWindowedNamedBatchSession &) = delete;
    DeviceResidentWindowedNamedBatchSession(DeviceResidentWindowedNamedBatchSession &&) = delete;
    DeviceResidentWindowedNamedBatchSession &operator=(DeviceResidentWindowedNamedBatchSession &&) = delete;


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

    [[nodiscard]] const std::shared_ptr<const DeviceResidentNamedDataset> &getWindowedDeviceDataset() const {
        return windowedDataset.getShared();
    }
    [[nodiscard]] uint64_t getBatchQueueDepth() const { return batchQueueDepth; }
    [[nodiscard]] bool isCancelled() const { return cancelled.load(std::memory_order_acquire); }

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
        std::unique_ptr<IndexedDatasetReader::Session> readerSession;
        uint64_t nextBatchNum = 0;
        uint64_t nextLogicalPosition = 0;
        mutable std::mutex mutex;
        std::condition_variable notEmpty;
        Stream gatherStream;

        [[nodiscard]] uint64_t numExamples() const {
            return sourceIndices == nullptr ? 0 : static_cast<uint64_t>(sourceIndices->size());
        }
    };

    Thor::DatasetMaterializationDescription datasetDescription;
    Thor::DeviceDatasetSessionDescription sessionDescription;
    std::set<Thor::DatasetFieldId> requiredFieldIds;
    std::shared_ptr<IndexedDatasetReader> reader;
    Thor::DeviceDatasetLease windowedDataset;
    uint64_t batchQueueDepth = 0;
    uint64_t readerQueueDepth = 0;
    std::map<ExampleType, std::unique_ptr<SplitRuntime>> splitRuntimes;
    std::atomic<bool> cancelled{false};

    void initializeSplit(ExampleType exampleType,
                         std::shared_ptr<const Thor::ExampleIndexSet> sourceIndices,
                         bool randomized,
                         std::optional<uint64_t> seed);
    [[nodiscard]] SplitRuntime &runtimeFor(ExampleType exampleType);
    [[nodiscard]] const SplitRuntime &runtimeFor(ExampleType exampleType) const;
    [[nodiscard]] std::map<std::string, ThorImplementation::Tensor> allocateBatchTensorSet() const;
    void fillRowIndexTensor(SplitRuntime &runtime);
    void validateReturnedBatch(const std::map<std::string, ThorImplementation::Tensor> &tensors) const;
};
