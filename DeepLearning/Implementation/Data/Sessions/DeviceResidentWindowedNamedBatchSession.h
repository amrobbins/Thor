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

struct DeviceResidentWindowedSelectionSlot;
struct DeviceResidentWindowedDirectSlot;
struct DeviceResidentWindowedPendingSelection;
struct DeviceResidentWindowedPendingDirect;

/**
 * Compact file-backed device session. Windowed fields are always resident and
 * returned as DeviceBatchReferences. Direct fields are also references when
 * their compact record ranges were admitted; otherwise they remain CPU-backed
 * tensors, preserving the hybrid fallback path.
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
    [[nodiscard]] Thor::BatchFieldSourceDescription getBatchFieldSourceDescription(
        const std::string &fieldName) const override;
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
        std::deque<std::shared_ptr<DeviceResidentWindowedSelectionSlot>> availableSelections;
        std::deque<std::shared_ptr<DeviceResidentWindowedDirectSlot>> availableDirectSlots;
        std::deque<DeviceResidentWindowedPendingSelection> pendingSelections;
        std::deque<DeviceResidentWindowedPendingDirect> pendingDirectSlots;
        std::unique_ptr<FullPeriodRandom> randomizer;
        std::unique_ptr<IndexedDatasetReader::Session> readerSession;
        uint64_t nextBatchNum = 0;
        uint64_t nextLogicalPosition = 0;
        mutable std::mutex mutex;
        std::mutex readerMutex;
        std::condition_variable notEmpty;
        Stream selectionUploadStream;

        [[nodiscard]] uint64_t numExamples() const {
            return sourceIndices == nullptr ? 0 : static_cast<uint64_t>(sourceIndices->size());
        }
    };

    Thor::DatasetMaterializationDescription datasetDescription;
    Thor::DeviceDatasetSessionDescription sessionDescription;
    std::set<Thor::DatasetFieldId> requiredFieldIds;
    std::set<std::string> directFieldNames;
    std::set<std::string> residentFieldNames;
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
    [[nodiscard]] std::map<std::string, ThorImplementation::Tensor> allocateDirectTensorSet() const;
    [[nodiscard]] std::shared_ptr<DeviceResidentWindowedSelectionSlot> allocateSelectionSlot(
        uint64_t slotIndex) const;
    [[nodiscard]] std::shared_ptr<DeviceResidentWindowedDirectSlot> allocateDirectSlot(
        uint64_t slotIndex) const;
    void fillRowIndexTensor(
        SplitRuntime &runtime,
        DeviceResidentWindowedSelectionSlot &selectionSlot);
    void validateReturnedBatch(const Batch &batch) const;
    void releaseSelectionSlot(
        ExampleType exampleType,
        std::shared_ptr<DeviceResidentWindowedSelectionSlot> selectionSlot,
        std::vector<Event> consumedEvents) noexcept;
    void releaseDirectSlot(
        ExampleType exampleType,
        std::shared_ptr<DeviceResidentWindowedDirectSlot> directSlot,
        std::vector<Event> consumedEvents) noexcept;
};
