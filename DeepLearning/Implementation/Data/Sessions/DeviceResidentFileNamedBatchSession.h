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

struct DeviceResidentFileSelectionSlot;
struct DeviceResidentFileDirectSlot;
struct DeviceResidentFileRaggedSlot;
struct DeviceResidentFilePendingSelection;
struct DeviceResidentFilePendingDirect;
struct DeviceResidentFilePendingRagged;

/**
 * Compact file-backed device session. Windowed fields are returned as
 * DeviceBatchReferences, ragged fields are gathered into reusable GPU
 * RaggedTensors, and direct fields are references when their compact record
 * ranges were admitted or CPU-backed tensors in the hybrid fallback path.
 */
class DeviceResidentFileNamedBatchSession : public Thor::BatchSession {
   public:
    DeviceResidentFileNamedBatchSession(
        Thor::DatasetMaterializationDescription datasetDescription,
        Thor::DeviceDatasetSessionDescription sessionDescription,
        Thor::DeviceDatasetLease residentDataset,
        uint64_t batchQueueDepth = 2,
        uint64_t readerQueueDepth = 32,
        std::string datasetName = {});
    ~DeviceResidentFileNamedBatchSession() override;

    DeviceResidentFileNamedBatchSession(const DeviceResidentFileNamedBatchSession &) = delete;
    DeviceResidentFileNamedBatchSession &operator=(const DeviceResidentFileNamedBatchSession &) = delete;
    DeviceResidentFileNamedBatchSession(DeviceResidentFileNamedBatchSession &&) = delete;
    DeviceResidentFileNamedBatchSession &operator=(DeviceResidentFileNamedBatchSession &&) = delete;

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;
    [[nodiscard]] std::optional<ThorImplementation::TensorPlacement> getBatchTensorPlacement(
        const std::string &tensorName) const override;
    [[nodiscard]] Thor::BatchFieldSourceDescription getBatchFieldSourceDescription(
        const std::string &fieldName) const override;
    [[nodiscard]] std::vector<Event> getSynchronizeEvents() const override;
    [[nodiscard]] const Thor::DatasetFieldMaterializationRequirements&
    getDatasetFieldMaterializationRequirements() const override {
        return fieldRequirements;
    }
    void cancel() override;

    [[nodiscard]] const std::shared_ptr<const DeviceResidentNamedDataset> &getDeviceDataset() const {
        return residentDataset.getShared();
    }
    [[nodiscard]] uint64_t getBatchQueueDepth() const { return batchQueueDepth; }
    [[nodiscard]] bool isCancelled() const { return cancelled.load(std::memory_order_acquire); }

   private:
    Batch acquireBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void recycleBatch(ExampleType exampleType, Batch &&batch) override;
    void setBatchTailModeForRuntimeImpl(ThorImplementation::BatchTailMode mode) override;

    struct SplitRuntime {
        ExampleType exampleType = ExampleType::TRAIN;
        std::string splitName;
        std::shared_ptr<const Thor::ExampleIndexSet> sourceIndices;
        bool randomized = false;
        std::optional<uint64_t> seed;
        uint64_t batchesPerEpoch = 0;
        std::deque<std::shared_ptr<DeviceResidentFileSelectionSlot>> availableSelections;
        std::deque<std::shared_ptr<DeviceResidentFileDirectSlot>> availableDirectSlots;
        std::deque<std::shared_ptr<DeviceResidentFileRaggedSlot>> availableRaggedSlots;
        std::deque<DeviceResidentFilePendingSelection> pendingSelections;
        std::deque<DeviceResidentFilePendingDirect> pendingDirectSlots;
        std::deque<DeviceResidentFilePendingRagged> pendingRaggedSlots;
        std::unique_ptr<FullPeriodRandom> randomizer;
        std::unique_ptr<IndexedDatasetReader::Session> readerSession;
        uint64_t nextBatchNum = 0;
        uint64_t nextLogicalPosition = 0;
        mutable std::mutex mutex;
        std::mutex readerMutex;
        std::condition_variable notEmpty;
        Stream selectionUploadStream;
        Stream raggedGatherStream;

        [[nodiscard]] uint64_t numExamples() const {
            return sourceIndices == nullptr ? 0 : static_cast<uint64_t>(sourceIndices->size());
        }
    };

    Thor::DatasetMaterializationDescription datasetDescription;
    Thor::DeviceDatasetSessionDescription sessionDescription;
    Thor::DatasetFieldMaterializationRequirements fieldRequirements;
    std::set<std::string> directFieldNames;
    std::set<std::string> residentReferenceFieldNames;
    std::set<std::string> raggedFieldNames;
    std::shared_ptr<IndexedDatasetReader> reader;
    Thor::DeviceDatasetLease residentDataset;
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
    [[nodiscard]] std::shared_ptr<DeviceResidentFileSelectionSlot> allocateSelectionSlot(
        uint64_t slotIndex) const;
    [[nodiscard]] std::shared_ptr<DeviceResidentFileDirectSlot> allocateDirectSlot(
        uint64_t slotIndex) const;
    [[nodiscard]] std::shared_ptr<DeviceResidentFileRaggedSlot> allocateRaggedSlot(
        uint64_t slotIndex) const;
    void fillRowIndexTensor(
        SplitRuntime &runtime,
        DeviceResidentFileSelectionSlot &selectionSlot,
        uint32_t validExampleCount);
    void validateReturnedBatch(const Batch &batch) const;
    void releaseSelectionSlot(
        ExampleType exampleType,
        std::shared_ptr<DeviceResidentFileSelectionSlot> selectionSlot,
        std::vector<Event> consumedEvents) noexcept;
    void releaseDirectSlot(
        ExampleType exampleType,
        std::shared_ptr<DeviceResidentFileDirectSlot> directSlot,
        std::vector<Event> consumedEvents) noexcept;
    void releaseRaggedSlot(
        ExampleType exampleType,
        std::shared_ptr<DeviceResidentFileRaggedSlot> raggedSlot,
        std::vector<Event> consumedEvents) noexcept;
};
