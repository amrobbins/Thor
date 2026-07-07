#pragma once

#include "DeepLearning/Api/Loaders/Loader.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Loaders/DeviceResidentNamedDataset.h"
#include "Utilities/Random/FullPeriodRandom.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct DeviceResidentNamedBatchLoaderStats {
    std::string splitName;
    uint64_t batchesGathered = 0;
    uint64_t batchesReturned = 0;
    uint64_t currentAvailableBatches = 0;
    uint64_t batchQueueDepth = 0;
    uint64_t residentExamples = 0;
    uint64_t residentBytes = 0;
};

/**
 * Loader wrapper over a DeviceResidentNamedDataset.
 *
 * It preserves the logical named-loader API while producing device batch
 * tensors gathered from persistent split tensors on the target GPU.
 */
class DeviceResidentNamedBatchLoader : public Loader {
   public:
    explicit DeviceResidentNamedBatchLoader(std::shared_ptr<DeviceResidentNamedDataset> dataset,
                                            uint64_t batchQueueDepth = 2);
    ~DeviceResidentNamedBatchLoader() override;

    DeviceResidentNamedBatchLoader(const DeviceResidentNamedBatchLoader &) = delete;
    DeviceResidentNamedBatchLoader &operator=(const DeviceResidentNamedBatchLoader &) = delete;
    DeviceResidentNamedBatchLoader(DeviceResidentNamedBatchLoader &&) = delete;
    DeviceResidentNamedBatchLoader &operator=(DeviceResidentNamedBatchLoader &&) = delete;

    Batch getBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void returnBatchBuffers(ExampleType exampleType, Batch &&batch) override;

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;

    [[nodiscard]] std::shared_ptr<DeviceResidentNamedDataset> getDeviceDataset() const { return dataset; }
    [[nodiscard]] DeviceResidentNamedBatchLoaderStats getStatsSnapshot(ExampleType exampleType) const;

   private:
    struct SplitRuntime {
        const DeviceResidentNamedSplit *split = nullptr;
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
    };

    std::shared_ptr<DeviceResidentNamedDataset> dataset;
    uint64_t batchQueueDepth = 0;
    std::map<ExampleType, std::unique_ptr<SplitRuntime>> splitRuntimes;

    void initializeSplit(const DeviceResidentNamedSplit &split);
    [[nodiscard]] SplitRuntime &runtimeFor(ExampleType exampleType);
    [[nodiscard]] const SplitRuntime &runtimeFor(ExampleType exampleType) const;
    [[nodiscard]] std::map<std::string, ThorImplementation::Tensor> allocateBatchTensorSet() const;
    void validateReturnedBatch(const std::map<std::string, ThorImplementation::Tensor> &tensors) const;
    void fillRowIndexTensor(SplitRuntime &runtime);
};
