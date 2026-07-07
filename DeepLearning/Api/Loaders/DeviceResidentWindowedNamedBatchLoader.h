#pragma once

#include "DeepLearning/Api/Loaders/Loader.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Loaders/DeviceResidentNamedDataset.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"
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

/**
 * Hybrid loader used when best-effort device dataset storage cannot fit the
 * full materialized dataset but can fit the high-value windowed tensors.
 *
 * Direct dense tensors are loaded from the indexed dataset without materializing
 * window sources. Windowed tensors/masks are gathered from a device-resident
 * assembled window cache. This preserves source split/order semantics while
 * avoiding repeated wide history window reads/assembly/copies.
 */
class DeviceResidentWindowedNamedBatchLoader : public Loader {
   public:
    DeviceResidentWindowedNamedBatchLoader(DeviceDatasetMaterializationView view,
                                           std::shared_ptr<DeviceResidentNamedDataset> windowedDataset,
                                           uint64_t batchQueueDepth = 2,
                                           uint64_t readerQueueDepth = 32);
    ~DeviceResidentWindowedNamedBatchLoader() override;

    DeviceResidentWindowedNamedBatchLoader(const DeviceResidentWindowedNamedBatchLoader &) = delete;
    DeviceResidentWindowedNamedBatchLoader &operator=(const DeviceResidentWindowedNamedBatchLoader &) = delete;
    DeviceResidentWindowedNamedBatchLoader(DeviceResidentWindowedNamedBatchLoader &&) = delete;
    DeviceResidentWindowedNamedBatchLoader &operator=(DeviceResidentWindowedNamedBatchLoader &&) = delete;

    Batch getBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void returnBatchBuffers(ExampleType exampleType, Batch &&batch) override;

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;

    [[nodiscard]] std::shared_ptr<DeviceResidentNamedDataset> getWindowedDeviceDataset() const { return windowedDataset; }

   private:
    struct SplitRuntime {
        const DeviceResidentNamedSplit *split = nullptr;
        std::deque<std::map<std::string, ThorImplementation::Tensor>> availableBatches;
        ThorImplementation::Tensor rowIndicesHost;
        ThorImplementation::Tensor rowIndicesDevice;
        std::unique_ptr<FullPeriodRandom> randomizer;
        std::unique_ptr<IndexedLocalNamedExampleReader::Session> readerSession;
        uint64_t nextBatchNum = 0;
        uint64_t nextLogicalPosition = 0;
        mutable std::mutex mutex;
        std::condition_variable notEmpty;
        Stream gatherStream;
    };

    DeviceDatasetMaterializationView view;
    std::shared_ptr<IndexedLocalNamedExampleReader> reader;
    std::shared_ptr<DeviceResidentNamedDataset> windowedDataset;
    uint64_t batchQueueDepth = 0;
    uint64_t readerQueueDepth = 0;
    std::map<ExampleType, std::unique_ptr<SplitRuntime>> splitRuntimes;

    void initializeSplit(const DeviceResidentNamedSplit &split);
    [[nodiscard]] SplitRuntime &runtimeFor(ExampleType exampleType);
    [[nodiscard]] const SplitRuntime &runtimeFor(ExampleType exampleType) const;
    [[nodiscard]] std::map<std::string, ThorImplementation::Tensor> allocateBatchTensorSet() const;
    void fillRowIndexTensor(SplitRuntime &runtime);
    void validateReturnedBatch(const std::map<std::string, ThorImplementation::Tensor> &tensors) const;
};
