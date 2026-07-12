#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetSchema.h"

#include <cstdint>
#include <memory>
#include <set>
#include <string_view>

struct MaterializedNamedDatasetSnapshot;

namespace ThorImplementation {
class NamedDatasetRuntimeAccess;
}

namespace Thor {

struct DatasetMaterializationDescription;

class BatchSession;
class BatchPolicy;
class DatasetAccessPolicy;
class DatasetSplitManifest;
class TrainingData;

class NamedDataset : public std::enable_shared_from_this<NamedDataset> {
   public:
    NamedDataset();
    virtual ~NamedDataset();
    NamedDataset(const NamedDataset &) = delete;
    NamedDataset &operator=(const NamedDataset &) = delete;
    NamedDataset(NamedDataset &&) = delete;
    NamedDataset &operator=(NamedDataset &&) = delete;

    [[nodiscard]] virtual const DatasetId &getId() const = 0;
    [[nodiscard]] virtual uint64_t getNumExamples() const = 0;
    [[nodiscard]] virtual const DatasetSchema &getSchema() const = 0;
    [[nodiscard]] virtual const DatasetField &getField(std::string_view name) const = 0;

   protected:
    /** Backend hook used only by TrainingData to create fresh mutable iteration state. */
    [[nodiscard]] virtual std::shared_ptr<BatchSession> openBatchSession(
        const DatasetSplitManifest &splits,
        const BatchPolicy &batching,
        const DatasetAccessPolicy &accessPolicy,
        uint64_t maxInFlightBatches,
        const std::set<DatasetFieldId> &requiredFieldIds) const = 0;

   private:
    [[nodiscard]] virtual std::unique_ptr<DatasetMaterializationDescription>
    describeMaterializationForRuntime() const;
    [[nodiscard]] virtual MaterializedNamedDatasetSnapshot
    materializeSnapshotForRuntime(uint64_t readerQueueDepth) const;

    friend class TrainingData;
    friend class ThorImplementation::NamedDatasetRuntimeAccess;

    class Runtime;
    std::unique_ptr<Runtime> runtime;
};

}  // namespace Thor
