#pragma once

#include "DeepLearning/Api/Data/NamedDataset.h"

#include <filesystem>
#include <memory>

namespace ThorImplementation {
class FileDatasetRuntimeAccess;
}

namespace Thor {

class FileDataset final : public NamedDataset {
   public:
    [[nodiscard]] static std::shared_ptr<FileDataset> open(const std::filesystem::path &datasetPath);

    ~FileDataset() override;

    [[nodiscard]] const DatasetId &getId() const override { return id; }
    [[nodiscard]] uint64_t getNumExamples() const override;
    [[nodiscard]] const DatasetSchema &getSchema() const override { return schema; }
    [[nodiscard]] const DatasetField &getField(std::string_view name) const override { return schema.getField(name); }

    [[nodiscard]] const std::filesystem::path &getPath() const { return datasetPath; }

    void assertSchema(const DatasetSchema &expectedSchema) const;

   private:
    friend class ThorImplementation::FileDatasetRuntimeAccess;

    class Runtime;

    [[nodiscard]] std::shared_ptr<BatchSession> openBatchSession(
        const DatasetSplitManifest &splits,
        const BatchPolicy &batching,
        const DatasetAccessPolicy &accessPolicy,
        uint64_t maxInFlightBatches,
        const std::set<DatasetFieldId> &requiredFieldIds) const override;

    [[nodiscard]] std::unique_ptr<DatasetMaterializationDescription>
    describeMaterializationForRuntime() const override;
    [[nodiscard]] MaterializedNamedDatasetSnapshot
    materializeSnapshotForRuntime(uint64_t readerQueueDepth) const override;

    FileDataset(std::filesystem::path datasetPath,
                DatasetId id,
                DatasetSchema schema,
                std::unique_ptr<Runtime> runtime);

    std::filesystem::path datasetPath;
    DatasetId id;
    DatasetSchema schema;
    std::unique_ptr<Runtime> runtime;
};

}  // namespace Thor
