#pragma once

#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Api/Data/NamedDataset.h"

#include <filesystem>
#include <memory>

class IndexedLocalNamedExampleReader;
class IndexedNamedBatchSession;

namespace Thor {

class FileDataset final : public NamedDataset {
   public:
    [[nodiscard]] static std::shared_ptr<FileDataset> open(const std::filesystem::path &datasetPath);

    [[nodiscard]] const DatasetId &getId() const override { return id; }
    [[nodiscard]] uint64_t getNumExamples() const override;
    [[nodiscard]] const DatasetSchema &getSchema() const override { return schema; }
    [[nodiscard]] const DatasetField &getField(std::string_view name) const override { return schema.getField(name); }

    [[nodiscard]] const std::filesystem::path &getPath() const { return datasetPath; }
    [[nodiscard]] const DatasetLayout &getLayout() const;

    void assertSchema(const DatasetSchema &expectedSchema) const;
    void assertLayout(const DatasetLayout &expectedLayout) const;

   private:
    friend class ::IndexedNamedBatchSession;

    [[nodiscard]] std::shared_ptr<BatchSession> openBatchSession(
        const DatasetSplitManifest &splits,
        const BatchPolicy &batching,
        const DatasetAccessPolicy &accessPolicy,
        uint64_t maxInFlightBatches,
        const std::set<DatasetFieldId> &requiredFieldIds) const override;

    [[nodiscard]] const std::shared_ptr<IndexedLocalNamedExampleReader> &getReader() const { return reader; }

    FileDataset(std::filesystem::path datasetPath,
                DatasetId id,
                DatasetSchema schema,
                std::shared_ptr<IndexedLocalNamedExampleReader> reader);

    std::filesystem::path datasetPath;
    DatasetId id;
    DatasetSchema schema;
    std::shared_ptr<IndexedLocalNamedExampleReader> reader;
};

}  // namespace Thor
