#pragma once

#include "DeepLearning/Api/Data/NamedDataset.h"
#include "Utilities/Loaders/IndexedLocalNamedExampleReader.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"

#include <filesystem>
#include <memory>

class IndexedNamedBatchSession;

namespace Thor {

class LocalNamedDataset final : public NamedDataset {
   public:
    [[nodiscard]] static std::shared_ptr<LocalNamedDataset> open(const std::filesystem::path &datasetPath);

    [[nodiscard]] const DatasetId &getId() const override { return id; }
    [[nodiscard]] uint64_t getNumExamples() const override;
    [[nodiscard]] const DatasetSchema &getSchema() const override { return schema; }
    [[nodiscard]] const DatasetField &getField(std::string_view name) const override { return schema.getField(name); }

    [[nodiscard]] const std::filesystem::path &getPath() const { return datasetPath; }
    [[nodiscard]] const LocalNamedExampleLayout &getLayout() const { return reader->getLayout(); }

    void assertSchema(const DatasetSchema &expectedSchema) const;
    void assertLayout(const LocalNamedExampleLayout &expectedLayout) const;

   private:
    friend class ::IndexedNamedBatchSession;

    [[nodiscard]] const std::shared_ptr<IndexedLocalNamedExampleReader> &getReader() const { return reader; }

    LocalNamedDataset(std::filesystem::path datasetPath,
                      DatasetId id,
                      DatasetSchema schema,
                      std::shared_ptr<IndexedLocalNamedExampleReader> reader);

    std::filesystem::path datasetPath;
    DatasetId id;
    DatasetSchema schema;
    std::shared_ptr<IndexedLocalNamedExampleReader> reader;
};

}  // namespace Thor
