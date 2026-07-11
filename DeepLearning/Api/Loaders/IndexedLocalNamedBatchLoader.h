#pragma once

#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/BatchSession.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"
#include "DeepLearning/Api/Data/LocalNamedDataset.h"
#include "DeepLearning/Api/Loaders/Loader.h"
#include "Utilities/Loaders/IndexedLocalNamedBatchAssembler.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

/**
 * Per-run indexed named batch session backed by one immutable named dataset.
 *
 * The dataset reader exposes a canonical indexed row space.  train/validate/test are
 * logical views over that row space, supplied as index
 * arrays. For the LocalNamedDataset backend, fold-specific sessions share one
 * physical dataset instead of duplicating records per fold.
 */
class IndexedNamedBatchSession : public Thor::BatchSession {
   public:
    IndexedNamedBatchSession(std::shared_ptr<const Thor::LocalNamedDataset> dataset,
                             Thor::DatasetSplitManifest splits,
                             Thor::BatchPolicy batching,
                             uint64_t batchQueueDepth = 32,
                             std::set<Thor::DatasetFieldId> requiredFieldIds = {});

    IndexedNamedBatchSession(std::shared_ptr<const Thor::LocalNamedDataset> dataset,
                             std::vector<uint64_t> trainIndices,
                             std::vector<uint64_t> validateIndices,
                             std::optional<std::vector<uint64_t>> testIndices,
                             uint64_t batchSize,
                             uint64_t batchQueueDepth = 32,
                             bool randomizeTrain = true,
                             std::optional<uint64_t> seed = std::nullopt);

    /** Compatibility adapter. requestedLayout is validation-only. */
    IndexedNamedBatchSession(std::filesystem::path datasetPath,
                             LocalNamedExampleLayout requestedLayout,
                             std::vector<uint64_t> trainIndices,
                             std::vector<uint64_t> validateIndices,
                             std::optional<std::vector<uint64_t>> testIndices,
                             uint64_t batchSize,
                             uint64_t batchQueueDepth = 32,
                             bool randomizeTrain = true,
                             std::optional<uint64_t> seed = std::nullopt);
    ~IndexedNamedBatchSession() override;

    IndexedNamedBatchSession(const IndexedNamedBatchSession &) = delete;
    IndexedNamedBatchSession &operator=(const IndexedNamedBatchSession &) = delete;
    IndexedNamedBatchSession(IndexedNamedBatchSession &&) = delete;
    IndexedNamedBatchSession &operator=(IndexedNamedBatchSession &&) = delete;

    Batch getBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void returnBatchBuffers(ExampleType exampleType, Batch &&batch) override;

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;

    [[nodiscard]] const LocalNamedExampleLayout &getLayout() const;
    [[nodiscard]] const std::filesystem::path &getDatasetPath() const;
    [[nodiscard]] const std::shared_ptr<const Thor::LocalNamedDataset> &getDataset() const { return dataset; }
    [[nodiscard]] const Thor::DatasetSplitManifest &getSplitManifest() const { return splitManifest; }
    [[nodiscard]] const std::set<Thor::DatasetFieldId>& getRequiredDatasetFieldIds() const override {
        return requiredFieldIds;
    }
    [[nodiscard]] uint64_t getNumDatasetExamples() const;
    [[nodiscard]] uint64_t getBatchQueueDepth() const;
    [[nodiscard]] bool getRandomizeTrain() const;
    [[nodiscard]] std::optional<uint64_t> getRandomSeed() const;
    [[nodiscard]] const std::vector<uint64_t> &getSplitIndices(ExampleType exampleType) const;
    [[nodiscard]] bool hasExplicitTestSplit() const;
    void cancel() override;
    [[nodiscard]] IndexedLocalNamedBatchAssemblerStats getStatsSnapshot(ExampleType exampleType);

#ifdef THOR_GTEST
    [[nodiscard]] const IndexedLocalNamedExampleReader *getDatasetReaderForTesting() const {
        return dataset->getReader().get();
    }

    uint64_t getReadyBatchCountForTesting(ExampleType exampleType) {
        IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
        return assembler == nullptr ? 0 : assembler->getReadyBatchCountForTesting();
    }
#endif

   private:
    std::shared_ptr<const Thor::LocalNamedDataset> dataset;
    Thor::DatasetSplitManifest splitManifest;
    std::set<Thor::DatasetFieldId> requiredFieldIds;

    std::unique_ptr<IndexedLocalNamedBatchAssembler> trainAssembler;
    std::unique_ptr<IndexedLocalNamedBatchAssembler> validateAssembler;
    std::unique_ptr<IndexedLocalNamedBatchAssembler> testAssembler;

    uint64_t numDatasetExamples = 0;
    uint64_t batchQueueDepth = 32;
    bool randomizeTrain = true;
    std::optional<uint64_t> seed;
    std::atomic<bool> cancelled{false};

    void validateIndex(uint64_t index, const char *splitName) const;
    void validateIndices(const std::vector<uint64_t> &indices, const char *splitName) const;
    std::unique_ptr<IndexedLocalNamedBatchAssembler> createAssembler(
        std::shared_ptr<const std::vector<uint64_t>> indices,
        const char *splitName,
        bool randomized,
        std::optional<uint64_t> splitSeed) const;
    IndexedLocalNamedBatchAssembler *assemblerFor(ExampleType exampleType);
    const IndexedLocalNamedBatchAssembler *assemblerFor(ExampleType exampleType) const;
};

// Compatibility names retained for one migration release.  The execution
// object itself is a session and is not named after the local storage backend.
using IndexedNamedBatchLoader = IndexedNamedBatchSession;
using IndexedLocalNamedBatchLoader = IndexedNamedBatchSession;
