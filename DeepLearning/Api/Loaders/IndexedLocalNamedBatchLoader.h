#pragma once

#include "DeepLearning/Api/Loaders/Loader.h"
#include "Utilities/Loaders/IndexedLocalNamedBatchAssembler.h"
#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * Indexed, Thor-native named batch loader backed by one shared local named example dataset.
 *
 * The dataset's canonical row space is the manifest-declared indexed record order across
 * all shards.  train/validate/test are logical views over that row space, supplied as index
 * arrays.  This is the local named equivalent of IndexedNumpyFloat32DictBatchLoader:
 * fold-specific splits share one physical dataset instead of duplicating records per fold.
 */
class IndexedLocalNamedBatchLoader : public Loader {
   public:
    IndexedLocalNamedBatchLoader(std::filesystem::path datasetPath,
                                 LocalNamedExampleLayout requestedLayout,
                                 std::vector<uint64_t> trainIndices,
                                 std::vector<uint64_t> validateIndices,
                                 std::optional<std::vector<uint64_t>> testIndices,
                                 uint64_t batchSize,
                                 uint64_t batchQueueDepth = 32,
                                 bool randomizeTrain = true,
                                 std::optional<uint64_t> seed = std::nullopt);
    ~IndexedLocalNamedBatchLoader() override;

    IndexedLocalNamedBatchLoader(const IndexedLocalNamedBatchLoader &) = delete;
    IndexedLocalNamedBatchLoader &operator=(const IndexedLocalNamedBatchLoader &) = delete;
    IndexedLocalNamedBatchLoader(IndexedLocalNamedBatchLoader &&) = delete;
    IndexedLocalNamedBatchLoader &operator=(IndexedLocalNamedBatchLoader &&) = delete;

    Batch getBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void returnBatchBuffers(ExampleType exampleType, Batch &&batch) override;

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;

    [[nodiscard]] const LocalNamedExampleLayout &getLayout() const;
    [[nodiscard]] const std::filesystem::path &getDatasetPath() const;
    [[nodiscard]] uint64_t getNumDatasetExamples() const;
    [[nodiscard]] uint64_t getBatchQueueDepth() const;
    [[nodiscard]] bool getRandomizeTrain() const;
    [[nodiscard]] std::optional<uint64_t> getRandomSeed() const;
    [[nodiscard]] bool hasExplicitTestSplit() const;
    [[nodiscard]] IndexedLocalNamedBatchAssemblerStats getStatsSnapshot(ExampleType exampleType);

#ifdef THOR_GTEST
    uint64_t getReadyBatchCountForTesting(ExampleType exampleType) {
        IndexedLocalNamedBatchAssembler *assembler = assemblerFor(exampleType);
        return assembler == nullptr ? 0 : assembler->getReadyBatchCountForTesting();
    }
#endif

   private:
    std::filesystem::path datasetPath;
    LocalNamedExampleLayout layout;
    std::vector<std::shared_ptr<Shard>> shards;
    std::vector<uint64_t> shardGlobalStarts;
    std::vector<uint64_t> shardTrainCounts;

    std::unique_ptr<IndexedLocalNamedBatchAssembler> trainAssembler;
    std::unique_ptr<IndexedLocalNamedBatchAssembler> validateAssembler;
    std::unique_ptr<IndexedLocalNamedBatchAssembler> testAssembler;

    uint64_t numDatasetExamples = 0;
    uint64_t batchQueueDepth = 32;
    bool randomizeTrain = true;
    std::optional<uint64_t> seed;
    bool explicitTestSplit = false;

    struct IndexedShardManifestEntry {
        std::string filename;
        uint64_t globalStart = 0;
        uint64_t numExamples = 0;
    };

    static std::vector<IndexedShardManifestEntry> readIndexedShardManifestEntries(const std::filesystem::path &manifestPath);
    void openDataset(const LocalNamedExampleLayout &requestedLayout);
    void validateIndex(uint64_t index, const char *splitName) const;
    void validateIndices(const std::vector<uint64_t> &indices, const char *splitName) const;
    std::unique_ptr<IndexedLocalNamedBatchAssembler> createAssembler(std::vector<uint64_t> indices,
                                                                     const char *splitName,
                                                                     bool randomized,
                                                                     std::optional<uint64_t> splitSeed) const;
    IndexedLocalNamedBatchAssembler *assemblerFor(ExampleType exampleType);
    const IndexedLocalNamedBatchAssembler *assemblerFor(ExampleType exampleType) const;
};
