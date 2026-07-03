#pragma once

#include "DeepLearning/Api/Loaders/Loader.h"
#include "Utilities/Loaders/LocalNamedBatchAssembler.h"
#include "Utilities/Loaders/LocalNamedExampleDatasetWriter.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class LocalNamedBatchLoader : public Loader {
   public:
    LocalNamedBatchLoader(std::filesystem::path datasetPath,
                          LocalNamedExampleLayout requestedLayout,
                          uint64_t batchSize,
                          uint64_t batchQueueDepth = 32,
                          bool randomizeTrain = true,
                          std::optional<uint64_t> seed = std::nullopt);

    Batch getBatch(ExampleType exampleType, uint64_t &batchNum) override;
    void returnBatchBuffers(ExampleType exampleType, Batch &&batch) override;

    uint64_t getNumBatchesPerEpoch(ExampleType exampleType) override;
    uint64_t getNumExamples(ExampleType exampleType) override;
    uint64_t getNextBatchNum(ExampleType exampleType) override;

    [[nodiscard]] const LocalNamedExampleLayout &getLayout() const;
    [[nodiscard]] const std::filesystem::path &getDatasetPath() const;

   private:
    std::filesystem::path datasetPath;
    LocalNamedExampleLayout layout;
    std::vector<std::shared_ptr<Shard>> shards;

    std::shared_ptr<LocalNamedBatchAssembler> batchAssemblerTrain;
    std::shared_ptr<LocalNamedBatchAssembler> batchAssemblerValidate;
    std::shared_ptr<LocalNamedBatchAssembler> batchAssemblerTest;

    static std::vector<std::string> readShardFilenames(const std::filesystem::path &manifestPath);
    std::shared_ptr<LocalNamedBatchAssembler> maybeCreateAssembler(ExampleType exampleType,
                                                                   uint64_t batchQueueDepth,
                                                                   bool randomizeExamples,
                                                                   std::optional<uint64_t> seed);
    LocalNamedBatchAssembler &assemblerOrThrow(ExampleType exampleType);
    const LocalNamedBatchAssembler &assemblerOrThrow(ExampleType exampleType) const;
};
