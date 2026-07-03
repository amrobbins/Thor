#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

class LocalNamedExampleDatasetWriter {
   public:
    struct TensorView {
        ThorImplementation::DataType dataType;
        std::vector<uint64_t> dimensions;
        const void *data;
        uint64_t numBytes;
    };

    static constexpr const char *MANIFEST_FILENAME = "manifest.json";

    LocalNamedExampleDatasetWriter(std::filesystem::path datasetPath,
                                   LocalNamedExampleLayout layout,
                                   uint64_t examplesPerShard);
    ~LocalNamedExampleDatasetWriter();

    LocalNamedExampleDatasetWriter(const LocalNamedExampleDatasetWriter &) = delete;
    LocalNamedExampleDatasetWriter &operator=(const LocalNamedExampleDatasetWriter &) = delete;
    LocalNamedExampleDatasetWriter(LocalNamedExampleDatasetWriter &&) = delete;
    LocalNamedExampleDatasetWriter &operator=(LocalNamedExampleDatasetWriter &&) = delete;

    void writeExample(ExampleType exampleType,
                      const std::map<std::string, TensorView> &tensors,
                      const std::string &label = std::string(),
                      const std::string &filename = std::string());
    void close();

    [[nodiscard]] bool isClosed() const;
    [[nodiscard]] const std::filesystem::path &path() const;
    [[nodiscard]] std::filesystem::path manifestPath() const;
    [[nodiscard]] uint64_t numExamples() const;
    [[nodiscard]] uint64_t numExamples(ExampleType exampleType) const;
    [[nodiscard]] const LocalNamedExampleLayout &getLayout() const;

   private:
    struct ShardManifestEntry {
        std::string filename;
        uint64_t trainExamples = 0;
        uint64_t validateExamples = 0;
        uint64_t testExamples = 0;

        [[nodiscard]] uint64_t totalExamples() const;
        [[nodiscard]] uint64_t examples(ExampleType exampleType) const;
        void increment(ExampleType exampleType);
    };

    std::filesystem::path datasetPath;
    LocalNamedExampleLayout layout;
    uint64_t examplesPerShard;
    bool closed;

    std::unique_ptr<Shard> currentShard;
    uint64_t nextShardIndex;
    std::vector<ShardManifestEntry> shardEntries;
    uint64_t totalTrainExamples;
    uint64_t totalValidateExamples;
    uint64_t totalTestExamples;

    void validateWritable() const;
    void validateTensorMapExact(const std::map<std::string, TensorView> &tensors) const;
    std::vector<uint8_t> packRecord(const std::map<std::string, TensorView> &tensors) const;
    void ensureCurrentShard();
    void finalizeCurrentShard();
    void writeManifest() const;
};
