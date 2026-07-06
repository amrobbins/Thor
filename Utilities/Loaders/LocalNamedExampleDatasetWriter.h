#pragma once

#include "DeepLearning/Implementation/Tensor/DataType.h"
#include "Utilities/Loaders/LocalNamedExampleLayout.h"
#include "Utilities/Loaders/Shard.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

class LocalNamedExampleDatasetWriter {
   public:
    enum class StorageMode { SPLIT, INDEXED };

    struct TensorView {
        ThorImplementation::DataType dataType;
        std::vector<uint64_t> dimensions;
        const void *data;
        uint64_t numBytes;
    };

    struct TensorBatchView {
        ThorImplementation::DataType dataType;
        std::vector<uint64_t> dimensions;
        const void *data;
        uint64_t numBytes;
    };

    struct WindowedTensorReferenceView {
        ThorImplementation::DataType keyDataType;
        ThorImplementation::DataType indexDataType;
        const void *key;
        const void *start;
    };

    struct WindowedTensorReferenceBatchView {
        ThorImplementation::DataType keyDataType;
        ThorImplementation::DataType indexDataType;
        const void *keys;
        const void *starts;
        uint64_t count;
    };

    struct WindowedTensorSourceView {
        ThorImplementation::DataType dataType;
        const void *key;
        int64_t startIndex;
        std::vector<uint64_t> dimensions;
        const void *data;
        uint64_t numBytes;
    };

    static constexpr const char *MANIFEST_FILENAME = "manifest.json";
    static constexpr const char *STORAGE_MODE_SPLIT = "split";
    static constexpr const char *STORAGE_MODE_INDEXED = "indexed";

    LocalNamedExampleDatasetWriter(std::filesystem::path datasetPath,
                                   LocalNamedExampleLayout layout,
                                   uint64_t examplesPerShard,
                                   StorageMode storageMode = StorageMode::SPLIT,
                                   std::optional<uint64_t> expectedNumExamples = std::nullopt,
                                   bool preallocate = false);
    ~LocalNamedExampleDatasetWriter();

    LocalNamedExampleDatasetWriter(const LocalNamedExampleDatasetWriter &) = delete;
    LocalNamedExampleDatasetWriter &operator=(const LocalNamedExampleDatasetWriter &) = delete;
    LocalNamedExampleDatasetWriter(LocalNamedExampleDatasetWriter &&) = delete;
    LocalNamedExampleDatasetWriter &operator=(LocalNamedExampleDatasetWriter &&) = delete;

    void writeExample(ExampleType exampleType,
                      const std::map<std::string, TensorView> &tensors,
                      const std::string &label = std::string(),
                      const std::string &filename = std::string());
    void writeIndexedExample(const std::map<std::string, TensorView> &tensors,
                             const std::string &label = std::string(),
                             const std::string &filename = std::string());
    void writeIndexedExample(const std::map<std::string, TensorView> &tensors,
                             const std::map<std::string, WindowedTensorReferenceView> &windowedTensorReferences,
                             const std::string &label = std::string(),
                             const std::string &filename = std::string());
    void writeIndexedExamples(const std::map<std::string, TensorBatchView> &tensors);
    void writeIndexedExamples(const std::map<std::string, TensorBatchView> &tensors,
                              const std::map<std::string, WindowedTensorReferenceBatchView> &windowedTensorReferences);
    void writeWindowedTensorSource(std::string_view tensorName, const WindowedTensorSourceView &source);
    void close();

    [[nodiscard]] bool isClosed() const;
    [[nodiscard]] const std::filesystem::path &path() const;
    [[nodiscard]] std::filesystem::path manifestPath() const;
    [[nodiscard]] uint64_t numExamples() const;
    [[nodiscard]] uint64_t numExamples(ExampleType exampleType) const;
    [[nodiscard]] const LocalNamedExampleLayout &getLayout() const;
    [[nodiscard]] StorageMode getStorageMode() const;
    [[nodiscard]] std::optional<uint64_t> getExpectedNumExamples() const;
    [[nodiscard]] bool getPreallocate() const;

    static StorageMode storageModeFromString(std::string_view value);
    static const char *storageModeToString(StorageMode storageMode);
    static StorageMode readStorageMode(const std::filesystem::path &manifestPath);

   private:
    struct ShardManifestEntry {
        std::string filename;
        uint64_t globalStart = 0;
        uint64_t capacityExamples = 0;
        uint64_t trainExamples = 0;
        uint64_t validateExamples = 0;
        uint64_t testExamples = 0;

        [[nodiscard]] uint64_t totalExamples() const;
        [[nodiscard]] uint64_t examples(ExampleType exampleType) const;
        [[nodiscard]] uint64_t remainingCapacity() const;
        [[nodiscard]] uint64_t numBytes(uint64_t recordSizeBytes) const;
        void increment(ExampleType exampleType);
        void incrementBy(ExampleType exampleType, uint64_t count);
    };

    std::filesystem::path datasetPath;
    LocalNamedExampleLayout layout;
    uint64_t examplesPerShard;
    StorageMode storageMode;
    std::optional<uint64_t> expectedNumExamples;
    bool preallocate;
    bool closed;

    std::unique_ptr<Shard> currentShard;
    uint64_t nextShardIndex;
    std::vector<ShardManifestEntry> shardEntries;

    struct WindowedTensorSourceManifestEntry {
        std::string filename;
        uint64_t numBytes = 0;
        std::set<std::string> keyHexValues;
        std::vector<LocalNamedExampleLayout::WindowedTensorSourceSequence> sequences;
    };

    std::map<std::string, WindowedTensorSourceManifestEntry> windowedTensorSources;
    uint64_t totalTrainExamples;
    uint64_t totalValidateExamples;
    uint64_t totalTestExamples;

    void validateWritable() const;
    void validateTensorMapExact(const std::map<std::string, TensorView> &tensors) const;
    uint64_t validateTensorBatchMapExact(const std::map<std::string, TensorBatchView> &tensors) const;
    void validateWindowedTensorReferenceMapExact(
        const std::map<std::string, WindowedTensorReferenceView> &windowedTensorReferences) const;
    uint64_t validateTensorAndWindowedTensorReferenceBatchMapsExact(
        const std::map<std::string, TensorBatchView> &tensors,
        const std::map<std::string, WindowedTensorReferenceBatchView> &windowedTensorReferences) const;
    std::vector<uint8_t> packRecord(const std::map<std::string, TensorView> &tensors) const;
    std::vector<uint8_t> packRecord(const std::map<std::string, TensorView> &tensors,
                                    const std::map<std::string, WindowedTensorReferenceView> &windowedTensorReferences) const;
    std::vector<uint8_t> packRecords(const std::map<std::string, TensorBatchView> &tensors, uint64_t count) const;
    std::vector<uint8_t> packRecords(const std::map<std::string, TensorBatchView> &tensors,
                                     const std::map<std::string, WindowedTensorReferenceBatchView> &windowedTensorReferences,
                                     uint64_t count) const;
    uint64_t nextShardCapacity() const;
    void ensureCurrentShard();
    void finalizeCurrentShard();
    void writeManifest() const;
    void writePackedRecord(ExampleType exampleType,
                           const std::map<std::string, TensorView> &tensors,
                           const std::string &label,
                           const std::string &filename);
    void writePackedIndexedRecords(const uint8_t *records, uint64_t count);
};
