#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/DatasetLayout.h"
#include "DeepLearning/Implementation/Tensor/DataType.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

class Shard;

class DatasetWriter {
   public:
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

    /**
     * One compact reference formula for a segment appended by writeAffineExamples().
     * For segment-local row r, start(r) = base + r * stride + fieldOffset.
     */
    struct AffineWindowedTensorReferenceView {
        ThorImplementation::DataType keyDataType;
        const void *key;
        int64_t base = 0;
        int64_t stride = 1;
        int64_t fieldOffset = 0;
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
    static constexpr const char *STORAGE_MODE_INDEXED = "indexed";

    DatasetWriter(std::filesystem::path datasetPath,
                  DatasetLayout layout,
                  uint64_t examplesPerShard,
                  std::optional<uint64_t> expectedNumExamples = std::nullopt,
                  bool preallocate = false);
    ~DatasetWriter();

    DatasetWriter(const DatasetWriter &) = delete;
    DatasetWriter &operator=(const DatasetWriter &) = delete;
    DatasetWriter(DatasetWriter &&) = delete;
    DatasetWriter &operator=(DatasetWriter &&) = delete;

    void writeIndexedExample(const std::map<std::string, TensorView> &tensors);
    void writeIndexedExample(const std::map<std::string, TensorView> &tensors,
                             const std::map<std::string, WindowedTensorReferenceView> &windowedTensorReferences);
    void writeIndexedExamples(const std::map<std::string, TensorBatchView> &tensors);
    void writeIndexedExamples(const std::map<std::string, TensorBatchView> &tensors,
                              const std::map<std::string, WindowedTensorReferenceBatchView> &windowedTensorReferences);
    void writeAffineExamples(
        uint64_t count,
        const std::map<std::string, TensorBatchView> &tensors,
        const std::map<std::string, AffineWindowedTensorReferenceView> &windowedTensorReferences);
    void writeWindowSource(std::string_view sourceName, const WindowedTensorSourceView &source);
    void close();

    [[nodiscard]] bool isClosed() const;
    [[nodiscard]] const std::filesystem::path &path() const;
    [[nodiscard]] std::filesystem::path manifestPath() const;
    [[nodiscard]] uint64_t numExamples() const;
    [[nodiscard]] const Thor::DatasetId &getDatasetId() const { return datasetId; }
    [[nodiscard]] const DatasetLayout &getLayout() const;
    [[nodiscard]] std::optional<uint64_t> getExpectedNumExamples() const;
    [[nodiscard]] bool getPreallocate() const;

   private:
    struct ShardManifestEntry {
        std::string filename;
        uint64_t globalStart = 0;
        uint64_t capacityExamples = 0;
        uint64_t numExamples = 0;

        [[nodiscard]] uint64_t remainingCapacity() const;
        [[nodiscard]] uint64_t numBytes(uint64_t recordSizeBytes) const;
    };

    std::filesystem::path datasetPath;
    Thor::DatasetId datasetId;
    DatasetLayout layout;
    uint64_t examplesPerShard;
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
        std::vector<DatasetLayout::WindowedTensorSourceSequence> sequences;
    };

    std::map<std::string, WindowedTensorSourceManifestEntry> windowSources;
    struct AffineWindowedTensorReferenceManifestEntry {
        std::string keyHex;
        int64_t base = 0;
        int64_t stride = 1;
        int64_t fieldOffset = 0;
    };

    struct AffineWindowReferenceSegment {
        uint64_t rowStart = 0;
        uint64_t count = 0;
        std::map<std::string, AffineWindowedTensorReferenceManifestEntry> references;
    };

    std::vector<AffineWindowReferenceSegment> affineWindowReferenceSegments;

    uint64_t totalExamples;

    void validateWritable() const;
    void validateTensorMapExact(const std::map<std::string, TensorView> &tensors) const;
    uint64_t validateTensorBatchMapExact(const std::map<std::string, TensorBatchView> &tensors) const;
    void validateAffineWindowedTensorReferenceMapExact(
        const std::map<std::string, AffineWindowedTensorReferenceView> &windowedTensorReferences,
        uint64_t count) const;
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
    void writePackedIndexedRecords(const uint8_t *records, uint64_t count);
};
