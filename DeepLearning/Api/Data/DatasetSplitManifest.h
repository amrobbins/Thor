#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/NamedDataset.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace Thor {

/** One monotonically increasing arithmetic progression of canonical row ids. */
struct ExampleIndexRange {
    uint64_t start = 0;
    uint64_t count = 0;
    uint64_t stride = 1;

    [[nodiscard]] uint64_t at(uint64_t position) const;
    [[nodiscard]] uint64_t last() const;

    bool operator==(const ExampleIndexRange &rhs) const = default;
};

/**
 * Immutable canonical example membership for one dataset partition.
 *
 * Small or irregular partitions may use explicit indices. Large regular
 * partitions may instead use one or more duplicate-free strided ranges.
 * Random access remains O(1) for explicit indices and O(log R) for R ranges.
 */
class ExampleIndexSet {
   public:
    explicit ExampleIndexSet(std::vector<uint64_t> indices);
    explicit ExampleIndexSet(std::vector<ExampleIndexRange> ranges);

    [[nodiscard]] static ExampleIndexSet contiguous(uint64_t start, uint64_t count);
    [[nodiscard]] static ExampleIndexSet strided(uint64_t start, uint64_t count, uint64_t stride);

    [[nodiscard]] uint64_t at(uint64_t position) const;
    [[nodiscard]] uint64_t size() const { return logicalSize; }
    [[nodiscard]] bool empty() const { return logicalSize == 0; }
    [[nodiscard]] bool isRangeBacked() const { return rangeStorage != nullptr; }
    [[nodiscard]] const std::vector<ExampleIndexRange> &getRanges() const;
    [[nodiscard]] std::vector<uint64_t> materialize() const;

    bool operator==(const ExampleIndexSet &rhs) const;
    bool operator!=(const ExampleIndexSet &rhs) const { return !(*this == rhs); }

   private:
    struct RangeStorage {
        std::vector<ExampleIndexRange> ranges;
        std::vector<uint64_t> cumulativeEnds;
    };

    std::shared_ptr<const std::vector<uint64_t>> explicitIndices;
    std::shared_ptr<const RangeStorage> rangeStorage;
    uint64_t logicalSize = 0;
};

/**
 * Immutable train/validate/test membership bound to one immutable NamedDataset.
 *
 * The manifest contains only canonical row ids. It intentionally contains no
 * batching, randomization, queue, placement, or tensor state.
 */
class DatasetSplitManifest {
   public:
    DatasetSplitManifest(const NamedDataset &dataset,
                         std::vector<uint64_t> trainIndices,
                         std::vector<uint64_t> validateIndices,
                         std::optional<std::vector<uint64_t>> testIndices = std::nullopt);
    DatasetSplitManifest(const NamedDataset &dataset,
                         ExampleIndexSet trainIndices,
                         ExampleIndexSet validateIndices,
                         std::optional<ExampleIndexSet> testIndices = std::nullopt);

    [[nodiscard]] static DatasetSplitManifest load(const std::filesystem::path &path);
    void save(const std::filesystem::path &path) const;

    [[nodiscard]] const DatasetId &getDatasetId() const { return datasetId; }
    [[nodiscard]] uint64_t getNumExamples() const { return numExamples; }
    [[nodiscard]] const ExampleIndexSet &getTrain() const { return *train; }
    [[nodiscard]] const ExampleIndexSet &getValidate() const { return *validate; }
    [[nodiscard]] const ExampleIndexSet &getTest() const { return *test; }
    [[nodiscard]] const std::shared_ptr<const ExampleIndexSet> &getSharedTrain() const { return train; }
    [[nodiscard]] const std::shared_ptr<const ExampleIndexSet> &getSharedValidate() const { return validate; }
    [[nodiscard]] const std::shared_ptr<const ExampleIndexSet> &getSharedTest() const { return test; }
    [[nodiscard]] bool hasExplicitTestSplit() const { return explicitTestSplit; }
    [[nodiscard]] bool testAliasesValidate() const { return test == validate; }

    void validateAgainst(const NamedDataset &dataset) const;

    bool operator==(const DatasetSplitManifest &rhs) const;
    bool operator!=(const DatasetSplitManifest &rhs) const { return !(*this == rhs); }

   private:
    DatasetSplitManifest(DatasetId datasetId,
                         uint64_t numExamples,
                         ExampleIndexSet trainIndices,
                         ExampleIndexSet validateIndices,
                         std::optional<ExampleIndexSet> testIndices);

    DatasetId datasetId;
    uint64_t numExamples;
    std::shared_ptr<const ExampleIndexSet> train;
    std::shared_ptr<const ExampleIndexSet> validate;
    std::shared_ptr<const ExampleIndexSet> test;
    bool explicitTestSplit;

    static std::shared_ptr<const ExampleIndexSet> makeIndexSet(ExampleIndexSet indices,
                                                               uint64_t numExamples,
                                                               const char *partitionName);
};

}  // namespace Thor
