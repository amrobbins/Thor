#pragma once

#include "DeepLearning/Api/Data/DatasetId.h"
#include "DeepLearning/Api/Data/NamedDataset.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace Thor {

/** Immutable canonical example membership for one dataset partition. */
class ExampleIndexSet {
   public:
    explicit ExampleIndexSet(std::vector<uint64_t> indices);

    [[nodiscard]] const std::vector<uint64_t> &getIndices() const { return *indices; }
    [[nodiscard]] const std::shared_ptr<const std::vector<uint64_t>> &getSharedIndices() const { return indices; }
    [[nodiscard]] uint64_t size() const { return static_cast<uint64_t>(indices->size()); }
    [[nodiscard]] bool empty() const { return indices->empty(); }

    bool operator==(const ExampleIndexSet &rhs) const { return *indices == *rhs.indices; }
    bool operator!=(const ExampleIndexSet &rhs) const { return !(*this == rhs); }

   private:
    std::shared_ptr<const std::vector<uint64_t>> indices;
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

    [[nodiscard]] static DatasetSplitManifest load(const std::filesystem::path &path);
    void save(const std::filesystem::path &path) const;

    [[nodiscard]] const DatasetId &getDatasetId() const { return datasetId; }
    [[nodiscard]] uint64_t getNumExamples() const { return numExamples; }
    [[nodiscard]] const ExampleIndexSet &getTrain() const { return *train; }
    [[nodiscard]] const ExampleIndexSet &getValidate() const { return *validate; }
    [[nodiscard]] const ExampleIndexSet &getTest() const { return *test; }
    [[nodiscard]] bool hasExplicitTestSplit() const { return explicitTestSplit; }
    [[nodiscard]] bool testAliasesValidate() const { return test == validate; }

    void validateAgainst(const NamedDataset &dataset) const;

    bool operator==(const DatasetSplitManifest &rhs) const;
    bool operator!=(const DatasetSplitManifest &rhs) const { return !(*this == rhs); }

   private:
    DatasetSplitManifest(DatasetId datasetId,
                         uint64_t numExamples,
                         std::vector<uint64_t> trainIndices,
                         std::vector<uint64_t> validateIndices,
                         std::optional<std::vector<uint64_t>> testIndices);

    DatasetId datasetId;
    uint64_t numExamples;
    std::shared_ptr<const ExampleIndexSet> train;
    std::shared_ptr<const ExampleIndexSet> validate;
    std::shared_ptr<const ExampleIndexSet> test;
    bool explicitTestSplit;

    static std::shared_ptr<const ExampleIndexSet> makeIndexSet(std::vector<uint64_t> indices,
                                                               uint64_t numExamples,
                                                               const char *partitionName);
};

}  // namespace Thor
