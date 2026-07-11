#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class TestNamedDataset final : public Thor::NamedDataset {
   public:
    TestNamedDataset(Thor::DatasetId id, uint64_t numExamples)
        : id(std::move(id)),
          numExamples(numExamples),
          schema(std::vector<Thor::DatasetField>{{.id = 0,
                                                  .name = "x",
                                                  .dataType = ThorImplementation::DataType::FP32,
                                                  .dimensions = {1},
                                                  .kind = Thor::DatasetFieldKind::DENSE}}) {}

    const Thor::DatasetId &getId() const override { return id; }
    uint64_t getNumExamples() const override { return numExamples; }
    const Thor::DatasetSchema &getSchema() const override { return schema; }
    const Thor::DatasetField &getField(std::string_view name) const override { return schema.getField(name); }

   private:
    Thor::DatasetId id;
    uint64_t numExamples;
    Thor::DatasetSchema schema;
};

std::filesystem::path tempManifestPath() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("thor_dataset_split_manifest_" + std::to_string(now) + ".json");
}

TEST(DatasetSplitManifest, BindsMembershipToDatasetAndAliasesDefaultTestPartition) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-a"), 6);
    Thor::DatasetSplitManifest manifest(dataset, {0, 2, 4}, {1, 3, 5});

    EXPECT_EQ(manifest.getDatasetId(), dataset.getId());
    EXPECT_EQ(manifest.getNumExamples(), 6u);
    EXPECT_EQ(manifest.getTrain().getIndices(), (std::vector<uint64_t>{0, 2, 4}));
    EXPECT_EQ(manifest.getValidate().getIndices(), (std::vector<uint64_t>{1, 3, 5}));
    EXPECT_FALSE(manifest.hasExplicitTestSplit());
    EXPECT_TRUE(manifest.testAliasesValidate());
    EXPECT_EQ(&manifest.getValidate(), &manifest.getTest());
}

TEST(DatasetSplitManifest, ExplicitTestPartitionHasIndependentImmutableMembership) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-a"), 6);
    Thor::DatasetSplitManifest manifest(dataset, {0, 1, 2}, {3}, std::vector<uint64_t>{4, 5});

    EXPECT_TRUE(manifest.hasExplicitTestSplit());
    EXPECT_FALSE(manifest.testAliasesValidate());
    EXPECT_NE(&manifest.getValidate(), &manifest.getTest());
    EXPECT_EQ(manifest.getTest().getIndices(), (std::vector<uint64_t>{4, 5}));
}

TEST(DatasetSplitManifest, RejectsOutOfRangeAndDuplicateMembership) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-a"), 3);

    EXPECT_THROW((Thor::DatasetSplitManifest(dataset, {0, 3}, {1})), std::runtime_error);
    EXPECT_THROW((Thor::DatasetSplitManifest(dataset, {0, 0}, {1})), std::runtime_error);
}

TEST(DatasetSplitManifest, RejectsDifferentDatasetIdentityOrGenerationSize) {
    TestNamedDataset datasetA(Thor::DatasetId::fromStableMaterial("dataset-a"), 4);
    TestNamedDataset datasetB(Thor::DatasetId::fromStableMaterial("dataset-b"), 4);
    TestNamedDataset resizedA(datasetA.getId(), 5);
    Thor::DatasetSplitManifest manifest(datasetA, {0, 1, 2}, {3});

    EXPECT_NO_THROW(manifest.validateAgainst(datasetA));
    EXPECT_THROW(manifest.validateAgainst(datasetB), std::runtime_error);
    EXPECT_THROW(manifest.validateAgainst(resizedA), std::runtime_error);
}

TEST(DatasetSplitManifest, PersistenceRoundTripsAndPreservesValidateTestAlias) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-a"), 5);
    Thor::DatasetSplitManifest original(dataset, {0, 2, 4}, {1, 3});
    const std::filesystem::path path = tempManifestPath();

    original.save(path);
    Thor::DatasetSplitManifest loaded = Thor::DatasetSplitManifest::load(path);
    std::filesystem::remove(path);

    EXPECT_EQ(loaded, original);
    EXPECT_TRUE(loaded.testAliasesValidate());
    EXPECT_EQ(&loaded.getValidate(), &loaded.getTest());
}

TEST(DatasetSplitManifest, PersistenceRoundTripsExplicitTestMembership) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-a"), 6);
    Thor::DatasetSplitManifest original(dataset, {0, 1, 2}, {3}, std::vector<uint64_t>{4, 5});
    const std::filesystem::path path = tempManifestPath();

    original.save(path);
    Thor::DatasetSplitManifest loaded = Thor::DatasetSplitManifest::load(path);
    std::filesystem::remove(path);

    EXPECT_EQ(loaded, original);
    EXPECT_TRUE(loaded.hasExplicitTestSplit());
    EXPECT_FALSE(loaded.testAliasesValidate());
    EXPECT_EQ(loaded.getTest().getIndices(), (std::vector<uint64_t>{4, 5}));
}

TEST(BatchPolicy, ValidatesBatchingWithoutOwningExecutionState) {
    Thor::BatchPolicy policy(128, true, 17);
    EXPECT_EQ(policy.getBatchSize(), 128u);
    EXPECT_TRUE(policy.getRandomizeTrain());
    EXPECT_EQ(policy.getRandomSeed(), std::optional<uint64_t>(17));

    EXPECT_THROW((Thor::BatchPolicy(0)), std::runtime_error);
    EXPECT_THROW((Thor::BatchPolicy(32, false, 17)), std::runtime_error);
}

}  // namespace
