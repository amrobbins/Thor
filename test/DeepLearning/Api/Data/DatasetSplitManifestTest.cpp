#include "DeepLearning/Api/Data/BatchPolicy.h"
#include "DeepLearning/Api/Data/DatasetSplitManifest.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
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

   protected:
    std::shared_ptr<Thor::BatchSession> openBatchSession(
        const Thor::DatasetSplitManifest &,
        const Thor::BatchPolicy &,
        const Thor::DatasetAccessPolicy &,
        uint64_t,
        const Thor::DatasetFieldMaterializationRequirements &) const override {
        return nullptr;
    }

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
    EXPECT_EQ(manifest.getTrain().materialize(), (std::vector<uint64_t>{0, 2, 4}));
    EXPECT_EQ(manifest.getValidate().materialize(), (std::vector<uint64_t>{1, 3, 5}));
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
    EXPECT_EQ(manifest.getTest().materialize(), (std::vector<uint64_t>{4, 5}));
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
    EXPECT_EQ(loaded.getTest().materialize(), (std::vector<uint64_t>{4, 5}));
}


TEST(DatasetSplitManifest, SupportsNamedValidationPopulationsAndDefaultSelection) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-named-validation"), 10);
    Thor::DatasetSplitManifest manifest(
        dataset,
        Thor::ExampleIndexSet(std::vector<uint64_t>{0, 1, 2, 3}),
        Thor::DatasetSplitManifest::ValidationIndexSets{
            {"seen_sku", Thor::ExampleIndexSet(std::vector<uint64_t>{4, 5, 6})},
            {"unseen_sku", Thor::ExampleIndexSet(std::vector<uint64_t>{7, 8, 9})}},
        "unseen_sku");

    EXPECT_EQ(manifest.getValidationNames(), (std::vector<std::string>{"seen_sku", "unseen_sku"}));
    EXPECT_EQ(manifest.getDefaultValidationName(), "unseen_sku");
    EXPECT_EQ(manifest.getValidate().materialize(), (std::vector<uint64_t>{7, 8, 9}));
    EXPECT_EQ(manifest.getValidation("seen_sku").materialize(), (std::vector<uint64_t>{4, 5, 6}));
    EXPECT_EQ(&manifest.getValidate(), &manifest.getTest());

    Thor::DatasetSplitManifest seenDefault = manifest.withDefaultValidation("seen_sku");
    EXPECT_EQ(seenDefault.getDefaultValidationName(), "seen_sku");
    EXPECT_EQ(seenDefault.getValidate().materialize(), (std::vector<uint64_t>{4, 5, 6}));
    EXPECT_EQ(&seenDefault.getValidate(), &seenDefault.getTest());
    EXPECT_EQ(manifest.getDefaultValidationName(), "unseen_sku");

    Thor::DatasetSplitManifest explicitTest(
        dataset,
        Thor::ExampleIndexSet(std::vector<uint64_t>{0, 1}),
        Thor::DatasetSplitManifest::ValidationIndexSets{
            {"seen_sku", Thor::ExampleIndexSet(std::vector<uint64_t>{2, 3})},
            {"unseen_sku", Thor::ExampleIndexSet(std::vector<uint64_t>{4, 5})}},
        "unseen_sku",
        Thor::ExampleIndexSet(std::vector<uint64_t>{6, 7}));
    Thor::DatasetSplitManifest explicitSeen =
        explicitTest.withDefaultValidation("seen_sku");
    EXPECT_TRUE(explicitSeen.hasExplicitTestSplit());
    EXPECT_EQ(explicitSeen.getTest().materialize(),
              (std::vector<uint64_t>{6, 7}));
}

TEST(DatasetSplitManifest, NamedValidationPersistenceRoundTrips) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-named-validation-persist"), 12);
    Thor::DatasetSplitManifest original(
        dataset,
        Thor::ExampleIndexSet::contiguous(0, 6),
        Thor::DatasetSplitManifest::ValidationIndexSets{
            {"seen_sku", Thor::ExampleIndexSet::contiguous(6, 3)},
            {"unseen_sku", Thor::ExampleIndexSet::contiguous(9, 3)}},
        "unseen_sku");
    const std::filesystem::path path = tempManifestPath();

    original.save(path);
    std::ifstream in(path);
    const nlohmann::json persisted = nlohmann::json::parse(in);
    EXPECT_EQ(persisted.at("format"), "thor.dataset_split_manifest.v2");
    EXPECT_EQ(persisted.at("default_validation"), "unseen_sku");
    EXPECT_TRUE(persisted.at("partitions").at("validations").contains("seen_sku"));
    EXPECT_TRUE(persisted.at("partitions").at("validations").contains("unseen_sku"));

    Thor::DatasetSplitManifest loaded = Thor::DatasetSplitManifest::load(path);
    std::filesystem::remove(path);
    EXPECT_EQ(loaded, original);
    EXPECT_TRUE(loaded.testAliasesValidate());
    EXPECT_EQ(&loaded.getValidation("unseen_sku"), &loaded.getTest());
}

TEST(DatasetSplitManifest, LoadsLegacyV1AsSingleValidatePopulation) {
    const Thor::DatasetId datasetId = Thor::DatasetId::fromStableMaterial("legacy-v1");
    const std::filesystem::path path = tempManifestPath();
    nlohmann::json legacy = {
        {"format", "thor.dataset_split_manifest.v1"},
        {"dataset_id", datasetId.str()},
        {"num_examples", 4},
        {"partitions",
         {{"train", std::vector<uint64_t>{0, 1}},
          {"validate", std::vector<uint64_t>{2, 3}},
          {"test", {{"alias", "validate"}}}}}};
    std::ofstream(path) << legacy.dump(2) << '\n';

    Thor::DatasetSplitManifest loaded = Thor::DatasetSplitManifest::load(path);
    std::filesystem::remove(path);
    EXPECT_EQ(loaded.getDefaultValidationName(), "validate");
    EXPECT_EQ(loaded.getValidationNames(), (std::vector<std::string>{"validate"}));
    EXPECT_EQ(loaded.getValidate().materialize(), (std::vector<uint64_t>{2, 3}));
    EXPECT_TRUE(loaded.testAliasesValidate());
}

TEST(DatasetSplitManifest, RejectsInvalidNamedValidationConfiguration) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-invalid-validation"), 4);
    EXPECT_THROW(
        (Thor::DatasetSplitManifest(
            dataset,
            Thor::ExampleIndexSet(std::vector<uint64_t>{0}),
            Thor::DatasetSplitManifest::ValidationIndexSets{},
            "validate")),
        std::runtime_error);
    EXPECT_THROW(
        (Thor::DatasetSplitManifest(
            dataset,
            Thor::ExampleIndexSet(std::vector<uint64_t>{0}),
            Thor::DatasetSplitManifest::ValidationIndexSets{
                {"seen_sku", Thor::ExampleIndexSet(std::vector<uint64_t>{1})}},
            "unseen_sku")),
        std::runtime_error);
    EXPECT_THROW(
        (Thor::DatasetSplitManifest(
            dataset,
            Thor::ExampleIndexSet(std::vector<uint64_t>{0}),
            Thor::DatasetSplitManifest::ValidationIndexSets{
                {"bad name", Thor::ExampleIndexSet(std::vector<uint64_t>{1})}},
            "bad name")),
        std::runtime_error);
}

TEST(DatasetSplitManifest, RangeBackedMembershipProvidesRandomAccessWithoutMaterialization) {
    Thor::ExampleIndexSet indices(std::vector<Thor::ExampleIndexRange>{
        Thor::ExampleIndexRange{.start = 10, .count = 3, .stride = 2},
        Thor::ExampleIndexRange{.start = 20, .count = 2, .stride = 5}});

    EXPECT_TRUE(indices.isRangeBacked());
    EXPECT_EQ(indices.size(), 5u);
    EXPECT_EQ(indices.at(0), 10u);
    EXPECT_EQ(indices.at(2), 14u);
    EXPECT_EQ(indices.at(3), 20u);
    EXPECT_EQ(indices.at(4), 25u);
    EXPECT_EQ(indices.materialize(), (std::vector<uint64_t>{10, 12, 14, 20, 25}));
}

TEST(DatasetSplitManifest, CompactRangePersistenceRoundTripsWithoutExpandingIndices) {
    TestNamedDataset dataset(Thor::DatasetId::fromStableMaterial("dataset-ranges"), 100);
    Thor::DatasetSplitManifest original(
        dataset,
        Thor::ExampleIndexSet::strided(0, 20, 2),
        Thor::ExampleIndexSet::contiguous(40, 10),
        Thor::ExampleIndexSet(std::vector<Thor::ExampleIndexRange>{
            Thor::ExampleIndexRange{.start = 60, .count = 5, .stride = 3}}));
    const std::filesystem::path path = tempManifestPath();

    original.save(path);
    std::ifstream in(path);
    const nlohmann::json persisted = nlohmann::json::parse(in);
    ASSERT_TRUE(persisted.at("partitions").at("train").is_object());
    ASSERT_TRUE(persisted.at("partitions").at("train").contains("ranges"));
    EXPECT_EQ(persisted.at("partitions").at("train").at("ranges").size(), 1u);

    Thor::DatasetSplitManifest loaded = Thor::DatasetSplitManifest::load(path);
    std::filesystem::remove(path);
    EXPECT_EQ(loaded, original);
    EXPECT_TRUE(loaded.getTrain().isRangeBacked());
    EXPECT_TRUE(loaded.getValidate().isRangeBacked());
    EXPECT_TRUE(loaded.getTest().isRangeBacked());
}

TEST(DatasetSplitManifest, InterleavedStridedRangesAreAllowedWhenTheirRowsAreDisjoint) {
    Thor::ExampleIndexSet indices(std::vector<Thor::ExampleIndexRange>{
        Thor::ExampleIndexRange{.start = 0, .count = 4, .stride = 3},
        Thor::ExampleIndexRange{.start = 1, .count = 4, .stride = 3},
        Thor::ExampleIndexRange{.start = 2, .count = 4, .stride = 3}});

    EXPECT_EQ(indices.materialize(), (std::vector<uint64_t>{0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11}));
}

TEST(DatasetSplitManifest, RejectsInvalidOrIntersectingStridedRanges) {
    EXPECT_THROW((Thor::ExampleIndexSet(std::vector<Thor::ExampleIndexRange>{
                     Thor::ExampleIndexRange{.start = 0, .count = 1, .stride = 0}})),
                 std::runtime_error);
    EXPECT_THROW((Thor::ExampleIndexSet(std::vector<Thor::ExampleIndexRange>{
                     Thor::ExampleIndexRange{.start = 0, .count = 4, .stride = 2},
                     Thor::ExampleIndexRange{.start = 2, .count = 2, .stride = 4}})),
                 std::runtime_error);
    EXPECT_NO_THROW((Thor::ExampleIndexSet(std::vector<Thor::ExampleIndexRange>{
        Thor::ExampleIndexRange{.start = 0, .count = 3, .stride = 2},
        Thor::ExampleIndexRange{.start = 3, .count = 2, .stride = 2}})));
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
