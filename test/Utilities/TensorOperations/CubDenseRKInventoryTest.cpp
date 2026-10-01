#include "Utilities/TensorOperations/Cub/DenseRKInventory.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <set>
#include <utility>

using namespace ThorImplementation::CubReductionInternal;

namespace {

template <typename Predicate>
size_t countProduction(Predicate&& predicate) {
    return static_cast<size_t>(std::count_if(
        denseRKProductionInventory().begin(), denseRKProductionInventory().end(), std::forward<Predicate>(predicate)));
}

}  // namespace

TEST(CubDenseRKInventory, AdmitsExactlyKParallelAndRCooperativeStrategies) {
    const auto strategies = denseRKStrategies();
    ASSERT_EQ(strategies.size(), 2U);
    EXPECT_EQ(strategies[0], DenseRKStrategy::KParallel);
    EXPECT_EQ(strategies[1], DenseRKStrategy::RCooperative);
}

TEST(CubDenseRKInventory, EveryModernFamilyImplementationBelongsToOneOfTheTwoStrategies) {
    size_t modern_count = 0;
    for (const DenseRKProductionInventoryEntry& entry : denseRKProductionInventory()) {
        if (entry.status != DenseRKImplementationStatus::FamilyImplementation) {
            continue;
        }
        ++modern_count;
        ASSERT_TRUE(entry.strategy.has_value()) << entry.symbol;
        ASSERT_TRUE(entry.access.has_value()) << entry.symbol;
        EXPECT_TRUE(entry.strategy.value() == DenseRKStrategy::KParallel
                    || entry.strategy.value() == DenseRKStrategy::RCooperative)
            << entry.symbol;
        EXPECT_TRUE(entry.supports_complete || entry.supports_staged) << entry.symbol;
    }
    EXPECT_GT(modern_count, 0U);
}

TEST(CubDenseRKInventory, NonFamilyPathsCannotBecomeAnImplicitThirdStrategy) {
    size_t non_family_count = 0;
    for (const DenseRKProductionInventoryEntry& entry : denseRKProductionInventory()) {
        if (entry.status == DenseRKImplementationStatus::FamilyImplementation) {
            continue;
        }
        ++non_family_count;
        EXPECT_FALSE(entry.strategy.has_value()) << entry.symbol;
        EXPECT_FALSE(entry.access.has_value()) << entry.symbol;
        EXPECT_TRUE(entry.supports_complete || entry.supports_staged) << entry.symbol;
    }
    EXPECT_GT(non_family_count, 0U);
}

TEST(CubDenseRKInventory, KParallelProductionStageOwnsBothCompleteAndStagedProgress) {
    const auto& entry =
        denseRKProductionInventoryEntry(DenseRKProductionImplementation::KParallelPass);
    ASSERT_EQ(entry.status, DenseRKImplementationStatus::FamilyImplementation);
    ASSERT_EQ(entry.strategy, DenseRKStrategy::KParallel);
    ASSERT_EQ(entry.access, DenseRKAccess::Aligned);
    EXPECT_TRUE(entry.supports_complete);
    EXPECT_TRUE(entry.supports_staged);
    EXPECT_FALSE(entry.owns_successor);
}

TEST(CubDenseRKInventory, RCooperativeAlignedInventoryHasIndependentCompleteAndStagedImplementations) {
    const auto& complete = denseRKProductionInventoryEntry(
        DenseRKProductionImplementation::RCooperativeAlignedComplete);
    const auto& staged = denseRKProductionInventoryEntry(
        DenseRKProductionImplementation::RCooperativeAlignedStaged);

    ASSERT_EQ(complete.strategy, DenseRKStrategy::RCooperative);
    ASSERT_EQ(staged.strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(complete.access, DenseRKAccess::Aligned);
    EXPECT_EQ(staged.access, DenseRKAccess::Aligned);
    EXPECT_TRUE(complete.supports_complete);
    EXPECT_FALSE(complete.supports_staged);
    EXPECT_FALSE(staged.supports_complete);
    EXPECT_TRUE(staged.supports_staged);
    EXPECT_FALSE(complete.owns_successor);
    EXPECT_FALSE(staged.owns_successor);
}

TEST(CubDenseRKInventory, EveryModernFamilyImplementationIsExactlyOnePhysicalPass) {
    for (const DenseRKProductionInventoryEntry& entry : denseRKProductionInventory()) {
        if (entry.status != DenseRKImplementationStatus::FamilyImplementation) {
            continue;
        }
        EXPECT_FALSE(entry.owns_successor) << entry.symbol;
    }
}

TEST(CubDenseRKInventory, RotatedRCooperativeIsAnIndependentStagedFirstPass) {
    const auto& rotated = denseRKProductionInventoryEntry(
        DenseRKProductionImplementation::RCooperativeRotatedStaged);
    ASSERT_EQ(rotated.status, DenseRKImplementationStatus::FamilyImplementation);
    ASSERT_EQ(rotated.strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(rotated.access, DenseRKAccess::Rotated);
    EXPECT_FALSE(rotated.supports_complete);
    EXPECT_TRUE(rotated.supports_staged);
    EXPECT_FALSE(rotated.owns_successor);
    EXPECT_EQ(rotated.symbol, "launchAwkwardAlignmentRotatedShardedFirstStage");
}

TEST(CubDenseRKInventory, NarrowFlatRCooperativeIsAnIndependentStagedFirstPass) {
    const auto& flat = denseRKProductionInventoryEntry(
        DenseRKProductionImplementation::RCooperativeFlatRowsStaged);
    ASSERT_EQ(flat.status, DenseRKImplementationStatus::FamilyImplementation);
    ASSERT_EQ(flat.strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(flat.access, DenseRKAccess::FlatRows);
    EXPECT_FALSE(flat.supports_complete);
    EXPECT_TRUE(flat.supports_staged);
    EXPECT_FALSE(flat.owns_successor);
    EXPECT_EQ(flat.symbol, "launchNarrowLowPrecisionFlatRCooperativeFirstStage");
}

TEST(CubDenseRKInventory, ValueInventoryContainsOnlyModernFamilyImplementations) {
    for (const DenseRKProductionInventoryEntry& entry : denseRKProductionInventory()) {
        if (entry.semantics != DenseRKSemantics::Value) {
            continue;
        }
        EXPECT_EQ(entry.status, DenseRKImplementationStatus::FamilyImplementation) << entry.symbol;
        EXPECT_TRUE(entry.strategy.has_value()) << entry.symbol;
        EXPECT_TRUE(denseRKIsFamilyImplementation(entry.implementation)) << entry.symbol;
        EXPECT_FALSE(entry.owns_successor) << entry.symbol;
    }
}

TEST(CubDenseRKInventory, ArgUsesTheSameStrategyVocabularyWithoutCreatingArgFamilies) {
    const auto& narrow = denseRKProductionInventoryEntry(
        DenseRKProductionImplementation::AlignedAsyncNarrowTiledArg);
    const auto& cooperative = denseRKProductionInventoryEntry(
        DenseRKProductionImplementation::AlignedCooperativeTiledArg);
    const auto& fallback = denseRKProductionInventoryEntry(
        DenseRKProductionImplementation::DirectTiledFixedSegmentArg);

    EXPECT_EQ(narrow.semantics, DenseRKSemantics::Arg);
    EXPECT_EQ(cooperative.semantics, DenseRKSemantics::Arg);
    EXPECT_EQ(narrow.strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(cooperative.strategy, DenseRKStrategy::RCooperative);
    EXPECT_EQ(fallback.status, DenseRKImplementationStatus::LegacyOverlap);
    EXPECT_FALSE(fallback.strategy.has_value());
}

TEST(CubDenseRKInventory, InventoryIdentifiersAreUnique) {
    std::set<DenseRKProductionImplementation> production_ids;
    std::set<std::string_view> production_symbols;
    for (const DenseRKProductionInventoryEntry& entry : denseRKProductionInventory()) {
        EXPECT_TRUE(production_ids.insert(entry.implementation).second);
        EXPECT_TRUE(production_symbols.insert(entry.symbol).second) << entry.symbol;
    }

    std::set<DenseRKBenchmarkImplementation> benchmark_ids;
    for (const DenseRKBenchmarkInventoryEntry& entry : denseRKBenchmarkInventory()) {
        EXPECT_TRUE(benchmark_ids.insert(entry.implementation).second);
        EXPECT_FALSE(entry.source_file.empty());
    }
}

TEST(CubDenseRKInventory, FamilyProductionHarnessSpansStrategiesWithoutOwningAnotherKernel) {
    const auto inventory = denseRKBenchmarkInventory();
    const auto it = std::find_if(
        inventory.begin(),
        inventory.end(),
        [](const DenseRKBenchmarkInventoryEntry& entry) {
            return entry.implementation == DenseRKBenchmarkImplementation::FamilyProductionHarness;
        });
    ASSERT_NE(it, inventory.end());
    EXPECT_FALSE(it->strategy.has_value());
    EXPECT_FALSE(it->owns_cuda_kernel);
    EXPECT_FALSE(it->legacy_reference);
    EXPECT_EQ(it->source_file, "benchmarks/CubReductionBenchmarkDenseRKFamilyCandidate.cu");
}

TEST(CubDenseRKInventory, BenchmarkHarnessCompilesNoDuplicateRKCudaKernels) {
    const auto inventory = denseRKBenchmarkInventory();
    ASSERT_EQ(inventory.size(), 1U);
    const DenseRKBenchmarkInventoryEntry& entry = inventory.front();
    EXPECT_EQ(entry.implementation, DenseRKBenchmarkImplementation::FamilyProductionHarness);
    EXPECT_FALSE(entry.strategy.has_value());
    EXPECT_FALSE(entry.owns_cuda_kernel);
    EXPECT_FALSE(entry.legacy_reference);
    EXPECT_EQ(entry.source_file, "benchmarks/CubReductionBenchmarkDenseRKFamilyCandidate.cu");
}
