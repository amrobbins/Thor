#include "Utilities/TensorOperations/Cub/DenseRKInventory.h"

#include <array>
#include <stdexcept>

namespace ThorImplementation::CubReductionInternal {
namespace {

constexpr std::array<DenseRKStrategy, 2> STRATEGIES = {
    DenseRKStrategy::KParallel,
    DenseRKStrategy::RCooperative,
};

constexpr std::array<DenseRKProductionInventoryEntry, 8> PRODUCTION_INVENTORY = {{
    {DenseRKProductionImplementation::KParallelPass,
     DenseRKImplementationStatus::FamilyImplementation, DenseRKSemantics::Value,
     DenseRKStrategy::KParallel, DenseRKAccess::Aligned, true, true, false, "launchKParallelTiledStage"},
    {DenseRKProductionImplementation::RCooperativeAlignedComplete,
     DenseRKImplementationStatus::FamilyImplementation, DenseRKSemantics::Value,
     DenseRKStrategy::RCooperative, DenseRKAccess::Aligned, true, false, false,
     "launchPacketAdaptiveCooperativeTiledFixedSegmentReduction"},
    {DenseRKProductionImplementation::RCooperativeAlignedStaged,
     DenseRKImplementationStatus::FamilyImplementation, DenseRKSemantics::Value,
     DenseRKStrategy::RCooperative, DenseRKAccess::Aligned, false, true, false,
     "launchCooperativeShardedTiledFirstStage"},
    {DenseRKProductionImplementation::RCooperativeRotatedStaged,
     DenseRKImplementationStatus::FamilyImplementation, DenseRKSemantics::Value,
     DenseRKStrategy::RCooperative, DenseRKAccess::Rotated, false, true, false,
     "launchAwkwardAlignmentRotatedShardedFirstStage"},
    {DenseRKProductionImplementation::RCooperativeFlatRowsStaged,
     DenseRKImplementationStatus::FamilyImplementation, DenseRKSemantics::Value,
     DenseRKStrategy::RCooperative, DenseRKAccess::FlatRows, false, true, false,
     "launchNarrowLowPrecisionFlatRCooperativeFirstStage"},
    {DenseRKProductionImplementation::AlignedAsyncNarrowTiledArg,
     DenseRKImplementationStatus::FamilyImplementation, DenseRKSemantics::Arg,
     DenseRKStrategy::RCooperative, DenseRKAccess::Aligned, true, false, false,
     "launchAlignedAsyncNarrowTiledArgReduction"},
    {DenseRKProductionImplementation::AlignedCooperativeTiledArg,
     DenseRKImplementationStatus::FamilyImplementation, DenseRKSemantics::Arg,
     DenseRKStrategy::RCooperative, DenseRKAccess::Aligned, true, false, false,
     "launchAlignedCooperativeTiledArgReduction"},
    {DenseRKProductionImplementation::DirectTiledFixedSegmentArg,
     DenseRKImplementationStatus::LegacyOverlap, DenseRKSemantics::Arg,
     std::nullopt, std::nullopt, true, false, false,
     "launchDirectTiledFixedSegmentArgReductionForRowLanes"},
}};

constexpr std::array<DenseRKBenchmarkInventoryEntry, 1> BENCHMARK_INVENTORY = {{
    {DenseRKBenchmarkImplementation::FamilyProductionHarness,
     std::nullopt,
     false,
     false,
     "benchmarks/CubReductionBenchmarkDenseRKFamilyCandidate.cu"},
}};

}  // namespace

std::span<const DenseRKStrategy> denseRKStrategies() {
    return STRATEGIES;
}

std::span<const DenseRKProductionInventoryEntry> denseRKProductionInventory() {
    return PRODUCTION_INVENTORY;
}

const DenseRKProductionInventoryEntry& denseRKProductionInventoryEntry(
    DenseRKProductionImplementation implementation) {
    for (const DenseRKProductionInventoryEntry& entry : PRODUCTION_INVENTORY) {
        if (entry.implementation == implementation) {
            return entry;
        }
    }
    throw std::invalid_argument("Unknown dense RK production implementation.");
}

bool denseRKIsFamilyImplementation(DenseRKProductionImplementation implementation) {
    return denseRKProductionInventoryEntry(implementation).status
           == DenseRKImplementationStatus::FamilyImplementation;
}

std::span<const DenseRKBenchmarkInventoryEntry> denseRKBenchmarkInventory() {
    return BENCHMARK_INVENTORY;
}

}  // namespace ThorImplementation::CubReductionInternal
