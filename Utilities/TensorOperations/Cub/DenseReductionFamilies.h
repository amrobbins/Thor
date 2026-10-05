#pragma once

#include "Utilities/TensorOperations/Cub/DenseReductionPlanning.h"
#include "Utilities/TensorOperations/Cub/DenseRKInventory.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ThorImplementation::CubReductionInternal {

/**
 * Existing direct CUB/Thor execution capability used by the foundation family adapters.
 *
 * This is deliberately narrower than CubReductionPath: ComposedDense is orchestration rather than a reducer family,
 * OffsetSegmented is not an ordinary dense reduction, and view-specific physical access modes remain outside this
 * migration. Later family patches may replace this transitional direct-plan payload with family-specific plans.
 */
enum class DenseDirectReductionImplementation : uint8_t {
    DeviceTransformReduce = 0,
    ContiguousFixedSegment = 1,
    TiledFixedSegment = 2,
};

/**
 * Family-local physical plan for DENSE-FAMILY-BASE-1.
 *
 * The overall planner sees this only through DenseReductionPhysicalPlan. These fields intentionally mirror only the
 * operation-independent direct dense geometry already selected by CubReduction::analyzeGeometry(); packet width,
 * CTA shape, sharding, and concrete RK kernel ownership remain internal to the existing implementation for now.
 */
struct DenseDirectReductionPhysicalPlan final : DenseReductionPhysicalPlan {
    DenseDirectReductionImplementation implementation = DenseDirectReductionImplementation::DeviceTransformReduce;
    DensePassRole role = DensePassRole::Complete;

    std::vector<uint64_t> input_dimensions;
    uint32_t reduction_axis = 0;

    uint64_t input_elements = 0;
    uint64_t reduction_size = 0;
    uint64_t output_elements = 0;
    uint64_t outer_size = 0;
    uint64_t inner_size = 0;
};

/** Value operation semantics needed by the RK family's calibrated implementation gates. */
enum class DenseRKValueOperation : uint8_t {
    Sum = 0,
    Mean,
    L1,
    L2,
    SumSquares,
    Product,
    Minimum,
    Maximum,
};

enum class DenseRKProgress : uint8_t {
    Complete = 0,
    Staged = 1,
};

/** One modern RK family-local physical pass selected from KParallel or RCooperative. */
struct DenseRKFamilyPhysicalPlan final : DenseReductionPhysicalPlan {
    DenseRKStrategy strategy = DenseRKStrategy::KParallel;
    DenseRKAccess access = DenseRKAccess::Aligned;
    DenseRKProductionImplementation implementation = DenseRKProductionImplementation::KParallelPass;
    DenseRKProgress progress = DenseRKProgress::Complete;
    DensePassRole role = DensePassRole::Complete;

    std::vector<uint64_t> input_dimensions;
    uint32_t reduction_axis = 0;
    DataType input_dtype = DataType::FP32;
    DenseRKValueOperation value_operation = DenseRKValueOperation::Sum;

    uint64_t outer_size = 0;
    uint64_t reduction_size = 0;
    uint64_t inner_size = 0;

    // Family-local launch geometry. Fields unused by a particular strategy remain zero.
    size_t packet_bytes = 0;
    uint32_t block_threads = 0;
    uint32_t cooperative_warps_per_tile = 0;
    uint64_t rows_per_shard = 0;
    uint64_t shards_per_output = 1;
    uint64_t component_tiles = 0;
    uint64_t first_stage_blocks = 0;
    uint64_t useful_stage_warps = 0;

    // Exact/fallback residency facts used by the calibrated family policy. physical_stage_warps counts every launched
    // warp in the physical CTA. useful_stage_warps counts only warps that own useful reduction work. Idle physical
    // warps still consume CTA/warp residency, so useful_resident_warp_capacity is derived through the exact physical
    // block/warp occupancy rather than pretending that useful warps can occupy the device independently.
    uint32_t max_active_blocks_per_sm = 0;
    uint32_t max_active_warps_per_sm = 0;
    uint64_t physical_stage_warps = 0;
    uint64_t resident_device_warps = 0;
    uint64_t useful_resident_warp_capacity = 0;
    // Useful launch supply credited toward the family comfort/progress policy.
    uint64_t launch_warp_supply = 0;
    uint64_t wave_capacity_blocks = 0;
};

/** Device-relative occupancy result for one exact RK physical specialization. */
struct DenseRKOccupancyInfo {
    uint32_t max_active_blocks_per_sm = 0;
    uint32_t max_active_warps_per_sm = 0;
};

/**
 * Optional exact-occupancy hook supplied by CUDA-aware callers. Host-only tests may leave this null and use the
 * deterministic thread/block residency fallback in DenseRKPlanningContext. The query must describe the exact physical
 * specialization encoded by plan and must not perform family selection itself.
 */
using DenseRKOccupancyQuery = DenseRKOccupancyInfo (*)(const DenseRKFamilyPhysicalPlan& plan, void* user_context);

/**
 * Runtime facts needed to choose a modern RK physical pass.
 *
 * original_input_dtype is consulted only while problem.input_state == OriginalInput; every reducer-produced VALUE
 * partial is FP32. Device geometry is explicit so shadow planning remains deterministic and so the same policy scales
 * with the target GPU. occupancy_query may provide exact register/shared-memory-limited residency for each compiled
 * specialization; otherwise max_threads_per_sm/max_blocks_per_sm provide the conservative thread/block fallback.
 */
struct DenseRKPlanningContext {
    DataType original_input_dtype = DataType::FP32;
    DenseRKValueOperation value_operation = DenseRKValueOperation::Sum;
    uint32_t multiprocessors = 1;
    uint32_t warp_size = 32;
    uint32_t max_threads_per_sm = 0;
    uint32_t max_blocks_per_sm = 0;
    DenseRKOccupancyQuery occupancy_query = nullptr;
    void* occupancy_query_context = nullptr;

    // Coverage escape hatch for VALUE dtypes/shapes outside the calibrated modern RK inventory. The fallback is the
    // generic TiledFixedSegment adapter; modern-family census/tests leave this false so coverage gaps stay visible.
    bool allow_generic_rk_fallback = false;
};

/** Whole-domain [R] -> scalar direct reducer family. */
class ReducersDenseR {
   public:
    [[nodiscard]] static DenseReductionCandidate propose(const DenseReductionProblem& problem, DenseReductionSite site);
};

/** Trailing contiguous [K,R] -> [K] direct reducer family. */
class ReducersDenseKR {
   public:
    [[nodiscard]] static DenseReductionCandidate propose(const DenseReductionProblem& problem, DenseReductionSite site);
};

/**
 * Middle-axis [O,R,K] -> [O,K] reducer family.
 *
 * The durable strategy vocabulary is exactly KParallel and RCooperative. A caller may explicitly allow the generic
 * TiledFixedSegment adapter when neither modern strategy can represent the nominated VALUE geometry; that adapter is
 * coverage fallback, not a third RK strategy.
 */
class ReducersDenseRK {
   public:
    /**
     * Proposes one modern VALUE RK pass for the nominated site. K-parallel packet/CTA/shard geometry is evaluated
     * jointly; R-cooperative complete/staged/aligned/rotated choices remain family-local. The selected pass maximizes
     * useful reduction progress among candidates in the family's efficient device-work envelope and never owns a
     * continuation. ARG intentionally remains on the transitional direct adapter until DENSE-ARG-CUTOVER.
     */
    [[nodiscard]] static DenseReductionCandidate propose(const DenseReductionProblem& problem,
                                                         DenseReductionSite site,
                                                         const DenseRKPlanningContext& context);

    /**
     * Returns exactly the production KParallel choice for this RK site, without allowing RCooperative to compete.
     * This is a calibration/introspection hook: it reuses the production KParallel selector byte-for-byte so census
     * benchmarks can tell which geometries are already covered without duplicating selector policy.
     */
    [[nodiscard]] static std::optional<DenseReductionCandidate> proposeProductionKParallelForCalibration(
        const DenseReductionProblem& problem,
        DenseReductionSite site,
        const DenseRKPlanningContext& context);
};

}  // namespace ThorImplementation::CubReductionInternal
