#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"

#include "Utilities/TensorOperations/Cub/CubDataTypePolicy.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal {
namespace {

constexpr uint64_t K_PARALLEL_MIN_ROWS_PER_SHARD = 8;
constexpr uint64_t K_PARALLEL_DEEP_CROSSOVER_ROWS = 96;
// Realistic end-to-end calibration shows that expensive staged reductions are best served by making the deepest
// first-pass progress that still leaves enough independent K-parallel work to keep the device busy. Six one-warp
// CTAs per SM is the calibrated supply floor for that decision; unlike the old 16-row rule, this naturally scales
// rows-per-shard with O, K tiling, dtype/alignment, and R.
constexpr uint32_t K_PARALLEL_STAGED_PROGRESS_WARPS_PER_SM = 6;
constexpr uint32_t K_PARALLEL_MAX_WARPS = 8;
constexpr uint32_t K_PARALLEL_MAX_THREADS = 256;
constexpr std::array<uint64_t, 9> FP8_K_PARALLEL_ROW_DEPTHS = {4096, 2048, 1024, 512, 256, 128, 64, 32, 16};
constexpr uint32_t R_COOPERATIVE_BLOCK_WARPS = 8;
constexpr uint32_t R_COOPERATIVE_BLOCK_THREADS = R_COOPERATIVE_BLOCK_WARPS * 32;
constexpr uint32_t R_COOPERATIVE_MAX_COMPLETE_WARPS = 32;
constexpr uint64_t R_COOPERATIVE_COMPLETE_TARGET_WARPS =
    CubReductionTiledPolicy::TARGET_ACTIVE_WARPS * CubReductionTiledPolicy::DENSE_RK_TARGET_SM_WAVES;

// Calibrated from the modern single-pass occupancy sweep. The established FP16/BF16/FP32 staged policy keeps its
// one-warp CTA and 16 -> 8 -> 4 transaction preference unchanged. FP8 is calibrated separately below: 16/8/4/2-byte
// packet reducers may use 32/64/128/256-thread CTAs, and useful-warp saturation determines whether packet efficiency
// or additional K-side parallelism is the scarce resource.
constexpr uint32_t K_PARALLEL_COMPLETE_MIN_WARPS_PER_SM = 8;
constexpr uint64_t K_PARALLEL_COMPLETE_MAX_REDUCTION_ROWS = 2048;
constexpr uint32_t ALIGNED_R_COOPERATIVE_STAGED_COMFORT_WARPS_PER_SM = 24;
constexpr uint32_t ROTATED_LOW_PRECISION_STAGED_COMFORT_WARPS_PER_SM = 12;
constexpr uint32_t NARROW_FLAT_STAGED_COMFORT_WARPS_PER_SM = 12;
constexpr uint32_t ROTATED_FP32_STAGED_COMFORT_WARPS_PER_SM = 24;
constexpr uint32_t R_COOPERATIVE_COMPLETE_MIN_WARPS_PER_SM = 12;
constexpr uint64_t R_COOPERATIVE_COMPLETE_MAX_ROWS_PER_COOPERATIVE_WARP = 256;

// End-to-end family-boundary calibration shows a distinct FP32 regime where KParallel is forced down to one
// FP32 value per lane (4-byte packets) and the input stream is large enough for rotated RCooperative to amortize
// its cooperative reduction machinery. The measured crossover lies cleanly between the ~512 MiB cases, where
// KParallel remains materially faster, and the ~819 MiB+ cases, where rotated RCooperative saves tens of
// microseconds. Keep this as an explicit physical crossover rather than keying policy on particular K or R values.
constexpr uint64_t FP32_SCALAR_PACKET_RCOOPERATIVE_MIN_INPUT_BYTES = 768ULL * 1024ULL * 1024ULL;

[[nodiscard]] uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* message) {
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        throw std::overflow_error(message);
    }
    return lhs * rhs;
}

[[nodiscard]] uint64_t ceilDivide(uint64_t numerator, uint64_t denominator) {
    if (denominator == 0) {
        throw std::logic_error("Dense RK planner denominator must be non-zero.");
    }
    return numerator / denominator + static_cast<uint64_t>(numerator % denominator != 0);
}

[[nodiscard]] uint64_t dtypeBytes(DataType dtype) {
    const size_t bytes = TensorDescriptor::getElementSizeInBytes(dtype);
    if (bytes == 0 || bytes > std::numeric_limits<uint64_t>::max()) {
        throw std::invalid_argument("Dense RK planner received an invalid storage dtype.");
    }
    return static_cast<uint64_t>(bytes);
}

[[nodiscard]] bool kParallelStorageDTypeSupported(DataType dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
    if (dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2) {
        return true;
    }
#endif
    return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

[[nodiscard]] bool rCooperativeStorageDTypeSupported(DataType dtype) {
    // FP8 FlatRows/Rotated/Complete adaptations are intentionally benchmark-only until the dedicated single-pass
    // calibration establishes their line-rate regions. Keeping this gate unchanged guarantees the experiment cannot
    // silently alter production selection while physical kernel capability is being characterized.
    return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

[[nodiscard]] bool isFp8StorageDType(DataType dtype) {
#if THOR_CUB_ENABLE_FP8_TYPES
    return dtype == DataType::FP8_E4M3 || dtype == DataType::FP8_E5M2;
#else
    static_cast<void>(dtype);
    return false;
#endif
}

[[nodiscard]] bool additiveValueOperation(DenseRKValueOperation op) {
    switch (op) {
        case DenseRKValueOperation::Sum:
        case DenseRKValueOperation::Mean:
        case DenseRKValueOperation::L1:
        case DenseRKValueOperation::L2:
        case DenseRKValueOperation::SumSquares:
            return true;
        case DenseRKValueOperation::Product:
        case DenseRKValueOperation::Minimum:
        case DenseRKValueOperation::Maximum:
            return false;
    }
    return false;
}

[[nodiscard]] const char* familyName(DenseReducerFamily family) {
    switch (family) {
        case DenseReducerFamily::R:
            return "R";
        case DenseReducerFamily::KR:
            return "KR";
        case DenseReducerFamily::RK:
            return "RK";
    }
    return "unknown";
}

[[nodiscard]] std::vector<uint64_t> currentRunDimensions(const DenseReductionProblem& problem) {
    std::vector<uint64_t> dimensions;
    dimensions.reserve(problem.runs.size());
    for (const DenseReductionRun& run : problem.runs) {
        if (run.extent == 0) {
            throw std::invalid_argument("Dense family proposal received a zero-extent current run.");
        }
        dimensions.push_back(run.extent);
    }
    if (dimensions.empty()) {
        throw std::invalid_argument("Dense family proposal requires a current reduction site.");
    }
    return dimensions;
}

[[nodiscard]] DenseDirectReductionImplementation directImplementationForFamily(DenseReducerFamily family) {
    switch (family) {
        case DenseReducerFamily::R:
            return DenseDirectReductionImplementation::DeviceTransformReduce;
        case DenseReducerFamily::KR:
            return DenseDirectReductionImplementation::ContiguousFixedSegment;
        case DenseReducerFamily::RK:
            return DenseDirectReductionImplementation::TiledFixedSegment;
    }
    throw std::logic_error("Unknown dense reducer family.");
}

[[nodiscard]] CubReductionPath cubPathForImplementation(DenseDirectReductionImplementation implementation) {
    switch (implementation) {
        case DenseDirectReductionImplementation::DeviceTransformReduce:
            return CubReductionPath::DeviceTransformReduce;
        case DenseDirectReductionImplementation::ContiguousFixedSegment:
            return CubReductionPath::ContiguousFixedSegment;
        case DenseDirectReductionImplementation::TiledFixedSegment:
            return CubReductionPath::TiledFixedSegment;
    }
    throw std::logic_error("Unknown direct dense reduction implementation.");
}

[[nodiscard]] DenseReductionCandidate proposeDirect(const DenseReductionProblem& problem,
                                                     DenseReductionSite site,
                                                     DenseReducerFamily requested_family) {
    const DenseReducerFamily classified_family = classifyDenseReducerFamily(problem, site);
    if (classified_family != requested_family) {
        throw std::invalid_argument(std::string("Dense ") + familyName(requested_family)
                                    + " family cannot propose for a site classified as "
                                    + familyName(classified_family) + ".");
    }

    const std::vector<uint64_t> input_dimensions = currentRunDimensions(problem);
    if (site.run_index >= input_dimensions.size()) {
        throw std::invalid_argument("Dense family proposal site is outside the current run tensor.");
    }

    const std::vector<uint32_t> axes{site.run_index};
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input_dimensions, axes);
    const DenseDirectReductionImplementation implementation = directImplementationForFamily(requested_family);
    const CubReductionPath expected_path = cubPathForImplementation(implementation);
    if (geometry.path != expected_path) {
        throw std::logic_error(std::string("Dense ") + familyName(requested_family)
                               + " direct adapter did not resolve to its existing production direct path.");
    }
    if (geometry.axes.size() != 1 || geometry.axes.front() != site.run_index) {
        throw std::logic_error("Dense direct adapter geometry did not preserve the nominated site.");
    }

    DenseReductionProblem next_problem = advanceDenseReductionProblem(problem, site, 1);
    const DensePassRole role = deriveDensePassRole(problem, next_problem);

    auto physical_plan = std::make_shared<DenseDirectReductionPhysicalPlan>();
    physical_plan->implementation = implementation;
    physical_plan->role = role;
    physical_plan->input_dimensions = input_dimensions;
    physical_plan->reduction_axis = site.run_index;
    physical_plan->input_elements = geometry.input_elements;
    physical_plan->reduction_size = geometry.reduction_size;
    physical_plan->output_elements = geometry.output_elements;
    physical_plan->outer_size = geometry.outer_size;
    physical_plan->inner_size = geometry.inner_size;

    DenseReductionCandidate candidate;
    candidate.family = requested_family;
    candidate.site = site;
    candidate.physical_plan = std::move(physical_plan);
    candidate.next_problem = std::move(next_problem);
    candidate.next_bytes = denseReductionProblemBytes(candidate.next_problem);
    candidate.scratch_bytes = 0;
    return candidate;
}

struct DenseRKOption {
    std::shared_ptr<DenseRKFamilyPhysicalPlan> plan;
    DenseReductionProblem next_problem;
    uint64_t next_bytes = 0;
    uint64_t scratch_bytes = 0;
    bool calibrated_efficient = false;
    uint64_t comfort_supply = 0;
    uint64_t comfort_target = 1;
};

[[nodiscard]] DataType currentRKInputDType(const DenseReductionProblem& problem,
                                           const DenseRKPlanningContext& context) {
    return problem.input_state == DenseReductionInputState::OriginalInput ? context.original_input_dtype
                                                                          : DataType::FP32;
}

[[nodiscard]] DenseRKOption makeRKOption(const DenseReductionProblem& problem,
                                         DenseReductionSite site,
                                         std::shared_ptr<DenseRKFamilyPhysicalPlan> plan,
                                         uint64_t next_reduction_extent,
                                         bool calibrated_efficient) {
    DenseRKOption option;
    option.next_problem = advanceDenseReductionProblem(problem, site, next_reduction_extent);
    plan->role = deriveDensePassRole(problem, option.next_problem);
    plan->progress = next_reduction_extent == 1 ? DenseRKProgress::Complete : DenseRKProgress::Staged;
    option.plan = std::move(plan);
    option.next_bytes = denseReductionProblemBytes(option.next_problem);
    option.calibrated_efficient = calibrated_efficient;

    if (next_reduction_extent != 1) {
        option.scratch_bytes = checkedMultiply(
            checkedMultiply(option.plan->outer_size,
                            next_reduction_extent,
                            "Dense RK staged scratch outer/shard count overflows uint64_t."),
            option.plan->inner_size,
            "Dense RK staged scratch element count overflows uint64_t.");
        option.scratch_bytes = checkedMultiply(
            option.scratch_bytes, sizeof(float), "Dense RK staged scratch byte count overflows uint64_t.");
    }
    return option;
}

[[nodiscard]] DenseRKOccupancyInfo queryRKOccupancy(const DenseRKPlanningContext& context,
                                                      const DenseRKFamilyPhysicalPlan& plan) {
    if (plan.block_threads == 0 || context.warp_size == 0 || plan.block_threads % context.warp_size != 0) {
        throw std::logic_error("Dense RK occupancy policy received invalid CTA/warp geometry.");
    }
    if (context.occupancy_query != nullptr) {
        const DenseRKOccupancyInfo info = context.occupancy_query(plan, context.occupancy_query_context);
        if (info.max_active_blocks_per_sm == 0 || info.max_active_warps_per_sm == 0) {
            throw std::logic_error("Dense RK exact occupancy query returned zero residency.");
        }
        return info;
    }
    if (context.max_threads_per_sm == 0 || context.max_blocks_per_sm == 0) {
        throw std::invalid_argument("Dense RK fallback occupancy requires non-zero device thread/block limits.");
    }
    const uint32_t thread_limited_blocks = context.max_threads_per_sm / plan.block_threads;
    const uint32_t active_blocks = std::min(context.max_blocks_per_sm, thread_limited_blocks);
    if (active_blocks == 0) {
        throw std::logic_error("Dense RK fallback occupancy found no resident CTA for a legal specialization.");
    }
    const uint32_t warps_per_cta = plan.block_threads / context.warp_size;
    return DenseRKOccupancyInfo{active_blocks, active_blocks * warps_per_cta};
}

void annotateRKOccupancy(DenseRKFamilyPhysicalPlan& plan, const DenseRKPlanningContext& context) {
    const DenseRKOccupancyInfo occupancy = queryRKOccupancy(context, plan);
    plan.max_active_blocks_per_sm = occupancy.max_active_blocks_per_sm;
    plan.max_active_warps_per_sm = occupancy.max_active_warps_per_sm;

    if (plan.first_stage_blocks == 0 || plan.useful_stage_warps == 0) {
        throw std::logic_error("Dense RK occupancy policy requires non-zero physical blocks and useful warps.");
    }

    const uint64_t warps_per_cta = plan.block_threads / context.warp_size;
    plan.physical_stage_warps = checkedMultiply(
        plan.first_stage_blocks, warps_per_cta, "Dense RK physical launch warp count overflows uint64_t.");
    if (plan.useful_stage_warps > plan.physical_stage_warps) {
        throw std::logic_error("Dense RK useful launch warps exceed physical launch warps.");
    }

    plan.resident_device_warps = checkedMultiply(
        static_cast<uint64_t>(context.multiprocessors),
        occupancy.max_active_warps_per_sm,
        "Dense RK resident device warp capacity overflows uint64_t.");
    plan.wave_capacity_blocks = checkedMultiply(
        static_cast<uint64_t>(context.multiprocessors),
        occupancy.max_active_blocks_per_sm,
        "Dense RK resident device CTA capacity overflows uint64_t.");

    // Useful work cannot be resident independently of its CTA. Scale the useful-warps-per-launched-block ratio by the
    // exact physical resident block capacity, then cap by the exact physical resident warp capacity. This discounts
    // idle warps for progress while still charging them for the occupancy/resources they consume. Use 128-bit
    // arithmetic because the diagnostic/planning counts are uint64_t and the intermediate product need not fit.
    const unsigned __int128 useful_capacity_numerator =
        static_cast<unsigned __int128>(plan.wave_capacity_blocks)
        * static_cast<unsigned __int128>(plan.useful_stage_warps);
    const unsigned __int128 useful_capacity_by_blocks_wide =
        useful_capacity_numerator / static_cast<unsigned __int128>(plan.first_stage_blocks);
    const uint64_t useful_capacity_by_blocks =
        useful_capacity_by_blocks_wide > static_cast<unsigned __int128>(std::numeric_limits<uint64_t>::max())
            ? std::numeric_limits<uint64_t>::max()
            : static_cast<uint64_t>(useful_capacity_by_blocks_wide);
    plan.useful_resident_warp_capacity =
        std::min(plan.resident_device_warps, useful_capacity_by_blocks);
    plan.launch_warp_supply =
        std::min(plan.useful_stage_warps, plan.useful_resident_warp_capacity);
}

[[nodiscard]] uint64_t targetWarpSupply(const DenseRKFamilyPhysicalPlan& plan,
                                        const DenseRKPlanningContext& context,
                                        uint32_t comfort_warps_per_sm) {
    const uint32_t target_per_sm = std::min(comfort_warps_per_sm, plan.max_active_warps_per_sm);
    return checkedMultiply(static_cast<uint64_t>(context.multiprocessors),
                           target_per_sm,
                           "Dense RK comfort warp target overflows uint64_t.");
}

[[nodiscard]] bool meetsWarpComfort(const DenseRKFamilyPhysicalPlan& plan,
                                    const DenseRKPlanningContext& context,
                                    uint32_t comfort_warps_per_sm) {
    return plan.launch_warp_supply >= targetWarpSupply(plan, context, comfort_warps_per_sm);
}

[[nodiscard]] bool sameKParallelLaunch(const DenseRKOption& lhs, const DenseRKOption& rhs) {
    return lhs.plan->strategy == DenseRKStrategy::KParallel && rhs.plan->strategy == DenseRKStrategy::KParallel
           && lhs.plan->packet_bytes == rhs.plan->packet_bytes
           && lhs.plan->block_threads == rhs.plan->block_threads
           && lhs.plan->shards_per_output == rhs.plan->shards_per_output;
}

[[nodiscard]] std::pair<uint64_t, uint64_t> kParallelLaneUtilization(const DenseRKOption& option) {
    if (option.plan->strategy != DenseRKStrategy::KParallel || option.plan->packet_bytes == 0
        || option.plan->block_threads == 0 || option.plan->component_tiles == 0) {
        return {0, 1};
    }
    const uint64_t element_bytes = dtypeBytes(option.plan->input_dtype);
    const uint64_t items_per_lane = option.plan->packet_bytes / element_bytes;
    const uint64_t packet_owners = ceilDivide(option.plan->inner_size, items_per_lane);
    const uint64_t lane_slots = checkedMultiply(
        option.plan->component_tiles,
        option.plan->block_threads,
        "Dense K-parallel packet lane-slot count overflows uint64_t.");
    return {packet_owners, lane_slots};
}

[[nodiscard]] bool betterProgressOption(const DenseRKOption& candidate, const DenseRKOption& incumbent) {
    if (candidate.calibrated_efficient != incumbent.calibrated_efficient) {
        return candidate.calibrated_efficient;
    }
    if (!candidate.calibrated_efficient && candidate.comfort_target != 0 && incumbent.comfort_target != 0) {
        const long double candidate_fraction =
            static_cast<long double>(candidate.comfort_supply) / candidate.comfort_target;
        const long double incumbent_fraction =
            static_cast<long double>(incumbent.comfort_supply) / incumbent.comfort_target;
        if (candidate_fraction != incumbent_fraction) {
            return candidate_fraction > incumbent_fraction;
        }
    }
    if (candidate.next_bytes != incumbent.next_bytes) {
        return candidate.next_bytes < incumbent.next_bytes;
    }
    if (candidate.scratch_bytes != incumbent.scratch_bytes) {
        return candidate.scratch_bytes < incumbent.scratch_bytes;
    }
    if (candidate.plan->progress != incumbent.plan->progress) {
        return candidate.plan->progress == DenseRKProgress::Complete;
    }
    if (candidate.plan->strategy != incumbent.plan->strategy) {
        // R-cooperative is the established modern baseline; K-parallel must earn selection through strictly greater
        // useful progress rather than becoming the blanket tie winner.
        return candidate.plan->strategy == DenseRKStrategy::RCooperative;
    }
    if (candidate.plan->strategy == DenseRKStrategy::KParallel) {
        const auto [candidate_owners, candidate_slots] = kParallelLaneUtilization(candidate);
        const auto [incumbent_owners, incumbent_slots] = kParallelLaneUtilization(incumbent);
        const long double candidate_fill = static_cast<long double>(candidate_owners) / candidate_slots;
        const long double incumbent_fill = static_cast<long double>(incumbent_owners) / incumbent_slots;
        if (candidate_fill != incumbent_fill) {
            return candidate_fill > incumbent_fill;
        }
    }
    if (candidate.plan->packet_bytes != incumbent.plan->packet_bytes) {
        return candidate.plan->packet_bytes > incumbent.plan->packet_bytes;
    }
    if (candidate.plan->rows_per_shard != incumbent.plan->rows_per_shard) {
        return candidate.plan->rows_per_shard > incumbent.plan->rows_per_shard;
    }
    if (candidate.plan->block_threads != incumbent.plan->block_threads) {
        return candidate.plan->block_threads < incumbent.plan->block_threads;
    }
    return static_cast<uint8_t>(candidate.plan->implementation)
           < static_cast<uint8_t>(incumbent.plan->implementation);
}

[[nodiscard]] std::optional<DenseRKOption> chooseProgressWithinOccupancyEnvelope(
    std::vector<DenseRKOption> options,
    bool calibrated_only,
    const DenseRKPlanningContext& context,
    uint32_t comfort_warps_per_sm) {
    std::vector<DenseRKOption*> eligible;
    eligible.reserve(options.size());
    for (DenseRKOption& option : options) {
        if (!calibrated_only || option.calibrated_efficient) {
            eligible.push_back(&option);
        }
    }
    if (eligible.empty()) {
        return std::nullopt;
    }

    // First prefer candidates that reach the specialization-relative comfort target. Among those, maximize reduction
    // progress (smallest next aggregate) exactly as the architecture requires. If the problem is too small for any
    // candidate to reach the target, choose the candidate with the greatest fraction of its target supplied, then use
    // the same progress ordering. This preserves sensible small-problem behavior without inventing a separate latency
    // model.
    std::optional<DenseRKOption> best_comfortable;
    for (DenseRKOption* option : eligible) {
        if (!meetsWarpComfort(*option->plan, context, comfort_warps_per_sm)) {
            continue;
        }
        DenseRKOption comfortable = *option;
        comfortable.comfort_supply = comfortable.plan->launch_warp_supply;
        comfortable.comfort_target = targetWarpSupply(*comfortable.plan, context, comfort_warps_per_sm);
        if (!best_comfortable.has_value() || betterProgressOption(comfortable, *best_comfortable)) {
            best_comfortable = std::move(comfortable);
        }
    }
    if (best_comfortable.has_value()) {
        return best_comfortable;
    }

    DenseRKOption* best_supply = eligible.front();
    uint64_t best_target = targetWarpSupply(*best_supply->plan, context, comfort_warps_per_sm);
    for (DenseRKOption* option : eligible) {
        const uint64_t option_target = targetWarpSupply(*option->plan, context, comfort_warps_per_sm);
        const long double option_fraction = static_cast<long double>(option->plan->launch_warp_supply) / option_target;
        const long double best_fraction =
            static_cast<long double>(best_supply->plan->launch_warp_supply) / best_target;
        if (option_fraction > best_fraction
            || (option_fraction == best_fraction && betterProgressOption(*option, *best_supply))) {
            best_supply = option;
            best_target = option_target;
        }
    }
    DenseRKOption underfilled = *best_supply;
    underfilled.calibrated_efficient = false;
    underfilled.comfort_supply = underfilled.plan->launch_warp_supply;
    underfilled.comfort_target = targetWarpSupply(*underfilled.plan, context, comfort_warps_per_sm);
    return underfilled;
}

[[nodiscard]] bool kParallelHasLegalPacket(uint64_t inner_size, uint64_t element_bytes) {
    constexpr std::array<size_t, 4> PACKET_BYTES = {16, 8, 4, 2};
    for (size_t packet_bytes : PACKET_BYTES) {
        // Two-byte packets are an FP8-only specialization. Keep the established FP16/BF16/FP32 packet surface
        // byte-for-byte unchanged while giving one-byte storage one more naturally coalesced geometry.
        if (packet_bytes == 2 && element_bytes != 1) {
            continue;
        }
        if (packet_bytes < element_bytes || packet_bytes % element_bytes != 0) {
            continue;
        }
        if (inner_size % (packet_bytes / element_bytes) == 0) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] uint32_t kParallelBlockThreads(uint64_t inner_size,
                                             uint64_t input_element_bytes,
                                             size_t packet_bytes) {
    const uint64_t items_per_lane = packet_bytes / input_element_bytes;
    const uint64_t packets = ceilDivide(inner_size, items_per_lane);
    const uint64_t required_warps = ceilDivide(packets, 32);
    uint32_t warps = 1;
    while (warps < K_PARALLEL_MAX_WARPS && static_cast<uint64_t>(warps) < required_warps) {
        warps <<= 1;
    }
    return warps * 32;
}

[[nodiscard]] uint32_t fp8KParallelSaturationWarpsPerSm(const DenseRKPlanningContext& context,
                                                               size_t packet_bytes) {
    if (context.warp_size == 0 || context.max_threads_per_sm == 0 || context.max_threads_per_sm < context.warp_size) {
        throw std::invalid_argument("FP8 KParallel saturation policy requires valid device warp/thread limits.");
    }
    if (packet_bytes != 16 && packet_bytes != 8 && packet_bytes != 4 && packet_bytes != 2) {
        throw std::invalid_argument("FP8 KParallel saturation policy received unsupported packet width.");
    }
    const uint32_t hardware_warps_per_sm = context.max_threads_per_sm / context.warp_size;
    const uint32_t base = static_cast<uint32_t>(ceilDivide(hardware_warps_per_sm, 4));
    const uint32_t packet_multiplier = packet_bytes >= 8 ? 1U : static_cast<uint32_t>(8 / packet_bytes);
    const uint64_t target = checkedMultiply(base,
                                            packet_multiplier,
                                            "FP8 KParallel per-SM saturation target overflows uint64_t.");
    return static_cast<uint32_t>(std::min<uint64_t>(target, std::numeric_limits<uint32_t>::max()));
}

[[nodiscard]] uint64_t fp8KParallelSaturationTarget(const DenseRKPlanningContext& context,
                                                     size_t packet_bytes) {
    return checkedMultiply(static_cast<uint64_t>(context.multiprocessors),
                           fp8KParallelSaturationWarpsPerSm(context, packet_bytes),
                           "FP8 KParallel device saturation target overflows uint64_t.");
}

// Calibration of the saturated FP8 staged kernels shows that the 2-byte path needs at least four physical warps/CTA
// before it can reach the ~1.6 TB/s regime. Keep smaller CTAs available for latency/under-supplied reductions, but do
// not allow them to claim saturation merely because their useful-warp count reaches the device target.
[[nodiscard]] bool fp8KParallelLayoutCanClaimSaturation(const DenseRKFamilyPhysicalPlan& plan) {
    return plan.packet_bytes != 2 || plan.block_threads >= 128;
}

[[nodiscard]] bool fp8KParallelOptionIsSaturated(const DenseRKOption& option,
                                                  const DenseRKPlanningContext& context) {
    return fp8KParallelLayoutCanClaimSaturation(*option.plan)
           && option.plan->launch_warp_supply >= fp8KParallelSaturationTarget(context, option.plan->packet_bytes);
}

void addFp8KParallelShardCandidate(std::vector<uint64_t>& shards,
                                   uint64_t reduction_size,
                                   uint64_t candidate) {
    if (candidate <= 1 || candidate > reduction_size) {
        return;
    }
    if (reduction_size / candidate < K_PARALLEL_MIN_ROWS_PER_SHARD) {
        return;
    }
    if (std::find(shards.begin(), shards.end(), candidate) == shards.end()) {
        shards.push_back(candidate);
    }
}

void addFp8KParallelShardNeighborhood(std::vector<uint64_t>& shards,
                                      uint64_t reduction_size,
                                      uint64_t candidate) {
    if (candidate > 2) {
        addFp8KParallelShardCandidate(shards, reduction_size, candidate - 1);
    }
    addFp8KParallelShardCandidate(shards, reduction_size, candidate);
    if (candidate < std::numeric_limits<uint64_t>::max()) {
        addFp8KParallelShardCandidate(shards, reduction_size, candidate + 1);
    }
}

[[nodiscard]] uint64_t fp8KParallelRowDepthDistance(uint64_t rows_per_shard) {
    uint64_t best = std::numeric_limits<uint64_t>::max();
    for (uint64_t calibrated_rows : FP8_K_PARALLEL_ROW_DEPTHS) {
        const uint64_t distance = rows_per_shard > calibrated_rows ? rows_per_shard - calibrated_rows
                                                                    : calibrated_rows - rows_per_shard;
        best = std::min(best, distance);
    }
    return best;
}

[[nodiscard]] uint64_t fp8KParallelShardSlack(const DenseRKFamilyPhysicalPlan& plan) {
    const uint64_t covered_rows = checkedMultiply(
        plan.rows_per_shard, plan.shards_per_output, "FP8 KParallel shard coverage overflows uint64_t.");
    if (covered_rows < plan.reduction_size) {
        throw std::logic_error("FP8 KParallel shard coverage does not span the reduction.");
    }
    return covered_rows - plan.reduction_size;
}

[[nodiscard]] uint32_t fp8KParallelPreferredWidePacketSaturatedBlockThreads(
    const DenseRKFamilyPhysicalPlan& plan, uint32_t warp_size) {
    if (plan.packet_bytes != 16 && plan.packet_bytes != 8) {
        throw std::logic_error("FP8 wide-packet CTA preference requires an 8- or 16-byte packet.");
    }
    const uint64_t element_bytes = dtypeBytes(plan.input_dtype);
    const uint64_t items_per_lane = plan.packet_bytes / element_bytes;
    if (warp_size == 0) {
        throw std::invalid_argument("FP8 wide-packet CTA preference requires a non-zero warp size.");
    }
    const uint64_t packet_owners = ceilDivide(plan.inner_size, items_per_lane);
    const uint64_t owner_warps = ceilDivide(packet_owners, warp_size);

    // The saturated staged-layout sweep separates two concerns that the old resident-capacity ranking conflated.
    // Useful launch supply decides whether the packet has reached the streaming regime. Once it has, packet16/8
    // perform best when the physical CTA roughly matches the K-side owner geometry: at least two warps so tiny CTAs
    // do not pay excessive scheduling/epilogue overhead, but no more than four warps because extra non-cooperative
    // warps do not create reduction cooperation. This is an intrinsic packet-owner geometry rule, not an O/R/K
    // lookup table.
    uint32_t preferred_warps = 2;
    while (preferred_warps < 4 && static_cast<uint64_t>(preferred_warps) < owner_warps) {
        preferred_warps <<= 1;
    }
    return preferred_warps * warp_size;
}

[[nodiscard]] uint32_t fp8KParallelBlockThreadDistance(uint32_t lhs, uint32_t rhs) {
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

[[nodiscard]] bool betterFp8SaturatedLayout(const DenseRKOption& candidate,
                                             const DenseRKOption& incumbent,
                                             const DenseRKPlanningContext& context) {
    if (candidate.plan->packet_bytes != incumbent.plan->packet_bytes) {
        throw std::logic_error("FP8 saturated layout comparison requires one packet width.");
    }

    if (candidate.plan->packet_bytes >= 8) {
        // Once packet16/8 already expose the calibrated useful-warp supply, additional useful resident capacity is no
        // longer a reliable speed predictor. Prefer the CTA width implied by the independent K packet-owner geometry
        // instead. The sweep shows this avoids both the CTA32-over-CTA64 miss at K=512 and the CTA64-over-CTA128 miss
        // around K=2048 while leaving packet-width selection and the saturation frontier unchanged.
        const uint32_t preferred_threads =
            fp8KParallelPreferredWidePacketSaturatedBlockThreads(*candidate.plan, context.warp_size);
        const uint32_t candidate_distance =
            fp8KParallelBlockThreadDistance(candidate.plan->block_threads, preferred_threads);
        const uint32_t incumbent_distance =
            fp8KParallelBlockThreadDistance(incumbent.plan->block_threads, preferred_threads);
        if (candidate_distance != incumbent_distance) {
            return candidate_distance < incumbent_distance;
        }
    } else if (candidate.plan->useful_resident_warp_capacity != incumbent.plan->useful_resident_warp_capacity) {
        // Packet4/2 need materially more useful warps/SM to reach their measured frontiers. Preserve the established
        // residency discriminator for those narrow-packet regimes rather than perturbing policy that this calibration
        // did not target.
        return candidate.plan->useful_resident_warp_capacity > incumbent.plan->useful_resident_warp_capacity;
    }

    // The staged kernels are calibrated on the existing rows/shard ladder. Around the saturation boundary, prefer
    // the layout whose actual ceil-divided shard depth stays closest to that ladder rather than blindly taking one
    // fewer shard. This makes exact 512/1024/... partitions survive the quantization boundary when they are available.
    const uint64_t candidate_depth_distance = fp8KParallelRowDepthDistance(candidate.plan->rows_per_shard);
    const uint64_t incumbent_depth_distance = fp8KParallelRowDepthDistance(incumbent.plan->rows_per_shard);
    if (candidate_depth_distance != incumbent_depth_distance) {
        return candidate_depth_distance < incumbent_depth_distance;
    }

    // For equally calibrated depths, minimize inactive tail rows across the shard grid. This handles awkward R
    // without requiring shape-specific lookup tables.
    const uint64_t candidate_slack = fp8KParallelShardSlack(*candidate.plan);
    const uint64_t incumbent_slack = fp8KParallelShardSlack(*incumbent.plan);
    if (candidate_slack != incumbent_slack) {
        return candidate_slack < incumbent_slack;
    }

    const auto [candidate_owners, candidate_slots] = kParallelLaneUtilization(candidate);
    const auto [incumbent_owners, incumbent_slots] = kParallelLaneUtilization(incumbent);
    const long double candidate_fill = static_cast<long double>(candidate_owners) / candidate_slots;
    const long double incumbent_fill = static_cast<long double>(incumbent_owners) / incumbent_slots;
    if (candidate_fill != incumbent_fill) {
        return candidate_fill > incumbent_fill;
    }

    // Only after CTA geometry and partition quality are equivalent should continuation size decide the tie.
    if (candidate.next_bytes != incumbent.next_bytes) {
        return candidate.next_bytes < incumbent.next_bytes;
    }
    if (candidate.scratch_bytes != incumbent.scratch_bytes) {
        return candidate.scratch_bytes < incumbent.scratch_bytes;
    }
    if (candidate.plan->block_threads != incumbent.plan->block_threads) {
        return candidate.plan->block_threads < incumbent.plan->block_threads;
    }
    return static_cast<uint8_t>(candidate.plan->implementation)
           < static_cast<uint8_t>(incumbent.plan->implementation);
}

[[nodiscard]] DenseRKOption chooseFp8KParallelLayoutForPacket(std::vector<DenseRKOption> options,
                                                               const DenseRKPlanningContext& context) {
    if (options.empty()) {
        throw std::logic_error("FP8 KParallel packet policy received no layouts.");
    }

    uint64_t minimum_saturated_shards = std::numeric_limits<uint64_t>::max();
    for (DenseRKOption& option : options) {
        const uint64_t target = fp8KParallelSaturationTarget(context, option.plan->packet_bytes);
        option.comfort_supply = option.plan->launch_warp_supply;
        option.comfort_target = target;
        option.calibrated_efficient = fp8KParallelOptionIsSaturated(option, context);
        if (option.calibrated_efficient) {
            minimum_saturated_shards = std::min(minimum_saturated_shards, option.plan->shards_per_output);
        }
    }

    if (minimum_saturated_shards != std::numeric_limits<uint64_t>::max()) {
        // Preserve the architecture's deepest-progress preference, but leave one shard of quantization freedom around
        // the first saturated layout. The measured FP8 sweep showed that this immediate neighbor is often the clean
        // 512/1024-row partition while the minimally saturated count can be one shard short of it.
        const uint64_t saturated_shard_limit = minimum_saturated_shards < std::numeric_limits<uint64_t>::max()
                                                   ? minimum_saturated_shards + 1
                                                   : minimum_saturated_shards;
        std::optional<DenseRKOption> saturated;
        for (const DenseRKOption& option : options) {
            if (!option.calibrated_efficient || option.plan->shards_per_output > saturated_shard_limit) {
                continue;
            }
            if (!saturated.has_value() || betterFp8SaturatedLayout(option, *saturated, context)) {
                saturated = option;
            }
        }
        if (!saturated.has_value()) {
            throw std::logic_error("FP8 KParallel saturated layout envelope unexpectedly became empty.");
        }
        return *saturated;
    }

    DenseRKOption* best = &options.front();
    for (DenseRKOption& option : options) {
        if (option.plan->launch_warp_supply > best->plan->launch_warp_supply
            || (option.plan->launch_warp_supply == best->plan->launch_warp_supply
                && betterProgressOption(option, *best))) {
            best = &option;
        }
    }
    best->calibrated_efficient = false;
    best->comfort_supply = best->plan->launch_warp_supply;
    best->comfort_target = fp8KParallelSaturationTarget(context, best->plan->packet_bytes);
    return *best;
}

[[nodiscard]] DenseRKOption chooseFp8KParallelPacket(std::vector<DenseRKOption> packet_choices,
                                                       const DenseRKPlanningContext& context) {
    if (packet_choices.empty()) {
        throw std::logic_error("FP8 KParallel packet policy received no packet candidates.");
    }

    // Once line-rate useful supply is available, transaction efficiency is the stable discriminator: choose the
    // widest saturated packet. Below saturation, useful parallel work is the scarce resource, so maximize useful warp
    // supply and use packet width only as the tie-break. This reproduces the measured small-reduction behavior without
    // shape/K lookup tables and converges to 16-byte packets in the saturated streaming regime.
    std::optional<DenseRKOption> saturated;
    for (DenseRKOption& option : packet_choices) {
        if (!fp8KParallelOptionIsSaturated(option, context)) {
            continue;
        }
        if (!saturated.has_value()
            || option.plan->packet_bytes > saturated->plan->packet_bytes
            || (option.plan->packet_bytes == saturated->plan->packet_bytes
                && betterProgressOption(option, *saturated))) {
            saturated = option;
        }
    }
    if (saturated.has_value()) {
        return *saturated;
    }

    DenseRKOption* best = &packet_choices.front();
    for (DenseRKOption& option : packet_choices) {
        if (option.plan->launch_warp_supply > best->plan->launch_warp_supply
            || (option.plan->launch_warp_supply == best->plan->launch_warp_supply
                && (option.plan->packet_bytes > best->plan->packet_bytes
                    || (option.plan->packet_bytes == best->plan->packet_bytes
                        && betterProgressOption(option, *best))))) {
            best = &option;
        }
    }
    return *best;
}

[[nodiscard]] std::optional<DenseRKOption> selectFp8KParallel(const DenseReductionProblem& problem,
                                                               DenseReductionSite site,
                                                               const DenseRKPlanningContext& context,
                                                               const CubReductionGeometry& geometry,
                                                               const std::vector<uint64_t>& input_dimensions,
                                                               DataType input_dtype) {
    constexpr std::array<size_t, 4> PACKET_BYTES = {16, 8, 4, 2};
    constexpr std::array<uint32_t, 4> CTA_WIDTHS = {32, 64, 128, 256};
    std::vector<DenseRKOption> complete_packet_choices;
    std::vector<DenseRKOption> staged_packet_choices;

    for (size_t packet_bytes : PACKET_BYTES) {
        const uint64_t items_per_lane = packet_bytes;
        if (geometry.inner_size % items_per_lane != 0) {
            continue;
        }
        const uint64_t packet_owners = geometry.inner_size / items_per_lane;
        const uint64_t useful_warps_per_output_shard = ceilDivide(packet_owners, context.warp_size);
        const uint64_t useful_warps_per_shard = checkedMultiply(
            geometry.outer_size,
            useful_warps_per_output_shard,
            "FP8 KParallel useful warps per shard overflows uint64_t.");

        std::vector<DenseRKOption> packet_complete_layouts;
        std::vector<DenseRKOption> packet_staged_layouts;

        for (uint32_t block_threads : CTA_WIDTHS) {
            if (block_threads > K_PARALLEL_MAX_THREADS) {
                continue;
            }
            const uint64_t components_per_cta = checkedMultiply(
                block_threads, items_per_lane, "FP8 KParallel components-per-CTA overflows uint64_t.");
            const uint64_t component_tiles = ceilDivide(geometry.inner_size, components_per_cta);
            const uint64_t base_blocks_per_shard = checkedMultiply(
                geometry.outer_size, component_tiles, "FP8 KParallel blocks-per-shard overflows uint64_t.");

            std::vector<uint64_t> shard_counts{1};
            for (uint64_t requested_rows : FP8_K_PARALLEL_ROW_DEPTHS) {
                if (geometry.reduction_size <= requested_rows) {
                    continue;
                }
                addFp8KParallelShardCandidate(
                    shard_counts, geometry.reduction_size, ceilDivide(geometry.reduction_size, requested_rows));
            }

            // The measured saturation frontier is hardware-relative. Add the exact shard neighborhood that would
            // expose the required useful warp supply for this packet width, plus the maximum legal sharding point for
            // truly under-supplied problems. Exact physical residency is evaluated after the plan is materialized.
            if (useful_warps_per_shard != 0) {
                const uint64_t saturation_shards = ceilDivide(
                    fp8KParallelSaturationTarget(context, packet_bytes), useful_warps_per_shard);
                addFp8KParallelShardNeighborhood(shard_counts, geometry.reduction_size, saturation_shards);
            }
            addFp8KParallelShardCandidate(
                shard_counts, geometry.reduction_size, geometry.reduction_size / K_PARALLEL_MIN_ROWS_PER_SHARD);
            std::sort(shard_counts.begin(), shard_counts.end());

            for (uint64_t shards : shard_counts) {
                const uint64_t first_stage_blocks = checkedMultiply(
                    base_blocks_per_shard, shards, "FP8 KParallel first-stage block count overflows uint64_t.");
                const uint64_t useful_warps = checkedMultiply(
                    useful_warps_per_shard, shards, "FP8 KParallel useful warp count overflows uint64_t.");

                auto plan = std::make_shared<DenseRKFamilyPhysicalPlan>();
                plan->strategy = DenseRKStrategy::KParallel;
                plan->access = DenseRKAccess::Aligned;
                plan->implementation = DenseRKProductionImplementation::KParallelPass;
                plan->input_dimensions = input_dimensions;
                plan->reduction_axis = site.run_index;
                plan->input_dtype = input_dtype;
                plan->value_operation = context.value_operation;
                plan->outer_size = geometry.outer_size;
                plan->reduction_size = geometry.reduction_size;
                plan->inner_size = geometry.inner_size;
                plan->packet_bytes = packet_bytes;
                plan->block_threads = block_threads;
                plan->rows_per_shard = ceilDivide(geometry.reduction_size, shards);
                plan->shards_per_output = shards;
                plan->component_tiles = component_tiles;
                plan->first_stage_blocks = first_stage_blocks;
                plan->useful_stage_warps = useful_warps;
                plan->progress = shards == 1 ? DenseRKProgress::Complete : DenseRKProgress::Staged;
                annotateRKOccupancy(*plan, context);

                DenseRKOption option = makeRKOption(problem, site, std::move(plan), shards, false);
                option.comfort_supply = option.plan->launch_warp_supply;
                option.comfort_target = fp8KParallelSaturationTarget(context, packet_bytes);
                option.calibrated_efficient = fp8KParallelOptionIsSaturated(option, context);
                if (shards == 1) {
                    packet_complete_layouts.push_back(std::move(option));
                } else {
                    packet_staged_layouts.push_back(std::move(option));
                }
            }
        }

        if (!packet_complete_layouts.empty()) {
            complete_packet_choices.push_back(
                chooseFp8KParallelLayoutForPacket(std::move(packet_complete_layouts), context));
        }
        if (!packet_staged_layouts.empty()) {
            staged_packet_choices.push_back(
                chooseFp8KParallelLayoutForPacket(std::move(packet_staged_layouts), context));
        }
    }

    if (complete_packet_choices.empty() && staged_packet_choices.empty()) {
        return std::nullopt;
    }

    std::optional<DenseRKOption> complete;
    if (!complete_packet_choices.empty()) {
        complete = chooseFp8KParallelPacket(std::move(complete_packet_choices), context);
    }
    std::optional<DenseRKOption> staged;
    if (!staged_packet_choices.empty()) {
        staged = chooseFp8KParallelPacket(std::move(staged_packet_choices), context);
    }

    // Preserve the existing measured Complete/Staged crossover. Packet/CTA/shard selection inside each topology is
    // now FP8 saturation-aware, but the topology itself retains the established end-to-end continuation calibration.
    if (geometry.reduction_size < K_PARALLEL_DEEP_CROSSOVER_ROWS && complete.has_value()) {
        return complete;
    }

    const bool complete_saturated = complete.has_value() && fp8KParallelOptionIsSaturated(*complete, context);
    const bool complete_calibrated = complete.has_value()
                                     && geometry.reduction_size <= K_PARALLEL_COMPLETE_MAX_REDUCTION_ROWS
                                     && complete_saturated;
    if (complete.has_value()) {
        complete->calibrated_efficient = complete_calibrated;
    }
    if (staged.has_value()) {
        staged->calibrated_efficient = fp8KParallelOptionIsSaturated(*staged, context);
    }

    if (geometry.reduction_size >= K_PARALLEL_DEEP_CROSSOVER_ROWS
        && !complete_calibrated
        && staged.has_value()) {
        return staged;
    }
    if (!complete.has_value()) {
        return staged;
    }
    if (!staged.has_value()) {
        return complete;
    }
    return betterProgressOption(*staged, *complete) ? staged : complete;
}

[[nodiscard]] std::optional<DenseRKOption> selectKParallel(const DenseReductionProblem& problem,
                                                            DenseReductionSite site,
                                                            const DenseRKPlanningContext& context,
                                                            const CubReductionGeometry& geometry,
                                                            const std::vector<uint64_t>& input_dimensions) {
    const DataType input_dtype = currentRKInputDType(problem, context);
    if (!kParallelStorageDTypeSupported(input_dtype)) {
        return std::nullopt;
    }
    const uint64_t element_bytes = dtypeBytes(input_dtype);
    const bool fp8_input = isFp8StorageDType(input_dtype);
    if (fp8_input) {
        return selectFp8KParallel(problem, site, context, geometry, input_dimensions, input_dtype);
    }

    std::vector<DenseRKOption> options;
    constexpr std::array<uint64_t, 9> ROW_DEPTHS = {4096, 2048, 1024, 512, 256, 128, 64, 32, 16};
    constexpr std::array<size_t, 4> PACKET_BYTES = {16, 8, 4, 2};

    for (size_t packet_bytes : PACKET_BYTES) {
        // FP8 owns 16/8/4/2-byte KParallel packets. The 2-byte specialization is deliberately not admitted for the
        // established FP16/BF16/FP32 populations, so their candidate set and calibrated policy remain unchanged.
        if (!fp8_input && packet_bytes == 2) {
            continue;
        }
        if (packet_bytes < element_bytes || packet_bytes % element_bytes != 0) {
            continue;
        }
        const uint64_t items_per_lane = packet_bytes / element_bytes;
        if (geometry.inner_size % items_per_lane != 0) {
            continue;
        }
        const uint64_t packet_owners = geometry.inner_size / items_per_lane;

        const uint32_t complete_block_threads =
            kParallelBlockThreads(geometry.inner_size, element_bytes, packet_bytes);
        if (complete_block_threads == 0 || complete_block_threads > K_PARALLEL_MAX_THREADS) {
            continue;
        }
        std::vector<uint64_t> shard_counts{1};
        for (uint64_t requested_rows : ROW_DEPTHS) {
            if (geometry.reduction_size <= requested_rows) {
                continue;
            }
            const uint64_t shards = ceilDivide(geometry.reduction_size, requested_rows);
            if (shards <= 1 || geometry.reduction_size / shards < K_PARALLEL_MIN_ROWS_PER_SHARD) {
                continue;
            }
            if (std::find(shard_counts.begin(), shard_counts.end(), shards) == shard_counts.end()) {
                shard_counts.push_back(shards);
            }
        }

        for (uint64_t shards : shard_counts) {
            // Complete is a serial-R terminal pass and uses the coverage-shaped CTA chosen above. Staged K-parallel is
            // different: lanes do not cooperate across a CTA, so making the CTA wider merely bundles independent K
            // packet owners into one scheduling unit. Keep staged CTAs at one warp and expose more component tiles
            // instead. This is the simplest geometry consistent with the kernel's actual independence structure.
            const uint32_t block_threads = shards == 1 ? complete_block_threads : 32U;
            const uint64_t components_per_cta = checkedMultiply(
                block_threads, items_per_lane, "Dense K-parallel components-per-CTA overflows uint64_t.");
            const uint64_t component_tiles = ceilDivide(geometry.inner_size, components_per_cta);
            const uint64_t output_tiles = checkedMultiply(
                geometry.outer_size, component_tiles, "Dense K-parallel output-tile count overflows uint64_t.");
            const uint64_t first_stage_blocks = checkedMultiply(
                output_tiles, shards, "Dense K-parallel first-stage block count overflows uint64_t.");
            const uint64_t useful_warps_per_output_shard = ceilDivide(packet_owners, 32);
            const uint64_t output_shards = checkedMultiply(
                geometry.outer_size, shards, "Dense K-parallel output/shard count overflows uint64_t.");
            const uint64_t useful_warps = checkedMultiply(
                output_shards,
                useful_warps_per_output_shard,
                "Dense K-parallel useful warp count overflows uint64_t.");

            auto plan = std::make_shared<DenseRKFamilyPhysicalPlan>();
            plan->strategy = DenseRKStrategy::KParallel;
            plan->access = DenseRKAccess::Aligned;
            plan->implementation = DenseRKProductionImplementation::KParallelPass;
            plan->input_dimensions = input_dimensions;
            plan->reduction_axis = site.run_index;
            plan->input_dtype = input_dtype;
            plan->value_operation = context.value_operation;
            plan->outer_size = geometry.outer_size;
            plan->reduction_size = geometry.reduction_size;
            plan->inner_size = geometry.inner_size;
            plan->packet_bytes = packet_bytes;
            plan->block_threads = block_threads;
            plan->rows_per_shard = ceilDivide(geometry.reduction_size, shards);
            plan->shards_per_output = shards;
            plan->component_tiles = component_tiles;
            plan->first_stage_blocks = first_stage_blocks;
            plan->useful_stage_warps = useful_warps;
            plan->progress = shards == 1 ? DenseRKProgress::Complete : DenseRKProgress::Staged;
            annotateRKOccupancy(*plan, context);

            const bool calibrated = shards > 1
                                    || (geometry.reduction_size <= K_PARALLEL_COMPLETE_MAX_REDUCTION_ROWS
                                        && meetsWarpComfort(*plan, context, K_PARALLEL_COMPLETE_MIN_WARPS_PER_SM));
            const uint64_t launch_warp_supply = plan->launch_warp_supply;
            const uint64_t complete_comfort_target = shards == 1
                                                         ? targetWarpSupply(*plan, context, K_PARALLEL_COMPLETE_MIN_WARPS_PER_SM)
                                                         : 1;
            DenseRKOption option = makeRKOption(problem, site, std::move(plan), shards, calibrated);
            if (shards == 1) {
                option.comfort_supply = launch_warp_supply;
                option.comfort_target = complete_comfort_target;
            }
            bool duplicate = false;
            for (DenseRKOption& existing : options) {
                if (sameKParallelLaunch(option, existing)) {
                    existing.plan->rows_per_shard = std::max(existing.plan->rows_per_shard,
                                                              option.plan->rows_per_shard);
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                options.push_back(std::move(option));
            }
        }
    }

    // Complete and staged K-parallel have different measured comfort envelopes. Keep them separate here: a shallow
    // Complete pass needs less launch supply, while a staged pass must create enough independent work to enter the
    // line-rate region. Once a candidate is inside its own envelope, reduction progress (next_bytes) decides.
    std::optional<DenseRKOption> complete;
    std::vector<DenseRKOption> staged;
    staged.reserve(options.size());
    for (DenseRKOption& option : options) {
        if (option.plan->progress == DenseRKProgress::Complete) {
            if (!complete.has_value() || betterProgressOption(option, *complete)) {
                complete = option;
            }
        } else {
            staged.push_back(std::move(option));
        }
    }

    // For staged K-parallel, choose progress from the physical work supplied by the first pass rather than from a
    // fixed rows-per-shard table. Preserve the widest legal transaction first; packet narrowing is not a substitute
    // for useful reduction work. Within that packet width, choose the deepest pass that still supplies the calibrated
    // number of one-warp CTAs to the device. If the problem is too small to reach that floor, choose the best-supplied
    // candidate at the same packet width.
    size_t staged_packet_bytes = 0;
    for (size_t preferred_packet_bytes : PACKET_BYTES) {
        if (std::any_of(staged.begin(), staged.end(), [&](const DenseRKOption& option) {
                return option.plan->packet_bytes == preferred_packet_bytes;
            })) {
            staged_packet_bytes = preferred_packet_bytes;
            break;
        }
    }
    if (staged_packet_bytes != 0) {
        staged.erase(
            std::remove_if(staged.begin(), staged.end(), [&](const DenseRKOption& option) {
                return option.plan->packet_bytes != staged_packet_bytes;
            }),
            staged.end());
    }
    std::optional<DenseRKOption> staged_choice = chooseProgressWithinOccupancyEnvelope(
        std::move(staged), true, context, K_PARALLEL_STAGED_PROGRESS_WARPS_PER_SM);

    // End-to-end calibration still shows that shallow reductions are genuinely Complete-favored once continuation
    // launch cost is included. Staged remains a fallback when Complete is unavailable.
    if (geometry.reduction_size < K_PARALLEL_DEEP_CROSSOVER_ROWS && complete.has_value()) {
        return complete;
    }

    const bool complete_underfilled =
        complete.has_value()
        && complete->comfort_target != 0
        && complete->comfort_supply < complete->comfort_target;
    if (geometry.reduction_size >= K_PARALLEL_DEEP_CROSSOVER_ROWS
        && complete_underfilled
        && staged_choice.has_value()) {
        return staged_choice;
    }

    std::optional<DenseRKOption> selected;
    if (complete.has_value()) {
        selected = complete;
    }
    if (staged_choice.has_value()
        && (!selected.has_value() || betterProgressOption(*staged_choice, *selected))) {
        selected = staged_choice;
    }
    return selected;
}

[[nodiscard]] size_t choosePacketAdaptiveBytes(uint64_t inner_size, uint64_t element_bytes) {
    if (element_bytes == 0 || 16 % element_bytes != 0) {
        throw std::logic_error("Dense R-cooperative packet planner received unsupported element width.");
    }
    const uint64_t items16 = 16 / element_bytes;
    const uint64_t items8 = 8 / element_bytes;
    const uint64_t components16 = 32 * items16;
    const uint64_t components8 = 32 * items8;
    const uint64_t row_bytes = checkedMultiply(
        inner_size, element_bytes, "Dense R-cooperative row byte count overflows uint64_t.");

    size_t packet_bytes = 16;
    if (inner_size <= components16 / 2 && row_bytes % 8 == 0) {
        packet_bytes = 8;
        if (inner_size <= components8 / 2 && row_bytes % 4 == 0) {
            packet_bytes = 4;
        }
    }
    return packet_bytes;
}

[[nodiscard]] uint32_t packetAdaptiveWarpsPerTile(uint64_t outer_size,
                                                  uint64_t reduction_size,
                                                  uint64_t inner_size,
                                                  uint64_t element_bytes,
                                                  size_t packet_bytes) {
    const uint64_t items_per_lane = packet_bytes / element_bytes;
    const uint64_t components_per_tile = 32 * items_per_lane;
    const uint64_t component_tiles = ceilDivide(inner_size, components_per_tile);
    const uint64_t output_tiles = std::max<uint64_t>(
        1, checkedMultiply(outer_size, component_tiles, "Dense R-cooperative output tile count overflows uint64_t."));
    const uint64_t desired_warps = ceilDivide(R_COOPERATIVE_COMPLETE_TARGET_WARPS, output_tiles);

    uint32_t warps = 1;
    while (warps < R_COOPERATIVE_MAX_COMPLETE_WARPS && static_cast<uint64_t>(warps) < desired_warps
           && static_cast<uint64_t>(warps) < reduction_size) {
        warps <<= 1;
    }
    while (warps < R_COOPERATIVE_MAX_COMPLETE_WARPS && reduction_size > static_cast<uint64_t>(warps)
           && (checkedMultiply(inner_size,
                               checkedMultiply(element_bytes,
                                               warps,
                                               "Dense R-cooperative row/warp byte stride overflows uint64_t."),
                               "Dense R-cooperative row/warp byte stride overflows uint64_t.")
               % packet_bytes)
                  != 0) {
        warps <<= 1;
    }
    return warps;
}

[[nodiscard]] std::optional<DenseRKOption> packetAdaptiveCompleteOption(const DenseReductionProblem& problem,
                                                                        DenseReductionSite site,
                                                                        const DenseRKPlanningContext& context,
                                                                        const CubReductionGeometry& geometry,
                                                                        const std::vector<uint64_t>& input_dimensions) {
    const DataType input_dtype = currentRKInputDType(problem, context);
    if (!rCooperativeStorageDTypeSupported(input_dtype) || !additiveValueOperation(context.value_operation)) {
        return std::nullopt;
    }
    const uint64_t element_bytes = dtypeBytes(input_dtype);
    const size_t packet_bytes = choosePacketAdaptiveBytes(geometry.inner_size, element_bytes);
    const uint64_t items_per_lane = packet_bytes / element_bytes;
    const uint64_t components_per_tile = 32 * items_per_lane;
    const uint64_t component_tiles = ceilDivide(geometry.inner_size, components_per_tile);
    const uint64_t output_tiles = checkedMultiply(
        geometry.outer_size, component_tiles, "Dense packet-adaptive output-tile count overflows uint64_t.");
    const uint32_t warps_per_tile = packetAdaptiveWarpsPerTile(
        geometry.outer_size, geometry.reduction_size, geometry.inner_size, element_bytes, packet_bytes);
    const uint32_t block_warps = std::max<uint32_t>(R_COOPERATIVE_BLOCK_WARPS, warps_per_tile);
    const uint32_t groups_per_block = block_warps / warps_per_tile;
    const uint64_t first_stage_blocks = ceilDivide(output_tiles, groups_per_block);
    const uint64_t useful_warps = checkedMultiply(
        output_tiles, warps_per_tile, "Dense packet-adaptive useful warp count overflows uint64_t.");

    auto plan = std::make_shared<DenseRKFamilyPhysicalPlan>();
    plan->strategy = DenseRKStrategy::RCooperative;
    plan->access = DenseRKAccess::Aligned;
    plan->implementation = DenseRKProductionImplementation::RCooperativeAlignedComplete;
    plan->input_dimensions = input_dimensions;
    plan->reduction_axis = site.run_index;
    plan->input_dtype = input_dtype;
    plan->value_operation = context.value_operation;
    plan->outer_size = geometry.outer_size;
    plan->reduction_size = geometry.reduction_size;
    plan->inner_size = geometry.inner_size;
    plan->packet_bytes = packet_bytes;
    plan->block_threads = block_warps * 32;
    plan->cooperative_warps_per_tile = warps_per_tile;
    plan->rows_per_shard = geometry.reduction_size;
    plan->shards_per_output = 1;
    plan->component_tiles = component_tiles;
    plan->first_stage_blocks = first_stage_blocks;
    plan->useful_stage_warps = useful_warps;
    plan->progress = DenseRKProgress::Complete;
    annotateRKOccupancy(*plan, context);

    const uint64_t rows_per_cooperative_warp = ceilDivide(geometry.reduction_size, warps_per_tile);
    const bool calibrated = geometry.reduction_size >= CubReductionTiledPolicy::PACKET_COOPERATIVE_MIN_REDUCTION_SIZE
                            && geometry.inner_size >= CubReductionTiledPolicy::PACKET_COOPERATIVE_MIN_INNER_SIZE
                            && geometry.inner_size <= CubReductionTiledPolicy::PACKET_COOPERATIVE_MAX_INNER_SIZE
                            && warps_per_tile >= CubReductionTiledPolicy::PACKET_COOPERATIVE_MIN_WARPS_PER_TILE
                            && rows_per_cooperative_warp <= R_COOPERATIVE_COMPLETE_MAX_ROWS_PER_COOPERATIVE_WARP
                            && meetsWarpComfort(*plan, context, R_COOPERATIVE_COMPLETE_MIN_WARPS_PER_SM);
    DenseRKOption option = makeRKOption(problem, site, std::move(plan), 1, calibrated);
    option.comfort_target = targetWarpSupply(*option.plan, context, R_COOPERATIVE_COMPLETE_MIN_WARPS_PER_SM);
    option.comfort_supply = calibrated ? option.plan->launch_warp_supply : 0;
    return option;
}

void appendAlignedStagedOptions(std::vector<DenseRKOption>& options,
                                const DenseReductionProblem& problem,
                                DenseReductionSite site,
                                const DenseRKPlanningContext& context,
                                const CubReductionGeometry& geometry,
                                const std::vector<uint64_t>& input_dimensions) {
    const DataType input_dtype = currentRKInputDType(problem, context);
    if (!rCooperativeStorageDTypeSupported(input_dtype)) {
        return;
    }
    const uint64_t element_bytes = dtypeBytes(input_dtype);
    // Aligned cooperative staging is the wide-K path. Narrow low-precision rows have their own calibrated FlatRows
    // implementation below; do not use this generic aligned path to encode that narrow selector envelope.
    if (geometry.inner_size < 256 || 16 % element_bytes != 0) {
        return;
    }
    const uint64_t vector_elements = 16 / element_bytes;
    if (geometry.inner_size % vector_elements != 0) {
        return;
    }

    const uint64_t components_per_tile = 32 * vector_elements;
    const uint64_t component_tiles = ceilDivide(geometry.inner_size, components_per_tile);
    const uint64_t outer_component_tiles = checkedMultiply(
        geometry.outer_size, component_tiles, "Dense aligned R-cooperative output-tile count overflows uint64_t.");

    constexpr std::array<uint64_t, 4> ROW_DEPTHS = {1024, 512, 256, 128};
    for (uint64_t rows_per_shard : ROW_DEPTHS) {
        if (geometry.reduction_size <= rows_per_shard) {
            continue;
        }
        const uint64_t shards = ceilDivide(geometry.reduction_size, rows_per_shard);
        const uint64_t first_stage_blocks = checkedMultiply(
            outer_component_tiles, shards, "Dense aligned R-cooperative first-stage blocks overflow uint64_t.");
        const uint64_t useful_warps = checkedMultiply(
            first_stage_blocks,
            R_COOPERATIVE_BLOCK_WARPS,
            "Dense aligned R-cooperative useful warp count overflows uint64_t.");

        auto plan = std::make_shared<DenseRKFamilyPhysicalPlan>();
        plan->strategy = DenseRKStrategy::RCooperative;
        plan->access = DenseRKAccess::Aligned;
        plan->implementation = DenseRKProductionImplementation::RCooperativeAlignedStaged;
        plan->input_dimensions = input_dimensions;
        plan->reduction_axis = site.run_index;
        plan->input_dtype = input_dtype;
        plan->value_operation = context.value_operation;
        plan->outer_size = geometry.outer_size;
        plan->reduction_size = geometry.reduction_size;
        plan->inner_size = geometry.inner_size;
        plan->packet_bytes = 16;
        plan->block_threads = R_COOPERATIVE_BLOCK_THREADS;
        plan->cooperative_warps_per_tile = R_COOPERATIVE_BLOCK_WARPS;
        plan->rows_per_shard = rows_per_shard;
        plan->shards_per_output = shards;
        plan->component_tiles = component_tiles;
        plan->first_stage_blocks = first_stage_blocks;
        plan->useful_stage_warps = useful_warps;
        plan->progress = DenseRKProgress::Staged;
        annotateRKOccupancy(*plan, context);
        options.push_back(makeRKOption(problem, site, std::move(plan), shards, true));
    }
}

void appendRotatedStagedOptions(std::vector<DenseRKOption>& options,
                                const DenseReductionProblem& problem,
                                DenseReductionSite site,
                                const DenseRKPlanningContext& context,
                                const CubReductionGeometry& geometry,
                                const std::vector<uint64_t>& input_dimensions) {
    const DataType input_dtype = currentRKInputDType(problem, context);
    if (!rCooperativeStorageDTypeSupported(input_dtype)) {
        return;
    }
    const uint64_t element_bytes = dtypeBytes(input_dtype);
    if ((element_bytes != 2 && element_bytes != 4) || 16 % element_bytes != 0) {
        return;
    }
    const uint64_t vector_elements = 16 / element_bytes;
    // Below K=256, packet-legal widths are handled by KParallel or the calibrated narrow FlatRows path. Rotated remains
    // the modern coverage path for packet-misaligned widths when FlatRows cannot stage (for example shallow R).
    if (geometry.inner_size < 256 && kParallelHasLegalPacket(geometry.inner_size, element_bytes)) {
        return;
    }
    const uint64_t packet_span_elements = 32 * vector_elements;
    const uint64_t logical_tile_elements = packet_span_elements - (vector_elements - 1);
    const uint64_t component_tiles = ceilDivide(geometry.inner_size, logical_tile_elements);
    const uint64_t outer_component_tiles = checkedMultiply(
        geometry.outer_size, component_tiles, "Dense rotated R-cooperative output-tile count overflows uint64_t.");

    constexpr std::array<uint64_t, 4> ROW_DEPTHS = {1024, 512, 256, 128};
    for (uint64_t rows_per_shard : ROW_DEPTHS) {
        if (geometry.reduction_size <= rows_per_shard) {
            continue;
        }
        const uint64_t shards = ceilDivide(geometry.reduction_size, rows_per_shard);
        const uint64_t first_stage_blocks = checkedMultiply(
            outer_component_tiles, shards, "Dense rotated R-cooperative first-stage block count overflows uint64_t.");
        const uint64_t useful_warps = checkedMultiply(
            first_stage_blocks,
            R_COOPERATIVE_BLOCK_WARPS,
            "Dense rotated R-cooperative useful warp count overflows uint64_t.");

        auto plan = std::make_shared<DenseRKFamilyPhysicalPlan>();
        plan->strategy = DenseRKStrategy::RCooperative;
        plan->access = DenseRKAccess::Rotated;
        plan->implementation = DenseRKProductionImplementation::RCooperativeRotatedStaged;
        plan->input_dimensions = input_dimensions;
        plan->reduction_axis = site.run_index;
        plan->input_dtype = input_dtype;
        plan->value_operation = context.value_operation;
        plan->outer_size = geometry.outer_size;
        plan->reduction_size = geometry.reduction_size;
        plan->inner_size = geometry.inner_size;
        plan->packet_bytes = 16;
        plan->block_threads = R_COOPERATIVE_BLOCK_THREADS;
        plan->cooperative_warps_per_tile = R_COOPERATIVE_BLOCK_WARPS;
        plan->rows_per_shard = rows_per_shard;
        plan->shards_per_output = shards;
        plan->component_tiles = component_tiles;
        plan->first_stage_blocks = first_stage_blocks;
        plan->useful_stage_warps = useful_warps;
        plan->progress = DenseRKProgress::Staged;
        annotateRKOccupancy(*plan, context);
        options.push_back(makeRKOption(problem, site, std::move(plan), shards, true));
    }
}

void appendNarrowFlatStagedOptions(std::vector<DenseRKOption>& options,
                                   const DenseReductionProblem& problem,
                                   DenseReductionSite site,
                                   const DenseRKPlanningContext& context,
                                   const CubReductionGeometry& geometry,
                                   const std::vector<uint64_t>& input_dimensions) {
    const DataType input_dtype = currentRKInputDType(problem, context);
    const bool low_precision = input_dtype == DataType::FP16 || input_dtype == DataType::BF16;
    const bool calibrated_flat_k = geometry.inner_size != 0 && geometry.inner_size <= 32
                                   && ((geometry.inner_size & 1U) != 0 || geometry.inner_size >= 16);
    if (!low_precision || !calibrated_flat_k) {
        return;
    }

    // The complete K=1..32 low-precision calibration sweep establishes one operation-independent ownership rule:
    // odd K uses FlatRows; even K<=14 stays KParallel; even K>=16 uses FlatRows. It also establishes a clean FlatRows
    // shard-depth transition: K<20 uses 512 rows/shard and K>=20 uses 128 rows/shard. Keep this measured geometry
    // beneath RCooperative instead of asking the generic occupancy/progress tie-breaker to rediscover it.
    const uint64_t rows_per_shard = geometry.inner_size < 20 ? 512 : 128;
    if (geometry.reduction_size <= rows_per_shard) {
        return;
    }
    {
        const uint64_t shards = ceilDivide(geometry.reduction_size, rows_per_shard);
        const uint64_t first_stage_blocks = checkedMultiply(
            geometry.outer_size, shards, "Dense narrow-flat R-cooperative first-stage block count overflows uint64_t.");
        const uint64_t useful_warps = checkedMultiply(
            first_stage_blocks,
            R_COOPERATIVE_BLOCK_WARPS,
            "Dense narrow-flat R-cooperative useful warp count overflows uint64_t.");

        auto plan = std::make_shared<DenseRKFamilyPhysicalPlan>();
        plan->strategy = DenseRKStrategy::RCooperative;
        plan->access = DenseRKAccess::FlatRows;
        plan->implementation = DenseRKProductionImplementation::RCooperativeFlatRowsStaged;
        plan->input_dimensions = input_dimensions;
        plan->reduction_axis = site.run_index;
        plan->input_dtype = input_dtype;
        plan->value_operation = context.value_operation;
        plan->outer_size = geometry.outer_size;
        plan->reduction_size = geometry.reduction_size;
        plan->inner_size = geometry.inner_size;
        plan->packet_bytes = 16;
        plan->block_threads = R_COOPERATIVE_BLOCK_THREADS;
        plan->cooperative_warps_per_tile = R_COOPERATIVE_BLOCK_WARPS;
        plan->rows_per_shard = rows_per_shard;
        plan->shards_per_output = shards;
        plan->component_tiles = 1;
        plan->first_stage_blocks = first_stage_blocks;
        plan->useful_stage_warps = useful_warps;
        plan->progress = DenseRKProgress::Staged;
        annotateRKOccupancy(*plan, context);
        options.push_back(makeRKOption(problem, site, std::move(plan), shards, true));
    }
}

[[nodiscard]] std::optional<DenseRKOption> selectRCooperative(const DenseReductionProblem& problem,
                                                               DenseReductionSite site,
                                                               const DenseRKPlanningContext& context,
                                                               const CubReductionGeometry& geometry,
                                                               const std::vector<uint64_t>& input_dimensions) {
    std::optional<DenseRKOption> complete =
        packetAdaptiveCompleteOption(problem, site, context, geometry, input_dimensions);

    std::vector<DenseRKOption> aligned_options;
    appendAlignedStagedOptions(aligned_options, problem, site, context, geometry, input_dimensions);
    std::vector<DenseRKOption> rotated_options;
    appendRotatedStagedOptions(rotated_options, problem, site, context, geometry, input_dimensions);
    std::vector<DenseRKOption> narrow_flat_options;
    appendNarrowFlatStagedOptions(narrow_flat_options, problem, site, context, geometry, input_dimensions);

    std::optional<DenseRKOption> selected;
    const auto consider = [&](const std::optional<DenseRKOption>& option) {
        if (option.has_value() && (!selected.has_value() || betterProgressOption(*option, *selected))) {
            selected = *option;
        }
    };

    if (complete.has_value() && complete->calibrated_efficient) {
        consider(complete);
    }
    consider(chooseProgressWithinOccupancyEnvelope(
        std::move(aligned_options), true, context, ALIGNED_R_COOPERATIVE_STAGED_COMFORT_WARPS_PER_SM));

    const DataType input_dtype = currentRKInputDType(problem, context);
    const uint32_t rotated_target = input_dtype == DataType::FP32
                                        ? ROTATED_FP32_STAGED_COMFORT_WARPS_PER_SM
                                        : ROTATED_LOW_PRECISION_STAGED_COMFORT_WARPS_PER_SM;
    // The K=1..32 calibration establishes FlatRows ownership for its admitted narrow low-precision band. Do not let
    // the generic same-strategy implementation-id tie-breaker silently choose rotated there. If no flat staged
    // candidate exists (for example shallow R), rotated remains the coverage path.
    if (narrow_flat_options.empty()) {
        consider(chooseProgressWithinOccupancyEnvelope(
            std::move(rotated_options), true, context, rotated_target));
    } else {
        consider(chooseProgressWithinOccupancyEnvelope(
            std::move(narrow_flat_options), true, context, NARROW_FLAT_STAGED_COMFORT_WARPS_PER_SM));
    }

    if (selected.has_value()) {
        return selected;
    }

    // A shallow problem may have no staged geometry at all. Keep packet-adaptive Complete as the modern correctness
    // endpoint even outside its calibrated throughput envelope; cross-strategy selection will still prefer any
    // calibrated K-parallel candidate.
    return complete;
}

[[nodiscard]] DenseReductionCandidate makeRKCandidate(DenseReductionSite site, DenseRKOption option) {
    if (!denseRKIsFamilyImplementation(option.plan->implementation)) {
        throw std::logic_error("Dense RK modern selector chose an implementation outside the family inventory.");
    }
    const DenseRKProductionInventoryEntry& inventory =
        denseRKProductionInventoryEntry(option.plan->implementation);
    if (!inventory.strategy.has_value() || *inventory.strategy != option.plan->strategy
        || !inventory.access.has_value() || *inventory.access != option.plan->access
        || inventory.owns_successor) {
        throw std::logic_error("Dense RK modern selector disagrees with the declared production inventory.");
    }
    if (option.plan->progress == DenseRKProgress::Complete && !inventory.supports_complete) {
        throw std::logic_error("Dense RK modern selector chose Complete progress on a staged-only implementation.");
    }
    if (option.plan->progress == DenseRKProgress::Staged && !inventory.supports_staged) {
        throw std::logic_error("Dense RK modern selector chose Staged progress on a complete-only implementation.");
    }

    DenseReductionCandidate candidate;
    candidate.family = DenseReducerFamily::RK;
    candidate.site = site;
    candidate.physical_plan = std::move(option.plan);
    candidate.next_problem = std::move(option.next_problem);
    candidate.next_bytes = option.next_bytes;
    candidate.scratch_bytes = option.scratch_bytes;
    return candidate;
}

}  // namespace

DenseReductionCandidate ReducersDenseR::propose(const DenseReductionProblem& problem, DenseReductionSite site) {
    return proposeDirect(problem, site, DenseReducerFamily::R);
}

DenseReductionCandidate ReducersDenseKR::propose(const DenseReductionProblem& problem, DenseReductionSite site) {
    return proposeDirect(problem, site, DenseReducerFamily::KR);
}

std::optional<DenseReductionCandidate> ReducersDenseRK::proposeProductionKParallelForCalibration(
    const DenseReductionProblem& problem,
    DenseReductionSite site,
    const DenseRKPlanningContext& context) {
    if (classifyDenseReducerFamily(problem, site) != DenseReducerFamily::RK) {
        throw std::invalid_argument("Dense RK KParallel calibration hook requires an RK-classified nominated site.");
    }
    if (context.multiprocessors == 0 || context.warp_size == 0) {
        throw std::invalid_argument("Dense RK planning context requires non-zero multiprocessor/warp geometry.");
    }
    if (context.occupancy_query == nullptr
        && (context.max_threads_per_sm == 0 || context.max_blocks_per_sm == 0)) {
        throw std::invalid_argument("Dense RK fallback occupancy requires non-zero device thread/block limits.");
    }
    if (problem.aggregate_kind == DenseReductionAggregateKind::Arg) {
        return std::nullopt;
    }

    const std::vector<uint64_t> input_dimensions = currentRunDimensions(problem);
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(
        input_dimensions, std::vector<uint32_t>{site.run_index});
    if (geometry.path != CubReductionPath::TiledFixedSegment || geometry.axes.size() != 1
        || geometry.axes.front() != site.run_index || geometry.inner_size <= 1) {
        throw std::logic_error("Dense RK KParallel calibration hook expected one ordinary TiledFixedSegment site.");
    }

    std::optional<DenseRKOption> selected = selectKParallel(problem, site, context, geometry, input_dimensions);
    if (!selected.has_value()) {
        return std::nullopt;
    }
    return makeRKCandidate(site, std::move(*selected));
}

DenseReductionCandidate ReducersDenseRK::propose(const DenseReductionProblem& problem,
                                                 DenseReductionSite site,
                                                 const DenseRKPlanningContext& context) {
    if (classifyDenseReducerFamily(problem, site) != DenseReducerFamily::RK) {
        throw std::invalid_argument("Dense RK family can only propose for an RK-classified nominated site.");
    }
    if (context.multiprocessors == 0 || context.warp_size == 0) {
        throw std::invalid_argument("Dense RK planning context requires non-zero multiprocessor/warp geometry.");
    }
    if (context.occupancy_query == nullptr
        && (context.max_threads_per_sm == 0 || context.max_blocks_per_sm == 0)) {
        throw std::invalid_argument("Dense RK fallback occupancy requires non-zero device thread/block limits.");
    }

    // DENSE-ARG-CUTOVER is intentionally later. Preserve the already-proven ARG topology adapter rather than creating
    // a second ARG-specific RK planner inside this patch.
    if (problem.aggregate_kind == DenseReductionAggregateKind::Arg) {
        return proposeDirect(problem, site, DenseReducerFamily::RK);
    }

    const std::vector<uint64_t> input_dimensions = currentRunDimensions(problem);
    const CubReductionGeometry geometry = CubReduction::analyzeGeometry(input_dimensions, std::vector<uint32_t>{site.run_index});
    if (geometry.path != CubReductionPath::TiledFixedSegment || geometry.axes.size() != 1
        || geometry.axes.front() != site.run_index || geometry.inner_size <= 1) {
        throw std::logic_error("Dense RK family expected one ordinary TiledFixedSegment nominated site.");
    }

    const std::optional<DenseRKOption> k_parallel =
        selectKParallel(problem, site, context, geometry, input_dimensions);
    const std::optional<DenseRKOption> r_cooperative =
        selectRCooperative(problem, site, context, geometry, input_dimensions);

    std::optional<DenseRKOption> selected;
    if (k_parallel.has_value()) {
        selected = k_parallel;
    }
    if (r_cooperative.has_value()) {
        // FlatRows owns its calibrated narrow low-precision band whenever a legal staged flat pass exists. Otherwise,
        // when KParallel already supplies its calibrated physical-work floor and makes at least as much R progress as
        // the cooperative pass, keep the simpler independent-packet implementation. At that point RCooperative is not
        // buying launch parallelism; it is only adding cooperative machinery.
        //
        // There is one measured wide-FP32 exception. If KParallel is forced to scalar-per-lane (4-byte) packets and
        // the input stream is at least 768 MiB, family-boundary calibration shows that rotated RCooperative's wider
        // vectorized/cooperative geometry wins by meaningful absolute wall time. This is intentionally expressed in
        // terms of packet geometry and streamed bytes, not benchmark-specific K/R values or a rows-per-shard lookup.
        const uint64_t input_bytes = checkedMultiply(
            geometry.input_elements, dtypeBytes(currentRKInputDType(problem, context)),
            "Dense RK input byte count overflows uint64_t.");
        const bool prefer_large_fp32_rotated =
            selected.has_value()
            && selected->plan->strategy == DenseRKStrategy::KParallel
            && selected->plan->input_dtype == DataType::FP32
            && selected->plan->packet_bytes == 4
            && input_bytes >= FP32_SCALAR_PACKET_RCOOPERATIVE_MIN_INPUT_BYTES
            && r_cooperative->plan->strategy == DenseRKStrategy::RCooperative
            && r_cooperative->plan->access == DenseRKAccess::Rotated;

        const bool k_parallel_sufficient =
            selected.has_value()
            && selected->plan->strategy == DenseRKStrategy::KParallel
            && selected->plan->progress == DenseRKProgress::Staged
            && selected->comfort_target != 0
            && selected->comfort_supply >= selected->comfort_target;
        const bool k_parallel_progress_is_at_least_as_deep =
            k_parallel_sufficient && selected->next_bytes <= r_cooperative->next_bytes;
        if (prefer_large_fp32_rotated
            || r_cooperative->plan->access == DenseRKAccess::FlatRows
            || !selected.has_value()
            || (!k_parallel_progress_is_at_least_as_deep
                && betterProgressOption(*r_cooperative, *selected))) {
            selected = r_cooperative;
        }
    }
    if (selected.has_value()) {
        return makeRKCandidate(site, std::move(*selected));
    }

    if (context.allow_generic_rk_fallback) {
        return proposeDirect(problem, site, DenseReducerFamily::RK);
    }
    throw std::logic_error(
        "Dense RK modern family has no calibrated/feasible KParallel or RCooperative implementation for this site; "
        "generic RK fallback was not enabled.");
}

}  // namespace ThorImplementation::CubReductionInternal
