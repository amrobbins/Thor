#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKStampAutotuner.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKBoundSelection.h"
#include "Utilities/TensorOperations/Reduction/RK/Fp8/Fp8RKProductionCandidates.h"
#include "Utilities/TensorOperations/Reduction/RK/RKRuntimeProposals.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Reduction/RK/RKSelectionKey.h"
#include "Utilities/TensorOperations/Reduction/RK/RKDescriptorSelection.h"
#include "Utilities/TensorOperations/Reduction/RK/RKSelectedContinuationBenchmark.h"
#include "Utilities/TensorOperations/Reduction/Planner/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Reduction/Arg/Fp8ArgPhysicalPlan.h"
#include "Utilities/TensorOperations/Reduction/ArgReducer.h"
#include "Utilities/TensorOperations/Reduction/Contest/ReductionAutotuner.h"
#include "Utilities/TensorOperations/Reduction/Contest/ReductionGpuMeasurementLock.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation::ReductionInternal::RK::Fp8 {
namespace {
struct Shape {
    uint64_t outer = 1, r = 0, k = 1;
};
uint64_t checkedMul(uint64_t a, uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("FP8 RK stamp shape overflow");
    return a * b;
}
Shape shapeAtSite(const DenseReductionProblem& problem, DenseReductionSite site) {
    if (classifyDenseReducerFamily(problem, site) != DenseReducerFamily::RK)
        throw std::invalid_argument("FP8 RK autotuning needs a middle-axis RK site");
    Shape shape;
    for (size_t i = 0; i < problem.runs.size(); ++i) {
        if (i < site.run_index) shape.outer = checkedMul(shape.outer, problem.runs[i].extent);
        else if (i == site.run_index) shape.r = problem.runs[i].extent;
        else shape.k = checkedMul(shape.k, problem.runs[i].extent);
    }
    return shape;
}
// The shared, per-ordinal cached device identity of the dense reduction selection cache.
RKDeviceSelectionIdentity deviceIdentity(const Stream& stream) {
    return denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
}

// The input alignment an original FP8 RK site's selection honours: the input's, at most 16 (as FP32's).
uint64_t siteAlignment(uint64_t input_alignment_bytes) {
    if (input_alignment_bytes == 0 || (input_alignment_bytes & (input_alignment_bytes - 1)) != 0)
        throw std::invalid_argument("FP8 RK input alignment must be a power of two");
    return std::min<uint64_t>(input_alignment_bytes, 16);
}

const char* familyName(PhysicalFamily family) {
    switch (family) {
        case PhysicalFamily::OneWindow: return "one_window";
        case PhysicalFamily::DeepRSmallK: return "deep_r_small_k";
        case PhysicalFamily::AlignedK: return "aligned_k";
        case PhysicalFamily::RuntimeK: return "runtime_k";
    }
    throw std::logic_error("Unknown physical FP8 RK family");
}
}

RKSelection autotuneIsolatedRKSite(
    const DenseReductionProblem& problem, DenseReductionSite site, ReductionOp operation,
    uint64_t original_reduction_size, float output_scale,
    const StridedSpan& representative_input, const Stream& stream, uint64_t workspace_budget_bytes,
    ReductionTuningDepth tuning_depth, uint64_t input_alignment_bytes) {
    if (representative_input.getDataType() != DataType::FP8_E4M3 &&
        representative_input.getDataType() != DataType::FP8_E5M2)
        throw std::invalid_argument("FP8 RK autotuning requires FP8 input");
    const uint64_t alignment = siteAlignment(input_alignment_bytes);
    if (problem.input_state != DenseReductionInputState::OriginalInput)
        throw std::invalid_argument("Only the original FP8 input is supported by this tuning entry point");
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const Shape shape = shapeAtSite(problem, site);
    if (!representative_input.isDenseContiguous() ||
        representative_input.getTotalNumElements() != checkedMul(checkedMul(shape.outer, shape.r), shape.k) ||
        representative_input.getPlacement().getDeviceNum() != stream.getGpuNum())
        throw std::invalid_argument("FP8 RK autotune input must match contiguous selected-site geometry and stream");
    if (original_reduction_size == 0)
        throw std::invalid_argument("FP8 RK original reduction size must be nonzero");
    const bool final_site = !hasDenseReductionSites(advanceDenseReductionProblem(problem, site, 1));
    const auto semantics = makeValueReductionStageSemantics(
        operation, final_site ? DensePassRole::Complete : DensePassRole::First,
        original_reduction_size);
    const auto device = deviceIdentity(stream);
    const auto occupancy = [&](const PhysicalKernel& kernel) {
        return runtimeKResidency(representative_input.getDataType(), semantics, kernel);
    };
    const auto runtime = enumerateRetainedRuntimeK(
        {shape.outer, shape.r, shape.k, device.sm_count}, occupancy);
    const ReductionTuningPlan tuning = reductionTuningPlan(tuning_depth);
    const BoundSelectionRequest request{
        shape.outer, shape.r, shape.k, device.sm_count,
        representative_input.getDataType(),
        final_site ? problem.final_output.value_dtype : DataType::FP32,
        semantics, workspace_budget_bytes, tuning.candidates, alignment};
    auto ranked = selectBoundFirstStages(request, runtime);
    // A single, owning production context supplies the exact same FP32
    // occupancy and physical policy as ordinary dense planning. It remains
    // alive throughout every dependency selection (outside the GPU lock).
    // With the original semantics and the tuning depth, which the FP32 routing requires.
    const DenseProductionPhysicalSelectionContext fp32_context(
        representative_input.getDataType(), denseRKValueOperation(operation), stream, original_reduction_size,
        output_scale, operation, workspace_budget_bytes, tuning_depth);
    std::vector<SelectedIsolatedBenchmark> executables;
    std::vector<ReductionAutotuning::Candidate> candidates;
    executables.reserve(ranked.size());
    candidates.reserve(ranked.size());
    for (const auto& kernel : ranked) {
        // Ranker filtering already guarantees a legal first physical kernel.
        // Only a typed candidate-local epilogue/budget exclusion may skip a
        // choice. In particular, invalid FP32 selector decisions, unexpected
        // resource-query errors and CUDA failures must not disappear here.
        IsolatedRKSite isolated = planIsolatedRKSite(problem, site, kernel, semantics);
        try {
            auto executable = makeSelectedIsolatedBenchmark(
                problem, site, std::move(isolated), kernel, fp32_context.context(),
                output_scale, workspace_budget_bytes, stream);
            candidates.push_back(executable.asAutotuneCandidate(
                std::string(familyName(kernel.family)) + "#" + std::to_string(candidates.size())));
            executables.push_back(std::move(executable));
        } catch (const UnavailableIsolatedCandidate&) {
            continue;
        }
    }
    if (candidates.empty())
        throw std::runtime_error("No legal retained FP8 RK complete candidate under production workspace budget");

    const auto& first = executables.front();
    const TensorDescriptor output_descriptor(first.site.site_output_dtype,
                                               {shape.outer, shape.k});
    Stream tuning_stream = stream; // shared CUDA stream, independent mutable handle
    // Candidate construction and hypothetical FP32 selection finish BEFORE
    // acquiring this measurement lock.
    // Keep the entire interleaved contest serialized on this GPU; the
    // autotuner excludes buffer setup from its timed CUDA event intervals.
    const auto result = [&] {
        ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
        ReductionAutotuning::Options options;
        options.measurement_count = tuning.rounds;  // drop fastest and slowest, average the rest
        options.trim_count = tuning.trimmed;
        // Launch counts and abandonment are the shared contest policy (ReductionAutotuning::Options), as for KR.
        return ReductionAutotuning::autotune(
            candidates, representative_input, output_descriptor, tuning_stream, options);
    }();
    if (result.winner_index >= executables.size())
        throw std::logic_error("FP8 RK autotuner returned an out-of-range winner");
    RKSelection selection;
    // The selected physical kernel is the ONLY executable identity in the
    // cache. Lookahead passes, timing buffers, and their workspace accounting
    // die with the temporary measurement programs above.
    selection.physical = executables[result.winner_index].first_physical;
    selection.winner_index = result.winner_index;
    selection.winner_name = result.winner_name;
    selection.candidate_count = candidates.size();
    return selection;
}


// FP8 ARG ranks and autotunes its own candidates (no VALUE proxy). The candidates are the ARG twins of the complete
// VALUE MIN/MAX inventory plus the ARG-only rotated passes (retainedFp8ArgCandidates), ranked by the same geometry
// model with the ARG aggregate feature, the top reductionTuningPlan(depth).candidates timed as the production pass runs
// them (launchRetainedFp8ArgPlan: packed staged passes include their packed continuation).
// The selection caches only the winner's ARG identity (retainedFp8ArgPhysicalIdentity); candidate_count 0 means no ARG
// pass is legal at this site.
RKSelection autotuneArgRKSite(
    const DenseReductionProblem& problem, DenseReductionSite site, bool maximum,
    uint64_t original_reduction_size, const StridedSpan& representative_input, const Stream& stream,
    uint64_t workspace_budget_bytes, ReductionTuningDepth tuning_depth, uint64_t input_alignment_bytes) {
    const DataType input_dtype = representative_input.getDataType();
    if (input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2)
        throw std::invalid_argument("FP8 ARG autotuning requires FP8 input");
    const uint64_t alignment = siteAlignment(input_alignment_bytes);
    if (problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.input_state != DenseReductionInputState::OriginalInput ||
        problem.runs.at(site.run_index).kind != DenseReductionRunKind::OriginalReduction ||
        original_reduction_size == 0)
        throw std::invalid_argument("FP8 ARG autotuning requires an original FP8 ARG RK site");
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const Shape shape = shapeAtSite(problem, site);
    if (!representative_input.isDenseContiguous() ||
        representative_input.getTotalNumElements() != checkedMul(checkedMul(shape.outer, shape.r), shape.k) ||
        representative_input.getPlacement().getDeviceNum() != stream.getGpuNum())
        throw std::invalid_argument("FP8 ARG autotune input must match contiguous selected-site geometry and stream");
    const bool final_site = !hasDenseReductionSites(advanceDenseReductionProblem(problem, site, 1));

    // The inventory is the VALUE MIN/MAX one (identity transform): the ARG twins share its ownership and occupancy.
    auto semantics = makeValueReductionStageSemantics(maximum ? ReductionOp::Max : ReductionOp::Min,
                                                      DensePassRole::Complete, shape.r);
    semantics.combine = maximum ? ReductionStageCombine::Maximum : ReductionStageCombine::Minimum;
    semantics.input_transform = ReductionStageInputTransform::Identity;
    const auto device = deviceIdentity(stream);
    const auto runtime = enumerateRetainedRuntimeK(
        {shape.outer, shape.r, shape.k, device.sm_count},
        [&](const PhysicalKernel& kernel) { return runtimeKResidency(input_dtype, semantics, kernel); });
    const BoundSelectionRequest request{shape.outer, shape.r, shape.k, device.sm_count, input_dtype, DataType::FP32,
                                        semantics, std::numeric_limits<uint64_t>::max(), 0};
    std::vector<RetainedFp8ValueSource> sources;
    for (const auto& entry : enumerateBoundInventory(request, runtime)) sources.push_back({entry.kernel, entry.local_rank});
    auto generated = retainedFp8ArgCandidates(sources, input_dtype, shape.outer, shape.r, shape.k);
    // And only the passes that read the input's alignment (the rotated passes read any address, at every R).
    std::erase_if(generated, [&](const RetainedFp8ArgCandidate& c) {
        return retainedFp8ArgScratchBytes(c.plan) > workspace_budget_bytes ||
               retainedFp8ArgPassInputAlignment(c.plan) > alignment;
    });
    RKSelection selection;
    if (generated.empty()) return selection;  // candidate_count 0: no ARG pass

    // Rank: the model's features of each ARG pass at its local rank, with the ARG aggregate of this pass's outputs.
    namespace G = Fp8RKPolicy;
    const G::RankerSemantics typed{input_dtype == DataType::FP8_E4M3 ? G::Fp8Format::E4M3 : G::Fp8Format::E5M2,
                                   maximum ? G::Combine::Max : G::Combine::Min, G::InputTransform::Identity};
    const bool produce_value = !final_site || problem.final_output.produce_value;
    const bool produce_index = !final_site || problem.final_output.produce_index;
    const G::RankerAggregate aggregate = produce_value ? G::RankerAggregate::ArgValueIndex : G::RankerAggregate::ArgIndex;
    std::vector<std::pair<float, size_t>> scored;
    for (size_t i = 0; i < generated.size(); ++i) {
        const auto features = G::Detail::rankerFeatures(rankerPlanForKernel(generated[i].identity, generated[i].local_rank),
                                                        shape.outer, shape.r, shape.k, typed, generated[i].local_rank,
                                                        aggregate);
        scored.emplace_back(Fp8GeometryRankerModel::margin(features), i);
    }
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    const ReductionTuningPlan tuning = reductionTuningPlan(tuning_depth);
    if (scored.size() > tuning.candidates) scored.resize(tuning.candidates);

    // Outputs exactly as the stamped pass writes them: the requested outputs on the final site, else the FP32 value +
    // carried index intermediate. One UINT8 output slot holds the value, then the index (at nextSliceOffset: 256-byte aligned, 128 bytes past the value).
    const DataType value_dtype = final_site ? problem.final_output.value_dtype : DataType::FP32;
    const DataType index_dtype = final_site ? problem.final_output.index_dtype : problem.carried_arg_index_dtype;
    const std::vector<uint64_t> output_dims{shape.outer, shape.k};
    const uint64_t value_bytes = produce_value ? TensorDescriptor(value_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t index_offset = ReductionAutotuning::nextSliceOffset(value_bytes);
    const uint64_t index_bytes = produce_index ? TensorDescriptor(index_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t domain_stride = denseArgOriginalIndexStride(problem.runs[site.run_index]).value_or(0);
    const ReductionGeometry geometry = ArgReducer::analyzeDenseGeometry({shape.outer, shape.r, shape.k}, {1});
    std::vector<std::shared_ptr<const Fp8ArgPhysicalPlan>> plans;
    std::vector<ReductionAutotuning::Candidate> candidates;
    for (const auto& [margin, index] : scored) {
        auto plan = std::make_shared<const Fp8ArgPhysicalPlan>(generated[index].plan);
        ReductionAutotuning::Candidate candidate;
        candidate.name = std::string(familyName(plan->family)) + "_arg#" + std::to_string(candidates.size());
        candidate.scratch_bytes = retainedFp8ArgScratchBytes(*plan);
        candidate.launch = [=](const StridedSpan& input, const StridedSpan& output, const StridedSpan& scratch, Stream& run_stream) {
            StridedSpan site_input = input;
            site_input.reshape({shape.outer, shape.r, shape.k});
            std::optional<StridedSpan> value, index;
            if (produce_value) value = output.bytesAs(0, value_dtype, output_dims);
            if (produce_index) index = output.bytesAs(index_offset, index_dtype, output_dims);
            uint32_t* packed = plan->packed_partials ? static_cast<uint32_t*>(scratch.getMemPtr<void>()) : nullptr;
            launchRetainedFp8ArgPlan(*plan, site_input, value ? &*value : nullptr, index ? &*index : nullptr,
                                     geometry, packed, maximum, domain_stride, run_stream);
        };
        candidates.push_back(std::move(candidate));
        plans.push_back(std::move(plan));
    }
    Stream tuning_stream = stream;
    const auto result = [&] {
        ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
        ReductionAutotuning::Options options;
        options.measurement_count = tuning.rounds;
        options.trim_count = tuning.trimmed;  // the shared contest policy otherwise (as above)
        return ReductionAutotuning::autotune(candidates, representative_input,
                                             TensorDescriptor(DataType::UINT8, {index_offset + index_bytes}),
                                             tuning_stream, options);
    }();
    if (result.winner_index >= plans.size())
        throw std::logic_error("FP8 ARG autotuner returned an out-of-range winner");
    selection.physical = retainedFp8ArgPhysicalIdentity(*plans[result.winner_index]);
    selection.winner_index = result.winner_index;
    selection.winner_name = result.winner_name;
    selection.candidate_count = candidates.size();
    return selection;
}

namespace {

// Validation stays outside getOrTune: a cache hit must not make a malformed
// descriptor, unsupported input dtype, or invalid reduction site acceptable.
Shape validateDescriptorSelection(
    const DenseReductionProblem& problem, DenseReductionSite site, DataType input_dtype,
    uint64_t original_reduction_size, float output_scale) {
    if (input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2)
        throw std::invalid_argument("FP8 RK cached selection requires FP8 input");
    if (problem.input_state != DenseReductionInputState::OriginalInput ||
        problem.aggregate_kind != DenseReductionAggregateKind::Value ||
        !problem.final_output.produce_value || problem.final_output.produce_index ||
        original_reduction_size == 0 || !std::isfinite(output_scale))
        throw std::invalid_argument("Invalid FP8 RK cached selection semantics");
    return shapeAtSite(problem, site);
}

RKSelection selectCachedDescriptorImpl(
    const DenseReductionProblem& problem, DenseReductionSite site, ReductionOp operation,
    uint64_t original_reduction_size, float output_scale, DataType input_dtype,
    const StridedSpan* representative_input, const Stream& stream, uint64_t workspace_budget_bytes,
    ReductionTuningDepth tuning_depth, uint64_t input_alignment_bytes) {
    const Shape shape = validateDescriptorSelection(
        problem, site, input_dtype, original_reduction_size, output_scale);
    const uint64_t alignment = siteAlignment(input_alignment_bytes);
    if (representative_input != nullptr &&
        (!representative_input->isInitialized() ||
         !representative_input->isDenseContiguous() ||
         representative_input->getDataType() != input_dtype ||
         representative_input->getTotalNumElements() != checkedMul(checkedMul(shape.outer, shape.r), shape.k) ||
         representative_input->getPlacement().getMemDevice() != TensorPlacement::MemDevices::GPU ||
         representative_input->getPlacement().getDeviceNum() != stream.getGpuNum())) {
        throw std::invalid_argument("FP8 RK cached selection requires matching contiguous input and GPU stream");
    }

    ScopedGpu scoped_gpu(stream.getGpuNum());
    const auto identity = deviceIdentity(stream);
    // The one dense reduction selection cache; it applies the depth rule (a deeper selection, completed or in flight,
    // answers a shallower request).
    auto key = toDenseReductionSelectionKey(makeRKDescriptorSelectionKey(
        problem, site, operation, original_reduction_size, output_scale, input_dtype,
        identity, workspace_budget_bytes, tuning_depth));
    key.site.push_back(alignment);  // as FP32's

    // The cache is the allocation boundary. Only a single-flight miss leader
    // constructs benchmark input. Hits need neither Tensor nor continuation.
    // Representative allocation/initialization/release are serialized with
    // measurement contests; FP32 dependency selection is deliberately outside
    // that lock, and the lock is not held across the whole cache callback.
    return denseReductionSelectionCache().getOrTune<RKSelection>(key, [&] {
        if (representative_input != nullptr) {
            return autotuneIsolatedRKSite(problem, site, operation, original_reduction_size,
                                          output_scale, *representative_input, stream,
                                          workspace_budget_bytes, tuning_depth, alignment);
        }
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        Tensor temporary;
        try {
            {
                ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
                temporary = Tensor(placement, TensorDescriptor(input_dtype,
                                                                {shape.outer, shape.r, shape.k}));
                Stream tuning_stream = stream;
                temporary.fill(1.0f, tuning_stream);
                tuning_stream.synchronize();
            }
            // The official FP32 dependencies are selected outside the timing
            // lock. Autotuning then acquires the same per-device lock for its
            // interleaved contest and synchronizes before returning.
            auto winner = autotuneIsolatedRKSite(problem, site, operation,
                                                 original_reduction_size, output_scale,
                                                 temporary, stream, workspace_budget_bytes, tuning_depth, alignment);
            {
                ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
                temporary.dropReference();
            }
            return winner;
        } catch (...) {
            // cudaFree may synchronize. Even error-path teardown must not
            // interfere with another selection's GPU timing contest.
            ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
            temporary.dropReference();
            throw;
        }
    });
}

} // namespace

RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem, DenseReductionSite site, ReductionOp operation,
    uint64_t original_reduction_size, float output_scale, DataType input_dtype,
    const Stream& stream, uint64_t workspace_budget_bytes, ReductionTuningDepth tuning_depth,
    uint64_t input_alignment_bytes) {
    return selectCachedDescriptorImpl(problem, site, operation, original_reduction_size,
                                      output_scale, input_dtype, nullptr, stream, workspace_budget_bytes,
                                      tuning_depth, input_alignment_bytes);
}

RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem, DenseReductionSite site, ReductionOp operation,
    uint64_t original_reduction_size, float output_scale, const StridedSpan& representative_input,
    const Stream& stream, uint64_t workspace_budget_bytes, ReductionTuningDepth tuning_depth,
    uint64_t input_alignment_bytes) {
    if (!representative_input.isInitialized())
        throw std::invalid_argument("FP8 RK cached selection requires initialized input");
    return selectCachedDescriptorImpl(problem, site, operation, original_reduction_size,
                                      output_scale, representative_input.getDataType(),
                                      &representative_input, stream, workspace_budget_bytes, tuning_depth,
                                      input_alignment_bytes);
}

RKSelection selectCachedArgRKSite(
    const DenseReductionProblem& problem, DenseReductionSite site, bool maximum,
    uint64_t original_reduction_size, DataType input_dtype, const Stream& stream,
    uint64_t workspace_budget_bytes, ReductionTuningDepth tuning_depth, uint64_t input_alignment_bytes) {
    if (input_dtype != DataType::FP8_E4M3 && input_dtype != DataType::FP8_E5M2)
        throw std::invalid_argument("FP8 ARG cached selection requires FP8 input");
    const uint64_t alignment = siteAlignment(input_alignment_bytes);
    if (problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.input_state != DenseReductionInputState::OriginalInput || original_reduction_size == 0)
        throw std::invalid_argument("Invalid FP8 ARG cached selection semantics");
    const Shape shape = shapeAtSite(problem, site);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const auto identity = deviceIdentity(stream);
    const ReductionOp operation = maximum ? ReductionOp::Max : ReductionOp::Min;
    auto key = toDenseReductionSelectionKey(makeRKDescriptorSelectionKey(
        problem, site, operation, original_reduction_size, 1.0f, input_dtype, identity, workspace_budget_bytes,
        tuning_depth));
    key.site.push_back(alignment);
    return denseReductionSelectionCache().getOrTune<RKSelection>(key, [&] {
        // As the VALUE miss path: a temporary representative input, allocated and released under the measurement
        // lock; the contest itself takes the lock separately.
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        Tensor temporary;
        try {
            {
                ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
                temporary = Tensor(placement, TensorDescriptor(input_dtype, {shape.outer, shape.r, shape.k}));
                Stream fill_stream = stream;
                temporary.fill(1.0f, fill_stream);
                fill_stream.synchronize();
            }
            auto winner = autotuneArgRKSite(problem, site, maximum, original_reduction_size, temporary, stream,
                                            workspace_budget_bytes, tuning_depth, alignment);
            {
                ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
                temporary.dropReference();
            }
            return winner;
        } catch (...) {
            ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
            temporary.dropReference();
            throw;
        }
    });
}

} // namespace ThorImplementation::ReductionInternal::RK::Fp8
