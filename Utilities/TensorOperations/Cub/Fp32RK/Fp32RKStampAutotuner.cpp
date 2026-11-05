#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKStampAutotuner.h"

#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/Fp32CarriedArgContestInput.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKArgCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp32RK/Fp32RKCarriedArgCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKDescriptorSelection.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKIsolatedSite.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"
#include "Utilities/TensorOperations/Cub/ReductionAutotuner.h"
#include "Utilities/TensorOperations/Cub/ReductionGpuMeasurementLock.h"
#include "Utilities/Common/ScopedGpu.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp32RK {
namespace {
using Fp8RK::PhysicalFamily;
using Fp8RK::PhysicalKernel;

struct Shape {
    uint64_t outer = 1, r = 0, k = 1;
};
uint64_t checkedMul(uint64_t a, uint64_t b) {
    if (b && a > UINT64_MAX / b) throw std::overflow_error("FP32 RK stamp shape overflow");
    return a * b;
}
Shape shapeAtSite(const DenseReductionProblem& problem, DenseReductionSite site) {
    if (classifyDenseReducerFamily(problem, site) != DenseReducerFamily::RK)
        throw std::invalid_argument("FP32 RK selection needs a middle-axis RK site");
    Shape shape;
    for (size_t i = 0; i < problem.runs.size(); ++i) {
        if (i < site.run_index) shape.outer = checkedMul(shape.outer, problem.runs[i].extent);
        else if (i == site.run_index) shape.r = problem.runs[i].extent;
        else shape.k = checkedMul(shape.k, problem.runs[i].extent);
    }
    return shape;
}
// The FP32 VALUE domain, checked before the cache so a hit cannot make an invalid request acceptable: an original FP32
// site, or (1207) an intermediate site, whose input is FP32 partials of any dtype's earlier passes.
Shape validateValueSite(const DenseReductionProblem& problem, DenseReductionSite site, uint64_t original_reduction_size,
                        float output_scale) {
    if (problem.aggregate_kind != DenseReductionAggregateKind::Value || !problem.final_output.produce_value ||
        problem.final_output.produce_index || original_reduction_size == 0 || !std::isfinite(output_scale))
        throw std::invalid_argument("Invalid FP32 RK VALUE selection semantics");
    return shapeAtSite(problem, site);
}
// The FP32 ARG domain: an original ARG RK site.
Shape validateArgSite(const DenseReductionProblem& problem, DenseReductionSite site, uint64_t original_reduction_size) {
    if (problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.input_state != DenseReductionInputState::OriginalInput ||
        problem.runs.at(site.run_index).kind != DenseReductionRunKind::OriginalReduction ||
        original_reduction_size == 0)
        throw std::invalid_argument("FP32 RK ARG selection requires an original FP32 ARG RK site");
    return shapeAtSite(problem, site);
}
// 1215: the carried FP32 ARG domain: an ARG RK site over an earlier pass's FP32 values and carried index (the run an
// original or an intermediate reduction).
Shape validateCarriedArgSite(const DenseReductionProblem& problem, DenseReductionSite site,
                             uint64_t original_reduction_size) {
    if (problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.input_state != DenseReductionInputState::IntermediateAggregate || original_reduction_size == 0 ||
        (problem.carried_arg_index_dtype != DataType::UINT32 && problem.carried_arg_index_dtype != DataType::UINT64))
        throw std::invalid_argument("FP32 RK carried ARG selection requires a carried-index FP32 ARG RK site");
    return shapeAtSite(problem, site);
}
const char* familyName(PhysicalFamily family) {
    switch (family) {
        case PhysicalFamily::OneWindow: return "fp32_one_window";
        case PhysicalFamily::DeepRSmallK: return "fp32_deep_r_small_k";
        case PhysicalFamily::AlignedK: return "fp32_aligned_k";
        case PhysicalFamily::RuntimeK: return "fp32_runtime_k";
    }
    throw std::logic_error("Unknown physical FP32 RK family");
}
// 1222: the input alignment a site's selection honours: the original input's (at most 16) at an original-input site;
// 16 at an intermediate (fresh, aligned allocations).
uint64_t siteAlignment(const DenseReductionProblem& problem, uint64_t input_alignment_bytes) {
    if (input_alignment_bytes == 0 || (input_alignment_bytes & (input_alignment_bytes - 1)) != 0)
        throw std::invalid_argument("FP32 RK input alignment must be a power of two");
    if (problem.input_state != DenseReductionInputState::OriginalInput) return 16;
    return std::min<uint64_t>(input_alignment_bytes, 16);
}
// Keeps the passes that run at `alignment` (fp32RKPassInputAlignment).
template <typename Entry, typename Kernel>
void keepAlignedPasses(std::vector<Entry>& entries, uint64_t alignment, Kernel kernel) {
    std::erase_if(entries, [&](const Entry& entry) { return fp32RKPassInputAlignment(kernel(entry)) > alignment; });
    if (entries.empty())
        throw std::runtime_error("No retained FP32 RK pass runs at this input's " + std::to_string(alignment) +
                                 "-byte alignment");
}
}  // namespace

Fp32RKSelection autotuneValueSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                  CubReductionOp operation, uint64_t original_reduction_size, float output_scale,
                                  const Tensor& representative_input, const Stream& stream,
                                  uint64_t workspace_budget_bytes, CubReductionTuningDepth tuning_depth,
                                  uint64_t input_alignment_bytes) {
    if (representative_input.getDataType() != DataType::FP32)
        throw std::invalid_argument("FP32 RK autotuning requires FP32 input");
    const Shape shape = validateValueSite(problem, site, original_reduction_size, output_scale);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    if (!representative_input.isDenseContiguous() ||
        representative_input.getTotalNumElements() != checkedMul(checkedMul(shape.outer, shape.r), shape.k) ||
        representative_input.getPlacement().getDeviceNum() != stream.getGpuNum())
        throw std::invalid_argument("FP32 RK autotune input must match contiguous selected-site geometry and stream");
    // The pass role: Complete / First on original input, Final / Intermediate on an intermediate (1207: no input
    // transform there; the finalizer on the last pass only).
    const auto semantics = makeValueReductionStageSemantics(
        operation, deriveDensePassRole(problem, advanceDenseReductionProblem(problem, site, 1)),
        original_reduction_size);
    const auto device = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));

    const Fp32RKSite candidate_site{shape.outer, shape.r, shape.k, device.sm_count, semantics, workspace_budget_bytes};
    auto pool = fp32RKCandidatePool(candidate_site, [&](const PhysicalKernel& kernel) {
        return fp32RKCompiledResidency(kernel, shape.k, semantics);
    });
    keepAlignedPasses(pool, siteAlignment(problem, input_alignment_bytes),
                      [](const Fp32RKPoolEntry& entry) { return entry.kernel; });
    const auto ranked = fp32RKRankedKernels(pool, candidate_site, fp32RKShortlistLength(tuning_depth));

    // The production FP32 policy selects each staged candidate's continuation (outside the GPU lock), as for FP8. 1220:
    // the context carries the original semantics and the tuning depth, which the retained FP32 routing requires.
    const DenseProductionPhysicalSelectionContext fp32_context(DataType::FP32, denseRKValueOperation(operation),
                                                               stream, original_reduction_size, output_scale,
                                                               operation, workspace_budget_bytes, tuning_depth);
    std::vector<Fp8RK::SelectedIsolatedBenchmark> executables;
    std::vector<ReductionAutotuning::Candidate> candidates;
    for (const PhysicalKernel& kernel : ranked) {
        Fp8RK::IsolatedRKSite isolated = Fp8RK::planIsolatedRKSite(problem, site, kernel, semantics);
        try {
            auto executable = Fp8RK::makeSelectedIsolatedBenchmark(problem, site, std::move(isolated), kernel,
                                                                   fp32_context.context(), output_scale,
                                                                   workspace_budget_bytes, stream);
            candidates.push_back(executable.asAutotuneCandidate(std::string(familyName(kernel.family)) + "#" +
                                                                std::to_string(candidates.size())));
            executables.push_back(std::move(executable));
        } catch (const Fp8RK::UnavailableIsolatedCandidate&) {
            continue;
        }
    }
    if (candidates.empty())
        throw std::runtime_error("No legal retained FP32 RK candidate under the workspace budget");

    const TensorDescriptor output_descriptor(executables.front().site.site_output_dtype, {shape.outer, shape.k});
    const Fp8RK::Fp8RKTuningPlan tuning = Fp8RK::fp8RKTuningPlan(tuning_depth);
    Stream tuning_stream = stream;
    const auto result = [&] {
        ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
        ReductionAutotuning::Options options;
        options.measurement_count = tuning.rounds;
        options.trim_count = tuning.trimmed;
        return ReductionAutotuning::autotune(candidates, representative_input, output_descriptor, tuning_stream,
                                             options);
    }();
    if (result.winner_index >= executables.size())
        throw std::logic_error("FP32 RK autotuner returned an out-of-range winner");
    Fp32RKSelection selection;
    selection.physical = executables[result.winner_index].first_physical;
    selection.winner_index = result.winner_index;
    selection.winner_name = result.winner_name;
    selection.candidate_count = candidates.size();
    return selection;
}

Fp32RKSelection autotuneArgSite(const DenseReductionProblem& problem, DenseReductionSite site, bool maximum,
                                uint64_t original_reduction_size, const Tensor& representative_input,
                                const Stream& stream, uint64_t workspace_budget_bytes,
                                CubReductionTuningDepth tuning_depth, uint64_t input_alignment_bytes) {
    if (representative_input.getDataType() != DataType::FP32)
        throw std::invalid_argument("FP32 RK ARG autotuning requires FP32 input");
    const Shape shape = validateArgSite(problem, site, original_reduction_size);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    if (!representative_input.isDenseContiguous() ||
        representative_input.getTotalNumElements() != checkedMul(checkedMul(shape.outer, shape.r), shape.k) ||
        representative_input.getPlacement().getDeviceNum() != stream.getGpuNum())
        throw std::invalid_argument(
            "FP32 RK ARG autotune input must match contiguous selected-site geometry and stream");
    const bool final_site = !hasDenseReductionSites(advanceDenseReductionProblem(problem, site, 1));

    // The VALUE MIN / MAX pool (identity transform): the ARG twins share its identities.
    auto semantics = makeValueReductionStageSemantics(maximum ? CubReductionOp::Max : CubReductionOp::Min,
                                                      DensePassRole::Complete, shape.r);
    semantics.combine = maximum ? CubReductionStageCombine::Maximum : CubReductionStageCombine::Minimum;
    semantics.input_transform = CubReductionStageInputTransform::Identity;
    const auto device = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    const Fp32RKSite candidate_site{shape.outer, shape.r, shape.k, device.sm_count, semantics,
                                    std::numeric_limits<uint64_t>::max()};
    const auto value_pool = fp32RKCandidatePool(candidate_site, [&](const PhysicalKernel& kernel) {
        return fp32RKCompiledResidency(kernel, shape.k, semantics);
    });
    auto generated = fp32RKArgCandidates(value_pool, shape.outer, shape.r, shape.k, workspace_budget_bytes);
    Fp32RKSelection selection;
    if (generated.empty()) return selection;  // candidate_count 0: no FP32 ARG pass
    keepAlignedPasses(generated, siteAlignment(problem, input_alignment_bytes),
                      [](const Fp32RKArgCandidate& candidate) { return candidate.identity; });

    // Rank with the ARG aggregate of this pass's outputs (1 index, 2 value and index).
    const bool produce_value = !final_site || problem.final_output.produce_value;
    const bool produce_index = !final_site || problem.final_output.produce_index;
    const uint32_t aggregate = produce_value ? 2U : 1U;
    const auto ranked = fp32RKRankedKernels(fp32RKArgRankerPool(generated), candidate_site,
                                            fp32RKShortlistLength(tuning_depth), aggregate);

    // Outputs exactly as the stamped pass writes them: the requested outputs on the final site, else the FP32 value +
    // carried index intermediate. One UINT8 output slot holds the value then (256-byte aligned) the index.
    const DataType value_dtype = final_site ? problem.final_output.value_dtype : DataType::FP32;
    const DataType index_dtype = final_site ? problem.final_output.index_dtype : problem.carried_arg_index_dtype;
    const std::vector<uint64_t> output_dims{shape.outer, shape.k};
    const uint64_t value_bytes = produce_value ? TensorDescriptor(value_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t index_offset = (value_bytes + 255) & ~uint64_t{255};
    const uint64_t index_bytes = produce_index ? TensorDescriptor(index_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t domain_stride = denseArgOriginalIndexStride(problem.runs[site.run_index]).value_or(0);
    const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({shape.outer, shape.r, shape.k}, {1});
    std::vector<PhysicalKernel> passes;
    std::vector<ReductionAutotuning::Candidate> candidates;
    for (const PhysicalKernel& identity : ranked) {
        ReductionAutotuning::Candidate candidate;
        candidate.name = std::string(familyName(identity.family)) + "_arg#" + std::to_string(candidates.size());
        candidate.scratch_bytes = fp32RKArgScratchBytes(identity, shape.outer, shape.r, shape.k);
        const bool staged = candidate.scratch_bytes != 0;
        candidate.launch = [=](const Tensor& input, Tensor& output, Tensor& scratch, Stream& run_stream) {
            Tensor site_input = input;
            site_input.reshape({shape.outer, shape.r, shape.k});
            std::optional<Tensor> value, index;
            if (produce_value) value = output.aliasByteStorageAs(value_dtype, output_dims, 0);
            if (produce_index) index = output.aliasByteStorageAs(index_dtype, output_dims, index_offset);
            uint64_t* packed = staged ? static_cast<uint64_t*>(scratch.getMemPtr<void>()) : nullptr;
            launchFp32RKArgPass(identity, site_input, value ? &*value : nullptr, index ? &*index : nullptr, geometry,
                                packed, maximum, domain_stride, run_stream);
        };
        candidates.push_back(std::move(candidate));
        passes.push_back(identity);
    }
    const Fp8RK::Fp8RKTuningPlan tuning = Fp8RK::fp8RKTuningPlan(tuning_depth);
    Stream tuning_stream = stream;
    const auto result = [&] {
        ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
        ReductionAutotuning::Options options;
        options.measurement_count = tuning.rounds;
        options.trim_count = tuning.trimmed;
        return ReductionAutotuning::autotune(candidates, representative_input,
                                             TensorDescriptor(DataType::UINT8, {index_offset + index_bytes}),
                                             tuning_stream, options);
    }();
    if (result.winner_index >= passes.size())
        throw std::logic_error("FP32 RK ARG autotuner returned an out-of-range winner");
    selection.physical = passes[result.winner_index];
    selection.winner_index = result.winner_index;
    selection.winner_name = result.winner_name;
    selection.candidate_count = candidates.size();
    return selection;
}

Fp32RKSelection selectCachedArgSite(const DenseReductionProblem& problem, DenseReductionSite site, bool maximum,
                                    uint64_t original_reduction_size, const Stream& stream,
                                    uint64_t workspace_budget_bytes, CubReductionTuningDepth tuning_depth,
                                    uint64_t input_alignment_bytes) {
    const Shape shape = validateArgSite(problem, site, original_reduction_size);
    const uint64_t alignment = siteAlignment(problem, input_alignment_bytes);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const auto identity = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    DenseReductionSelectionKey key = Fp8RK::toDenseReductionSelectionKey(Fp8RK::makeFp8RKDescriptorSelectionKey(
        problem, site, maximum ? CubReductionOp::Max : CubReductionOp::Min, original_reduction_size, 1.0f,
        DataType::FP32, identity, workspace_budget_bytes, tuning_depth));
    key.kind = DenseReductionSelectionKind::Fp32RK;
    key.site.push_back(alignment);  // 1222
    return denseReductionSelectionCache().getOrTune<Fp32RKSelection>(key, [&] {
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        Tensor temporary;
        try {
            {
                ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
                temporary = Tensor(placement, TensorDescriptor(DataType::FP32, {shape.outer, shape.r, shape.k}));
                Stream fill_stream = stream;
                temporary.fill(1.0f, fill_stream);
                fill_stream.synchronize();
            }
            auto winner = autotuneArgSite(problem, site, maximum, original_reduction_size, temporary, stream,
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

namespace {
// The carried contest at a validated site over a contest input (Fp32CarriedArgContestInput.h) of its geometry.
Fp32RKSelection tuneCarriedArgContest(const DenseReductionProblem& problem, DenseReductionSite site, const Shape& shape,
                                      bool maximum, const Tensor& contest_input, const Stream& stream,
                                      uint64_t workspace_budget_bytes, CubReductionTuningDepth tuning_depth) {
    const uint64_t elements = checkedMul(checkedMul(shape.outer, shape.r), shape.k);
    const bool final_site = !hasDenseReductionSites(advanceDenseReductionProblem(problem, site, 1));
    const auto generated = fp32RKCarriedArgCandidates(shape.outer, shape.r, shape.k, workspace_budget_bytes);
    Fp32RKSelection selection;
    if (generated.empty()) return selection;  // candidate_count 0: no carried FP32 ARG pass

    // Rank as the carried sweep cells: the MIN / MAX semantics and the carried aggregate (3 index, 4 value and index).
    auto semantics = makeValueReductionStageSemantics(maximum ? CubReductionOp::Max : CubReductionOp::Min,
                                                      DensePassRole::Complete, shape.r);
    semantics.combine = maximum ? CubReductionStageCombine::Maximum : CubReductionStageCombine::Minimum;
    semantics.input_transform = CubReductionStageInputTransform::Identity;
    const auto device = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    const Fp32RKSite candidate_site{shape.outer, shape.r, shape.k, device.sm_count, semantics,
                                    std::numeric_limits<uint64_t>::max()};
    const bool produce_value = !final_site || problem.final_output.produce_value;
    const bool produce_index = !final_site || problem.final_output.produce_index;
    const uint32_t aggregate = produce_value ? 4U : 3U;
    const auto ranked = fp32RKRankedKernels(fp32RKArgRankerPool(generated), candidate_site,
                                            fp32RKShortlistLength(tuning_depth), aggregate);

    // Outputs as the stamped pass writes them (autotuneArgSite's slot); the composed index needs the run's
    // original-domain stride (0 for an intermediate run, whose carried index is already composed).
    const DataType value_dtype = final_site ? problem.final_output.value_dtype : DataType::FP32;
    const DataType index_dtype = final_site ? problem.final_output.index_dtype : problem.carried_arg_index_dtype;
    const std::vector<uint64_t> output_dims{shape.outer, shape.k};
    const uint64_t value_bytes = produce_value ? TensorDescriptor(value_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t index_offset = (value_bytes + 255) & ~uint64_t{255};
    const uint64_t index_bytes = produce_index ? TensorDescriptor(index_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t row_stride = denseArgOriginalIndexStride(problem.runs[site.run_index]).value_or(0);
    const CubReductionGeometry geometry = CubArgReduction::analyzeDenseGeometry({shape.outer, shape.r, shape.k}, {1});
    const Fp32CarriedArgContestInput layout = fp32CarriedArgContestInput(elements, problem.carried_arg_index_dtype);
    const std::vector<uint64_t> input_dims{shape.outer, shape.r, shape.k};
    std::vector<PhysicalKernel> passes;
    std::vector<ReductionAutotuning::Candidate> candidates;
    for (const PhysicalKernel& identity : ranked) {
        ReductionAutotuning::Candidate candidate;
        candidate.name = std::string(familyName(identity.family)) + "_carried_arg#" + std::to_string(candidates.size());
        candidate.scratch_bytes = fp32RKCarriedArgScratchBytes(identity, shape.outer, shape.r, shape.k);
        const bool staged = candidate.scratch_bytes != 0;
        candidate.launch = [=](const Tensor& input, Tensor& output, Tensor& scratch, Stream& run_stream) {
            const Tensor values = layout.values(input, input_dims);
            const Tensor carried = layout.carriedIndex(input, input_dims);
            std::optional<Tensor> value, index;
            if (produce_value) value = output.aliasByteStorageAs(value_dtype, output_dims, 0);
            if (produce_index) index = output.aliasByteStorageAs(index_dtype, output_dims, index_offset);
            launchFp32RKCarriedArgPass(identity, values, carried, value ? &*value : nullptr,
                                       index ? &*index : nullptr, geometry,
                                       staged ? scratch.getMemPtr<void>() : nullptr, maximum, row_stride, run_stream);
        };
        candidates.push_back(std::move(candidate));
        passes.push_back(identity);
    }
    const Fp8RK::Fp8RKTuningPlan tuning = Fp8RK::fp8RKTuningPlan(tuning_depth);
    Stream tuning_stream = stream;
    const auto result = [&] {
        ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
        ReductionAutotuning::Options options;
        options.measurement_count = tuning.rounds;
        options.trim_count = tuning.trimmed;
        return ReductionAutotuning::autotune(candidates, contest_input,
                                             TensorDescriptor(DataType::UINT8, {index_offset + index_bytes}),
                                             tuning_stream, options);
    }();
    if (result.winner_index >= passes.size())
        throw std::logic_error("FP32 RK carried ARG autotuner returned an out-of-range winner");
    selection.physical = passes[result.winner_index];
    selection.winner_index = result.winner_index;
    selection.winner_name = result.winner_name;
    selection.candidate_count = candidates.size();
    return selection;
}

// A contest input made and released under the measurement lock around `tune`.
template <typename Make>
Fp32RKSelection withCarriedContestInput(const Stream& stream, Make make,
                                        const std::function<Fp32RKSelection(const Tensor&)>& tune) {
    Tensor contest_input;
    const auto release = [&] {
        ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
        contest_input.dropReference();
    };
    try {
        {
            ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
            contest_input = make();
        }
        Fp32RKSelection selection = tune(contest_input);
        release();
        return selection;
    } catch (...) {
        release();
        throw;
    }
}
}  // namespace

Fp32RKSelection autotuneCarriedArgSite(const DenseReductionProblem& problem, DenseReductionSite site, bool maximum,
                                       uint64_t original_reduction_size, const Tensor& representative_values,
                                       const Tensor& representative_carried_index, const Stream& stream,
                                       uint64_t workspace_budget_bytes, CubReductionTuningDepth tuning_depth) {
    const Shape shape = validateCarriedArgSite(problem, site, original_reduction_size);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const uint64_t elements = checkedMul(checkedMul(shape.outer, shape.r), shape.k);
    const auto on_site = [&](const Tensor& tensor, DataType dtype) {
        return tensor.getDataType() == dtype && tensor.isDenseContiguous() &&
               tensor.getTotalNumElements() == elements && tensor.getPlacement().getDeviceNum() == stream.getGpuNum();
    };
    if (!on_site(representative_values, DataType::FP32) ||
        !on_site(representative_carried_index, problem.carried_arg_index_dtype))
        throw std::invalid_argument(
            "FP32 RK carried ARG autotune inputs must be contiguous FP32 values and the site's carried index dtype at "
            "the site's geometry on the stream's GPU");
    const Fp32CarriedArgContestInput layout = fp32CarriedArgContestInput(elements, problem.carried_arg_index_dtype);
    return withCarriedContestInput(
        stream,
        [&] { return makeFp32CarriedArgContestInput(layout, stream, &representative_values,
                                                    &representative_carried_index); },
        [&](const Tensor& contest_input) {
            return tuneCarriedArgContest(problem, site, shape, maximum, contest_input, stream, workspace_budget_bytes,
                                         tuning_depth);
        });
}

Fp32RKSelection selectCachedCarriedArgSite(const DenseReductionProblem& problem, DenseReductionSite site, bool maximum,
                                           uint64_t original_reduction_size, const Stream& stream,
                                           uint64_t workspace_budget_bytes, CubReductionTuningDepth tuning_depth) {
    const Shape shape = validateCarriedArgSite(problem, site, original_reduction_size);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const auto identity = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    // FP8 RK's descriptor: the input state, carried index dtype and run kinds / strides separate it from original ARG.
    DenseReductionSelectionKey key = Fp8RK::toDenseReductionSelectionKey(Fp8RK::makeFp8RKDescriptorSelectionKey(
        problem, site, maximum ? CubReductionOp::Max : CubReductionOp::Min, original_reduction_size, 1.0f,
        DataType::FP32, identity, workspace_budget_bytes, tuning_depth));
    key.kind = DenseReductionSelectionKind::Fp32RK;
    return denseReductionSelectionCache().getOrTune<Fp32RKSelection>(key, [&] {
        // A representative contest input: values 1.0, carried index 0.
        const Fp32CarriedArgContestInput layout = fp32CarriedArgContestInput(
            checkedMul(checkedMul(shape.outer, shape.r), shape.k), problem.carried_arg_index_dtype);
        return withCarriedContestInput(
            stream, [&] { return makeFp32CarriedArgContestInput(layout, stream); },
            [&](const Tensor& contest_input) {
                return tuneCarriedArgContest(problem, site, shape, maximum, contest_input, stream,
                                             workspace_budget_bytes, tuning_depth);
            });
    });
}

Fp32RKSelection selectCachedValueSite(const DenseReductionProblem& problem, DenseReductionSite site,
                                      CubReductionOp operation, uint64_t original_reduction_size, float output_scale,
                                      const Stream& stream, uint64_t workspace_budget_bytes,
                                      CubReductionTuningDepth tuning_depth, uint64_t input_alignment_bytes) {
    const Shape shape = validateValueSite(problem, site, original_reduction_size, output_scale);
    const uint64_t alignment = siteAlignment(problem, input_alignment_bytes);
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const auto identity = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    // FP8 RK's descriptor (every field that can change the shortlist, continuation or winner), under kind Fp32RK.
    DenseReductionSelectionKey key = Fp8RK::toDenseReductionSelectionKey(Fp8RK::makeFp8RKDescriptorSelectionKey(
        problem, site, operation, original_reduction_size, output_scale, DataType::FP32, identity,
        workspace_budget_bytes, tuning_depth));
    key.kind = DenseReductionSelectionKind::Fp32RK;
    key.site.push_back(alignment);  // 1222
    return denseReductionSelectionCache().getOrTune<Fp32RKSelection>(key, [&] {
        // A temporary representative input, allocated and released under the measurement lock; the contest takes the
        // lock separately (as FP8 RK's miss path).
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        Tensor temporary;
        try {
            {
                ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
                temporary = Tensor(placement, TensorDescriptor(DataType::FP32, {shape.outer, shape.r, shape.k}));
                Stream fill_stream = stream;
                temporary.fill(1.0f, fill_stream);
                fill_stream.synchronize();
            }
            auto winner = autotuneValueSite(problem, site, operation, original_reduction_size, output_scale, temporary,
                                            stream, workspace_budget_bytes, tuning_depth, alignment);
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

}  // namespace ThorImplementation::CubReductionInternal::Fp32RK
