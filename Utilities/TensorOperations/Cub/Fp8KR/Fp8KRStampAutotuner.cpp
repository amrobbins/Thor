#include "Utilities/TensorOperations/Cub/Fp8KR/Fp8KRStampAutotuner.h"

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/DensePhysicalPassExecution.h"
#include "Utilities/TensorOperations/Cub/DenseReductionFamilies.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"
#include "Utilities/TensorOperations/Cub/ReductionAutotuner.h"
#include "Utilities/TensorOperations/Cub/ReductionGpuMeasurementLock.h"
#include "Utilities/Common/ScopedGpu.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp8KR {
namespace {

// 1171: R / KR selections live in the one dense reduction selection cache (kind Fp8KR). The site is every field that
// can change the shortlist, the measurement or the winner's legality: kind (VALUE / ARG), dtype, operation, the R / KR
// geometry, whether the pass is final and its role, and the outputs it writes (final) or carries (staged).
constexpr uint64_t KrValueSite = 1, KrArgSite = 2;

DenseReductionSelectionKey selectionKey(uint64_t kind, const DenseReductionProblem& problem, DenseReductionSite site,
                                        uint64_t operation, DataType input_dtype, const Stream& stream,
                                        CubReductionTuningDepth depth) {
    const Fp8KRSiteGeometry geometry = fp8KRSiteGeometry(problem, site);
    const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
    const bool final_pass = !hasDenseReductionSites(next);
    DenseReductionSelectionKey key;
    key.kind = DenseReductionSelectionKind::Fp8KR;
    key.device = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    key.tuning_depth = depth;
    key.site = {kind,
                static_cast<uint64_t>(input_dtype),
                operation,
                geometry.rows,
                geometry.row_bytes,
                static_cast<uint64_t>(final_pass),
                static_cast<uint64_t>(deriveDensePassRole(problem, next))};
    if (final_pass) {
        key.site.insert(key.site.end(), {static_cast<uint64_t>(problem.final_output.produce_value),
                                         static_cast<uint64_t>(problem.final_output.value_dtype),
                                         static_cast<uint64_t>(problem.final_output.produce_index),
                                         static_cast<uint64_t>(problem.final_output.index_dtype)});
    } else {
        key.site.push_back(static_cast<uint64_t>(problem.carried_arg_index_dtype));
    }
    return key;
}

// A miss allocates a representative input (1.0 everywhere) under the measurement lock, tunes, and releases it under
// the lock; the contest takes the lock itself. Hits allocate nothing.
Fp8KRSelection selectCached(const DenseReductionSelectionKey& key, uint64_t input_elements, DataType input_dtype,
                            const Stream& stream, const std::function<Fp8KRSelection(const Tensor&)>& tune) {
    return denseReductionSelectionCache().getOrTune<Fp8KRSelection>(key, [&] {
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        Tensor representative;
        try {
            {
                ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
                representative = Tensor(placement, TensorDescriptor(input_dtype, {input_elements}));
                Stream fill_stream = stream;
                representative.fill(1.0, fill_stream);
                fill_stream.synchronize();
            }
            Fp8KRSelection selection = tune(representative);
            {
                ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
                representative.dropReference();
            }
            return selection;
        } catch (...) {
            ScopedReductionGpuMeasurement cleanup(static_cast<int>(stream.getGpuNum()));
            representative.dropReference();
            throw;
        }
    });
}

// 1158: how many ranked shapes a depth times (the trained ranker's shortlist; without a trained model, the heuristic
// list's prefix at the RK depths).
size_t shortlistLength(CubReductionTuningDepth depth) {
    if (!fp8KRRankerTrained()) return Fp8RK::fp8RKTuningPlan(depth).candidates;
    switch (depth) {
        case CubReductionTuningDepth::Quick: return FP8_KR_SHORTLIST_DEPTHS[0];
        case CubReductionTuningDepth::Regular: return FP8_KR_SHORTLIST_DEPTHS[1];
        case CubReductionTuningDepth::Ultra: return FP8_KR_SHORTLIST_DEPTHS[2];
    }
    return FP8_KR_SHORTLIST_DEPTHS[1];
}

ReductionAutotuning::RunResult runContest(const std::vector<ReductionAutotuning::Candidate>& candidates,
                                          const Tensor& representative, const TensorDescriptor& output_descriptor,
                                          const Stream& stream, CubReductionTuningDepth depth) {
    const Fp8RK::Fp8RKTuningPlan tuning = Fp8RK::fp8RKTuningPlan(depth);
    ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
    ReductionAutotuning::Options options;
    options.measurement_count = tuning.rounds;
    options.trim_count = tuning.trimmed;
    // 1162: the launch counts and abandonment are the shared contest policy (ReductionAutotuning::Options).
    Stream tuning_stream = stream;
    return ReductionAutotuning::autotune(candidates, representative, output_descriptor, tuning_stream, options);
}

}  // namespace

Fp8KRSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                     DenseReductionSite site,
                                     CubReductionOp operation,
                                     uint64_t original_reduction_size,
                                     float output_scale,
                                     DataType input_dtype,
                                     uint32_t multiprocessors,
                                     const Stream& stream,
                                     CubReductionTuningDepth tuning_depth) {
    if (!isOriginalFp8KRSite(problem, site, input_dtype) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Value || !problem.final_output.produce_value ||
        problem.final_output.produce_index || original_reduction_size == 0)
        throw std::invalid_argument("Retained FP8 KR VALUE selection requires an original FP8 VALUE R/KR site");
    {
        // Semantics the retained kernels do not compile (none of the public operations today) keep the direct pass.
        const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
        if (!fp8KRValueSupports(makeValueReductionStageSemantics(operation, deriveDensePassRole(problem, next),
                                                                 original_reduction_size)))
            return {};
    }
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const Fp8KRSiteGeometry geometry = fp8KRSiteGeometry(problem, site);
    const DenseReductionSelectionKey key = selectionKey(KrValueSite, problem, site, static_cast<uint64_t>(operation),
                                                        input_dtype, stream, tuning_depth);

    return selectCached(key, geometry.rows * geometry.row_bytes, input_dtype, stream,
                        [&](const Tensor& representative) {
        const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
        const bool final_pass = !hasDenseReductionSites(next);
        const CubReductionStageSemantics semantics =
            makeValueReductionStageSemantics(operation, deriveDensePassRole(problem, next), original_reduction_size);
        const DataType output_dtype = final_pass ? problem.final_output.value_dtype : DataType::FP32;
        const float scale = final_pass ? output_scale : 1.0f;

        // 1148 (user, 2026-11-01): the retained shapes only; the direct CUB pass is no longer a candidate.
        // 1158: the ranked shortlist (the trained ranker over the pool, or the heuristic list's prefix).
        const std::vector<Fp8KRSegmentShape> order =
            fp8KRRankedShapes(geometry.rows, geometry.row_bytes, multiprocessors,
                              fp8KRValueRankerSite(semantics, input_dtype), shortlistLength(tuning_depth));

        std::vector<ReductionAutotuning::Candidate> candidates;
        std::vector<Fp8KRSegmentShape> candidate_shapes;
        std::vector<uint64_t> output_dimensions;
        for (const Fp8KRSegmentShape& shape : order) {
            auto plan = makeRetainedFp8KRValueCandidate(problem, site, shape, input_dtype).physical_plan;
            const DensePhysicalPassLayout layout = describeDensePhysicalPass(*plan);
            if (output_dimensions.empty()) output_dimensions = layout.geometry.output_dimensions;
            const size_t scratch_bytes = queryDensePhysicalPassScratchBytes(
                *plan, semantics, input_dtype, nullptr, geometry.rows * geometry.row_bytes, output_dtype, nullptr,
                layout.geometry, scale, stream);
            ReductionAutotuning::Candidate candidate;
            candidate.name = "fp8_kr:" + fp8KRShapeName(shape);
            candidate.scratch_bytes = scratch_bytes;
            candidate.launch = [plan, semantics, layout, scratch_bytes, scale](const Tensor& input, Tensor& output,
                                                                              Tensor& scratch, Stream& run_stream) {
                Tensor pass_input = input;
                pass_input.reshape(layout.input_dimensions);
                launchDensePhysicalPass(*plan, semantics, scratch, scratch_bytes, pass_input, output, layout.geometry,
                                        scale, run_stream);
            };
            candidates.push_back(std::move(candidate));
            candidate_shapes.push_back(shape);
        }
        if (candidates.empty()) throw std::logic_error("Retained FP8 KR VALUE selection has no candidate");

        const auto result = runContest(candidates, representative, TensorDescriptor(output_dtype, output_dimensions),
                                       stream, tuning_depth);
        if (result.winner_index >= candidates.size())
            throw std::logic_error("Retained FP8 KR VALUE autotuner returned an out-of-range winner");
        Fp8KRSelection selection;
        selection.shape = candidate_shapes[result.winner_index];
        selection.candidate_count = candidates.size();
        selection.winner_name = result.winner_name;
        return selection;
    });
}

Fp8KRSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                   DenseReductionSite site,
                                   bool maximum,
                                   DataType input_dtype,
                                   uint32_t multiprocessors,
                                   const Stream& stream,
                                   CubReductionTuningDepth tuning_depth) {
    if (!isOriginalFp8KRSite(problem, site, input_dtype) || problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.runs[site.run_index].kind != DenseReductionRunKind::OriginalReduction)
        throw std::invalid_argument("Retained FP8 KR ARG selection requires an original FP8 ARG R/KR site");
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const Fp8KRSiteGeometry geometry = fp8KRSiteGeometry(problem, site);
    const DenseReductionSelectionKey key =
        selectionKey(KrArgSite, problem, site, maximum ? 1 : 0, input_dtype, stream, tuning_depth);

    return selectCached(key, geometry.rows * geometry.row_bytes, input_dtype, stream,
                        [&](const Tensor& representative) {
        const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
        const bool final_pass = !hasDenseReductionSites(next);
        // Outputs exactly as the stamped pass writes them: the requested outputs on the final site, else the FP32
        // value + carried index intermediate. One UINT8 output slot holds the value, then (256-byte aligned) the index.
        const bool produce_value = !final_pass || problem.final_output.produce_value;
        const bool produce_index = !final_pass || problem.final_output.produce_index;
        const DataType value_dtype = final_pass ? problem.final_output.value_dtype : DataType::FP32;
        const DataType index_dtype = final_pass ? problem.final_output.index_dtype : problem.carried_arg_index_dtype;
        const std::vector<uint64_t> output_dims{geometry.rows};
        const uint64_t value_bytes =
            produce_value ? TensorDescriptor(value_dtype, output_dims).getArraySizeInBytes() : 0;
        const uint64_t index_offset = (value_bytes + 255) & ~uint64_t{255};
        const uint64_t index_bytes =
            produce_index ? TensorDescriptor(index_dtype, output_dims).getArraySizeInBytes() : 0;
        const uint64_t domain_stride = denseArgOriginalIndexStride(problem.runs[site.run_index]).value_or(0);

        Fp8KRRankerSite ranker_site;
        ranker_site.kind = produce_value ? 2 : 1;
        ranker_site.combine = maximum ? 3 : 2;
        ranker_site.e5m2 = input_dtype == DataType::FP8_E5M2;
        const std::vector<Fp8KRSegmentShape> shapes = fp8KRRankedShapes(
            geometry.rows, geometry.row_bytes, multiprocessors, ranker_site, shortlistLength(tuning_depth));
        if (shapes.empty()) throw std::logic_error("Retained FP8 KR ARG selection has no candidate");

        std::vector<ReductionAutotuning::Candidate> candidates;
        for (const Fp8KRSegmentShape& shape : shapes) {
            ReductionAutotuning::Candidate candidate;
            candidate.name = "fp8_arg_kr:" + fp8KRShapeName(shape);
            candidate.scratch_bytes = fp8KRScratchBytes(shape, true);
            const uint64_t scratch_bytes = candidate.scratch_bytes;
            candidate.launch = [=](const Tensor& input, Tensor& output, Tensor& scratch, Stream& run_stream) {
                std::optional<Tensor> value, index;
                if (produce_value) value = output.aliasByteStorageAs(value_dtype, output_dims, 0);
                if (produce_index) index = output.aliasByteStorageAs(index_dtype, output_dims, index_offset);
                void* partials = scratch_bytes != 0 ? scratch.getMemPtr<void>() : nullptr;
                launchFp8KRArgSegments(input, value ? &*value : nullptr, index ? &*index : nullptr, shape, partials,
                                       scratch_bytes, maximum, domain_stride, run_stream);
            };
            candidates.push_back(std::move(candidate));
        }
        const auto result = runContest(candidates, representative,
                                       TensorDescriptor(DataType::UINT8, {index_offset + index_bytes}), stream,
                                       tuning_depth);
        if (result.winner_index >= candidates.size())
            throw std::logic_error("Retained FP8 KR ARG autotuner returned an out-of-range winner");
        Fp8KRSelection selection;
        selection.shape = shapes[result.winner_index];
        selection.candidate_count = candidates.size();
        selection.winner_name = result.winner_name;
        return selection;
    });
}

}  // namespace ThorImplementation::CubReductionInternal::Fp8KR

namespace ThorImplementation::CubReductionInternal {

Fp8KR::Fp8KRRankerSite fp8KRValueRankerSite(const CubReductionStageSemantics& semantics, DataType input_dtype) {
    Fp8KR::Fp8KRRankerSite site;
    site.kind = 0;
    switch (semantics.combine) {
        case CubReductionStageCombine::Sum: site.combine = 0; break;
        case CubReductionStageCombine::Product: site.combine = 1; break;
        case CubReductionStageCombine::Minimum: site.combine = 2; break;
        case CubReductionStageCombine::Maximum: site.combine = 3; break;
    }
    switch (semantics.input_transform) {
        case CubReductionStageInputTransform::Identity: site.transform = 0; break;
        case CubReductionStageInputTransform::AbsoluteValue: site.transform = 1; break;
        case CubReductionStageInputTransform::Square: site.transform = 2; break;
    }
    // 1233: the slot is the alternate format: E5M2 for FP8, BF16 for the 2-byte formats (FP16 / BF16 plan, decision 1).
    site.e5m2 = input_dtype == DataType::FP8_E5M2 || input_dtype == DataType::BF16;
    return site;
}

}  // namespace ThorImplementation::CubReductionInternal
