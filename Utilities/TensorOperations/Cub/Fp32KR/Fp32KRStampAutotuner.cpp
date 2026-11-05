#include "Utilities/TensorOperations/Cub/Fp32KR/Fp32KRStampAutotuner.h"

#include "Utilities/TensorOperations/Cub/Fp32KR/Fp32KRRankerModel.h"

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8KRPhysicalPlan.h"
#include "Utilities/TensorOperations/Cub/Fp32CarriedArgContestInput.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKProductionCandidates.h"
#include "Utilities/TensorOperations/Cub/ReductionAutotuner.h"
#include "Utilities/TensorOperations/Cub/ReductionGpuMeasurementLock.h"
#include "Utilities/Common/ScopedGpu.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp32KR {
namespace {

// The site of a kind-Fp32KR cache key: every field that can change the shortlist, the measurement or the winner's
// legality (the FP8 KR key's fields; the dtype is always FP32, kept so the layouts match).
constexpr uint64_t KrValueSite = 1, KrArgSite = 2, KrCarriedArgSite = 3;

DenseReductionSelectionKey selectionKey(uint64_t kind, const DenseReductionProblem& problem, DenseReductionSite site,
                                        uint64_t operation, const Stream& stream, CubReductionTuningDepth depth) {
    const Fp32KRSiteGeometry geometry = fp32KRSiteGeometry(problem, site);
    const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
    const bool final_pass = !hasDenseReductionSites(next);
    DenseReductionSelectionKey key;
    key.kind = DenseReductionSelectionKind::Fp32KR;
    key.device = denseReductionDeviceIdentity(static_cast<int>(stream.getGpuNum()));
    key.tuning_depth = depth;
    key.site = {kind,
                static_cast<uint64_t>(DataType::FP32),
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
    if (kind == KrCarriedArgSite) {
        // 1215: the carried index the pass reads and the reduced run's kind (original or intermediate).
        key.site.insert(key.site.end(), {static_cast<uint64_t>(problem.carried_arg_index_dtype),
                                         static_cast<uint64_t>(problem.runs[site.run_index].kind)});
    }
    return key;
}

// A miss allocates a representative FP32 input (1.0 everywhere) under the measurement lock, tunes, and releases it
// under the lock; the contest takes the lock itself. Hits allocate nothing.
Fp32KRSelection selectCached(const DenseReductionSelectionKey& key, uint64_t input_elements, const Stream& stream,
                             const std::function<Fp32KRSelection(const Tensor&)>& tune) {
    return denseReductionSelectionCache().getOrTune<Fp32KRSelection>(key, [&] {
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
        Tensor representative;
        try {
            {
                ScopedReductionGpuMeasurement preparation(static_cast<int>(stream.getGpuNum()));
                representative = Tensor(placement, TensorDescriptor(DataType::FP32, {input_elements}));
                Stream fill_stream = stream;
                representative.fill(1.0, fill_stream);
                fill_stream.synchronize();
            }
            Fp32KRSelection selection = tune(representative);
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

// The shared contest with the depth's rounds and trim (the RK depths), under the measurement lock.
ReductionAutotuning::RunResult runContest(const std::vector<ReductionAutotuning::Candidate>& candidates,
                                          const Tensor& representative, const TensorDescriptor& output_descriptor,
                                          const Stream& stream, CubReductionTuningDepth depth) {
    const Fp8RK::Fp8RKTuningPlan tuning = Fp8RK::fp8RKTuningPlan(depth);
    ScopedReductionGpuMeasurement measurement(static_cast<int>(stream.getGpuNum()));
    ReductionAutotuning::Options options;
    options.measurement_count = tuning.rounds;
    options.trim_count = tuning.trimmed;
    Stream tuning_stream = stream;
    return ReductionAutotuning::autotune(candidates, representative, output_descriptor, tuning_stream, options);
}

}  // namespace

Fp32KRSiteGeometry fp32KRSiteGeometry(const DenseReductionProblem& problem, DenseReductionSite site) {
    const Fp8KRSiteGeometry elements = fp8KRSiteGeometry(problem, site);  // rows and the row's element count
    if (elements.row_bytes > (uint64_t{1} << 60)) throw std::invalid_argument("Retained FP32 KR row is too long");
    return {elements.rows, 4 * elements.row_bytes};
}

bool isOriginalFp32KRSite(const DenseReductionProblem& problem, DenseReductionSite site,
                          DataType original_input_dtype) {
    if (original_input_dtype != DataType::FP32) return false;
    if (problem.input_state != DenseReductionInputState::OriginalInput) return false;
    if (site.run_index >= problem.runs.size()) return false;
    const DenseReducerFamily family = classifyDenseReducerFamily(problem, site);
    return family == DenseReducerFamily::R || family == DenseReducerFamily::KR;
}

bool isCarriedFp32KRArgSite(const DenseReductionProblem& problem, DenseReductionSite site) {
    if (problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.input_state != DenseReductionInputState::IntermediateAggregate ||
        (problem.carried_arg_index_dtype != DataType::UINT32 && problem.carried_arg_index_dtype != DataType::UINT64) ||
        site.run_index >= problem.runs.size())
        return false;
    const DenseReducerFamily family = classifyDenseReducerFamily(problem, site);
    return family == DenseReducerFamily::R || family == DenseReducerFamily::KR;
}

bool isFp32KRValueSite(const DenseReductionProblem& problem, DenseReductionSite site, DataType original_input_dtype) {
    if (problem.input_state == DenseReductionInputState::OriginalInput)
        return isOriginalFp32KRSite(problem, site, original_input_dtype);
    if (site.run_index >= problem.runs.size()) return false;
    const DenseReducerFamily family = classifyDenseReducerFamily(problem, site);
    return family == DenseReducerFamily::R || family == DenseReducerFamily::KR;
}

bool fp32KRRankerTrained() { return Fp32KRRankerModel::TRAINED; }

std::vector<Fp8KR::Fp8KRPoolEntry> fp32KRCandidatePool(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                       const Fp8KR::Fp8KRRankerSite& site) {
    std::vector<Fp8KR::Fp8KRPoolEntry> pool = Fp8KR::retainedKRCandidatePool(rows, row_bytes, multiprocessors, 4);
    if (site.kind != 0) {
        std::erase_if(pool, [](const Fp8KR::Fp8KRPoolEntry& entry) {
            return !Fp8KR::fp32KRArgShapeLegal(entry.shape);
        });
    }
    return pool;
}

std::vector<Fp8KR::Fp8KRSegmentShape> fp32KRRankedShapes(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                         const Fp8KR::Fp8KRRankerSite& site, size_t count) {
    const std::vector<Fp8KR::Fp8KRPoolEntry> pool = fp32KRCandidatePool(rows, row_bytes, multiprocessors, site);
    std::vector<Fp8KR::Fp8KRSegmentShape> result;
    if (!fp32KRRankerTrained()) {
        for (const Fp8KR::Fp8KRPoolEntry& entry : pool) {
            if (result.size() >= count) break;
            if (entry.list_position != 0) result.push_back(entry.shape);
        }
        return result;
    }
    static_assert(Fp32KRRankerModel::FEATURE_COUNT == Fp8KR::FP8_KR_RANKER_FEATURES);
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(pool.size());
    for (size_t i = 0; i < pool.size(); ++i)
        scored.emplace_back(Fp32KRRankerModel::margin(Fp8KR::fp8KRRankerFeatures(pool[i], site, multiprocessors)), i);
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < scored.size() && result.size() < count; ++i) result.push_back(pool[scored[i].second].shape);
    return result;
}

size_t fp32KRShortlistLength(CubReductionTuningDepth depth) {
    if (!fp32KRRankerTrained()) return Fp8RK::fp8RKTuningPlan(depth).candidates;
    switch (depth) {
        case CubReductionTuningDepth::Quick: return FP32_KR_SHORTLIST_DEPTHS[0];
        case CubReductionTuningDepth::Regular: return FP32_KR_SHORTLIST_DEPTHS[1];
        case CubReductionTuningDepth::Ultra: return FP32_KR_SHORTLIST_DEPTHS[2];
    }
    return FP32_KR_SHORTLIST_DEPTHS[1];
}

std::vector<Fp8KR::Fp8KRSegmentShape> fp32KRShortlist(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                      const Fp8KR::Fp8KRRankerSite& site,
                                                      CubReductionTuningDepth depth) {
    return fp32KRRankedShapes(rows, row_bytes, multiprocessors, site, fp32KRShortlistLength(depth));
}

Fp8KR::Fp8KRRankerSite fp32KRArgRankerSite(bool maximum, bool produce_value, bool carried) {
    Fp8KR::Fp8KRRankerSite site;
    site.kind = static_cast<uint8_t>((produce_value ? 2 : 1) + (carried ? 2 : 0));
    site.combine = maximum ? 3 : 2;
    site.e5m2 = false;
    return site;
}

Fp32KRSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                      DenseReductionSite site,
                                      CubReductionOp operation,
                                      uint64_t original_reduction_size,
                                      float output_scale,
                                      uint32_t multiprocessors,
                                      const Stream& stream,
                                      CubReductionTuningDepth tuning_depth) {
    if (!isFp32KRValueSite(problem, site, DataType::FP32) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Value || !problem.final_output.produce_value ||
        problem.final_output.produce_index || original_reduction_size == 0)
        throw std::invalid_argument("Retained FP32 KR VALUE selection requires an FP32 VALUE R/KR site");
    const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
    const CubReductionStageSemantics semantics =
        makeValueReductionStageSemantics(operation, deriveDensePassRole(problem, next), original_reduction_size);
    if (!fp8KRValueSupports(semantics)) return {};  // none of the public operations today

    ScopedGpu scoped_gpu(stream.getGpuNum());
    const Fp32KRSiteGeometry geometry = fp32KRSiteGeometry(problem, site);
    const uint64_t input_elements = geometry.rows * (geometry.row_bytes / 4);
    const DenseReductionSelectionKey key =
        selectionKey(KrValueSite, problem, site, static_cast<uint64_t>(operation), stream, tuning_depth);

    return selectCached(key, input_elements, stream, [&](const Tensor& representative) {
        const bool final_pass = !hasDenseReductionSites(next);
        const DataType output_dtype = final_pass ? problem.final_output.value_dtype : DataType::FP32;
        const float scale = final_pass ? output_scale : 1.0f;
        const std::vector<Fp8KR::Fp8KRSegmentShape> shapes =
            fp32KRShortlist(geometry.rows, geometry.row_bytes, multiprocessors,
                            fp8KRValueRankerSite(semantics, DataType::FP32), tuning_depth);
        if (shapes.empty()) throw std::logic_error("Retained FP32 KR VALUE selection has no candidate");

        std::vector<ReductionAutotuning::Candidate> candidates;
        for (const Fp8KR::Fp8KRSegmentShape& shape : shapes) {
            ReductionAutotuning::Candidate candidate;
            candidate.name = "fp32_kr:" + Fp8KR::fp8KRShapeName(shape);
            candidate.scratch_bytes = Fp8KR::fp8KRScratchBytes(shape, false);
            const uint64_t scratch_bytes = candidate.scratch_bytes;
            candidate.launch = [shape, semantics, scale, scratch_bytes](const Tensor& input, Tensor& output,
                                                                       Tensor& scratch, Stream& run_stream) {
                void* partials = scratch_bytes != 0 ? scratch.getMemPtr<void>() : nullptr;
                launchFp32KRValueSegments(semantics, input, output, shape, partials, scratch_bytes, scale,
                                          run_stream);
            };
            candidates.push_back(std::move(candidate));
        }

        const ReductionAutotuning::RunResult result = runContest(
            candidates, representative, TensorDescriptor(output_dtype, {geometry.rows}), stream, tuning_depth);
        if (result.winner_index >= candidates.size())
            throw std::logic_error("Retained FP32 KR VALUE autotuner returned an out-of-range winner");
        Fp32KRSelection selection;
        selection.shape = shapes[result.winner_index];
        selection.candidate_count = candidates.size();
        selection.winner_name = result.winner_name;
        return selection;
    });
}

Fp32KRSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                    DenseReductionSite site,
                                    bool maximum,
                                    uint32_t multiprocessors,
                                    const Stream& stream,
                                    CubReductionTuningDepth tuning_depth) {
    if (!isOriginalFp32KRSite(problem, site, DataType::FP32) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.runs[site.run_index].kind != DenseReductionRunKind::OriginalReduction)
        throw std::invalid_argument("Retained FP32 KR ARG selection requires an original FP32 ARG R/KR site");
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const Fp32KRSiteGeometry geometry = fp32KRSiteGeometry(problem, site);
    const uint64_t input_elements = geometry.rows * (geometry.row_bytes / 4);
    const DenseReductionSelectionKey key =
        selectionKey(KrArgSite, problem, site, maximum ? 1 : 0, stream, tuning_depth);

    return selectCached(key, input_elements, stream, [&](const Tensor& representative) {
        const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
        const bool final_pass = !hasDenseReductionSites(next);
        // Outputs exactly as the stamped pass will write them: the requested outputs on the final site, else the FP32
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

        const std::vector<Fp8KR::Fp8KRSegmentShape> shapes =
            fp32KRShortlist(geometry.rows, geometry.row_bytes, multiprocessors,
                            fp32KRArgRankerSite(maximum, produce_value), tuning_depth);
        if (shapes.empty()) throw std::logic_error("Retained FP32 KR ARG selection has no candidate");

        std::vector<ReductionAutotuning::Candidate> candidates;
        for (const Fp8KR::Fp8KRSegmentShape& shape : shapes) {
            ReductionAutotuning::Candidate candidate;
            candidate.name = "fp32_arg_kr:" + Fp8KR::fp8KRShapeName(shape);
            candidate.scratch_bytes = Fp8KR::fp8KRScratchBytes(shape, true);
            const uint64_t scratch_bytes = candidate.scratch_bytes;
            candidate.launch = [=](const Tensor& input, Tensor& output, Tensor& scratch, Stream& run_stream) {
                std::optional<Tensor> value, index;
                if (produce_value) value = output.aliasByteStorageAs(value_dtype, output_dims, 0);
                if (produce_index) index = output.aliasByteStorageAs(index_dtype, output_dims, index_offset);
                void* partials = scratch_bytes != 0 ? scratch.getMemPtr<void>() : nullptr;
                launchFp32KRArgSegments(input, value ? &*value : nullptr, index ? &*index : nullptr, shape, partials,
                                        scratch_bytes, maximum, domain_stride, run_stream);
            };
            candidates.push_back(std::move(candidate));
        }
        const auto result = runContest(candidates, representative,
                                       TensorDescriptor(DataType::UINT8, {index_offset + index_bytes}), stream,
                                       tuning_depth);
        if (result.winner_index >= candidates.size())
            throw std::logic_error("Retained FP32 KR ARG autotuner returned an out-of-range winner");
        Fp32KRSelection selection;
        selection.shape = shapes[result.winner_index];
        selection.candidate_count = candidates.size();
        selection.winner_name = result.winner_name;
        return selection;
    });
}

namespace {
// The carried contest at a validated site over a contest input (Fp32CarriedArgContestInput.h) of its geometry.
Fp32KRSelection tuneCarriedArgContest(const DenseReductionProblem& problem, DenseReductionSite site, bool maximum,
                                      uint32_t multiprocessors, const Tensor& contest_input, const Stream& stream,
                                      CubReductionTuningDepth tuning_depth) {
    const Fp32KRSiteGeometry geometry = fp32KRSiteGeometry(problem, site);
    const uint64_t elements = geometry.rows * (geometry.row_bytes / 4);
    const Fp32CarriedArgContestInput layout = fp32CarriedArgContestInput(elements, problem.carried_arg_index_dtype);
    if (contest_input.getDataType() != DataType::UINT8 || contest_input.getTotalNumElements() != layout.bytes ||
        !contest_input.isDenseContiguous() || contest_input.getPlacement().getDeviceNum() != stream.getGpuNum())
        throw std::invalid_argument("Retained FP32 KR carried ARG contest input does not match the site");
    const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
    const bool final_pass = !hasDenseReductionSites(next);
    // Outputs as selectCachedArgSite's slot; the composed index needs the run's original-domain stride (0 for an
    // intermediate run, whose carried index is already composed).
    const bool produce_value = !final_pass || problem.final_output.produce_value;
    const bool produce_index = !final_pass || problem.final_output.produce_index;
    const DataType value_dtype = final_pass ? problem.final_output.value_dtype : DataType::FP32;
    const DataType index_dtype = final_pass ? problem.final_output.index_dtype : problem.carried_arg_index_dtype;
    const std::vector<uint64_t> output_dims{geometry.rows};
    const uint64_t value_bytes = produce_value ? TensorDescriptor(value_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t index_offset = (value_bytes + 255) & ~uint64_t{255};
    const uint64_t index_bytes = produce_index ? TensorDescriptor(index_dtype, output_dims).getArraySizeInBytes() : 0;
    const uint64_t row_stride = denseArgOriginalIndexStride(problem.runs[site.run_index]).value_or(0);
    const std::vector<uint64_t> input_dims{geometry.rows, geometry.row_bytes / 4};

    const std::vector<Fp8KR::Fp8KRSegmentShape> shapes =
        fp32KRShortlist(geometry.rows, geometry.row_bytes, multiprocessors,
                        fp32KRArgRankerSite(maximum, produce_value, true), tuning_depth);
    if (shapes.empty()) throw std::logic_error("Retained FP32 KR carried ARG selection has no candidate");
    std::vector<ReductionAutotuning::Candidate> candidates;
    for (const Fp8KR::Fp8KRSegmentShape& shape : shapes) {
        ReductionAutotuning::Candidate candidate;
        candidate.name = "fp32_carried_arg_kr:" + Fp8KR::fp8KRShapeName(shape);
        candidate.scratch_bytes = fp32KRCarriedArgScratchBytes(shape);
        const uint64_t scratch_bytes = candidate.scratch_bytes;
        candidate.launch = [=](const Tensor& input, Tensor& output, Tensor& scratch, Stream& run_stream) {
            const Tensor values = layout.values(input, input_dims);
            const Tensor carried = layout.carriedIndex(input, input_dims);
            std::optional<Tensor> value, index;
            if (produce_value) value = output.aliasByteStorageAs(value_dtype, output_dims, 0);
            if (produce_index) index = output.aliasByteStorageAs(index_dtype, output_dims, index_offset);
            void* partials = scratch_bytes != 0 ? scratch.getMemPtr<void>() : nullptr;
            launchFp32KRCarriedArgSegments(values, carried, value ? &*value : nullptr, index ? &*index : nullptr,
                                           shape, partials, scratch_bytes, maximum, row_stride, run_stream);
        };
        candidates.push_back(std::move(candidate));
    }
    const auto result = runContest(candidates, contest_input,
                                   TensorDescriptor(DataType::UINT8, {index_offset + index_bytes}), stream,
                                   tuning_depth);
    if (result.winner_index >= candidates.size())
        throw std::logic_error("Retained FP32 KR carried ARG autotuner returned an out-of-range winner");
    Fp32KRSelection selection;
    selection.shape = shapes[result.winner_index];
    selection.candidate_count = candidates.size();
    selection.winner_name = result.winner_name;
    return selection;
}

// A contest input made and released under the measurement lock around `tune`.
template <typename Make, typename Tune>
Fp32KRSelection withCarriedContestInput(const Stream& stream, Make make, Tune tune) {
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
        Fp32KRSelection selection = tune(contest_input);
        release();
        return selection;
    } catch (...) {
        release();
        throw;
    }
}
}  // namespace

Fp32KRSelection autotuneCarriedArgSite(const DenseReductionProblem& problem,
                                       DenseReductionSite site,
                                       bool maximum,
                                       uint32_t multiprocessors,
                                       const Tensor& representative_values,
                                       const Tensor& representative_carried_index,
                                       const Stream& stream,
                                       CubReductionTuningDepth tuning_depth) {
    if (!isCarriedFp32KRArgSite(problem, site))
        throw std::invalid_argument(
            "Retained FP32 KR carried ARG selection requires a carried-index FP32 ARG R/KR site");
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const Fp32KRSiteGeometry geometry = fp32KRSiteGeometry(problem, site);
    const Fp32CarriedArgContestInput layout =
        fp32CarriedArgContestInput(geometry.rows * (geometry.row_bytes / 4), problem.carried_arg_index_dtype);
    const auto on_site = [&](const Tensor& tensor) {
        return tensor.isDenseContiguous() && tensor.getPlacement().getDeviceNum() == stream.getGpuNum();
    };
    if (!on_site(representative_values) || !on_site(representative_carried_index))
        throw std::invalid_argument(
            "Retained FP32 KR carried ARG autotune inputs must be contiguous on the stream's GPU");
    return withCarriedContestInput(
        stream,
        [&] { return makeFp32CarriedArgContestInput(layout, stream, &representative_values,
                                                    &representative_carried_index); },
        [&](const Tensor& contest_input) {
            return tuneCarriedArgContest(problem, site, maximum, multiprocessors, contest_input, stream,
                                         tuning_depth);
        });
}

Fp32KRSelection selectCachedCarriedArgSite(const DenseReductionProblem& problem,
                                           DenseReductionSite site,
                                           bool maximum,
                                           uint32_t multiprocessors,
                                           const Stream& stream,
                                           CubReductionTuningDepth tuning_depth) {
    if (!isCarriedFp32KRArgSite(problem, site))
        throw std::invalid_argument(
            "Retained FP32 KR carried ARG selection requires a carried-index FP32 ARG R/KR site");
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const Fp32KRSiteGeometry geometry = fp32KRSiteGeometry(problem, site);
    const DenseReductionSelectionKey key =
        selectionKey(KrCarriedArgSite, problem, site, maximum ? 1 : 0, stream, tuning_depth);
    return denseReductionSelectionCache().getOrTune<Fp32KRSelection>(key, [&] {
        // A representative contest input: values 1.0, carried index 0.
        const Fp32CarriedArgContestInput layout =
            fp32CarriedArgContestInput(geometry.rows * (geometry.row_bytes / 4), problem.carried_arg_index_dtype);
        return withCarriedContestInput(
            stream, [&] { return makeFp32CarriedArgContestInput(layout, stream); },
            [&](const Tensor& contest_input) {
                return tuneCarriedArgContest(problem, site, maximum, multiprocessors, contest_input, stream,
                                             tuning_depth);
            });
    });
}

}  // namespace ThorImplementation::CubReductionInternal::Fp32KR
