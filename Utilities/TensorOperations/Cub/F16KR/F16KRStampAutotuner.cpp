#include "Utilities/TensorOperations/Cub/F16KR/F16KRStampAutotuner.h"

#include "Utilities/TensorOperations/Cub/F16KR/F16KRRankerModel.h"

#include "Utilities/TensorOperations/Cub/CubReductionInternal.h"
#include "Utilities/TensorOperations/Cub/CubReductionStageSemantics.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8KRPhysicalPlan.h"
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

namespace ThorImplementation::CubReductionInternal::F16KR {
namespace {

// The site of a kind-F16KR cache key: the Fp32KR key's fields, with the real input dtype (FP16 or BF16).
constexpr uint64_t KrValueSite = 1, KrArgSite = 2;

DenseReductionSelectionKey selectionKey(uint64_t kind, const DenseReductionProblem& problem, DenseReductionSite site,
                                        DataType input_dtype, uint64_t operation, const Stream& stream,
                                        CubReductionTuningDepth depth) {
    const F16KRSiteGeometry geometry = f16KRSiteGeometry(problem, site);
    const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
    const bool final_pass = !hasDenseReductionSites(next);
    DenseReductionSelectionKey key;
    key.kind = DenseReductionSelectionKind::F16KR;
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

// A miss allocates a representative input of the site's dtype (1.0 everywhere) under the measurement lock, tunes,
// and releases it under the lock; the contest takes the lock itself. Hits allocate nothing.
F16KRSelection selectCached(const DenseReductionSelectionKey& key, DataType input_dtype, uint64_t input_elements,
                            const Stream& stream, const std::function<F16KRSelection(const Tensor&)>& tune) {
    return denseReductionSelectionCache().getOrTune<F16KRSelection>(key, [&] {
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
            F16KRSelection selection = tune(representative);
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

F16KRSiteGeometry f16KRSiteGeometry(const DenseReductionProblem& problem, DenseReductionSite site) {
    const Fp8KRSiteGeometry elements = fp8KRSiteGeometry(problem, site);  // rows and the row's element count
    if (elements.row_bytes > (uint64_t{1} << 61))
        throw std::invalid_argument("Retained FP16 / BF16 KR row is too long");
    return {elements.rows, 2 * elements.row_bytes};
}

bool isF16KRInputDtype(DataType dtype) { return dtype == DataType::FP16 || dtype == DataType::BF16; }

bool isOriginalF16KRSite(const DenseReductionProblem& problem, DenseReductionSite site,
                         DataType original_input_dtype) {
    if (!isF16KRInputDtype(original_input_dtype)) return false;
    if (problem.input_state != DenseReductionInputState::OriginalInput) return false;
    if (site.run_index >= problem.runs.size()) return false;
    const DenseReducerFamily family = classifyDenseReducerFamily(problem, site);
    return family == DenseReducerFamily::R || family == DenseReducerFamily::KR;
}

bool f16KRRankerTrained() { return F16KRRankerModel::TRAINED; }

std::vector<Fp8KR::Fp8KRPoolEntry> f16KRCandidatePool(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                      const Fp8KR::Fp8KRRankerSite& site) {
    std::vector<Fp8KR::Fp8KRPoolEntry> pool = Fp8KR::retainedKRCandidatePool(rows, row_bytes, multiprocessors, 2);
    if (site.kind != 0) {
        std::erase_if(pool, [](const Fp8KR::Fp8KRPoolEntry& entry) {
            return !Fp8KR::f16KRArgShapeLegal(entry.shape);
        });
    }
    return pool;
}

std::vector<Fp8KR::Fp8KRSegmentShape> f16KRRankedShapes(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                        const Fp8KR::Fp8KRRankerSite& site, size_t count) {
    const std::vector<Fp8KR::Fp8KRPoolEntry> pool = f16KRCandidatePool(rows, row_bytes, multiprocessors, site);
    std::vector<Fp8KR::Fp8KRSegmentShape> result;
    if (!f16KRRankerTrained()) {
        for (const Fp8KR::Fp8KRPoolEntry& entry : pool) {
            if (result.size() >= count) break;
            if (entry.list_position != 0) result.push_back(entry.shape);
        }
        return result;
    }
    static_assert(F16KRRankerModel::FEATURE_COUNT == Fp8KR::FP8_KR_RANKER_FEATURES);
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(pool.size());
    for (size_t i = 0; i < pool.size(); ++i)
        scored.emplace_back(F16KRRankerModel::margin(Fp8KR::fp8KRRankerFeatures(pool[i], site, multiprocessors)), i);
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t i = 0; i < scored.size() && result.size() < count; ++i) result.push_back(pool[scored[i].second].shape);
    return result;
}

size_t f16KRShortlistLength(CubReductionTuningDepth depth) {
    if (!f16KRRankerTrained()) return Fp8RK::fp8RKTuningPlan(depth).candidates;
    switch (depth) {
        case CubReductionTuningDepth::Quick: return F16_KR_SHORTLIST_DEPTHS[0];
        case CubReductionTuningDepth::Regular: return F16_KR_SHORTLIST_DEPTHS[1];
        case CubReductionTuningDepth::Ultra: return F16_KR_SHORTLIST_DEPTHS[2];
    }
    return F16_KR_SHORTLIST_DEPTHS[1];
}

std::vector<Fp8KR::Fp8KRSegmentShape> f16KRShortlist(uint64_t rows, uint64_t row_bytes, uint32_t multiprocessors,
                                                     const Fp8KR::Fp8KRRankerSite& site,
                                                     CubReductionTuningDepth depth) {
    return f16KRRankedShapes(rows, row_bytes, multiprocessors, site, f16KRShortlistLength(depth));
}

Fp8KR::Fp8KRRankerSite f16KRArgRankerSite(bool maximum, bool produce_value, bool bf16) {
    Fp8KR::Fp8KRRankerSite site;
    site.kind = static_cast<uint8_t>(produce_value ? 2 : 1);
    site.combine = maximum ? 3 : 2;
    site.e5m2 = bf16;
    return site;
}

F16KRSelection selectCachedValueSite(const DenseReductionProblem& problem,
                                     DenseReductionSite site,
                                     DataType input_dtype,
                                     CubReductionOp operation,
                                     uint64_t original_reduction_size,
                                     float output_scale,
                                     uint32_t multiprocessors,
                                     const Stream& stream,
                                     CubReductionTuningDepth tuning_depth) {
    if (!isOriginalF16KRSite(problem, site, input_dtype) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Value || !problem.final_output.produce_value ||
        problem.final_output.produce_index || original_reduction_size == 0)
        throw std::invalid_argument("Retained FP16 / BF16 KR VALUE selection requires an original 2-byte VALUE R/KR "
                                    "site");
    const DenseReductionProblem next = advanceDenseReductionProblem(problem, site, 1);
    const CubReductionStageSemantics semantics =
        makeValueReductionStageSemantics(operation, deriveDensePassRole(problem, next), original_reduction_size);
    if (!fp8KRValueSupports(semantics)) return {};  // none of the public operations today

    ScopedGpu scoped_gpu(stream.getGpuNum());
    const F16KRSiteGeometry geometry = f16KRSiteGeometry(problem, site);
    const uint64_t input_elements = geometry.rows * (geometry.row_bytes / 2);
    const DenseReductionSelectionKey key = selectionKey(KrValueSite, problem, site, input_dtype,
                                                        static_cast<uint64_t>(operation), stream, tuning_depth);

    return selectCached(key, input_dtype, input_elements, stream, [&](const Tensor& representative) {
        const bool final_pass = !hasDenseReductionSites(next);
        const DataType output_dtype = final_pass ? problem.final_output.value_dtype : DataType::FP32;
        const float scale = final_pass ? output_scale : 1.0f;
        const std::vector<Fp8KR::Fp8KRSegmentShape> shapes =
            f16KRShortlist(geometry.rows, geometry.row_bytes, multiprocessors,
                           fp8KRValueRankerSite(semantics, input_dtype), tuning_depth);
        if (shapes.empty()) throw std::logic_error("Retained FP16 / BF16 KR VALUE selection has no candidate");

        std::vector<ReductionAutotuning::Candidate> candidates;
        for (const Fp8KR::Fp8KRSegmentShape& shape : shapes) {
            ReductionAutotuning::Candidate candidate;
            candidate.name = "f16_kr:" + Fp8KR::fp8KRShapeName(shape);
            candidate.scratch_bytes = Fp8KR::fp8KRScratchBytes(shape, false);
            const uint64_t scratch_bytes = candidate.scratch_bytes;
            candidate.launch = [shape, semantics, scale, scratch_bytes](const Tensor& input, Tensor& output,
                                                                       Tensor& scratch, Stream& run_stream) {
                void* partials = scratch_bytes != 0 ? scratch.getMemPtr<void>() : nullptr;
                launchF16KRValueSegments(semantics, input, output, shape, partials, scratch_bytes, scale, run_stream);
            };
            candidates.push_back(std::move(candidate));
        }

        const ReductionAutotuning::RunResult result = runContest(
            candidates, representative, TensorDescriptor(output_dtype, {geometry.rows}), stream, tuning_depth);
        if (result.winner_index >= candidates.size())
            throw std::logic_error("Retained FP16 / BF16 KR VALUE autotuner returned an out-of-range winner");
        F16KRSelection selection;
        selection.shape = shapes[result.winner_index];
        selection.candidate_count = candidates.size();
        selection.winner_name = result.winner_name;
        return selection;
    });
}

F16KRSelection selectCachedArgSite(const DenseReductionProblem& problem,
                                   DenseReductionSite site,
                                   DataType input_dtype,
                                   bool maximum,
                                   uint32_t multiprocessors,
                                   const Stream& stream,
                                   CubReductionTuningDepth tuning_depth) {
    if (!isOriginalF16KRSite(problem, site, input_dtype) ||
        problem.aggregate_kind != DenseReductionAggregateKind::Arg ||
        problem.runs[site.run_index].kind != DenseReductionRunKind::OriginalReduction)
        throw std::invalid_argument("Retained FP16 / BF16 KR ARG selection requires an original 2-byte ARG R/KR site");
    ScopedGpu scoped_gpu(stream.getGpuNum());
    const F16KRSiteGeometry geometry = f16KRSiteGeometry(problem, site);
    const uint64_t input_elements = geometry.rows * (geometry.row_bytes / 2);
    const DenseReductionSelectionKey key =
        selectionKey(KrArgSite, problem, site, input_dtype, maximum ? 1 : 0, stream, tuning_depth);

    return selectCached(key, input_dtype, input_elements, stream, [&](const Tensor& representative) {
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
            f16KRShortlist(geometry.rows, geometry.row_bytes, multiprocessors,
                           f16KRArgRankerSite(maximum, produce_value, input_dtype == DataType::BF16), tuning_depth);
        if (shapes.empty()) throw std::logic_error("Retained FP16 / BF16 KR ARG selection has no candidate");

        std::vector<ReductionAutotuning::Candidate> candidates;
        for (const Fp8KR::Fp8KRSegmentShape& shape : shapes) {
            ReductionAutotuning::Candidate candidate;
            candidate.name = "f16_arg_kr:" + Fp8KR::fp8KRShapeName(shape);
            candidate.scratch_bytes = Fp8KR::fp8KRScratchBytes(shape, true);
            const uint64_t scratch_bytes = candidate.scratch_bytes;
            candidate.launch = [=](const Tensor& input, Tensor& output, Tensor& scratch, Stream& run_stream) {
                std::optional<Tensor> value, index;
                if (produce_value) value = output.aliasByteStorageAs(value_dtype, output_dims, 0);
                if (produce_index) index = output.aliasByteStorageAs(index_dtype, output_dims, index_offset);
                void* partials = scratch_bytes != 0 ? scratch.getMemPtr<void>() : nullptr;
                launchF16KRArgSegments(input, value ? &*value : nullptr, index ? &*index : nullptr, shape, partials,
                                       scratch_bytes, maximum, domain_stride, run_stream);
            };
            candidates.push_back(std::move(candidate));
        }
        const auto result = runContest(candidates, representative,
                                       TensorDescriptor(DataType::UINT8, {index_offset + index_bytes}), stream,
                                       tuning_depth);
        if (result.winner_index >= candidates.size())
            throw std::logic_error("Retained FP16 / BF16 KR ARG autotuner returned an out-of-range winner");
        F16KRSelection selection;
        selection.shape = shapes[result.winner_index];
        selection.candidate_count = candidates.size();
        selection.winner_name = result.winner_name;
        return selection;
    });
}

}  // namespace ThorImplementation::CubReductionInternal::F16KR
