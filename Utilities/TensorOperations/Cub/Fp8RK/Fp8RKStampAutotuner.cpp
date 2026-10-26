#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKBoundSelection.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKRuntimeProposals.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectionCache.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKDescriptorSelection.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/ReductionAutotuner.h"
#include "Utilities/TensorOperations/Cub/ReductionGpuMeasurementLock.h"
#include "Utilities/Common/ScopedGpu.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation::CubReductionInternal::Fp8RK {
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
// CUDA identity and resource geometry are process-lifetime device properties.
// Cache them by ordinal to keep *every* FP8 recipe cache hit free of CUDA
// property queries. No selection or physical-plan state is cached here.
Fp8RKDeviceSelectionIdentity deviceIdentity(const Stream& stream) {
    static std::mutex mutex;
    static std::unordered_map<int, Fp8RKDeviceSelectionIdentity> identities;
    const int ordinal = static_cast<int>(stream.getGpuNum());
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = identities.find(ordinal); it != identities.end())
        return it->second;
    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, ordinal);
    if (status != cudaSuccess || properties.multiProcessorCount <= 0)
        throw std::runtime_error("FP8 RK could not query CUDA device identity");
    Fp8RKDeviceSelectionIdentity identity;
    static_assert(sizeof(properties.uuid.bytes) == 16);
    std::memcpy(identity.uuid.data(), properties.uuid.bytes, identity.uuid.size());
    identity.ordinal = ordinal;
    identity.sm_count = static_cast<uint32_t>(properties.multiProcessorCount);
    identity.compute_major = static_cast<uint32_t>(properties.major);
    identity.compute_minor = static_cast<uint32_t>(properties.minor);
    return identities.emplace(ordinal, identity).first->second;
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

Fp8RKSelection autotuneIsolatedRKSite(
    const DenseReductionProblem& problem, DenseReductionSite site, CubReductionOp operation,
    uint64_t original_reduction_size, float output_scale,
    const Tensor& representative_input, const Stream& stream, uint64_t workspace_budget_bytes) {
    if (representative_input.getDataType() != DataType::FP8_E4M3 &&
        representative_input.getDataType() != DataType::FP8_E5M2)
        throw std::invalid_argument("FP8 RK autotuning requires FP8 input");
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
    const BoundSelectionRequest request{
        shape.outer, shape.r, shape.k, device.sm_count,
        representative_input.getDataType(),
        final_site ? problem.final_output.value_dtype : DataType::FP32,
        semantics, workspace_budget_bytes};
    auto ranked = selectBoundFirstStages(request, runtime);
    // A single, owning production context supplies the exact same FP32
    // occupancy and physical policy as ordinary dense planning. It remains
    // alive throughout every dependency selection (outside the GPU lock).
    const DenseProductionPhysicalSelectionContext fp32_context(
        representative_input.getDataType(), denseRKValueOperation(operation), stream);
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
        return ReductionAutotuning::autotune(
            candidates, representative_input, output_descriptor, tuning_stream);
    }();
    if (result.winner_index >= executables.size())
        throw std::logic_error("FP8 RK autotuner returned an out-of-range winner");
    Fp8RKSelection selection;
    // The selected physical kernel is the ONLY executable identity in the
    // cache. Lookahead passes, timing buffers, and their workspace accounting
    // die with the temporary measurement programs above.
    selection.physical = executables[result.winner_index].first_physical;
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

Fp8RKSelection selectCachedDescriptorImpl(
    const DenseReductionProblem& problem, DenseReductionSite site, CubReductionOp operation,
    uint64_t original_reduction_size, float output_scale, DataType input_dtype,
    const Tensor* representative_input, const Stream& stream, uint64_t workspace_budget_bytes) {
    const Shape shape = validateDescriptorSelection(
        problem, site, input_dtype, original_reduction_size, output_scale);
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
    const auto key = makeFp8RKDescriptorSelectionKey(
        problem, site, operation, original_reduction_size, output_scale, input_dtype,
        identity, workspace_budget_bytes);

    // The cache is the allocation boundary. Only a single-flight miss leader
    // constructs benchmark input. Hits need neither Tensor nor continuation.
    // Representative allocation/initialization/release are serialized with
    // measurement contests; FP32 dependency selection is deliberately outside
    // that lock, and the lock is not held across the whole cache callback.
    return fp8RKSelectionCache().getOrTune(key, [&] {
        if (representative_input != nullptr) {
            return autotuneIsolatedRKSite(problem, site, operation, original_reduction_size,
                                          output_scale, *representative_input, stream,
                                          workspace_budget_bytes);
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
                                                 temporary, stream, workspace_budget_bytes);
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

Fp8RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem, DenseReductionSite site, CubReductionOp operation,
    uint64_t original_reduction_size, float output_scale, DataType input_dtype,
    const Stream& stream, uint64_t workspace_budget_bytes) {
    return selectCachedDescriptorImpl(problem, site, operation, original_reduction_size,
                                      output_scale, input_dtype, nullptr, stream, workspace_budget_bytes);
}

Fp8RKSelection selectCachedIsolatedRKSite(
    const DenseReductionProblem& problem, DenseReductionSite site, CubReductionOp operation,
    uint64_t original_reduction_size, float output_scale, const Tensor& representative_input,
    const Stream& stream, uint64_t workspace_budget_bytes) {
    if (!representative_input.isInitialized())
        throw std::invalid_argument("FP8 RK cached selection requires initialized input");
    return selectCachedDescriptorImpl(problem, site, operation, original_reduction_size,
                                      output_scale, representative_input.getDataType(),
                                      &representative_input, stream, workspace_budget_bytes);
}

} // namespace ThorImplementation::CubReductionInternal::Fp8RK
