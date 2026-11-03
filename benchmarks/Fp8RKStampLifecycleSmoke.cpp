// Phase 3: integration regression for the selection-cache/stamp/run boundary.
// This is a host compilation unit linked to Thor; it instantiates no CUDA kernels.
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/DenseReductionSelectionCache.h"
#include "Utilities/TensorOperations/Cub/DenseRetainedFp8RKPhysicalPlan.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ThorImplementation {
// Test-only visibility into stamp-owned physical pass resources. The
// selection cache contains an immutable first-kernel recipe only.
struct Fp8RKStampLifecycleProbe {
    using Pass = CubReductionInternal::DenseExecutablePass;
    static const std::vector<Pass>& passes(const StampedCubReduction& stamp) {
        return stamp.dense_executable_plan.passes;
    }
    static const CubReductionInternal::Fp8RK::PhysicalKernel& physical(
        const StampedCubReduction& stamp) {
        const auto& selected = passes(stamp);
        if (selected.empty()) throw std::logic_error("FP8 stamp has no physical passes");
        const auto* first = dynamic_cast<const CubReductionInternal::DenseRetainedFp8RKPhysicalPlan*>(
            selected.front().physical_plan.get());
        if (!first) throw std::logic_error("FP8 stamp did not select retained first physical kernel");
        return first->kernel;
    }
};
}  // namespace ThorImplementation

using namespace ThorImplementation;
namespace RK = ThorImplementation::CubReductionInternal::Fp8RK;

namespace {
constexpr uint64_t kO = 16;
constexpr uint64_t kR = 512;
constexpr uint64_t kK = 64;

void require(bool condition, const char* message) {
    if (!condition) throw std::logic_error(message);
}

void checkOutput(Tensor& output, Stream& stream, float expected) {
    std::vector<float> values(kO * kK);
    const cudaError_t status = cudaMemcpyAsync(values.data(), output.getMemPtr<void>(),
                                                values.size() * sizeof(float),
                                                cudaMemcpyDeviceToHost, stream.getStream());
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("FP8 RK lifecycle download failed: ") +
                                 cudaGetErrorString(status));
    stream.synchronize();
    for (float value : values)
        if (!std::isfinite(value) || std::abs(value - expected) > 1e-3f)
            throw std::logic_error("FP8 RK independent stamped result incorrect");
}

// This invariant holds for stamps on either side of a cache clear. They may
// materialize *different* winning kernels after a fresh tuning contest.
void requireIndependentExecution(const StampedCubReduction& a,
                                 const StampedCubReduction& b) {
    require(a.usesDenseExecutablePlan() && b.usesDenseExecutablePlan(),
            "Retained FP8 RK stamps must use the ordinary physical-pass sequence");
    const auto& ap = Fp8RKStampLifecycleProbe::passes(a);
    const auto& bp = Fp8RKStampLifecycleProbe::passes(b);
    require(!ap.empty() && !bp.empty() && ap.data() != bp.data(),
            "Independent stamps must own different physical-pass sequences");

    const auto checkWorkspace = [](const StampedCubReduction& stamp) {
        size_t total = 0;
        for (const auto& pass : Fp8RKStampLifecycleProbe::passes(stamp)) {
            require(pass.temp_storage.isInitialized() == (pass.temp_storage_bytes != 0),
                    "Pass scratch allocation differs from its query");
            require((pass.intermediate_bytes != 0) || pass.output.isInitialized(),
                    "Selected physical output has no storage");
            total += pass.workspaceBytes();
        }
        require(total == stamp.getWorkspaceSizeInBytes(),
                "Stamp workspace must equal owned physical pass requirements");
        size_t described = 0;
        const auto diagnostics = stamp.getDensePhysicalPassDiagnostics();
        require(diagnostics.size() == Fp8RKStampLifecycleProbe::passes(stamp).size(),
                "Stamp physical diagnostics did not preserve pass count");
        for (const auto& pass : diagnostics)
            described += pass.intermediate_bytes + pass.scratch_bytes;
        require(described == total,
                "Stamp physical diagnostics disagree with the actual owned buffers");
    };
    checkWorkspace(a);
    checkWorkspace(b);
    // After cache-clear retuning the two winners may have DIFFERENT numbers
    // of passes. Compare allocated regions regardless of their pass indices.
    for (const auto& left : ap) {
        for (const auto& right : bp) {
            if (left.intermediate_bytes != 0 && right.intermediate_bytes != 0)
                require(left.output.getMemPtr<void>() != right.output.getMemPtr<void>(),
                        "Independent stamps share mutable FP32 intermediate output");
            if (left.temp_storage_bytes != 0 && right.temp_storage_bytes != 0)
                require(left.temp_storage.getMemPtr<void>() != right.temp_storage.getMemPtr<void>(),
                        "Independent stamps share physical scratch storage");
        }
    }
}

// Only a cache HIT is required to reproduce exactly the cached selection.
// After clear() a fresh timing contest can legitimately select another family
// or specialization, even with the same geometry and legal candidate count.
void requireSameCachedSelection(const StampedCubReduction& a,
                                const StampedCubReduction& b) {
    if (Fp8RKStampLifecycleProbe::physical(a) != Fp8RKStampLifecycleProbe::physical(b) ||
        a.getModernRKStrategyChain() != b.getModernRKStrategyChain()) {
        throw std::logic_error(
            "Cache hit changed the selected FP8 RK kernel: first=" +
            a.getModernRKStrategyChain() + " second=" + b.getModernRKStrategyChain());
    }
    require(a.getWorkspaceSizeInBytes() == b.getWorkspaceSizeInBytes(),
            "Stamps sharing a cache recipe must have the same scratch requirement");
}

void exercise(DataType dtype) {
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream_a(placement);
    Stream stream_b(placement);
    require(stream_a.getStream() != stream_b.getStream(),
            "Independent execution test requires distinct CUDA streams");

    Tensor input_a(placement, TensorDescriptor(dtype, {kO, kR, kK}));
    Tensor input_b(placement, TensorDescriptor(dtype, {kO, kR, kK}));
    Tensor output_a(placement, TensorDescriptor(DataType::FP32, {kO, kK}));
    Tensor output_b(placement, TensorDescriptor(DataType::FP32, {kO, kK}));
    input_a.fill(1.0f, stream_a);
    input_b.fill(-0.5f, stream_b);
    // No other GPU launches should overlap the cold-cache tuning contest.
    stream_a.synchronize();
    stream_b.synchronize();

    ThorImplementation::CubReductionInternal::denseReductionSelectionCache().clear();
    const uint64_t attempts_before = ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts(ThorImplementation::CubReductionInternal::DenseReductionSelectionKind::Fp8RK);
    auto a = CubReduction(CubReductionOp::Sum, {1}, DataType::FP32)
                 .stamp(input_a, output_a, stream_a);
    require(ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts(ThorImplementation::CubReductionInternal::DenseReductionSelectionKind::Fp8RK) == attempts_before + 1 &&
            ThorImplementation::CubReductionInternal::denseReductionSelectionCache().size() == 1,
            "First stamp must tune once and cache only a selection recipe");
    auto b = CubReduction(CubReductionOp::Sum, {1}, DataType::FP32)
                 .stamp(input_b, output_b, stream_b);
    require(ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts(ThorImplementation::CubReductionInternal::DenseReductionSelectionKind::Fp8RK) == attempts_before + 1,
            "Second independent stamp with identical configuration must be a cache hit");
    requireIndependentExecution(*a, *b);
    requireSameCachedSelection(*a, *b);
    require(a->getInputTensor().getMemPtr<void>() != b->getInputTensor().getMemPtr<void>() &&
            a->getOutputTensor().getMemPtr<void>() != b->getOutputTensor().getMemPtr<void>(),
            "Independent stamps must bind their own input and output storage");

    // Clear selection recipes while both stamps remain alive. Running these
    // stamps must neither consult nor repopulate the recipe cache.
    ThorImplementation::CubReductionInternal::denseReductionSelectionCache().clear();
    require(ThorImplementation::CubReductionInternal::denseReductionSelectionCache().size() == 0, "Selection clear must empty the cache");
    const uint64_t before_runs = ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts(ThorImplementation::CubReductionInternal::DenseReductionSelectionKind::Fp8RK);
    auto launch_a = std::async(std::launch::async, [&] {
        for (int iteration = 0; iteration < 16; ++iteration) a->runOn(stream_a);
        checkOutput(output_a, stream_a, static_cast<float>(kR));
    });
    auto launch_b = std::async(std::launch::async, [&] {
        for (int iteration = 0; iteration < 16; ++iteration) b->runOn(stream_b);
        checkOutput(output_b, stream_b, -0.5f * static_cast<float>(kR));
    });
    // Propagate failures from both threads even if the first future throws.
    std::exception_ptr failure;
    try { launch_a.get(); } catch (...) { failure = std::current_exception(); }
    try { launch_b.get(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
    require(ThorImplementation::CubReductionInternal::denseReductionSelectionCache().size() == 0 &&
            ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts(ThorImplementation::CubReductionInternal::DenseReductionSelectionKind::Fp8RK) == before_runs,
            "Executing existing stamps must not access, populate, or retune selection cache");

    // A new stamp after clearing must tune anew, without disturbing the old
    // stamp objects. All three remain independently executable.
    Tensor output_c(placement, TensorDescriptor(DataType::FP32, {kO, kK}));
    auto c = CubReduction(CubReductionOp::Sum, {1}, DataType::FP32)
                 .stamp(input_a, output_c, stream_a);
    require(ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts(ThorImplementation::CubReductionInternal::DenseReductionSelectionKind::Fp8RK) == before_runs + 1 &&
            ThorImplementation::CubReductionInternal::denseReductionSelectionCache().size() == 1,
            "Fresh stamp after cache clear must retune and retain one recipe");
    // c was tuned after clear(), so it need not match a/b's physical winner.
    requireIndependentExecution(*a, *c);
    requireIndependentExecution(*b, *c);
    b->runOn(stream_b);
    c->runOn(stream_a);
    checkOutput(output_b, stream_b, -0.5f * static_cast<float>(kR));
    checkOutput(output_c, stream_a, static_cast<float>(kR));
    const uint64_t final_attempts = ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts(ThorImplementation::CubReductionInternal::DenseReductionSelectionKind::Fp8RK);
    b.reset();  // Destruction of one stamped instance must not affect peers.
    a->runOn(stream_a);
    checkOutput(output_a, stream_a, static_cast<float>(kR));
    require(ThorImplementation::CubReductionInternal::denseReductionSelectionCache().tuningAttempts(ThorImplementation::CubReductionInternal::DenseReductionSelectionKind::Fp8RK) == final_attempts,
            "Previously stamped reduction unexpectedly retuned after a later cache miss");

    std::cout << "FP8 RK lifecycle dtype=" << static_cast<int>(dtype)
              << " scratch_bytes=" << a->getWorkspaceSizeInBytes()
              << " concurrent_stamps=2 cache_misses=2 passed\n";
}
}  // namespace

int main() {
    try {
        exercise(DataType::FP8_E4M3);
        exercise(DataType::FP8_E5M2);
        std::cout << "FP8 RK stamp lifecycle smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FP8 RK stamp lifecycle smoke failed: " << error.what() << '\n';
        return 1;
    }
}
