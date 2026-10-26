// GPU integration smoke: real retained physical FP8 RK candidates, complete
// FP32 continuations and cached replay. No new CUDA kernels are compiled.
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKStampAutotuner.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectionCache.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectedContinuationBenchmark.h"
#include "Utilities/TensorOperations/Cub/DenseProductionPhysicalSelectionContext.h"
#include "Utilities/TensorOperations/Cub/CubReduction.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

using namespace ThorImplementation;
namespace Cub = ThorImplementation::CubReductionInternal;
namespace RK = ThorImplementation::CubReductionInternal::Fp8RK;

int main() {
    try {
        const TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
        Stream stream(placement);
        constexpr uint64_t O = 16, R = 512, K = 64;
        for (const DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            Tensor input(placement, TensorDescriptor(dtype, {O, R, K}));
            input.fill(1.0f, stream);
            Cub::DenseReductionOutputSpec spec;
            spec.value_dtype = DataType::FP32;
            const auto problem = Cub::makeInitialDenseReductionProblem(
                {O, R, K}, {1}, Cub::DenseReductionAggregateKind::Value, spec);
            const auto sites = Cub::enumerateDenseReductionSites(problem);
            if (sites.size() != 1) throw std::logic_error("Expected one RK site");
            auto tuned = RK::autotuneIsolatedRKSite(
                problem, sites.front(), CubReductionOp::Sum, R, 1.0f,
                input, stream, uint64_t{256} * 1024 * 1024);
            if (!tuned.candidate_count || tuned.candidate_count > 28 ||
                tuned.winner_index >= tuned.candidate_count)
                throw std::logic_error("FP8 RK autotune shortlist/continuation contract failed");
            Tensor output(placement, TensorDescriptor(DataType::FP32, {O, K}));
            // A caller may provide either squeezed or keep-dimension output
            // shape. Both handles alias the same allocation; no candidate
            // owns (or allocates) a separate output tensor.
            Tensor keepdims_output = output;
            keepdims_output.reshape({O, 1, K});
            if (keepdims_output.getMemPtr<void>() != output.getMemPtr<void>())
                throw std::logic_error("FP8 RK output view unexpectedly changed backing storage");
            // Exercise the live benchmark builder directly: there is no
            // benchmark-only stamp materialization API or independent FP32
            // continuation selector to preserve for this smoke.
            const auto semantics = Cub::makeValueReductionStageSemantics(
                CubReductionOp::Sum, Cub::DensePassRole::Complete, R);
            auto isolated = RK::planIsolatedRKSite(
                problem, sites.front(), tuned.physical, semantics);
            const Cub::DenseProductionPhysicalSelectionContext fp32_context(
                dtype, Cub::denseRKValueOperation(CubReductionOp::Sum), stream);
            auto site_executable = RK::makeSelectedIsolatedBenchmark(
                problem, sites.front(), std::move(isolated), tuned.physical,
                fp32_context.context(), 1.0f, uint64_t{256} * 1024 * 1024, stream);
            if (Cub::hasDenseReductionSites(site_executable.site.remaining_problem))
                throw std::logic_error("Selected site left reduction axes unexpectedly");
            Tensor site_workspace;
            if (site_executable.scratch_bytes) {
                site_workspace = Tensor(placement, TensorDescriptor(
                    DataType::UINT8, std::vector<uint64_t>{static_cast<uint64_t>(site_executable.scratch_bytes)}));
            }
            site_executable.launch(input, output, site_workspace, stream);
            site_executable.launch(input, keepdims_output, site_workspace, stream);
            std::vector<float> actual(O * K);
            const auto error = cudaMemcpyAsync(actual.data(), output.getMemPtr<void>(),
                actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
            if (error != cudaSuccess) throw std::runtime_error("FP8 RK smoke download failed");
            stream.synchronize();
            for (float value : actual)
                if (!std::isfinite(value) || std::abs(value - static_cast<float>(R)) > 1e-3f)
                    throw std::logic_error("FP8 RK autotuned result is incorrect");
            // The retained selector is now the ordinary production default.
            // Stamping must put the physical first stage in the normal pass list.
            RK::fp8RKSelectionCache().clear();
            const auto attempts_before = RK::fp8RKSelectionCache().tuningAttempts();
            auto stamped = CubReduction(CubReductionOp::Sum, {1}, DataType::FP32)
                               .stamp(input, output, stream);
            if (RK::fp8RKSelectionCache().tuningAttempts() != attempts_before + 1 ||
                RK::fp8RKSelectionCache().size() != 1)
                throw std::logic_error("First production FP8 RK stamp must populate exactly one selection recipe");
            Tensor independent_output(placement, TensorDescriptor(DataType::FP32, {O, K}));
            auto independently_stamped = CubReduction(CubReductionOp::Sum, {1}, DataType::FP32)
                                             .stamp(input, independent_output, stream);
            if (RK::fp8RKSelectionCache().tuningAttempts() != attempts_before + 1 ||
                !independently_stamped->usesDenseExecutablePlan())
                throw std::logic_error("Identical independent stamp unexpectedly retuned");
            independently_stamped->runOn(stream);
            std::vector<float> independently_actual(O * K);
            const auto second_status = cudaMemcpyAsync(independently_actual.data(),
                independent_output.getMemPtr<void>(), independently_actual.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream.getStream());
            if (second_status != cudaSuccess) throw std::runtime_error("Independent stamped output download failed");
            stream.synchronize();
            for (float value : independently_actual)
                if (!std::isfinite(value) || std::abs(value - static_cast<float>(R)) > 1e-3f)
                    throw std::logic_error("Independent cached-selection stamp produced an incorrect result");
            // Eviction of the *selection recipe* must not invalidate the
            // already materialized stamped executors or their workspaces.
            RK::fp8RKSelectionCache().clear();
            independently_stamped->runOn(stream);
            const auto replay_status = cudaMemcpyAsync(independently_actual.data(),
                independent_output.getMemPtr<void>(), independently_actual.size() * sizeof(float),
                cudaMemcpyDeviceToHost, stream.getStream());
            if (replay_status != cudaSuccess) throw std::runtime_error("Cache-eviction replay download failed");
            stream.synchronize();
            for (float value : independently_actual)
                if (!std::isfinite(value) || std::abs(value - static_cast<float>(R)) > 1e-3f)
                    throw std::logic_error("Eviction invalidated a stamped FP8 RK executor");
            if (!stamped->usesDenseExecutablePlan() ||
                stamped->getDenseExecutablePassCount() == 0 ||
                stamped->getModernRKStrategyChain().find("retained_fp8_rk:") != 0)
                throw std::logic_error("Production FP8 RK stamp did not materialize retained physical passes");
            stamped->runOn(stream);
            const auto checkStamped = [&](float expected) {
                const auto status = cudaMemcpyAsync(actual.data(), output.getMemPtr<void>(),
                    actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
                if (status != cudaSuccess) throw std::runtime_error("FP8 RK stamped output download failed");
                stream.synchronize();
                for (float value : actual)
                    if (!std::isfinite(value) || std::abs(value - expected) > 1e-3f)
                        throw std::logic_error("Production FP8 RK stamped execution returned wrong result");
            };
            checkStamped(static_cast<float>(R));
            if (!stamped->usesDenseExecutablePlan() || stamped->getDenseExecutablePassCount() < 1)
                throw std::logic_error("Retained FP8 RK stamp failed to use the ordinary pass list");
            if (stamped->getWorkspaceSizeInBytes() > uint64_t{256} * 1024 * 1024)
                throw std::logic_error("Retained FP8 RK tuned workspace exceeds budget");
            // Complete retained FP8 kernels apply runtime scale at final FP32
            // writeback. Staged candidates defer it to their final FP32 pass.
            stamped->runOn(stream, 2.0f);
            checkStamped(static_cast<float>(2 * R));
            stamped->runOn(stream, -0.5f);
            checkStamped(-0.5f * static_cast<float>(R));
            stamped->runOn(stream);
            checkStamped(static_cast<float>(R));
            // Every VALUE operation and floating output dtype is wired through the
            // retained selector: MEAN finalizes once, FP16 output converts once.
            auto mean_stamp = CubReduction(CubReductionOp::Mean, {1}, DataType::FP32)
                                  .stamp(input, output, stream);
            if (mean_stamp->getModernRKStrategyChain().find("retained_fp8_rk:") != 0)
                throw std::logic_error("FP8 RK MEAN did not use the retained physical selector");
            mean_stamp->runOn(stream);
            checkStamped(1.0f);
            Tensor fp16_output(placement, TensorDescriptor(DataType::FP16, {O, K}));
            auto fp16_stamp = CubReduction(CubReductionOp::Sum, {1}, DataType::FP16)
                                  .stamp(input, fp16_output, stream);
            if (fp16_stamp->getModernRKStrategyChain().find("retained_fp8_rk:") != 0)
                throw std::logic_error("FP8 RK FP16-output SUM did not use the retained physical selector");
            fp16_stamp->runOn(stream);
            std::vector<__half> fp16_actual(O * K);
            const auto fp16_status = cudaMemcpyAsync(fp16_actual.data(), fp16_output.getMemPtr<void>(),
                fp16_actual.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream.getStream());
            if (fp16_status != cudaSuccess) throw std::runtime_error("FP8 RK FP16 output download failed");
            stream.synchronize();
            for (const __half value : fp16_actual)
                if (__half2float(value) != static_cast<float>(R))
                    throw std::logic_error("FP8 RK FP16-output SUM returned wrong result");
            // Those stamps tuned their own recipes; restore the cleared-cache state
            // that the eviction checks below rely on.
            RK::fp8RKSelectionCache().clear();
            stamped->runOn(stream);
            checkStamped(static_cast<float>(R));
            // The cache was cleared above. Existing stamps must survive that
            // eviction, but the next *new* stamp must tune one fresh physical
            // recipe. A subsequent identical stamp must then be a cache hit.
            // Neither stamp may fall back to the superseded FP8 RK backend.
            const uint64_t attempts_after_eviction = RK::fp8RKSelectionCache().tuningAttempts();
            auto another = CubReduction(CubReductionOp::Sum, {1}, DataType::FP32)
                               .stamp(input, output, stream);
            if (!another->usesDenseExecutablePlan() ||
                another->getModernRKStrategyChain().find("retained_fp8_rk:") != 0 ||
                RK::fp8RKSelectionCache().tuningAttempts() != attempts_after_eviction + 1 ||
                RK::fp8RKSelectionCache().size() != 1)
                throw std::logic_error("First FP8 RK stamp after cache eviction did not retune one retained recipe");
            another->runOn(stream);
            checkStamped(static_cast<float>(R));
            auto cached_again = CubReduction(CubReductionOp::Sum, {1}, DataType::FP32)
                                    .stamp(input, output, stream);
            if (!cached_again->usesDenseExecutablePlan() ||
                cached_again->getModernRKStrategyChain().find("retained_fp8_rk:") != 0 ||
                RK::fp8RKSelectionCache().tuningAttempts() != attempts_after_eviction + 1 ||
                RK::fp8RKSelectionCache().size() != 1)
                throw std::logic_error("Identical FP8 RK stamp missed the repopulated selection cache");
            cached_again->runOn(stream);
            checkStamped(static_cast<float>(R));
            auto scaled_stamp = CubReduction(CubReductionOp::Sum, {1}, DataType::FP32, 2.0f)
                                    .stamp(input, output, stream);
            scaled_stamp->runOn(stream);
            checkStamped(static_cast<float>(2 * R));
            scaled_stamp->runOn(stream, -1.5f);
            checkStamped(-1.5f * static_cast<float>(R));
            std::cout << "FP8 RK dtype=" << static_cast<int>(dtype)
                      << " candidates=" << tuned.candidate_count
                      << " winner=" << tuned.winner_name
                      << " stamped_passes=" << stamped->getDenseExecutablePassCount() << '\n';
        }
        std::cout << "FP8 RK stamp autotuning smoke passed\n";
    } catch (const std::exception& error) {
        std::cerr << "FP8 RK stamp autotuning smoke failed: " << error.what() << '\n';
        return 1;
    }
}
