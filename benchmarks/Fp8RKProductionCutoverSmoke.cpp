// Public-API regression for default ordinary FP8 RK VALUE production.
// No opt-in flags, hidden continuations, or special stamped executors.
#include "Utilities/TensorOperations/Cub/CubReduction.h"
#include "Utilities/TensorOperations/Cub/Fp8RK/Fp8RKSelectionCache.h"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;
namespace RK = ThorImplementation::CubReductionInternal::Fp8RK;

namespace {
void require(bool result, const char* message) {
    if (!result) throw std::logic_error(message);
}

void checkOutput(const Tensor& output, uint64_t elements, float expected, Stream& stream) {
    std::vector<float> actual(elements);
    const cudaError_t result = cudaMemcpyAsync(actual.data(), output.getMemPtr<void>(),
        elements * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    stream.synchronize();
    for (float value : actual) {
        if (!std::isfinite(value) || std::abs(value - expected) > 1e-2f)
            throw std::logic_error("P2-D ordinary FP8 RK output differs from FP32 reference");
    }
}

void checkExpandedSurface(const Tensor& input, Stream& stream) {
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, stream.getGpuNum());
    Tensor mean_output(placement, TensorDescriptor(DataType::FP32, {16, 1, 64}));
    Tensor half_output(placement, TensorDescriptor(DataType::FP16, {16, 1, 64}));
    const CubReduction mean(CubReductionOp::Mean, {1}, DataType::FP32);
    const CubReduction half_sum(CubReductionOp::Sum, {1}, DataType::FP16);
    auto mean_stamp = mean.stamp(input, mean_output, stream);
    auto half_stamp = half_sum.stamp(input, half_output, stream);
    require(mean_stamp->getModernRKStrategyChain().find("retained_fp8_rk:") == 0 &&
            half_stamp->getModernRKStrategyChain().find("retained_fp8_rk:") == 0,
            "New FP8 RK VALUE semantics must not fall through to legacy production");
    mean_stamp->runOn(stream);
    checkOutput(mean_output, 16 * 64, 0.25f, stream);
    half_stamp->runOn(stream);
    std::vector<__half> half_values(16 * 64);
    const cudaError_t copied = cudaMemcpyAsync(half_values.data(), half_output.getMemPtr<void>(),
        half_values.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream.getStream());
    if (copied != cudaSuccess) throw std::runtime_error(cudaGetErrorString(copied));
    stream.synchronize();
    for (const __half value : half_values)
        require(std::abs(__half2float(value) - 128.0f) < 0.01f,
                "Retained FP8 RK SUM did not convert final output to FP16");
}

void exercise(DataType dtype, bool multi_axis) {
    const TensorPlacement placement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(placement), alternate(placement);
    const std::vector<uint64_t> shape = multi_axis
        ? std::vector<uint64_t>{512, 2, 64}
        : std::vector<uint64_t>{16, 512, 64};
    const std::vector<uint32_t> axes = multi_axis
        ? std::vector<uint32_t>{0, 2}
        : std::vector<uint32_t>{1};
    const std::vector<uint64_t> output_shape = multi_axis
        ? std::vector<uint64_t>{1, 2, 1}
        : std::vector<uint64_t>{16, 1, 64};
    const uint64_t result_elements = multi_axis ? 2 : 16 * 64;
    const float expected = multi_axis ? (512.0f * 64.0f * 0.25f) : (512.0f * 0.25f);
    Tensor input(placement, TensorDescriptor(dtype, shape));
    Tensor output(placement, TensorDescriptor(DataType::FP32, output_shape));
    input.fill(0.25f, stream);
    stream.synchronize();

    const CubReduction reduction(CubReductionOp::Sum, axes, DataType::FP32);
    // Query and stamp must traverse the SAME physical selection channel; a
    // cached first-stage recipe must not force its benchmark continuation.
    const size_t planned_workspace = reduction.queryWorkspaceSizeInBytes(
        TensorDescriptor(dtype, shape), stream);
    auto stamp = reduction.stamp(input, output, stream);
    require(stamp->usesDenseExecutablePlan() && stamp->getDenseExecutablePassCount() > 0,
            "P2-D ordinary FP8 VALUE must use the flat physical-pass executor");
    require(stamp->getWorkspaceSizeInBytes() == planned_workspace,
            "Descriptor and stamped physical workspace do not agree");
    const auto passes = stamp->getDensePhysicalPassDiagnostics();
    require(passes.size() == stamp->getDenseExecutablePassCount(),
            "Physical diagnostics do not describe the complete stamped pass list");
    size_t diagnostic_workspace = 0;
    for (size_t i = 0; i < passes.size(); ++i) {
        const auto& pass = passes[i];
        diagnostic_workspace += pass.intermediate_bytes + pass.scratch_bytes;
        require(!pass.family.empty() && !pass.input_dimensions.empty() &&
                    !pass.output_dimensions.empty(),
                "Stamped pass diagnostics lost physical family or tensor geometry");
        if (pass.intermediate_bytes != 0)
            require(pass.output_dtype == DataType::FP32,
                    "Nonfinal physical output must remain FP32");
        if (i + 1 < passes.size())
            require(pass.output_dtype == passes[i + 1].input_dtype,
                    "Next physical pass must consume the previous output dtype");
    }
    require(diagnostic_workspace == planned_workspace,
            "Physical pass diagnostics double-counted or omitted workspace");
    if (!multi_axis)
        require(passes.front().family.find("retained_fp8_rk:") == 0 &&
                    passes.front().scratch_bytes == 0,
                "Retained first-stage diagnostics have the wrong family or phantom scratch");
    if (!multi_axis)
        require(stamp->getModernRKStrategyChain().find("retained_fp8_rk:") == 0,
                "P2-D original-input FP8 RK fell back to legacy RK");
    else
        require(stamp->getDenseExecutablePassCount() >= 2,
                "Multi-axis reduction should use independently planned physical passes");

    RK::fp8RKSelectionCache().clear();
    const uint64_t attempts = RK::fp8RKSelectionCache().tuningAttempts();
    for (int iteration = 0; iteration < 3; ++iteration)
        stamp->runOn(alternate);
    checkOutput(output, result_elements, expected, alternate);
    stamp->runOn(alternate, -0.5f);
    checkOutput(output, result_elements, expected * -0.5f, alternate);
    stamp->runOn(alternate);
    checkOutput(output, result_elements, expected, alternate);
    require(RK::fp8RKSelectionCache().size() == 0 &&
            RK::fp8RKSelectionCache().tuningAttempts() == attempts,
            "Stamped run must not consult, populate or retune the FP8 selection cache");
    // A nonunit stamp-time scale also applies only on the final mathematical pass.
    const CubReduction scaled_reduction(CubReductionOp::Sum, axes, DataType::FP32, 2.0f);
    auto scaled = scaled_reduction.stamp(input, output, stream);
    require(scaled_reduction.queryWorkspaceSizeInBytes(TensorDescriptor(dtype, shape), stream) ==
                scaled->getWorkspaceSizeInBytes(),
            "Scaled descriptor/stamp workspace accounting diverged");
    scaled->runOn(alternate);
    checkOutput(output, result_elements, expected * 2.0f, alternate);
    scaled->runOn(alternate, -1.5f);
    checkOutput(output, result_elements, expected * -1.5f, alternate);
    if (!multi_axis) {
        // MEAN and FP16 output were temporary cutover restrictions. Exercise
        // their real production execution without a legacy FP8 fallback.
        checkExpandedSurface(input, stream);
    }
    std::cout << "P2-D dtype=" << static_cast<int>(dtype)
              << " multi_axis=" << multi_axis
              << " physical_passes=" << stamp->getDenseExecutablePassCount()
              << " workspace=" << stamp->getWorkspaceSizeInBytes()
              << " strategy=" << stamp->getModernRKStrategyChain() << " passed\n";
}
} // namespace

int main() {
    try {
        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
            exercise(dtype, false);
            exercise(dtype, true);
        }
        std::cout << "P2-D default ordinary FP8 RK production cutover smoke passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P2-D default ordinary FP8 RK production smoke failed: " << error.what() << '\n';
        return 1;
    }
}
