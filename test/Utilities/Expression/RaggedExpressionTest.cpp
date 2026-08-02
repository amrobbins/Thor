#include "Utilities/Expression/RaggedExpression.h"

#include "Utilities/Expression/AutoDiff.h"

#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

RaggedTensorDescriptor makeDescriptor(DataType values_dtype = DataType::FP32,
                                      std::vector<uint64_t> trailing_dimensions = {4},
                                      uint64_t batch_size = 3,
                                      uint64_t max_total_values = 9,
                                      DataType offsets_dtype = DataType::UINT32) {
    return RaggedTensorDescriptor(values_dtype, trailing_dimensions, batch_size, max_total_values, offsets_dtype);
}

ExprNode outputNode(const Expression& expression) {
    const PhysicalExpression physical = expression.expression();
    return physical.nodes.at(physical.output_node);
}

struct MarkedValueNodes {
    ExprNode marker;
    ExprNode values;
};

MarkedValueNodes markedValueNodes(const Expression& expression) {
    const PhysicalExpression physical = expression.expression();
    const ExprNode marker = physical.nodes.at(physical.output_node);
    if (marker.op != ExprOp::RAGGED_VALUEWISE_EXTENT) {
        throw std::runtime_error("test expected a ragged runtime extent marker.");
    }
    return MarkedValueNodes{marker, physical.nodes.at(marker.lhs)};
}

#define REQUIRE_CUDA_DEVICE()                                                                                          \
    do {                                                                                                                \
        int cuda_device_count_for_test = 0;                                                                             \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                       \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                    \
            GTEST_SKIP() << "CUDA device is required for ragged expression execution tests.";                          \
        }                                                                                                               \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dim : tensor.getDimensions()) {
        numel *= dim;
    }
    return numel;
}

template <typename T>
DataType dtypeFor();

template <>
DataType dtypeFor<float>() {
    return DataType::FP32;
}

template <>
DataType dtypeFor<uint32_t>() {
    return DataType::UINT32;
}

template <>
DataType dtypeFor<uint64_t>() {
    return DataType::UINT64;
}

template <typename T>
Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), dims));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    }
    T* ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(dtypeFor<T>(), dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

template <typename T>
void overwriteGpuTensor(Tensor& gpu, const std::vector<T>& values, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtypeFor<T>(), gpu.getDimensions()));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("overwriteGpuTensor value count mismatch.");
    }
    T* ptr = cpu.getMemPtr<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
}


Tensor makeGpuTensorFromFloats(const std::vector<uint64_t>& dims,
                               const std::vector<float>& values,
                               DataType dtype,
                               Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(dtype, dims));
    if (tensorNumel(cpu) != values.size()) {
        throw std::runtime_error("makeGpuTensorFromFloats value count mismatch.");
    }
    switch (dtype) {
        case DataType::FP16: {
            __half* ptr = cpu.getMemPtr<__half>();
            for (size_t i = 0; i < values.size(); ++i) ptr[i] = __float2half(values[i]);
            break;
        }
        case DataType::BF16: {
            __nv_bfloat16* ptr = cpu.getMemPtr<__nv_bfloat16>();
            for (size_t i = 0; i < values.size(); ++i) ptr[i] = __float2bfloat16(values[i]);
            break;
        }
        case DataType::FP32: {
            float* ptr = cpu.getMemPtr<float>();
            for (size_t i = 0; i < values.size(); ++i) ptr[i] = values[i];
            break;
        }
        default:
            throw std::runtime_error("makeGpuTensorFromFloats test helper supports FP16/BF16/FP32 only.");
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(dtype, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyToCpuFloatValues(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, gpu.getDimensions()));
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const float* ptr = cpu.getMemPtr<float>();
    return std::vector<float>(ptr, ptr + cpu.getTotalNumElements());
}

std::vector<float> copyToCpuValues(const Tensor& gpu, Stream& stream) {
    Tensor cpu = gpu.clone(cpuPlacement);
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const float* ptr = cpu.getMemPtr<float>();
    return std::vector<float>(ptr, ptr + cpu.getTotalNumElements());
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], atol) << "index " << i;
    }
}

std::vector<float> cpuSegmentSoftmax(const std::vector<float>& values,
                                     const std::vector<uint64_t>& offsets,
                                     bool log_softmax) {
    std::vector<float> output(values.size(), 0.0F);
    if (offsets.empty()) {
        return output;
    }

    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        if (begin == end) {
            continue;
        }

        float row_max = values.at(begin);
        for (uint64_t i = begin + 1; i < end; ++i) {
            row_max = std::max(row_max, values.at(i));
        }

        double denominator = 0.0;
        for (uint64_t i = begin; i < end; ++i) {
            denominator += std::exp(static_cast<double>(values.at(i) - row_max));
        }
        const double log_denominator = std::log(denominator);

        for (uint64_t i = begin; i < end; ++i) {
            if (log_softmax) {
                output.at(i) = static_cast<float>(static_cast<double>(values.at(i) - row_max) - log_denominator);
            } else {
                output.at(i) = static_cast<float>(std::exp(static_cast<double>(values.at(i) - row_max)) / denominator);
            }
        }
    }
    return output;
}

double cpuWeightedSegmentSoftmaxObjective(const std::vector<float>& values,
                                         const std::vector<uint64_t>& offsets,
                                         const std::vector<float>& upstream,
                                         bool log_softmax) {
    const std::vector<float> output = cpuSegmentSoftmax(values, offsets, log_softmax);
    double objective = 0.0;
    const uint64_t active_values = offsets.empty() ? 0 : offsets.back();
    for (uint64_t i = 0; i < active_values; ++i) {
        objective += static_cast<double>(output.at(i)) * static_cast<double>(upstream.at(i));
    }
    return objective;
}

std::vector<float> finiteDifferenceSegmentSoftmaxGradient(const std::vector<float>& values,
                                                          const std::vector<uint64_t>& offsets,
                                                          const std::vector<float>& upstream,
                                                          bool log_softmax,
                                                          float epsilon = 1.0e-3F) {
    std::vector<float> gradient(values.size(), 0.0F);
    const uint64_t active_values = offsets.empty() ? 0 : offsets.back();
    for (uint64_t i = 0; i < active_values; ++i) {
        std::vector<float> plus = values;
        std::vector<float> minus = values;
        plus.at(i) += epsilon;
        minus.at(i) -= epsilon;
        const double plus_objective = cpuWeightedSegmentSoftmaxObjective(plus, offsets, upstream, log_softmax);
        const double minus_objective = cpuWeightedSegmentSoftmaxObjective(minus, offsets, upstream, log_softmax);
        gradient.at(i) = static_cast<float>((plus_objective - minus_objective) / (2.0 * static_cast<double>(epsilon)));
    }
    return gradient;
}

std::vector<float> cpuVectorSegmentSoftmax(const std::vector<float>& values,
                                           const std::vector<uint64_t>& offsets,
                                           uint64_t elements_per_value,
                                           bool log_softmax) {
    std::vector<float> output(values.size(), 0.0F);
    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        for (uint64_t component = 0; component < elements_per_value; ++component) {
            if (begin == end) continue;
            float row_max = values.at(begin * elements_per_value + component);
            for (uint64_t value_index = begin + 1; value_index < end; ++value_index) {
                row_max = std::max(row_max, values.at(value_index * elements_per_value + component));
            }
            double denominator = 0.0;
            for (uint64_t value_index = begin; value_index < end; ++value_index) {
                denominator += std::exp(
                    static_cast<double>(values.at(value_index * elements_per_value + component) - row_max));
            }
            const double log_denominator = std::log(denominator);
            for (uint64_t value_index = begin; value_index < end; ++value_index) {
                const double shifted = static_cast<double>(values.at(value_index * elements_per_value + component) - row_max);
                output.at(value_index * elements_per_value + component) = log_softmax
                    ? static_cast<float>(shifted - log_denominator)
                    : static_cast<float>(std::exp(shifted) / denominator);
            }
        }
    }
    return output;
}

double cpuWeightedVectorSegmentSoftmaxObjective(const std::vector<float>& values,
                                                 const std::vector<uint64_t>& offsets,
                                                 uint64_t elements_per_value,
                                                 const std::vector<float>& upstream,
                                                 bool log_softmax) {
    const std::vector<float> output = cpuVectorSegmentSoftmax(values, offsets, elements_per_value, log_softmax);
    double objective = 0.0;
    const uint64_t active_elements = (offsets.empty() ? 0 : offsets.back()) * elements_per_value;
    for (uint64_t i = 0; i < active_elements; ++i) {
        objective += static_cast<double>(output.at(i)) * static_cast<double>(upstream.at(i));
    }
    return objective;
}

std::vector<float> finiteDifferenceVectorSegmentSoftmaxGradient(const std::vector<float>& values,
                                                                const std::vector<uint64_t>& offsets,
                                                                uint64_t elements_per_value,
                                                                const std::vector<float>& upstream,
                                                                bool log_softmax,
                                                                float epsilon = 1.0e-3F) {
    std::vector<float> gradient(values.size(), 0.0F);
    const uint64_t active_elements = (offsets.empty() ? 0 : offsets.back()) * elements_per_value;
    for (uint64_t i = 0; i < active_elements; ++i) {
        std::vector<float> plus = values;
        std::vector<float> minus = values;
        plus.at(i) += epsilon;
        minus.at(i) -= epsilon;
        const double plus_objective =
            cpuWeightedVectorSegmentSoftmaxObjective(plus, offsets, elements_per_value, upstream, log_softmax);
        const double minus_objective =
            cpuWeightedVectorSegmentSoftmaxObjective(minus, offsets, elements_per_value, upstream, log_softmax);
        gradient.at(i) = static_cast<float>((plus_objective - minus_objective) / (2.0 * static_cast<double>(epsilon)));
    }
    return gradient;
}

double cpuWeightedVectorSegmentMinMaxObjective(const std::vector<float>& values,
                                                const std::vector<uint64_t>& offsets,
                                                uint64_t elements_per_value,
                                                const std::vector<float>& upstream,
                                                bool minimum) {
    double objective = 0.0;
    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        for (uint64_t component = 0; component < elements_per_value; ++component) {
            if (begin == end) continue;
            float winner = values.at(begin * elements_per_value + component);
            for (uint64_t value_index = begin + 1; value_index < end; ++value_index) {
                const float candidate = values.at(value_index * elements_per_value + component);
                winner = minimum ? std::min(winner, candidate) : std::max(winner, candidate);
            }
            objective += static_cast<double>(winner) *
                         static_cast<double>(upstream.at(row * elements_per_value + component));
        }
    }
    return objective;
}

std::vector<float> finiteDifferenceVectorSegmentMinMaxGradient(const std::vector<float>& values,
                                                               const std::vector<uint64_t>& offsets,
                                                               uint64_t elements_per_value,
                                                               const std::vector<float>& upstream,
                                                               bool minimum,
                                                               float epsilon = 1.0e-3F) {
    std::vector<float> gradient(values.size(), 0.0F);
    const uint64_t active_elements = (offsets.empty() ? 0 : offsets.back()) * elements_per_value;
    for (uint64_t i = 0; i < active_elements; ++i) {
        std::vector<float> plus = values;
        std::vector<float> minus = values;
        plus.at(i) += epsilon;
        minus.at(i) -= epsilon;
        const double plus_objective =
            cpuWeightedVectorSegmentMinMaxObjective(plus, offsets, elements_per_value, upstream, minimum);
        const double minus_objective =
            cpuWeightedVectorSegmentMinMaxObjective(minus, offsets, elements_per_value, upstream, minimum);
        gradient.at(i) = static_cast<float>((plus_objective - minus_objective) / (2.0 * static_cast<double>(epsilon)));
    }
    return gradient;
}

double cpuWeightedSegmentMinMaxObjective(const std::vector<float>& values,
                                         const std::vector<uint64_t>& offsets,
                                         const std::vector<float>& upstream,
                                         bool minimum) {
    double objective = 0.0;
    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        if (begin == end) {
            continue;
        }
        float winner = values.at(begin);
        for (uint64_t i = begin + 1; i < end; ++i) {
            winner = minimum ? std::min(winner, values.at(i)) : std::max(winner, values.at(i));
        }
        objective += static_cast<double>(winner) * static_cast<double>(upstream.at(row));
    }
    return objective;
}

std::vector<float> finiteDifferenceSegmentMinMaxGradient(const std::vector<float>& values,
                                                         const std::vector<uint64_t>& offsets,
                                                         const std::vector<float>& upstream,
                                                         bool minimum,
                                                         float epsilon = 1.0e-3F) {
    std::vector<float> gradient(values.size(), 0.0F);
    const uint64_t active_values = offsets.empty() ? 0 : offsets.back();
    for (uint64_t i = 0; i < active_values; ++i) {
        std::vector<float> plus = values;
        std::vector<float> minus = values;
        plus.at(i) += epsilon;
        minus.at(i) -= epsilon;
        const double plus_objective = cpuWeightedSegmentMinMaxObjective(plus, offsets, upstream, minimum);
        const double minus_objective = cpuWeightedSegmentMinMaxObjective(minus, offsets, upstream, minimum);
        gradient.at(i) = static_cast<float>((plus_objective - minus_objective) / (2.0 * static_cast<double>(epsilon)));
    }
    return gradient;
}

Tensor runExpressionOutput(const Expression& expression,
                           const std::unordered_map<std::string, Tensor>& inputs,
                           const std::string& output_name,
                           Stream& stream,
                           const std::optional<Tensor>& preallocated_output = std::nullopt) {
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{output_name, expression}}).physicalOutputs(), 0);
    std::unordered_map<std::string, Tensor> preallocated;
    if (preallocated_output.has_value()) {
        preallocated.emplace(output_name, preallocated_output.value());
    }
    StampedExecutionPlan plan = equation.stamp(inputs, stream, {}, preallocated);
    plan.run();

    // Keep the stamped plan and its operation-specific temporary storage alive
    // until the enqueued work has completed. In particular, CUB segmented
    // reductions retain their temporary storage through execution. Destroying
    // the plan before synchronization can surface as a cudaFree failure during
    // stack unwinding instead of a useful test failure at the operation site.
    stream.synchronize();
    return plan.output(output_name);
}

Tensor runBackwardOutput(const Expression& forward_expression,
                         const std::unordered_map<std::string, Tensor>& inputs,
                         const std::string& wrt_name,
                         const std::string& upstream_input_name,
                         Stream& stream,
                         const std::optional<Tensor>& preallocated_output = std::nullopt) {
    FusedEquation forward =
        FusedEquation::compile(Expression::outputs({{"y", forward_expression}}).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({wrt_name}, upstream_input_name);

    const std::string output_name = wrt_name + "_grad";
    std::unordered_map<std::string, Tensor> preallocated;
    if (preallocated_output.has_value()) {
        preallocated.emplace(output_name, preallocated_output.value());
    }

    StampedExecutionPlan plan = backward.stamp(inputs, stream, {}, preallocated);
    plan.run();
    stream.synchronize();
    return plan.output(output_name);
}

bool containsOp(const PhysicalOutputs& outputs, ExprOp op) {
    if (!outputs.expr) {
        return false;
    }
    for (const ExprNode& node : outputs.expr->nodes) {
        if (node.op == op) {
            return true;
        }
    }
    return false;
}

void resolveRaggedBackwardTestDTypes(PhysicalOutputs& outputs, DataType offsets_dtype) {
    if (!outputs.expr) {
        throw std::runtime_error("resolveRaggedBackwardTestDTypes requires non-null outputs.expr.");
    }

    std::vector<DataType> input_dtypes(outputs.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : outputs.expr->inputs) {
        if (input.slot >= input_dtypes.size()) {
            throw std::runtime_error("ragged backward test input slot is out of range.");
        }
        if (input.name == "x.offsets") {
            input_dtypes[input.slot] = offsets_dtype;
        }
    }
    resolveOutputsDTypesInPlace(outputs, input_dtypes);
}

template <typename OffsetT>
void runSegmentSoftmaxAutodiffCase(bool log_softmax, float unused_gradient_sentinel) {
    Stream stream(0);

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 8;
    const RaggedExpression ragged = RaggedExpression::input(
        "x", makeDescriptor(DataType::FP32, {}, batch_size, max_total_values, dtypeFor<OffsetT>()));

    const std::vector<float> values_host{0.7F, -1.2F, 2.1F, 0.3F, -0.8F, 50.0F, 60.0F, 70.0F};
    const std::vector<uint64_t> offsets_host{0ULL, 2ULL, 2ULL, 5ULL};
    const std::vector<OffsetT> offsets_typed{
        static_cast<OffsetT>(0), static_cast<OffsetT>(2), static_cast<OffsetT>(2), static_cast<OffsetT>(5)};
    const std::vector<float> upstream_host{0.4F, -1.1F, 0.7F, 1.3F, -0.2F, 17.0F, 19.0F, 23.0F};

    Tensor values = makeGpuTensor<float>({max_total_values}, values_host, stream);
    Tensor offsets = makeGpuTensor<OffsetT>({batch_size + 1}, offsets_typed, stream);
    Tensor upstream = makeGpuTensor<float>({max_total_values}, upstream_host, stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values}));
    gradient.fill(unused_gradient_sentinel, stream);

    const Expression forward =
        log_softmax ? ragged.segment_log_softmax().getValues() : ragged.segment_softmax().getValues();
    const Tensor result = runBackwardOutput(forward,
                                            {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                            "x.values",
                                            "dy",
                                            stream,
                                            gradient);

    std::vector<float> expected = finiteDifferenceSegmentSoftmaxGradient(
        values_host, offsets_host, upstream_host, log_softmax);
    for (uint64_t i = offsets_host.back(); i < max_total_values; ++i) {
        expected.at(i) = unused_gradient_sentinel;
    }
    expectNear(copyToCpuValues(result, stream), expected, log_softmax ? 4.0e-3F : 2.0e-3F);
}

template <typename OffsetT>
void runVectorSegmentSoftmaxAutodiffCase(bool log_softmax, float unused_gradient_sentinel) {
    Stream stream(0);

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 7;
    constexpr uint64_t elements_per_value = 2;
    const RaggedExpression ragged = RaggedExpression::input(
        "x", makeDescriptor(DataType::FP32, {elements_per_value}, batch_size, max_total_values, dtypeFor<OffsetT>()));

    const std::vector<float> values_host{
        0.2F, -0.4F,
        1.1F, 0.3F,
        -0.7F, 1.5F,
        0.6F, -1.2F,
        1.8F, 0.9F,
        99.0F, 100.0F,
        101.0F, 102.0F};
    const std::vector<uint64_t> offsets_host{0ULL, 2ULL, 2ULL, 5ULL};
    const std::vector<OffsetT> offsets_typed{
        static_cast<OffsetT>(0), static_cast<OffsetT>(2), static_cast<OffsetT>(2), static_cast<OffsetT>(5)};
    const std::vector<float> upstream_host{
        0.7F, -1.1F,
        -0.2F, 0.5F,
        1.3F, -0.4F,
        0.6F, 0.8F,
        -0.9F, 1.4F,
        3.0F, 4.0F,
        5.0F, 6.0F};

    Tensor values = makeGpuTensor<float>({max_total_values, elements_per_value}, values_host, stream);
    Tensor offsets = makeGpuTensor<OffsetT>({batch_size + 1}, offsets_typed, stream);
    Tensor upstream = makeGpuTensor<float>({max_total_values, elements_per_value}, upstream_host, stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, elements_per_value}));
    gradient.fill(unused_gradient_sentinel, stream);

    const Expression forward = log_softmax ? ragged.segment_log_softmax().getValues() : ragged.segment_softmax().getValues();
    const Tensor result = runBackwardOutput(forward,
                                            {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                            "x.values",
                                            "dy",
                                            stream,
                                            gradient);

    std::vector<float> expected = finiteDifferenceVectorSegmentSoftmaxGradient(
        values_host, offsets_host, elements_per_value, upstream_host, log_softmax);
    for (uint64_t value_index = offsets_host.back(); value_index < max_total_values; ++value_index) {
        for (uint64_t component = 0; component < elements_per_value; ++component) {
            expected.at(value_index * elements_per_value + component) = unused_gradient_sentinel;
        }
    }
    expectNear(copyToCpuValues(result, stream), expected, 3.0e-3F);
}

template <typename OffsetT>
void runVectorSegmentMinMaxFiniteDifferenceCase(bool minimum, float unused_gradient_sentinel) {
    Stream stream(0);

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 7;
    constexpr uint64_t elements_per_value = 3;
    const RaggedExpression ragged = RaggedExpression::input(
        "x", makeDescriptor(DataType::FP32, {elements_per_value}, batch_size, max_total_values, dtypeFor<OffsetT>()));

    const std::vector<float> values_host{
        2.0F, -1.0F, 0.5F,
        -0.5F, 3.0F, 1.7F,
        0.25F, -2.0F, 4.0F,
        1.2F, 0.6F, -3.0F,
        -1.5F, 2.2F, 0.9F,
        99.0F, 100.0F, 101.0F,
        102.0F, 103.0F, 104.0F};
    const std::vector<uint64_t> offsets_host{0ULL, 2ULL, 2ULL, 5ULL};
    const std::vector<OffsetT> offsets_typed{
        static_cast<OffsetT>(0), static_cast<OffsetT>(2), static_cast<OffsetT>(2), static_cast<OffsetT>(5)};
    const std::vector<float> upstream_host{
        0.7F, -1.1F, 0.4F,
        4.0F, 5.0F, 6.0F,
        -0.3F, 1.2F, 2.5F};

    Tensor values = makeGpuTensor<float>({max_total_values, elements_per_value}, values_host, stream);
    Tensor offsets = makeGpuTensor<OffsetT>({batch_size + 1}, offsets_typed, stream);
    Tensor upstream = makeGpuTensor<float>({batch_size, elements_per_value}, upstream_host, stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, elements_per_value}));
    gradient.fill(unused_gradient_sentinel, stream);

    const Expression forward = minimum ? ragged.segment_min() : ragged.segment_max();
    const Tensor result = runBackwardOutput(forward,
                                            {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                            "x.values",
                                            "dy",
                                            stream,
                                            gradient);

    std::vector<float> expected = finiteDifferenceVectorSegmentMinMaxGradient(
        values_host, offsets_host, elements_per_value, upstream_host, minimum);
    for (uint64_t value_index = offsets_host.back(); value_index < max_total_values; ++value_index) {
        for (uint64_t component = 0; component < elements_per_value; ++component) {
            expected.at(value_index * elements_per_value + component) = unused_gradient_sentinel;
        }
    }
    expectNear(copyToCpuValues(result, stream), expected, 2.0e-3F);
}

template <typename OffsetT>
void runSegmentMinMaxFiniteDifferenceCase(bool minimum, float unused_gradient_sentinel) {
    Stream stream(0);

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 8;
    const RaggedExpression ragged = RaggedExpression::input(
        "x", makeDescriptor(DataType::FP32, {}, batch_size, max_total_values, dtypeFor<OffsetT>()));

    const std::vector<float> values_host{2.0F, -1.0F, 0.25F, 3.0F, -4.0F, 1.5F, 99.0F, 100.0F};
    const std::vector<uint64_t> offsets_host{0ULL, 2ULL, 4ULL, 6ULL};
    const std::vector<OffsetT> offsets_typed{
        static_cast<OffsetT>(0), static_cast<OffsetT>(2), static_cast<OffsetT>(4), static_cast<OffsetT>(6)};
    const std::vector<float> upstream_host{0.7F, -1.2F, 2.5F};

    Tensor values = makeGpuTensor<float>({max_total_values}, values_host, stream);
    Tensor offsets = makeGpuTensor<OffsetT>({batch_size + 1}, offsets_typed, stream);
    Tensor upstream = makeGpuTensor<float>({batch_size}, upstream_host, stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values}));
    gradient.fill(unused_gradient_sentinel, stream);

    const Expression forward = minimum ? ragged.segment_min() : ragged.segment_max();
    const Tensor result = runBackwardOutput(forward,
                                            {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                            "x.values",
                                            "dy",
                                            stream,
                                            gradient);

    std::vector<float> expected =
        finiteDifferenceSegmentMinMaxGradient(values_host, offsets_host, upstream_host, minimum);
    for (uint64_t i = offsets_host.back(); i < max_total_values; ++i) {
        expected.at(i) = unused_gradient_sentinel;
    }
    expectNear(copyToCpuValues(result, stream), expected, 2.0e-3F);
}

template <typename OffsetT>
void runSegmentMinMaxAutodiffCase(bool minimum, float unused_gradient_sentinel) {
    Stream stream(0);

    constexpr uint64_t batch_size = 4;
    constexpr uint64_t max_total_values = 10;
    const RaggedExpression ragged = RaggedExpression::input(
        "x", makeDescriptor(DataType::FP32, {}, batch_size, max_total_values, dtypeFor<OffsetT>()));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> values_host{
        2.0F, -1.0F, -1.0F,
        nan, 5.0F, nan,
        4.0F, 4.0F,
        99.0F, 100.0F};
    const std::vector<OffsetT> offsets_host{
        static_cast<OffsetT>(0), static_cast<OffsetT>(3), static_cast<OffsetT>(3),
        static_cast<OffsetT>(6), static_cast<OffsetT>(8)};
    const std::vector<float> upstream_host{10.0F, 20.0F, 30.0F, 40.0F};

    Tensor values = makeGpuTensor<float>({max_total_values}, values_host, stream);
    Tensor offsets = makeGpuTensor<OffsetT>({batch_size + 1}, offsets_host, stream);
    Tensor upstream = makeGpuTensor<float>({batch_size}, upstream_host, stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values}));
    gradient.fill(unused_gradient_sentinel, stream);

    const Expression forward = minimum ? ragged.segment_min() : ragged.segment_max();
    const Tensor result = runBackwardOutput(forward,
                                            {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                            "x.values",
                                            "dy",
                                            stream,
                                            gradient);

    std::vector<float> expected(max_total_values, 0.0F);
    if (minimum) {
        expected[1] = 10.0F;  // first -1 wins row-0 tie
    } else {
        expected[0] = 10.0F;
    }
    // Row 1 is empty. In row 2 NaNs propagate and the first NaN wins.
    expected[3] = 30.0F;
    // Row 3 is a tie; the lowest packed index wins.
    expected[6] = 40.0F;
    expected[8] = unused_gradient_sentinel;
    expected[9] = unused_gradient_sentinel;

    expectNear(copyToCpuValues(result, stream), expected);
}


}  // namespace

TEST(RaggedExpression, WrapsValuesOffsetsAndBuildsRuntimeExtentAlias) {
    const RaggedTensorDescriptor descriptor = makeDescriptor(DataType::FP32, {4, 2}, 5, 17, DataType::UINT64);
    const Expression values = Expression::input("labels.values");
    const Expression offsets = Expression::input("labels.offsets");

    const RaggedExpression ragged(values, offsets, descriptor);

    EXPECT_TRUE(ragged.isInitialized());
    const MarkedValueNodes marked = markedValueNodes(ragged.getValues());
    EXPECT_EQ(marked.values.op, ExprOp::INPUT);
    EXPECT_EQ(marked.marker.ragged_runtime_batch_size, descriptor.getBatchSize());
    EXPECT_EQ(marked.marker.ragged_runtime_max_active_values, descriptor.getMaxTotalValues());
    EXPECT_EQ(marked.marker.ragged_runtime_elements_per_value, 8ULL);
    EXPECT_TRUE(ragged.getOffsets().isSameLogicalNode(offsets));
    EXPECT_EQ(ragged.getDescriptor(), descriptor);

    const RaggedExpressionRuntimeExtent& extent = ragged.getRuntimeExtent();
    EXPECT_TRUE(extent.isInitialized());
    EXPECT_EQ(extent.maxActiveValues, descriptor.getMaxTotalValues());
    EXPECT_EQ(extent.elementsPerValue, 8);
    EXPECT_EQ(extent.maxLaunchElements(), 17 * 8);

    const ExprNode activeCountNode = outputNode(extent.activeValueCount);
    EXPECT_EQ(activeCountNode.op, ExprOp::STRIDED_VIEW);
    EXPECT_EQ(activeCountNode.view_dims, std::vector<uint64_t>{1});
    EXPECT_EQ(activeCountNode.view_strides, std::vector<uint64_t>{1});
    EXPECT_EQ(activeCountNode.view_element_offset, descriptor.getBatchSize());
}

TEST(RaggedExpression, LogicalInputCreatesValuesAndOffsetsInputs) {
    const RaggedExpression ragged = RaggedExpression::input("labels", makeDescriptor(DataType::FP16, {7}, 2, 11));

    EXPECT_EQ(ragged.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(ragged.getOffsetsDataType(), DataType::UINT32);

    const std::set<std::string> allInputs = ragged.getInputNames();
    EXPECT_TRUE(allInputs.contains("labels.values"));
    EXPECT_TRUE(allInputs.contains("labels.offsets"));

    const std::set<std::string> differentiableInputs = ragged.getDifferentiableInputNames();
    EXPECT_TRUE(differentiableInputs.contains("labels.values"));
    EXPECT_FALSE(differentiableInputs.contains("labels.offsets"));

    const std::set<std::string> metadataInputs = ragged.getMetadataInputNames();
    EXPECT_FALSE(metadataInputs.contains("labels.values"));
    EXPECT_TRUE(metadataInputs.contains("labels.offsets"));
}

TEST(RaggedExpression, UnaryValuewiseOpPreservesOffsetsAndRuntimeExtent) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor());

    const RaggedExpression result = ragged.abs();

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getRuntimeExtent().maxActiveValues, ragged.getRuntimeExtent().maxActiveValues);
    EXPECT_EQ(result.getRuntimeExtent().elementsPerValue, ragged.getRuntimeExtent().elementsPerValue);
    EXPECT_EQ(result.getDescriptor(), ragged.getDescriptor());

    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::ABS);
}

TEST(RaggedExpression, CastChangesValuesDTypeButPreservesOffsetsMetadataAndExtent) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {3}, 4, 12, DataType::UINT64));

    const RaggedExpression result = ragged.cast(DataType::FP16);

    EXPECT_EQ(result.getValuesDataType(), DataType::FP16);
    EXPECT_EQ(result.getOffsetsDataType(), DataType::UINT64);
    EXPECT_EQ(result.getValuesDimensions(), ragged.getValuesDimensions());
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::CAST);
}

TEST(RaggedExpression, BinaryOpWithSameOffsetsSucceeds) {
    const RaggedTensorDescriptor descriptor = makeDescriptor();
    const Expression offsets = Expression::input("shared.offsets");
    const RaggedExpression lhs(Expression::input("lhs.values"), offsets, descriptor);
    const RaggedExpression rhs(Expression::input("rhs.values"), offsets, descriptor);

    const RaggedExpression result = lhs + rhs;

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(offsets));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(lhs.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getDescriptor(), lhs.getDescriptor());
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::ADD);
}

TEST(RaggedExpression, BinaryOpWithDifferentOffsetsRejects) {
    const RaggedTensorDescriptor descriptor = makeDescriptor();
    const RaggedExpression lhs(Expression::input("lhs.values"), Expression::input("lhs.offsets"), descriptor);
    const RaggedExpression rhs(Expression::input("rhs.values"), Expression::input("rhs.offsets"), descriptor);

    EXPECT_THROW((void)(lhs + rhs), std::invalid_argument);
}

TEST(RaggedExpression, BinaryOpWithDifferentValuesDescriptorRejects) {
    const Expression offsets = Expression::input("shared.offsets");
    const RaggedExpression lhs(Expression::input("lhs.values"), offsets, makeDescriptor(DataType::FP32, {4}, 3, 9));
    const RaggedExpression rhs(Expression::input("rhs.values"), offsets, makeDescriptor(DataType::FP32, {5}, 3, 9));

    EXPECT_THROW((void)(lhs + rhs), std::invalid_argument);
}

TEST(RaggedExpression, ComparisonOpProducesBooleanValuesAndPreservesOffsets) {
    const RaggedTensorDescriptor descriptor = makeDescriptor();
    const Expression offsets = Expression::input("shared.offsets");
    const RaggedExpression lhs(Expression::input("lhs.values"), offsets, descriptor);
    const RaggedExpression rhs(Expression::input("rhs.values"), offsets, descriptor);

    const RaggedExpression result = lhs.lessThan(rhs);

    EXPECT_EQ(result.getValuesDataType(), DataType::BOOLEAN);
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(offsets));
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::LESS);
}

TEST(RaggedExpression, NonScalarConvenienceSegmentOpsUseGeneralizedSegmentSemantics) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {4}, 3, 9));

    EXPECT_NO_THROW((void)ragged.softmax());
    EXPECT_NO_THROW((void)ragged.reduce_sum());
    EXPECT_EQ(outputNode(ragged.reduce_sum()).ragged_runtime_elements_per_value, 4ULL);
}


TEST(RaggedExpression, SegmentReductionsBuildDensePerRowOutputsForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 4, 12));

    const Expression sum = ragged.segment_sum();
    const Expression min = ragged.segment_min();
    const Expression max = ragged.segment_max();
    const Expression mean = ragged.segment_mean();

    EXPECT_EQ(outputNode(sum).op, ExprOp::SEGMENTED_REDUCE_SUM);
    EXPECT_EQ(outputNode(min).op, ExprOp::SEGMENTED_REDUCE_MIN);
    EXPECT_EQ(outputNode(max).op, ExprOp::SEGMENTED_REDUCE_MAX);
    EXPECT_EQ(outputNode(mean).op, ExprOp::SEGMENTED_REDUCE_MEAN);
    EXPECT_EQ(outputNode(sum).ragged_runtime_batch_size, 4ULL);
    EXPECT_EQ(outputNode(sum).ragged_runtime_max_active_values, 12ULL);
}

TEST(RaggedExpression, SegmentReductionsCarryVectorElementsPerValueMetadata) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 3}, 3, 9));

    for (const Expression& reduction :
         {ragged.segment_sum(), ragged.segment_min(), ragged.segment_max(), ragged.segment_mean()}) {
        const ExprNode node = outputNode(reduction);
        EXPECT_EQ(node.ragged_runtime_batch_size, 3ULL);
        EXPECT_EQ(node.ragged_runtime_max_active_values, 9ULL);
        EXPECT_EQ(node.ragged_runtime_elements_per_value, 6ULL);
    }
    EXPECT_NO_THROW((void)ragged.segment_softmax());
    EXPECT_NO_THROW((void)ragged.segment_log_softmax());
}

TEST(RaggedExpression, SegmentMeanBuildsOneDirectSegmentedReductionForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const Expression mean = ragged.segment_mean();
    const PhysicalExpression physical = mean.expression();

    ASSERT_EQ(physical.nodes.size(), 3U);
    const ExprNode& mean_node = physical.nodes.at(physical.output_node);
    EXPECT_EQ(mean_node.op, ExprOp::SEGMENTED_REDUCE_MEAN);
    EXPECT_EQ(physical.nodes.at(mean_node.lhs).op, ExprOp::INPUT);
    EXPECT_EQ(physical.nodes.at(mean_node.rhs).op, ExprOp::INPUT);
}

TEST(RaggedExpression, SegmentMeanAcceptsFp8BecauseCentralCubMeanAccumulatesInFp32) {
    const RaggedExpression e4m3 = RaggedExpression::input("e4m3", makeDescriptor(DataType::FP8_E4M3, {}, 3, 9));
    const RaggedExpression e5m2 = RaggedExpression::input("e5m2", makeDescriptor(DataType::FP8_E5M2, {}, 3, 9));

    EXPECT_EQ(outputNode(e4m3.segment_mean()).op, ExprOp::SEGMENTED_REDUCE_MEAN);
    EXPECT_EQ(outputNode(e5m2.segment_mean()).op, ExprOp::SEGMENTED_REDUCE_MEAN);
}

TEST(RaggedExpression, SegmentSoftmaxPreservesOffsetsAndRuntimeExtentForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const RaggedExpression result = ragged.segment_softmax();

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getDescriptor(), ragged.getDescriptor());
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::DIV);
}

TEST(RaggedExpression, SegmentLogSoftmaxPreservesOffsetsAndRuntimeExtentForScalarValues) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const RaggedExpression result = ragged.segment_log_softmax();

    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getDescriptor(), ragged.getDescriptor());
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::SUB);
}

TEST(RaggedExpression, SegmentSoftmaxAndLogSoftmaxAutodiffBuildThroughExistingSegmentedScanBackward) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const PhysicalOutputs softmax_outputs =
        Expression::outputs({{"y", ragged.segment_softmax().getValues()}}).physicalOutputs();
    size_t softmax_segmented_scans = 0;
    for (const ExprNode& node : softmax_outputs.expr->nodes) {
        if (node.op == ExprOp::SEGMENTED_SCAN) {
            ++softmax_segmented_scans;
            EXPECT_EQ(node.ragged_runtime_batch_size, 3ULL);
            EXPECT_EQ(node.ragged_runtime_max_active_values, 9ULL);
            EXPECT_EQ(node.ragged_runtime_elements_per_value, 1ULL);
        }
    }
    EXPECT_GT(softmax_segmented_scans, 0U);

    PhysicalOutputs softmax_backward = buildBackwardOutputs(softmax_outputs, {"x.values"});
    resolveRaggedBackwardTestDTypes(softmax_backward, DataType::UINT32);
    EXPECT_TRUE(containsOp(softmax_backward, ExprOp::RAGGED_VALUEWISE_EXTENT));
    EXPECT_TRUE(containsOp(softmax_backward, ExprOp::SEGMENTED_SCAN));
    EXPECT_TRUE(containsOp(softmax_backward, ExprOp::SEGMENTED_SCAN_MAX_BACKWARD));
    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(softmax_backward));
    for (const ExprNode& node : softmax_backward.expr->nodes) {
        if (node.op == ExprOp::SEGMENTED_SCAN || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
            EXPECT_EQ(node.ragged_runtime_batch_size, 3ULL);
            EXPECT_EQ(node.ragged_runtime_max_active_values, 9ULL);
            EXPECT_EQ(node.ragged_runtime_elements_per_value, 1ULL);
        }
    }
    EXPECT_THROW((void)buildBackwardOutputs(softmax_outputs, {"x.offsets"}), std::runtime_error);

    const PhysicalOutputs log_softmax_outputs =
        Expression::outputs({{"y", ragged.segment_log_softmax().getValues()}}).physicalOutputs();
    size_t log_softmax_segmented_scans = 0;
    for (const ExprNode& node : log_softmax_outputs.expr->nodes) {
        if (node.op == ExprOp::SEGMENTED_SCAN) {
            ++log_softmax_segmented_scans;
            EXPECT_EQ(node.ragged_runtime_batch_size, 3ULL);
            EXPECT_EQ(node.ragged_runtime_max_active_values, 9ULL);
            EXPECT_EQ(node.ragged_runtime_elements_per_value, 1ULL);
        }
    }
    EXPECT_GT(log_softmax_segmented_scans, 0U);

    PhysicalOutputs log_softmax_backward = buildBackwardOutputs(log_softmax_outputs, {"x.values"});
    resolveRaggedBackwardTestDTypes(log_softmax_backward, DataType::UINT32);
    EXPECT_TRUE(containsOp(log_softmax_backward, ExprOp::RAGGED_VALUEWISE_EXTENT));
    EXPECT_TRUE(containsOp(log_softmax_backward, ExprOp::SEGMENTED_SCAN));
    EXPECT_TRUE(containsOp(log_softmax_backward, ExprOp::SEGMENTED_SCAN_MAX_BACKWARD));
    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(log_softmax_backward));
    for (const ExprNode& node : log_softmax_backward.expr->nodes) {
        if (node.op == ExprOp::SEGMENTED_SCAN || node.op == ExprOp::SEGMENTED_SCAN_MAX_BACKWARD) {
            EXPECT_EQ(node.ragged_runtime_batch_size, 3ULL);
            EXPECT_EQ(node.ragged_runtime_max_active_values, 9ULL);
            EXPECT_EQ(node.ragged_runtime_elements_per_value, 1ULL);
        }
    }
    EXPECT_THROW((void)buildBackwardOutputs(log_softmax_outputs, {"x.offsets"}), std::runtime_error);
}

TEST(RaggedExpression, SegmentedScanCanCarryReverseFlagForRowLocalBroadcasts) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 9));

    const Expression reverseScan =
        Expression::input("x.values", DataType::FP32).segmentedScan(ragged.getOffsets(), ScanOp::Sum, true, true);

    const ExprNode scanNode = outputNode(reverseScan);
    EXPECT_EQ(scanNode.op, ExprOp::SEGMENTED_SCAN);
    EXPECT_TRUE(scanNode.scan_reverse);
}


TEST(RaggedExpression, ValuewiseAutodiffPreservesRuntimeExtentAndRejectsOffsets) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2}, 3, 6));
    const PhysicalOutputs outputs = Expression::outputs({{"y", ragged.relu().getValues()}}).physicalOutputs();

    const PhysicalOutputs backward = buildBackwardOutputs(outputs, {"x.values"});
    EXPECT_TRUE(containsOp(backward, ExprOp::RAGGED_VALUEWISE_EXTENT));
    EXPECT_THROW((void)buildBackwardOutputs(outputs, {"x.offsets"}), std::runtime_error);
}

TEST(RaggedExpression, SegmentSumAndMeanAutodiffLowerThroughSegmentedBroadcast) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 7));

    const PhysicalOutputs sum_outputs = Expression::outputs({{"y", ragged.segment_sum()}}).physicalOutputs();
    const PhysicalOutputs sum_backward = buildBackwardOutputs(sum_outputs, {"x.values"});
    ASSERT_TRUE(containsOp(sum_backward, ExprOp::SEGMENTED_BROADCAST));
    EXPECT_TRUE(containsOp(sum_backward, ExprOp::RAGGED_VALUEWISE_EXTENT));

    bool found_sum_broadcast = false;
    for (const ExprNode& node : sum_backward.expr->nodes) {
        if (node.op == ExprOp::SEGMENTED_BROADCAST) {
            found_sum_broadcast = true;
            EXPECT_FALSE(node.segmented_broadcast_normalize_by_length);
            EXPECT_EQ(node.ragged_runtime_max_active_values, 7ULL);
        }
    }
    EXPECT_TRUE(found_sum_broadcast);

    const PhysicalOutputs mean_outputs = Expression::outputs({{"y", ragged.segment_mean()}}).physicalOutputs();
    const PhysicalOutputs mean_backward = buildBackwardOutputs(mean_outputs, {"x.values"});
    bool found_mean_broadcast = false;
    for (const ExprNode& node : mean_backward.expr->nodes) {
        if (node.op == ExprOp::SEGMENTED_BROADCAST) {
            found_mean_broadcast = true;
            EXPECT_TRUE(node.segmented_broadcast_normalize_by_length);
            EXPECT_EQ(node.ragged_runtime_max_active_values, 7ULL);
        }
    }
    EXPECT_TRUE(found_mean_broadcast);

    EXPECT_THROW((void)buildBackwardOutputs(sum_outputs, {"x.offsets"}), std::runtime_error);
    EXPECT_THROW((void)buildBackwardOutputs(mean_outputs, {"x.offsets"}), std::runtime_error);
}

TEST(RaggedExpression, VectorSegmentSumAndMeanAutodiffCarryGeneralizedBroadcastMetadata) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 3}, 3, 7));

    for (const Expression& forward : {ragged.segment_sum(), ragged.segment_mean()}) {
        const PhysicalOutputs backward =
            buildBackwardOutputs(Expression::outputs({{"y", forward}}).physicalOutputs(), {"x.values"});
        bool found_broadcast = false;
        bool found_extent = false;
        for (const ExprNode& node : backward.expr->nodes) {
            if (node.op == ExprOp::SEGMENTED_BROADCAST) {
                found_broadcast = true;
                EXPECT_EQ(node.ragged_runtime_max_active_values, 7ULL);
                EXPECT_EQ(node.ragged_runtime_elements_per_value, 6ULL);
            }
            if (node.op == ExprOp::RAGGED_VALUEWISE_EXTENT) {
                found_extent = true;
                EXPECT_EQ(node.ragged_runtime_elements_per_value, 6ULL);
            }
        }
        EXPECT_TRUE(found_broadcast);
        EXPECT_TRUE(found_extent);
    }
}

TEST(RaggedExpression, SegmentMinMaxAutodiffLowersThroughSegmentedArgReductionBackward) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 4, 10));

    PhysicalOutputs min_backward =
        buildBackwardOutputs(Expression::outputs({{"y", ragged.segment_min()}}).physicalOutputs(), {"x.values"});
    resolveRaggedBackwardTestDTypes(min_backward, DataType::UINT32);
    EXPECT_TRUE(containsOp(min_backward, ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD));
    EXPECT_TRUE(containsOp(min_backward, ExprOp::RAGGED_VALUEWISE_EXTENT));
    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(min_backward));

    PhysicalOutputs max_backward =
        buildBackwardOutputs(Expression::outputs({{"y", ragged.segment_max()}}).physicalOutputs(), {"x.values"});
    resolveRaggedBackwardTestDTypes(max_backward, DataType::UINT64);
    EXPECT_TRUE(containsOp(max_backward, ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD));
    EXPECT_TRUE(containsOp(max_backward, ExprOp::RAGGED_VALUEWISE_EXTENT));
    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(max_backward));

    EXPECT_THROW(
        (void)buildBackwardOutputs(Expression::outputs({{"y", ragged.segment_min()}}).physicalOutputs(), {"x.offsets"}),
        std::runtime_error);
    EXPECT_THROW(
        (void)buildBackwardOutputs(Expression::outputs({{"y", ragged.segment_max()}}).physicalOutputs(), {"x.offsets"}),
        std::runtime_error);
}

TEST(RaggedExpression, VectorSegmentMinMaxAutodiffLowersThroughGeneralizedWinnerRouting) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 2}, 4, 10));

    PhysicalOutputs min_backward =
        buildBackwardOutputs(Expression::outputs({{"y", ragged.segment_min()}}).physicalOutputs(), {"x.values"});
    resolveRaggedBackwardTestDTypes(min_backward, DataType::UINT32);
    EXPECT_TRUE(containsOp(min_backward, ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD));
    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(min_backward));
    bool found_min = false;
    for (const ExprNode& node : min_backward.expr->nodes) {
        if (node.op == ExprOp::SEGMENTED_REDUCE_MIN_BACKWARD) {
            found_min = true;
            EXPECT_EQ(node.ragged_runtime_elements_per_value, 4ULL);
        }
    }
    EXPECT_TRUE(found_min);

    PhysicalOutputs max_backward =
        buildBackwardOutputs(Expression::outputs({{"y", ragged.segment_max()}}).physicalOutputs(), {"x.values"});
    resolveRaggedBackwardTestDTypes(max_backward, DataType::UINT64);
    EXPECT_TRUE(containsOp(max_backward, ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD));
    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(max_backward));
    bool found_max = false;
    for (const ExprNode& node : max_backward.expr->nodes) {
        if (node.op == ExprOp::SEGMENTED_REDUCE_MAX_BACKWARD) {
            found_max = true;
            EXPECT_EQ(node.ragged_runtime_elements_per_value, 4ULL);
        }
    }
    EXPECT_TRUE(found_max);
}

TEST(RaggedExpression, SegmentMinMaxAutodiffMatchesFiniteDifferencesAwayFromTies) {
    REQUIRE_CUDA_DEVICE();

    runSegmentMinMaxFiniteDifferenceCase<uint32_t>(true, 777.0F);
    runSegmentMinMaxFiniteDifferenceCase<uint64_t>(true, 778.0F);
    runSegmentMinMaxFiniteDifferenceCase<uint32_t>(false, -777.0F);
    runSegmentMinMaxFiniteDifferenceCase<uint64_t>(false, -778.0F);
}

TEST(RaggedExpression, SegmentMinMaxAutodiffRoutesFirstWinnerForTiesNansAndEmptyRows) {
    REQUIRE_CUDA_DEVICE();

    runSegmentMinMaxAutodiffCase<uint32_t>(true, 777.0F);
    runSegmentMinMaxAutodiffCase<uint64_t>(true, 778.0F);
    runSegmentMinMaxAutodiffCase<uint32_t>(false, -777.0F);
    runSegmentMinMaxAutodiffCase<uint64_t>(false, -778.0F);
}

TEST(RaggedExpression, ValuewiseAutodiffExecutesOnlyOverActivePackedValues) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2}, 3, 6));
    Tensor values = makeGpuTensor<float>({6, 2},
                                         {-1.0F, 2.0F,
                                          3.0F, -4.0F,
                                          5.0F, 6.0F,
                                          -7.0F, 8.0F,
                                          9.0F, -10.0F,
                                          11.0F, 12.0F},
                                         stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 1U, 1U, 3U}, stream);
    Tensor upstream = makeGpuTensor<float>({6, 2},
                                           {1.0F, 2.0F,
                                            3.0F, 4.0F,
                                            5.0F, 6.0F,
                                            7.0F, 8.0F,
                                            9.0F, 10.0F,
                                            11.0F, 12.0F},
                                           stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {6, 2}));
    gradient.fill(777.0, stream);

    const Tensor result = runBackwardOutput(ragged.relu().getValues(),
                                            {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                            "x.values",
                                            "dy",
                                            stream,
                                            gradient);

    expectNear(copyToCpuValues(result, stream),
               {0.0F, 2.0F,
                3.0F, 0.0F,
                5.0F, 6.0F,
                777.0F, 777.0F,
                777.0F, 777.0F,
                777.0F, 777.0F});
}

TEST(RaggedExpression, SegmentSoftmaxAutodiffMatchesFiniteDifferencesForBothOffsetDTypes) {
    REQUIRE_CUDA_DEVICE();

    runSegmentSoftmaxAutodiffCase<uint32_t>(false, 777.0F);
    runSegmentSoftmaxAutodiffCase<uint64_t>(false, 778.0F);
}

TEST(RaggedExpression, SegmentLogSoftmaxAutodiffMatchesFiniteDifferencesForBothOffsetDTypes) {
    REQUIRE_CUDA_DEVICE();

    runSegmentSoftmaxAutodiffCase<uint32_t>(true, -777.0F);
    runSegmentSoftmaxAutodiffCase<uint64_t>(true, -778.0F);
}

TEST(RaggedExpression, SegmentSumAndMeanAutodiffExecuteWithEmptyRowsAndUnusedCapacity) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged =
        RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 7, DataType::UINT64));
    Tensor values = makeGpuTensor<float>({7}, {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 99.0F, 100.0F}, stream);
    Tensor offsets = makeGpuTensor<uint64_t>({4}, {0ULL, 2ULL, 2ULL, 5ULL}, stream);
    Tensor upstream = makeGpuTensor<float>({3}, {10.0F, 20.0F, 30.0F}, stream);

    Tensor sum_gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {7}));
    sum_gradient.fill(777.0, stream);
    const Tensor sum_result = runBackwardOutput(ragged.segment_sum(),
                                                {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                "x.values",
                                                "dy",
                                                stream,
                                                sum_gradient);
    expectNear(copyToCpuValues(sum_result, stream), {10.0F, 10.0F, 30.0F, 30.0F, 30.0F, 777.0F, 777.0F});

    Tensor mean_gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {7}));
    mean_gradient.fill(888.0, stream);
    const Tensor mean_result = runBackwardOutput(ragged.segment_mean(),
                                                 {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                 "x.values",
                                                 "dy",
                                                 stream,
                                                 mean_gradient);
    expectNear(copyToCpuValues(mean_result, stream), {5.0F, 5.0F, 10.0F, 10.0F, 10.0F, 888.0F, 888.0F});
}

TEST(RaggedExpression, VectorSegmentSumAndMeanAutodiffExecuteAcrossTrailingDimensions) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 7;
    const RaggedExpression ragged =
        RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 2}, batch_size, max_total_values, DataType::UINT64));
    const std::vector<float> values_host{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
        13.0F, 14.0F, 15.0F, 16.0F,
        17.0F, 18.0F, 19.0F, 20.0F,
        99.0F, 100.0F, 101.0F, 102.0F,
        103.0F, 104.0F, 105.0F, 106.0F};
    Tensor values = makeGpuTensor<float>({max_total_values, 2, 2}, values_host, stream);
    Tensor offsets = makeGpuTensor<uint64_t>({batch_size + 1}, {0ULL, 2ULL, 2ULL, 5ULL}, stream);
    const std::vector<float> upstream_host{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F};
    Tensor upstream = makeGpuTensor<float>({batch_size, 2, 2}, upstream_host, stream);

    Tensor sum_gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, 2, 2}));
    sum_gradient.fill(777.0, stream);
    const Tensor sum_result = runBackwardOutput(ragged.segment_sum(),
                                                {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                "x.values",
                                                "dy",
                                                stream,
                                                sum_gradient);
    const std::vector<float> sum_expected{
        1.0F, 2.0F, 3.0F, 4.0F,
        1.0F, 2.0F, 3.0F, 4.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
        777.0F, 777.0F, 777.0F, 777.0F,
        777.0F, 777.0F, 777.0F, 777.0F};
    expectNear(copyToCpuValues(sum_result, stream), sum_expected);

    Tensor mean_gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, 2, 2}));
    mean_gradient.fill(888.0, stream);
    const Tensor mean_result = runBackwardOutput(ragged.segment_mean(),
                                                 {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                 "x.values",
                                                 "dy",
                                                 stream,
                                                 mean_gradient);
    const std::vector<float> mean_expected{
        0.5F, 1.0F, 1.5F, 2.0F,
        0.5F, 1.0F, 1.5F, 2.0F,
        3.0F, 10.0F / 3.0F, 11.0F / 3.0F, 4.0F,
        3.0F, 10.0F / 3.0F, 11.0F / 3.0F, 4.0F,
        3.0F, 10.0F / 3.0F, 11.0F / 3.0F, 4.0F,
        888.0F, 888.0F, 888.0F, 888.0F,
        888.0F, 888.0F, 888.0F, 888.0F};
    expectNear(copyToCpuValues(mean_result, stream), mean_expected, 2.0e-5F);
}

TEST(RaggedExpression, VectorSegmentAutodiffExecutesForFp16Bf16AndFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 5;
    const std::vector<float> values_host{
        1.0F, 5.0F,
        2.0F, 4.0F,
        -1.0F, 3.0F,
        0.0F, 2.0F,
        99.0F, 100.0F};
    const std::vector<float> upstream_host{3.0F, 4.0F, 5.0F, 6.0F};
    Tensor offsets = makeGpuTensor<uint32_t>({batch_size + 1}, {0U, 2U, 4U}, stream);

    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        SCOPED_TRACE(TensorDescriptor::getElementTypeName(dtype));
        const RaggedExpression ragged =
            RaggedExpression::input("x", makeDescriptor(dtype, {2}, batch_size, max_total_values, DataType::UINT32));
        Tensor values = makeGpuTensorFromFloats({max_total_values, 2}, values_host, dtype, stream);
        Tensor upstream = makeGpuTensorFromFloats({batch_size, 2}, upstream_host, dtype, stream);

        Tensor sum_gradient(gpuPlacement, TensorDescriptor(dtype, {max_total_values, 2}));
        sum_gradient.fill(77.0, stream);
        const Tensor sum_result = runBackwardOutput(ragged.segment_sum(),
                                                    {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                    "x.values",
                                                    "dy",
                                                    stream,
                                                    sum_gradient);
        expectNear(copyToCpuFloatValues(sum_result, stream),
                   {3.0F, 4.0F, 3.0F, 4.0F, 5.0F, 6.0F, 5.0F, 6.0F, 77.0F, 77.0F},
                   2.0e-2F);

        Tensor min_gradient(gpuPlacement, TensorDescriptor(dtype, {max_total_values, 2}));
        min_gradient.fill(88.0, stream);
        const Tensor min_result = runBackwardOutput(ragged.segment_min(),
                                                    {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                    "x.values",
                                                    "dy",
                                                    stream,
                                                    min_gradient);
        expectNear(copyToCpuFloatValues(min_result, stream),
                   {3.0F, 0.0F, 0.0F, 4.0F, 5.0F, 0.0F, 0.0F, 6.0F, 88.0F, 88.0F},
                   2.0e-2F);
    }
}

TEST(RaggedExpression, VectorSegmentMinMaxAutodiffMatchesFiniteDifferencesAndPreservesUnusedCapacity) {
    REQUIRE_CUDA_DEVICE();

    runVectorSegmentMinMaxFiniteDifferenceCase<uint32_t>(true, 777.0F);
    runVectorSegmentMinMaxFiniteDifferenceCase<uint64_t>(true, 778.0F);
    runVectorSegmentMinMaxFiniteDifferenceCase<uint32_t>(false, -777.0F);
    runVectorSegmentMinMaxFiniteDifferenceCase<uint64_t>(false, -778.0F);
}

TEST(RaggedExpression, VectorSegmentMinMaxAutodiffRoutesFirstWinnerForTiesNansAndEmptyRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batch_size = 4;
    constexpr uint64_t max_total_values = 10;
    constexpr uint64_t elements_per_value = 2;
    const RaggedExpression ragged = RaggedExpression::input(
        "x", makeDescriptor(DataType::FP32, {elements_per_value}, batch_size, max_total_values, DataType::UINT64));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> values_host{
        2.0F, 4.0F,
        -1.0F, 7.0F,
        -1.0F, 7.0F,
        nan, 5.0F,
        3.0F, -2.0F,
        nan, 8.0F,
        4.0F, 1.0F,
        4.0F, 1.0F,
        99.0F, 100.0F,
        101.0F, 102.0F};
    Tensor values = makeGpuTensor<float>({max_total_values, elements_per_value}, values_host, stream);
    Tensor offsets = makeGpuTensor<uint64_t>({batch_size + 1}, {0ULL, 3ULL, 3ULL, 6ULL, 8ULL}, stream);
    Tensor upstream = makeGpuTensor<float>({batch_size, elements_per_value},
                                           {10.0F, 11.0F, 20.0F, 21.0F, 30.0F, 31.0F, 40.0F, 41.0F},
                                           stream);

    for (bool minimum : {true, false}) {
        const float sentinel = minimum ? 777.0F : -777.0F;
        Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, elements_per_value}));
        gradient.fill(sentinel, stream);
        const Expression forward = minimum ? ragged.segment_min() : ragged.segment_max();
        const Tensor result = runBackwardOutput(forward,
                                                {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                "x.values",
                                                "dy",
                                                stream,
                                                gradient);
        std::vector<float> expected(max_total_values * elements_per_value, 0.0F);
        if (minimum) {
            expected[1 * elements_per_value + 0] = 10.0F;  // first -1 in row 0
            expected[0 * elements_per_value + 1] = 11.0F;  // 4 is row-0 minimum for component 1
        } else {
            expected[0 * elements_per_value + 0] = 10.0F;  // 2 is row-0 maximum for component 0
            expected[1 * elements_per_value + 1] = 11.0F;  // first 7 in row-0 tie
        }
        // Row 1 is empty. For row 2, NaNs propagate and the first NaN wins component 0.
        expected[3 * elements_per_value + 0] = 30.0F;
        if (minimum) {
            expected[4 * elements_per_value + 1] = 31.0F;
        } else {
            expected[5 * elements_per_value + 1] = 31.0F;
        }
        // Row 3 ties in both components; the first packed value wins.
        expected[6 * elements_per_value + 0] = 40.0F;
        expected[6 * elements_per_value + 1] = 41.0F;
        for (uint64_t value_index = 8; value_index < max_total_values; ++value_index) {
            for (uint64_t component = 0; component < elements_per_value; ++component) {
                expected[value_index * elements_per_value + component] = sentinel;
            }
        }
        expectNear(copyToCpuValues(result, stream), expected);
    }
}

TEST(RaggedExpression, VectorSegmentSoftmaxAndLogSoftmaxAutodiffMatchFiniteDifferences) {
    REQUIRE_CUDA_DEVICE();

    runVectorSegmentSoftmaxAutodiffCase<uint32_t>(false, 777.0F);
    runVectorSegmentSoftmaxAutodiffCase<uint64_t>(false, 778.0F);
    runVectorSegmentSoftmaxAutodiffCase<uint32_t>(true, -777.0F);
    runVectorSegmentSoftmaxAutodiffCase<uint64_t>(true, -778.0F);
}

TEST(RaggedExpression, ValuewiseExecutionReadsActiveExtentOnDeviceAndReusesOneStampedPlan) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedTensorDescriptor descriptor = makeDescriptor(DataType::FP32, {2}, 3, 6);
    const RaggedExpression ragged = RaggedExpression::input("x", descriptor);
    const Expression output_expression = ragged.relu().getValues();

    Tensor values = makeGpuTensor<float>({6, 2},
                                         {-1.0F, 2.0F,
                                          -3.0F, 4.0F,
                                          5.0F, -6.0F,
                                          7.0F, -8.0F,
                                          9.0F, -10.0F,
                                          11.0F, -12.0F},
                                         stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 2U, 2U, 3U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {6, 2}));
    output.fill(777.0, stream);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", output_expression}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"x.values", values}, {"x.offsets", offsets}}, stream, {}, {{"y", output}});
    plan.run();
    expectNear(copyToCpuValues(output, stream),
               {0.0F, 2.0F, 0.0F, 4.0F, 5.0F, 0.0F,
                777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F});

    // Change only offsets[B] and replay the already-stamped plan. This exercises
    // device-side logical extent without shape recompilation or host readback.
    overwriteGpuTensor<uint32_t>(offsets, {0U, 1U, 3U, 5U}, stream);
    output.fill(888.0, stream);
    plan.run();
    expectNear(copyToCpuValues(output, stream),
               {0.0F, 2.0F, 0.0F, 4.0F, 5.0F, 0.0F, 7.0F, 0.0F, 9.0F, 0.0F,
                888.0F, 888.0F});
}

TEST(RaggedExpression, SegmentSumMinMaxAndMeanExecuteForEmptyAndSkewedRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 4, 9));
    Tensor values = makeGpuTensor<float>({9}, {1.0F, -2.0F, 4.0F, 5.0F, 7.0F, -1.0F, 8.0F, 99.0F, 100.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({5}, {0U, 3U, 3U, 5U, 7U}, stream);
    const std::unordered_map<std::string, Tensor> inputs{{"x.values", values}, {"x.offsets", offsets}};

    const Tensor sum = runExpressionOutput(ragged.segment_sum(), inputs, "sum", stream);
    const Tensor min = runExpressionOutput(ragged.segment_min(), inputs, "min", stream);
    const Tensor max = runExpressionOutput(ragged.segment_max(), inputs, "max", stream);
    const Tensor mean = runExpressionOutput(ragged.segment_mean(), inputs, "mean", stream);

    expectNear(copyToCpuValues(sum, stream), {3.0F, 0.0F, 12.0F, 7.0F});

    const std::vector<float> min_values = copyToCpuValues(min, stream);
    ASSERT_EQ(min_values.size(), 4U);
    EXPECT_FLOAT_EQ(min_values[0], -2.0F);
    EXPECT_FLOAT_EQ(min_values[2], 5.0F);
    EXPECT_FLOAT_EQ(min_values[3], -1.0F);

    const std::vector<float> max_values = copyToCpuValues(max, stream);
    ASSERT_EQ(max_values.size(), 4U);
    EXPECT_FLOAT_EQ(max_values[0], 4.0F);
    EXPECT_FLOAT_EQ(max_values[2], 7.0F);
    EXPECT_FLOAT_EQ(max_values[3], 8.0F);

    expectNear(copyToCpuValues(mean, stream), {1.0F, 0.0F, 6.0F, 3.5F});
}

TEST(RaggedExpression, SegmentSoftmaxExecutesPerRowAndLeavesUnusedCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 8));
    Tensor values = makeGpuTensor<float>({8}, {1.0F, 2.0F, 0.0F, -1.0F, 1.0F, 50.0F, 60.0F, 70.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 2U, 2U, 5U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {8}));
    output.fill(321.0, stream);

    const Tensor actual = runExpressionOutput(ragged.segment_softmax().getValues(),
                                              {{"x.values", values}, {"x.offsets", offsets}},
                                              "softmax",
                                              stream,
                                              output);

    const float row0_denom = std::exp(1.0F) + std::exp(2.0F);
    const float row2_denom = std::exp(0.0F) + std::exp(-1.0F) + std::exp(1.0F);
    expectNear(copyToCpuValues(actual, stream),
               {std::exp(1.0F) / row0_denom,
                std::exp(2.0F) / row0_denom,
                std::exp(0.0F) / row2_denom,
                std::exp(-1.0F) / row2_denom,
                std::exp(1.0F) / row2_denom,
                321.0F,
                321.0F,
                321.0F},
               2.0e-5F);
}

TEST(RaggedExpression, SegmentLogSoftmaxExecutesPerRowAndLeavesUnusedCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {}, 3, 8));
    Tensor values = makeGpuTensor<float>({8}, {1.0F, 2.0F, 0.0F, -1.0F, 1.0F, 50.0F, 60.0F, 70.0F}, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 2U, 2U, 5U}, stream);
    Tensor output(gpuPlacement, TensorDescriptor(DataType::FP32, {8}));
    output.fill(-456.0, stream);

    const Tensor actual = runExpressionOutput(ragged.segment_log_softmax().getValues(),
                                              {{"x.values", values}, {"x.offsets", offsets}},
                                              "log_softmax",
                                              stream,
                                              output);

    const float row0_log_denom = std::log(std::exp(1.0F) + std::exp(2.0F));
    const float row2_log_denom = std::log(std::exp(0.0F) + std::exp(-1.0F) + std::exp(1.0F));
    expectNear(copyToCpuValues(actual, stream),
               {1.0F - row0_log_denom,
                2.0F - row0_log_denom,
                0.0F - row2_log_denom,
                -1.0F - row2_log_denom,
                1.0F - row2_log_denom,
                -456.0F,
                -456.0F,
                -456.0F},
               2.0e-5F);
}

TEST(RaggedExpression, VectorSegmentReductionsExecuteForFp16Bf16AndFp32) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const std::vector<float> values{
        1.0F, 10.0F,
        3.0F, 6.0F,
        -1.0F, 4.0F,
        5.0F, -2.0F,
        7.0F, 8.0F,
        91.0F, 92.0F,
        93.0F, 94.0F,
    };
    Tensor offsets = makeGpuTensor<uint32_t>({5}, {0U, 4U, 4U, 5U, 5U}, stream);

    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        SCOPED_TRACE(TensorDescriptor::getElementTypeName(dtype));
        const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(dtype, {2}, 4, 7));
        Tensor gpu_values = makeGpuTensorFromFloats({7, 2}, values, dtype, stream);
        const std::unordered_map<std::string, Tensor> inputs{{"x.values", gpu_values}, {"x.offsets", offsets}};

        const Tensor sum = runExpressionOutput(ragged.segment_sum(), inputs, "sum", stream);
        const Tensor min = runExpressionOutput(ragged.segment_min(), inputs, "min", stream);
        const Tensor max = runExpressionOutput(ragged.segment_max(), inputs, "max", stream);
        const Tensor mean = runExpressionOutput(ragged.segment_mean(), inputs, "mean", stream);

        EXPECT_EQ(sum.getDimensions(), (std::vector<uint64_t>{4, 2}));
        expectNear(copyToCpuFloatValues(sum, stream), {8.0F, 18.0F, 0.0F, 0.0F, 7.0F, 8.0F, 0.0F, 0.0F}, 2.0e-2F);
        expectNear(copyToCpuFloatValues(mean, stream), {2.0F, 4.5F, 0.0F, 0.0F, 7.0F, 8.0F, 0.0F, 0.0F}, 2.0e-2F);

        const std::vector<float> min_values = copyToCpuFloatValues(min, stream);
        ASSERT_EQ(min_values.size(), 8U);
        EXPECT_NEAR(min_values[0], -1.0F, 2.0e-2F);
        EXPECT_NEAR(min_values[1], -2.0F, 2.0e-2F);
        EXPECT_TRUE(std::isinf(min_values[2]) && min_values[2] > 0.0F);
        EXPECT_TRUE(std::isinf(min_values[3]) && min_values[3] > 0.0F);
        EXPECT_NEAR(min_values[4], 7.0F, 2.0e-2F);
        EXPECT_NEAR(min_values[5], 8.0F, 2.0e-2F);
        EXPECT_TRUE(std::isinf(min_values[6]) && min_values[6] > 0.0F);
        EXPECT_TRUE(std::isinf(min_values[7]) && min_values[7] > 0.0F);

        const std::vector<float> max_values = copyToCpuFloatValues(max, stream);
        ASSERT_EQ(max_values.size(), 8U);
        EXPECT_NEAR(max_values[0], 5.0F, 2.0e-2F);
        EXPECT_NEAR(max_values[1], 10.0F, 2.0e-2F);
        EXPECT_TRUE(std::isinf(max_values[2]) && max_values[2] < 0.0F);
        EXPECT_TRUE(std::isinf(max_values[3]) && max_values[3] < 0.0F);
        EXPECT_NEAR(max_values[4], 7.0F, 2.0e-2F);
        EXPECT_NEAR(max_values[5], 8.0F, 2.0e-2F);
        EXPECT_TRUE(std::isinf(max_values[6]) && max_values[6] < 0.0F);
        EXPECT_TRUE(std::isinf(max_values[7]) && max_values[7] < 0.0F);
    }
}

TEST(RaggedExpression, VectorSegmentSoftmaxExecutesPerComponentAndLeavesUnusedCapacityUntouched) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t max_total_values = 32;
    std::vector<float> values(max_total_values * 2, 91.0F);
    const std::vector<float> active_values{
        1.0F, 10.0F,
        3.0F, 6.0F,
        -1.0F, 4.0F,
        5.0F, -2.0F,
        7.0F, 8.0F,
    };
    std::copy(active_values.begin(), active_values.end(), values.begin());
    Tensor offsets = makeGpuTensor<uint32_t>({5}, {0U, 2U, 2U, 3U, 5U}, stream);

    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        SCOPED_TRACE(TensorDescriptor::getElementTypeName(dtype));
        const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(dtype, {2}, 4, max_total_values));
        Tensor gpu_values = makeGpuTensorFromFloats({max_total_values, 2}, values, dtype, stream);
        Tensor output(gpuPlacement, TensorDescriptor(dtype, {max_total_values, 2}));
        output.fill(77.0, stream);

        const Tensor actual = runExpressionOutput(ragged.segment_softmax().getValues(),
                                                  {{"x.values", gpu_values}, {"x.offsets", offsets}},
                                                  "softmax",
                                                  stream,
                                                  output);
        const float e2 = std::exp(2.0F);
        const float e10 = std::exp(10.0F);
        std::vector<float> expected(max_total_values * 2, 77.0F);
        const std::vector<float> active_expected{
            1.0F / (1.0F + e2), e10 / (e10 + std::exp(6.0F)),
            e2 / (1.0F + e2), std::exp(6.0F) / (e10 + std::exp(6.0F)),
            1.0F, 1.0F,
            1.0F / (1.0F + e2), 1.0F / (1.0F + e10),
            e2 / (1.0F + e2), e10 / (1.0F + e10),
        };
        std::copy(active_expected.begin(), active_expected.end(), expected.begin());
        expectNear(copyToCpuFloatValues(actual, stream), expected, dtype == DataType::FP32 ? 3.0e-5F : 2.0e-2F);
    }
}

TEST(RaggedExpression, VectorSegmentOperationsSupportMultipleTrailingDimensionsAndUint64Offsets) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    const RaggedExpression ragged =
        RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 2}, 3, 6, DataType::UINT64));
    Tensor values = makeGpuTensor<float>({6, 2, 2},
                                         {1.0F, 2.0F, 3.0F, 4.0F,
                                          5.0F, 6.0F, 7.0F, 8.0F,
                                          -1.0F, -2.0F, -3.0F, -4.0F,
                                          9.0F, 10.0F, 11.0F, 12.0F,
                                          13.0F, 14.0F, 15.0F, 16.0F,
                                          101.0F, 102.0F, 103.0F, 104.0F},
                                         stream);
    Tensor offsets = makeGpuTensor<uint64_t>({4}, {0ULL, 2ULL, 2ULL, 5ULL}, stream);
    const std::unordered_map<std::string, Tensor> inputs{{"x.values", values}, {"x.offsets", offsets}};

    const Tensor sum = runExpressionOutput(ragged.segment_sum(), inputs, "sum", stream);
    EXPECT_EQ(sum.getDimensions(), (std::vector<uint64_t>{3, 2, 2}));
    expectNear(copyToCpuValues(sum, stream),
               {6.0F, 8.0F, 10.0F, 12.0F,
                0.0F, 0.0F, 0.0F, 0.0F,
                21.0F, 22.0F, 23.0F, 24.0F});

    Tensor softmax_output(gpuPlacement, TensorDescriptor(DataType::FP32, {6, 2, 2}));
    softmax_output.fill(555.0, stream);
    const Tensor softmax = runExpressionOutput(ragged.segment_softmax().getValues(), inputs, "softmax", stream, softmax_output);
    const std::vector<float> softmax_values = copyToCpuValues(softmax, stream);
    ASSERT_EQ(softmax_values.size(), 24U);
    const float small = 1.0F / (1.0F + std::exp(4.0F));
    const float large = 1.0F - small;
    for (size_t component = 0; component < 4; ++component) {
        EXPECT_NEAR(softmax_values[component], small, 3.0e-5F);
        EXPECT_NEAR(softmax_values[4 + component], large, 3.0e-5F);
    }
    for (size_t i = 20; i < 24; ++i) {
        EXPECT_FLOAT_EQ(softmax_values[i], 555.0F);
    }

    Tensor log_softmax_output(gpuPlacement, TensorDescriptor(DataType::FP32, {6, 2, 2}));
    log_softmax_output.fill(-555.0, stream);
    const Tensor log_softmax =
        runExpressionOutput(ragged.segment_log_softmax().getValues(), inputs, "log_softmax", stream, log_softmax_output);
    const std::vector<float> log_softmax_values = copyToCpuValues(log_softmax, stream);
    ASSERT_EQ(log_softmax_values.size(), 24U);
    for (size_t component = 0; component < 4; ++component) {
        EXPECT_NEAR(log_softmax_values[component], std::log(small), 3.0e-5F);
        EXPECT_NEAR(log_softmax_values[4 + component], std::log(large), 3.0e-5F);
    }
    for (size_t i = 20; i < 24; ++i) {
        EXPECT_FLOAT_EQ(log_softmax_values[i], -555.0F);
    }
}
