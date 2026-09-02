#include "Utilities/Expression/RaggedExpression.h"

#include "Utilities/Expression/AutoDiff.h"
#include "Utilities/Common/CudnnFrontendPlan.h"

#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/CudaSourceEmitter.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequence.h"

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
    return RaggedTensorDescriptor(values_dtype,
                                  trailing_dimensions,
                                  batch_size,
                                  max_total_values,
                                  max_total_values,
                                  offsets_dtype);
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

void expectNearRelative(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        float atol,
                        float rtol) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = std::max(atol, rtol * std::fabs(expected[i]));
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
    }
}

std::vector<float> cpuRaggedCausalConv1d(const std::vector<float>& packed_values,
                                           const std::vector<uint64_t>& offsets,
                                           const std::vector<float>& filter,
                                           uint64_t max_total_values,
                                           uint64_t input_channels,
                                           uint64_t output_channels,
                                           uint64_t kernel_width,
                                           uint64_t dilation,
                                           float inactive_sentinel,
                                           uint64_t groups = 1) {
    if (packed_values.size() != max_total_values * input_channels) {
        throw std::runtime_error("cpuRaggedCausalConv1d packed-value size mismatch.");
    }
    if (groups == 0 || input_channels % groups != 0 || output_channels % groups != 0) {
        throw std::runtime_error("cpuRaggedCausalConv1d invalid group geometry.");
    }
    const uint64_t input_channels_per_group = input_channels / groups;
    const uint64_t output_channels_per_group = output_channels / groups;
    if (filter.size() != output_channels * input_channels_per_group * kernel_width) {
        throw std::runtime_error("cpuRaggedCausalConv1d filter size mismatch.");
    }
    if (offsets.empty()) {
        throw std::runtime_error("cpuRaggedCausalConv1d requires canonical row offsets.");
    }

    std::vector<float> output(max_total_values * output_channels, inactive_sentinel);
    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        if (begin > end || end > max_total_values) {
            throw std::runtime_error("cpuRaggedCausalConv1d offsets exceed packed capacity.");
        }
        const uint64_t row_length = end - begin;
        for (uint64_t timestep = 0; timestep < row_length; ++timestep) {
            for (uint64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
                float accumulator = 0.0F;
                for (uint64_t filter_position = 0; filter_position < kernel_width; ++filter_position) {
                    const uint64_t lag = (kernel_width - 1 - filter_position) * dilation;
                    if (timestep < lag) {
                        continue;
                    }
                    const uint64_t source_value = begin + timestep - lag;
                    const uint64_t group = output_channel / output_channels_per_group;
                    const uint64_t input_channel_begin = group * input_channels_per_group;
                    for (uint64_t input_channel_in_group = 0; input_channel_in_group < input_channels_per_group;
                         ++input_channel_in_group) {
                        const uint64_t input_channel = input_channel_begin + input_channel_in_group;
                        const size_t input_index = source_value * input_channels + input_channel;
                        const size_t filter_index =
                            (output_channel * input_channels_per_group + input_channel_in_group) * kernel_width +
                            filter_position;
                        accumulator += packed_values.at(input_index) * filter.at(filter_index);
                    }
                }
                output.at((begin + timestep) * output_channels + output_channel) = accumulator;
            }
        }
    }
    return output;
}

std::vector<float> cpuRaggedCausalConv1dDgrad(const std::vector<float>& packed_grad_output,
                                                const std::vector<uint64_t>& offsets,
                                                const std::vector<float>& filter,
                                                uint64_t max_total_values,
                                                uint64_t input_channels,
                                                uint64_t output_channels,
                                                uint64_t kernel_width,
                                                uint64_t dilation,
                                                float inactive_sentinel,
                                                uint64_t groups = 1) {
    if (packed_grad_output.size() != max_total_values * output_channels) {
        throw std::runtime_error("cpuRaggedCausalConv1dDgrad packed dY size mismatch.");
    }
    if (groups == 0 || input_channels % groups != 0 || output_channels % groups != 0) {
        throw std::runtime_error("cpuRaggedCausalConv1dDgrad invalid group geometry.");
    }
    const uint64_t input_channels_per_group = input_channels / groups;
    const uint64_t output_channels_per_group = output_channels / groups;
    if (filter.size() != output_channels * input_channels_per_group * kernel_width) {
        throw std::runtime_error("cpuRaggedCausalConv1dDgrad filter size mismatch.");
    }

    std::vector<float> output(max_total_values * input_channels, inactive_sentinel);
    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        if (begin > end || end > max_total_values) {
            throw std::runtime_error("cpuRaggedCausalConv1dDgrad offsets exceed packed capacity.");
        }
        for (uint64_t value = begin; value < end; ++value) {
            for (uint64_t input_channel = 0; input_channel < input_channels; ++input_channel) {
                output[value * input_channels + input_channel] = 0.0F;
            }
        }
        const uint64_t row_length = end - begin;
        for (uint64_t timestep = 0; timestep < row_length; ++timestep) {
            for (uint64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
                const float dy = packed_grad_output[(begin + timestep) * output_channels + output_channel];
                const uint64_t group = output_channel / output_channels_per_group;
                const uint64_t input_channel_begin = group * input_channels_per_group;
                for (uint64_t filter_position = 0; filter_position < kernel_width; ++filter_position) {
                    const uint64_t lag = (kernel_width - 1 - filter_position) * dilation;
                    if (timestep < lag) {
                        continue;
                    }
                    const uint64_t source_value = begin + timestep - lag;
                    for (uint64_t input_channel_in_group = 0; input_channel_in_group < input_channels_per_group;
                         ++input_channel_in_group) {
                        const uint64_t input_channel = input_channel_begin + input_channel_in_group;
                        const size_t filter_index =
                            (output_channel * input_channels_per_group + input_channel_in_group) * kernel_width +
                            filter_position;
                        output[source_value * input_channels + input_channel] += dy * filter[filter_index];
                    }
                }
            }
        }
    }
    return output;
}

std::vector<float> cpuRaggedCausalConv1dWgrad(const std::vector<float>& packed_values,
                                                const std::vector<float>& packed_grad_output,
                                                const std::vector<uint64_t>& offsets,
                                                uint64_t max_total_values,
                                                uint64_t input_channels,
                                                uint64_t output_channels,
                                                uint64_t kernel_width,
                                                uint64_t dilation,
                                                uint64_t groups = 1) {
    if (packed_values.size() != max_total_values * input_channels) {
        throw std::runtime_error("cpuRaggedCausalConv1dWgrad packed X size mismatch.");
    }
    if (packed_grad_output.size() != max_total_values * output_channels) {
        throw std::runtime_error("cpuRaggedCausalConv1dWgrad packed dY size mismatch.");
    }
    if (groups == 0 || input_channels % groups != 0 || output_channels % groups != 0) {
        throw std::runtime_error("cpuRaggedCausalConv1dWgrad invalid group geometry.");
    }
    const uint64_t input_channels_per_group = input_channels / groups;
    const uint64_t output_channels_per_group = output_channels / groups;
    std::vector<float> dw(output_channels * input_channels_per_group * kernel_width, 0.0F);
    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        if (begin > end || end > max_total_values) {
            throw std::runtime_error("cpuRaggedCausalConv1dWgrad offsets exceed packed capacity.");
        }
        const uint64_t row_length = end - begin;
        for (uint64_t timestep = 0; timestep < row_length; ++timestep) {
            for (uint64_t output_channel = 0; output_channel < output_channels; ++output_channel) {
                const float dy = packed_grad_output[(begin + timestep) * output_channels + output_channel];
                const uint64_t group = output_channel / output_channels_per_group;
                const uint64_t input_channel_begin = group * input_channels_per_group;
                for (uint64_t filter_position = 0; filter_position < kernel_width; ++filter_position) {
                    const uint64_t lag = (kernel_width - 1 - filter_position) * dilation;
                    if (timestep < lag) {
                        continue;
                    }
                    const uint64_t source_value = begin + timestep - lag;
                    for (uint64_t input_channel_in_group = 0; input_channel_in_group < input_channels_per_group;
                         ++input_channel_in_group) {
                        const uint64_t input_channel = input_channel_begin + input_channel_in_group;
                        const size_t filter_index =
                            (output_channel * input_channels_per_group + input_channel_in_group) * kernel_width +
                            filter_position;
                        dw[filter_index] += packed_values[source_value * input_channels + input_channel] * dy;
                    }
                }
            }
        }
    }
    return dw;
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

std::vector<float> denseRowByRowCausalConv1d(const std::vector<float>& packed_values,
                                             const std::vector<uint64_t>& offsets,
                                             const Tensor& gpu_filter,
                                             uint64_t max_total_values,
                                             uint64_t input_channels,
                                             uint64_t output_channels,
                                             uint64_t kernel_width,
                                             uint64_t dilation,
                                             float inactive_sentinel,
                                             Stream& stream,
                                             uint64_t groups = 1) {
    std::vector<float> packed_output(max_total_values * output_channels, inactive_sentinel);
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::causal(kernel_width, 1, static_cast<int32_t>(dilation));
    const Expression dense_input_expr = Expression::input("dense_input", DataType::FP32, DataType::FP32);
    const Expression dense_filter_expr = Expression::input("dense_filter", DataType::FP32, DataType::FP32);
    const Expression dense_output_expr =
        Expression::conv1d(dense_input_expr, dense_filter_expr, spatial, DataType::FP32, DataType::FP32, groups);

    for (size_t row = 0; row + 1 < offsets.size(); ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t end = offsets[row + 1];
        const uint64_t width = end - begin;
        if (width == 0) {
            continue;
        }
        std::vector<float> ncw(input_channels * width, 0.0F);
        for (uint64_t c = 0; c < input_channels; ++c) {
            for (uint64_t t = 0; t < width; ++t) {
                ncw[c * width + t] = packed_values[(begin + t) * input_channels + c];
            }
        }
        Tensor gpu_row = makeGpuTensor<float>({1, input_channels, width}, ncw, stream);
        Tensor gpu_row_output = runExpressionOutput(dense_output_expr,
                                                    {{"dense_input", gpu_row}, {"dense_filter", gpu_filter}},
                                                    "dense_output",
                                                    stream);
        const std::vector<float> row_output = copyToCpuValues(gpu_row_output, stream);
        if (row_output.size() != output_channels * width) {
            throw std::runtime_error("denseRowByRowCausalConv1d output size mismatch.");
        }
        for (uint64_t t = 0; t < width; ++t) {
            for (uint64_t k = 0; k < output_channels; ++k) {
                packed_output[(begin + t) * output_channels + k] = row_output[k * width + t];
            }
        }
    }
    return packed_output;
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

TEST(RaggedExpression, WithValuesCanChangeTrailingShapeAndRecomputesRuntimeWidth) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {8}, 4, 12, DataType::UINT64));
    const RaggedTensorDescriptor narrowed_descriptor = makeDescriptor(DataType::FP32, {3}, 4, 12, DataType::UINT64);

    const RaggedExpression result = ragged.withValues(Expression::input("narrow.values"), narrowed_descriptor);

    EXPECT_EQ(result.getTrailingDimensions(), std::vector<uint64_t>({3}));
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getRuntimeExtent().maxActiveValues, 12ULL);
    EXPECT_EQ(result.getRuntimeExtent().elementsPerValue, 3ULL);
    EXPECT_EQ(result.getRuntimeExtent().maxLaunchElements(), 36ULL);
    EXPECT_EQ(markedValueNodes(result.getValues()).marker.ragged_runtime_elements_per_value, 3ULL);
}

TEST(RaggedExpression, TrailingSlicePreservesPartitionAndUsesSourceStrides) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {8}, 3, 9));

    const RaggedExpression result = ragged.sliceLastDimension(2, 3);

    EXPECT_EQ(result.getValuesDimensions(), std::vector<uint64_t>({9, 3}));
    EXPECT_EQ(result.getTrailingDimensions(), std::vector<uint64_t>({3}));
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getRuntimeExtent().elementsPerValue, 3ULL);

    const MarkedValueNodes marked = markedValueNodes(result.getValues());
    EXPECT_EQ(marked.values.op, ExprOp::STRIDED_VIEW);
    EXPECT_EQ(marked.values.view_dims, std::vector<uint64_t>({9, 3}));
    EXPECT_EQ(marked.values.view_strides, std::vector<uint64_t>({8, 1}));
    EXPECT_EQ(marked.values.view_element_offset, 2ULL);
    EXPECT_EQ(marked.marker.ragged_runtime_elements_per_value, 3ULL);
}

TEST(RaggedExpression, TrailingSliceSupportsNonLastTrailingAxis) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {4, 6}, 3, 9));

    const RaggedExpression result = ragged.sliceTrailingDimension(0, 1, 2);

    EXPECT_EQ(result.getValuesDimensions(), std::vector<uint64_t>({9, 2, 6}));
    EXPECT_EQ(result.getTrailingDimensions(), std::vector<uint64_t>({2, 6}));
    EXPECT_EQ(result.getRuntimeExtent().elementsPerValue, 12ULL);

    const MarkedValueNodes marked = markedValueNodes(result.getValues());
    EXPECT_EQ(marked.values.op, ExprOp::STRIDED_VIEW);
    EXPECT_EQ(marked.values.view_dims, std::vector<uint64_t>({9, 2, 6}));
    EXPECT_EQ(marked.values.view_strides, std::vector<uint64_t>({24, 6, 1}));
    EXPECT_EQ(marked.values.view_element_offset, 6ULL);
}

TEST(RaggedExpression, TrailingSliceRejectsInvalidRanges) {
    const RaggedExpression vector = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {8}, 3, 9));
    const RaggedExpression scalar = RaggedExpression::input("scalar", makeDescriptor(DataType::FP32, {}, 3, 9));

    EXPECT_THROW((void)vector.sliceTrailingDimension(1, 0, 1), std::invalid_argument);
    EXPECT_THROW((void)vector.sliceLastDimension(8, 1), std::invalid_argument);
    EXPECT_THROW((void)vector.sliceLastDimension(7, 2), std::invalid_argument);
    EXPECT_THROW((void)vector.sliceLastDimension(0, 0), std::invalid_argument);
    EXPECT_THROW((void)scalar.sliceLastDimension(0, 1), std::invalid_argument);
}

TEST(RaggedExpression, BinaryCompositionAfterTrailingSlicePreservesNarrowRuntimeExtent) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {8}, 3, 9));
    const RaggedExpression lhs = ragged.sliceLastDimension(0, 4);
    const RaggedExpression rhs = ragged.sliceLastDimension(4, 4);

    const RaggedExpression result = lhs + rhs;

    EXPECT_EQ(result.getTrailingDimensions(), std::vector<uint64_t>({4}));
    EXPECT_EQ(result.getRuntimeExtent().elementsPerValue, 4ULL);
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_EQ(markedValueNodes(result.getValues()).values.op, ExprOp::ADD);
}

TEST(RaggedExpression, NestedTrailingSlicesCanonicalizeToOriginalStorageStrides) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {8}, 3, 9));
    const RaggedExpression first = ragged.sliceLastDimension(2, 4);
    const RaggedExpression second = first.sliceLastDimension(1, 2);

    EXPECT_EQ(second.getTrailingDimensions(), std::vector<uint64_t>({2}));
    EXPECT_EQ(second.getRuntimeExtent().elementsPerValue, 2ULL);
    const MarkedValueNodes marked = markedValueNodes(second.getValues());
    EXPECT_EQ(marked.values.op, ExprOp::STRIDED_VIEW);
    EXPECT_EQ(marked.values.view_dims, std::vector<uint64_t>({9, 2}));
    EXPECT_EQ(marked.values.view_strides, std::vector<uint64_t>({8, 1}));
    EXPECT_EQ(marked.values.view_element_offset, 3ULL);
}

TEST(RaggedExpression, TrailingSliceAutodiffRewrapsScatterWithSourceRowWidth) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {6}, 3, 9));
    const RaggedExpression first_half = ragged.sliceLastDimension(0, 3);
    const RaggedExpression second_half = ragged.sliceLastDimension(3, 3);
    const RaggedExpression combined = first_half + second_half;

    const PhysicalOutputs forward = Expression::outputs({{"y", combined.getValues()}}).physicalOutputs();
    PhysicalOutputs backward = buildBackwardOutputs(forward, {"x.values"});
    resolveRaggedBackwardTestDTypes(backward, DataType::UINT32);

    bool found_source_width_marker = false;
    for (const ExprNode& node : backward.expr->nodes) {
        if (node.op != ExprOp::RAGGED_VALUEWISE_EXTENT || node.ragged_runtime_elements_per_value != 6ULL) continue;
        ASSERT_LT(node.lhs, backward.expr->nodes.size());
        if (backward.expr->nodes.at(node.lhs).op != ExprOp::STRIDED_VIEW_BACKWARD) continue;

        found_source_width_marker = true;
        EXPECT_EQ(node.ragged_runtime_batch_size, 3ULL);
        EXPECT_EQ(node.ragged_runtime_max_active_values, 9ULL);
        ASSERT_NE(node.rhs, UINT32_MAX);
        ASSERT_LT(node.rhs, backward.expr->nodes.size());
        const ExprNode& offsets = backward.expr->nodes.at(node.rhs);
        EXPECT_EQ(offsets.op, ExprOp::INPUT);
        ASSERT_LT(offsets.input_slot, backward.expr->inputs.size());
        EXPECT_EQ(backward.expr->inputs.at(offsets.input_slot).name, "x.offsets");
    }
    EXPECT_TRUE(found_source_width_marker);
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(backward);
    ASSERT_EQ(stages.size(), 1U);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    const std::string source = CudaSourceEmitter::emitFlat(stages[0], "ragged_trailing_slice_backward");
    EXPECT_NE(source.find("runtime_numel_u64 = active_values * 6ULL"), std::string::npos);
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

TEST(RaggedExpression, SegmentSumCanEmitFp32WithoutWideningLowPrecisionInputMaterialization) {
    for (DataType sourceDType : {DataType::FP8_E4M3, DataType::FP8_E5M2, DataType::FP16, DataType::BF16}) {
        const RaggedExpression ragged =
            RaggedExpression::input("x", makeDescriptor(sourceDType, {2}, 3, 9, DataType::UINT32));
        PhysicalOutputs outputs =
            Expression::outputs({{"y", ragged.segment_sum().withOutputDType(DataType::FP32)}}).physicalOutputs();

        std::vector<DataType> inputDTypes(outputs.expr->inputs.size(), sourceDType);
        for (const NamedInput& input : outputs.expr->inputs) {
            if (input.name == "x.offsets")
                inputDTypes.at(input.slot) = DataType::UINT32;
        }
        ASSERT_NO_THROW(resolveOutputsDTypesInPlace(outputs, inputDTypes));

        bool foundSegmentedSum = false;
        bool foundValuesInput = false;
        for (uint32_t nodeIndex = 0; nodeIndex < outputs.expr->nodes.size(); ++nodeIndex) {
            const ExprNode& node = outputs.expr->nodes[nodeIndex];
            if (node.op == ExprOp::SEGMENTED_REDUCE_SUM) {
                foundSegmentedSum = true;
                ASSERT_TRUE(node.output_dtype.has_value());
                EXPECT_EQ(node.output_dtype.value(), DataType::FP32);
            }
            if (node.op == ExprOp::INPUT && node.input_slot < outputs.expr->inputs.size() &&
                outputs.expr->inputs.at(node.input_slot).name == "x.values") {
                foundValuesInput = true;
                ASSERT_TRUE(node.input_tensor_dtype.has_value());
                ASSERT_TRUE(node.output_dtype.has_value());
                EXPECT_EQ(node.input_tensor_dtype.value(), sourceDType);
                EXPECT_EQ(node.output_dtype.value(), sourceDType);
                const std::optional<DataType> storageDType = materializedValueStorageDType(*outputs.expr, nodeIndex);
                ASSERT_TRUE(storageDType.has_value());
                EXPECT_EQ(storageDType.value(), sourceDType);
            }
        }
        EXPECT_TRUE(foundSegmentedSum);
        EXPECT_TRUE(foundValuesInput);
    }
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

TEST(RaggedExpression, SegmentReductionCanonicalizationSupportsAllOpsAndIncludesRaggedMetadata) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 3}, 3, 9));

    for (const Expression& reduction :
         {ragged.segment_sum(), ragged.segment_min(), ragged.segment_max(), ragged.segment_mean()}) {
        const std::string canonical = canonicalize(reduction.expression());
        EXPECT_NE(canonical.find("batch=3"), std::string::npos);
        EXPECT_NE(canonical.find("maxActive=9"), std::string::npos);
        EXPECT_NE(canonical.find("elementsPerValue=6"), std::string::npos);
    }

    const RaggedExpression larger_capacity =
        RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 3}, 3, 12));
    EXPECT_NE(canonicalize(ragged.segment_sum().expression()), canonicalize(larger_capacity.segment_sum().expression()));
}

TEST(RaggedExpression, SegmentedMinMaxBackwardCanonicalizationIncludesRaggedMetadata) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 3}, 3, 9));

    for (const Expression& reduction : {ragged.segment_min(), ragged.segment_max()}) {
        const PhysicalOutputs forward = Expression::outputs({{"y", reduction}}).physicalOutputs();
        const PhysicalOutputs backward = buildBackwardOutputs(forward, {"x.values"});
        const std::string canonical = canonicalize(backward);
        EXPECT_NE(canonical.find("batch=3"), std::string::npos);
        EXPECT_NE(canonical.find("maxActive=9"), std::string::npos);
        EXPECT_NE(canonical.find("elementsPerValue=6"), std::string::npos);
    }
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

TEST(RaggedExpression, BroadcastParameterGradientReducesOnlyAuthoritativePackedRows) {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 8;
    constexpr uint64_t width = 4;

    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression offsets = Expression::input("offsets", std::nullopt, DataType::UINT32);
    const Expression y = (x * scale).withRaggedRuntimeExtent(offsets, batchSize, maxTotalValues, width);
    const PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();

    const std::unordered_map<std::string, std::vector<uint64_t>> inputDims{
        {"x", {maxTotalValues, width}},
        {"scale", {width}},
        {"offsets", {batchSize + 1}},
    };
    PhysicalOutputs backward = buildBackwardOutputs(forward, {"scale"}, std::nullopt, inputDims);

    std::vector<DataType> backwardInputDTypes(backward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : backward.expr->inputs) {
        if (input.name == "offsets") {
            ASSERT_LT(input.slot, backwardInputDTypes.size());
            backwardInputDTypes[input.slot] = DataType::UINT32;
        }
    }
    resolveOutputsDTypesInPlace(backward, backwardInputDTypes);

    EXPECT_TRUE(containsOp(backward, ExprOp::SEGMENTED_REDUCE_SUM));
    EXPECT_TRUE(containsOp(backward, ExprOp::REDUCE_SUM));
    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(backward));
}

TEST(RaggedExpression, T9DAllDifferentiableActiveLocalBroadcastParametersUseSegmentedReduction) {
    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t width = 4;

    enum class BinaryKind { Add, Sub, Mul, Div, Pow, Min, Max, Where };
    for (BinaryKind kind : {BinaryKind::Add,
                            BinaryKind::Sub,
                            BinaryKind::Mul,
                            BinaryKind::Div,
                            BinaryKind::Pow,
                            BinaryKind::Min,
                            BinaryKind::Max,
                            BinaryKind::Where}) {
        SCOPED_TRACE(static_cast<int>(kind));
        const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
        const Expression parameter = Expression::input("parameter", DataType::FP32, DataType::FP32);
        const Expression offsets = Expression::input("offsets", std::nullopt, DataType::UINT32);
        const Expression value = [&]() -> Expression {
            switch (kind) {
                case BinaryKind::Add:
                    return x + parameter;
                case BinaryKind::Sub:
                    return x - parameter;
                case BinaryKind::Mul:
                    return x * parameter;
                case BinaryKind::Div:
                    return x / parameter;
                case BinaryKind::Pow:
                    return x.pow(parameter);
                case BinaryKind::Min:
                    return x.min(parameter);
                case BinaryKind::Max:
                    return x.max(parameter);
                case BinaryKind::Where:
                    return Expression::where(x > Expression::constantScalar(0.0), x, parameter);
            }
            throw std::logic_error("unhandled T9D binary-kind test case");
        }();
        const PhysicalOutputs forward = Expression::outputs(
            {{"y", value.withRaggedRuntimeExtent(offsets, batch_size, max_total_values, width)}})
                                            .physicalOutputs();
        PhysicalOutputs backward = buildBackwardOutputs(
            forward,
            {"parameter"},
            std::nullopt,
            std::unordered_map<std::string, std::vector<uint64_t>>{
                {"x", {max_total_values, width}},
                {"parameter", {width}},
                {"offsets", {batch_size + 1}},
            });
        std::vector<DataType> input_dtypes(backward.expr->inputs.size(), DataType::FP32);
        for (const NamedInput& input : backward.expr->inputs) {
            if (input.name == "offsets") {
                input_dtypes.at(input.slot) = DataType::UINT32;
            }
        }
        resolveOutputsDTypesInPlace(backward, input_dtypes);
        EXPECT_TRUE(containsOp(backward, ExprOp::SEGMENTED_REDUCE_SUM));
        EXPECT_TRUE(containsOp(backward, ExprOp::REDUCE_SUM));
        EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(backward));
    }

    // Scalar-tensor parameters reduce both the authoritative packed axis and
    // trailing feature axes. They still must consume packed axis 0 through the
    // segmented reduction before any ordinary dense reduction.
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression scalar_parameter = Expression::input("scalar_parameter", DataType::FP32, DataType::FP32);
    const Expression offsets = Expression::input("offsets", std::nullopt, DataType::UINT32);
    const PhysicalOutputs scalar_forward = Expression::outputs(
        {{"y", (x * scalar_parameter).withRaggedRuntimeExtent(offsets, batch_size, max_total_values, width)}})
                                                     .physicalOutputs();
    PhysicalOutputs scalar_backward = buildBackwardOutputs(
        scalar_forward,
        {"scalar_parameter"},
        std::nullopt,
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {max_total_values, width}},
            {"scalar_parameter", {1}},
            {"offsets", {batch_size + 1}},
        });
    std::vector<DataType> scalar_input_dtypes(scalar_backward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& input : scalar_backward.expr->inputs) {
        if (input.name == "offsets") {
            scalar_input_dtypes.at(input.slot) = DataType::UINT32;
        }
    }
    resolveOutputsDTypesInPlace(scalar_backward, scalar_input_dtypes);
    EXPECT_TRUE(containsOp(scalar_backward, ExprOp::SEGMENTED_REDUCE_SUM));
    EXPECT_TRUE(containsOp(scalar_backward, ExprOp::REDUCE_SUM));
    EXPECT_NO_THROW((void)EquationCompiler::splitAtReductionBoundaries(scalar_backward));
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

TEST(RaggedExpression, TrailingSlicesComposeAndAutodiffOnlyAcrossActivePackedRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 5;
    constexpr uint64_t source_width = 6;
    constexpr uint64_t half_width = 3;
    const RaggedExpression ragged =
        RaggedExpression::input("x", makeDescriptor(DataType::FP32, {source_width}, batch_size, max_total_values));
    const RaggedExpression first_half = ragged.sliceLastDimension(0, half_width);
    const RaggedExpression second_half = ragged.sliceLastDimension(half_width, half_width);
    const RaggedExpression combined = first_half + second_half;

    Tensor values = makeGpuTensor<float>({max_total_values, source_width},
                                         {1.0F, 2.0F, 3.0F, 10.0F, 20.0F, 30.0F,
                                          4.0F, 5.0F, 6.0F, 40.0F, 50.0F, 60.0F,
                                          7.0F, 8.0F, 9.0F, 70.0F, 80.0F, 90.0F,
                                          1001.0F, 1002.0F, 1003.0F, 1010.0F, 1020.0F, 1030.0F,
                                          2001.0F, 2002.0F, 2003.0F, 2010.0F, 2020.0F, 2030.0F},
                                         stream);
    Tensor offsets = makeGpuTensor<uint32_t>({batch_size + 1}, {0U, 1U, 1U, 3U}, stream);

    Tensor forward_output(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, half_width}));
    forward_output.fill(777.0, stream);
    const Tensor forward_result = runExpressionOutput(
        combined.getValues(), {{"x.values", values}, {"x.offsets", offsets}}, "y", stream, forward_output);
    expectNear(copyToCpuValues(forward_result, stream),
               {11.0F, 22.0F, 33.0F,
                44.0F, 55.0F, 66.0F,
                77.0F, 88.0F, 99.0F,
                777.0F, 777.0F, 777.0F,
                777.0F, 777.0F, 777.0F});

    Tensor upstream = makeGpuTensor<float>({max_total_values, half_width},
                                           {1.0F, 2.0F, 3.0F,
                                            4.0F, 5.0F, 6.0F,
                                            7.0F, 8.0F, 9.0F,
                                            100.0F, 200.0F, 300.0F,
                                            400.0F, 500.0F, 600.0F},
                                           stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, source_width}));
    gradient.fill(777.0, stream);

    const Tensor backward_result = runBackwardOutput(combined.getValues(),
                                                     {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                     "x.values",
                                                     "dy",
                                                     stream,
                                                     gradient);
    expectNear(copyToCpuValues(backward_result, stream),
               {1.0F, 2.0F, 3.0F, 1.0F, 2.0F, 3.0F,
                4.0F, 5.0F, 6.0F, 4.0F, 5.0F, 6.0F,
                7.0F, 8.0F, 9.0F, 7.0F, 8.0F, 9.0F,
                777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F,
                777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F});
}

TEST(RaggedExpression, NestedTrailingSlicesExecuteAndAutodiffThroughComposedViews) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 4;
    constexpr uint64_t source_width = 8;
    const RaggedExpression ragged =
        RaggedExpression::input("x", makeDescriptor(DataType::FP32, {source_width}, batch_size, max_total_values));
    const RaggedExpression nested = ragged.sliceLastDimension(2, 4).sliceLastDimension(1, 2);

    // Consecutive ragged slices must collapse to one storage view.  The first slice
    // selects columns [2, 6) with row stride 8; the second selects [1, 3) within
    // that view, so the canonical storage view is columns [3, 5) with the original
    // row stride preserved.
    const PhysicalExpression nested_physical = nested.getValues().expression();
    const ExprNode& extent_node = nested_physical.nodes.at(nested_physical.output_node);
    ASSERT_EQ(extent_node.op, ExprOp::RAGGED_VALUEWISE_EXTENT);
    const ExprNode& nested_view = nested_physical.nodes.at(extent_node.lhs);
    ASSERT_EQ(nested_view.op, ExprOp::STRIDED_VIEW);
    EXPECT_EQ(nested_view.view_dims, (std::vector<uint64_t>{max_total_values, 2}));
    EXPECT_EQ(nested_view.view_strides, (std::vector<uint64_t>{source_width, 1}));
    EXPECT_EQ(nested_view.view_element_offset, 3U);
    ASSERT_LT(nested_view.lhs, nested_physical.nodes.size());
    EXPECT_NE(nested_physical.nodes.at(nested_view.lhs).op, ExprOp::STRIDED_VIEW);

    Tensor values = makeGpuTensor<float>({max_total_values, source_width},
                                         {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
                                          11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F, 18.0F,
                                          101.0F, 102.0F, 103.0F, 104.0F, 105.0F, 106.0F, 107.0F, 108.0F,
                                          201.0F, 202.0F, 203.0F, 204.0F, 205.0F, 206.0F, 207.0F, 208.0F},
                                         stream);
    Tensor offsets = makeGpuTensor<uint32_t>({batch_size + 1}, {0U, 1U, 2U}, stream);

    Tensor forward_output(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, 2}));
    forward_output.fill(777.0, stream);
    const Tensor forward_result =
        runExpressionOutput(nested.getValues(), {{"x.values", values}, {"x.offsets", offsets}}, "y", stream, forward_output);
    expectNear(copyToCpuValues(forward_result, stream),
               {4.0F, 5.0F,
                14.0F, 15.0F,
                777.0F, 777.0F,
                777.0F, 777.0F});

    Tensor upstream = makeGpuTensor<float>({max_total_values, 2},
                                           {2.0F, 3.0F,
                                            5.0F, 7.0F,
                                            100.0F, 200.0F,
                                            300.0F, 400.0F},
                                           stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, source_width}));
    gradient.fill(777.0, stream);
    const Tensor backward_result = runBackwardOutput(nested.getValues(),
                                                     {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                     "x.values",
                                                     "dy",
                                                     stream,
                                                     gradient);
    expectNear(copyToCpuValues(backward_result, stream),
               {0.0F, 0.0F, 0.0F, 2.0F, 3.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 5.0F, 7.0F, 0.0F, 0.0F, 0.0F,
                777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F,
                777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F, 777.0F});
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

TEST(RaggedExpression, VectorSegmentMinBackwardReusesDynamicActivePrefixWithoutTouchingReservedCapacity) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 5;
    constexpr uint64_t elements_per_value = 2;
    const RaggedExpression ragged = RaggedExpression::input(
        "x", makeDescriptor(DataType::FP32, {elements_per_value}, batch_size, max_total_values, DataType::UINT32));
    Tensor values = makeGpuTensor<float>({max_total_values, elements_per_value},
                                         {1.0F, 10.0F,
                                          2.0F, 9.0F,
                                          3.0F, 8.0F,
                                          4.0F, 7.0F,
                                          99.0F, 100.0F},
                                         stream);
    Tensor offsets = makeGpuTensor<uint32_t>({batch_size + 1}, {0U, 1U, 3U}, stream);
    Tensor upstream = makeGpuTensor<float>({batch_size, elements_per_value}, {10.0F, 11.0F, 20.0F, 21.0F}, stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, elements_per_value}));
    gradient.fill(777.0, stream);

    FusedEquation forward =
        FusedEquation::compile(Expression::outputs({{"y", ragged.segment_min()}}).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"x.values"}, "dy");
    StampedExecutionPlan plan = backward.stamp({{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                               stream,
                                               {},
                                               {{"x.values_grad", gradient}});

    plan.run();
    stream.synchronize();
    expectNear(copyToCpuValues(plan.output("x.values_grad"), stream),
               {10.0F, 11.0F,
                20.0F, 0.0F,
                0.0F, 21.0F,
                777.0F, 777.0F,
                777.0F, 777.0F});

    overwriteGpuTensor<uint32_t>(offsets, {0U, 2U, 4U}, stream);
    gradient.fill(888.0, stream);
    plan.run();
    stream.synchronize();
    expectNear(copyToCpuValues(plan.output("x.values_grad"), stream),
               {10.0F, 0.0F,
                0.0F, 11.0F,
                20.0F, 0.0F,
                0.0F, 21.0F,
                888.0F, 888.0F});
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

TEST(RaggedExpression, MapValuesWrapsWholePointwiseTailInOneRaggedExtent) {
    RaggedExpression input = RaggedExpression::input("tokens", makeDescriptor(DataType::FP32, {3}, 2, 8));
    RaggedExpression mapped = input.mapValues([](const Expression& values) {
        return (values.swish() + Expression(0.25)).tanh();
    });

    const PhysicalExpression physical = mapped.getValues().expression();
    const ExprNode& marker = physical.nodes.at(physical.output_node);
    ASSERT_EQ(marker.op, ExprOp::RAGGED_VALUEWISE_EXTENT);
    ASSERT_EQ(marker.ragged_runtime_max_active_values, 8ULL);
    ASSERT_EQ(marker.ragged_runtime_elements_per_value, 3ULL);
    EXPECT_EQ(mapped.getOffsets().getInputNames(), (std::set<std::string>{"tokens.offsets"}));
    EXPECT_EQ(mapped.getDifferentiableInputNames(), (std::set<std::string>{"tokens.values"}));

    const PhysicalOutputs outputs = Expression::outputs({{"y", mapped.getValues()}}).physicalOutputs();
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1U);
    EXPECT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
}

TEST(RaggedExpression, MapValuesExecutesAndAutodiffsOnlyAcrossActivePackedRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t max_total_values = 6;
    const RaggedExpression input =
        RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2}, 3, max_total_values));
    const RaggedExpression mapped = input.mapValues([](const Expression& values) {
        return values.max(Expression(0.0)) + Expression(0.5);
    });

    Tensor values = makeGpuTensor<float>({max_total_values, 2},
                                         {-2.0F, 1.0F,
                                          3.0F, -4.0F,
                                          5.0F, 6.0F,
                                          101.0F, 102.0F,
                                          103.0F, 104.0F,
                                          105.0F, 106.0F},
                                         stream);
    Tensor offsets = makeGpuTensor<uint32_t>({4}, {0U, 1U, 1U, 3U}, stream);

    Tensor forward_output(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, 2}));
    forward_output.fill(777.0, stream);
    const Tensor forward_result = runExpressionOutput(
        mapped.getValues(), {{"x.values", values}, {"x.offsets", offsets}}, "y", stream, forward_output);
    expectNear(copyToCpuValues(forward_result, stream),
               {0.5F, 1.5F,
                3.5F, 0.5F,
                5.5F, 6.5F,
                777.0F, 777.0F,
                777.0F, 777.0F,
                777.0F, 777.0F});

    Tensor upstream = makeGpuTensor<float>({max_total_values, 2},
                                           {1.0F, 2.0F,
                                            3.0F, 4.0F,
                                            5.0F, 6.0F,
                                            100.0F, 200.0F,
                                            300.0F, 400.0F,
                                            500.0F, 600.0F},
                                           stream);
    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, 2}));
    gradient.fill(777.0, stream);
    const Tensor backward_result = runBackwardOutput(mapped.getValues(),
                                                     {{"x.values", values}, {"x.offsets", offsets}, {"dy", upstream}},
                                                     "x.values",
                                                     "dy",
                                                     stream,
                                                     gradient);
    expectNear(copyToCpuValues(backward_result, stream),
               {0.0F, 2.0F,
                3.0F, 0.0F,
                5.0F, 6.0F,
                777.0F, 777.0F,
                777.0F, 777.0F,
                777.0F, 777.0F});
}

TEST(RaggedExpression, PackedRmsNormLowersRowPartitionAsStructuralStageInput) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression offsets = Expression::input("offsets", DataType::UINT64, DataType::UINT64);
    const Expression packed = x.withRaggedRuntimeExtent(offsets, 3, 9, 4);
    const Expression y = Expression::rmsNorm(packed, scale, 4, 1.0e-5, DataType::FP32, DataType::FP32, 9);

    PhysicalOutputs outputs = Expression::outputs({{"y", y}}).physicalOutputs();
    ASSERT_NE(outputs.expr, nullptr);
    std::vector<DataType> input_dtypes(outputs.expr->numInputs(), DataType::FP32);
    for (const NamedInput& input : outputs.expr->inputs) {
        ASSERT_LT(input.slot, input_dtypes.size());
        if (input.name == "offsets") {
            input_dtypes[input.slot] = DataType::UINT64;
        }
    }
    resolveOutputsDTypesInPlace(outputs, input_dtypes);
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1U);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::RmsNorm);
    ASSERT_EQ(stages[0].input_value_ids.size(), 3U);

    const std::shared_ptr<CompiledRmsNorm> compiled = EquationCompiler::compileRmsNorm(stages[0].expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->packed_row_capacity, 9U);
    EXPECT_EQ(compiled->ragged_offsets_input_slot, 2U);
    EXPECT_EQ(compiled->ragged_batch_size, 3U);
}



TEST(RaggedExpression, PackedRmsNormC6OwnsFiniteExecutableFamiliesAndNeverPreparesAtRuntime) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t capacity = 16;
    constexpr uint64_t hidden = 4;
    constexpr uint64_t batch_size = 2;
    const std::vector<uint64_t> buckets = makeRaggedRmsNormCapacityBuckets(capacity);
    ASSERT_EQ(buckets, (std::vector<uint64_t>{8, 16}));

    Stream stream_a(0);
    Stream stream_b(0);
    Tensor x_a(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor x_b(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor scale_a(gpuPlacement, TensorDescriptor(DataType::FP32, {hidden}));
    Tensor scale_b(gpuPlacement, TensorDescriptor(DataType::FP32, {hidden}));
    Tensor y_a(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor y_b(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor offsets_a = makeGpuTensor<uint32_t>({batch_size + 1}, {0, 4, 8}, stream_a);
    Tensor offsets_b = makeGpuTensor<uint32_t>({batch_size + 1}, {0, 4, 8}, stream_b);
    x_a.fill(0.5, stream_a);
    x_b.fill(0.5, stream_b);
    scale_a.fill(1.0, stream_a);
    scale_b.fill(1.0, stream_b);

    auto compiled = std::make_shared<CompiledRmsNorm>();
    compiled->normalized_feature_count = hidden;
    compiled->packed_row_capacity = capacity;
    compiled->ragged_batch_size = batch_size;
    compiled->epsilon = 1.0e-5;
    compiled->input_dtype = DataType::FP32;
    compiled->scale_dtype = DataType::FP32;
    compiled->output_dtype = DataType::FP32;
    compiled->compute_dtype = DataType::FP32;
    compiled->debug_name = "c6_packed_rmsnorm_forward";

    CudnnRmsNorm& rms_norm = CudnnRmsNorm::instance();
    rms_norm.clearSelectionCache();
    auto stamp_a = std::make_shared<StampedRmsNorm>(compiled, x_a, scale_a, y_a, stream_a, offsets_a);
    auto stamp_b = std::make_shared<StampedRmsNorm>(compiled, x_b, scale_b, y_b, stream_b, offsets_b);

    ASSERT_EQ(stamp_a->executablePlanCount(), buckets.size());
    ASSERT_EQ(stamp_b->executablePlanCount(), buckets.size());
    EXPECT_EQ(stamp_a->planSelections(), stamp_b->planSelections());
    const std::vector<uintptr_t> ids_a = stamp_a->executablePlanIds();
    const std::vector<uintptr_t> ids_b = stamp_b->executablePlanIds();
    ASSERT_EQ(ids_a.size(), ids_b.size());
    for (size_t i = 0; i < ids_a.size(); ++i) EXPECT_NE(ids_a[i], ids_b[i]);
    EXPECT_EQ(rms_norm.cachedSelectionCount(), buckets.size());

    Tensor dy(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor dx(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor dscale(gpuPlacement, TensorDescriptor(DataType::FP32, {hidden}));
    dy.fill(1.0, stream_a);
    auto compiled_backward = std::make_shared<CompiledRmsNormBackward>();
    compiled_backward->normalized_feature_count = hidden;
    compiled_backward->packed_row_capacity = capacity;
    compiled_backward->ragged_batch_size = batch_size;
    compiled_backward->epsilon = compiled->epsilon;
    compiled_backward->input_dtype = DataType::FP32;
    compiled_backward->scale_dtype = DataType::FP32;
    compiled_backward->dy_dtype = DataType::FP32;
    compiled_backward->dx_dtype = DataType::FP32;
    compiled_backward->dscale_dtype = DataType::FP32;
    compiled_backward->compute_dtype = DataType::FP32;
    compiled_backward->debug_name = "c6_packed_rmsnorm_backward";

    StampedRmsNormBackward standalone_backward(
        compiled_backward, x_a, scale_a, dy, dx, dscale, stream_a, offsets_a);
    EXPECT_EQ(standalone_backward.backwardExecutablePlanCount(), buckets.size());
    EXPECT_EQ(standalone_backward.fallbackForwardExecutablePlanCount(), buckets.size());

    // Everything below is execution only. Selection state can disappear and
    // every bucket transition must choose among already-owned local plans.
    rms_norm.clearSelectionCache();
    ASSERT_EQ(rms_norm.cachedSelectionCount(), 0U);
    const uint64_t preparations_after_stamping = cudnnFrontendExecutablePreparationCountForTests();
    for (const uint64_t active_rows : std::vector<uint64_t>{7, 9, capacity}) {
        SCOPED_TRACE("activeRows=" + std::to_string(active_rows));
        RowPartitionRuntime(offsets_a, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32))
            .setHostActiveValueCount(active_rows);
        RowPartitionRuntime(offsets_b, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32))
            .setHostActiveValueCount(active_rows);
        stamp_a->runOn(stream_a);
        stamp_b->runOn(stream_b);
        standalone_backward.runOn(stream_a);
        stream_a.synchronize();
        stream_b.synchronize();
        EXPECT_EQ(rms_norm.cachedSelectionCount(), 0U);
        EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparations_after_stamping)
            << "RMSNorm bucket transitions must not build/replay/deserialize plans at runtime.";
    }
}

TEST(RaggedExpression, PackedLayerNormUsesFiniteRmsNormCapacityFamilyWithoutTailSanitation) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t capacity = 16;
    constexpr uint64_t hidden = 4;
    constexpr uint64_t batch_size = 2;
    constexpr float inactive_sentinel = 32123.0F;
    const std::vector<uint64_t> buckets = makeRaggedRmsNormCapacityBuckets(capacity);
    ASSERT_EQ(buckets, (std::vector<uint64_t>{8, 16}));

    Stream stream_a(0);
    Stream stream_b(0);
    Tensor x_a(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor x_b(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor scale_a(gpuPlacement, TensorDescriptor(DataType::FP32, {hidden}));
    Tensor scale_b(gpuPlacement, TensorDescriptor(DataType::FP32, {hidden}));
    Tensor bias_a(gpuPlacement, TensorDescriptor(DataType::FP32, {hidden}));
    Tensor bias_b(gpuPlacement, TensorDescriptor(DataType::FP32, {hidden}));
    Tensor y_a(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor y_b(gpuPlacement, TensorDescriptor(DataType::FP32, {capacity, hidden}));
    Tensor offsets_a = makeGpuTensor<uint32_t>({batch_size + 1}, {0, 4, 7}, stream_a);
    Tensor offsets_b = makeGpuTensor<uint32_t>({batch_size + 1}, {0, 4, 7}, stream_b);
    x_a.fill(0.5, stream_a);
    x_b.fill(0.75, stream_b);
    scale_a.fill(1.0, stream_a);
    scale_b.fill(1.0, stream_b);
    bias_a.fill(0.0, stream_a);
    bias_b.fill(0.0, stream_b);
    y_a.fill(inactive_sentinel, stream_a);
    y_b.fill(inactive_sentinel, stream_b);

    auto compiled = std::make_shared<CompiledLayerNorm>();
    compiled->normalized_feature_count = hidden;
    compiled->packed_row_capacity = capacity;
    compiled->ragged_batch_size = batch_size;
    compiled->epsilon = 1.0e-5;
    compiled->input_dtype = DataType::FP32;
    compiled->scale_dtype = DataType::FP32;
    compiled->bias_dtype = DataType::FP32;
    compiled->output_dtype = DataType::FP32;
    compiled->compute_dtype = DataType::FP32;
    compiled->debug_name = "packed_layernorm_finite_capacity_family";

    CudnnLayerNorm& layer_norm = CudnnLayerNorm::instance();
    layer_norm.clearSelectionCache();
    auto stamp_a = std::make_shared<StampedLayerNorm>(compiled, x_a, scale_a, bias_a, y_a, stream_a, offsets_a);
    auto stamp_b = std::make_shared<StampedLayerNorm>(compiled, x_b, scale_b, bias_b, y_b, stream_b, offsets_b);

    ASSERT_EQ(stamp_a->executablePlanCount(), buckets.size());
    ASSERT_EQ(stamp_b->executablePlanCount(), buckets.size());
    EXPECT_EQ(stamp_a->planSelections(), stamp_b->planSelections());
    const std::vector<uintptr_t> ids_a = stamp_a->executablePlanIds();
    const std::vector<uintptr_t> ids_b = stamp_b->executablePlanIds();
    ASSERT_EQ(ids_a.size(), ids_b.size());
    for (size_t i = 0; i < ids_a.size(); ++i) EXPECT_NE(ids_a[i], ids_b[i]);
    EXPECT_EQ(layer_norm.cachedSelectionCount(), buckets.size());

    layer_norm.clearSelectionCache();
    ASSERT_EQ(layer_norm.cachedSelectionCount(), 0U);
    const uint64_t preparations_after_stamping = cudnnFrontendExecutablePreparationCountForTests();

    RowPartitionRuntime(offsets_a, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32))
        .setHostActiveValueCount(7);
    RowPartitionRuntime(offsets_b, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32))
        .setHostActiveValueCount(7);
    stamp_a->runOn(stream_a);
    stamp_b->runOn(stream_b);
    stream_a.synchronize();
    stream_b.synchronize();

    EXPECT_EQ(layer_norm.cachedSelectionCount(), 0U);
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparations_after_stamping)
        << "LayerNorm bucket selection must use only stamp-local prebuilt executables at runtime.";

    // Seven active rows select the 8-row bucket. LayerNorm is row-local, so it
    // deliberately does not sanitize the inactive row inside that bucket; rows
    // outside the selected bucket must not be touched at all.
    const std::vector<float> actual = copyToCpuValues(y_a, stream_a);
    for (uint64_t row = 8; row < capacity; ++row) {
        for (uint64_t channel = 0; channel < hidden; ++channel) {
            EXPECT_EQ(actual[row * hidden + channel], inactive_sentinel)
                << "row outside selected LayerNorm bucket was unexpectedly written";
        }
    }

    // Crossing into the 16-row bucket must select another executable that was
    // already prepared when the stamp was built.
    RowPartitionRuntime(offsets_a, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32))
        .setHostActiveValueCount(9);
    RowPartitionRuntime(offsets_b, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32))
        .setHostActiveValueCount(9);
    stamp_a->runOn(stream_a);
    stamp_b->runOn(stream_b);
    stream_a.synchronize();
    stream_b.synchronize();
    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), preparations_after_stamping)
        << "LayerNorm bucket transitions must not prepare executables at runtime.";
}

TEST(RaggedExpression, PackedRmsNormAutodiffUsesBucketedCudnnBackwardWithStructuralOffsets) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression scale = Expression::input("scale", DataType::FP32, DataType::FP32);
    const Expression offsets = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression packed = x.withRaggedRuntimeExtent(offsets, 3, 9, 4);
    const Expression y = Expression::rmsNorm(packed, scale, 4, 1.0e-5, DataType::FP32, DataType::FP32, 9);

    PhysicalOutputs forward = Expression::outputs({{"y", y}}).physicalOutputs();
    std::vector<DataType> forward_input_dtypes;
    forward_input_dtypes.reserve(forward.expr->inputs.size());
    for (const NamedInput& input : forward.expr->inputs) {
        forward_input_dtypes.push_back(input.name == "offsets" ? DataType::UINT32 : DataType::FP32);
    }
    resolveOutputsDTypesInPlace(forward, forward_input_dtypes);

    PhysicalOutputs backward = buildBackwardOutputs(
        forward,
        {"x", "scale"},
        std::unordered_map<std::string, std::string>{{"y", "dy"}},
        std::unordered_map<std::string, DataType>{{"y", DataType::FP32}},
        std::unordered_map<std::string, std::vector<uint64_t>>{
            {"x", {9, 4}},
            {"scale", {4}},
            {"offsets", {4}},
        });
    std::vector<DataType> backward_input_dtypes;
    backward_input_dtypes.reserve(backward.expr->inputs.size());
    for (const NamedInput& input : backward.expr->inputs) {
        backward_input_dtypes.push_back(input.name == "offsets" ? DataType::UINT32 : DataType::FP32);
    }
    resolveOutputsDTypesInPlace(backward, backward_input_dtypes);

    size_t dx_routes = 0;
    size_t dscale_routes = 0;
    for (const ExprNode& node : backward.expr->nodes) {
        dx_routes += node.op == ExprOp::RMSNORM_BACKWARD_X ? 1u : 0u;
        dscale_routes += node.op == ExprOp::RMSNORM_BACKWARD_SCALE ? 1u : 0u;
        EXPECT_NE(node.op, ExprOp::SQRT);
        EXPECT_NE(node.op, ExprOp::REDUCE_AVG);
        EXPECT_NE(node.op, ExprOp::REDUCE_SUM);
    }
    EXPECT_EQ(dx_routes, 1u);
    EXPECT_EQ(dscale_routes, 1u);

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(backward);
    size_t rms_backward_stage_count = 0;
    for (const PhysicalExecutionStage& stage : stages) {
        if (stage.kind != PhysicalExecutionStage::Kind::RmsNormBackward) {
            continue;
        }
        ++rms_backward_stage_count;
        EXPECT_EQ(stage.input_value_ids.size(), 4u);
        ASSERT_EQ(stage.outputs.size(), 2u);
        const std::shared_ptr<CompiledRmsNormBackward> compiled = EquationCompiler::compileRmsNormBackward(stage.expr);
        ASSERT_NE(compiled, nullptr);
        EXPECT_EQ(compiled->packed_row_capacity, 9u);
        EXPECT_EQ(compiled->ragged_offsets_input_slot, 3u);
        EXPECT_EQ(compiled->ragged_batch_size, 3u);
    }
    EXPECT_EQ(rms_backward_stage_count, 1u);
}


TEST(RaggedExpression, PackedMatmulForwardAndAutodiffUseOffsetsRuntimeWithoutValuesMetadata) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t capacity = 8;
    constexpr uint64_t active_rows = 3;
    constexpr uint64_t width = 2;

    Tensor values = makeGpuTensor<float>({capacity, width},
                                         {1.0F, 2.0F,
                                          3.0F, 4.0F,
                                          5.0F, 6.0F,
                                          0.0F, 0.0F,
                                          0.0F, 0.0F,
                                          0.0F, 0.0F,
                                          0.0F, 0.0F,
                                          0.0F, 0.0F},
                                         stream);
    Tensor weights = makeGpuTensor<float>({width, width},
                                          {2.0F, 0.0F,
                                           0.0F, 3.0F},
                                          stream);
    Tensor offsets = makeGpuTensor<uint32_t>({batch_size + 1}, {0U, 1U, 3U}, stream);
    RowPartitionRuntime(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32))
        .setHostActiveValueCount(active_rows);


    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression w = Expression::input("w", DataType::FP32, DataType::FP32);
    const Expression row_offsets = Expression::input("offsets", DataType::UINT32, DataType::UINT32);
    const Expression packed_x = x.withRaggedRuntimeExtent(row_offsets, batch_size, capacity, width);
    const Expression y = Expression::matmul(packed_x,
                                             w,
                                             false,
                                             false,
                                             DataType::FP32,
                                             DataType::FP32,
                                             capacity);

    const std::unordered_map<std::string, Tensor> forward_inputs{{"x", values}, {"w", weights}, {"offsets", offsets}};
    const Tensor forward = runExpressionOutput(y, forward_inputs, "y", stream);
    const std::vector<float> forward_values = copyToCpuValues(forward, stream);
    ASSERT_GE(forward_values.size(), active_rows * width);
    expectNear(std::vector<float>(forward_values.begin(), forward_values.begin() + active_rows * width),
               {2.0F, 6.0F,
                6.0F, 12.0F,
                10.0F, 18.0F});

    Tensor upstream = makeGpuTensor<float>({capacity, width},
                                           {1.0F, 2.0F,
                                            3.0F, 4.0F,
                                            5.0F, 6.0F,
                                            0.0F, 0.0F,
                                            0.0F, 0.0F,
                                            0.0F, 0.0F,
                                            0.0F, 0.0F,
                                            0.0F, 0.0F},
                                           stream);
    const std::unordered_map<std::string, Tensor> backward_inputs{
        {"x", values}, {"w", weights}, {"offsets", offsets}, {"dy", upstream}};

    const Tensor dx = runBackwardOutput(y, backward_inputs, "x", "dy", stream);
    const std::vector<float> dx_values = copyToCpuValues(dx, stream);
    ASSERT_GE(dx_values.size(), active_rows * width);
    expectNear(std::vector<float>(dx_values.begin(), dx_values.begin() + active_rows * width),
               {2.0F, 6.0F,
                6.0F, 12.0F,
                10.0F, 18.0F});

    const Tensor dw = runBackwardOutput(y, backward_inputs, "w", "dy", stream);
    expectNear(copyToCpuValues(dw, stream),
               {35.0F, 44.0F,
                44.0F, 56.0F});

    EXPECT_EQ(RowPartitionRuntime(offsets, RowPartitionDescriptor(batch_size, capacity, DataType::UINT32))
                  .requireHostActiveValueCount(),
              active_rows);
}

TEST(RaggedExpression, RaggedAttentionBackwardConsumesMatchingQueryExtentOnUpstreamGradient) {
    auto build = [](bool use_query_partition_for_do) {
        auto expr = std::make_shared<PhysicalExpression>();
        expr->inputs = {
            NamedInput{"q", 0, NamedInput::Kind::Tensor},
            NamedInput{"k", 1, NamedInput::Kind::Tensor},
            NamedInput{"v", 2, NamedInput::Kind::Tensor},
            NamedInput{"do", 3, NamedInput::Kind::Tensor},
            NamedInput{"q_offsets", 4, NamedInput::Kind::Tensor},
            NamedInput{"kv_offsets", 5, NamedInput::Kind::Tensor},
        };

        auto addInput = [&](uint32_t slot, DataType dtype) {
            ExprNode input;
            input.op = ExprOp::INPUT;
            input.input_slot = slot;
            input.input_tensor_dtype = dtype;
            input.output_dtype = dtype;
            input.compute_dtype = dtype == DataType::FP32 ? std::optional<DataType>(DataType::FP32) : std::nullopt;
            expr->nodes.push_back(std::move(input));
        };
        addInput(0, DataType::FP32);
        addInput(1, DataType::FP32);
        addInput(2, DataType::FP32);
        addInput(3, DataType::FP32);
        addInput(4, DataType::UINT32);
        addInput(5, DataType::UINT32);

        ExprNode extent;
        extent.op = ExprOp::RAGGED_VALUEWISE_EXTENT;
        extent.lhs = 3;
        extent.rhs = use_query_partition_for_do ? 4 : 5;
        extent.ragged_runtime_batch_size = 2;
        extent.ragged_runtime_max_active_values = 6;
        extent.ragged_runtime_elements_per_value = 4;
        extent.output_dtype = DataType::FP32;
        extent.compute_dtype = DataType::FP32;
        expr->nodes.push_back(std::move(extent));

        // Match the real ragged-query Attention training path: the active-aware dO
        // extent can flow through a metadata-only shape transform before cuDNN backward.
        ExprNode reshape;
        reshape.op = ExprOp::RESHAPE;
        reshape.lhs = 6;
        reshape.reshape_dims = {6, 1, 4};
        reshape.output_dtype = DataType::FP32;
        reshape.compute_dtype = DataType::FP32;
        expr->nodes.push_back(std::move(reshape));

        ExprNode backward;
        backward.op = ExprOp::ATTENTION_BACKWARD_Q;
        backward.lhs = 0;
        backward.rhs = 1;
        backward.aux = 2;
        backward.alpha_node = 7;
        backward.attention_use_ragged_offsets = true;
        backward.attention_ragged_offset_q_node = 4;
        backward.attention_ragged_offset_kv_node = 5;
        backward.output_dtype = DataType::FP32;
        backward.compute_dtype = DataType::FP32;
        expr->nodes.push_back(std::move(backward));
        expr->output_node = 8;

        return PhysicalOutputs{
            .expr = std::move(expr),
            .outputs = {NamedOutput{"dq", 8}},
        };
    };

    const PhysicalOutputs matching = build(true);
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(matching);
    ASSERT_FALSE(stages.empty());
    EXPECT_EQ(stages.back().kind, PhysicalExecutionStage::Kind::AttentionBackward);

    const PhysicalOutputs mismatched = build(false);
    try {
        (void)EquationCompiler::splitAtReductionBoundaries(mismatched);
        FAIL() << "Expected ragged Attention backward to reject dO carrying the KV partition.";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("same row partition as the query/output domain"), std::string::npos);
    }
}

TEST(RaggedExpression, CausalConv1dPreservesRowPartitionAndRecordsLogicalContract) {
    const RaggedTensorDescriptor descriptor = makeDescriptor(DataType::FP32, {3}, 4, 13, DataType::UINT64);
    const RaggedExpression input = RaggedExpression::input("tokens", descriptor);
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP32);

    const RaggedExpression output = input.causalConv1d(filter, 5, 3, 2, DataType::FP32, DataType::FP32);

    EXPECT_EQ(output.getDescriptor().getRowPartition(), input.getDescriptor().getRowPartition());
    EXPECT_TRUE(output.getOffsets().isSameLogicalNode(input.getOffsets()));
    EXPECT_EQ(output.getTrailingDimensions(), (std::vector<uint64_t>{5}));
    EXPECT_EQ(output.getValuesDimensions(), (std::vector<uint64_t>{13, 5}));
    EXPECT_EQ(output.getRuntimeExtent().maxActiveValues, 13ULL);
    EXPECT_EQ(output.getRuntimeExtent().elementsPerValue, 5ULL);
    EXPECT_EQ(output.getMetadataInputNames(), (std::set<std::string>{"tokens.offsets"}));
    EXPECT_EQ(output.getDifferentiableInputNames(), (std::set<std::string>{"filter", "tokens.values"}));

    const MarkedValueNodes marked = markedValueNodes(output.getValues());
    const ExprNode& node = marked.values;
    ASSERT_EQ(node.op, ExprOp::RAGGED_CONV1D_CAUSAL);
    EXPECT_EQ(node.ragged_conv_spatial_1d.stride, 1);
    EXPECT_EQ(node.ragged_conv_spatial_1d.dilation, 2);
    EXPECT_EQ(node.ragged_conv_spatial_1d.pre_padding, 4);
    EXPECT_EQ(node.ragged_conv_spatial_1d.post_padding, 0);
    EXPECT_EQ(node.ragged_conv1d_input_channels, 3ULL);
    EXPECT_EQ(node.ragged_conv1d_output_channels, 5ULL);
    EXPECT_EQ(node.ragged_conv1d_kernel_width, 3ULL);
    EXPECT_EQ(node.ragged_runtime_batch_size, 4ULL);
    EXPECT_EQ(node.ragged_runtime_max_active_values, 13ULL);
    EXPECT_EQ(node.ragged_runtime_max_values_per_row, 13ULL);
    EXPECT_EQ(node.ragged_runtime_elements_per_value, 5ULL);
}

TEST(RaggedExpression, CausalConv1dRequiresPlacementTimeMaxValuesPerRow) {
    const RaggedTensorDescriptor descriptor(
        DataType::FP32, {3}, 4, 13, DataType::UINT32);
    ASSERT_FALSE(descriptor.hasMaxValuesPerRow());
    const RaggedExpression input = RaggedExpression::input("tokens", descriptor);
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP32);

    try {
        (void)input.causalConv1d(filter, 5, 3, 1, DataType::FP32, DataType::FP32);
        FAIL() << "Expected ragged Conv1D to require max_values_per_row for placement.";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("max_values_per_row"), std::string::npos);
    }
}

TEST(RaggedExpression, Conv1dT6ARejectsNonCausalOrStridedGeometry) {
    const RaggedExpression input = RaggedExpression::input("tokens", makeDescriptor(DataType::FP32, {3}, 2, 8));
    const Expression filter = Expression::input("filter");

    EXPECT_THROW((void)input.conv1d(filter, 4, 3, ConvolutionSpatial1d::valid()), std::invalid_argument);
    EXPECT_THROW((void)input.conv1d(filter, 4, 3, ConvolutionSpatial1d::sameUpper(8, 3)), std::invalid_argument);
    EXPECT_THROW((void)input.conv1d(filter, 4, 3, ConvolutionSpatial1d::causal(3, 2, 1)), std::invalid_argument);
    EXPECT_THROW((void)input.conv1d(filter, 0, 3, ConvolutionSpatial1d::causal(3)), std::invalid_argument);
    EXPECT_THROW((void)input.conv1d(filter, 4, 0, ConvolutionSpatial1d::valid()), std::invalid_argument);

    const RaggedExpression rank_two_trailing =
        RaggedExpression::input("matrix", makeDescriptor(DataType::FP32, {2, 3}, 2, 8));
    EXPECT_THROW((void)rank_two_trailing.causalConv1d(filter, 4, 3), std::invalid_argument);
}

TEST(RaggedExpression, CausalConv1dPhysicalSerializationRoundTripsLogicalMetadata) {
    const RaggedExpression input = RaggedExpression::input("tokens", makeDescriptor(DataType::FP16, {8}, 3, 11, DataType::UINT32));
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP16);
    const RaggedExpression output = input.causalConv1d(filter, 12, 5, 3, DataType::FP32, DataType::FP16, 4);

    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Expression::outputs({{"y", output.getValues()}}));
    const nlohmann::json payload = definition.architectureJson();

    bool found = false;
    for (const auto& node : payload.at("nodes")) {
        if (node.at("op").get<std::string>() != "ragged_conv1d_causal") {
            continue;
        }
        found = true;
        EXPECT_EQ(node.at("ragged_conv_stride").get<int32_t>(), 1);
        EXPECT_EQ(node.at("ragged_conv_dilation").get<int32_t>(), 3);
        EXPECT_EQ(node.at("ragged_conv_pre_padding").get<int32_t>(), 12);
        EXPECT_EQ(node.at("ragged_conv_post_padding").get<int32_t>(), 0);
        EXPECT_EQ(node.at("ragged_conv1d_input_channels").get<uint64_t>(), 8ULL);
        EXPECT_EQ(node.at("ragged_conv1d_output_channels").get<uint64_t>(), 12ULL);
        EXPECT_EQ(node.at("ragged_conv1d_kernel_width").get<uint64_t>(), 5ULL);
        EXPECT_EQ(node.at("ragged_conv1d_groups").get<uint64_t>(), 4ULL);
        EXPECT_EQ(node.at("ragged_runtime_batch_size").get<uint64_t>(), 3ULL);
        EXPECT_EQ(node.at("ragged_runtime_max_active_values").get<uint64_t>(), 11ULL);
        EXPECT_EQ(node.at("ragged_runtime_max_values_per_row").get<uint64_t>(), 11ULL);
        EXPECT_EQ(node.at("ragged_runtime_elements_per_value").get<uint64_t>(), 12ULL);
    }
    ASSERT_TRUE(found);

    const ExpressionDefinition loaded = ExpressionDefinition::deserialize(payload);
    EXPECT_EQ(loaded.architectureJson(), payload);
    EXPECT_EQ(loaded.canonical_hash, definition.canonical_hash);

    // Convolution serialization is intentionally current-schema-only. Do not
    // silently synthesize placement geometry from an older aggregate bound.
    nlohmann::json legacyPayload = payload;
    for (auto& node : legacyPayload.at("nodes")) {
        if (node.at("op").get<std::string>() == "ragged_conv1d_causal") {
            node.erase("ragged_runtime_max_values_per_row");
        }
    }
    try {
        (void)ExpressionDefinition::deserialize(legacyPayload);
        FAIL() << "Expected legacy ragged Conv1D serialization to be rejected.";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("legacy convolution serialization is not supported"), std::string::npos);
    }
}

TEST(RaggedExpression, CausalConv1dCanonicalIdentityIncludesDilationAndKernelGeometry) {
    // Four channels keeps the grouped=2 identity variant structurally valid;
    // this test is about canonical identity, not invalid-group validation.
    const RaggedExpression input = RaggedExpression::input("tokens", makeDescriptor(DataType::FP32, {4}, 2, 8));
    const Expression filter = Expression::input("filter");
    const RaggedExpression d1 = input.causalConv1d(filter, 4, 3, 1);
    const RaggedExpression d2 = input.causalConv1d(filter, 4, 3, 2);
    const RaggedExpression k5 = input.causalConv1d(filter, 4, 5, 1);
    const RaggedExpression grouped = input.causalConv1d(filter, 4, 3, 1, DataType::FP32, DataType::FP32, 2);

    const std::string c1 = canonicalize(d1.getValues().expression());
    const std::string c2 = canonicalize(d2.getValues().expression());
    const std::string c3 = canonicalize(k5.getValues().expression());
    const std::string c4 = canonicalize(grouped.getValues().expression());
    EXPECT_NE(c1, c2);
    EXPECT_NE(c1, c3);
    EXPECT_NE(c1, c4);
    EXPECT_NE(c1.find("RAGGED_CONV1D_CAUSAL"), std::string::npos);
    EXPECT_NE(c2.find("dilation=2"), std::string::npos);
    EXPECT_NE(c4.find("groups=2"), std::string::npos);
}

TEST(RaggedExpression, CausalConv1dRejectsNonDivisibleGroupedChannelGeometry) {
    const Expression filter = Expression::input("filter");

    const RaggedExpression invalid_input_channels =
        RaggedExpression::input("tokens3", makeDescriptor(DataType::FP32, {3}, 2, 8));
    EXPECT_THROW((void)invalid_input_channels.causalConv1d(
                     filter, 4, 3, 1, DataType::FP32, DataType::FP32, 2),
                 std::invalid_argument);

    const RaggedExpression invalid_output_channels =
        RaggedExpression::input("tokens4", makeDescriptor(DataType::FP32, {4}, 2, 8));
    EXPECT_THROW((void)invalid_output_channels.causalConv1d(
                     filter, 3, 3, 1, DataType::FP32, DataType::FP32, 2),
                 std::invalid_argument);
}

TEST(RaggedExpression, CausalConv1dT9AAutodiffBuildsOnlyInputGradientAndTreatsOffsetsAsStructural) {
    const RaggedExpression input =
        RaggedExpression::input("tokens", makeDescriptor(DataType::FP32, {3}, 2, 8, DataType::UINT32));
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression output = input.causalConv1d(filter, 4, 3, 2, DataType::FP32, DataType::FP32);
    const PhysicalOutputs forward = Expression::outputs({{"y", output.getValues()}}).physicalOutputs();

    const std::unordered_map<std::string, std::vector<uint64_t>> dims{
        {"tokens.values", {8, 3}},
        {"tokens.offsets", {3}},
        {"filter", {4, 3, 3}},
    };

    PhysicalOutputs backward = buildBackwardOutputs(forward, {"tokens.values"}, std::nullopt, dims);
    resolveRaggedBackwardTestDTypes(backward, DataType::UINT32);
    EXPECT_TRUE(containsOp(backward, ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_DATA));
    EXPECT_FALSE(containsOp(backward, ExprOp::CONV2D_BACKWARD_FILTER));
    EXPECT_FALSE(containsOp(backward, ExprOp::CONV3D_BACKWARD_FILTER));

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(backward);
    EXPECT_TRUE(std::any_of(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausalBackwardData;
    }));

    EXPECT_THROW((void)buildBackwardOutputs(forward, {"tokens.offsets"}, std::nullopt, dims), std::runtime_error);
}

TEST(RaggedExpression, CausalConv1dT9BAutodiffBuildsWeightGradientAndSupportsMixedDgradWgrad) {
    const RaggedExpression input =
        RaggedExpression::input("tokens", makeDescriptor(DataType::FP32, {4}, 3, 13, DataType::UINT32));
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression output = input.causalConv1d(filter, 6, 3, 2, DataType::FP32, DataType::FP32, 2);
    const PhysicalOutputs forward = Expression::outputs({{"y", output.getValues()}}).physicalOutputs();
    const std::unordered_map<std::string, std::vector<uint64_t>> dims{
        {"tokens.values", {13, 4}},
        {"tokens.offsets", {4}},
        {"filter", {6, 2, 3}},
    };

    PhysicalOutputs wgrad_only = buildBackwardOutputs(forward, {"filter"}, std::nullopt, dims);
    resolveRaggedBackwardTestDTypes(wgrad_only, DataType::UINT32);
    EXPECT_TRUE(containsOp(wgrad_only, ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_FILTER));
    EXPECT_FALSE(containsOp(wgrad_only, ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_DATA));
    const std::vector<PhysicalExecutionStage> wgrad_stages = EquationCompiler::splitAtReductionBoundaries(wgrad_only);
    EXPECT_TRUE(std::any_of(wgrad_stages.begin(), wgrad_stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausalBackwardFilter;
    }));

    PhysicalOutputs mixed = buildBackwardOutputs(forward, {"tokens.values", "filter"}, std::nullopt, dims);
    resolveRaggedBackwardTestDTypes(mixed, DataType::UINT32);
    EXPECT_TRUE(containsOp(mixed, ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_DATA));
    EXPECT_TRUE(containsOp(mixed, ExprOp::RAGGED_CONV1D_CAUSAL_BACKWARD_FILTER));
}

TEST(RaggedExpression, CausalConv1dT9ACompilerCarriesPaddedDgradLayouts) {
    const RaggedExpression input =
        RaggedExpression::input("tokens", makeDescriptor(DataType::FP16, {4}, 3, 13, DataType::UINT64));
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP16);
    const RaggedExpression output =
        input.causalConv1d(filter, 8, 3, 2, DataType::FP32, DataType::FP16, 2);
    const PhysicalOutputs forward = Expression::outputs({{"y", output.getValues()}}).physicalOutputs();
    const std::unordered_map<std::string, std::vector<uint64_t>> dims{
        {"tokens.values", {13, 4}},
        {"tokens.offsets", {4}},
        {"filter", {8, 2, 3}},
    };
    PhysicalOutputs backward = buildBackwardOutputs(forward, {"tokens.values"}, std::nullopt, dims);
    resolveRaggedBackwardTestDTypes(backward, DataType::UINT64);

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(backward);
    const auto stage_it = std::find_if(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausalBackwardData;
    });
    ASSERT_NE(stage_it, stages.end());
    const auto compiled = EquationCompiler::compileRaggedConv1dCausalBackwardData(stage_it->expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->filter_dtype, DataType::FP16);
    EXPECT_EQ(compiled->grad_output_dtype, DataType::FP16);
    EXPECT_EQ(compiled->output_dtype, DataType::FP16);
    EXPECT_EQ(compiled->compute_dtype, DataType::FP32);
    EXPECT_EQ(compiled->offset_dtype, DataType::UINT64);
    EXPECT_EQ(compiled->groups, 2u);
    EXPECT_EQ(compiled->padded_grad_output_layout.channels, 8u);
    EXPECT_EQ(compiled->padded_output_layout.channels, 4u);
    EXPECT_EQ(compiled->padded_grad_output_layout.max_values_per_row, 13u);
    EXPECT_EQ(compiled->padded_output_layout.max_values_per_row, 13u);
}

TEST(RaggedExpression, CausalConv1dT9BCompilerCarriesTwoPaddedInputsAndDenseWeightGradient) {
    const RaggedExpression input =
        RaggedExpression::input("tokens", makeDescriptor(DataType::BF16, {4}, 3, 13, DataType::UINT64));
    const Expression filter = Expression::input("filter", std::nullopt, DataType::BF16);
    const RaggedExpression output =
        input.causalConv1d(filter, 8, 3, 2, DataType::FP32, DataType::BF16, 2);
    const PhysicalOutputs forward = Expression::outputs({{"y", output.getValues()}}).physicalOutputs();
    const std::unordered_map<std::string, std::vector<uint64_t>> dims{
        {"tokens.values", {13, 4}},
        {"tokens.offsets", {4}},
        {"filter", {8, 2, 3}},
    };
    PhysicalOutputs backward = buildBackwardOutputs(forward, {"filter"}, std::nullopt, dims);
    resolveRaggedBackwardTestDTypes(backward, DataType::UINT64);

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(backward);
    const auto stage_it = std::find_if(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausalBackwardFilter;
    });
    ASSERT_NE(stage_it, stages.end());
    const auto compiled = EquationCompiler::compileRaggedConv1dCausalBackwardFilter(stage_it->expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->input_dtype, DataType::BF16);
    EXPECT_EQ(compiled->grad_output_dtype, DataType::BF16);
    EXPECT_EQ(compiled->output_dtype, DataType::BF16);
    EXPECT_EQ(compiled->compute_dtype, DataType::FP32);
    EXPECT_EQ(compiled->offset_dtype, DataType::UINT64);
    EXPECT_EQ(compiled->groups, 2u);
    EXPECT_EQ(compiled->padded_input_layout.channels, 4u);
    EXPECT_EQ(compiled->padded_grad_output_layout.channels, 8u);
    EXPECT_EQ(compiled->padded_input_layout.max_values_per_row, 13u);
    EXPECT_EQ(compiled->padded_grad_output_layout.max_values_per_row, 13u);
}

TEST(RaggedExpression, CausalConv1dCompilerLowersToDedicatedRaggedConvStage) {
    const RaggedExpression input = RaggedExpression::input("tokens", makeDescriptor(DataType::FP32, {3}, 2, 8));
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression output = input.causalConv1d(filter, 4, 3, 1, DataType::FP32, DataType::FP32);
    const PhysicalOutputs physical = Expression::outputs({{"y", output.getValues()}}).physicalOutputs();

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(physical);
    ASSERT_GE(stages.size(), 1U);
    EXPECT_TRUE(std::any_of(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausal;
    }));
}

TEST(RaggedExpression, CausalConv1dCompilerCarriesT7R2PaddedInputAndOutputLayouts) {
    const RaggedExpression input =
        RaggedExpression::input("tokens", makeDescriptor(DataType::FP16, {3}, 4, 13, DataType::UINT64));
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP16);
    const RaggedExpression output =
        input.causalConv1d(filter, 5, 3, 2, DataType::FP32, DataType::FP16);
    const PhysicalOutputs physical = Expression::outputs({{"y", output.getValues()}}).physicalOutputs();

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(physical);
    const auto stage_it = std::find_if(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausal;
    });
    ASSERT_NE(stage_it, stages.end());

    const std::shared_ptr<CompiledRaggedConv1dCausal> compiled =
        EquationCompiler::compileRaggedConv1dCausal(stage_it->expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->padded_input_layout.values_dtype, DataType::FP16);
    EXPECT_EQ(compiled->padded_input_layout.offset_dtype, DataType::UINT64);
    EXPECT_EQ(compiled->padded_input_layout.batch_size, 4u);
    EXPECT_EQ(compiled->padded_input_layout.max_total_values, 13u);
    EXPECT_EQ(compiled->padded_input_layout.max_values_per_row, 13u);
    EXPECT_EQ(compiled->max_values_per_row, 13u);
    EXPECT_EQ(compiled->padded_input_layout.channels, 3u);
    EXPECT_EQ(compiled->padded_output_layout.values_dtype, DataType::FP16);
    EXPECT_EQ(compiled->padded_output_layout.offset_dtype, DataType::UINT64);
    EXPECT_EQ(compiled->padded_output_layout.batch_size, 4u);
    EXPECT_EQ(compiled->padded_output_layout.max_total_values, 13u);
    EXPECT_EQ(compiled->padded_output_layout.max_values_per_row, 13u);
    EXPECT_EQ(compiled->padded_output_layout.channels, 5u);
}

TEST(RaggedExpression, CausalConv1dCompilerSignatureIncludesAllOperandsAndGeometry) {
    const RaggedTensorDescriptor descriptor = makeDescriptor(DataType::FP32, {4}, 2, 8, DataType::UINT32);
    const Expression values = Expression::input("values", std::nullopt, DataType::FP32);
    const Expression offsets_a = Expression::input("offsets_a", std::nullopt, DataType::UINT32);
    const Expression offsets_b = Expression::input("offsets_b", std::nullopt, DataType::UINT32);
    const Expression filter_a = Expression::input("filter_a", std::nullopt, DataType::FP32);
    const Expression filter_b = Expression::input("filter_b", std::nullopt, DataType::FP32);
    const RaggedExpression input_a(values, offsets_a, descriptor);
    const RaggedExpression input_b(values, offsets_b, descriptor);

    const RaggedExpression base = input_a.causalConv1d(filter_a, 4, 3, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression different_dilation = input_a.causalConv1d(filter_a, 4, 3, 2, DataType::FP32, DataType::FP32);
    const RaggedExpression different_filter = input_a.causalConv1d(filter_b, 4, 3, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression different_offsets = input_b.causalConv1d(filter_a, 4, 3, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression different_groups = input_a.causalConv1d(filter_a, 4, 3, 1, DataType::FP32, DataType::FP32, 2);
    const PhysicalOutputs physical = Expression::outputs({{"base", base.getValues()},
                                                          {"dilation", different_dilation.getValues()},
                                                          {"filter", different_filter.getValues()},
                                                          {"offsets", different_offsets.getValues()},
                                                          {"groups", different_groups.getValues()}})
                                         .physicalOutputs();

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(physical);
    const size_t ragged_conv_stages = std::count_if(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausal;
    });
    EXPECT_EQ(ragged_conv_stages, 5U);
}

TEST(RaggedExpression, CausalConv1dCompilerPreservesRequestedComputeDtypeForCudnnBackend) {
    const RaggedExpression input = RaggedExpression::input("tokens", makeDescriptor(DataType::FP16, {3}, 2, 8));
    const Expression filter = Expression::input("filter", std::nullopt, DataType::FP16);
    const RaggedExpression output = input.causalConv1d(filter, 4, 3, 1, DataType::FP16, DataType::FP16);
    const PhysicalOutputs physical = Expression::outputs({{"y", output.getValues()}}).physicalOutputs();

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(physical);
    const auto stage_it = std::find_if(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausal;
    });
    ASSERT_NE(stage_it, stages.end());

    const std::shared_ptr<CompiledRaggedConv1dCausal> compiled =
        EquationCompiler::compileRaggedConv1dCausal(stage_it->expr);
    ASSERT_NE(compiled, nullptr);
    EXPECT_EQ(compiled->input_dtype, DataType::FP16);
    EXPECT_EQ(compiled->filter_dtype, DataType::FP16);
    EXPECT_EQ(compiled->output_dtype, DataType::FP16);
    EXPECT_EQ(compiled->compute_dtype, DataType::FP16);
}

TEST(RaggedExpression, CausalConv1dT7R6RejectsUnsupportedCudnnValueDtypeWithoutFallback) {
    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 2;
    constexpr uint64_t kernel_width = 3;

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP64, {input_channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP64);
    const RaggedExpression result =
        input.causalConv1d(filter_expr, output_channels, kernel_width, 1, DataType::FP64, DataType::FP64);
    const PhysicalOutputs physical = Expression::outputs({{"y", result.getValues()}}).physicalOutputs();
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(physical);
    const auto stage_it = std::find_if(stages.begin(), stages.end(), [](const PhysicalExecutionStage& stage) {
        return stage.kind == PhysicalExecutionStage::Kind::RaggedConv1dCausal;
    });
    ASSERT_NE(stage_it, stages.end());

    try {
        (void)EquationCompiler::compileRaggedConv1dCausal(stage_it->expr);
        FAIL() << "Expected ragged Conv1D to reject a value dtype unsupported by its cuDNN backend.";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("no alternate convolution backend"), std::string::npos);
    }
}

TEST(RaggedExpression, CausalConv1dForwardRespectsRowsAndPoisonedCapacity) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 4;
    constexpr uint64_t max_total_values = 11;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 2;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t dilation = 1;
    constexpr float inactive_sentinel = 777.0F;

    const std::vector<uint32_t> offsets32{0, 3, 3, 7, 9};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    // The last value of rows 0 and 2 is deliberately enormous. A kernel that
    // treats packed adjacency as temporal adjacency will contaminate the first
    // output of the next non-empty row. The final two capacity values are also
    // poisoned and must never be consumed or written.
    const std::vector<float> values{
        1.0F, 2.0F,
        3.0F, 4.0F,
        -5000.0F, 6000.0F,
        1.5F, -2.0F,
        2.0F, -3.0F,
        4.0F, 5.0F,
        7000.0F, -8000.0F,
        -1.0F, 0.5F,
        8.0F, 9.0F,
        12345.0F, -12345.0F,
        22222.0F, -22222.0F,
    };
    const std::vector<float> filter{
        0.25F, -0.5F, 1.0F,
        2.0F, 0.5F, -1.0F,
        -0.75F, 0.25F, 0.5F,
        1.25F, -0.5F, 0.75F,
    };

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, input_channels}, values, stream);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels, kernel_width}, filter, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(gpu_offsets,
                                  RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostActiveValueCount(offsets.back());
    partition.setHostMaxActiveRowLength(4);
    Tensor output_storage = makeGpuTensor<float>(
        {max_total_values, output_channels},
        std::vector<float>(max_total_values * output_channels, inactive_sentinel),
        stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression result =
        input.causalConv1d(filter_expr, output_channels, kernel_width, dilation, DataType::FP32, DataType::FP32);

    const Tensor actual_tensor = runExpressionOutput(result.getValues(),
                                                     {{"tokens.values", gpu_values},
                                                      {"tokens.offsets", gpu_offsets},
                                                      {"filter", gpu_filter}},
                                                     "y",
                                                     stream,
                                                     output_storage);
    const std::vector<float> actual = copyToCpuValues(actual_tensor, stream);
    const std::vector<float> expected = cpuRaggedCausalConv1d(values,
                                                              offsets,
                                                              filter,
                                                              max_total_values,
                                                              input_channels,
                                                              output_channels,
                                                              kernel_width,
                                                              dilation,
                                                              inactive_sentinel);
    expectNear(actual, expected, 1.0e-5F);
}

TEST(RaggedExpression, CausalConv1dForwardSupportsDilationAndUint64Offsets) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t input_channels = 1;
    constexpr uint64_t output_channels = 1;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t dilation = 2;
    constexpr float inactive_sentinel = -333.0F;

    const std::vector<uint64_t> offsets{0, 5, 7, 10};
    const std::vector<float> values{
        1.0F, 2.0F, 3.0F, 4.0F, 5000.0F,
        1.5F, -6000.0F,
        2.0F, 3.0F, 4.0F,
        12345.0F, -12345.0F,
    };
    const std::vector<float> filter{2.0F, -1.0F, 0.5F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, input_channels}, values, stream);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels, kernel_width}, filter, stream);
    Tensor gpu_offsets = makeGpuTensor<uint64_t>({batch_size + 1}, offsets, stream);
    RowPartitionRuntime partition(gpu_offsets,
                                  RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT64, max_total_values));
    partition.setHostActiveValueCount(offsets.back());
    partition.setHostMaxActiveRowLength(5);
    Tensor output_storage = makeGpuTensor<float>(
        {max_total_values, output_channels},
        std::vector<float>(max_total_values * output_channels, inactive_sentinel),
        stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT64));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression result =
        input.causalConv1d(filter_expr, output_channels, kernel_width, dilation, DataType::FP32, DataType::FP32);

    const Tensor actual_tensor = runExpressionOutput(result.getValues(),
                                                     {{"tokens.values", gpu_values},
                                                      {"tokens.offsets", gpu_offsets},
                                                      {"filter", gpu_filter}},
                                                     "y",
                                                     stream,
                                                     output_storage);
    const std::vector<float> actual = copyToCpuValues(actual_tensor, stream);
    const std::vector<float> expected = cpuRaggedCausalConv1d(values,
                                                              offsets,
                                                              filter,
                                                              max_total_values,
                                                              input_channels,
                                                              output_channels,
                                                              kernel_width,
                                                              dilation,
                                                              inactive_sentinel);
    expectNear(actual, expected, 1.0e-5F);
}

TEST(RaggedExpression, CausalConv1dForwardSupportsFp16Bf16AndFp32) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 6;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 2;
    constexpr uint64_t kernel_width = 2;
    constexpr uint64_t dilation = 1;

    const std::vector<uint32_t> offsets32{0, 3, 5};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> values{
        1.0F, -2.0F,
        3.0F, 0.5F,
        4.0F, -5.0F,
        2.0F, 1.0F,
        -1.5F, 3.0F,
        12.0F, -12.0F,
    };
    const std::vector<float> filter{
        0.5F, 1.0F,
        -0.25F, 0.75F,
        1.25F, -0.5F,
        0.5F, 0.25F,
    };
    const std::vector<float> expected = cpuRaggedCausalConv1d(values,
                                                              offsets,
                                                              filter,
                                                              max_total_values,
                                                              input_channels,
                                                              output_channels,
                                                              kernel_width,
                                                              dilation,
                                                              0.0F);

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(gpu_offsets,
                                  RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostActiveValueCount(offsets.back());
    partition.setHostMaxActiveRowLength(3);
    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        SCOPED_TRACE(TensorDescriptor::getElementTypeName(dtype));
        Tensor gpu_values = makeGpuTensorFromFloats({max_total_values, input_channels}, values, dtype, stream);
        Tensor gpu_filter = makeGpuTensorFromFloats({output_channels, input_channels, kernel_width}, filter, dtype, stream);

        const RaggedExpression input =
            RaggedExpression::input("tokens", makeDescriptor(dtype, {input_channels}, batch_size, max_total_values, DataType::UINT32));
        const Expression filter_expr = Expression::input("filter", std::nullopt, dtype);
        const RaggedExpression result =
            input.causalConv1d(filter_expr, output_channels, kernel_width, dilation, DataType::FP32, dtype);

        const Tensor actual_tensor = runExpressionOutput(result.getValues(),
                                                         {{"tokens.values", gpu_values},
                                                          {"tokens.offsets", gpu_offsets},
                                                          {"filter", gpu_filter}},
                                                         "y",
                                                         stream);
        const std::vector<float> actual = copyToCpuFloatValues(actual_tensor, stream);
        const size_t active_elements = static_cast<size_t>(offsets.back() * output_channels);
        ASSERT_GE(actual.size(), active_elements);
        for (size_t i = 0; i < active_elements; ++i) {
            EXPECT_NEAR(actual[i], expected[i], dtype == DataType::FP32 ? 1.0e-5F : 3.0e-2F) << "index " << i;
        }
    }
}

TEST(RaggedExpression, CausalConv1dRuntimeFailsLoudlyWhenProducerOmitsMaxActiveRowLength) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 6;
    constexpr uint64_t input_channels = 1;
    constexpr uint64_t output_channels = 1;
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F, 99.0F, 99.0F};
    const std::vector<uint32_t> offsets{0, 2, 4};
    const std::vector<float> filter{0.5F, 1.0F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, input_channels}, values, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets, stream);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels, 2}, filter, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression result =
        input.causalConv1d(filter_expr, output_channels, 2, 1, DataType::FP32, DataType::FP32);
    FusedEquation equation =
        FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter", gpu_filter}},
                                               stream);

    try {
        plan.run();
        FAIL() << "Expected ragged Conv1D to reject missing max_active_row_length metadata.";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("max_active_row_length"), std::string::npos);
        EXPECT_NE(std::string(error.what()).find("implicit device-to-host"), std::string::npos);
    }
}

TEST(RaggedExpression, CausalConv1dT7R2PaddedCudnnMatchesIndependentDenseRows) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 5;
    constexpr uint64_t max_total_values = 24;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 3;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t dilation = 2;
    constexpr float inactive_sentinel = -9191.0F;

    const std::vector<uint32_t> offsets32{0, 3, 3, 8, 10, 19};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    std::vector<float> values(max_total_values * input_channels, 77777.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < input_channels; ++channel) {
            values[value * input_channels + channel] =
                static_cast<float>((value + 1) * (channel == 0 ? 0.25 : -0.375));
        }
    }
    // Make row ends extremely distinctive. Packed-adjacency leakage would be obvious.
    values[(offsets[1] - 1) * input_channels + 0] = 5000.0F;
    values[(offsets[3] - 1) * input_channels + 1] = -7000.0F;
    values[(offsets[4] - 1) * input_channels + 0] = 9000.0F;

    const std::vector<float> filter{
        0.25F, -0.5F, 1.0F,
        -0.75F, 0.5F, 0.125F,
        1.25F, -0.25F, 0.75F,
        0.5F, 0.25F, -1.0F,
        -0.375F, 0.625F, 0.5F,
        0.875F, -0.125F, 0.25F,
    };

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, input_channels}, values, stream);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels, kernel_width}, filter, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression result =
        input.causalConv1d(filter_expr, output_channels, kernel_width, dilation, DataType::FP32, DataType::FP32);

    // T7R2 requires only scalar dispatch metadata; no full host offsets mirror is needed.
    Tensor scalar_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime scalar_partition(
        scalar_offsets, RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    scalar_partition.setHostActiveValueCount(offsets.back());
    scalar_partition.setHostMaxActiveRowLength(9);
    Tensor scalar_output = makeGpuTensor<float>(
        {max_total_values, output_channels},
        std::vector<float>(max_total_values * output_channels, inactive_sentinel),
        stream);
    FusedEquation scalar_equation =
        FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan scalar_plan = scalar_equation.stamp({{"tokens.values", gpu_values},
                                                              {"tokens.offsets", scalar_offsets},
                                                              {"filter", gpu_filter}},
                                                             stream,
                                                             {},
                                                             {{"y", scalar_output}});
    scalar_plan.run();
    stream.synchronize();
    const std::vector<float> scalar_metadata_result = copyToCpuValues(scalar_plan.output("y"), stream);
    const std::vector<RaggedConv1dStageDiagnostic> scalar_diagnostics =
        scalar_plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(scalar_diagnostics.size(), 1u);

    // Publishing complete host offsets derives the same scalar metadata and must select the same padded representation.
    Tensor padded_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime(padded_offsets,
                        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values))
        .setHostOffsets(offsets);
    Tensor padded_output = makeGpuTensor<float>(
        {max_total_values, output_channels},
        std::vector<float>(max_total_values * output_channels, inactive_sentinel),
        stream);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values},
                                                {"tokens.offsets", padded_offsets},
                                                {"filter", gpu_filter}},
                                               stream,
                                               {},
                                               {{"y", padded_output}});
    plan.run();
    stream.synchronize();

    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
    const std::vector<float> cpu_expected = cpuRaggedCausalConv1d(values,
                                                                  offsets,
                                                                  filter,
                                                                  max_total_values,
                                                                  input_channels,
                                                                  output_channels,
                                                                  kernel_width,
                                                                  dilation,
                                                                  inactive_sentinel);
    const std::vector<float> dense_rows = denseRowByRowCausalConv1d(values,
                                                                    offsets,
                                                                    gpu_filter,
                                                                    max_total_values,
                                                                    input_channels,
                                                                    output_channels,
                                                                    kernel_width,
                                                                    dilation,
                                                                    inactive_sentinel,
                                                                    stream);
    expectNear(actual, scalar_metadata_result, 1.0e-5F);
    expectNear(actual, cpu_expected, 1.0e-5F);
    expectNear(actual, dense_rows, 1.0e-5F);

    const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(diagnostics.size(), 1u);
    const RaggedConv1dStageDiagnostic& diagnostic = diagnostics.front();
    EXPECT_EQ(diagnostic.active_values, offsets.back());
    EXPECT_EQ(diagnostic.selected_width_capacity, 16u);
    EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u);
    const uint64_t element_bytes = sizeof(float);
    EXPECT_EQ(diagnostic.input_padded_value_bytes,
              batch_size * input_channels * diagnostic.selected_width_capacity * element_bytes);
    EXPECT_EQ(diagnostic.output_padded_value_bytes,
              batch_size * output_channels * diagnostic.selected_width_capacity * element_bytes);
    EXPECT_EQ(diagnostic.allocated_padded_value_bytes,
              batch_size * max_total_values * (input_channels + output_channels) * element_bytes);

    const uint64_t active_elements = offsets.back() * output_channels;
    for (uint64_t i = active_elements; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i], inactive_sentinel) << "packed spare output element " << i;
    }
}

TEST(RaggedExpression, CausalConv1dT7R2PaddedCudnnSupportsFp16Bf16AndFp32) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 9;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 2;
    constexpr uint64_t kernel_width = 2;
    constexpr uint64_t dilation = 1;
    const std::vector<uint32_t> offsets32{0, 3, 5, 8};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> values{
        1.0F, -2.0F, 3.0F, 0.5F, 4.0F, -5.0F,
        2.0F, 1.0F, -1.5F, 3.0F,
        0.25F, -0.75F, 1.25F, 2.0F, -3.0F, 0.125F,
        12345.0F, -12345.0F,
    };
    const std::vector<float> filter{
        0.5F, 1.0F,
        -0.25F, 0.75F,
        1.25F, -0.5F,
        0.5F, 0.25F,
    };
    const std::vector<float> expected = cpuRaggedCausalConv1d(values,
                                                              offsets,
                                                              filter,
                                                              max_total_values,
                                                              input_channels,
                                                              output_channels,
                                                              kernel_width,
                                                              dilation,
                                                              0.0F);

    Stream stream(0);
    for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP32}) {
        SCOPED_TRACE(TensorDescriptor::getElementTypeName(dtype));
        Tensor gpu_values = makeGpuTensorFromFloats({max_total_values, input_channels}, values, dtype, stream);
        Tensor gpu_filter = makeGpuTensorFromFloats({output_channels, input_channels, kernel_width}, filter, dtype, stream);
        Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
        RowPartitionRuntime(gpu_offsets,
                            RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32))
            .setHostOffsets(offsets);

        const RaggedExpression input = RaggedExpression::input(
            "tokens", makeDescriptor(dtype, {input_channels}, batch_size, max_total_values, DataType::UINT32));
        const Expression filter_expr = Expression::input("filter", std::nullopt, dtype);
        const RaggedExpression result =
            input.causalConv1d(filter_expr, output_channels, kernel_width, dilation, DataType::FP32, dtype);

        FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
        StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values},
                                                    {"tokens.offsets", gpu_offsets},
                                                    {"filter", gpu_filter}},
                                                   stream);
        plan.run();
        stream.synchronize();
        const std::vector<float> actual = copyToCpuFloatValues(plan.output("y"), stream);
        const size_t active_elements = static_cast<size_t>(offsets.back() * output_channels);
        for (size_t i = 0; i < active_elements; ++i) {
            EXPECT_NEAR(actual[i], expected[i], dtype == DataType::FP32 ? 1.0e-5F : 3.0e-2F) << "index " << i;
        }
        const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
        ASSERT_EQ(diagnostics.size(), 1u);
        EXPECT_EQ(diagnostics.front().explicit_unfold_workspace_bytes, 0u);
    }
}

TEST(RaggedExpression, CausalConv1dT7CGroupedAndDepthwiseMatchCpuAndDenseRows) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 4;
    constexpr uint64_t max_total_values = 18;
    constexpr uint64_t input_channels = 4;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t dilation = 2;
    constexpr float inactive_sentinel = -54321.0F;
    const std::vector<uint32_t> offsets32{0, 5, 5, 11, 15};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> values(max_total_values * input_channels, 7777.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < input_channels; ++channel) {
            values[value * input_channels + channel] =
                static_cast<float>(static_cast<int>((value * 7 + channel * 3) % 19) - 9) * 0.125F;
        }
    }
    // Distinctive row-end values make any cross-row read obvious.
    values[(offsets[1] - 1) * input_channels + 0] = 6000.0F;
    values[(offsets[3] - 1) * input_channels + 2] = -8000.0F;
    values[(offsets[4] - 1) * input_channels + 3] = 10000.0F;

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, input_channels}, values, stream);

    auto run_case = [&](uint64_t output_channels, uint64_t groups, const char* label) {
        SCOPED_TRACE(label);
        const uint64_t input_channels_per_group = input_channels / groups;
        std::vector<float> filter(output_channels * input_channels_per_group * kernel_width);
        for (size_t i = 0; i < filter.size(); ++i) {
            filter[i] = static_cast<float>(static_cast<int>((i * 5) % 17) - 8) * 0.075F;
        }
        Tensor gpu_filter =
            makeGpuTensor<float>({output_channels, input_channels_per_group, kernel_width}, filter, stream);
        Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
        RowPartitionRuntime(gpu_offsets,
                            RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values))
            .setHostOffsets(offsets);

        const RaggedExpression input = RaggedExpression::input(
            "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT32));
        const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
        const RaggedExpression result = input.causalConv1d(filter_expr,
                                                           output_channels,
                                                           kernel_width,
                                                           dilation,
                                                           DataType::FP32,
                                                           DataType::FP32,
                                                           groups);
        Tensor packed_output = makeGpuTensor<float>(
            {max_total_values, output_channels},
            std::vector<float>(max_total_values * output_channels, inactive_sentinel),
            stream);
        FusedEquation equation =
            FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
        StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values},
                                                    {"tokens.offsets", gpu_offsets},
                                                    {"filter", gpu_filter}},
                                                   stream,
                                                   {},
                                                   {{"y", packed_output}});
        const uint64_t expected_flops =
            2 * offsets.back() * output_channels * input_channels_per_group * kernel_width;
        EXPECT_EQ(plan.flopCount(), expected_flops);
        plan.run();
        stream.synchronize();
        const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
        const std::vector<float> cpu_expected = cpuRaggedCausalConv1d(values,
                                                                      offsets,
                                                                      filter,
                                                                      max_total_values,
                                                                      input_channels,
                                                                      output_channels,
                                                                      kernel_width,
                                                                      dilation,
                                                                      inactive_sentinel,
                                                                      groups);
        const std::vector<float> dense_expected = denseRowByRowCausalConv1d(values,
                                                                            offsets,
                                                                            gpu_filter,
                                                                            max_total_values,
                                                                            input_channels,
                                                                            output_channels,
                                                                            kernel_width,
                                                                            dilation,
                                                                            inactive_sentinel,
                                                                            stream,
                                                                            groups);
        // The distinctive row-end sentinels deliberately drive some valid outputs
        // into the thousands. Grouped cuDNN execution may choose different legal
        // FP32 convolution engines across placements/runs; those engines can differ
        // slightly in multiply/accumulation precision and ordering from both the
        // scalar CPU reference and the row-by-row dense reference. Keep a small
        // absolute allowance near zero (the observed engine-to-reference drift is
        // O(1e-5)) and a few FP32 ulps of relative error for the deliberately large
        // sentinel-driven values. This is still far tighter than the error required
        // to hide a row-boundary/grouping bug.
        constexpr float atol = 1.0e-4F;
        constexpr float rtol = 4.0F * std::numeric_limits<float>::epsilon();
        expectNearRelative(actual, cpu_expected, atol, rtol);
        expectNearRelative(actual, dense_expected, atol, rtol);

        const uint64_t active_elements = offsets.back() * output_channels;
        for (uint64_t i = active_elements; i < actual.size(); ++i) {
            EXPECT_EQ(actual[i], inactive_sentinel) << "packed spare output element " << i;
        }
        const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
        ASSERT_EQ(diagnostics.size(), 1u);
        EXPECT_EQ(diagnostics.front().explicit_unfold_workspace_bytes, 0u);
    };

    run_case(6, 2, "groups=2");
    run_case(4, 4, "depthwise");
}

TEST(RaggedExpression, CausalConv1dT8ARetainsOnePaddedRepresentationAcrossCompatibleRegion) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t hidden_channels = 3;
    constexpr uint64_t output_channels = 2;
    constexpr uint64_t kernel_width = 3;
    const std::vector<uint32_t> offsets32{0, 4, 4, 9};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> values(max_total_values * input_channels, 7777.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < input_channels; ++channel) {
            values[value * input_channels + channel] =
                static_cast<float>(1 + value * input_channels + channel) * 0.125F;
        }
    }
    std::vector<float> filter1(hidden_channels * input_channels * kernel_width);
    for (size_t i = 0; i < filter1.size(); ++i) {
        filter1[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125F;
    }
    std::vector<float> filter2(output_channels * hidden_channels * kernel_width);
    for (size_t i = 0; i < filter2.size(); ++i) {
        filter2[i] = static_cast<float>(static_cast<int>((i * 3) % 11) - 5) * 0.0625F;
    }

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, input_channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({hidden_channels, input_channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({output_channels, hidden_channels, kernel_width}, filter2, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, hidden_channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression result =
        hidden.causalConv1d(filter2_expr, output_channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> named_inputs{{"tokens.values", gpu_values},
                                                               {"tokens.offsets", gpu_offsets},
                                                               {"filter1", gpu_filter1},
                                                               {"filter2", gpu_filter2}};
    const std::shared_ptr<CompiledOutputs> compiled = equation.compileForInputs(named_inputs);
    ASSERT_NE(compiled, nullptr);
    ASSERT_GE(compiled->padded_ragged_values.size(), 2u);
    ASSERT_EQ(compiled->final_outputs.size(), 1u);
    std::vector<const CompiledExecutionStage*> ragged_stages;
    for (const CompiledExecutionStage& stage : compiled->stages) {
        if (stage.kind == CompiledExecutionStage::Kind::RaggedConv1dCausal) {
            ragged_stages.push_back(&stage);
        }
    }
    ASSERT_EQ(ragged_stages.size(), 2u);
    ASSERT_EQ(ragged_stages[0]->outputs.size(), 1u);
    ASSERT_EQ(ragged_stages[1]->outputs.size(), 1u);
    EXPECT_TRUE(compiled->padded_ragged_values.contains(
        paddedRaggedRepresentationKey(compiled->final_outputs[0].value_id, ragged_stages[1]->input_value_ids[2])));
    EXPECT_EQ(ragged_stages[1]->input_value_ids[0], ragged_stages[0]->outputs[0].value_id);
    const auto entry_representation_it = compiled->padded_ragged_values.find(
        paddedRaggedRepresentationKey(ragged_stages[0]->input_value_ids[0], ragged_stages[0]->input_value_ids[2]));
    ASSERT_NE(entry_representation_it, compiled->padded_ragged_values.end());
    EXPECT_EQ(entry_representation_it->second.offsets_value_id, ragged_stages[0]->input_value_ids[2]);
    EXPECT_EQ(entry_representation_it->second.layout, ragged_stages[0]->ragged_conv1d_causal->padded_input_layout);
    for (const CompiledExecutionStage* stage : ragged_stages) {
        const auto representation_it = compiled->padded_ragged_values.find(
            paddedRaggedRepresentationKey(stage->outputs[0].value_id, stage->input_value_ids[2]));
        ASSERT_NE(representation_it, compiled->padded_ragged_values.end());
        EXPECT_EQ(representation_it->second.offsets_value_id, stage->input_value_ids[2]);
        EXPECT_FALSE(representation_it->second.width_capacities.empty());
        EXPECT_GE(representation_it->second.width_capacities.back(), max_total_values);
    }

    constexpr float inactive_sentinel = 6543.25F;
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, output_channels},
        std::vector<float>(max_total_values * output_channels, inactive_sentinel),
        stream);
    StampedExecutionPlan plan = equation.stamp(named_inputs, stream, {}, {{"y", packed_output}});
    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}));

    plan.run();
    stream.synchronize();
    const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(diagnostics.size(), 2u);
    EXPECT_EQ(diagnostics[0].selected_width_capacity, diagnostics[1].selected_width_capacity);
    EXPECT_EQ(diagnostics[0].active_values, diagnostics[1].active_values);
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
    const std::vector<float> expected_hidden = cpuRaggedCausalConv1d(values,
                                                                     offsets,
                                                                     filter1,
                                                                     max_total_values,
                                                                     input_channels,
                                                                     hidden_channels,
                                                                     kernel_width,
                                                                     1,
                                                                     0.0F);
    const std::vector<float> expected = cpuRaggedCausalConv1d(expected_hidden,
                                                              offsets,
                                                              filter2,
                                                              max_total_values,
                                                              hidden_channels,
                                                              output_channels,
                                                              kernel_width,
                                                              1,
                                                              0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < output_channels; ++channel) {
            const uint64_t index = value * output_channels + channel;
            EXPECT_NEAR(actual[index], expected[index], 1.0e-5F) << "active output index " << index;
        }
    }
    for (uint64_t index = offsets.back() * output_channels; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index], inactive_sentinel) << "packed spare output element " << index;
    }

    // The retained region must carry one newly selected width through every
    // compatible value when the row partition changes between executions.
    const std::vector<uint32_t> wider_offsets32{0, 9, 9, 9};
    const std::vector<uint64_t> wider_offsets(wider_offsets32.begin(), wider_offsets32.end());
    overwriteGpuTensor<uint32_t>(gpu_offsets, wider_offsets32, stream);
    partition.setHostOffsets(wider_offsets);
    plan.run();
    stream.synchronize();
    const std::vector<RaggedConv1dStageDiagnostic> wider_diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(wider_diagnostics.size(), 2u);
    EXPECT_EQ(wider_diagnostics[0].selected_width_capacity, 12u);
    EXPECT_EQ(wider_diagnostics[1].selected_width_capacity, 12u);
    const std::vector<float> wider_actual = copyToCpuValues(plan.output("y"), stream);
    const std::vector<float> wider_hidden = cpuRaggedCausalConv1d(values,
                                                                  wider_offsets,
                                                                  filter1,
                                                                  max_total_values,
                                                                  input_channels,
                                                                  hidden_channels,
                                                                  kernel_width,
                                                                  1,
                                                                  0.0F);
    const std::vector<float> wider_expected = cpuRaggedCausalConv1d(wider_hidden,
                                                                    wider_offsets,
                                                                    filter2,
                                                                    max_total_values,
                                                                    hidden_channels,
                                                                    output_channels,
                                                                    kernel_width,
                                                                    1,
                                                                    0.0F);
    for (uint64_t value = 0; value < wider_offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < output_channels; ++channel) {
            const uint64_t index = value * output_channels + channel;
            EXPECT_NEAR(wider_actual[index], wider_expected[index], 1.0e-5F) << "wider active output index " << index;
        }
    }
}

TEST(RaggedExpression, CausalConv1dT8ASharesOneEntryPackAcrossCompatibleFanout) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t channels = 1;
    constexpr uint64_t kernel_width = 3;
    const std::vector<uint32_t> offsets32{0, 3, 6};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> values{1.0F, -2.0F, 3.0F, 4.0F, 0.5F, -1.0F, 7777.0F, 7777.0F};
    const std::vector<float> filter1{0.5F, -0.25F, 1.0F};
    const std::vector<float> filter2{-0.75F, 0.5F, 0.25F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel_width}, filter2, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime(
        gpu_offsets, RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values))
        .setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const RaggedExpression y1 =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression y2 =
        input.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(
        Expression::outputs({{"y1", y1.getValues()}, {"y2", y2.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> named_inputs{{"tokens.values", gpu_values},
                                                               {"tokens.offsets", gpu_offsets},
                                                               {"filter1", gpu_filter1},
                                                               {"filter2", gpu_filter2}};
    constexpr float inactive_sentinel = -8765.5F;
    Tensor output1 = makeGpuTensor<float>(
        {max_total_values, channels}, std::vector<float>(max_total_values * channels, inactive_sentinel), stream);
    Tensor output2 = makeGpuTensor<float>(
        {max_total_values, channels}, std::vector<float>(max_total_values * channels, inactive_sentinel), stream);
    StampedExecutionPlan plan = equation.stamp(named_inputs, stream, {}, {{"y1", output1}, {"y2", output2}});

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPack"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausal"), 2);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 2);

    plan.run();
    stream.synchronize();
    const std::vector<float> actual1 = copyToCpuValues(plan.output("y1"), stream);
    const std::vector<float> actual2 = copyToCpuValues(plan.output("y2"), stream);
    const std::vector<float> expected1 = cpuRaggedCausalConv1d(values,
                                                               offsets,
                                                               filter1,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel_width,
                                                               1,
                                                               0.0F);
    const std::vector<float> expected2 = cpuRaggedCausalConv1d(values,
                                                               offsets,
                                                               filter2,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel_width,
                                                               1,
                                                               0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        EXPECT_NEAR(actual1[value], expected1[value], 1.0e-5F) << "y1 active output index " << value;
        EXPECT_NEAR(actual2[value], expected2[value], 1.0e-5F) << "y2 active output index " << value;
    }
    for (uint64_t value = offsets.back(); value < max_total_values; ++value) {
        EXPECT_EQ(actual1[value], inactive_sentinel) << "y1 packed spare output index " << value;
        EXPECT_EQ(actual2[value], inactive_sentinel) << "y2 packed spare output index " << value;
    }
}

TEST(RaggedExpression, CausalConv1dT8BRetainsPointwiseActivationAcrossCompatibleRegion) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t channels = 1;
    constexpr uint64_t kernel_width = 2;
    const std::vector<uint32_t> offsets32{0, 3, 6};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> values{-2.0F, 1.0F, 3.0F, -1.0F, 4.0F, 2.0F, 7777.0F, 7777.0F};
    const std::vector<float> filter1{-0.5F, 1.0F};
    const std::vector<float> filter2{0.25F, 0.75F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel_width}, filter2, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    // T8B keeps this valuewise activation in the same padded physical region.
    // The inactive tail is deliberately undefined; the following causal
    // convolution must not observe it while computing active positions.
    const RaggedExpression activated = hidden.relu();
    const RaggedExpression result =
        activated.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(
        Expression::outputs({{"root", input.getValues()}, {"y", result.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> named_inputs{{"tokens.values", gpu_values},
                                                               {"tokens.offsets", gpu_offsets},
                                                               {"filter1", gpu_filter1},
                                                               {"filter2", gpu_filter2}};
    constexpr float inactive_sentinel = 4321.5F;
    Tensor packed_root_output = makeGpuTensor<float>(
        {max_total_values, channels}, std::vector<float>(max_total_values * channels, inactive_sentinel), stream);
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, channels}, std::vector<float>(max_total_values * channels, inactive_sentinel), stream);
    StampedExecutionPlan plan =
        equation.stamp(named_inputs, stream, {}, {{"root", packed_root_output}, {"y", packed_output}});

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPack"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPointwise"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausal"), 2);

    plan.run();
    stream.synchronize();
    const std::vector<float> root_actual = copyToCpuValues(plan.output("root"), stream);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        EXPECT_EQ(root_actual[value], values[value]) << "packed root output index " << value;
    }
    for (uint64_t value = offsets.back(); value < max_total_values; ++value) {
        EXPECT_EQ(root_actual[value], inactive_sentinel) << "packed root spare output index " << value;
    }
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
    std::vector<float> expected_hidden = cpuRaggedCausalConv1d(values,
                                                               offsets,
                                                               filter1,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel_width,
                                                               1,
                                                               0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        expected_hidden[value] = std::max(expected_hidden[value], 0.0F);
    }
    const std::vector<float> expected = cpuRaggedCausalConv1d(expected_hidden,
                                                              offsets,
                                                              filter2,
                                                              max_total_values,
                                                              channels,
                                                              channels,
                                                              kernel_width,
                                                              1,
                                                              0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        EXPECT_NEAR(actual[value], expected[value], 1.0e-5F) << "active output index " << value;
    }
    for (uint64_t value = offsets.back(); value < max_total_values; ++value) {
        EXPECT_EQ(actual[value], inactive_sentinel) << "packed spare output index " << value;
    }
}


TEST(RaggedExpression, CausalConv1dT8BRetainsChannelBiasAndActivationWithUndefinedTail) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel_width = 3;
    const std::vector<uint32_t> offsets32{0, 4, 4, 9};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> values(max_total_values * channels, 7777.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            values[value * channels + channel] =
                static_cast<float>(static_cast<int>(value * channels + channel) - 6) * 0.25F;
        }
    }
    std::vector<float> filter1(channels * channels * kernel_width);
    std::vector<float> filter2(channels * channels * kernel_width);
    for (size_t i = 0; i < filter1.size(); ++i) {
        filter1[i] = static_cast<float>(static_cast<int>((i * 3) % 9) - 4) * 0.125F;
        filter2[i] = static_cast<float>(static_cast<int>((i * 5) % 11) - 5) * 0.0625F;
    }
    const std::vector<float> bias{2.0F, 1.25F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel_width}, filter2, stream);
    Tensor gpu_bias = makeGpuTensor<float>({channels}, bias, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression bias_expr = Expression::input("bias", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression biased_relu = hidden.mapValues(
        [&](const Expression& x) { return (x + bias_expr).max(Expression::constantScalar(0.0)); });
    const RaggedExpression result =
        biased_relu.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> named_inputs{{"tokens.values", gpu_values},
                                                               {"tokens.offsets", gpu_offsets},
                                                               {"filter1", gpu_filter1},
                                                               {"filter2", gpu_filter2},
                                                               {"bias", gpu_bias}};
    constexpr float inactive_sentinel = -5432.25F;
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, channels},
        std::vector<float>(max_total_values * channels, inactive_sentinel),
        stream);
    StampedExecutionPlan plan = equation.stamp(named_inputs, stream, {}, {{"y", packed_output}});

    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedPointwise",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}));

    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
    std::vector<float> expected_hidden = cpuRaggedCausalConv1d(values,
                                                               offsets,
                                                               filter1,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel_width,
                                                               1,
                                                               0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const uint64_t index = value * channels + channel;
            expected_hidden[index] = std::max(expected_hidden[index] + bias[channel], 0.0F);
        }
    }
    const std::vector<float> expected = cpuRaggedCausalConv1d(expected_hidden,
                                                              offsets,
                                                              filter2,
                                                              max_total_values,
                                                              channels,
                                                              channels,
                                                              kernel_width,
                                                              1,
                                                              0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const uint64_t index = value * channels + channel;
            EXPECT_NEAR(actual[index], expected[index], 1.0e-5F) << "active output index " << index;
        }
    }
    for (uint64_t index = offsets.back() * channels; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index], inactive_sentinel) << "packed spare output index " << index;
    }
}

TEST(RaggedExpression, CausalConv1dT8BRetainsRuntimeScalarPointwiseBeforeConvolution) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    // Use a placement-time row capacity that genuinely has more than one
    // canonical selected width ({8, 16}). The old fixture used 8, whose finite
    // width family is intentionally just {8}, so it could not test the
    // no-runtime-compilation width-switch contract asserted below.
    constexpr uint64_t max_total_values = 16;
    constexpr uint64_t channels = 1;
    constexpr uint64_t kernel_width = 3;
    const std::vector<uint32_t> offsets32{0, 3, 6};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> values{1.0F,
                                    -2.0F,
                                    3.0F,
                                    4.0F,
                                    0.5F,
                                    -1.0F,
                                    2.25F,
                                    -3.5F,
                                    1.75F,
                                    0.25F,
                                    -2.5F,
                                    4.5F,
                                    3.25F,
                                    -1.25F,
                                    0.75F,
                                    -4.0F};
    const std::vector<float> filter{0.5F, -0.25F, 1.0F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter = makeGpuTensor<float>({channels, channels, kernel_width}, filter, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets, RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression alpha = Expression::runtimeScalar("alpha", DataType::FP32, DataType::FP32);
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression scaled = input.mapValues([&](const Expression& x) { return x * alpha; });
    const RaggedExpression result =
        scaled.causalConv1d(filter_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> named_inputs{{"tokens.values", gpu_values},
                                                               {"tokens.offsets", gpu_offsets},
                                                               {"filter", gpu_filter}};
    constexpr float inactive_sentinel = 6789.5F;
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, channels}, std::vector<float>(max_total_values * channels, inactive_sentinel), stream);
    StampedExecutionPlan plan = equation.stamp(named_inputs, stream, {}, {{"y", packed_output}});

    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "PaddedRaggedPointwise",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}));
    EXPECT_EQ(plan.runtimeScalarNames(), (std::unordered_set<std::string>{"alpha"}));

    const std::vector<size_t> pre_stamped_width_counts = plan.paddedRaggedPointwisePreStampedWidthCounts();
    ASSERT_EQ(pre_stamped_width_counts.size(), 1u);
    EXPECT_GT(pre_stamped_width_counts.front(), 1u);

    auto run_and_check = [&](float alpha_value, const std::vector<uint64_t>& run_offsets) {
        overwriteGpuTensor<float>(packed_output,
                                  std::vector<float>(max_total_values * channels, inactive_sentinel),
                                  stream);
        plan.run({{"alpha", alpha_value}});
        stream.synchronize();
        const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
        std::vector<float> scaled_values = values;
        for (uint64_t value = 0; value < run_offsets.back(); ++value) {
            scaled_values[value] *= alpha_value;
        }
        const std::vector<float> expected = cpuRaggedCausalConv1d(scaled_values,
                                                                  run_offsets,
                                                                  filter,
                                                                  max_total_values,
                                                                  channels,
                                                                  channels,
                                                                  kernel_width,
                                                                  1,
                                                                  0.0F);
        for (uint64_t value = 0; value < run_offsets.back(); ++value) {
            EXPECT_NEAR(actual[value], expected[value], 1.0e-5F) << "active output index " << value;
        }
        for (uint64_t value = run_offsets.back(); value < max_total_values; ++value) {
            EXPECT_EQ(actual[value], inactive_sentinel) << "packed spare output index " << value;
        }
    };

    run_and_check(2.0F, offsets);
    run_and_check(-0.5F, offsets);

    // Move from selected W=8 to selected W=16 without compiling anything at
    // runtime; both pointwise invocations must already exist from stamping.
    const std::vector<uint32_t> wider_offsets32{0, 10, 10};
    const std::vector<uint64_t> wider_offsets(wider_offsets32.begin(), wider_offsets32.end());
    overwriteGpuTensor<uint32_t>(gpu_offsets, wider_offsets32, stream);
    partition.setHostOffsets(wider_offsets);
    run_and_check(1.5F, wider_offsets);

    const std::vector<uint32_t> empty_offsets32{0, 0, 0};
    const std::vector<uint64_t> empty_offsets(empty_offsets32.begin(), empty_offsets32.end());
    overwriteGpuTensor<uint32_t>(gpu_offsets, empty_offsets32, stream);
    partition.setHostOffsets(empty_offsets);
    run_and_check(3.0F, empty_offsets);

    EXPECT_EQ(plan.paddedRaggedPointwisePreStampedWidthCounts(), pre_stamped_width_counts);
}

TEST(RaggedExpression, CausalConv1dT8BRetainsCastsAcrossPaddedRegion) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t channels = 1;
    constexpr uint64_t kernel_width = 2;
    const std::vector<uint32_t> offsets32{0, 3, 6};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> values{0.1F, -0.35F, 1.2F, 2.3F, -4.7F, 0.6F, 7777.0F, 7777.0F};
    const std::vector<float> filter{0.5F, -0.25F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter = makeGpuTensorFromFloats({channels, channels, kernel_width}, filter, DataType::FP16, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime(
        gpu_offsets, RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values))
        .setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP16);
    const RaggedExpression quantized = input.cast(DataType::FP16);
    const RaggedExpression result =
        quantized.causalConv1d(filter_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP16);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    constexpr float inactive_sentinel = -123.5F;
    Tensor packed_output = makeGpuTensorFromFloats(
        {max_total_values, channels},
        std::vector<float>(max_total_values * channels, inactive_sentinel),
        DataType::FP16,
        stream);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter", gpu_filter}},
                                               stream,
                                               {},
                                               {{"y", packed_output}});

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPack"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPointwise"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausal"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 1);

    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuFloatValues(plan.output("y"), stream);
    std::vector<float> quantized_values = values;
    std::vector<float> quantized_filter = filter;
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        quantized_values[value] = __half2float(__float2half(quantized_values[value]));
    }
    for (float& coefficient : quantized_filter) {
        coefficient = __half2float(__float2half(coefficient));
    }
    const std::vector<float> expected = cpuRaggedCausalConv1d(quantized_values,
                                                               offsets,
                                                               quantized_filter,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel_width,
                                                               1,
                                                               0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        EXPECT_NEAR(actual[value], expected[value], 2.0e-2F) << "active output index " << value;
    }
    const float stored_sentinel = __half2float(__float2half(inactive_sentinel));
    for (uint64_t value = offsets.back(); value < max_total_values; ++value) {
        EXPECT_EQ(actual[value], stored_sentinel) << "packed spare output index " << value;
    }
}

TEST(RaggedExpression, CausalConv1dT8BRetainsSamePartitionBinaryValuewiseRegion) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel_width = 2;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> values(max_total_values * channels, 7777.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            values[value * channels + channel] =
                static_cast<float>(static_cast<int>((value * 5 + channel * 3) % 17) - 8) * 0.25F;
        }
    }
    const std::vector<float> filter1{
        0.5F, -0.25F,
        0.75F, 0.125F,
        -0.5F, 0.25F,
        0.375F, -0.625F,
    };
    const std::vector<float> filter2{
        -0.125F, 0.625F,
        0.25F, 0.5F,
        0.875F, -0.375F,
        -0.25F, 0.125F,
    };
    const std::vector<float> filter3{
        0.5F, 0.25F,
        -0.125F, 0.75F,
        0.375F, -0.5F,
        0.625F, 0.125F,
    };

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel_width}, filter2, stream);
    Tensor gpu_filter3 = makeGpuTensor<float>({channels, channels, kernel_width}, filter3, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime(
        gpu_offsets, RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values))
        .setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression filter3_expr = Expression::input("filter3", std::nullopt, DataType::FP32);
    const RaggedExpression branch1 =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression branch2 =
        input.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression combined = (branch1 + branch2).relu();
    const RaggedExpression result =
        combined.causalConv1d(filter3_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    constexpr float inactive_sentinel = 2468.5F;
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, channels}, std::vector<float>(max_total_values * channels, inactive_sentinel), stream);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter1", gpu_filter1},
                                                {"filter2", gpu_filter2},
                                                {"filter3", gpu_filter3}},
                                               stream,
                                               {},
                                               {{"y", packed_output}});

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPack"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausal"), 3);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPointwise"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 1);

    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
    const std::vector<float> branch1_expected = cpuRaggedCausalConv1d(values,
                                                                       offsets,
                                                                       filter1,
                                                                       max_total_values,
                                                                       channels,
                                                                       channels,
                                                                       kernel_width,
                                                                       1,
                                                                       0.0F);
    const std::vector<float> branch2_expected = cpuRaggedCausalConv1d(values,
                                                                       offsets,
                                                                       filter2,
                                                                       max_total_values,
                                                                       channels,
                                                                       channels,
                                                                       kernel_width,
                                                                       1,
                                                                       0.0F);
    std::vector<float> combined_expected(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const uint64_t index = value * channels + channel;
            combined_expected[index] = std::max(branch1_expected[index] + branch2_expected[index], 0.0F);
        }
    }
    const std::vector<float> expected = cpuRaggedCausalConv1d(combined_expected,
                                                               offsets,
                                                               filter3,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel_width,
                                                               1,
                                                               0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const uint64_t index = value * channels + channel;
            EXPECT_NEAR(actual[index], expected[index], 1.0e-5F) << "active output index " << index;
        }
    }
    for (uint64_t index = offsets.back() * channels; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index], inactive_sentinel) << "packed spare output index " << index;
    }
}

TEST(RaggedExpression, CausalConv1dT8BFallsBackAtUnsupportedPerTimestepBroadcastBoundary) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel_width = 2;
    const std::vector<uint32_t> offsets32{0, 3, 6};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> values(max_total_values * channels, 7777.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            values[value * channels + channel] = static_cast<float>(value * channels + channel + 1) * 0.25F;
        }
    }
    const std::vector<float> filter1{
        0.5F, -0.25F,
        0.75F, 0.125F,
        -0.5F, 0.25F,
        0.375F, -0.625F,
    };
    const std::vector<float> filter2{
        0.25F, 0.5F,
        -0.125F, 0.75F,
        0.625F, -0.25F,
        0.5F, 0.125F,
    };
    std::vector<float> timestep_bias(max_total_values, 99.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        timestep_bias[value] = static_cast<float>(static_cast<int>(value) - 2) * 0.5F;
    }

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel_width}, filter2, stream);
    Tensor gpu_timestep_bias = makeGpuTensor<float>({max_total_values, 1}, timestep_bias, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime(
        gpu_offsets, RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values))
        .setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression timestep_bias_expr = Expression::input("timestep_bias", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression biased = hidden.mapValues([&](const Expression& x) { return x + timestep_bias_expr; });
    const RaggedExpression result =
        biased.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    constexpr float inactive_sentinel = 1357.25F;
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, channels}, std::vector<float>(max_total_values * channels, inactive_sentinel), stream);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter1", gpu_filter1},
                                                {"filter2", gpu_filter2},
                                                {"timestep_bias", gpu_timestep_bias}},
                                               stream,
                                               {},
                                               {{"y", packed_output}});

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPointwise"), 0);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPack"), 2);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 2);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausal"), 2);

    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
    std::vector<float> expected_hidden = cpuRaggedCausalConv1d(values,
                                                               offsets,
                                                               filter1,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel_width,
                                                               1,
                                                               0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            expected_hidden[value * channels + channel] += timestep_bias[value];
        }
    }
    const std::vector<float> expected = cpuRaggedCausalConv1d(expected_hidden,
                                                               offsets,
                                                               filter2,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel_width,
                                                               1,
                                                               0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const uint64_t index = value * channels + channel;
            EXPECT_NEAR(actual[index], expected[index], 1.0e-5F) << "active output index " << index;
        }
    }
    for (uint64_t index = offsets.back() * channels; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index], inactive_sentinel) << "packed spare output index " << index;
    }
}

TEST(RaggedExpression, T8BDoesNotPadStandaloneValuewiseExpressionWithoutCompatibleRegion) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t channels = 1;
    const std::vector<uint32_t> offsets32{0, 3, 6};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> values{-2.0F, 1.0F, 3.0F, -1.0F, 4.0F, 2.0F, 7777.0F, 7777.0F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime(
        gpu_offsets, RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values))
        .setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const RaggedExpression result = input.relu();
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values}, {"tokens.offsets", gpu_offsets}}, stream);

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPack"), 0);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPointwise"), 0);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 0);
    EXPECT_GE(std::count(stage_names.begin(), stage_names.end(), "FusedKernel"), 1);

    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        EXPECT_EQ(actual[value], std::max(values[value], 0.0F));
    }
}

TEST(RaggedExpression, CausalConv1dT7R5PrebuildsFixedPlanFamilyBeforeExecutionAndNeverGrowsIt) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 24;
    constexpr uint64_t input_channels = 1;
    constexpr uint64_t output_channels = 1;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t dilation = 1;
    std::vector<float> values(max_total_values, 9999.0F);
    for (uint64_t i = 0; i < 14; ++i) values[i] = static_cast<float>(i + 1);
    const std::vector<float> filter{0.5F, -1.0F, 2.0F};
    const std::vector<uint32_t> first_offsets32{0, 9, 13, 14};
    const std::vector<uint64_t> first_offsets(first_offsets32.begin(), first_offsets32.end());
    const std::vector<uint32_t> second_offsets32{0, 7, 7, 12};
    const std::vector<uint64_t> second_offsets(second_offsets32.begin(), second_offsets32.end());

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, input_channels}, values, stream);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels, kernel_width}, filter, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, first_offsets32, stream);
    RowPartitionRuntime partition(gpu_offsets,
                                  RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(first_offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression result =
        input.causalConv1d(filter_expr, output_channels, kernel_width, dilation, DataType::FP32, DataType::FP32);
    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_values},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter", gpu_filter}},
                                               stream);

    const std::vector<RaggedConv1dStageDiagnostic> stamped_diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(stamped_diagnostics.size(), 1u);
    EXPECT_GT(stamped_diagnostics.front().width_capacity_count, 0u);
    EXPECT_EQ(stamped_diagnostics.front().prebuilt_cudnn_plan_count,
              stamped_diagnostics.front().width_capacity_count);
    const uint64_t placement_plan_count = stamped_diagnostics.front().prebuilt_cudnn_plan_count;
    const uint64_t placement_workspace_bytes = stamped_diagnostics.front().cudnn_workspace_bytes;

    plan.run();
    stream.synchronize();
    std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);
    std::vector<float> expected = cpuRaggedCausalConv1d(values,
                                                        first_offsets,
                                                        filter,
                                                        max_total_values,
                                                        input_channels,
                                                        output_channels,
                                                        kernel_width,
                                                        dilation,
                                                        0.0F);
    for (uint64_t i = 0; i < first_offsets.back(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1.0e-5F) << "first partition index " << i;
    }
    {
        const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
        ASSERT_EQ(diagnostics.size(), 1u);
        EXPECT_EQ(diagnostics.front().selected_width_capacity, 16u);
        EXPECT_EQ(diagnostics.front().prebuilt_cudnn_plan_count, placement_plan_count);
        EXPECT_EQ(diagnostics.front().width_capacity_count, placement_plan_count);
        EXPECT_EQ(diagnostics.front().cudnn_workspace_bytes, placement_workspace_bytes);
    }

    // A generic offsets write invalidates the host mirror; publish the new
    // already-known metadata after the copy, exactly as NetworkInput does.
    overwriteGpuTensor<uint32_t>(gpu_offsets, second_offsets32, stream);
    partition.setHostOffsets(second_offsets);
    plan.run();
    stream.synchronize();
    actual = copyToCpuValues(plan.output("y"), stream);
    expected = cpuRaggedCausalConv1d(values,
                                     second_offsets,
                                     filter,
                                     max_total_values,
                                     input_channels,
                                     output_channels,
                                     kernel_width,
                                     dilation,
                                     0.0F);
    for (uint64_t i = 0; i < second_offsets.back(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1.0e-5F) << "second partition index " << i;
    }

    const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(diagnostics.size(), 1u);
    EXPECT_EQ(diagnostics.front().selected_width_capacity, 8u);
    EXPECT_EQ(diagnostics.front().prebuilt_cudnn_plan_count, placement_plan_count);
    EXPECT_EQ(diagnostics.front().width_capacity_count, placement_plan_count);
    EXPECT_EQ(diagnostics.front().cudnn_workspace_bytes, placement_workspace_bytes);

    // Repeated runtime shape changes may select different prebuilt widths but
    // can never construct another cuDNN graph/plan or grow the shared workspace.
    for (int iteration = 0; iteration < 8; ++iteration) {
        const bool use_first = (iteration % 2) == 0;
        overwriteGpuTensor<uint32_t>(gpu_offsets, use_first ? first_offsets32 : second_offsets32, stream);
        partition.setHostOffsets(use_first ? first_offsets : second_offsets);
        plan.run();
        stream.synchronize();
        const std::vector<RaggedConv1dStageDiagnostic> repeated = plan.raggedConv1dStageDiagnostics();
        ASSERT_EQ(repeated.size(), 1u);
        EXPECT_EQ(repeated.front().prebuilt_cudnn_plan_count, placement_plan_count);
        EXPECT_EQ(repeated.front().width_capacity_count, placement_plan_count);
        EXPECT_EQ(repeated.front().cudnn_workspace_bytes, placement_workspace_bytes);
        EXPECT_EQ(repeated.front().selected_width_capacity, use_first ? 16u : 8u);
    }
}

TEST(RaggedExpression, CausalConv1dT9ADgradMatchesPackedReferenceForGroupedAndDepthwise) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 16;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t dilation = 1;
    constexpr float inactive_sentinel = -8181.0F;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    struct Geometry {
        uint64_t input_channels;
        uint64_t output_channels;
        uint64_t groups;
    };
    for (const Geometry geometry : {Geometry{3, 4, 1}, Geometry{4, 6, 2}, Geometry{4, 4, 4}}) {
        SCOPED_TRACE("groups=" + std::to_string(geometry.groups));
        const uint64_t input_channels_per_group = geometry.input_channels / geometry.groups;
        std::vector<float> filter(geometry.output_channels * input_channels_per_group * kernel_width);
        for (size_t i = 0; i < filter.size(); ++i) {
            filter[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125F;
        }
        std::vector<float> dy(max_total_values * geometry.output_channels, 12345.0F);
        for (uint64_t value = 0; value < offsets.back(); ++value) {
            for (uint64_t channel = 0; channel < geometry.output_channels; ++channel) {
                dy[value * geometry.output_channels + channel] =
                    static_cast<float>((value + 1) * (channel + 2)) * 0.0625F - 0.4F;
            }
        }

        Stream stream(0);
        Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
        RowPartitionRuntime partition(
            gpu_offsets,
            RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
        partition.setHostOffsets(offsets);
        Tensor gpu_filter = makeGpuTensor<float>(
            {geometry.output_channels, input_channels_per_group, kernel_width}, filter, stream);
        Tensor gpu_dy = makeGpuTensor<float>({max_total_values, geometry.output_channels}, dy, stream);
        Tensor gpu_values = makeGpuTensor<float>(
            {max_total_values, geometry.input_channels},
            std::vector<float>(max_total_values * geometry.input_channels, 0.0F),
            stream);

        const RaggedExpression input = RaggedExpression::input(
            "tokens",
            makeDescriptor(DataType::FP32, {geometry.input_channels}, batch_size, max_total_values, DataType::UINT32));
        const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
        const RaggedExpression output = input.causalConv1d(filter_expr,
                                                           geometry.output_channels,
                                                           kernel_width,
                                                           dilation,
                                                           DataType::FP32,
                                                           DataType::FP32,
                                                           geometry.groups);
        FusedEquation forward =
            FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
        FusedEquation backward = forward.compileBackward({"tokens.values"}, "dy");

        Tensor dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, geometry.input_channels}));
        dx.fill(inactive_sentinel, stream);
        StampedExecutionPlan plan = backward.stamp({{"tokens.values", gpu_values},
                                                    {"filter", gpu_filter},
                                                    {"tokens.offsets", gpu_offsets},
                                                    {"dy", gpu_dy}},
                                                   stream,
                                                   {},
                                                   {{"tokens.values_grad", dx}});
        plan.run();
        stream.synchronize();

        const std::vector<float> actual = copyToCpuValues(plan.output("tokens.values_grad"), stream);
        const std::vector<float> expected = cpuRaggedCausalConv1dDgrad(dy,
                                                                       offsets,
                                                                       filter,
                                                                       max_total_values,
                                                                       geometry.input_channels,
                                                                       geometry.output_channels,
                                                                       kernel_width,
                                                                       dilation,
                                                                       inactive_sentinel,
                                                                       geometry.groups);
        expectNear(actual, expected, 1.0e-5F);

        const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
        ASSERT_EQ(diagnostics.size(), 1u);
        EXPECT_EQ(diagnostics.front().active_values, offsets.back());
        EXPECT_EQ(diagnostics.front().selected_width_capacity, 8u);
        EXPECT_GT(diagnostics.front().width_capacity_count, 0u);
        EXPECT_EQ(diagnostics.front().prebuilt_cudnn_plan_count, diagnostics.front().width_capacity_count);
        EXPECT_EQ(diagnostics.front().explicit_unfold_workspace_bytes, 0u);
    }
}

TEST(RaggedExpression, CausalConv1dT9AAllEmptyUsesWidthZeroAndLeavesDxInactiveStorageUndefined) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 2;
    constexpr float sentinel = 4545.0F;
    const std::vector<uint32_t> offsets32{0, 0, 0, 0};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> filter{0.5F, -0.25F, 1.0F, 0.75F, -0.5F, 0.125F,
                                    -0.2F, 0.4F, 0.6F, -0.8F, 0.3F, 0.7F};
    std::vector<float> dy(max_total_values * output_channels, std::numeric_limits<float>::quiet_NaN());

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels, 3}, filter, stream);
    Tensor gpu_dy = makeGpuTensor<float>({max_total_values, output_channels}, dy, stream);
    Tensor gpu_values = makeGpuTensor<float>(
        {max_total_values, input_channels}, std::vector<float>(max_total_values * input_channels, 0.0F), stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression output = input.causalConv1d(filter_expr, output_channels, 3, 1, DataType::FP32, DataType::FP32);
    FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"tokens.values"}, "dy");

    Tensor dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, input_channels}));
    dx.fill(sentinel, stream);
    StampedExecutionPlan plan = backward.stamp({{"tokens.values", gpu_values},
                                                {"filter", gpu_filter},
                                                {"tokens.offsets", gpu_offsets},
                                                {"dy", gpu_dy}},
                                               stream,
                                               {},
                                               {{"tokens.values_grad", dx}});
    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("tokens.values_grad"), stream);
    for (float value : actual) {
        EXPECT_EQ(value, sentinel);
    }
    const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(diagnostics.size(), 1u);
    EXPECT_EQ(diagnostics.front().active_values, 0u);
    EXPECT_EQ(diagnostics.front().selected_width_capacity, 0u);
    EXPECT_EQ(diagnostics.front().prebuilt_cudnn_plan_count, diagnostics.front().width_capacity_count);
}

TEST(RaggedExpression, CausalConv1dT9ADgradConsumerSanitizesPoisonedRetainedDyWithoutMutatingProducer) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t max_values_per_row = 8;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 2;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t selected_width = 8;
    constexpr float inactive_sentinel = -9911.0F;
    const std::vector<uint32_t> offsets32{0, 2, 5};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> filter{0.5F, -0.25F, 1.0F, -0.75F, 0.4F, 0.2F,
                                    -0.3F, 0.6F, 0.8F, 0.125F, -0.5F, 0.9F};
    std::vector<float> packed_dy(max_total_values * output_channels, 3333.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        packed_dy[value * output_channels + 0] = 0.2F * static_cast<float>(value + 1);
        packed_dy[value * output_channels + 1] = -0.15F * static_cast<float>(value + 2);
    }

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_values_per_row));
    partition.setHostOffsets(offsets);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels, kernel_width}, filter, stream);

    PaddedRaggedSequencePlan dy_plan =
        preparePaddedRaggedSequencePlan(partition, output_channels, DataType::FP32, selected_width);
    PaddedRaggedSequencePlan dx_plan =
        preparePaddedRaggedSequencePlan(partition, input_channels, DataType::FP32, selected_width);
    auto padded_dy = std::make_shared<PaddedRaggedSequence>(dy_plan, gpu_offsets, gpuPlacement, selected_width);
    auto padded_dx = std::make_shared<PaddedRaggedSequence>(dx_plan, gpu_offsets, gpuPlacement, selected_width);
    auto compiled = std::make_shared<CompiledRaggedConv1dCausalBackwardData>(DataType::FP32,
                                                                             DataType::FP32,
                                                                             DataType::FP32,
                                                                             DataType::FP32,
                                                                             DataType::UINT32,
                                                                             batch_size,
                                                                             max_total_values,
                                                                             max_values_per_row,
                                                                             input_channels,
                                                                             output_channels,
                                                                             kernel_width,
                                                                             1,
                                                                             1);
    StampedRaggedConv1dCausalBackwardData dgrad(compiled, gpu_filter, padded_dy, gpu_offsets, padded_dx, stream);
    // Plan-family construction is placement-only and may leave the shared value
    // configured at the last prebuilt W. Establish this execution's retained W
    // exactly as an upstream retained producer would before poisoning its tail.
    padded_dy->reconfigure(dy_plan);

    std::vector<float> poisoned(batch_size * output_channels * selected_width,
                                std::numeric_limits<float>::quiet_NaN());
    for (uint64_t row = 0; row < batch_size; ++row) {
        const uint64_t begin = offsets[row];
        const uint64_t row_length = offsets[row + 1] - begin;
        for (uint64_t channel = 0; channel < output_channels; ++channel) {
            for (uint64_t timestep = 0; timestep < row_length; ++timestep) {
                poisoned[(row * output_channels + channel) * selected_width + timestep] =
                    packed_dy[(begin + timestep) * output_channels + channel];
            }
            for (uint64_t timestep = row_length; timestep < selected_width; ++timestep) {
                const size_t index = (row * output_channels + channel) * selected_width + timestep;
                poisoned[index] = ((index & 1U) == 0U) ? std::numeric_limits<float>::infinity()
                                                       : std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    Tensor dy_storage = padded_dy->getPaddedValuesStorage();
    overwriteGpuTensor<float>(dy_storage, poisoned, stream);
    const std::vector<float> producer_before = copyToCpuValues(dy_storage, stream);

    Stream other_stream(0);
    EXPECT_THROW(dgrad.runOn(other_stream), std::runtime_error);
    dgrad.runOn(stream);
    stream.synchronize();

    Tensor packed_dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, input_channels}));
    packed_dx.fill(inactive_sentinel, stream);
    padded_dx->unpackTo(packed_dx, stream);
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(packed_dx, stream);
    const std::vector<float> expected = cpuRaggedCausalConv1dDgrad(packed_dy,
                                                                   offsets,
                                                                   filter,
                                                                   max_total_values,
                                                                   input_channels,
                                                                   output_channels,
                                                                   kernel_width,
                                                                   1,
                                                                   inactive_sentinel);
    expectNear(actual, expected, 1.0e-5F);
    for (uint64_t i = 0; i < offsets.back() * input_channels; ++i) {
        EXPECT_TRUE(std::isfinite(actual[i])) << "active dX index " << i;
    }

    const std::vector<float> producer_after = copyToCpuValues(dy_storage, stream);
    ASSERT_EQ(producer_after.size(), producer_before.size());
    for (size_t i = 0; i < producer_before.size(); ++i) {
        if (std::isnan(producer_before[i])) {
            EXPECT_TRUE(std::isnan(producer_after[i])) << "producer dY index " << i;
        } else {
            EXPECT_EQ(producer_after[i], producer_before[i]) << "producer dY index " << i;
        }
    }
    const RaggedConv1dStageDiagnostic diagnostic = dgrad.diagnostic();
    EXPECT_EQ(diagnostic.active_values, offsets.back());
    EXPECT_EQ(diagnostic.selected_width_capacity, selected_width);
    EXPECT_EQ(diagnostic.prebuilt_cudnn_plan_count, diagnostic.width_capacity_count);
    EXPECT_GT(diagnostic.prebuilt_cudnn_plan_count, 0u);
    EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u);
}

TEST(RaggedExpression, CausalConv1dT9BWgradMatchesPackedReferenceForGroupedAndDepthwise) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 16;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t dilation = 1;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    struct Geometry {
        uint64_t input_channels;
        uint64_t output_channels;
        uint64_t groups;
    };
    for (const Geometry geometry : {Geometry{3, 4, 1}, Geometry{4, 6, 2}, Geometry{4, 4, 4}}) {
        SCOPED_TRACE("groups=" + std::to_string(geometry.groups));
        const uint64_t input_channels_per_group = geometry.input_channels / geometry.groups;
        std::vector<float> x(max_total_values * geometry.input_channels, 7777.0F);
        std::vector<float> dy(max_total_values * geometry.output_channels, -8888.0F);
        for (uint64_t value = 0; value < offsets.back(); ++value) {
            for (uint64_t channel = 0; channel < geometry.input_channels; ++channel) {
                x[value * geometry.input_channels + channel] =
                    static_cast<float>(static_cast<int>((value + 2) * (channel + 1)) - 5) * 0.125F;
            }
            for (uint64_t channel = 0; channel < geometry.output_channels; ++channel) {
                dy[value * geometry.output_channels + channel] =
                    static_cast<float>(static_cast<int>((value + 1) * (channel + 3)) - 7) * 0.0625F;
            }
        }
        std::vector<float> filter(geometry.output_channels * input_channels_per_group * kernel_width, 0.0F);

        Stream stream(0);
        Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
        RowPartitionRuntime partition(
            gpu_offsets,
            RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
        partition.setHostOffsets(offsets);
        Tensor gpu_x = makeGpuTensor<float>({max_total_values, geometry.input_channels}, x, stream);
        Tensor gpu_dy = makeGpuTensor<float>({max_total_values, geometry.output_channels}, dy, stream);
        Tensor gpu_filter = makeGpuTensor<float>(
            {geometry.output_channels, input_channels_per_group, kernel_width}, filter, stream);

        const RaggedExpression input = RaggedExpression::input(
            "tokens",
            makeDescriptor(DataType::FP32, {geometry.input_channels}, batch_size, max_total_values, DataType::UINT32));
        const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
        const RaggedExpression output = input.causalConv1d(filter_expr,
                                                           geometry.output_channels,
                                                           kernel_width,
                                                           dilation,
                                                           DataType::FP32,
                                                           DataType::FP32,
                                                           geometry.groups);
        FusedEquation forward =
            FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
        FusedEquation backward = forward.compileBackward({"filter"}, "dy");

        StampedExecutionPlan plan = backward.stamp({{"tokens.values", gpu_x},
                                                    {"filter", gpu_filter},
                                                    {"tokens.offsets", gpu_offsets},
                                                    {"dy", gpu_dy}},
                                                   stream);
        plan.run();
        stream.synchronize();

        const std::vector<float> actual = copyToCpuValues(plan.output("filter_grad"), stream);
        const std::vector<float> expected = cpuRaggedCausalConv1dWgrad(x,
                                                                       dy,
                                                                       offsets,
                                                                       max_total_values,
                                                                       geometry.input_channels,
                                                                       geometry.output_channels,
                                                                       kernel_width,
                                                                       dilation,
                                                                       geometry.groups);
        expectNear(actual, expected, 1.0e-5F);
        const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
        ASSERT_EQ(diagnostics.size(), 1u);
        EXPECT_EQ(diagnostics.front().active_values, offsets.back());
        EXPECT_EQ(diagnostics.front().selected_width_capacity, 8u);
        EXPECT_GT(diagnostics.front().width_capacity_count, 0u);
        EXPECT_EQ(diagnostics.front().prebuilt_cudnn_plan_count, diagnostics.front().width_capacity_count);
        EXPECT_EQ(diagnostics.front().explicit_unfold_workspace_bytes, 0u);
    }
}

TEST(RaggedExpression, CausalConv1dT9BAllEmptyProducesExactZeroWeightGradient) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t input_channels = 4;
    constexpr uint64_t output_channels = 4;
    constexpr uint64_t groups = 4;
    constexpr uint64_t kernel_width = 3;
    const std::vector<uint32_t> offsets32{0, 0, 0, 0};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    std::vector<float> x(max_total_values * input_channels, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> dy(max_total_values * output_channels, std::numeric_limits<float>::infinity());
    std::vector<float> filter(output_channels * (input_channels / groups) * kernel_width, 0.0F);

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);
    Tensor gpu_x = makeGpuTensor<float>({max_total_values, input_channels}, x, stream);
    Tensor gpu_dy = makeGpuTensor<float>({max_total_values, output_channels}, dy, stream);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels / groups, kernel_width}, filter, stream);
    Tensor dw(gpuPlacement, TensorDescriptor(DataType::FP32, {output_channels, input_channels / groups, kernel_width}));
    dw.fill(37.0F, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {input_channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression output =
        input.causalConv1d(filter_expr, output_channels, kernel_width, 1, DataType::FP32, DataType::FP32, groups);
    FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"filter"}, "dy");
    StampedExecutionPlan plan = backward.stamp({{"tokens.values", gpu_x},
                                                {"filter", gpu_filter},
                                                {"tokens.offsets", gpu_offsets},
                                                {"dy", gpu_dy}},
                                               stream,
                                               {},
                                               {{"filter_grad", dw}});
    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("filter_grad"), stream);
    for (float value : actual) {
        EXPECT_EQ(value, 0.0F);
        EXPECT_FALSE(std::signbit(value));
    }
    const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(diagnostics.size(), 1u);
    EXPECT_EQ(diagnostics.front().active_values, 0u);
    EXPECT_EQ(diagnostics.front().selected_width_capacity, 0u);
    EXPECT_EQ(diagnostics.front().prebuilt_cudnn_plan_count, diagnostics.front().width_capacity_count);
}

TEST(RaggedExpression, CausalConv1dT9BWgradConsumerSanitizesBothInputsAndCatchesZeroTimesNaN) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 16;
    constexpr uint64_t max_values_per_row = 16;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 2;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t selected_width = 8;
    const std::vector<uint32_t> offsets32{0, 2, 5};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    std::vector<float> packed_x(max_total_values * input_channels, 1234.0F);
    std::vector<float> packed_dy(max_total_values * output_channels, -5678.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < input_channels; ++channel) {
            packed_x[value * input_channels + channel] =
                static_cast<float>(static_cast<int>((value + 1) * (channel + 2)) - 4) * 0.2F;
        }
        for (uint64_t channel = 0; channel < output_channels; ++channel) {
            packed_dy[value * output_channels + channel] =
                static_cast<float>(static_cast<int>((value + 2) * (channel + 1)) - 3) * 0.15F;
        }
    }
    const std::vector<float> expected = cpuRaggedCausalConv1dWgrad(
        packed_x, packed_dy, offsets, max_total_values, input_channels, output_channels, kernel_width, 1);

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_values_per_row));
    partition.setHostOffsets(offsets);
    PaddedRaggedSequencePlan x_plan =
        preparePaddedRaggedSequencePlan(partition, input_channels, DataType::FP32, selected_width);
    PaddedRaggedSequencePlan dy_plan =
        preparePaddedRaggedSequencePlan(partition, output_channels, DataType::FP32, selected_width);
    auto padded_x = std::make_shared<PaddedRaggedSequence>(x_plan, gpu_offsets, gpuPlacement, selected_width);
    auto padded_dy = std::make_shared<PaddedRaggedSequence>(dy_plan, gpu_offsets, gpuPlacement, selected_width);
    Tensor dw(gpuPlacement, TensorDescriptor(DataType::FP32, {output_channels, input_channels, kernel_width}));
    auto compiled = std::make_shared<CompiledRaggedConv1dCausalBackwardFilter>(DataType::FP32,
                                                                               DataType::FP32,
                                                                               DataType::FP32,
                                                                               DataType::FP32,
                                                                               DataType::UINT32,
                                                                               batch_size,
                                                                               max_total_values,
                                                                               max_values_per_row,
                                                                               input_channels,
                                                                               output_channels,
                                                                               kernel_width,
                                                                               1,
                                                                               1);
    StampedRaggedConv1dCausalBackwardFilter wgrad(compiled, padded_x, padded_dy, gpu_offsets, dw, stream);
    EXPECT_EQ(padded_x->getPlan().widthCapacity, selected_width);
    EXPECT_EQ(padded_dy->getPlan().widthCapacity, selected_width);

    auto make_poisoned_padded = [&](uint64_t channels, const std::vector<float>& packed, bool poison_tail) {
        std::vector<float> values(batch_size * channels * selected_width, 0.0F);
        for (uint64_t row = 0; row < batch_size; ++row) {
            const uint64_t begin = offsets[row];
            const uint64_t row_length = offsets[row + 1] - begin;
            for (uint64_t channel = 0; channel < channels; ++channel) {
                for (uint64_t timestep = 0; timestep < row_length; ++timestep) {
                    values[(row * channels + channel) * selected_width + timestep] =
                        packed[(begin + timestep) * channels + channel];
                }
                if (poison_tail) {
                    for (uint64_t timestep = row_length; timestep < selected_width; ++timestep) {
                        const size_t index = (row * channels + channel) * selected_width + timestep;
                        values[index] = ((index & 1U) == 0U) ? std::numeric_limits<float>::quiet_NaN()
                                                               : std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
        return values;
    };

    // First execution is the specifically adversarial 0 * NaN case: inactive
    // dY is exactly zero while inactive X is NaN/Inf. Sanitizing dY alone is
    // insufficient because IEEE 0 * NaN is NaN.
    Tensor x_storage = padded_x->getPaddedValuesStorage();
    Tensor dy_storage = padded_dy->getPaddedValuesStorage();
    const std::vector<float> poisoned_x = make_poisoned_padded(input_channels, packed_x, true);
    const std::vector<float> zero_tail_dy = make_poisoned_padded(output_channels, packed_dy, false);
    overwriteGpuTensor<float>(x_storage, poisoned_x, stream);
    overwriteGpuTensor<float>(dy_storage, zero_tail_dy, stream);
    const std::vector<float> x_before = copyToCpuValues(x_storage, stream);
    const std::vector<float> dy_before = copyToCpuValues(dy_storage, stream);
    Stream other_stream(0);
    EXPECT_THROW(wgrad.runOn(other_stream), std::runtime_error);
    wgrad.runOn(stream);
    stream.synchronize();
    expectNear(copyToCpuValues(dw, stream), expected, 1.0e-5F);
    for (float value : copyToCpuValues(dw, stream)) EXPECT_TRUE(std::isfinite(value));

    auto expect_storage_unchanged = [&](const Tensor& storage, const std::vector<float>& before, const char* label) {
        const std::vector<float> after = copyToCpuValues(storage, stream);
        ASSERT_EQ(after.size(), before.size());
        for (size_t i = 0; i < before.size(); ++i) {
            if (std::isnan(before[i])) {
                EXPECT_TRUE(std::isnan(after[i])) << label << " index " << i;
            } else {
                EXPECT_EQ(after[i], before[i]) << label << " index " << i;
            }
        }
    };
    expect_storage_unchanged(x_storage, x_before, "producer X");
    expect_storage_unchanged(dy_storage, dy_before, "producer dY");

    // Second execution independently proves dY sanitation: keep inactive X
    // finite/zero and poison inactive dY with NaN/Inf.
    const std::vector<float> zero_tail_x = make_poisoned_padded(input_channels, packed_x, false);
    const std::vector<float> poisoned_dy = make_poisoned_padded(output_channels, packed_dy, true);
    overwriteGpuTensor<float>(x_storage, zero_tail_x, stream);
    overwriteGpuTensor<float>(dy_storage, poisoned_dy, stream);
    const std::vector<float> x2_before = copyToCpuValues(x_storage, stream);
    const std::vector<float> dy2_before = copyToCpuValues(dy_storage, stream);
    wgrad.runOn(stream);
    stream.synchronize();
    expectNear(copyToCpuValues(dw, stream), expected, 1.0e-5F);
    for (float value : copyToCpuValues(dw, stream)) EXPECT_TRUE(std::isfinite(value));
    expect_storage_unchanged(x_storage, x2_before, "producer X second run");
    expect_storage_unchanged(dy_storage, dy2_before, "producer dY second run");

    const RaggedConv1dStageDiagnostic diagnostic = wgrad.diagnostic();
    EXPECT_EQ(diagnostic.active_values, offsets.back());
    EXPECT_EQ(diagnostic.selected_width_capacity, selected_width);
    EXPECT_EQ(diagnostic.prebuilt_cudnn_plan_count, diagnostic.width_capacity_count);
    EXPECT_GT(diagnostic.prebuilt_cudnn_plan_count, 0u);
    EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u);
}

TEST(RaggedExpression, CausalConv1dT9CRetainsReluBackwardBetweenConvolutions) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel_width = 2;
    constexpr float inactive_sentinel = -9191.0F;
    const std::vector<uint32_t> offsets32{0, 3, 5};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> x(max_total_values * channels, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        x[value * channels + 0] = static_cast<float>(static_cast<int>(value) - 2) * 0.5F + 0.125F;
        x[value * channels + 1] = static_cast<float>(3 - static_cast<int>(value)) * 0.25F + 0.0625F;
    }
    const std::vector<float> filter1{0.5F, -0.25F, 0.25F, 0.5F,
                                     -0.5F, 0.25F, 0.5F, -0.25F};
    const std::vector<float> filter2{0.25F, 0.5F, -0.5F, 0.25F,
                                     0.5F, -0.25F, 0.25F, 0.5F};
    std::vector<float> dy(max_total_values * channels, std::numeric_limits<float>::infinity());
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        dy[value * channels + 0] = static_cast<float>(value + 1) * 0.125F;
        dy[value * channels + 1] = -static_cast<float>(value + 2) * 0.0625F;
    }

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);
    Tensor gpu_x = makeGpuTensor<float>({max_total_values, channels}, x, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel_width}, filter2, stream);
    Tensor gpu_dy = makeGpuTensor<float>({max_total_values, channels}, dy, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression activated = hidden.relu();
    const RaggedExpression output =
        activated.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan forward_plan = forward.stamp({{"tokens.values", gpu_x},
                                                       {"tokens.offsets", gpu_offsets},
                                                       {"filter1", gpu_filter1},
                                                       {"filter2", gpu_filter2}},
                                                      stream);
    const std::vector<std::string> forward_stage_names = forward_plan.stageKindNames();
    EXPECT_EQ(std::count(forward_stage_names.begin(), forward_stage_names.end(), "PaddedRaggedPack"), 1);
    EXPECT_EQ(std::count(forward_stage_names.begin(), forward_stage_names.end(), "PaddedRaggedPointwise"), 1);
    EXPECT_EQ(std::count(forward_stage_names.begin(), forward_stage_names.end(), "RaggedConv1dCausal"), 2);
    EXPECT_EQ(std::count(forward_stage_names.begin(), forward_stage_names.end(), "PaddedRaggedUnpack"), 1);

    FusedEquation backward = forward.compileBackward({"tokens.values"}, "dy");
    Tensor dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, channels}));
    dx.fill(inactive_sentinel, stream);
    StampedExecutionPlan backward_plan = backward.stamp({{"tokens.values", gpu_x},
                                                         {"tokens.offsets", gpu_offsets},
                                                         {"filter1", gpu_filter1},
                                                         {"filter2", gpu_filter2},
                                                         {"dy", gpu_dy}},
                                                        stream,
                                                        {},
                                                        {{"tokens.values_grad", dx}});
    const std::vector<std::string> backward_stage_names = backward_plan.stageKindNames();
    EXPECT_EQ(std::count(backward_stage_names.begin(), backward_stage_names.end(), "RaggedConv1dCausalBackwardData"), 2);
    EXPECT_EQ(std::count(backward_stage_names.begin(), backward_stage_names.end(), "PaddedRaggedPointwise"), 1);
    EXPECT_EQ(std::count(backward_stage_names.begin(), backward_stage_names.end(), "PaddedRaggedUnpack"), 1);
    ASSERT_EQ(backward_plan.paddedRaggedPointwisePreStampedWidthCounts().size(), 1u);
    EXPECT_GT(backward_plan.paddedRaggedPointwisePreStampedWidthCounts().front(), 0u);

    backward_plan.run();
    stream.synchronize();

    std::vector<float> hidden_cpu = cpuRaggedCausalConv1d(
        x, offsets, filter1, max_total_values, channels, channels, kernel_width, 1, 0.0F);
    std::vector<float> dactivated = cpuRaggedCausalConv1dDgrad(
        dy, offsets, filter2, max_total_values, channels, channels, kernel_width, 1, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const size_t index = value * channels + channel;
            dactivated[index] *= hidden_cpu[index] > 0.0F ? 1.0F : 0.0F;
        }
    }
    const std::vector<float> expected = cpuRaggedCausalConv1dDgrad(
        dactivated, offsets, filter1, max_total_values, channels, channels, kernel_width, 1, inactive_sentinel);
    const std::vector<float> actual = copyToCpuValues(backward_plan.output("tokens.values_grad"), stream);
    expectNear(actual, expected, 1.0e-5F);
    for (uint64_t value = offsets.back(); value < max_total_values; ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            EXPECT_EQ(actual[value * channels + channel], inactive_sentinel);
        }
    }
}

TEST(RaggedExpression, CausalConv1dM7TerminalOutputCastStaysRetainedAfterDgrad) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel_width = 2;
    const std::vector<uint32_t> offsets32{0, 2, 5};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> x(max_total_values * channels, 0.25F);
    const std::vector<float> filter(channels * channels * kernel_width, 0.125F);
    const std::vector<float> dy(max_total_values * channels, 0.5F);

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);
    Tensor gpu_x = makeGpuTensor<float>({max_total_values, channels}, x, stream);
    Tensor gpu_filter = makeGpuTensor<float>({channels, channels, kernel_width}, filter, stream);
    Tensor gpu_dy = makeGpuTensor<float>({max_total_values, channels}, dy, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
    const RaggedExpression output =
        input.causalConv1d(filter_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    PhysicalOutputs forward = Expression::outputs({{"y", output.getValues()}}).physicalOutputs();

    const std::unordered_map<std::string, std::vector<uint64_t>> dims{
        {"tokens.values", {max_total_values, channels}},
        {"tokens.offsets", {batch_size + 1}},
        {"filter", {channels, channels, kernel_width}},
    };
    PhysicalOutputs backward = buildBackwardOutputs(
        forward, {"tokens.values"}, std::optional<std::string>{"dy"}, dims);

    std::vector<DataType> backward_input_dtypes(backward.expr->inputs.size(), DataType::FP32);
    for (const NamedInput& named_input : backward.expr->inputs) {
        if (named_input.name == "tokens.offsets") {
            backward_input_dtypes.at(named_input.slot) = DataType::UINT32;
        }
    }
    resolveOutputsDTypesInPlace(backward, backward_input_dtypes);

    ASSERT_EQ(backward.outputs.size(), 1u);
    ASSERT_EQ(backward.outputs.front().name, "tokens.values_grad");
    ASSERT_TRUE(backward.outputs.front().materialization.storage_dtype.has_value());
    EXPECT_EQ(backward.outputs.front().materialization.storage_dtype.value(), DataType::FP32);
    EXPECT_FALSE(containsOp(backward, ExprOp::CAST));

    // Force a physical output-storage conversion without changing the mathematical
    // backward graph. M3 must append this CAST only in the compiler-local view, and
    // T9C must keep it in the retained padded representation after dgrad.
    backward.outputs.front().materialization.storage_dtype = DataType::FP16;

    FusedEquation equation = FusedEquation::compile(backward, 0);
    StampedExecutionPlan plan = equation.stamp({{"tokens.values", gpu_x},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter", gpu_filter},
                                                {"dy", gpu_dy}},
                                               stream);

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausalBackwardData"), 1);
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPointwise"), 1)
        << "compiler-local terminal CAST must remain active-local in the retained padded region";
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 1);
    ASSERT_EQ(plan.paddedRaggedPointwisePreStampedWidthCounts().size(), 1u);
    EXPECT_GT(plan.paddedRaggedPointwisePreStampedWidthCounts().front(), 0u);
    EXPECT_EQ(plan.output("tokens.values_grad").getDescriptor().getDataType(), DataType::FP16);

    // Compilation/stamping must not write the physical CAST back into AutoDiff's
    // persistent mathematical expression.
    EXPECT_FALSE(containsOp(backward, ExprOp::CAST));
}

TEST(RaggedExpression, CausalConv1dT9CRetainsTerminalActiveLocalBackwardAfterDgrad) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel_width = 2;
    constexpr float inactive_sentinel = 7171.0F;
    const std::vector<uint32_t> offsets32{0, 2, 5};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    std::vector<float> x(max_total_values * channels, 9999.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        x[value * channels + 0] = static_cast<float>(static_cast<int>(value) - 2) * 0.25F + 0.0625F;
        x[value * channels + 1] = static_cast<float>(3 - static_cast<int>(value)) * 0.125F + 0.03125F;
    }
    const std::vector<float> filter{0.5F, -0.25F, 0.25F, 0.5F,
                                    -0.5F, 0.25F, 0.5F, -0.25F};
    std::vector<float> dy(max_total_values * channels, -9999.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        dy[value * channels + 0] = static_cast<float>(value + 1) * 0.125F;
        dy[value * channels + 1] = -static_cast<float>(value + 2) * 0.0625F;
    }

    enum class ActivationKind { Tanh, Sigmoid, CastRoundTrip };
    for (ActivationKind activation_kind : {ActivationKind::Tanh, ActivationKind::Sigmoid, ActivationKind::CastRoundTrip}) {
        SCOPED_TRACE(static_cast<int>(activation_kind));
        Stream stream(0);
        Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
        RowPartitionRuntime partition(
            gpu_offsets,
            RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
        partition.setHostOffsets(offsets);
        Tensor gpu_x = makeGpuTensor<float>({max_total_values, channels}, x, stream);
        Tensor gpu_filter = makeGpuTensor<float>({channels, channels, kernel_width}, filter, stream);
        Tensor gpu_dy = makeGpuTensor<float>({max_total_values, channels}, dy, stream);

        const RaggedExpression input = RaggedExpression::input(
            "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
        RaggedExpression activated;
        switch (activation_kind) {
            case ActivationKind::Tanh:
                activated = input.mapValues([](const Expression& values) { return values.tanh(); });
                break;
            case ActivationKind::Sigmoid:
                activated = input.mapValues([](const Expression& values) { return values.sigmoid(); });
                break;
            case ActivationKind::CastRoundTrip:
                activated = input.cast(DataType::FP16).cast(DataType::FP32);
                break;
        }
        const Expression filter_expr = Expression::input("filter", std::nullopt, DataType::FP32);
        const RaggedExpression output =
            activated.causalConv1d(filter_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
        FusedEquation forward = FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
        StampedExecutionPlan forward_plan = forward.stamp({{"tokens.values", gpu_x},
                                                           {"tokens.offsets", gpu_offsets},
                                                           {"filter", gpu_filter}},
                                                          stream);
        const std::vector<std::string> forward_stage_names = forward_plan.stageKindNames();
        EXPECT_EQ(std::count(forward_stage_names.begin(), forward_stage_names.end(), "PaddedRaggedPack"), 1);
        EXPECT_EQ(std::count(forward_stage_names.begin(), forward_stage_names.end(), "PaddedRaggedPointwise"), 1);
        EXPECT_EQ(std::count(forward_stage_names.begin(), forward_stage_names.end(), "RaggedConv1dCausal"), 1);
        EXPECT_EQ(std::count(forward_stage_names.begin(), forward_stage_names.end(), "PaddedRaggedUnpack"), 1);

        FusedEquation backward = forward.compileBackward({"tokens.values"}, "dy");
        Tensor dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, channels}));
        dx.fill(inactive_sentinel, stream);
        StampedExecutionPlan backward_plan = backward.stamp({{"tokens.values", gpu_x},
                                                             {"tokens.offsets", gpu_offsets},
                                                             {"filter", gpu_filter},
                                                             {"dy", gpu_dy}},
                                                            stream,
                                                            {},
                                                            {{"tokens.values_grad", dx}});
        const std::vector<std::string> backward_stage_names = backward_plan.stageKindNames();
        EXPECT_EQ(std::count(backward_stage_names.begin(), backward_stage_names.end(), "RaggedConv1dCausalBackwardData"), 1);
        EXPECT_EQ(std::count(backward_stage_names.begin(), backward_stage_names.end(), "PaddedRaggedPointwise"), 1);
        EXPECT_EQ(std::count(backward_stage_names.begin(), backward_stage_names.end(), "PaddedRaggedUnpack"), 1);
        ASSERT_EQ(backward_plan.paddedRaggedPointwisePreStampedWidthCounts().size(), 1u);
        EXPECT_GT(backward_plan.paddedRaggedPointwisePreStampedWidthCounts().front(), 0u);

        backward_plan.run();
        stream.synchronize();
        std::vector<float> local_grad = cpuRaggedCausalConv1dDgrad(
            dy, offsets, filter, max_total_values, channels, channels, kernel_width, 1, 0.0F);
        if (activation_kind != ActivationKind::CastRoundTrip) {
            for (uint64_t value = 0; value < offsets.back(); ++value) {
                for (uint64_t channel = 0; channel < channels; ++channel) {
                    const size_t index = value * channels + channel;
                    if (activation_kind == ActivationKind::Tanh) {
                        const float t = std::tanh(x[index]);
                        local_grad[index] *= 1.0F - t * t;
                    } else {
                        const float sigmoid = 1.0F / (1.0F + std::exp(-x[index]));
                        local_grad[index] *= sigmoid * (1.0F - sigmoid);
                    }
                }
            }
        }
        for (uint64_t value = offsets.back(); value < max_total_values; ++value) {
            for (uint64_t channel = 0; channel < channels; ++channel) {
                local_grad[value * channels + channel] = inactive_sentinel;
            }
        }
        const std::vector<float> actual = copyToCpuValues(backward_plan.output("tokens.values_grad"), stream);
        expectNear(actual, local_grad, activation_kind == ActivationKind::CastRoundTrip ? 1.0e-3F : 2.0e-5F);
    }
}

TEST(RaggedExpression, CausalConv1dT9DParameterReductionsExitRetainedSpineAtLogicalBoundary) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel_width = 2;
    constexpr float inactive_dx_sentinel = 8181.0F;
    const std::vector<uint32_t> offsets32{0, 3, 3, 7};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> x(max_total_values * channels, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        x[value * channels + 0] = static_cast<float>(static_cast<int>(value) - 3) * 0.20F + 0.05F;
        x[value * channels + 1] = static_cast<float>(4 - static_cast<int>(value)) * 0.15F - 0.025F;
    }
    const std::vector<float> filter1{0.5F, -0.25F, 0.125F, 0.75F,
                                     -0.4F, 0.3F, 0.6F, -0.2F};
    const std::vector<float> filter2{-0.25F, 0.5F, 0.4F, 0.125F,
                                     0.75F, -0.5F, -0.3F, 0.6F};
    const std::vector<float> scale{1.25F, -0.75F};
    const std::vector<float> bias{-0.10F, 0.20F};
    std::vector<float> dy(max_total_values * channels, std::numeric_limits<float>::infinity());
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        dy[value * channels + 0] = 0.10F * static_cast<float>(value + 1);
        dy[value * channels + 1] = -0.075F * static_cast<float>(value + 2);
    }

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);
    Tensor gpu_x = makeGpuTensor<float>({max_total_values, channels}, x, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel_width}, filter2, stream);
    Tensor gpu_scale = makeGpuTensor<float>({channels}, scale, stream);
    Tensor gpu_bias = makeGpuTensor<float>({channels}, bias, stream);
    Tensor gpu_dy = makeGpuTensor<float>({max_total_values, channels}, dy, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression scale_expr = Expression::input("scale", std::nullopt, DataType::FP32);
    const Expression bias_expr = Expression::input("bias", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression activated = hidden.mapValues([&](const Expression& values) {
        return (values * scale_expr + bias_expr).relu();
    });
    const RaggedExpression output =
        activated.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation forward =
        FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"tokens.values", "scale", "bias"}, "dy");
    Tensor dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, channels}));
    dx.fill(inactive_dx_sentinel, stream);
    StampedExecutionPlan plan = backward.stamp({{"tokens.values", gpu_x},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter1", gpu_filter1},
                                                {"filter2", gpu_filter2},
                                                {"scale", gpu_scale},
                                                {"bias", gpu_bias},
                                                {"dy", gpu_dy}},
                                               stream,
                                               {},
                                               {{"tokens.values_grad", dx}});

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausalBackwardData"), 2);
    EXPECT_GE(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPointwise"), 1)
        << "the dX spine must remain retained across the active-local scale/bias/ReLU backward";
    EXPECT_GE(std::count(stage_names.begin(), stage_names.end(), "SegmentedReduction"), 2)
        << "channel scale and bias gradients must reduce only logical ragged values";
    EXPECT_GE(std::count(stage_names.begin(), stage_names.end(), "Reduction"), 2);
    EXPECT_GE(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 2)
        << "parameter reductions must exit retained storage through packed logical values while dX has its own final exit";

    // Every segmented parameter reduction in this graph must be downstream of
    // an explicit retained->packed exit. The exit may feed a small packed
    // pointwise contribution stage before the segmented sum, so check ancestry
    // rather than requiring the unpack to be the reduction's direct parent.
    const std::vector<std::vector<uint32_t>> dependencies = plan.stageDependencyIndices();
    ASSERT_EQ(dependencies.size(), stage_names.size());
    for (uint32_t reduction_idx = 0; reduction_idx < stage_names.size(); ++reduction_idx) {
        if (stage_names[reduction_idx] != "SegmentedReduction") {
            continue;
        }
        bool found_unpack_ancestor = false;
        std::vector<uint32_t> pending = dependencies[reduction_idx];
        std::set<uint32_t> visited;
        while (!pending.empty()) {
            const uint32_t ancestor = pending.back();
            pending.pop_back();
            if (ancestor >= stage_names.size() || !visited.insert(ancestor).second) {
                continue;
            }
            if (stage_names[ancestor] == "PaddedRaggedUnpack") {
                found_unpack_ancestor = true;
                break;
            }
            pending.insert(pending.end(), dependencies[ancestor].begin(), dependencies[ancestor].end());
        }
        EXPECT_TRUE(found_unpack_ancestor)
            << "segmented reduction stage " << reduction_idx
            << " must consume a logical packed view, never retained padded storage";
    }

    plan.run();
    stream.synchronize();

    const std::vector<float> hidden_cpu = cpuRaggedCausalConv1d(
        x, offsets, filter1, max_total_values, channels, channels, kernel_width, 1, 0.0F);
    std::vector<float> scaled(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const size_t index = value * channels + channel;
            scaled[index] = hidden_cpu[index] * scale[channel] + bias[channel];
        }
    }
    const std::vector<float> grad_activated = cpuRaggedCausalConv1dDgrad(
        dy, offsets, filter2, max_total_values, channels, channels, kernel_width, 1, 0.0F);
    std::vector<float> grad_scaled(max_total_values * channels, 0.0F);
    std::vector<float> expected_scale_grad(channels, 0.0F);
    std::vector<float> expected_bias_grad(channels, 0.0F);
    std::vector<float> grad_hidden(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const size_t index = value * channels + channel;
            grad_scaled[index] = scaled[index] > 0.0F ? grad_activated[index] : 0.0F;
            expected_scale_grad[channel] += grad_scaled[index] * hidden_cpu[index];
            expected_bias_grad[channel] += grad_scaled[index];
            grad_hidden[index] = grad_scaled[index] * scale[channel];
        }
    }
    const std::vector<float> expected_dx = cpuRaggedCausalConv1dDgrad(
        grad_hidden,
        offsets,
        filter1,
        max_total_values,
        channels,
        channels,
        kernel_width,
        1,
        inactive_dx_sentinel);

    const std::vector<float> actual_dx = copyToCpuValues(plan.output("tokens.values_grad"), stream);
    const std::vector<float> actual_scale_grad = copyToCpuValues(plan.output("scale_grad"), stream);
    const std::vector<float> actual_bias_grad = copyToCpuValues(plan.output("bias_grad"), stream);
    expectNear(actual_dx, expected_dx, 2.0e-5F);
    expectNear(actual_scale_grad, expected_scale_grad, 2.0e-5F);
    expectNear(actual_bias_grad, expected_bias_grad, 2.0e-5F);
    for (float value : actual_scale_grad) {
        EXPECT_TRUE(std::isfinite(value));
    }
    for (float value : actual_bias_grad) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(RaggedExpression, CausalConv1dT9DAllEmptyParameterReductionsProduceExactZero) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel_width = 2;
    constexpr float inactive_dx_sentinel = -6262.0F;
    const std::vector<uint32_t> offsets32{0, 0, 0, 0};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> x(max_total_values * channels, std::numeric_limits<float>::quiet_NaN());
    const std::vector<float> dy(max_total_values * channels, std::numeric_limits<float>::infinity());
    const std::vector<float> filter1{0.5F, -0.25F, 0.125F, 0.75F,
                                     -0.4F, 0.3F, 0.6F, -0.2F};
    const std::vector<float> filter2{-0.25F, 0.5F, 0.4F, 0.125F,
                                     0.75F, -0.5F, -0.3F, 0.6F};
    const std::vector<float> scale{1.25F, -0.75F};
    const std::vector<float> bias{-0.10F, 0.20F};

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);
    Tensor gpu_x = makeGpuTensor<float>({max_total_values, channels}, x, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel_width}, filter2, stream);
    Tensor gpu_scale = makeGpuTensor<float>({channels}, scale, stream);
    Tensor gpu_bias = makeGpuTensor<float>({channels}, bias, stream);
    Tensor gpu_dy = makeGpuTensor<float>({max_total_values, channels}, dy, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression scale_expr = Expression::input("scale", std::nullopt, DataType::FP32);
    const Expression bias_expr = Expression::input("bias", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression activated = hidden.mapValues([&](const Expression& values) {
        return (values * scale_expr + bias_expr).relu();
    });
    const RaggedExpression output =
        activated.causalConv1d(filter2_expr, channels, kernel_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation forward =
        FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"tokens.values", "scale", "bias"}, "dy");
    Tensor dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, channels}));
    dx.fill(inactive_dx_sentinel, stream);
    StampedExecutionPlan plan = backward.stamp({{"tokens.values", gpu_x},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter1", gpu_filter1},
                                                {"filter2", gpu_filter2},
                                                {"scale", gpu_scale},
                                                {"bias", gpu_bias},
                                                {"dy", gpu_dy}},
                                               stream,
                                               {},
                                               {{"tokens.values_grad", dx}});

    const std::vector<std::string> stage_names = plan.stageKindNames();
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausalBackwardData"), 2);
    EXPECT_GE(std::count(stage_names.begin(), stage_names.end(), "SegmentedReduction"), 2);

    plan.run();
    stream.synchronize();
    const std::vector<float> actual_dx = copyToCpuValues(plan.output("tokens.values_grad"), stream);
    for (float value : actual_dx) {
        EXPECT_EQ(value, inactive_dx_sentinel);
    }
    for (const std::string& output_name : {std::string("scale_grad"), std::string("bias_grad")}) {
        const std::vector<float> actual = copyToCpuValues(plan.output(output_name), stream);
        ASSERT_EQ(actual.size(), channels);
        for (float value : actual) {
            EXPECT_EQ(value, 0.0F);
            EXPECT_FALSE(std::signbit(value));
        }
    }
    for (const RaggedConv1dStageDiagnostic& diagnostic : plan.raggedConv1dStageDiagnostics()) {
        EXPECT_EQ(diagnostic.active_values, 0u);
        EXPECT_EQ(diagnostic.selected_width_capacity, 0u);
    }
}


TEST(RaggedExpression, CausalConv1dT9EThreeConvBackwardRegionRetainsSpineAndMatchesPackedReference) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel1_width = 2;
    constexpr uint64_t kernel2_width = 3;
    constexpr uint64_t kernel3_width = 2;
    constexpr float inactive_dx_sentinel = -7373.0F;
    const std::vector<uint32_t> offsets32{0, 3, 3, 7};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> x(max_total_values * channels, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        x[value * channels + 0] = static_cast<float>(static_cast<int>(value) - 3) * 0.20F + 0.05F;
        x[value * channels + 1] = static_cast<float>(4 - static_cast<int>(value)) * 0.15F - 0.025F;
    }
    const std::vector<float> filter1{0.50F, -0.25F, 0.125F, 0.75F,
                                     -0.40F, 0.30F, 0.60F, -0.20F};
    const std::vector<float> filter2{0.25F, -0.50F, 0.125F,
                                     0.40F, 0.30F, -0.20F,
                                     -0.35F, 0.20F, 0.45F,
                                     0.15F, -0.25F, 0.50F};
    const std::vector<float> filter3{-0.25F, 0.50F, 0.40F, 0.125F,
                                     0.75F, -0.50F, -0.30F, 0.60F};
    const std::vector<float> bias{-0.10F, 0.20F};
    std::vector<float> dy(max_total_values * channels, std::numeric_limits<float>::infinity());
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        dy[value * channels + 0] = 0.10F * static_cast<float>(value + 1);
        dy[value * channels + 1] = -0.075F * static_cast<float>(value + 2);
    }

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);
    Tensor gpu_x = makeGpuTensor<float>({max_total_values, channels}, x, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel1_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel2_width}, filter2, stream);
    Tensor gpu_filter3 = makeGpuTensor<float>({channels, channels, kernel3_width}, filter3, stream);
    Tensor gpu_bias = makeGpuTensor<float>({channels}, bias, stream);
    Tensor gpu_dy = makeGpuTensor<float>({max_total_values, channels}, dy, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression filter3_expr = Expression::input("filter3", std::nullopt, DataType::FP32);
    const Expression bias_expr = Expression::input("bias", std::nullopt, DataType::FP32);

    const RaggedExpression conv1 =
        input.causalConv1d(filter1_expr, channels, kernel1_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression activation1 = conv1.mapValues([&](const Expression& values) {
        return (values + bias_expr).relu();
    });
    const RaggedExpression conv2 =
        activation1.causalConv1d(filter2_expr, channels, kernel2_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression activation2 = conv2.mapValues([](const Expression& values) {
        return values.tanh();
    });
    const RaggedExpression output =
        activation2.causalConv1d(filter3_expr, channels, kernel3_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation forward =
        FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> forward_inputs{{"tokens.values", gpu_x},
                                                                 {"tokens.offsets", gpu_offsets},
                                                                 {"filter1", gpu_filter1},
                                                                 {"filter2", gpu_filter2},
                                                                 {"filter3", gpu_filter3},
                                                                 {"bias", gpu_bias}};
    StampedExecutionPlan forward_plan = forward.stamp(forward_inputs, stream);
    EXPECT_EQ(forward_plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedPointwise",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedPointwise",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}))
        << "the realistic forward chain must remain one retained padded region";

    FusedEquation backward =
        forward.compileBackward({"tokens.values", "filter1", "bias", "filter2", "filter3"}, "dy");

    Tensor dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, channels}));
    dx.fill(inactive_dx_sentinel, stream);
    StampedExecutionPlan plan = backward.stamp({{"tokens.values", gpu_x},
                                                {"tokens.offsets", gpu_offsets},
                                                {"filter1", gpu_filter1},
                                                {"filter2", gpu_filter2},
                                                {"filter3", gpu_filter3},
                                                {"bias", gpu_bias},
                                                {"dy", gpu_dy}},
                                               stream,
                                               {},
                                               {{"tokens.values_grad", dx}});

    const std::vector<std::string> stage_names = plan.stageKindNames();
    const std::vector<std::vector<uint32_t>> dependencies = plan.stageDependencyIndices();
    ASSERT_EQ(stage_names.size(), dependencies.size());
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausalBackwardData"), 3)
        << "the complete three-convolution dX spine must use retained dgrad at every convolution";
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "RaggedConv1dCausalBackwardFilter"), 3)
        << "T9E integrates T9B wgrad for every convolution in the realistic chain";
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "SegmentedReduction"), 1)
        << "the single channel bias has exactly one logical ragged reduction boundary";
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "Reduction"), 1)
        << "the channel-bias segmented result has one dense batch reduction";
    EXPECT_EQ(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedUnpack"), 2)
        << "only the bias-reduction branch and the final public dX may exit retained storage";
    EXPECT_GE(std::count(stage_names.begin(), stage_names.end(), "PaddedRaggedPointwise"), 2)
        << "ReLU/bias and tanh backward must remain in retained pointwise regions";

    auto ancestor_set = [&](uint32_t stage_index) {
        std::set<uint32_t> ancestors;
        std::vector<uint32_t> pending = dependencies.at(stage_index);
        while (!pending.empty()) {
            const uint32_t current = pending.back();
            pending.pop_back();
            if (!ancestors.insert(current).second) {
                continue;
            }
            if (current >= dependencies.size()) {
                throw std::runtime_error("T9E stage dependency index is out of range.");
            }
            pending.insert(pending.end(), dependencies[current].begin(), dependencies[current].end());
        }
        return ancestors;
    };
    auto count_kind_in = [&](const std::set<uint32_t>& stages, const std::string& kind) {
        return std::count_if(stages.begin(), stages.end(), [&](uint32_t stage_index) {
            return stage_names.at(stage_index) == kind;
        });
    };

    std::vector<uint32_t> unpack_indices;
    std::vector<uint32_t> segmented_reduction_indices;
    for (uint32_t stage_index = 0; stage_index < stage_names.size(); ++stage_index) {
        if (stage_names[stage_index] == "PaddedRaggedUnpack") {
            unpack_indices.push_back(stage_index);
        } else if (stage_names[stage_index] == "SegmentedReduction") {
            segmented_reduction_indices.push_back(stage_index);
        }
    }
    ASSERT_EQ(unpack_indices.size(), 2u);
    ASSERT_EQ(segmented_reduction_indices.size(), 1u);

    // Identify the public dX exit structurally: it is the only unpack whose
    // ancestry contains all three dgrad stages. No earlier unpack may occur on
    // that retained gradient spine.
    std::optional<uint32_t> dx_unpack_index;
    std::optional<uint32_t> bias_unpack_index;
    for (uint32_t unpack_index : unpack_indices) {
        const std::set<uint32_t> ancestors = ancestor_set(unpack_index);
        const int dgrad_ancestors = count_kind_in(ancestors, "RaggedConv1dCausalBackwardData");
        if (dgrad_ancestors == 3) {
            ASSERT_FALSE(dx_unpack_index.has_value());
            dx_unpack_index = unpack_index;
        } else {
            ASSERT_FALSE(bias_unpack_index.has_value());
            bias_unpack_index = unpack_index;
        }
    }
    ASSERT_TRUE(dx_unpack_index.has_value());
    ASSERT_TRUE(bias_unpack_index.has_value());

    const std::set<uint32_t> dx_ancestors = ancestor_set(*dx_unpack_index);
    EXPECT_EQ(count_kind_in(dx_ancestors, "RaggedConv1dCausalBackwardData"), 3);
    EXPECT_EQ(count_kind_in(dx_ancestors, "PaddedRaggedUnpack"), 0)
        << "there must be no representation boundary between compatible backward operators";
    EXPECT_EQ(count_kind_in(dx_ancestors, "SegmentedReduction"), 0)
        << "the parameter-reduction branch must remain a sibling, not enter the dX spine";

    const uint32_t segmented_reduction_index = segmented_reduction_indices.front();
    const std::set<uint32_t> reduction_ancestors = ancestor_set(segmented_reduction_index);
    EXPECT_TRUE(reduction_ancestors.contains(*bias_unpack_index))
        << "bias reduction must consume the explicit retained-to-packed logical exit";
    EXPECT_FALSE(reduction_ancestors.contains(*dx_unpack_index));
    EXPECT_EQ(count_kind_in(reduction_ancestors, "PaddedRaggedUnpack"), 1)
        << "the bias branch must cross exactly one representation boundary before reduction";
    EXPECT_EQ(count_kind_in(reduction_ancestors, "RaggedConv1dCausalBackwardData"), 2)
        << "bias gradient branches after conv2 dgrad; conv1 dgrad remains on the sibling dX spine";

    forward_plan.run();
    plan.run();
    stream.synchronize();

    // Independent packed CPU forward/backward reference for the entire chain.
    const std::vector<float> conv1_cpu = cpuRaggedCausalConv1d(
        x, offsets, filter1, max_total_values, channels, channels, kernel1_width, 1, 0.0F);
    std::vector<float> activation1_cpu(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const size_t index = value * channels + channel;
            activation1_cpu[index] = std::max(conv1_cpu[index] + bias[channel], 0.0F);
        }
    }
    const std::vector<float> conv2_cpu = cpuRaggedCausalConv1d(
        activation1_cpu, offsets, filter2, max_total_values, channels, channels, kernel2_width, 1, 0.0F);
    std::vector<float> activation2_cpu(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const size_t index = value * channels + channel;
            activation2_cpu[index] = std::tanh(conv2_cpu[index]);
        }
    }
    const std::vector<float> expected_y = cpuRaggedCausalConv1d(
        activation2_cpu, offsets, filter3, max_total_values, channels, channels, kernel3_width, 1, 0.0F);
    const std::vector<float> actual_y = copyToCpuValues(forward_plan.output("y"), stream);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const size_t index = value * channels + channel;
            EXPECT_NEAR(actual_y[index], expected_y[index], 3.0e-5F) << "forward active index " << index;
        }
    }

    const std::vector<float> expected_filter3_grad = cpuRaggedCausalConv1dWgrad(
        activation2_cpu, dy, offsets, max_total_values, channels, channels, kernel3_width, 1);
    const std::vector<float> grad_activation2 = cpuRaggedCausalConv1dDgrad(
        dy, offsets, filter3, max_total_values, channels, channels, kernel3_width, 1, 0.0F);
    std::vector<float> grad_conv2(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const size_t index = value * channels + channel;
            const float t = activation2_cpu[index];
            grad_conv2[index] = grad_activation2[index] * (1.0F - t * t);
        }
    }
    const std::vector<float> expected_filter2_grad = cpuRaggedCausalConv1dWgrad(
        activation1_cpu, grad_conv2, offsets, max_total_values, channels, channels, kernel2_width, 1);
    const std::vector<float> grad_activation1 = cpuRaggedCausalConv1dDgrad(
        grad_conv2, offsets, filter2, max_total_values, channels, channels, kernel2_width, 1, 0.0F);
    std::vector<float> grad_conv1(max_total_values * channels, 0.0F);
    std::vector<float> expected_bias_grad(channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const size_t index = value * channels + channel;
            grad_conv1[index] =
                (conv1_cpu[index] + bias[channel] > 0.0F) ? grad_activation1[index] : 0.0F;
            expected_bias_grad[channel] += grad_conv1[index];
        }
    }
    const std::vector<float> expected_filter1_grad = cpuRaggedCausalConv1dWgrad(
        x, grad_conv1, offsets, max_total_values, channels, channels, kernel1_width, 1);
    const std::vector<float> expected_dx = cpuRaggedCausalConv1dDgrad(
        grad_conv1,
        offsets,
        filter1,
        max_total_values,
        channels,
        channels,
        kernel1_width,
        1,
        inactive_dx_sentinel);

    expectNear(copyToCpuValues(plan.output("tokens.values_grad"), stream), expected_dx, 3.0e-5F);
    expectNear(copyToCpuValues(plan.output("filter1_grad"), stream), expected_filter1_grad, 5.0e-5F);
    expectNear(copyToCpuValues(plan.output("filter2_grad"), stream), expected_filter2_grad, 5.0e-5F);
    expectNear(copyToCpuValues(plan.output("filter3_grad"), stream), expected_filter3_grad, 5.0e-5F);
    expectNear(copyToCpuValues(plan.output("bias_grad"), stream), expected_bias_grad, 3.0e-5F);

    for (const std::string& output_name : {std::string("filter1_grad"),
                                           std::string("filter2_grad"),
                                           std::string("filter3_grad"),
                                           std::string("bias_grad")}) {
        for (float value : copyToCpuValues(plan.output(output_name), stream)) {
            EXPECT_TRUE(std::isfinite(value)) << output_name;
        }
    }
    const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
    size_t backward_conv_diagnostics = 0;
    std::optional<uint64_t> backward_selected_width;
    for (const RaggedConv1dStageDiagnostic& diagnostic : diagnostics) {
        ASSERT_LT(diagnostic.stage_index, stage_names.size());
        const std::string& kind = stage_names[diagnostic.stage_index];
        if (kind != "RaggedConv1dCausalBackwardData" && kind != "RaggedConv1dCausalBackwardFilter") {
            continue;
        }
        ++backward_conv_diagnostics;
        EXPECT_EQ(diagnostic.active_values, offsets.back());
        EXPECT_GT(diagnostic.selected_width_capacity, 0u);
        if (!backward_selected_width.has_value()) {
            backward_selected_width = diagnostic.selected_width_capacity;
        } else {
            EXPECT_EQ(diagnostic.selected_width_capacity, *backward_selected_width)
                << "all retained backward convolution consumers share the partition-selected W";
        }
        EXPECT_EQ(diagnostic.prebuilt_cudnn_plan_count, diagnostic.width_capacity_count);
        EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u);
    }
    EXPECT_EQ(backward_conv_diagnostics, 6u);
}


TEST(RaggedExpression, CausalConv1dT9FRuntimeWidthFamilyRemainsFiniteAndAllocationStable) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 64;
    constexpr uint64_t max_values_per_row = 32;
    constexpr uint64_t channels = 2;
    constexpr uint64_t kernel1_width = 3;
    constexpr uint64_t kernel2_width = 2;
    constexpr size_t stress_iterations = 2048;

    // The explicit transition family exercises every placement-built width and
    // the all-empty path, then repeats for thousands of executions:
    //     8 -> 16 -> 32 -> 8 -> 0 -> ...
    struct RuntimePartition {
        std::vector<uint32_t> offsets32;
        std::vector<uint64_t> offsets;
        uint64_t expected_width;
    };
    const std::vector<RuntimePartition> transitions{
        {{0, 8, 8}, {0, 8, 8}, 8},
        {{0, 16, 16}, {0, 16, 16}, 16},
        {{0, 32, 32}, {0, 32, 32}, 32},
        {{0, 5, 8}, {0, 5, 8}, 8},
        {{0, 0, 0}, {0, 0, 0}, 0},
    };

    std::vector<float> x(max_total_values * channels, 0.0F);
    std::vector<float> dy(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < max_total_values; ++value) {
        x[value * channels + 0] = 0.01F * static_cast<float>(value + 1);
        x[value * channels + 1] = -0.0075F * static_cast<float>(value + 3);
        dy[value * channels + 0] = 0.005F * static_cast<float>(value + 2);
        dy[value * channels + 1] = -0.003F * static_cast<float>(value + 5);
    }
    const std::vector<float> filter1{
        0.25F, -0.50F, 0.75F,
        -0.20F, 0.10F, 0.30F,
        0.40F, -0.35F, 0.15F,
        0.05F, 0.60F, -0.45F,
    };
    const std::vector<float> filter2{
        0.50F, -0.25F,
        0.20F, 0.30F,
        -0.40F, 0.10F,
        0.35F, 0.15F,
    };

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, transitions.front().offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_values_per_row));
    partition.setHostOffsets(transitions.front().offsets);
    Tensor gpu_x = makeGpuTensor<float>({max_total_values, channels}, x, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel1_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel2_width}, filter2, stream);
    Tensor gpu_dy = makeGpuTensor<float>({max_total_values, channels}, dy, stream);

    const RaggedExpression input = RaggedExpression::input(
        "tokens",
        RaggedTensorDescriptor(DataType::FP32,
                               {channels},
                               batch_size,
                               max_total_values,
                               max_values_per_row,
                               DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel1_width, 1, DataType::FP32, DataType::FP32).relu();
    const RaggedExpression output =
        hidden.causalConv1d(filter2_expr, channels, kernel2_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation forward =
        FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
    StampedExecutionPlan forward_plan = forward.stamp({{"tokens.values", gpu_x},
                                                       {"tokens.offsets", gpu_offsets},
                                                       {"filter1", gpu_filter1},
                                                       {"filter2", gpu_filter2}},
                                                      stream);

    FusedEquation backward = forward.compileBackward({"tokens.values", "filter1", "filter2"}, "dy");
    StampedExecutionPlan backward_plan = backward.stamp({{"tokens.values", gpu_x},
                                                         {"tokens.offsets", gpu_offsets},
                                                         {"filter1", gpu_filter1},
                                                         {"filter2", gpu_filter2},
                                                         {"dy", gpu_dy}},
                                                        stream);

    const std::vector<RaggedConv1dStageDiagnostic> forward_baseline = forward_plan.raggedConv1dStageDiagnostics();
    const std::vector<RaggedConv1dStageDiagnostic> backward_baseline = backward_plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(forward_baseline.size(), 2u);
    ASSERT_FALSE(backward_baseline.empty());

    const std::vector<size_t> forward_pointwise_width_counts =
        forward_plan.paddedRaggedPointwisePreStampedWidthCounts();
    const std::vector<size_t> backward_pointwise_width_counts =
        backward_plan.paddedRaggedPointwisePreStampedWidthCounts();
    ASSERT_EQ(forward_pointwise_width_counts.size(), 1u);
    ASSERT_FALSE(backward_pointwise_width_counts.empty());
    EXPECT_EQ(forward_pointwise_width_counts.front(), 3u);
    for (size_t count : backward_pointwise_width_counts) {
        EXPECT_EQ(count, 3u);
    }

    auto assert_stamp_time_family = [](const std::vector<RaggedConv1dStageDiagnostic>& diagnostics) {
        for (const RaggedConv1dStageDiagnostic& diagnostic : diagnostics) {
            EXPECT_EQ(diagnostic.width_capacity_count, 3u);
            EXPECT_EQ(diagnostic.prebuilt_cudnn_plan_count, 3u);
            EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u);
            if (diagnostic.cudnn_workspace_bytes == 0) {
                EXPECT_EQ(diagnostic.cudnn_workspace_state_id, 0u);
            } else {
                EXPECT_NE(diagnostic.cudnn_workspace_state_id, 0u);
            }
        }
    };
    assert_stamp_time_family(forward_baseline);
    assert_stamp_time_family(backward_baseline);

    const uint64_t cudnn_preparations_after_stamping = cudnnFrontendExecutablePreparationCountForTests();
    const uint64_t equation_builds_after_stamping = EquationCompiler::compiledEquationBuildCountForTests();
    const size_t equation_cache_entries_after_stamping = EquationCompiler::compiledEquationCacheEntryCountForTests();

    // Stronger than merely checking "no growth": remove the global convolution
    // selection recipes entirely. Every runtime execution below must use only
    // operation-local plans prepared during stamping.
    clearCudnnFrontendConvolutionSelectionCacheForTests();
    ASSERT_EQ(cachedCudnnFrontendConvolutionSelectionCountForTests(), 0u);
    const uint64_t selection_hits_after_clear = cudnnFrontendConvolutionSelectionCacheHitCountForTests();
    const uint64_t selection_misses_after_clear = cudnnFrontendConvolutionSelectionCacheMissCountForTests();

    auto assert_runtime_family =
        [&](const StampedExecutionPlan& plan,
            const std::vector<RaggedConv1dStageDiagnostic>& baseline,
            const std::vector<size_t>& pointwise_width_counts,
            const RuntimePartition& runtime_partition) {
            const std::vector<RaggedConv1dStageDiagnostic> diagnostics = plan.raggedConv1dStageDiagnostics();
            ASSERT_EQ(diagnostics.size(), baseline.size());
            for (size_t stage = 0; stage < diagnostics.size(); ++stage) {
                const RaggedConv1dStageDiagnostic& current = diagnostics[stage];
                const RaggedConv1dStageDiagnostic& stamped = baseline[stage];
                EXPECT_EQ(current.active_values, runtime_partition.offsets.back());
                EXPECT_EQ(current.selected_width_capacity, runtime_partition.expected_width);
                EXPECT_EQ(current.width_capacity_count, stamped.width_capacity_count);
                EXPECT_EQ(current.prebuilt_cudnn_plan_count, stamped.prebuilt_cudnn_plan_count);
                EXPECT_EQ(current.cudnn_workspace_bytes, stamped.cudnn_workspace_bytes);
                EXPECT_EQ(current.cudnn_workspace_state_id, stamped.cudnn_workspace_state_id)
                    << "runtime width selection must never replace the operation-local cuDNN workspace";
                EXPECT_EQ(current.allocated_padded_value_bytes, stamped.allocated_padded_value_bytes)
                    << "retained/sanitation storage is reserved at placement and must never grow";
                EXPECT_EQ(current.explicit_unfold_workspace_bytes, 0u);
            }
            EXPECT_EQ(plan.paddedRaggedPointwisePreStampedWidthCounts(), pointwise_width_counts)
                << "runtime width transitions may select a pre-stamped pointwise invocation but never add one";
        };

    for (size_t iteration = 0; iteration < stress_iterations; ++iteration) {
        const RuntimePartition& runtime_partition = transitions[iteration % transitions.size()];

        overwriteGpuTensor<uint32_t>(gpu_offsets, runtime_partition.offsets32, stream);
        partition.setHostOffsets(runtime_partition.offsets);

        forward_plan.run();
        backward_plan.run();

        assert_runtime_family(forward_plan, forward_baseline, forward_pointwise_width_counts, runtime_partition);
        assert_runtime_family(backward_plan, backward_baseline, backward_pointwise_width_counts, runtime_partition);

        if ((iteration + 1) % 64 == 0) {
            stream.synchronize();
            EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), cudnn_preparations_after_stamping);
            EXPECT_EQ(EquationCompiler::compiledEquationBuildCountForTests(), equation_builds_after_stamping);
            EXPECT_EQ(EquationCompiler::compiledEquationCacheEntryCountForTests(), equation_cache_entries_after_stamping);
            EXPECT_EQ(cachedCudnnFrontendConvolutionSelectionCountForTests(), 0u);
        }
    }
    stream.synchronize();

    EXPECT_EQ(cudnnFrontendExecutablePreparationCountForTests(), cudnn_preparations_after_stamping)
        << "runtime may never prepare another cuDNN executable";
    EXPECT_EQ(EquationCompiler::compiledEquationBuildCountForTests(), equation_builds_after_stamping)
        << "runtime may never compile/materialize another fused CUDA equation";
    EXPECT_EQ(EquationCompiler::compiledEquationCacheEntryCountForTests(), equation_cache_entries_after_stamping)
        << "runtime may never populate a fused-kernel cache";
    EXPECT_EQ(cachedCudnnFrontendConvolutionSelectionCountForTests(), 0u)
        << "runtime may never consult/repopulate the cleared convolution-selection cache";
    EXPECT_EQ(cudnnFrontendConvolutionSelectionCacheHitCountForTests(), selection_hits_after_clear);
    EXPECT_EQ(cudnnFrontendConvolutionSelectionCacheMissCountForTests(), selection_misses_after_clear);
}


TEST(RaggedExpression, CausalConv1dT9GPoisonedFanoutKeepsProducerStorageImmutableAndConsumersCorrect) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 2;
    constexpr uint64_t max_total_values = 8;
    constexpr uint64_t max_values_per_row = 8;
    constexpr uint64_t selected_width = 8;
    constexpr uint64_t input_channels = 2;
    constexpr uint64_t output_channels = 2;
    constexpr uint64_t kernel_width = 3;
    constexpr float packed_inactive_sentinel = -7123.0F;
    constexpr float output_inactive_sentinel = 8451.0F;

    const std::vector<uint32_t> offsets32{0, 2, 5};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());
    const std::vector<float> filter{
        0.50F, -0.25F, 0.75F,
        -0.40F, 0.20F, 0.10F,
        0.30F, 0.60F, -0.50F,
        -0.15F, 0.45F, 0.80F,
    };

    std::vector<float> packed_x(max_total_values * input_channels, packed_inactive_sentinel);
    std::vector<float> packed_dy(max_total_values * output_channels, packed_inactive_sentinel);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        packed_x[value * input_channels + 0] = 0.20F * static_cast<float>(value + 1);
        packed_x[value * input_channels + 1] = -0.125F * static_cast<float>(value + 2);
        packed_dy[value * output_channels + 0] = 0.075F * static_cast<float>(value + 3);
        packed_dy[value * output_channels + 1] = -0.05F * static_cast<float>(value + 1);
    }

    Stream stream(0);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_values_per_row));
    partition.setHostOffsets(offsets);
    Tensor gpu_filter = makeGpuTensor<float>({output_channels, input_channels, kernel_width}, filter, stream);

    const PaddedRaggedSequencePlan x_plan =
        preparePaddedRaggedSequencePlan(partition, input_channels, DataType::FP32, selected_width);
    const PaddedRaggedSequencePlan dy_plan =
        preparePaddedRaggedSequencePlan(partition, output_channels, DataType::FP32, selected_width);
    const PaddedRaggedSequencePlan y_plan =
        preparePaddedRaggedSequencePlan(partition, output_channels, DataType::FP32, selected_width);
    const PaddedRaggedSequencePlan dx_plan =
        preparePaddedRaggedSequencePlan(partition, input_channels, DataType::FP32, selected_width);

    auto padded_x =
        std::make_shared<PaddedRaggedSequence>(x_plan, gpu_offsets, gpuPlacement, selected_width);
    auto padded_dy =
        std::make_shared<PaddedRaggedSequence>(dy_plan, gpu_offsets, gpuPlacement, selected_width);
    auto padded_y =
        std::make_shared<PaddedRaggedSequence>(y_plan, gpu_offsets, gpuPlacement, selected_width);
    auto padded_dx =
        std::make_shared<PaddedRaggedSequence>(dx_plan, gpu_offsets, gpuPlacement, selected_width);

    auto compiled_forward = std::make_shared<CompiledRaggedConv1dCausal>(DataType::FP32,
                                                                         DataType::FP32,
                                                                         DataType::FP32,
                                                                         DataType::FP32,
                                                                         DataType::UINT32,
                                                                         batch_size,
                                                                         max_total_values,
                                                                         max_values_per_row,
                                                                         input_channels,
                                                                         output_channels,
                                                                         kernel_width,
                                                                         1,
                                                                         1);
    auto compiled_dgrad = std::make_shared<CompiledRaggedConv1dCausalBackwardData>(DataType::FP32,
                                                                                   DataType::FP32,
                                                                                   DataType::FP32,
                                                                                   DataType::FP32,
                                                                                   DataType::UINT32,
                                                                                   batch_size,
                                                                                   max_total_values,
                                                                                   max_values_per_row,
                                                                                   input_channels,
                                                                                   output_channels,
                                                                                   kernel_width,
                                                                                   1,
                                                                                   1);
    auto compiled_wgrad = std::make_shared<CompiledRaggedConv1dCausalBackwardFilter>(DataType::FP32,
                                                                                     DataType::FP32,
                                                                                     DataType::FP32,
                                                                                     DataType::FP32,
                                                                                     DataType::UINT32,
                                                                                     batch_size,
                                                                                     max_total_values,
                                                                                     max_values_per_row,
                                                                                     input_channels,
                                                                                     output_channels,
                                                                                     kernel_width,
                                                                                     1,
                                                                                     1);

    Tensor gpu_dw(gpuPlacement, TensorDescriptor(DataType::FP32, {output_channels, input_channels, kernel_width}));
    StampedRaggedConv1dCausal compatible_forward(
        compiled_forward, padded_x, gpu_filter, gpu_offsets, padded_y, stream);
    StampedRaggedConv1dCausalBackwardData sanitizing_dgrad(
        compiled_dgrad, gpu_filter, padded_dy, gpu_offsets, padded_dx, stream);
    StampedRaggedConv1dCausalBackwardFilter sanitizing_wgrad(
        compiled_wgrad, padded_x, padded_dy, gpu_offsets, gpu_dw, stream);

    // Placement-time family construction may inspect every W. Re-establish the
    // actual retained producer state before injecting undefined-tail poison.
    padded_x->reconfigure(x_plan);
    padded_dy->reconfigure(dy_plan);

    auto make_poisoned_retained = [&](uint64_t channels, const std::vector<float>& packed, uint64_t salt) {
        std::vector<float> values(batch_size * channels * selected_width, 0.0F);
        for (uint64_t row = 0; row < batch_size; ++row) {
            const uint64_t begin = offsets[row];
            const uint64_t row_length = offsets[row + 1] - begin;
            for (uint64_t channel = 0; channel < channels; ++channel) {
                for (uint64_t timestep = 0; timestep < selected_width; ++timestep) {
                    const size_t index = (row * channels + channel) * selected_width + timestep;
                    if (timestep < row_length) {
                        values[index] = packed[(begin + timestep) * channels + channel];
                    } else {
                        switch ((index + salt) % 3) {
                            case 0:
                                values[index] = std::numeric_limits<float>::quiet_NaN();
                                break;
                            case 1:
                                values[index] = std::numeric_limits<float>::infinity();
                                break;
                            default:
                                values[index] = -std::numeric_limits<float>::infinity();
                                break;
                        }
                    }
                }
            }
        }
        return values;
    };

    Tensor x_storage = padded_x->getPaddedValuesStorage();
    Tensor dy_storage = padded_dy->getPaddedValuesStorage();
    overwriteGpuTensor<float>(x_storage, make_poisoned_retained(input_channels, packed_x, 0), stream);
    overwriteGpuTensor<float>(dy_storage, make_poisoned_retained(output_channels, packed_dy, 1), stream);
    stream.synchronize();

    const std::vector<float> x_before = copyToCpuValues(x_storage, stream);
    const std::vector<float> dy_before = copyToCpuValues(dy_storage, stream);

    auto expect_storage_unchanged = [&](const Tensor& storage,
                                        const std::vector<float>& before,
                                        const char* producer_name) {
        const std::vector<float> after = copyToCpuValues(storage, stream);
        ASSERT_EQ(after.size(), before.size());
        for (size_t i = 0; i < before.size(); ++i) {
            if (std::isnan(before[i])) {
                EXPECT_TRUE(std::isnan(after[i])) << producer_name << " index " << i;
            } else {
                EXPECT_EQ(after[i], before[i]) << producer_name << " index " << i;
            }
        }
    };
    auto expect_both_producers_unchanged = [&]() {
        expect_storage_unchanged(x_storage, x_before, "retained X producer");
        expect_storage_unchanged(dy_storage, dy_before, "retained dY producer");
    };

    const std::vector<float> expected_y = cpuRaggedCausalConv1d(packed_x,
                                                                 offsets,
                                                                 filter,
                                                                 max_total_values,
                                                                 input_channels,
                                                                 output_channels,
                                                                 kernel_width,
                                                                 1,
                                                                 output_inactive_sentinel);
    const std::vector<float> expected_dx = cpuRaggedCausalConv1dDgrad(packed_dy,
                                                                      offsets,
                                                                      filter,
                                                                      max_total_values,
                                                                      input_channels,
                                                                      output_channels,
                                                                      kernel_width,
                                                                      1,
                                                                      output_inactive_sentinel);
    const std::vector<float> expected_dw = cpuRaggedCausalConv1dWgrad(packed_x,
                                                                      packed_dy,
                                                                      offsets,
                                                                      max_total_values,
                                                                      input_channels,
                                                                      output_channels,
                                                                      kernel_width,
                                                                      1);

    auto verify_compatible_forward = [&]() {
        compatible_forward.runOn(stream);
        Tensor packed_y(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, output_channels}));
        packed_y.fill(output_inactive_sentinel, stream);
        padded_y->unpackTo(packed_y, stream);
        stream.synchronize();
        expectNear(copyToCpuValues(packed_y, stream), expected_y, 1.0e-5F);
        expect_both_producers_unchanged();
    };

    auto verify_incompatible_unpack = [&]() {
        Tensor unpacked_x(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, input_channels}));
        unpacked_x.fill(output_inactive_sentinel, stream);
        padded_x->unpackTo(unpacked_x, stream);
        stream.synchronize();
        std::vector<float> expected = packed_x;
        for (uint64_t i = offsets.back() * input_channels; i < expected.size(); ++i) {
            expected[i] = output_inactive_sentinel;
        }
        EXPECT_EQ(copyToCpuValues(unpacked_x, stream), expected);
        expect_both_producers_unchanged();
    };

    auto verify_dgrad = [&]() {
        sanitizing_dgrad.runOn(stream);
        Tensor packed_dx(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, input_channels}));
        packed_dx.fill(output_inactive_sentinel, stream);
        padded_dx->unpackTo(packed_dx, stream);
        stream.synchronize();
        const std::vector<float> actual = copyToCpuValues(packed_dx, stream);
        expectNear(actual, expected_dx, 1.0e-5F);
        for (uint64_t i = 0; i < offsets.back() * input_channels; ++i) {
            EXPECT_TRUE(std::isfinite(actual[i])) << "active dX index " << i;
        }
        expect_both_producers_unchanged();
    };

    auto verify_wgrad = [&]() {
        gpu_dw.fill(std::numeric_limits<float>::quiet_NaN(), stream);
        sanitizing_wgrad.runOn(stream);
        stream.synchronize();
        const std::vector<float> actual = copyToCpuValues(gpu_dw, stream);
        expectNear(actual, expected_dw, 1.0e-5F);
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_TRUE(std::isfinite(actual[i])) << "dW index " << i;
        }
        expect_both_producers_unchanged();
    };

    // One retained X producer has three semantically different consumers:
    //   * compatible forward Conv1D, which may observe undefined tails only at
    //     output positions that are themselves inactive;
    //   * an incompatible packed exit adapter;
    //   * wgrad, which must sanitize privately.
    // The retained dY producer is shared by the independently sanitizing dgrad
    // and wgrad consumers. Exercise two different consumer orders to make the
    // ownership rule independent of fanout scheduling order.
    verify_compatible_forward();
    verify_incompatible_unpack();
    verify_dgrad();
    verify_wgrad();

    verify_wgrad();
    verify_dgrad();
    verify_incompatible_unpack();
    verify_compatible_forward();

    expect_both_producers_unchanged();
    EXPECT_EQ(compatible_forward.diagnostic().selected_width_capacity, selected_width);
    EXPECT_EQ(sanitizing_dgrad.diagnostic().selected_width_capacity, selected_width);
    EXPECT_EQ(sanitizing_wgrad.diagnostic().selected_width_capacity, selected_width);
}

TEST(RaggedExpression, CausalConv1dT10RetainedTrainingGateCoversDtypesGroupedDepthwiseAndMultilayerBackward) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t max_values_per_row = 8;
    constexpr uint64_t kernel1_width = 2;
    constexpr uint64_t kernel2_width = 3;
    constexpr float inactive_dx_sentinel = -17.0F;
    const std::vector<uint32_t> offsets32{0, 3, 3, 7};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    struct ProductionCase {
        DataType dtype;
        uint64_t channels;
        uint64_t groups;
        float tolerance;
        const char* label;
    };
    const std::vector<ProductionCase> cases{
        {DataType::FP16, 2, 1, 6.0e-2F, "fp16-ordinary"},
        {DataType::BF16, 4, 2, 1.2e-1F, "bf16-grouped"},
        {DataType::FP32, 4, 4, 4.0e-4F, "fp32-depthwise"},
    };

    for (const ProductionCase& production_case : cases) {
        SCOPED_TRACE(production_case.label);
        const uint64_t channels = production_case.channels;
        const uint64_t groups = production_case.groups;
        const uint64_t channels_per_group = channels / groups;

        std::vector<float> x(max_total_values * channels, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> dy(max_total_values * channels, std::numeric_limits<float>::infinity());
        for (uint64_t value = 0; value < offsets.back(); ++value) {
            for (uint64_t channel = 0; channel < channels; ++channel) {
                x[value * channels + channel] =
                    0.10F + 0.015F * static_cast<float>(value + 1) + 0.01F * static_cast<float>(channel + 1);
                dy[value * channels + channel] =
                    (channel % 2 == 0 ? 1.0F : -1.0F) * 0.02F * static_cast<float>(value + channel + 2);
            }
        }

        std::vector<float> filter1(channels * channels_per_group * kernel1_width, 0.0F);
        std::vector<float> filter2(channels * channels_per_group * kernel2_width, 0.0F);
        for (uint64_t output_channel = 0; output_channel < channels; ++output_channel) {
            for (uint64_t input_channel = 0; input_channel < channels_per_group; ++input_channel) {
                const size_t base1 = (output_channel * channels_per_group + input_channel) * kernel1_width;
                const size_t base2 = (output_channel * channels_per_group + input_channel) * kernel2_width;
                const float diagonalish = input_channel == (output_channel % channels_per_group) ? 1.0F : 0.35F;
                filter1[base1 + 0] = 0.08F * diagonalish;
                filter1[base1 + 1] = 0.55F * diagonalish;
                filter2[base2 + 0] = 0.05F * diagonalish;
                filter2[base2 + 1] = 0.10F * diagonalish;
                filter2[base2 + 2] = 0.45F * diagonalish;
            }
        }

        Stream stream(0);
        Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
        RowPartitionRuntime partition(
            gpu_offsets,
            RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_values_per_row));
        partition.setHostOffsets(offsets);
        Tensor gpu_x = makeGpuTensorFromFloats({max_total_values, channels}, x, production_case.dtype, stream);
        Tensor gpu_filter1 = makeGpuTensorFromFloats(
            {channels, channels_per_group, kernel1_width}, filter1, production_case.dtype, stream);
        Tensor gpu_filter2 = makeGpuTensorFromFloats(
            {channels, channels_per_group, kernel2_width}, filter2, production_case.dtype, stream);
        Tensor gpu_dy = makeGpuTensorFromFloats({max_total_values, channels}, dy, production_case.dtype, stream);

        const RaggedExpression input = RaggedExpression::input(
            "tokens",
            RaggedTensorDescriptor(production_case.dtype,
                                   {channels},
                                   batch_size,
                                   max_total_values,
                                   max_values_per_row,
                                   DataType::UINT32));
        const Expression filter1_expr = Expression::input("filter1", std::nullopt, production_case.dtype);
        const Expression filter2_expr = Expression::input("filter2", std::nullopt, production_case.dtype);
        const RaggedExpression hidden = input.causalConv1d(filter1_expr,
                                                           channels,
                                                           kernel1_width,
                                                           1,
                                                           DataType::FP32,
                                                           production_case.dtype,
                                                           groups)
                                            .relu();
        const RaggedExpression output = hidden.causalConv1d(filter2_expr,
                                                             channels,
                                                             kernel2_width,
                                                             1,
                                                             DataType::FP32,
                                                             production_case.dtype,
                                                             groups);

        FusedEquation forward =
            FusedEquation::compile(Expression::outputs({{"y", output.getValues()}}).physicalOutputs(), 0);
        StampedExecutionPlan forward_plan = forward.stamp({{"tokens.values", gpu_x},
                                                           {"tokens.offsets", gpu_offsets},
                                                           {"filter1", gpu_filter1},
                                                           {"filter2", gpu_filter2}},
                                                          stream);
        EXPECT_EQ(forward_plan.stageKindNames(),
                  (std::vector<std::string>{"PaddedRaggedPack",
                                            "RaggedConv1dCausal",
                                            "PaddedRaggedPointwise",
                                            "RaggedConv1dCausal",
                                            "PaddedRaggedUnpack"}))
            << "T10 forward topology may not acquire an internal representation boundary";

        FusedEquation backward = forward.compileBackward({"tokens.values", "filter1", "filter2"}, "dy");
        Tensor dx = makeGpuTensorFromFloats({max_total_values, channels},
                                            std::vector<float>(max_total_values * channels, inactive_dx_sentinel),
                                            production_case.dtype,
                                            stream);
        StampedExecutionPlan backward_plan = backward.stamp({{"tokens.values", gpu_x},
                                                             {"tokens.offsets", gpu_offsets},
                                                             {"filter1", gpu_filter1},
                                                             {"filter2", gpu_filter2},
                                                             {"dy", gpu_dy}},
                                                            stream,
                                                            {},
                                                            {{"tokens.values_grad", dx}});
        const std::vector<std::string> backward_stage_names = backward_plan.stageKindNames();
        EXPECT_EQ(std::count(backward_stage_names.begin(),
                             backward_stage_names.end(),
                             "RaggedConv1dCausalBackwardData"),
                  2);
        EXPECT_EQ(std::count(backward_stage_names.begin(),
                             backward_stage_names.end(),
                             "RaggedConv1dCausalBackwardFilter"),
                  2);
        EXPECT_EQ(std::count(backward_stage_names.begin(), backward_stage_names.end(), "PaddedRaggedUnpack"), 1)
            << "the public dX exit must be the only packed boundary in this reduction-free backward spine";
        EXPECT_GE(std::count(backward_stage_names.begin(), backward_stage_names.end(), "PaddedRaggedPointwise"), 1)
            << "ReLU backward must remain active-local and retained";

        forward_plan.run();
        backward_plan.run();
        stream.synchronize();

        std::vector<float> expected_hidden_pre = cpuRaggedCausalConv1d(x,
                                                                       offsets,
                                                                       filter1,
                                                                       max_total_values,
                                                                       channels,
                                                                       channels,
                                                                       kernel1_width,
                                                                       1,
                                                                       0.0F,
                                                                       groups);
        std::vector<float> expected_hidden = expected_hidden_pre;
        for (uint64_t value = 0; value < offsets.back(); ++value) {
            for (uint64_t channel = 0; channel < channels; ++channel) {
                const size_t index = value * channels + channel;
                expected_hidden[index] = std::max(expected_hidden[index], 0.0F);
            }
        }
        const std::vector<float> expected_y = cpuRaggedCausalConv1d(expected_hidden,
                                                                     offsets,
                                                                     filter2,
                                                                     max_total_values,
                                                                     channels,
                                                                     channels,
                                                                     kernel2_width,
                                                                     1,
                                                                     0.0F,
                                                                     groups);
        const std::vector<float> expected_dw2 = cpuRaggedCausalConv1dWgrad(expected_hidden,
                                                                           dy,
                                                                           offsets,
                                                                           max_total_values,
                                                                           channels,
                                                                           channels,
                                                                           kernel2_width,
                                                                           1,
                                                                           groups);
        std::vector<float> expected_dhidden = cpuRaggedCausalConv1dDgrad(dy,
                                                                         offsets,
                                                                         filter2,
                                                                         max_total_values,
                                                                         channels,
                                                                         channels,
                                                                         kernel2_width,
                                                                         1,
                                                                         0.0F,
                                                                         groups);
        for (uint64_t value = 0; value < offsets.back(); ++value) {
            for (uint64_t channel = 0; channel < channels; ++channel) {
                const size_t index = value * channels + channel;
                if (expected_hidden_pre[index] <= 0.0F) {
                    expected_dhidden[index] = 0.0F;
                }
            }
        }
        const std::vector<float> expected_dw1 = cpuRaggedCausalConv1dWgrad(x,
                                                                           expected_dhidden,
                                                                           offsets,
                                                                           max_total_values,
                                                                           channels,
                                                                           channels,
                                                                           kernel1_width,
                                                                           1,
                                                                           groups);
        const std::vector<float> expected_dx = cpuRaggedCausalConv1dDgrad(expected_dhidden,
                                                                          offsets,
                                                                          filter1,
                                                                          max_total_values,
                                                                          channels,
                                                                          channels,
                                                                          kernel1_width,
                                                                          1,
                                                                          inactive_dx_sentinel,
                                                                          groups);

        const std::vector<float> actual_y = copyToCpuFloatValues(forward_plan.output("y"), stream);
        const std::vector<float> actual_dx = copyToCpuFloatValues(backward_plan.output("tokens.values_grad"), stream);
        const std::vector<float> actual_dw1 = copyToCpuFloatValues(backward_plan.output("filter1_grad"), stream);
        const std::vector<float> actual_dw2 = copyToCpuFloatValues(backward_plan.output("filter2_grad"), stream);
        const uint64_t active_elements = offsets.back() * channels;
        for (uint64_t index = 0; index < active_elements; ++index) {
            EXPECT_NEAR(actual_y[index], expected_y[index], production_case.tolerance) << "forward index " << index;
            EXPECT_NEAR(actual_dx[index], expected_dx[index], production_case.tolerance) << "dX index " << index;
            EXPECT_TRUE(std::isfinite(actual_y[index]));
            EXPECT_TRUE(std::isfinite(actual_dx[index]));
        }
        for (uint64_t index = active_elements; index < actual_dx.size(); ++index) {
            EXPECT_NEAR(actual_dx[index], inactive_dx_sentinel, 1.0e-3F)
                << "inactive packed dX capacity must remain untouched";
        }
        expectNear(actual_dw1, expected_dw1, production_case.tolerance);
        expectNear(actual_dw2, expected_dw2, production_case.tolerance);

        for (const RaggedConv1dStageDiagnostic& diagnostic : forward_plan.raggedConv1dStageDiagnostics()) {
            EXPECT_GT(diagnostic.width_capacity_count, 0u);
            EXPECT_EQ(diagnostic.prebuilt_cudnn_plan_count, diagnostic.width_capacity_count);
            EXPECT_EQ(diagnostic.selected_width_capacity, 8u);
            EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u);
        }
        for (const RaggedConv1dStageDiagnostic& diagnostic : backward_plan.raggedConv1dStageDiagnostics()) {
            EXPECT_GT(diagnostic.width_capacity_count, 0u);
            EXPECT_EQ(diagnostic.prebuilt_cudnn_plan_count, diagnostic.width_capacity_count);
            EXPECT_EQ(diagnostic.selected_width_capacity, 8u);
            EXPECT_EQ(diagnostic.explicit_unfold_workspace_bytes, 0u)
                << "T10 forbids im2col/unfold-like temporary storage in retained training";
        }
    }
}

TEST(RaggedExpression, CausalConv1dT8CRmsNormIsExplicitPaddedRepresentationBoundary) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t channels = 4;
    constexpr uint64_t kernel1_width = 3;
    constexpr uint64_t kernel2_width = 2;
    constexpr float epsilon = 1.0e-5F;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> values(max_total_values * channels, 7777.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            values[value * channels + channel] =
                static_cast<float>(static_cast<int>((value + 1) * (channel + 2)) - 7) * 0.125F;
        }
    }
    std::vector<float> filter1(channels * channels * kernel1_width, 0.0F);
    std::vector<float> filter2(channels * channels * kernel2_width, 0.0F);
    for (uint64_t channel = 0; channel < channels; ++channel) {
        filter1[(channel * channels + channel) * kernel1_width + 0] = 0.25F;
        filter1[(channel * channels + channel) * kernel1_width + 1] = -0.5F;
        filter1[(channel * channels + channel) * kernel1_width + 2] = 1.0F;
        filter2[(channel * channels + channel) * kernel2_width + 0] = 0.5F;
        filter2[(channel * channels + channel) * kernel2_width + 1] = 0.5F;
    }
    const std::vector<float> scale{1.0F, 0.5F, 1.5F, -2.0F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel1_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel2_width}, filter2, stream);
    Tensor gpu_scale = makeGpuTensor<float>({channels}, scale, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression scale_expr = Expression::input("scale", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel1_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression normalized = hidden.rmsNorm(scale_expr, epsilon, DataType::FP32, DataType::FP32);
    const RaggedExpression result =
        normalized.causalConv1d(filter2_expr, channels, kernel2_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> named_inputs{{"tokens.values", gpu_values},
                                                               {"tokens.offsets", gpu_offsets},
                                                               {"filter1", gpu_filter1},
                                                               {"filter2", gpu_filter2},
                                                               {"scale", gpu_scale}};
    constexpr float inactive_sentinel = -8765.25F;
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, channels},
        std::vector<float>(max_total_values * channels, inactive_sentinel),
        stream);
    StampedExecutionPlan plan = equation.stamp(named_inputs, stream, {}, {{"y", packed_output}});

    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack",
                                        "SanitizePackedTail",
                                        "RmsNorm",
                                        "PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}));
    const std::vector<RaggedConv1dStageDiagnostic> conv_diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(conv_diagnostics.size(), 2U);

    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);

    std::vector<float> expected_hidden = cpuRaggedCausalConv1d(values,
                                                               offsets,
                                                               filter1,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel1_width,
                                                               1,
                                                               0.0F);
    std::vector<float> expected_normalized(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        float sum_sq = 0.0F;
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const float x = expected_hidden[value * channels + channel];
            sum_sq += x * x;
        }
        const float inv_rms = 1.0F / std::sqrt(sum_sq / static_cast<float>(channels) + epsilon);
        for (uint64_t channel = 0; channel < channels; ++channel) {
            expected_normalized[value * channels + channel] =
                expected_hidden[value * channels + channel] * inv_rms * scale[channel];
        }
    }
    const std::vector<float> expected = cpuRaggedCausalConv1d(expected_normalized,
                                                              offsets,
                                                              filter2,
                                                              max_total_values,
                                                              channels,
                                                              channels,
                                                              kernel2_width,
                                                              1,
                                                              0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const uint64_t index = value * channels + channel;
            EXPECT_NEAR(actual[index], expected[index], 2.0e-4F) << "active output index " << index;
        }
    }
    for (uint64_t index = offsets.back() * channels; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index], inactive_sentinel) << "packed spare output index " << index;
    }
}

TEST(RaggedExpression, LayerNormT8CCanonicalIdentityAndSerializationPreserveConfiguration) {
    const RaggedExpression input =
        RaggedExpression::input("tokens", makeDescriptor(DataType::FP32, {4}, 3, 11, DataType::UINT32));
    const Expression scale = Expression::input("scale", std::nullopt, DataType::FP32);
    const Expression bias = Expression::input("bias", std::nullopt, DataType::FP32);
    const RaggedExpression normalized = input.layerNorm(scale, bias, 2.5e-4, DataType::FP32, DataType::FP16);
    const RaggedExpression different_epsilon = input.layerNorm(scale, bias, 5.0e-4, DataType::FP32, DataType::FP16);

    const std::string canonical = canonicalize(normalized.getValues().expression());
    EXPECT_NE(canonical.find("LAYERNORM"), std::string::npos);
    EXPECT_NE(canonical.find("hidden=4"), std::string::npos);
    EXPECT_NE(canonical.find("packedRowsCapacity=11"), std::string::npos);
    EXPECT_NE(canonical, canonicalize(different_epsilon.getValues().expression()));

    const ExpressionDefinition definition =
        ExpressionDefinition::fromOutputs(Expression::outputs({{"y", normalized.getValues()}}));
    const nlohmann::json payload = definition.architectureJson();
    bool found = false;
    for (const auto& node : payload.at("nodes")) {
        if (node.at("op").get<std::string>() != "layernorm") continue;
        found = true;
        EXPECT_EQ(node.at("layer_norm_normalized_feature_count").get<uint64_t>(), 4ULL);
        EXPECT_DOUBLE_EQ(node.at("layer_norm_epsilon").get<double>(), 2.5e-4);
        EXPECT_EQ(node.at("layer_norm_packed_row_capacity").get<uint64_t>(), 11ULL);
    }
    ASSERT_TRUE(found);

    const ExpressionDefinition loaded = ExpressionDefinition::deserialize(payload);
    EXPECT_EQ(loaded.architectureJson(), payload);
    EXPECT_EQ(loaded.canonical_hash, definition.canonical_hash);
}

TEST(RaggedExpression, CausalConv1dT8CLayerNormIsExplicitPaddedRepresentationBoundary) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t channels = 4;
    constexpr uint64_t kernel1_width = 3;
    constexpr uint64_t kernel2_width = 2;
    constexpr float epsilon = 1.0e-5F;
    const std::vector<uint32_t> offsets32{0, 3, 3, 8};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> values(max_total_values * channels, 7777.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            values[value * channels + channel] =
                static_cast<float>(static_cast<int>((value + 2) * (channel + 1)) - 6) * 0.125F;
        }
    }
    std::vector<float> filter1(channels * channels * kernel1_width, 0.0F);
    std::vector<float> filter2(channels * channels * kernel2_width, 0.0F);
    for (uint64_t channel = 0; channel < channels; ++channel) {
        filter1[(channel * channels + channel) * kernel1_width + 0] = -0.25F;
        filter1[(channel * channels + channel) * kernel1_width + 1] = 0.5F;
        filter1[(channel * channels + channel) * kernel1_width + 2] = 1.0F;
        filter2[(channel * channels + channel) * kernel2_width + 0] = 0.75F;
        filter2[(channel * channels + channel) * kernel2_width + 1] = -0.25F;
    }
    const std::vector<float> scale{1.0F, 0.5F, 1.5F, -2.0F};
    const std::vector<float> bias{0.25F, -0.5F, 1.0F, 0.75F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel1_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel2_width}, filter2, stream);
    Tensor gpu_scale = makeGpuTensor<float>({channels}, scale, stream);
    Tensor gpu_bias = makeGpuTensor<float>({channels}, bias, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression scale_expr = Expression::input("scale", std::nullopt, DataType::FP32);
    const Expression bias_expr = Expression::input("bias", std::nullopt, DataType::FP32);
    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel1_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression normalized =
        hidden.layerNorm(scale_expr, bias_expr, epsilon, DataType::FP32, DataType::FP32);
    const RaggedExpression result =
        normalized.causalConv1d(filter2_expr, channels, kernel2_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> named_inputs{{"tokens.values", gpu_values},
                                                               {"tokens.offsets", gpu_offsets},
                                                               {"filter1", gpu_filter1},
                                                               {"filter2", gpu_filter2},
                                                               {"scale", gpu_scale},
                                                               {"bias", gpu_bias}};
    constexpr float inactive_sentinel = -7654.5F;
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, channels},
        std::vector<float>(max_total_values * channels, inactive_sentinel),
        stream);
    StampedExecutionPlan plan = equation.stamp(named_inputs, stream, {}, {{"y", packed_output}});

    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack",
                                        "LayerNorm",
                                        "PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}));
    const std::vector<RaggedConv1dStageDiagnostic> conv_diagnostics = plan.raggedConv1dStageDiagnostics();
    ASSERT_EQ(conv_diagnostics.size(), 2U);

    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);

    std::vector<float> expected_hidden = cpuRaggedCausalConv1d(values,
                                                               offsets,
                                                               filter1,
                                                               max_total_values,
                                                               channels,
                                                               channels,
                                                               kernel1_width,
                                                               1,
                                                               0.0F);
    std::vector<float> expected_normalized(max_total_values * channels, 0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        float mean = 0.0F;
        for (uint64_t channel = 0; channel < channels; ++channel) {
            mean += expected_hidden[value * channels + channel];
        }
        mean /= static_cast<float>(channels);
        float variance = 0.0F;
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const float centered = expected_hidden[value * channels + channel] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(channels);
        const float inv_std = 1.0F / std::sqrt(variance + epsilon);
        for (uint64_t channel = 0; channel < channels; ++channel) {
            expected_normalized[value * channels + channel] =
                (expected_hidden[value * channels + channel] - mean) * inv_std * scale[channel] + bias[channel];
        }
    }
    const std::vector<float> expected = cpuRaggedCausalConv1d(expected_normalized,
                                                              offsets,
                                                              filter2,
                                                              max_total_values,
                                                              channels,
                                                              channels,
                                                              kernel2_width,
                                                              1,
                                                              0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const uint64_t index = value * channels + channel;
            EXPECT_NEAR(actual[index], expected[index], 3.0e-4F) << "active output index " << index;
        }
    }
    for (uint64_t index = offsets.back() * channels; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index], inactive_sentinel) << "packed spare output index " << index;
    }
}

TEST(RaggedExpression, CausalConv1dT8CMixedPointwiseRegionStopsAtNormalizationBoundaries) {
    REQUIRE_CUDA_DEVICE();

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 12;
    constexpr uint64_t channels = 4;
    constexpr uint64_t kernel1_width = 3;
    constexpr uint64_t kernel2_width = 2;
    constexpr float layer_epsilon = 1.0e-5F;
    constexpr float rms_epsilon = 2.0e-5F;
    const std::vector<uint32_t> offsets32{0, 4, 4, 9};
    const std::vector<uint64_t> offsets(offsets32.begin(), offsets32.end());

    std::vector<float> values(max_total_values * channels, 9999.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            values[value * channels + channel] =
                static_cast<float>(static_cast<int>((value + 1) * 3 + channel * 2) - 11) * 0.125F;
        }
    }
    std::vector<float> filter1(channels * channels * kernel1_width, 0.0F);
    std::vector<float> filter2(channels * channels * kernel2_width, 0.0F);
    for (uint64_t channel = 0; channel < channels; ++channel) {
        filter1[(channel * channels + channel) * kernel1_width + 0] = 0.25F;
        filter1[(channel * channels + channel) * kernel1_width + 1] = -0.5F;
        filter1[(channel * channels + channel) * kernel1_width + 2] = 1.0F;
        filter2[(channel * channels + channel) * kernel2_width + 0] = 0.5F;
        filter2[(channel * channels + channel) * kernel2_width + 1] = 0.5F;
    }
    const std::vector<float> pointwise_bias{0.25F, -0.5F, 0.75F, -1.0F};
    const std::vector<float> layer_scale{1.0F, 0.75F, 1.25F, -1.5F};
    const std::vector<float> layer_bias{0.5F, -0.25F, 0.125F, 0.75F};
    const std::vector<float> rms_scale{1.0F, 0.5F, 1.5F, 2.0F};

    Stream stream(0);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, channels}, values, stream);
    Tensor gpu_filter1 = makeGpuTensor<float>({channels, channels, kernel1_width}, filter1, stream);
    Tensor gpu_filter2 = makeGpuTensor<float>({channels, channels, kernel2_width}, filter2, stream);
    Tensor gpu_pointwise_bias = makeGpuTensor<float>({channels}, pointwise_bias, stream);
    Tensor gpu_layer_scale = makeGpuTensor<float>({channels}, layer_scale, stream);
    Tensor gpu_layer_bias = makeGpuTensor<float>({channels}, layer_bias, stream);
    Tensor gpu_rms_scale = makeGpuTensor<float>({channels}, rms_scale, stream);
    Tensor gpu_offsets = makeGpuTensor<uint32_t>({batch_size + 1}, offsets32, stream);
    RowPartitionRuntime partition(
        gpu_offsets,
        RowPartitionDescriptor(batch_size, max_total_values, DataType::UINT32, max_total_values));
    partition.setHostOffsets(offsets);

    const RaggedExpression input = RaggedExpression::input(
        "tokens", makeDescriptor(DataType::FP32, {channels}, batch_size, max_total_values, DataType::UINT32));
    const Expression filter1_expr = Expression::input("filter1", std::nullopt, DataType::FP32);
    const Expression filter2_expr = Expression::input("filter2", std::nullopt, DataType::FP32);
    const Expression pointwise_bias_expr = Expression::input("pointwise_bias", std::nullopt, DataType::FP32);
    const Expression layer_scale_expr = Expression::input("layer_scale", std::nullopt, DataType::FP32);
    const Expression layer_bias_expr = Expression::input("layer_bias", std::nullopt, DataType::FP32);
    const Expression rms_scale_expr = Expression::input("rms_scale", std::nullopt, DataType::FP32);

    const RaggedExpression hidden =
        input.causalConv1d(filter1_expr, channels, kernel1_width, 1, DataType::FP32, DataType::FP32);
    const RaggedExpression biased = hidden.mapValues([&](const Expression& x) { return x + pointwise_bias_expr; });
    const RaggedExpression layer_normalized =
        biased.layerNorm(layer_scale_expr, layer_bias_expr, layer_epsilon, DataType::FP32, DataType::FP32);
    const RaggedExpression activated = layer_normalized.relu();
    const RaggedExpression rms_normalized =
        activated.rmsNorm(rms_scale_expr, rms_epsilon, DataType::FP32, DataType::FP32);
    const RaggedExpression result =
        rms_normalized.causalConv1d(filter2_expr, channels, kernel2_width, 1, DataType::FP32, DataType::FP32);

    FusedEquation equation = FusedEquation::compile(Expression::outputs({{"y", result.getValues()}}).physicalOutputs(), 0);
    const std::unordered_map<std::string, Tensor> named_inputs{{"tokens.values", gpu_values},
                                                               {"tokens.offsets", gpu_offsets},
                                                               {"filter1", gpu_filter1},
                                                               {"filter2", gpu_filter2},
                                                               {"pointwise_bias", gpu_pointwise_bias},
                                                               {"layer_scale", gpu_layer_scale},
                                                               {"layer_bias", gpu_layer_bias},
                                                               {"rms_scale", gpu_rms_scale}};
    constexpr float inactive_sentinel = -4321.75F;
    Tensor packed_output = makeGpuTensor<float>(
        {max_total_values, channels},
        std::vector<float>(max_total_values * channels, inactive_sentinel),
        stream);
    StampedExecutionPlan plan = equation.stamp(named_inputs, stream, {}, {{"y", packed_output}});

    // Normalization is an explicit padded-representation boundary. Because the
    // channel-broadcast add is layout-compatible on either side of that boundary,
    // the compiler may place the unpack before it and execute the add as an
    // ordinary packed fused kernel. That is preferable here: it touches only the
    // active packed values and does not introduce another representation change.
    EXPECT_EQ(plan.stageKindNames(),
              (std::vector<std::string>{"PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack",
                                        "FusedKernel",
                                        "LayerNorm",
                                        "FusedKernel",
                                        "SanitizePackedTail",
                                        "RmsNorm",
                                        "PaddedRaggedPack",
                                        "RaggedConv1dCausal",
                                        "PaddedRaggedUnpack"}));

    plan.run();
    stream.synchronize();
    const std::vector<float> actual = copyToCpuValues(plan.output("y"), stream);

    std::vector<float> expected = cpuRaggedCausalConv1d(values,
                                                        offsets,
                                                        filter1,
                                                        max_total_values,
                                                        channels,
                                                        channels,
                                                        kernel1_width,
                                                        1,
                                                        0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            expected[value * channels + channel] += pointwise_bias[channel];
        }
        float mean = 0.0F;
        for (uint64_t channel = 0; channel < channels; ++channel) {
            mean += expected[value * channels + channel];
        }
        mean /= static_cast<float>(channels);
        float variance = 0.0F;
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const float centered = expected[value * channels + channel] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(channels);
        const float inv_std = 1.0F / std::sqrt(variance + layer_epsilon);
        for (uint64_t channel = 0; channel < channels; ++channel) {
            float normalized =
                (expected[value * channels + channel] - mean) * inv_std * layer_scale[channel] + layer_bias[channel];
            expected[value * channels + channel] = std::max(normalized, 0.0F);
        }
        float sum_sq = 0.0F;
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const float x = expected[value * channels + channel];
            sum_sq += x * x;
        }
        const float inv_rms = 1.0F / std::sqrt(sum_sq / static_cast<float>(channels) + rms_epsilon);
        for (uint64_t channel = 0; channel < channels; ++channel) {
            expected[value * channels + channel] *= inv_rms * rms_scale[channel];
        }
    }
    expected = cpuRaggedCausalConv1d(expected,
                                     offsets,
                                     filter2,
                                     max_total_values,
                                     channels,
                                     channels,
                                     kernel2_width,
                                     1,
                                     0.0F);
    for (uint64_t value = 0; value < offsets.back(); ++value) {
        for (uint64_t channel = 0; channel < channels; ++channel) {
            const uint64_t index = value * channels + channel;
            EXPECT_NEAR(actual[index], expected[index], 4.0e-4F) << "active output index " << index;
        }
    }
    for (uint64_t index = offsets.back() * channels; index < actual.size(); ++index) {
        EXPECT_EQ(actual[index], inactive_sentinel) << "packed spare output index " << index;
    }
}

TEST(RaggedExpression, TrailingTransposePreservesPartitionAndUsesPermutationStrides) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 3, 4}, 3, 9));

    const RaggedExpression result = ragged.transposeTrailingDimensions();

    EXPECT_EQ(result.getValuesDimensions(), std::vector<uint64_t>({9, 2, 4, 3}));
    EXPECT_EQ(result.getTrailingDimensions(), std::vector<uint64_t>({2, 4, 3}));
    EXPECT_TRUE(result.getOffsets().isSameLogicalNode(ragged.getOffsets()));
    EXPECT_TRUE(result.getRuntimeExtent().activeValueCount.isSameLogicalNode(ragged.getRuntimeExtent().activeValueCount));
    EXPECT_EQ(result.getRuntimeExtent().elementsPerValue, 24ULL);

    const MarkedValueNodes marked = markedValueNodes(result.getValues());
    EXPECT_EQ(marked.values.op, ExprOp::STRIDED_VIEW);
    EXPECT_EQ(marked.values.view_dims, std::vector<uint64_t>({9, 2, 4, 3}));
    EXPECT_EQ(marked.values.view_strides, std::vector<uint64_t>({24, 12, 1, 4}));
    EXPECT_EQ(marked.values.view_element_offset, 0ULL);
    EXPECT_EQ(marked.marker.ragged_runtime_elements_per_value, 24ULL);
}

TEST(RaggedExpression, TrailingTransposeRequiresAtLeastTwoTrailingDimensions) {
    const RaggedExpression vector = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {8}, 3, 9));
    EXPECT_THROW((void)vector.transposeTrailingDimensions(), std::invalid_argument);
}

TEST(RaggedExpression, TrailingTransposeAutodiffScattersPermutationWithinActiveRows) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 3, 4}, 3, 9));
    const RaggedExpression transposed = ragged.transposeTrailingDimensions();

    const PhysicalOutputs forward = Expression::outputs({{"y", transposed.getValues()}}).physicalOutputs();
    PhysicalOutputs backward = buildBackwardOutputs(forward, {"x.values"});
    resolveRaggedBackwardTestDTypes(backward, DataType::UINT32);

    bool found_scatter = false;
    for (const ExprNode& node : backward.expr->nodes) {
        if (node.op != ExprOp::RAGGED_VALUEWISE_EXTENT || node.ragged_runtime_elements_per_value != 24ULL) continue;
        ASSERT_LT(node.lhs, backward.expr->nodes.size());
        const ExprNode& scatter = backward.expr->nodes.at(node.lhs);
        if (scatter.op != ExprOp::STRIDED_VIEW_BACKWARD) continue;

        found_scatter = true;
        EXPECT_EQ(scatter.fill_dims, std::vector<uint64_t>({9, 2, 3, 4}));
        EXPECT_EQ(scatter.view_dims, std::vector<uint64_t>({9, 2, 4, 3}));
        EXPECT_EQ(scatter.view_strides, std::vector<uint64_t>({24, 12, 1, 4}));
        EXPECT_EQ(scatter.view_element_offset, 0ULL);
        EXPECT_EQ(node.ragged_runtime_batch_size, 3ULL);
        EXPECT_EQ(node.ragged_runtime_max_active_values, 9ULL);
    }
    EXPECT_TRUE(found_scatter);

    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(backward);
    ASSERT_EQ(stages.size(), 1U);
    ASSERT_EQ(stages[0].kind, PhysicalExecutionStage::Kind::FusedKernel);
    const std::string source = CudaSourceEmitter::emitFlat(stages[0], "ragged_trailing_transpose_backward");
    EXPECT_NE(source.find("runtime_numel_u64 = active_values * 24ULL"), std::string::npos);
}

TEST(RaggedExpression, TrailingTransposeComposesWithPrecedingTrailingSlice) {
    const RaggedExpression ragged = RaggedExpression::input("x", makeDescriptor(DataType::FP32, {2, 4, 6}, 3, 9));
    const RaggedExpression sliced = ragged.sliceLastDimension(/*start=*/1, /*length=*/3);
    const RaggedExpression transposed = sliced.transposeTrailingDimensions();

    EXPECT_EQ(transposed.getValuesDimensions(), std::vector<uint64_t>({9, 2, 3, 4}));
    EXPECT_TRUE(transposed.getOffsets().isSameLogicalNode(ragged.getOffsets()));

    const MarkedValueNodes marked = markedValueNodes(transposed.getValues());
    ASSERT_EQ(marked.values.op, ExprOp::STRIDED_VIEW);
    EXPECT_EQ(marked.values.view_dims, std::vector<uint64_t>({9, 2, 3, 4}));
    EXPECT_EQ(marked.values.view_strides, std::vector<uint64_t>({48, 24, 1, 6}));
    EXPECT_EQ(marked.values.view_element_offset, 1ULL);
}

TEST(RaggedExpression, TrailingTransposeBackwardRunsOnlyAcrossActivePackedRows) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t batch_size = 3;
    constexpr uint64_t max_total_values = 5;
    constexpr uint64_t first = 2;
    constexpr uint64_t second = 3;
    constexpr uint64_t elements_per_value = first * second;

    const RaggedExpression ragged = RaggedExpression::input(
        "x", makeDescriptor(DataType::FP32, {first, second}, batch_size, max_total_values, DataType::UINT32));
    const RaggedExpression transposed = ragged.transposeTrailingDimensions();

    std::vector<float> values(max_total_values * elements_per_value, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t i = 0; i < 3 * elements_per_value; ++i) values[i] = static_cast<float>(i + 1);
    Tensor gpu_values = makeGpuTensor<float>({max_total_values, first, second}, values, stream);
    Tensor offsets = makeGpuTensor<uint32_t>({batch_size + 1}, {0U, 1U, 1U, 3U}, stream);

    std::vector<float> upstream_values(max_total_values * elements_per_value, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t i = 0; i < 3 * elements_per_value; ++i) upstream_values[i] = static_cast<float>(100 + i);
    Tensor upstream = makeGpuTensor<float>({max_total_values, second, first}, upstream_values, stream);

    Tensor gradient(gpuPlacement, TensorDescriptor(DataType::FP32, {max_total_values, first, second}));
    constexpr float inactive_sentinel = 777.0F;
    gradient.fill(inactive_sentinel, stream);

    const Tensor result = runBackwardOutput(transposed.getValues(),
                                            {{"x.values", gpu_values}, {"x.offsets", offsets}, {"dy", upstream}},
                                            "x.values",
                                            "dy",
                                            stream,
                                            gradient);
    const std::vector<float> actual = copyToCpuValues(result, stream);

    std::vector<float> expected(max_total_values * elements_per_value, inactive_sentinel);
    for (uint64_t value = 0; value < 3; ++value) {
        const uint64_t base = value * elements_per_value;
        for (uint64_t i = 0; i < first; ++i) {
            for (uint64_t j = 0; j < second; ++j) {
                expected[base + i * second + j] = upstream_values[base + j * first + i];
            }
        }
    }
    expectNear(actual, expected);
}
