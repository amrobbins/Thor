#include "Utilities/Expression/ConvolutionSpatial.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int cuda_device_count_for_test = 0;                                                                              \
        const cudaError_t cuda_status_for_test = cudaGetDeviceCount(&cuda_device_count_for_test);                        \
        if (cuda_status_for_test != cudaSuccess || cuda_device_count_for_test <= 0) {                                     \
            GTEST_SKIP() << "CUDA device is required for asymmetric Conv2D execution tests.";                            \
        }                                                                                                                \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t numel(const std::vector<uint64_t>& dims) {
    uint64_t count = 1;
    for (uint64_t dim : dims) {
        count *= dim;
    }
    return count;
}

Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    if (numel(dims) != values.size()) {
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    }
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, dims));
    auto* cpu_ptr = static_cast<float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i) {
        cpu_ptr[i] = values[i];
    }

    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyToCpu(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(gpu.getDataType(), gpu.getDimensions()));
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();

    std::vector<float> values(numel(cpu.getDimensions()));
    const auto* cpu_ptr = static_cast<const float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = cpu_ptr[i];
    }
    return values;
}

void expectAllClose(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance = 1.0e-4F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
    }
}

uint64_t outputExtent(uint64_t input,
                      uint64_t filter,
                      int32_t stride,
                      int32_t dilation,
                      int32_t pre_padding,
                      int32_t post_padding) {
    const int64_t effective_filter = static_cast<int64_t>(dilation) * (static_cast<int64_t>(filter) - 1) + 1;
    const int64_t numerator = static_cast<int64_t>(input) + pre_padding + post_padding - effective_filter;
    if (numerator < 0) {
        throw std::runtime_error("CPU reference Conv2D produced negative output extent.");
    }
    return static_cast<uint64_t>(numerator / stride + 1);
}

size_t nchwIndex(uint64_t n, uint64_t c, uint64_t h, uint64_t w, uint64_t channels, uint64_t height, uint64_t width) {
    return static_cast<size_t>(((n * channels + c) * height + h) * width + w);
}

std::vector<float> conv2dForwardReference(const std::vector<float>& input,
                                          const std::vector<float>& filter,
                                          uint64_t n_count,
                                          uint64_t c_count,
                                          uint64_t input_h,
                                          uint64_t input_w,
                                          uint64_t k_count,
                                          uint64_t filter_h,
                                          uint64_t filter_w,
                                          const ConvolutionSpatial2d& spatial,
                                          uint64_t groups = 1) {
    const uint64_t output_h = outputExtent(
        input_h, filter_h, spatial.stride_h, spatial.dilation_h, spatial.pre_padding_h, spatial.post_padding_h);
    const uint64_t output_w = outputExtent(
        input_w, filter_w, spatial.stride_w, spatial.dilation_w, spatial.pre_padding_w, spatial.post_padding_w);
    if (groups == 0 || c_count % groups != 0 || k_count % groups != 0)
        throw std::runtime_error("Grouped Conv2D forward reference received invalid channel geometry.");
    const uint64_t c_per_group = c_count / groups;
    const uint64_t k_per_group = k_count / groups;
    std::vector<float> output(n_count * k_count * output_h * output_w, 0.0F);

    for (uint64_t n = 0; n < n_count; ++n) {
        for (uint64_t k = 0; k < k_count; ++k) {
            const uint64_t group = k / k_per_group;
            for (uint64_t oh = 0; oh < output_h; ++oh) {
                for (uint64_t ow = 0; ow < output_w; ++ow) {
                    float sum = 0.0F;
                    for (uint64_t local_c = 0; local_c < c_per_group; ++local_c) {
                        const uint64_t c = group * c_per_group + local_c;
                        for (uint64_t r = 0; r < filter_h; ++r) {
                            for (uint64_t s = 0; s < filter_w; ++s) {
                                const int64_t ih = static_cast<int64_t>(oh * spatial.stride_h + r * spatial.dilation_h) -
                                                   spatial.pre_padding_h;
                                const int64_t iw = static_cast<int64_t>(ow * spatial.stride_w + s * spatial.dilation_w) -
                                                   spatial.pre_padding_w;
                                if (ih < 0 || iw < 0 || ih >= static_cast<int64_t>(input_h) || iw >= static_cast<int64_t>(input_w)) {
                                    continue;
                                }
                                const size_t input_idx = nchwIndex(n, c, ih, iw, c_count, input_h, input_w);
                                const size_t filter_idx = nchwIndex(k, local_c, r, s, c_per_group, filter_h, filter_w);
                                sum += input[input_idx] * filter[filter_idx];
                            }
                        }
                    }
                    output[nchwIndex(n, k, oh, ow, k_count, output_h, output_w)] = sum;
                }
            }
        }
    }
    return output;
}

std::vector<float> conv2dDataGradReference(const std::vector<float>& grad_output,
                                           const std::vector<float>& filter,
                                           uint64_t n_count,
                                           uint64_t c_count,
                                           uint64_t input_h,
                                           uint64_t input_w,
                                           uint64_t k_count,
                                           uint64_t filter_h,
                                           uint64_t filter_w,
                                           const ConvolutionSpatial2d& spatial,
                                           uint64_t groups = 1) {
    const uint64_t output_h = outputExtent(
        input_h, filter_h, spatial.stride_h, spatial.dilation_h, spatial.pre_padding_h, spatial.post_padding_h);
    const uint64_t output_w = outputExtent(
        input_w, filter_w, spatial.stride_w, spatial.dilation_w, spatial.pre_padding_w, spatial.post_padding_w);
    if (groups == 0 || c_count % groups != 0 || k_count % groups != 0)
        throw std::runtime_error("Grouped Conv2D dgrad reference received invalid channel geometry.");
    const uint64_t c_per_group = c_count / groups;
    const uint64_t k_per_group = k_count / groups;
    std::vector<float> grad_input(n_count * c_count * input_h * input_w, 0.0F);

    for (uint64_t n = 0; n < n_count; ++n) {
        for (uint64_t k = 0; k < k_count; ++k) {
            const uint64_t group = k / k_per_group;
            for (uint64_t oh = 0; oh < output_h; ++oh) {
                for (uint64_t ow = 0; ow < output_w; ++ow) {
                    const float dy = grad_output[nchwIndex(n, k, oh, ow, k_count, output_h, output_w)];
                    for (uint64_t local_c = 0; local_c < c_per_group; ++local_c) {
                        const uint64_t c = group * c_per_group + local_c;
                        for (uint64_t r = 0; r < filter_h; ++r) {
                            for (uint64_t s = 0; s < filter_w; ++s) {
                                const int64_t ih = static_cast<int64_t>(oh * spatial.stride_h + r * spatial.dilation_h) -
                                                   spatial.pre_padding_h;
                                const int64_t iw = static_cast<int64_t>(ow * spatial.stride_w + s * spatial.dilation_w) -
                                                   spatial.pre_padding_w;
                                if (ih < 0 || iw < 0 || ih >= static_cast<int64_t>(input_h) || iw >= static_cast<int64_t>(input_w)) {
                                    continue;
                                }
                                const size_t input_idx = nchwIndex(n, c, ih, iw, c_count, input_h, input_w);
                                const size_t filter_idx = nchwIndex(k, local_c, r, s, c_per_group, filter_h, filter_w);
                                grad_input[input_idx] += dy * filter[filter_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_input;
}

std::vector<float> conv2dFilterGradReference(const std::vector<float>& input,
                                             const std::vector<float>& grad_output,
                                             uint64_t n_count,
                                             uint64_t c_count,
                                             uint64_t input_h,
                                             uint64_t input_w,
                                             uint64_t k_count,
                                             uint64_t filter_h,
                                             uint64_t filter_w,
                                             const ConvolutionSpatial2d& spatial,
                                             uint64_t groups = 1) {
    const uint64_t output_h = outputExtent(
        input_h, filter_h, spatial.stride_h, spatial.dilation_h, spatial.pre_padding_h, spatial.post_padding_h);
    const uint64_t output_w = outputExtent(
        input_w, filter_w, spatial.stride_w, spatial.dilation_w, spatial.pre_padding_w, spatial.post_padding_w);
    if (groups == 0 || c_count % groups != 0 || k_count % groups != 0)
        throw std::runtime_error("Grouped Conv2D wgrad reference received invalid channel geometry.");
    const uint64_t c_per_group = c_count / groups;
    const uint64_t k_per_group = k_count / groups;
    std::vector<float> grad_filter(k_count * c_per_group * filter_h * filter_w, 0.0F);

    for (uint64_t n = 0; n < n_count; ++n) {
        for (uint64_t k = 0; k < k_count; ++k) {
            const uint64_t group = k / k_per_group;
            for (uint64_t oh = 0; oh < output_h; ++oh) {
                for (uint64_t ow = 0; ow < output_w; ++ow) {
                    const float dy = grad_output[nchwIndex(n, k, oh, ow, k_count, output_h, output_w)];
                    for (uint64_t local_c = 0; local_c < c_per_group; ++local_c) {
                        const uint64_t c = group * c_per_group + local_c;
                        for (uint64_t r = 0; r < filter_h; ++r) {
                            for (uint64_t s = 0; s < filter_w; ++s) {
                                const int64_t ih = static_cast<int64_t>(oh * spatial.stride_h + r * spatial.dilation_h) -
                                                   spatial.pre_padding_h;
                                const int64_t iw = static_cast<int64_t>(ow * spatial.stride_w + s * spatial.dilation_w) -
                                                   spatial.pre_padding_w;
                                if (ih < 0 || iw < 0 || ih >= static_cast<int64_t>(input_h) || iw >= static_cast<int64_t>(input_w)) {
                                    continue;
                                }
                                const size_t input_idx = nchwIndex(n, c, ih, iw, c_count, input_h, input_w);
                                const size_t filter_idx = nchwIndex(k, local_c, r, s, c_per_group, filter_h, filter_w);
                                grad_filter[filter_idx] += dy * input[input_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_filter;
}

Outputs asymmetricConv2dOutputs(const ConvolutionSpatial2d& spatial, uint64_t groups = 1) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);
    return Expression::outputs(
        {{"output", Expression::conv2d(input, filter, spatial, DataType::FP32, DataType::FP32, groups)}});
}

}  // namespace

TEST(ExpressionConv2dAsymmetricPadding, ForwardMatchesIndependentCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t n_count = 1;
    constexpr uint64_t c_count = 2;
    constexpr uint64_t input_h = 3;
    constexpr uint64_t input_w = 4;
    constexpr uint64_t k_count = 2;
    constexpr uint64_t filter_h = 2;
    constexpr uint64_t filter_w = 3;

    ConvolutionSpatial2d spatial;
    spatial.stride_h = 1;
    spatial.stride_w = 2;
    spatial.dilation_h = 1;
    spatial.dilation_w = 1;
    spatial.pre_padding_h = 1;
    spatial.post_padding_h = 2;
    spatial.pre_padding_w = 3;
    spatial.post_padding_w = 0;

    std::vector<float> input(n_count * c_count * input_h * input_w);
    std::vector<float> filter(k_count * c_count * filter_h * filter_w);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25F;
    }
    for (size_t i = 0; i < filter.size(); ++i) {
        filter[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.2F;
    }

    Tensor input_gpu = makeGpuTensor({n_count, c_count, input_h, input_w}, input, stream);
    Tensor filter_gpu = makeGpuTensor({k_count, c_count, filter_h, filter_w}, filter, stream);

    FusedEquation equation = FusedEquation::compile(asymmetricConv2dOutputs(spatial).physicalOutputs(), 0);
    StampedExecutionPlan plan = equation.stamp({{"input", input_gpu}, {"filter", filter_gpu}}, stream);
    plan.run();
    stream.synchronize();

    const uint64_t expected_h = outputExtent(input_h, filter_h, spatial.stride_h, spatial.dilation_h, 1, 2);
    const uint64_t expected_w = outputExtent(input_w, filter_w, spatial.stride_w, spatial.dilation_w, 3, 0);
    EXPECT_EQ(plan.output("output").getDimensions(), (std::vector<uint64_t>{n_count, k_count, expected_h, expected_w}));
    expectAllClose(copyToCpu(plan.output("output"), stream),
                   conv2dForwardReference(input,
                                          filter,
                                          n_count,
                                          c_count,
                                          input_h,
                                          input_w,
                                          k_count,
                                          filter_h,
                                          filter_w,
                                          spatial));
}

TEST(ExpressionConv2dAsymmetricPadding, DataAndFilterGradientsMatchIndependentCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t n_count = 1;
    constexpr uint64_t c_count = 2;
    constexpr uint64_t input_h = 3;
    constexpr uint64_t input_w = 4;
    constexpr uint64_t k_count = 2;
    constexpr uint64_t filter_h = 2;
    constexpr uint64_t filter_w = 3;

    ConvolutionSpatial2d spatial;
    spatial.stride_h = 1;
    spatial.stride_w = 2;
    spatial.dilation_h = 1;
    spatial.dilation_w = 1;
    spatial.pre_padding_h = 1;
    spatial.post_padding_h = 2;
    spatial.pre_padding_w = 3;
    spatial.post_padding_w = 0;

    const uint64_t output_h = outputExtent(input_h, filter_h, spatial.stride_h, spatial.dilation_h, 1, 2);
    const uint64_t output_w = outputExtent(input_w, filter_w, spatial.stride_w, spatial.dilation_w, 3, 0);

    std::vector<float> input(n_count * c_count * input_h * input_w);
    std::vector<float> filter(k_count * c_count * filter_h * filter_w);
    std::vector<float> grad_output(n_count * k_count * output_h * output_w);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.3F;
    }
    for (size_t i = 0; i < filter.size(); ++i) {
        filter[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.4F;
    }
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_output[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.1F;
    }

    Tensor input_gpu = makeGpuTensor({n_count, c_count, input_h, input_w}, input, stream);
    Tensor filter_gpu = makeGpuTensor({k_count, c_count, filter_h, filter_w}, filter, stream);
    Tensor grad_output_gpu = makeGpuTensor({n_count, k_count, output_h, output_w}, grad_output, stream);

    FusedEquation forward = FusedEquation::compile(asymmetricConv2dOutputs(spatial).physicalOutputs(), 0);
    FusedEquation backward = forward.compileBackward({"input", "filter"}, std::optional<std::string>{"doutput"});
    StampedExecutionPlan plan = backward.stamp(
        {{"input", input_gpu}, {"filter", filter_gpu}, {"doutput", grad_output_gpu}}, stream);
    plan.run();
    stream.synchronize();

    expectAllClose(copyToCpu(plan.output("input_grad"), stream),
                   conv2dDataGradReference(grad_output,
                                           filter,
                                           n_count,
                                           c_count,
                                           input_h,
                                           input_w,
                                           k_count,
                                           filter_h,
                                           filter_w,
                                           spatial));
    expectAllClose(copyToCpu(plan.output("filter_grad"), stream),
                   conv2dFilterGradReference(input,
                                             grad_output,
                                             n_count,
                                             c_count,
                                             input_h,
                                             input_w,
                                             k_count,
                                             filter_h,
                                             filter_w,
                                             spatial));
}


TEST(ExpressionConv2dGrouped, ForwardDataGradAndFilterGradMatchIndependentCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t n_count = 2;
    constexpr uint64_t c_count = 4;
    constexpr uint64_t input_h = 5;
    constexpr uint64_t input_w = 6;
    constexpr uint64_t k_count = 6;
    constexpr uint64_t groups = 2;
    constexpr uint64_t filter_h = 3;
    constexpr uint64_t filter_w = 2;
    constexpr uint64_t c_per_group = c_count / groups;

    ConvolutionSpatial2d spatial;
    spatial.stride_h = 1;
    spatial.stride_w = 2;
    spatial.dilation_h = 2;
    spatial.dilation_w = 1;
    spatial.pre_padding_h = 2;
    spatial.post_padding_h = 1;
    spatial.pre_padding_w = 1;
    spatial.post_padding_w = 0;

    const uint64_t output_h = outputExtent(input_h,
                                           filter_h,
                                           spatial.stride_h,
                                           spatial.dilation_h,
                                           spatial.pre_padding_h,
                                           spatial.post_padding_h);
    const uint64_t output_w = outputExtent(input_w,
                                           filter_w,
                                           spatial.stride_w,
                                           spatial.dilation_w,
                                           spatial.pre_padding_w,
                                           spatial.post_padding_w);

    std::vector<float> input(n_count * c_count * input_h * input_w);
    std::vector<float> filter(k_count * c_per_group * filter_h * filter_w);
    std::vector<float> grad_output(n_count * k_count * output_h * output_w);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int>((i * 3) % 17) - 8) * 0.075F;
    for (size_t i = 0; i < filter.size(); ++i)
        filter[i] = static_cast<float>(static_cast<int>((i * 5) % 13) - 6) * 0.09F;
    for (size_t i = 0; i < grad_output.size(); ++i)
        grad_output[i] = static_cast<float>(static_cast<int>((i * 7) % 19) - 9) * 0.06F;

    Tensor input_gpu = makeGpuTensor({n_count, c_count, input_h, input_w}, input, stream);
    Tensor filter_gpu = makeGpuTensor({k_count, c_per_group, filter_h, filter_w}, filter, stream);
    Tensor grad_output_gpu = makeGpuTensor({n_count, k_count, output_h, output_w}, grad_output, stream);

    FusedEquation forward = FusedEquation::compile(asymmetricConv2dOutputs(spatial, groups).physicalOutputs(), 0);
    StampedExecutionPlan forward_plan = forward.stamp({{"input", input_gpu}, {"filter", filter_gpu}}, stream);
    forward_plan.run();
    stream.synchronize();
    expectAllClose(copyToCpu(forward_plan.output("output"), stream),
                   conv2dForwardReference(input,
                                          filter,
                                          n_count,
                                          c_count,
                                          input_h,
                                          input_w,
                                          k_count,
                                          filter_h,
                                          filter_w,
                                          spatial,
                                          groups));

    FusedEquation backward = forward.compileBackward({"input", "filter"}, std::optional<std::string>{"doutput"});
    StampedExecutionPlan backward_plan = backward.stamp(
        {{"input", input_gpu}, {"filter", filter_gpu}, {"doutput", grad_output_gpu}}, stream);
    backward_plan.run();
    stream.synchronize();
    expectAllClose(copyToCpu(backward_plan.output("input_grad"), stream),
                   conv2dDataGradReference(grad_output,
                                           filter,
                                           n_count,
                                           c_count,
                                           input_h,
                                           input_w,
                                           k_count,
                                           filter_h,
                                           filter_w,
                                           spatial,
                                           groups));
    expectAllClose(copyToCpu(backward_plan.output("filter_grad"), stream),
                   conv2dFilterGradReference(input,
                                             grad_output,
                                             n_count,
                                             c_count,
                                             input_h,
                                             input_w,
                                             k_count,
                                             filter_h,
                                             filter_w,
                                             spatial,
                                             groups));
}
