#include "Utilities/Expression/ConvolutionSpatial.h"
#include "Utilities/Expression/ConvolutionKernelValidation.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/EquationCompiler.h"
#include "Utilities/Expression/ExpressionDTypeResolution.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {

#define REQUIRE_CUDA_DEVICE()                                                                                           \
    do {                                                                                                                 \
        int count = 0;                                                                                                   \
        const cudaError_t status = cudaGetDeviceCount(&count);                                                          \
        if (status != cudaSuccess || count <= 0) {                                                                       \
            GTEST_SKIP() << "CUDA device is required for dense Conv1D execution tests.";                               \
        }                                                                                                                \
    } while (false)

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t numel(const std::vector<uint64_t>& dims) {
    uint64_t result = 1;
    for (uint64_t dim : dims)
        result *= dim;
    return result;
}

Tensor makeGpuTensor(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    if (numel(dims) != values.size())
        throw std::runtime_error("makeGpuTensor value count mismatch.");
    Tensor cpu(cpuPlacement, TensorDescriptor(DataType::FP32, dims));
    auto* ptr = static_cast<float*>(cpu.getMemPtr());
    for (size_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
    Tensor gpu(gpuPlacement, TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(cpu, stream);
    stream.synchronize();
    return gpu;
}

std::vector<float> copyToCpu(const Tensor& gpu, Stream& stream) {
    Tensor cpu(cpuPlacement, TensorDescriptor(gpu.getDataType(), gpu.getDimensions()));
    cpu.copyFromAsync(gpu, stream);
    stream.synchronize();
    const auto* ptr = static_cast<const float*>(cpu.getMemPtr());
    return std::vector<float>(ptr, ptr + numel(cpu.getDimensions()));
}

void expectClose(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance = 1.0e-4F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "index " << i;
}

uint64_t outputWidth(uint64_t input_width, uint64_t kernel_width, const ConvolutionSpatial1d& spatial) {
    const int64_t effective = static_cast<int64_t>(spatial.dilation) * (static_cast<int64_t>(kernel_width) - 1) + 1;
    const int64_t numerator = static_cast<int64_t>(input_width) + spatial.pre_padding + spatial.post_padding - effective;
    if (numerator < 0)
        throw std::runtime_error("Conv1D reference produced negative output extent.");
    return static_cast<uint64_t>(numerator / spatial.stride + 1);
}

size_t ncwIndex(uint64_t n, uint64_t c, uint64_t w, uint64_t channels, uint64_t width) {
    return static_cast<size_t>((n * channels + c) * width + w);
}

std::vector<float> forwardReference(const std::vector<float>& input,
                                    const std::vector<float>& filter,
                                    uint64_t n_count,
                                    uint64_t c_count,
                                    uint64_t input_width,
                                    uint64_t k_count,
                                    uint64_t kernel_width,
                                    const ConvolutionSpatial1d& spatial,
                                    uint64_t groups = 1) {
    const uint64_t out_width = outputWidth(input_width, kernel_width, spatial);
    if (groups == 0 || c_count % groups != 0 || k_count % groups != 0)
        throw std::runtime_error("Grouped Conv1D reference received invalid channel geometry.");
    const uint64_t c_per_group = c_count / groups;
    const uint64_t k_per_group = k_count / groups;
    std::vector<float> output(n_count * k_count * out_width, 0.0F);
    for (uint64_t n = 0; n < n_count; ++n) {
        for (uint64_t k = 0; k < k_count; ++k) {
            const uint64_t group = k / k_per_group;
            for (uint64_t ow = 0; ow < out_width; ++ow) {
                float sum = 0.0F;
                for (uint64_t local_c = 0; local_c < c_per_group; ++local_c) {
                    const uint64_t c = group * c_per_group + local_c;
                    for (uint64_t r = 0; r < kernel_width; ++r) {
                        const int64_t iw = static_cast<int64_t>(ow * spatial.stride + r * spatial.dilation) - spatial.pre_padding;
                        if (iw < 0 || iw >= static_cast<int64_t>(input_width))
                            continue;
                        sum += input[ncwIndex(n, c, static_cast<uint64_t>(iw), c_count, input_width)] *
                               filter[ncwIndex(k, local_c, r, c_per_group, kernel_width)];
                    }
                }
                output[ncwIndex(n, k, ow, k_count, out_width)] = sum;
            }
        }
    }
    return output;
}

std::vector<float> dataGradReference(const std::vector<float>& grad_output,
                                     const std::vector<float>& filter,
                                     uint64_t n_count,
                                     uint64_t c_count,
                                     uint64_t input_width,
                                     uint64_t k_count,
                                     uint64_t kernel_width,
                                     const ConvolutionSpatial1d& spatial,
                                     uint64_t groups = 1) {
    const uint64_t out_width = outputWidth(input_width, kernel_width, spatial);
    if (groups == 0 || c_count % groups != 0 || k_count % groups != 0)
        throw std::runtime_error("Grouped Conv1D dgrad reference received invalid channel geometry.");
    const uint64_t c_per_group = c_count / groups;
    const uint64_t k_per_group = k_count / groups;
    std::vector<float> grad_input(n_count * c_count * input_width, 0.0F);
    for (uint64_t n = 0; n < n_count; ++n) {
        for (uint64_t k = 0; k < k_count; ++k) {
            const uint64_t group = k / k_per_group;
            for (uint64_t ow = 0; ow < out_width; ++ow) {
                const float dy = grad_output[ncwIndex(n, k, ow, k_count, out_width)];
                for (uint64_t local_c = 0; local_c < c_per_group; ++local_c) {
                    const uint64_t c = group * c_per_group + local_c;
                    for (uint64_t r = 0; r < kernel_width; ++r) {
                        const int64_t iw = static_cast<int64_t>(ow * spatial.stride + r * spatial.dilation) - spatial.pre_padding;
                        if (iw < 0 || iw >= static_cast<int64_t>(input_width))
                            continue;
                        grad_input[ncwIndex(n, c, static_cast<uint64_t>(iw), c_count, input_width)] +=
                            dy * filter[ncwIndex(k, local_c, r, c_per_group, kernel_width)];
                    }
                }
            }
        }
    }
    return grad_input;
}

std::vector<float> filterGradReference(const std::vector<float>& input,
                                       const std::vector<float>& grad_output,
                                       uint64_t n_count,
                                       uint64_t c_count,
                                       uint64_t input_width,
                                       uint64_t k_count,
                                       uint64_t kernel_width,
                                       const ConvolutionSpatial1d& spatial,
                                       uint64_t groups = 1) {
    const uint64_t out_width = outputWidth(input_width, kernel_width, spatial);
    if (groups == 0 || c_count % groups != 0 || k_count % groups != 0)
        throw std::runtime_error("Grouped Conv1D wgrad reference received invalid channel geometry.");
    const uint64_t c_per_group = c_count / groups;
    const uint64_t k_per_group = k_count / groups;
    std::vector<float> grad_filter(k_count * c_per_group * kernel_width, 0.0F);
    for (uint64_t n = 0; n < n_count; ++n) {
        for (uint64_t k = 0; k < k_count; ++k) {
            const uint64_t group = k / k_per_group;
            for (uint64_t ow = 0; ow < out_width; ++ow) {
                const float dy = grad_output[ncwIndex(n, k, ow, k_count, out_width)];
                for (uint64_t local_c = 0; local_c < c_per_group; ++local_c) {
                    const uint64_t c = group * c_per_group + local_c;
                    for (uint64_t r = 0; r < kernel_width; ++r) {
                        const int64_t iw = static_cast<int64_t>(ow * spatial.stride + r * spatial.dilation) - spatial.pre_padding;
                        if (iw < 0 || iw >= static_cast<int64_t>(input_width))
                            continue;
                        grad_filter[ncwIndex(k, local_c, r, c_per_group, kernel_width)] +=
                            dy * input[ncwIndex(n, c, static_cast<uint64_t>(iw), c_count, input_width)];
                    }
                }
            }
        }
    }
    return grad_filter;
}

Outputs conv1dOutputs(const ConvolutionSpatial1d& spatial, uint64_t groups = 1) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);
    return Expression::outputs({{"output", Expression::conv1d(input, filter, spatial, DataType::FP32, DataType::FP32, groups)}});
}

Outputs explicitSingletonConv2dOutputs(const ConvolutionSpatial1d& spatial, uint64_t groups = 1) {
    const Expression input = Expression::input("input", DataType::FP32, DataType::FP32);
    const Expression filter = Expression::input("filter", DataType::FP32, DataType::FP32);
    ConvolutionSpatial2d spatial2d;
    spatial2d.stride_w = spatial.stride;
    spatial2d.dilation_w = spatial.dilation;
    spatial2d.pre_padding_w = spatial.pre_padding;
    spatial2d.post_padding_w = spatial.post_padding;
    constexpr uint64_t copy_dim = 0;
    constexpr uint64_t infer_dim = std::numeric_limits<uint64_t>::max();
    const Expression output =
        Expression::conv2d(input.reshape({copy_dim, copy_dim, 1, infer_dim}),
                           filter.reshape({copy_dim, copy_dim, 1, infer_dim}),
                           spatial2d,
                           DataType::FP32,
                           DataType::FP32,
                           groups)
            .reshape({copy_dim, copy_dim, infer_dim});
    return Expression::outputs({{"output", output}});
}

}  // namespace

TEST(ConvolutionSpatial1d, PaddingResolversProduceModernGeometry) {
    EXPECT_EQ(ConvolutionSpatial1d::valid(2, 3),
              (ConvolutionSpatial1d{.stride = 2, .dilation = 3, .pre_padding = 0, .post_padding = 0}));
    EXPECT_EQ(ConvolutionSpatial1d::explicitPadding(2, 5, 3, 2),
              (ConvolutionSpatial1d{.stride = 3, .dilation = 2, .pre_padding = 2, .post_padding = 5}));
    EXPECT_EQ(ConvolutionSpatial1d::causal(4, 1, 3),
              (ConvolutionSpatial1d{.stride = 1, .dilation = 3, .pre_padding = 9, .post_padding = 0}));
    // I=8, K=3, S=2 => O=4 and total padding=1; SAME_UPPER puts the extra cell on the right.
    EXPECT_EQ(ConvolutionSpatial1d::sameUpper(8, 3, 2, 1),
              (ConvolutionSpatial1d{.stride = 2, .dilation = 1, .pre_padding = 0, .post_padding = 1}));
}

TEST(ExpressionConv1d, CanonicalGraphEqualsExplicitSingletonHeightConv2d) {
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::explicitPadding(2, 1, 2, 2);
    EXPECT_EQ(canonicalize(conv1dOutputs(spatial).physicalOutputs()),
              canonicalize(explicitSingletonConv2dOutputs(spatial).physicalOutputs()));
}

TEST(ExpressionConv1d, SingletonHeightReshapesRemainStorageAliasesAroundOneConvolutionStage) {
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::causal(5, 1, 2);
    PhysicalOutputs outputs = conv1dOutputs(spatial).physicalOutputs();
    resolveOutputsDTypesInPlace(outputs, {DataType::FP32, DataType::FP32});
    const std::vector<PhysicalExecutionStage> stages = EquationCompiler::splitAtReductionBoundaries(outputs);
    ASSERT_EQ(stages.size(), 1u);
    EXPECT_EQ(stages.front().kind, PhysicalExecutionStage::Kind::Convolution);
}

TEST(ExpressionConv1d, ForwardAndBackwardMatchIndependentCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    cudaStream_t cudnn_stream = nullptr;
    ASSERT_EQ(cudnnGetStream(stream.getCudnnHandle(), &cudnn_stream), CUDNN_STATUS_SUCCESS);
    ASSERT_EQ(cudnn_stream, stream.getStream());

    constexpr uint64_t n_count = 2;
    constexpr uint64_t c_count = 2;
    constexpr uint64_t input_width = 9;
    constexpr uint64_t k_count = 3;
    constexpr uint64_t kernel_width = 3;
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::explicitPadding(2, 1, 2, 2);
    const uint64_t out_width = outputWidth(input_width, kernel_width, spatial);

    std::vector<float> input(n_count * c_count * input_width);
    std::vector<float> filter(k_count * c_count * kernel_width);
    std::vector<float> grad_output(n_count * k_count * out_width);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125F;
    for (size_t i = 0; i < filter.size(); ++i)
        filter[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.2F;
    for (size_t i = 0; i < grad_output.size(); ++i)
        grad_output[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.1F;

    Tensor input_gpu = makeGpuTensor({n_count, c_count, input_width}, input, stream);
    Tensor filter_gpu = makeGpuTensor({k_count, c_count, kernel_width}, filter, stream);
    Tensor grad_gpu = makeGpuTensor({n_count, k_count, out_width}, grad_output, stream);

    FusedEquation forward = FusedEquation::compile(conv1dOutputs(spatial).physicalOutputs(), 0);
    const std::vector<float> expected_forward =
        forwardReference(input, filter, n_count, c_count, input_width, k_count, kernel_width, spatial);
    // Repeated stamping exercises placement-time cuDNN Frontend autotuning. Candidate
    // selection mutates the scratch Frontend graph; the retained production graph must
    // be rebuilt pristine with the winning plan and remain immutable thereafter. This
    // catches the intermittent corruption that occurs when an autotuned scratch graph
    // is reused directly for real execution.
    for (int repetition = 0; repetition < 8; ++repetition) {
        SCOPED_TRACE("forward autotune repetition=" + std::to_string(repetition));
        StampedExecutionPlan forward_plan = forward.stamp({{"input", input_gpu}, {"filter", filter_gpu}}, stream);
        forward_plan.run();
        stream.synchronize();
        EXPECT_EQ(forward_plan.output("output").getDimensions(), (std::vector<uint64_t>{n_count, k_count, out_width}));
        expectClose(copyToCpu(forward_plan.output("output"), stream), expected_forward);
    }

    FusedEquation backward = forward.compileBackward({"input", "filter"}, std::optional<std::string>{"doutput"});
    const std::vector<float> expected_input_grad =
        dataGradReference(grad_output, filter, n_count, c_count, input_width, k_count, kernel_width, spatial);
    const std::vector<float> expected_filter_grad =
        filterGradReference(input, grad_output, n_count, c_count, input_width, k_count, kernel_width, spatial);
    // Backward contains independent dgrad and wgrad roots. Thor deliberately keeps
    // each operation-local cuDNN Frontend plan in its stamping execution domain, so
    // neither root may be migrated to a helper stream by the DAG scheduler.
    for (int repetition = 0; repetition < 8; ++repetition) {
        SCOPED_TRACE("backward autotune repetition=" + std::to_string(repetition));
        StampedExecutionPlan backward_plan = backward.stamp(
            {{"input", input_gpu}, {"filter", filter_gpu}, {"doutput", grad_gpu}}, stream);
        EXPECT_EQ(backward_plan.stageLaneIndices(),
                  std::vector<uint32_t>(backward_plan.stageKindNames().size(), 0));
        backward_plan.run();
        stream.synchronize();
        expectClose(copyToCpu(backward_plan.output("input_grad"), stream), expected_input_grad);
        expectClose(copyToCpu(backward_plan.output("filter_grad"), stream), expected_filter_grad);
    }
}

TEST(ExpressionConv1d, IndependentKernelValidatorMatchesCpuReferenceAndRejectsCorruption) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    constexpr uint64_t n_count = 2;
    constexpr uint64_t c_count = 4;
    constexpr uint64_t k_count = 6;
    constexpr uint64_t input_width = 8;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t groups = 2;
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::explicitPadding(2, 1, 2, 2);
    const uint64_t out_width = outputWidth(input_width, kernel_width, spatial);
    const uint64_t c_per_group = c_count / groups;

    Tensor input_gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {n_count, c_count, 1, input_width}));
    Tensor filter_gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {k_count, c_per_group, 1, kernel_width}));
    Tensor grad_gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {n_count, k_count, 1, out_width}));
    fillConvolutionKernelValidationTensor(input_gpu, 17, stream);
    fillConvolutionKernelValidationTensor(filter_gpu, 29, stream);
    fillConvolutionKernelValidationTensor(grad_gpu, 43, stream);
    stream.synchronize();

    const std::vector<float> input = copyToCpu(input_gpu, stream);
    const std::vector<float> filter = copyToCpu(filter_gpu, stream);
    const std::vector<float> grad_output = copyToCpu(grad_gpu, stream);

    ConvolutionKernelValidationResult preservation_result =
        validateConvolutionKernelValidationInputUnchanged(input_gpu, 17, stream);
    EXPECT_TRUE(preservation_result.passed) << describeConvolutionKernelValidationFailure(preservation_result);
    std::vector<float> corrupt_input = input;
    corrupt_input.at(corrupt_input.size() / 2) += 1.0F;
    Tensor corrupt_input_gpu = makeGpuTensor({n_count, c_count, 1, input_width}, corrupt_input, stream);
    preservation_result = validateConvolutionKernelValidationInputUnchanged(corrupt_input_gpu, 17, stream);
    EXPECT_FALSE(preservation_result.passed);
    EXPECT_GT(preservation_result.bad_elements, 0u);

    ConvolutionKernelValidationSpec spec;
    spec.is_3d = false;
    spec.groups = groups;
    spec.stride_h = 1;
    spec.stride_w = spatial.stride;
    spec.pre_padding_h = 0;
    spec.pre_padding_w = spatial.pre_padding;
    spec.dilation_h = 1;
    spec.dilation_w = spatial.dilation;
    spec.compute_dtype = DataType::FP32;

    const std::vector<float> expected_forward =
        forwardReference(input, filter, n_count, c_count, input_width, k_count, kernel_width, spatial, groups);
    Tensor forward_candidate =
        makeGpuTensor({n_count, k_count, 1, out_width}, expected_forward, stream);
    spec.kind = ConvolutionKernelValidationKind::Forward;
    ConvolutionKernelValidationResult forward_result =
        validateConvolutionKernelOutput(input_gpu, filter_gpu, forward_candidate, spec, stream);
    EXPECT_TRUE(forward_result.passed) << describeConvolutionKernelValidationFailure(forward_result);
    EXPECT_EQ(forward_result.checked_elements, expected_forward.size());

    std::vector<float> corrupt_forward = expected_forward;
    corrupt_forward.at(corrupt_forward.size() / 2) += 1.0F;
    Tensor corrupt_forward_candidate =
        makeGpuTensor({n_count, k_count, 1, out_width}, corrupt_forward, stream);
    forward_result = validateConvolutionKernelOutput(input_gpu, filter_gpu, corrupt_forward_candidate, spec, stream);
    EXPECT_FALSE(forward_result.passed);
    EXPECT_GT(forward_result.bad_elements, 0u);
    EXPECT_FLOAT_EQ(forward_result.first_bad_tolerance, 0.0F)
        << "small FP32-compute validation canaries must be exact rather than tolerance-based";

    const std::vector<float> expected_dgrad =
        dataGradReference(grad_output, filter, n_count, c_count, input_width, k_count, kernel_width, spatial, groups);
    Tensor dgrad_candidate = makeGpuTensor({n_count, c_count, 1, input_width}, expected_dgrad, stream);
    spec.kind = ConvolutionKernelValidationKind::BackwardData;
    ConvolutionKernelValidationResult dgrad_result =
        validateConvolutionKernelOutput(filter_gpu, grad_gpu, dgrad_candidate, spec, stream);
    EXPECT_TRUE(dgrad_result.passed) << describeConvolutionKernelValidationFailure(dgrad_result);
    EXPECT_EQ(dgrad_result.checked_elements, expected_dgrad.size());

    std::vector<float> corrupt_dgrad = expected_dgrad;
    corrupt_dgrad.at(corrupt_dgrad.size() / 3) -= 1.0F;
    Tensor corrupt_dgrad_candidate = makeGpuTensor({n_count, c_count, 1, input_width}, corrupt_dgrad, stream);
    dgrad_result = validateConvolutionKernelOutput(filter_gpu, grad_gpu, corrupt_dgrad_candidate, spec, stream);
    EXPECT_FALSE(dgrad_result.passed);
    EXPECT_GT(dgrad_result.bad_elements, 0u);

    const std::vector<float> expected_wgrad =
        filterGradReference(input, grad_output, n_count, c_count, input_width, k_count, kernel_width, spatial, groups);
    Tensor wgrad_candidate = makeGpuTensor({k_count, c_per_group, 1, kernel_width}, expected_wgrad, stream);
    spec.kind = ConvolutionKernelValidationKind::BackwardFilter;
    ConvolutionKernelValidationResult wgrad_result =
        validateConvolutionKernelOutput(input_gpu, grad_gpu, wgrad_candidate, spec, stream);
    EXPECT_TRUE(wgrad_result.passed) << describeConvolutionKernelValidationFailure(wgrad_result);
    EXPECT_EQ(wgrad_result.checked_elements, expected_wgrad.size());

    std::vector<float> corrupt_wgrad = expected_wgrad;
    corrupt_wgrad.at(corrupt_wgrad.size() / 4) += 1.0F;
    Tensor corrupt_wgrad_candidate = makeGpuTensor({k_count, c_per_group, 1, kernel_width}, corrupt_wgrad, stream);
    wgrad_result = validateConvolutionKernelOutput(input_gpu, grad_gpu, corrupt_wgrad_candidate, spec, stream);
    EXPECT_FALSE(wgrad_result.passed);
    EXPECT_GT(wgrad_result.bad_elements, 0u);
}


TEST(ExpressionConv1d, IndependentKernelValidatorCooperativeForwardAndDgradMatchCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // 192 channels/group * 3 filter taps = 576 reduction terms.  This is
    // intentionally above the validator's serial-reduction threshold so both
    // forward and dgrad exercise the block-cooperative int32 reference path.
    constexpr uint64_t n_count = 1;
    constexpr uint64_t c_count = 384;
    constexpr uint64_t k_count = 384;
    constexpr uint64_t input_width = 5;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t groups = 2;
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::explicitPadding(1, 1, 1, 1);
    const uint64_t out_width = outputWidth(input_width, kernel_width, spatial);
    const uint64_t c_per_group = c_count / groups;

    Tensor input_gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {n_count, c_count, 1, input_width}));
    Tensor filter_gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {k_count, c_per_group, 1, kernel_width}));
    Tensor grad_gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {n_count, k_count, 1, out_width}));
    fillConvolutionKernelValidationTensor(input_gpu, 101, stream);
    fillConvolutionKernelValidationTensor(filter_gpu, 103, stream);
    fillConvolutionKernelValidationTensor(grad_gpu, 107, stream);
    stream.synchronize();

    const std::vector<float> input = copyToCpu(input_gpu, stream);
    const std::vector<float> filter = copyToCpu(filter_gpu, stream);
    const std::vector<float> grad_output = copyToCpu(grad_gpu, stream);

    ConvolutionKernelValidationSpec spec;
    spec.is_3d = false;
    spec.groups = groups;
    spec.stride_h = 1;
    spec.stride_w = spatial.stride;
    spec.pre_padding_h = 0;
    spec.pre_padding_w = spatial.pre_padding;
    spec.dilation_h = 1;
    spec.dilation_w = spatial.dilation;
    spec.compute_dtype = DataType::FP32;

    const std::vector<float> expected_forward =
        forwardReference(input, filter, n_count, c_count, input_width, k_count, kernel_width, spatial, groups);
    Tensor forward_candidate = makeGpuTensor({n_count, k_count, 1, out_width}, expected_forward, stream);
    spec.kind = ConvolutionKernelValidationKind::Forward;
    const ConvolutionKernelValidationResult forward_result =
        validateConvolutionKernelOutput(input_gpu, filter_gpu, forward_candidate, spec, stream);
    EXPECT_TRUE(forward_result.passed) << describeConvolutionKernelValidationFailure(forward_result);

    const std::vector<float> expected_dgrad =
        dataGradReference(grad_output, filter, n_count, c_count, input_width, k_count, kernel_width, spatial, groups);
    Tensor dgrad_candidate = makeGpuTensor({n_count, c_count, 1, input_width}, expected_dgrad, stream);
    spec.kind = ConvolutionKernelValidationKind::BackwardData;
    const ConvolutionKernelValidationResult dgrad_result =
        validateConvolutionKernelOutput(filter_gpu, grad_gpu, dgrad_candidate, spec, stream);
    EXPECT_TRUE(dgrad_result.passed) << describeConvolutionKernelValidationFailure(dgrad_result);
}


TEST(ExpressionConv1d, IndependentKernelValidatorCooperativeWgradMatchesCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    // 4 batches * 130 output positions = 520 reduction terms, forcing the
    // backward-filter validator through the block-cooperative int32 path.
    constexpr uint64_t n_count = 4;
    constexpr uint64_t c_count = 4;
    constexpr uint64_t k_count = 6;
    constexpr uint64_t input_width = 130;
    constexpr uint64_t kernel_width = 3;
    constexpr uint64_t groups = 2;
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::explicitPadding(1, 1, 1, 1);
    const uint64_t out_width = outputWidth(input_width, kernel_width, spatial);
    ASSERT_EQ(out_width, 130u);
    const uint64_t c_per_group = c_count / groups;

    Tensor input_gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {n_count, c_count, 1, input_width}));
    Tensor grad_gpu(gpuPlacement, TensorDescriptor(DataType::FP32, {n_count, k_count, 1, out_width}));
    fillConvolutionKernelValidationTensor(input_gpu, 109, stream);
    fillConvolutionKernelValidationTensor(grad_gpu, 113, stream);
    stream.synchronize();

    const std::vector<float> input = copyToCpu(input_gpu, stream);
    const std::vector<float> grad_output = copyToCpu(grad_gpu, stream);
    const std::vector<float> expected_wgrad =
        filterGradReference(input, grad_output, n_count, c_count, input_width, k_count, kernel_width, spatial, groups);
    Tensor wgrad_candidate = makeGpuTensor({k_count, c_per_group, 1, kernel_width}, expected_wgrad, stream);

    ConvolutionKernelValidationSpec spec;
    spec.kind = ConvolutionKernelValidationKind::BackwardFilter;
    spec.is_3d = false;
    spec.groups = groups;
    spec.stride_h = 1;
    spec.stride_w = spatial.stride;
    spec.pre_padding_h = 0;
    spec.pre_padding_w = spatial.pre_padding;
    spec.dilation_h = 1;
    spec.dilation_w = spatial.dilation;
    spec.compute_dtype = DataType::FP32;

    const ConvolutionKernelValidationResult wgrad_result =
        validateConvolutionKernelOutput(input_gpu, grad_gpu, wgrad_candidate, spec, stream);
    EXPECT_TRUE(wgrad_result.passed) << describeConvolutionKernelValidationFailure(wgrad_result);
}

TEST(ExpressionConv1d, HandleBoundConvolutionPlanRejectsDifferentRuntimeStream) {
    REQUIRE_CUDA_DEVICE();
    Stream stamp_stream(0);
    Stream other_stream(0);

    constexpr uint64_t n_count = 1;
    constexpr uint64_t c_count = 2;
    constexpr uint64_t input_width = 5;
    constexpr uint64_t k_count = 3;
    constexpr uint64_t kernel_width = 3;
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::causal(kernel_width, 1, 1);

    std::vector<float> input(n_count * c_count * input_width, 0.25F);
    std::vector<float> filter(k_count * c_count * kernel_width, -0.125F);
    Tensor input_gpu = makeGpuTensor({n_count, c_count, input_width}, input, stamp_stream);
    Tensor filter_gpu = makeGpuTensor({k_count, c_count, kernel_width}, filter, stamp_stream);

    FusedEquation forward = FusedEquation::compile(conv1dOutputs(spatial).physicalOutputs(), 0);
    StampedExecutionPlan plan = forward.stamp({{"input", input_gpu}, {"filter", filter_gpu}}, stamp_stream);

    EXPECT_THROW(plan.runOn(other_stream), std::runtime_error);
    EXPECT_NO_THROW(plan.runOn(stamp_stream));
    stamp_stream.synchronize();
}

TEST(ExpressionConv1d, GroupedForwardBackwardAndDepthwiseMatchIndependentCpuReference) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);

    auto run_case = [&](uint64_t c_count, uint64_t k_count, uint64_t groups, const std::string& tag) {
        SCOPED_TRACE(tag);
        constexpr uint64_t n_count = 2;
        constexpr uint64_t input_width = 8;
        constexpr uint64_t kernel_width = 3;
        const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::causal(kernel_width, 1, 2);
        const uint64_t out_width = outputWidth(input_width, kernel_width, spatial);
        const uint64_t c_per_group = c_count / groups;

        std::vector<float> input(n_count * c_count * input_width);
        std::vector<float> filter(k_count * c_per_group * kernel_width);
        std::vector<float> grad_output(n_count * k_count * out_width);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<float>(static_cast<int>((i * 3) % 17) - 8) * 0.075F;
        for (size_t i = 0; i < filter.size(); ++i)
            filter[i] = static_cast<float>(static_cast<int>((i * 5) % 11) - 5) * 0.11F;
        for (size_t i = 0; i < grad_output.size(); ++i)
            grad_output[i] = static_cast<float>(static_cast<int>((i * 7) % 13) - 6) * 0.09F;

        Tensor input_gpu = makeGpuTensor({n_count, c_count, input_width}, input, stream);
        Tensor filter_gpu = makeGpuTensor({k_count, c_per_group, kernel_width}, filter, stream);
        Tensor grad_gpu = makeGpuTensor({n_count, k_count, out_width}, grad_output, stream);

        FusedEquation forward = FusedEquation::compile(conv1dOutputs(spatial, groups).physicalOutputs(), 0);
        StampedExecutionPlan forward_plan = forward.stamp({{"input", input_gpu}, {"filter", filter_gpu}}, stream);
        forward_plan.run();
        stream.synchronize();
        expectClose(copyToCpu(forward_plan.output("output"), stream),
                    forwardReference(input, filter, n_count, c_count, input_width, k_count, kernel_width, spatial, groups));

        FusedEquation backward = forward.compileBackward({"input", "filter"}, std::optional<std::string>{"doutput"});
        StampedExecutionPlan backward_plan =
            backward.stamp({{"input", input_gpu}, {"filter", filter_gpu}, {"doutput", grad_gpu}}, stream);
        backward_plan.run();
        stream.synchronize();
        expectClose(copyToCpu(backward_plan.output("input_grad"), stream),
                    dataGradReference(grad_output, filter, n_count, c_count, input_width, k_count, kernel_width, spatial, groups));
        expectClose(copyToCpu(backward_plan.output("filter_grad"), stream),
                    filterGradReference(input, grad_output, n_count, c_count, input_width, k_count, kernel_width, spatial, groups));
    };

    run_case(4, 6, 2, "groups=2");
    run_case(4, 4, 4, "depthwise");
}

TEST(ExpressionConv1d, GroupCountParticipatesInCanonicalIdentity) {
    const ConvolutionSpatial1d spatial = ConvolutionSpatial1d::causal(3, 1, 1);
    EXPECT_NE(canonicalize(conv1dOutputs(spatial, 1).physicalOutputs()),
              canonicalize(conv1dOutputs(spatial, 2).physicalOutputs()));
    EXPECT_EQ(canonicalize(conv1dOutputs(spatial, 2).physicalOutputs()),
              canonicalize(explicitSingletonConv2dOutputs(spatial, 2).physicalOutputs()));
}
