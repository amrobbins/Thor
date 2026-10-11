#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ThorImplementation;

namespace {
bool cudaAvailable() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Tensor uploadFloats(const std::vector<uint64_t>& dims, const std::vector<float>& data, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, dims));
    EXPECT_EQ(host.getArraySizeInBytes(), data.size() * sizeof(float));
    std::copy(data.begin(), data.end(), host.getMemPtr<float>());
    Tensor gpu(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), TensorDescriptor(DataType::FP32, dims));
    gpu.copyFromAsync(host, stream);
    stream.synchronize(); // Keep pageable host source alive until the transfer completes.
    return gpu;
}

std::vector<float> downloadFloats(const Tensor& gpu, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), gpu.getDescriptor());
    host.copyFromAsync(gpu, stream);
    stream.synchronize();
    const size_t count = host.getArraySizeInBytes() / sizeof(float);
    return std::vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + count);
}

FusedEquation backwardEquation(const std::vector<std::string>& wrt) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression gamma = Expression::input("gamma", DataType::FP32, DataType::FP32);
    const Expression beta = Expression::input("beta", DataType::FP32, DataType::FP32);
    const Expression y = Expression::layerNorm(x, gamma, beta, 128, 1.0e-5, DataType::FP32, DataType::FP32);
    return FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0)
        .compileBackward(wrt, "dy");
}
} // namespace

TEST(LayerNormBackwardDispatch1C2B2B, DenseThreeRoutesRunFromMatchingRealForward) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required";
    Stream stream(0);
    constexpr int outer = 3;
    constexpr int hidden = 128;
    std::vector<float> x(outer * hidden), dy(outer * hidden), gamma(hidden), beta(hidden);
    for (int i = 0; i < outer * hidden; ++i) {
        x[i] = 0.06f * static_cast<float>(i % 19) + 0.17f * static_cast<float>(i / hidden);
        dy[i] = 0.2f + 0.04f * static_cast<float>((i * 7) % 13);
    }
    for (int j = 0; j < hidden; ++j) {
        gamma[j] = 0.6f + 0.02f * static_cast<float>(j % 11);
        beta[j] = -0.2f + 0.01f * static_cast<float>(j % 5);
    }
    auto compiled = backwardEquation({"x", "gamma", "beta"});
    auto [forward, backward] = compiled.stampForwardBackwardPair(
        {{"x", uploadFloats({outer, hidden}, x, stream)},
         {"gamma", uploadFloats({hidden}, gamma, stream)},
         {"beta", uploadFloats({hidden}, beta, stream)},
         {"dy", uploadFloats({outer, hidden}, dy, stream)}}, stream);
    forward.run();
    backward.run();
    const auto actual_dx = downloadFloats(backward.output("x_grad"), stream);
    const auto actual_dgamma = downloadFloats(backward.output("gamma_grad"), stream);
    const auto actual_dbeta = downloadFloats(backward.output("beta_grad"), stream);
    ASSERT_EQ(actual_dx.size(), x.size());
    ASSERT_EQ(actual_dgamma.size(), gamma.size());
    ASSERT_EQ(actual_dbeta.size(), beta.size());
    std::vector<float> expected_dgamma(hidden, 0.0f), expected_dbeta(hidden, 0.0f);
    for (int r = 0; r < outer; ++r) {
        double mean = 0.0, variance = 0.0;
        for (int j = 0; j < hidden; ++j) mean += x[r * hidden + j];
        mean /= hidden;
        for (int j = 0; j < hidden; ++j) {
            const double centered = x[r * hidden + j] - mean;
            variance += centered * centered;
        }
        variance /= hidden;
        const double inverse_sd = 1.0 / std::sqrt(variance + 1.0e-5);
        double mean_dz = 0.0, mean_dz_xhat = 0.0;
        for (int j = 0; j < hidden; ++j) {
            const int i = r * hidden + j;
            const double xhat = (x[i] - mean) * inverse_sd;
            const double dz = dy[i] * gamma[j];
            expected_dgamma[j] += static_cast<float>(dy[i] * xhat);
            expected_dbeta[j] += dy[i];
            mean_dz += dz;
            mean_dz_xhat += dz * xhat;
        }
        mean_dz /= hidden;
        mean_dz_xhat /= hidden;
        for (int j = 0; j < hidden; ++j) {
            const int i = r * hidden + j;
            const double xhat = (x[i] - mean) * inverse_sd;
            const double dz = dy[i] * gamma[j];
            const double expected_dx = inverse_sd * (dz - mean_dz - xhat * mean_dz_xhat);
            EXPECT_NEAR(actual_dx[i], expected_dx, 4.0e-3) << "dx index " << i;
        }
    }
    for (int j = 0; j < hidden; ++j) {
        EXPECT_NEAR(actual_dgamma[j], expected_dgamma[j], 4.0e-3) << "dgamma index " << j;
        EXPECT_NEAR(actual_dbeta[j], expected_dbeta[j], 4.0e-3) << "dbeta index " << j;
    }
}

TEST(LayerNormBackwardDispatch1C2B2B, RequestedBiasOnlyDoesNotExposeOtherGradients) {
    if (!cudaAvailable()) GTEST_SKIP() << "CUDA device required";
    Stream stream(0);
    std::vector<float> x(256), dy(256, 0.75f), gamma(128, 1.2f), beta(128, 0.1f);
    for (int i = 0; i < 256; ++i) x[i] = static_cast<float>((i * 3) % 31) * 0.03f;
    auto backward = backwardEquation({"beta"});
    auto [forward_plan, backward_plan] = backward.stampForwardBackwardPair(
        {{"x", uploadFloats({2, 128}, x, stream)},
         {"gamma", uploadFloats({128}, gamma, stream)},
         {"beta", uploadFloats({128}, beta, stream)},
         {"dy", uploadFloats({2, 128}, dy, stream)}}, stream);
    forward_plan.run();
    backward_plan.run();
    const auto bias_grad = downloadFloats(backward_plan.output("beta_grad"), stream);
    ASSERT_EQ(bias_grad.size(), 128u);
    for (float v : bias_grad) EXPECT_NEAR(v, 1.5f, 2.0e-3f);
    EXPECT_THROW((void)backward_plan.output("x_grad"), std::exception);
    EXPECT_THROW((void)backward_plan.output("gamma_grad"), std::exception);
}
