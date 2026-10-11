#include "Utilities/Expression/PackedLayerNormBackwardKernel.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

bool hasCuda1E1() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Tensor upload1E1(const std::vector<uint64_t>& dims, const std::vector<float>& v, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, dims));
    if (host.getArraySizeInBytes() != sizeof(float) * v.size()) throw std::invalid_argument("test input size mismatch");
    std::copy(v.begin(), v.end(), host.getMemPtr<float>());
    Tensor device(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), host.getDescriptor());
    device.copyFromAsync(host, stream);
    stream.synchronize();
    return device;
}

std::vector<float> download1E1(const Tensor& device, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), device.getDescriptor());
    host.copyFromAsync(device, stream);
    stream.synchronize();
    const size_t n = host.getArraySizeInBytes() / sizeof(float);
    return {host.getMemPtr<float>(), host.getMemPtr<float>() + n};
}

void validateActivePrefix1E1(uint64_t capacity, uint64_t active) {
    Stream stream(0);
    constexpr uint64_t H = 128;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> x(capacity * H, nan), dy(capacity * H, nan);
    std::vector<float> mean(capacity, nan), inv(capacity, nan);
    std::vector<float> gamma(H);
    std::vector<double> reference_dx(active * H, 0.0);
    std::vector<double> reference_dgamma(H, 0.0), reference_dbeta(H, 0.0);
    for (uint64_t j = 0; j < H; ++j) gamma[j] = 0.7f + 0.02f * static_cast<float>(j % 11);
    for (uint64_t r = 0; r < active; ++r) {
        double mu = 0.0, variance = 0.0;
        for (uint64_t j = 0; j < H; ++j) {
            const uint64_t i = r * H + j;
            x[i] = 0.09f * static_cast<float>((13 * j + 7 * r) % 29) - 0.12f * static_cast<float>(r);
            dy[i] = 0.13f + 0.017f * static_cast<float>((5 * j + 3 * r) % 23);
            mu += x[i];
        }
        mu /= H;
        for (uint64_t j = 0; j < H; ++j) {
            const double delta = x[r * H + j] - mu;
            variance += delta * delta;
        }
        variance /= H;
        mean[r] = static_cast<float>(mu);
        inv[r] = static_cast<float>(1.0 / std::sqrt(variance + 1.0e-5));
        double sum_q = 0.0, sum_qz = 0.0;
        for (uint64_t j = 0; j < H; ++j) {
            const uint64_t i = r * H + j;
            const double z = (x[i] - mean[r]) * inv[r];
            const double q = dy[i] * gamma[j];
            sum_q += q;
            sum_qz += q * z;
            reference_dgamma[j] += dy[i] * z;
            reference_dbeta[j] += dy[i];
        }
        for (uint64_t j = 0; j < H; ++j) {
            const uint64_t i = r * H + j;
            const double z = (x[i] - mean[r]) * inv[r];
            const double q = dy[i] * gamma[j];
            reference_dx[i] = inv[r] * (q - sum_q / H - z * sum_qz / H);
        }
    }
    Tensor gpu_x = upload1E1({capacity, H}, x, stream);
    Tensor gpu_dy = upload1E1({capacity, H}, dy, stream);
    Tensor gpu_gamma = upload1E1({H}, gamma, stream);
    Tensor gpu_mean = upload1E1({capacity}, mean, stream);
    Tensor gpu_inv = upload1E1({capacity}, inv, stream);
    Tensor gpu_dx = upload1E1({capacity, H}, std::vector<float>(capacity * H, 7.25f), stream);
    Tensor gpu_dgamma = upload1E1({H}, std::vector<float>(H, -123.0f), stream);
    Tensor gpu_dbeta = upload1E1({H}, std::vector<float>(H, -123.0f), stream);
    launchPackedLayerNormBackwardActivePrefix(gpu_x, gpu_gamma, gpu_dy, gpu_mean, gpu_inv,
                                              gpu_dx, gpu_dgamma, gpu_dbeta, active, stream.getStream());
    const auto actual_dx = download1E1(gpu_dx, stream);
    const auto actual_dgamma = download1E1(gpu_dgamma, stream);
    const auto actual_dbeta = download1E1(gpu_dbeta, stream);
    for (uint64_t i = 0; i < active * H; ++i) {
        EXPECT_NEAR(actual_dx[i], reference_dx[i], 2.0e-4) << "dX active idx " << i;
    }
    for (uint64_t i = active * H; i < capacity * H; ++i) {
        EXPECT_FLOAT_EQ(actual_dx[i], 7.25f) << "inactive dX changed at " << i;
    }
    for (uint64_t j = 0; j < H; ++j) {
        EXPECT_NEAR(actual_dgamma[j], reference_dgamma[j], 1.0e-3) << "gamma idx " << j;
        EXPECT_NEAR(actual_dbeta[j], reference_dbeta[j], 1.0e-3) << "beta idx " << j;
    }
}

}  // namespace

TEST(PackedLayerNormBackwardActivePrefix1E1, PoisonedInactiveTailsNeverAffectGradientsOrDx) {
    if (!hasCuda1E1()) GTEST_SKIP() << "CUDA device required";
    validateActivePrefix1E1(8, 3);
    validateActivePrefix1E1(21, 11);
}

TEST(PackedLayerNormBackwardActivePrefix1E1, ZeroActiveProducesZeroAffineAndDoesNotTouchDx) {
    if (!hasCuda1E1()) GTEST_SKIP() << "CUDA device required";
    validateActivePrefix1E1(14, 0);
}

TEST(PackedLayerNormBackwardActivePrefix1E1, InvalidActiveExtentRejectedBeforeLaunching) {
    if (!hasCuda1E1()) GTEST_SKIP() << "CUDA device required";
    Stream stream(0);
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    Tensor dx(gpu, TensorDescriptor(DataType::FP32, {8, 128}));
    Tensor gamma(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor mean(gpu, TensorDescriptor(DataType::FP32, {8}));
    Tensor inv(gpu, TensorDescriptor(DataType::FP32, {8}));
    Tensor dg(gpu, TensorDescriptor(DataType::FP32, {128}));
    Tensor db(gpu, TensorDescriptor(DataType::FP32, {128}));
    EXPECT_THROW(launchPackedLayerNormBackwardActivePrefix(x, gamma, dy, mean, inv, dx, dg, db, 9,
                                                            stream.getStream()), std::invalid_argument);
    Tensor bad_mean(gpu, TensorDescriptor(DataType::FP32, {7}));
    EXPECT_THROW(launchPackedLayerNormBackwardActivePrefix(x, gamma, dy, bad_mean, inv, dx, dg, db, 2,
                                                            stream.getStream()), std::invalid_argument);
}
