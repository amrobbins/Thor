#include "Utilities/Expression/StampedEquation.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ThorImplementation;

namespace {
constexpr uint64_t kCapacity = 16;
constexpr uint64_t kBatch = 2;
constexpr uint64_t kHidden = 128;
constexpr float kSentinel = 975.0f;

bool gpuAvailableE2() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

Tensor gpuE2(const std::vector<uint64_t>& dims, const std::vector<float>& values, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::FP32, dims));
    if (host.getTotalNumElements() != values.size()) throw std::invalid_argument("Test value count mismatch");
    std::copy(values.begin(), values.end(), host.getMemPtr<float>());
    Tensor result(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), host.getDescriptor());
    result.copyFromAsync(host, stream);
    stream.synchronize();
    return result;
}

std::vector<float> readE2(const Tensor& gpu, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), gpu.getDescriptor());
    host.copyFromAsync(gpu, stream);
    stream.synchronize();
    const float* p = host.getMemPtr<float>();
    return {p, p + gpu.getTotalNumElements()};
}

struct CaseE2 {
    std::vector<float> x;
    std::vector<float> dy;
    std::vector<float> gamma;
    std::vector<float> beta;
    explicit CaseE2(uint64_t active) : x(kCapacity * kHidden, std::numeric_limits<float>::quiet_NaN()),
                                     dy(kCapacity * kHidden, std::numeric_limits<float>::quiet_NaN()),
                                     gamma(kHidden), beta(kHidden) {
        for (uint64_t j = 0; j < kHidden; ++j) {
            gamma[j] = 0.7f + 0.03f * static_cast<float>(j % 7);
            beta[j] = -0.2f + 0.01f * static_cast<float>(j % 11);
        }
        for (uint64_t r = 0; r < active; ++r) {
            for (uint64_t j = 0; j < kHidden; ++j) {
                const uint64_t idx = r * kHidden + j;
                x[idx] = 0.04f * static_cast<float>((r * 17 + j * 7) % 41) + 0.005f * static_cast<float>(r);
                dy[idx] = 0.17f + 0.006f * static_cast<float>((r * 11 + j * 13) % 37);
            }
        }
    }
};

void checkGradientsE2(uint64_t active, const CaseE2& testcase,
                     const std::vector<float>& dx, const std::vector<float>& dg, const std::vector<float>& db) {
    ASSERT_EQ(dx.size(), kCapacity * kHidden);
    ASSERT_EQ(dg.size(), kHidden);
    ASSERT_EQ(db.size(), kHidden);
    std::vector<double> expected_dg(kHidden, 0.0), expected_db(kHidden, 0.0);
    for (uint64_t r = 0; r < active; ++r) {
        double mean = 0.0, var = 0.0;
        for (uint64_t j = 0; j < kHidden; ++j) mean += testcase.x[r * kHidden + j];
        mean /= kHidden;
        for (uint64_t j = 0; j < kHidden; ++j) {
            const double d = static_cast<double>(testcase.x[r * kHidden + j]) - mean;
            var += d * d;
        }
        var /= kHidden;
        const double inv = 1.0 / std::sqrt(var + 1.0e-5);
        double qsum = 0.0, qzsum = 0.0;
        for (uint64_t j = 0; j < kHidden; ++j) {
            const uint64_t i = r * kHidden + j;
            const double z = (testcase.x[i] - mean) * inv;
            const double q = testcase.dy[i] * testcase.gamma[j];
            qsum += q;
            qzsum += q * z;
            expected_dg[j] += testcase.dy[i] * z;
            expected_db[j] += testcase.dy[i];
        }
        for (uint64_t j = 0; j < kHidden; ++j) {
            const uint64_t i = r * kHidden + j;
            const double z = (testcase.x[i] - mean) * inv;
            const double q = testcase.dy[i] * testcase.gamma[j];
            EXPECT_NEAR(dx[i], inv * (q - qsum / kHidden - z * qzsum / kHidden), 4e-3)
                << "active dX at " << i;
        }
    }
    for (uint64_t i = active * kHidden; i < kCapacity * kHidden; ++i) {
        EXPECT_FLOAT_EQ(dx[i], kSentinel) << "inactive dX " << i;
    }
    for (uint64_t j = 0; j < kHidden; ++j) {
        EXPECT_NEAR(dg[j], expected_dg[j], 3e-3) << "dGamma " << j;
        EXPECT_NEAR(db[j], expected_db[j], 3e-3) << "dBeta " << j;
    }
}

std::shared_ptr<CompiledLayerNorm> compiledForwardE2() {
    auto c = std::make_shared<CompiledLayerNorm>();
    c->normalized_feature_count = kHidden;
    c->packed_row_capacity = kCapacity;
    c->ragged_batch_size = kBatch;
    c->input_dtype = DataType::FP32;
    c->output_dtype = DataType::FP32;
    c->scale_dtype = DataType::FP32;
    c->bias_dtype = DataType::FP32;
    c->compute_dtype = DataType::FP32;
    return c;
}
std::shared_ptr<CompiledLayerNormBackward> compiledBackwardE2() {
    auto c = std::make_shared<CompiledLayerNormBackward>();
    c->normalized_feature_count = kHidden;
    c->packed_row_capacity = kCapacity;
    c->ragged_batch_size = kBatch;
    c->input_dtype = DataType::FP32;
    c->scale_dtype = DataType::FP32;
    c->dy_dtype = DataType::FP32;
    c->dx_dtype = DataType::FP32;
    c->dscale_dtype = DataType::FP32;
    c->dbias_dtype = DataType::FP32;
    c->compute_dtype = DataType::FP32;
    c->produces_dx = true;
    c->produces_dscale = true;
    c->produces_dbias = true;
    return c;
}

Tensor extentCarrierE2(Stream& stream) {
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Tensor offsets(gpu, TensorDescriptor(DataType::UINT32, {kBatch + 1}));
    return offsets;
}
void publishActiveE2(Tensor carrier, uint64_t active) {
    RowPartitionRuntime partition(carrier, RowPartitionDescriptor(kBatch, kCapacity, DataType::UINT32));
    partition.setHostOffsets({0, active / 2, active});
}
}  // namespace

TEST(PackedLayerNormStampedBackward1E2, LinkedTrainingReadsOnlyActivePrefixAcrossBucketTransitions) {
    if (!gpuAvailableE2()) GTEST_SKIP() << "CUDA device required";
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);
    Tensor carrier = extentCarrierE2(stream);
    Tensor gamma = gpuE2({kHidden}, CaseE2(0).gamma, stream);
    Tensor beta = gpuE2({kHidden}, CaseE2(0).beta, stream);
    Tensor x(gpu, TensorDescriptor(DataType::FP32, {kCapacity, kHidden}));
    Tensor dy(gpu, TensorDescriptor(DataType::FP32, {kCapacity, kHidden}));
    Tensor y(gpu, TensorDescriptor(DataType::FP32, {kCapacity, kHidden}));
    Tensor dx(gpu, TensorDescriptor(DataType::FP32, {kCapacity, kHidden}));
    Tensor dg(gpu, TensorDescriptor(DataType::FP32, {kHidden}));
    Tensor db(gpu, TensorDescriptor(DataType::FP32, {kHidden}));
    auto forward = std::make_shared<StampedLayerNorm>(compiledForwardE2(), x, gamma, beta, y, stream, carrier);
    StampedLayerNormBackward backward(compiledBackwardE2(), x, gamma, dy, dx, dg, db, stream, carrier);
    EXPECT_EQ(backward.backwardExecutablePlanCount(), 0U) << "Packed backward uses LN-E1, not cuDNN bucket reductions";
    EXPECT_THROW(backward.runOn(stream), std::runtime_error);
    ASSERT_TRUE(backward.tryLinkForwardStateFrom(forward));
    EXPECT_THROW(backward.runOn(stream), std::runtime_error) << "No real training forward executed";
    for (const uint64_t active : {3ULL, 11ULL, 0ULL, 16ULL}) {
        SCOPED_TRACE("active=" + std::to_string(active));
        const CaseE2 t(active);
        Tensor host_x = gpuE2({kCapacity, kHidden}, t.x, stream);
        Tensor host_dy = gpuE2({kCapacity, kHidden}, t.dy, stream);
        x.copyFromAsync(host_x, stream);
        dy.copyFromAsync(host_dy, stream);
        dx.fill(kSentinel, stream);
        stream.synchronize();
        publishActiveE2(carrier, active);
        forward->runOn(stream);
        ASSERT_TRUE(forward->trainingForwardState()->has_valid_state);
        backward.runOn(stream);
        checkGradientsE2(active, t, readE2(dx, stream), readE2(dg, stream), readE2(db, stream));
    }
}

TEST(PackedLayerNormStampedBackward1E2, RejectsExtentChangesSinceForward) {
    if (!gpuAvailableE2()) GTEST_SKIP() << "CUDA device required";
    const TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);
    Tensor carrier = extentCarrierE2(stream);
    CaseE2 t(7);
    Tensor x = gpuE2({kCapacity, kHidden}, t.x, stream);
    Tensor dy = gpuE2({kCapacity, kHidden}, t.dy, stream);
    Tensor gamma = gpuE2({kHidden}, t.gamma, stream);
    Tensor beta = gpuE2({kHidden}, t.beta, stream);
    Tensor y(gpu, TensorDescriptor(DataType::FP32, {kCapacity, kHidden}));
    Tensor dx(gpu, TensorDescriptor(DataType::FP32, {kCapacity, kHidden}));
    Tensor dg(gpu, TensorDescriptor(DataType::FP32, {kHidden}));
    Tensor db(gpu, TensorDescriptor(DataType::FP32, {kHidden}));
    auto forward = std::make_shared<StampedLayerNorm>(compiledForwardE2(), x, gamma, beta, y, stream, carrier);
    StampedLayerNormBackward backward(compiledBackwardE2(), x, gamma, dy, dx, dg, db, stream, carrier);
    ASSERT_TRUE(backward.tryLinkForwardStateFrom(forward));
    publishActiveE2(carrier, 7);
    forward->runOn(stream);
    publishActiveE2(carrier, 10);
    EXPECT_THROW(backward.runOn(stream), std::runtime_error);
}

TEST(PackedLayerNormStampedBackward1E2, FullExpressionGraphUsesHostExtentForThreeGradientRoutes) {
    if (!gpuAvailableE2()) GTEST_SKIP() << "CUDA device required";
    Stream stream(0);
    const uint64_t active = 5;
    CaseE2 t(active);
    Tensor x = gpuE2({kCapacity, kHidden}, t.x, stream);
    Tensor dy = gpuE2({kCapacity, kHidden}, t.dy, stream);
    Tensor gamma = gpuE2({kHidden}, t.gamma, stream);
    Tensor beta = gpuE2({kHidden}, t.beta, stream);
    Tensor carrier = extentCarrierE2(stream);
    publishActiveE2(carrier, active);
    Expression ex = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression ext = Expression::input("extent", DataType::UINT32, DataType::UINT32);
    ex = ex.withRaggedRuntimeExtent(ext, kBatch, kCapacity, kHidden, RaggedRuntimeExtentSource::HOST_EXTENT);
    const Expression scale = Expression::input("gamma", DataType::FP32, DataType::FP32);
    const Expression bias = Expression::input("beta", DataType::FP32, DataType::FP32);
    const Expression y = Expression::layerNorm(ex, scale, bias, kHidden, 1.0e-5,
                                              DataType::FP32, DataType::FP32, kCapacity);
    auto compiled = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0)
        .compileBackward({"x", "gamma", "beta"}, "dy");
    Tensor dx(TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
              TensorDescriptor(DataType::FP32, {kCapacity, kHidden}));
    dx.fill(kSentinel, stream);
    auto [forward, backward] = compiled.stampForwardBackwardPair(
        {{"x", x}, {"gamma", gamma}, {"beta", beta}, {"extent", carrier}, {"dy", dy}},
        stream, {}, {{"x_grad", dx}});
    forward.run();
    backward.run();
    checkGradientsE2(active, t, readE2(backward.output("x_grad"), stream),
                     readE2(backward.output("gamma_grad"), stream),
                     readE2(backward.output("beta_grad"), stream));
}

TEST(PackedLayerNormStampedBackward1E2, PackedBiasOnlyDoesNotExposeOtherGradientRoutes) {
    if (!gpuAvailableE2()) GTEST_SKIP() << "CUDA device required";
    Stream stream(0);
    const uint64_t active = 7;
    const CaseE2 t(active);
    Tensor x = gpuE2({kCapacity, kHidden}, t.x, stream);
    Tensor dy = gpuE2({kCapacity, kHidden}, t.dy, stream);
    Tensor gamma = gpuE2({kHidden}, t.gamma, stream);
    Tensor beta = gpuE2({kHidden}, t.beta, stream);
    Tensor carrier = extentCarrierE2(stream);
    publishActiveE2(carrier, active);
    Expression ex = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression ext = Expression::input("extent", DataType::UINT32, DataType::UINT32);
    ex = ex.withRaggedRuntimeExtent(ext, kBatch, kCapacity, kHidden, RaggedRuntimeExtentSource::HOST_EXTENT);
    const Expression scale = Expression::input("gamma", DataType::FP32, DataType::FP32);
    const Expression bias = Expression::input("beta", DataType::FP32, DataType::FP32);
    const Expression y = Expression::layerNorm(ex, scale, bias, kHidden, 1.0e-5,
                                              DataType::FP32, DataType::FP32, kCapacity);
    auto compiled = FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0)
        .compileBackward({"beta"}, "dy");
    auto [forward, backward] = compiled.stampForwardBackwardPair(
        {{"x", x}, {"gamma", gamma}, {"beta", beta}, {"extent", carrier}, {"dy", dy}}, stream);
    forward.run();
    backward.run();
    const auto db = readE2(backward.output("beta_grad"), stream);
    for (uint64_t j = 0; j < kHidden; ++j) {
        double reference = 0;
        for (uint64_t r = 0; r < active; ++r) reference += t.dy[r * kHidden + j];
        EXPECT_NEAR(db[j], reference, 3e-3);
    }
    EXPECT_THROW((void)backward.output("x_grad"), std::exception);
    EXPECT_THROW((void)backward.output("gamma_grad"), std::exception);
}
