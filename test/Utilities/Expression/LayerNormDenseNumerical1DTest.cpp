#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace ThorImplementation;

namespace {

bool hasCudaForLayerNorm1D() {
    int devices = 0;
    return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

Tensor toDevice1D(const std::vector<uint64_t>& dimensions, const std::vector<float>& data, Stream& stream) {
    Tensor source(TensorPlacement(TensorPlacement::MemDevices::CPU),
                  TensorDescriptor(DataType::FP32, dimensions));
    if (source.getArraySizeInBytes() != data.size() * sizeof(float)) {
        throw std::invalid_argument("LayerNorm 1D test input dimensions disagree with data length");
    }
    std::copy(data.begin(), data.end(), source.getMemPtr<float>());
    Tensor destination(TensorPlacement(TensorPlacement::MemDevices::GPU, 0), source.getDescriptor());
    destination.copyFromAsync(source, stream);
    stream.synchronize(); // The pageable host source cannot be destroyed before the copy completes.
    return destination;
}

void overwriteDevice1D(Tensor& destination, const std::vector<float>& data, Stream& stream) {
    Tensor source(TensorPlacement(TensorPlacement::MemDevices::CPU), destination.getDescriptor());
    if (source.getArraySizeInBytes() != data.size() * sizeof(float)) {
        throw std::invalid_argument("LayerNorm 1D parameter update dimensions disagree with data length");
    }
    std::copy(data.begin(), data.end(), source.getMemPtr<float>());
    destination.copyFromAsync(source, stream);
    stream.synchronize();
}

std::vector<float> fromDevice1D(const Tensor& source, Stream& stream) {
    Tensor host(TensorPlacement(TensorPlacement::MemDevices::CPU), source.getDescriptor());
    host.copyFromAsync(source, stream);
    stream.synchronize();
    const size_t count = host.getArraySizeInBytes() / sizeof(float);
    return {host.getMemPtr<float>(), host.getMemPtr<float>() + count};
}

FusedEquation backward1D(uint64_t hidden, double epsilon, const std::vector<std::string>& wrt) {
    const Expression x = Expression::input("x", DataType::FP32, DataType::FP32);
    const Expression gamma = Expression::input("gamma", DataType::FP32, DataType::FP32);
    const Expression beta = Expression::input("beta", DataType::FP32, DataType::FP32);
    const Expression y = Expression::layerNorm(x, gamma, beta, hidden, epsilon,
                                              DataType::FP32, DataType::FP32);
    return FusedEquation::compile(Expression::outputs({{"y", y}}).physicalOutputs(), 0)
        .compileBackward(wrt, "dy");
}

struct DenseCase1D {
    uint64_t rows;
    uint64_t hidden;
    double epsilon;
    std::vector<float> x;
    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> dy;
};

DenseCase1D makeCase1D(uint64_t rows, uint64_t hidden, double epsilon) {
    DenseCase1D c{rows, hidden, epsilon};
    c.x.resize(rows * hidden);
    c.dy.resize(rows * hidden);
    c.gamma.resize(hidden);
    c.beta.resize(hidden);
    for (uint64_t row = 0; row < rows; ++row) {
        for (uint64_t j = 0; j < hidden; ++j) {
            const size_t i = row * hidden + j;
            c.x[i] = 0.45f * std::sin(0.071f * static_cast<float>(j) + 0.37f * static_cast<float>(row)) +
                     0.28f * std::cos(0.23f * static_cast<float>(j) - 0.41f * static_cast<float>(row)) +
                     0.13f * static_cast<float>(row);
            c.dy[i] = 0.31f * std::cos(0.19f * static_cast<float>(j) + 0.53f * static_cast<float>(row)) +
                      0.17f * std::sin(0.097f * static_cast<float>(i)) + 0.08f;
        }
    }
    for (uint64_t j = 0; j < hidden; ++j) {
        c.gamma[j] = 0.86f + 0.23f * std::sin(0.11f * static_cast<float>(j));
        c.beta[j] = -0.13f + 0.09f * std::cos(0.16f * static_cast<float>(j));
    }
    return c;
}

// Independent double-precision CPU objective: L = sum(dY * LayerNorm(X, gamma, beta)).
// Using an independent forward evaluator avoids simply checking a backward formula against itself.
double cpuObjective1D(const DenseCase1D& c, const std::vector<float>& x,
                      const std::vector<float>& gamma, const std::vector<float>& beta,
                      std::vector<double>* forward = nullptr) {
    double total = 0.0;
    if (forward) forward->assign(x.size(), 0.0);
    for (uint64_t row = 0; row < c.rows; ++row) {
        double mean = 0.0;
        for (uint64_t j = 0; j < c.hidden; ++j) mean += x[row * c.hidden + j];
        mean /= static_cast<double>(c.hidden);
        double variance = 0.0;
        for (uint64_t j = 0; j < c.hidden; ++j) {
            const double delta = static_cast<double>(x[row * c.hidden + j]) - mean;
            variance += delta * delta;
        }
        variance /= static_cast<double>(c.hidden);
        const double inverse_sigma = 1.0 / std::sqrt(variance + c.epsilon);
        for (uint64_t j = 0; j < c.hidden; ++j) {
            const size_t i = row * c.hidden + j;
            const double y = ((static_cast<double>(x[i]) - mean) * inverse_sigma) * gamma[j] + beta[j];
            total += static_cast<double>(c.dy[i]) * y;
            if (forward) (*forward)[i] = y;
        }
    }
    return total;
}

enum class Operand1D { X, Gamma, Beta };

double finiteDifference1D(const DenseCase1D& c, Operand1D operand, size_t index) {
    // A reasonably large delta reduces float input-perturbation roundoff;
    // the CPU objective itself is accumulated in double precision.
    constexpr float h = 0.005f;
    std::vector<float> x = c.x, gamma = c.gamma, beta = c.beta;
    std::vector<float>* target = operand == Operand1D::X ? &x :
                                 operand == Operand1D::Gamma ? &gamma : &beta;
    const float original = (*target)[index];
    (*target)[index] = original + h;
    const double upper = cpuObjective1D(c, x, gamma, beta);
    (*target)[index] = original - h;
    const double lower = cpuObjective1D(c, x, gamma, beta);
    return (upper - lower) / static_cast<double>((original + h) - (original - h));
}

void checkFiniteDifferences1D(const DenseCase1D& c, Operand1D operand,
                              const std::vector<float>& actual) {
    const size_t expected_size = operand == Operand1D::X ? c.x.size() : c.gamma.size();
    ASSERT_EQ(actual.size(), expected_size);
    for (size_t i = 0; i < expected_size; ++i) {
        const double numerical = finiteDifference1D(c, operand, i);
        EXPECT_TRUE(std::isfinite(actual[i])) << "gradient index " << i;
        EXPECT_NEAR(actual[i], numerical, 0.008 + 0.003 * std::abs(numerical))
            << "gradient operand " << static_cast<int>(operand) << " index " << i;
    }
}

std::vector<float> targetFor1D(const DenseCase1D& c, const std::vector<float>& target_gamma,
                               const std::vector<float>& target_beta) {
    std::vector<double> values;
    (void)cpuObjective1D(c, c.x, target_gamma, target_beta, &values);
    return std::vector<float>(values.begin(), values.end());
}

double meanSquared1D(const std::vector<float>& prediction, const std::vector<float>& target) {
    if (prediction.size() != target.size()) throw std::invalid_argument("mismatched MSE sizes");
    double sum = 0.0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        const double delta = static_cast<double>(prediction[i]) - target[i];
        sum += delta * delta;
    }
    return sum / static_cast<double>(prediction.size());
}

} // namespace

TEST(LayerNormDenseNumerical1D, EveryInputAndAffineGradientMatchesCentralFiniteDifferences) {
    if (!hasCudaForLayerNorm1D()) GTEST_SKIP() << "CUDA required";
    Stream stream(0);
    for (const auto [rows, hidden, epsilon] :
         {std::tuple<uint64_t, uint64_t, double>{3, 128, 1.0e-5},
          std::tuple<uint64_t, uint64_t, double>{2, 256, 1.0e-3}}) {
        SCOPED_TRACE(::testing::Message() << "rows=" << rows << " hidden=" << hidden
                                         << " epsilon=" << epsilon);
        const auto c = makeCase1D(rows, hidden, epsilon);
        const auto equation = backward1D(hidden, epsilon, {"x", "gamma", "beta"});
        auto [forward, backward] = equation.stampForwardBackwardPair(
            {{"x", toDevice1D({rows, hidden}, c.x, stream)},
             {"gamma", toDevice1D({hidden}, c.gamma, stream)},
             {"beta", toDevice1D({hidden}, c.beta, stream)},
             {"dy", toDevice1D({rows, hidden}, c.dy, stream)}}, stream);
        forward.run();
        backward.run();
        std::vector<double> expected_forward;
        (void)cpuObjective1D(c, c.x, c.gamma, c.beta, &expected_forward);
        const auto actual_forward = fromDevice1D(forward.output("y"), stream);
        ASSERT_EQ(actual_forward.size(), expected_forward.size());
        for (size_t i = 0; i < actual_forward.size(); ++i) {
            EXPECT_NEAR(actual_forward[i], expected_forward[i], 0.003) << "forward index " << i;
        }
        checkFiniteDifferences1D(c, Operand1D::X, fromDevice1D(backward.output("x_grad"), stream));
        checkFiniteDifferences1D(c, Operand1D::Gamma, fromDevice1D(backward.output("gamma_grad"), stream));
        checkFiniteDifferences1D(c, Operand1D::Beta, fromDevice1D(backward.output("beta_grad"), stream));
    }
}

TEST(LayerNormDenseNumerical1D, RequestedSubsetsNumericallyCorrectAndDoNotExposeOtherRoutes) {
    if (!hasCudaForLayerNorm1D()) GTEST_SKIP() << "CUDA required";
    Stream stream(0);
    const auto c = makeCase1D(2, 128, 1.0e-4);
    for (const auto& wrt : {std::vector<std::string>{"x", "beta"},
                            std::vector<std::string>{"gamma"}}) {
        const auto equation = backward1D(c.hidden, c.epsilon, wrt);
        auto [forward, backward] = equation.stampForwardBackwardPair(
            {{"x", toDevice1D({c.rows, c.hidden}, c.x, stream)},
             {"gamma", toDevice1D({c.hidden}, c.gamma, stream)},
             {"beta", toDevice1D({c.hidden}, c.beta, stream)},
             {"dy", toDevice1D({c.rows, c.hidden}, c.dy, stream)}}, stream);
        forward.run();
        backward.run();
        for (const auto& name : {std::string("x"), std::string("gamma"), std::string("beta")}) {
            const bool requested = std::find(wrt.begin(), wrt.end(), name) != wrt.end();
            if (!requested) {
                EXPECT_THROW((void)backward.output(name + "_grad"), std::exception);
                continue;
            }
            const auto actual = fromDevice1D(backward.output(name + "_grad"), stream);
            const Operand1D operand = name == "x" ? Operand1D::X :
                                      name == "gamma" ? Operand1D::Gamma : Operand1D::Beta;
            checkFiniteDifferences1D(c, operand, actual);
        }
    }
}

TEST(LayerNormDenseNumerical1D, RejectsIncorrectInputAffineAndUpstreamShapes) {
    if (!hasCudaForLayerNorm1D()) GTEST_SKIP() << "CUDA required";
    Stream stream(0);
    const auto c = makeCase1D(2, 128, 1.0e-5);
    const auto equation = backward1D(c.hidden, c.epsilon, {"x", "gamma", "beta"});
    auto check_bad_shape = [&](const std::vector<uint64_t>& x_dims,
                               const std::vector<uint64_t>& gamma_dims,
                               const std::vector<uint64_t>& beta_dims,
                               const std::vector<uint64_t>& dy_dims) {
        const auto sized = [](const std::vector<uint64_t>& dims) {
            size_t n = 1;
            for (auto d : dims) n *= d;
            return std::vector<float>(n, 0.25f);
        };
        EXPECT_THROW((void)equation.stampForwardBackwardPair(
            {{"x", toDevice1D(x_dims, sized(x_dims), stream)},
             {"gamma", toDevice1D(gamma_dims, sized(gamma_dims), stream)},
             {"beta", toDevice1D(beta_dims, sized(beta_dims), stream)},
             {"dy", toDevice1D(dy_dims, sized(dy_dims), stream)}}, stream), std::exception);
    };
    check_bad_shape({2, 127}, {128}, {128}, {2, 127}); // normalized width mismatch
    check_bad_shape({2, 128}, {127}, {128}, {2, 128}); // gamma mismatch
    check_bad_shape({2, 128}, {128}, {127}, {2, 128}); // beta mismatch
    check_bad_shape({2, 128}, {128}, {128}, {1, 128}); // upstream mismatch
}

TEST(LayerNormDenseNumerical1D, GradientDrivenDeviceAffineUpdateDecreasesActualForwardLoss) {
    if (!hasCudaForLayerNorm1D()) GTEST_SKIP() << "CUDA required";
    Stream stream(0);
    auto c = makeCase1D(3, 128, 1.0e-5);
    std::vector<float> target_gamma(c.hidden), target_beta(c.hidden);
    for (size_t j = 0; j < c.hidden; ++j) {
        target_gamma[j] = c.gamma[j] + 0.25f + 0.03f * std::cos(0.1f * static_cast<float>(j));
        target_beta[j] = c.beta[j] - 0.24f;
    }
    const auto target = targetFor1D(c, target_gamma, target_beta);
    // dY is the derivative of the sum-of-features, mean-over-rows squared
    // error at the INITIAL parameters. Updating gamma/beta must move this
    // objective in the correct direction on a subsequent GPU forward pass.
    std::vector<double> initial_cpu_forward;
    (void)cpuObjective1D(c, c.x, c.gamma, c.beta, &initial_cpu_forward);
    for (size_t i = 0; i < c.dy.size(); ++i) {
        c.dy[i] = (2.0f / static_cast<float>(c.rows)) *
                  (static_cast<float>(initial_cpu_forward[i]) - target[i]);
    }
    Tensor x = toDevice1D({c.rows, c.hidden}, c.x, stream);
    Tensor gamma = toDevice1D({c.hidden}, c.gamma, stream);
    Tensor beta = toDevice1D({c.hidden}, c.beta, stream);
    Tensor dy = toDevice1D({c.rows, c.hidden}, c.dy, stream);
    const auto equation = backward1D(c.hidden, c.epsilon, {"gamma", "beta"});
    auto [forward, backward] = equation.stampForwardBackwardPair(
        {{"x", x}, {"gamma", gamma}, {"beta", beta}, {"dy", dy}}, stream);
    forward.run();
    const auto before_y = fromDevice1D(forward.output("y"), stream);
    const double before_loss = meanSquared1D(before_y, target);
    ASSERT_GT(before_loss, 0.01);
    backward.run();
    const auto dgamma = fromDevice1D(backward.output("gamma_grad"), stream);
    const auto dbeta = fromDevice1D(backward.output("beta_grad"), stream);
    ASSERT_EQ(dgamma.size(), c.hidden);
    ASSERT_EQ(dbeta.size(), c.hidden);
    constexpr float learning_rate = 0.06f;
    bool gamma_changed = false, beta_changed = false;
    for (size_t j = 0; j < c.hidden; ++j) {
        const float old_gamma = c.gamma[j], old_beta = c.beta[j];
        c.gamma[j] -= learning_rate * dgamma[j];
        c.beta[j] -= learning_rate * dbeta[j];
        gamma_changed |= std::abs(c.gamma[j] - old_gamma) > 1e-5f;
        beta_changed |= std::abs(c.beta[j] - old_beta) > 1e-5f;
    }
    EXPECT_TRUE(gamma_changed);
    EXPECT_TRUE(beta_changed);
    overwriteDevice1D(gamma, c.gamma, stream);
    overwriteDevice1D(beta, c.beta, stream);
    forward.run();
    const double after_loss = meanSquared1D(fromDevice1D(forward.output("y"), stream), target);
    EXPECT_LT(after_loss, before_loss * 0.95)
        << "Applying both affine gradients to the actual GPU tensors must improve the objective";
}
