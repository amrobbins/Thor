#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Learning/Attention.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Common/Stream.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/Attention.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;
constexpr uint64_t kQMax = 4;
constexpr uint64_t kKVMax = 6;
constexpr uint64_t kFeatures = 32;

std::vector<uint64_t> offsets2D2(uint32_t b, bool query, uint32_t pattern) {
    std::vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t row = 0; row < b; ++row) {
        // Empty query rows, uneven Q/KV partitions and a different pattern
        // on a second submission through the same placed network.
        const uint64_t length = query
            ? (pattern == 0 ? (row % 4 == 0 ? 0u : 1u + row % 3)
                            : (row % 3 == 0 ? 0u : 1u + row % 4))
            : (pattern == 0 ? 1u + (row * 3u + 1u) % 5u
                            : 1u + (row * 2u + 2u) % 5u);
        offsets[row + 1] = offsets[row] + length;
    }
    return offsets;
}

float value2D2(uint32_t which, uint64_t token, uint64_t f) {
    const float x = 0.025f * static_cast<float>(which + 1) +
        0.033f * static_cast<float>((token * 7 + f * 3 + which) % 13) -
        0.17f + 0.009f * static_cast<float>(f % 5);
    return __half2float(__float2half(x));
}

Impl::Tensor values2D2(uint32_t b, uint64_t m, uint32_t which,
                       const std::vector<uint64_t>& offsets) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor t(cpu, Impl::TensorDescriptor(DataType::FP16, {b * m, kFeatures}));
    auto* v = t.getMemPtr<half>();
    for (uint64_t token = 0; token < b * m; ++token) {
        for (uint64_t f = 0; f < kFeatures; ++f) {
            v[token * kFeatures + f] = token < offsets.back()
                ? __float2half(value2D2(which, token, f))
                : __float2half(std::numeric_limits<float>::quiet_NaN());
        }
    }
    return t;
}

Impl::Tensor physicalOffsets2D2(DataType type, const std::vector<uint64_t>& offsets) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor t(cpu, Impl::TensorDescriptor(type, {offsets.size()}));
    if (type == DataType::UINT32) {
        auto* p = t.getMemPtr<uint32_t>();
        for (size_t i = 0; i < offsets.size(); ++i) p[i] = static_cast<uint32_t>(offsets[i]);
    } else {
        auto* p = t.getMemPtr<uint64_t>();
        for (size_t i = 0; i < offsets.size(); ++i) p[i] = offsets[i];
    }
    return t;
}

Batch makeBatch2D2(uint32_t b, uint32_t pattern, const Api::RaggedTensor& q,
                   const Api::RaggedTensor& k, const Api::RaggedTensor& v) {
    const auto qo = offsets2D2(b, true, pattern);
    const auto ko = offsets2D2(b, false, pattern);
    Impl::RowPartitionRuntime qp(
        physicalOffsets2D2(DataType::UINT32, qo), q.getDescriptorForPlacement(b).getRowPartition());
    Impl::RowPartitionRuntime kp(
        physicalOffsets2D2(DataType::UINT64, ko), k.getDescriptorForPlacement(b).getRowPartition());
    qp.setHostOffsets(qo);
    kp.setHostOffsets(ko);
    Batch batch;
    batch.insert("q", Impl::RaggedTensor(values2D2(b, kQMax, 0, qo), qp));
    batch.insert("k", Impl::RaggedTensor(values2D2(b, kKVMax, 1, ko), kp));
    batch.insert("v", Impl::RaggedTensor(values2D2(b, kKVMax, 2, ko), kp));
    batch.setValidExampleCount(b);
    return batch;
}

// Q/K/V/output projection weights are all identity. This checks the *entire*
// composite computation (four trainable projections and SDPA), not standalone
// SDPA. The same deterministic weights are installed at both placements.
void setIdentityProjections2D2(const std::shared_ptr<Impl::Attention>& physical) {
    ASSERT_NE(physical, nullptr);
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Stream stream = physical->getStreams().at(0);
    for (const char* name : {"query_weights", "key_weights", "value_weights", "output_weights"}) {
        auto parameter = physical->getParameter(name);
        ASSERT_NE(parameter, nullptr);
        ASSERT_TRUE(parameter->getStorage().has_value());
        Impl::Tensor device = parameter->getStorage().value();
        ASSERT_EQ(device.getDimensions(), (std::vector<uint64_t>{kFeatures, kFeatures}));
        Impl::Tensor host(cpu, Impl::TensorDescriptor(DataType::FP16, {kFeatures, kFeatures}));
        auto* data = host.getMemPtr<half>();
        for (uint64_t row = 0; row < kFeatures; ++row) {
            for (uint64_t col = 0; col < kFeatures; ++col) {
                data[row*kFeatures + col] = __float2half(row == col ? 1.0f : 0.0f);
            }
        }
        device.copyFromAsync(host, stream);
        // Keep host backing storage alive until its asynchronous upload ends.
        stream.synchronize();
    }
}

void checkReference2D2(const Impl::RaggedTensor& result, uint32_t b, uint32_t pattern) {
    const auto qo = offsets2D2(b, true, pattern);
    const auto ko = offsets2D2(b, false, pattern);
    EXPECT_EQ(result.getValues().getDimensions(), (std::vector<uint64_t>{b*kQMax, kFeatures}));
    ASSERT_TRUE(result.getHostOffsetsIfAvailable().has_value());
    EXPECT_EQ(result.getHostOffsetsIfAvailable().value(), qo);
    ASSERT_EQ(result.getValues().getDataType(), DataType::FP16);
    const auto* output = result.getValues().getMemPtr<half>();
    uint64_t compared = 0;
    for (uint32_t row = 0; row < b; ++row) {
        for (uint64_t qi = qo[row]; qi < qo[row + 1]; ++qi) {
            std::vector<double> scores;
            for (uint64_t ki = ko[row]; ki < ko[row + 1]; ++ki) {
                double dot = 0.0;
                for (uint64_t f = 0; f < kFeatures; ++f) {
                    dot += static_cast<double>(value2D2(0, qi, f)) * value2D2(1, ki, f);
                }
                scores.push_back(dot / std::sqrt(static_cast<double>(kFeatures)));
            }
            ASSERT_FALSE(scores.empty());
            const double peak = *std::max_element(scores.begin(), scores.end());
            double denom = 0.0;
            for (double& score : scores) {
                score = std::exp(score - peak);
                denom += score;
            }
            for (uint64_t f = 0; f < kFeatures; ++f) {
                double expected = 0.0;
                for (uint64_t i = 0; i < scores.size(); ++i) {
                    expected += (scores[i] / denom) * value2D2(2, ko[row] + i, f);
                }
                const float got = __half2float(output[qi * kFeatures + f]);
                ASSERT_TRUE(std::isfinite(got)) << "active token=" << qi << " feature=" << f;
                EXPECT_NEAR(got, expected, 0.06f) << "row=" << row << " token=" << qi << " feature=" << f;
                ++compared;
            }
        }
    }
    EXPECT_GT(compared, 0u);
}

std::shared_ptr<Api::Attention> findAttention2D2(Api::Network& network) {
    for (uint32_t i = 0; i < network.getNumTrainableLayers(); ++i) {
        if (auto layer = std::dynamic_pointer_cast<Api::Attention>(network.getTrainableLayer(i))) {
            return layer;
        }
    }
    return {};
}
} // namespace

TEST(AttentionNeutral2D2, SavedCompositeCrossAttentionB2B7MatchesCpuForwardAndReusesPlacement) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        GTEST_SKIP() << "CUDA required for composite Attention forward numerical gate";
    }
    const auto archive = std::filesystem::temp_directory_path() /
        ("attention_neutral_2d2_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(archive);
    {
        Api::Network network("attention_neutral_2d2_saved");
        const auto q = Api::RaggedNetworkInput::Builder().network(network).name("q")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
            .trailingDimensions({kFeatures}).maxValuesPerRow(kQMax).build();
        const auto k = Api::RaggedNetworkInput::Builder().network(network).name("k")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
            .trailingDimensions({kFeatures}).maxValuesPerRow(kKVMax).build();
        const auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
            .valuesDataType(DataType::FP16).trailingDimensions({kFeatures}).partition(k).build();
        const auto attention = Api::Attention::Builder().network(network)
            .queryInput(q).keyInput(k).valueInput(v).numHeads(1).headDim(kFeatures)
            .valueDim(kFeatures).outputFeatures(kFeatures).hasBias(false).build();
        ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
        ASSERT_TRUE(attention.getRaggedFeatureOutput()->sharesPartitionWith(q));
        (void)Api::RaggedNetworkOutput::Builder().network(network).name("out")
            .inputTensor(*attention.getRaggedFeatureOutput()).build();
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Api::Network network("attention_neutral_2d2_saved");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto attention = findAttention2D2(network);
        ASSERT_NE(attention, nullptr);
        ASSERT_EQ(attention->architectureJson().at("version"), "2.1.0");
        const auto before = network.architectureJson();
        std::vector<Event> events;
        auto placed = network.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto physical = std::dynamic_pointer_cast<Impl::Attention>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(attention->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{b*kQMax, kFeatures}));
        setIdentityProjections2D2(physical);
        for (uint32_t pattern : {0u, 1u}) {
            SCOPED_TRACE(::testing::Message() << "pattern=" << pattern);
            auto result = placed->inferLogical(makeBatch2D2(
                b, pattern, *attention->getRaggedQueryInput(),
                *attention->getRaggedKeyInput(), *attention->getRaggedValueInput()));
            ASSERT_TRUE(result.contains("out"));
            ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(result.at("out")));
            checkReference2D2(std::get<Impl::RaggedTensor>(result.at("out")), b, pattern);
        }
        EXPECT_EQ(network.architectureJson(), before);
    }
    std::filesystem::remove_all(archive);
}

// 2D-3: a genuine RAW-loss training root on the saved neutral composite.
// The independent CPU VJP deliberately works at the four-projection level:
// replacing Attention backward with a detached SDPA-only backward cannot pass.
namespace {
Impl::DynamicExpression squaredLoss2D3(bool gradient) {
    auto p = Impl::Expression::input("predictions", DataType::FP16, DataType::FP32);
    auto y = Impl::Expression::input("labels", DataType::FP16, DataType::FP32);
    auto delta = p - y;
    auto result = gradient
        ? delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
        : delta * delta;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            gradient ? "predictions_grad" : "loss", result.withOutputDType(DataType::FP16)}})));
}

std::vector<float> copyNumeric2D3(const Impl::Tensor& source) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = source.clone(cpu);
    Impl::Tensor device = source;
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    host.copyFromAsync(device, stream);
    stream.synchronize();
    std::vector<float> result(host.getTotalNumElements());
    if (host.getDataType() == DataType::FP16) {
        const auto* ptr = host.getMemPtr<half>();
        for (size_t i = 0; i < result.size(); ++i) result[i] = __half2float(ptr[i]);
    } else if (host.getDataType() == DataType::FP32) {
        const auto* ptr = host.getMemPtr<float>();
        for (size_t i = 0; i < result.size(); ++i) result[i] = ptr[i];
    } else {
        throw std::runtime_error("2D3 unsupported diagnostic tensor dtype");
    }
    return result;
}

Batch makeTrainingBatch2D3(uint32_t b, const Api::RaggedTensor& q,
                          const Api::RaggedTensor& k, const Api::RaggedTensor& v,
                          const Api::RaggedTensor& labels) {
    Batch batch = makeBatch2D2(b, 0, q, k, v);
    const auto qo = offsets2D2(b, true, 0);
    Impl::RowPartitionRuntime qp(
        physicalOffsets2D2(DataType::UINT32, qo), labels.getDescriptorForPlacement(b).getRowPartition());
    qp.setHostOffsets(qo);
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor zero(cpu, labels.getDescriptorForPlacement(b).getValuesDescriptor());
    auto* ptr = zero.getMemPtr<half>();
    for (uint64_t i = 0; i < b * kQMax * kFeatures; ++i)
        ptr[i] = i < qo.back() * kFeatures
            ? __float2half(0.0f) : __float2half(std::numeric_limits<float>::quiet_NaN());
    batch.insert("labels", Impl::RaggedTensor(zero, qp));
    return batch;
}

struct BackwardReference2D3 {
    std::vector<double> dq, dk, dv;
    std::map<std::string, std::vector<double>> dw;
};

BackwardReference2D3 cpuBackward2D3(uint32_t b) {
    const auto qo = offsets2D2(b, true, 0), ko = offsets2D2(b, false, 0);
    BackwardReference2D3 r;
    r.dq.resize(b * kQMax * kFeatures, 0.0);
    r.dk.resize(b * kKVMax * kFeatures, 0.0);
    r.dv.resize(b * kKVMax * kFeatures, 0.0);
    for (const char* name : {"query_weights", "key_weights", "value_weights", "output_weights"})
        r.dw[name].resize(kFeatures * kFeatures, 0.0);
    // All four projection matrices are identity.  The CPU reference still
    // computes each projection's input and parameter VJP independently.
    for (uint32_t row = 0; row < b; ++row) {
        for (uint64_t qi = qo[row]; qi < qo[row + 1]; ++qi) {
            std::vector<double> p;
            for (uint64_t ki = ko[row]; ki < ko[row + 1]; ++ki) {
                double dot = 0.0;
                for (uint64_t d = 0; d < kFeatures; ++d)
                    dot += value2D2(0, qi, d) * value2D2(1, ki, d);
                p.push_back(dot / std::sqrt(static_cast<double>(kFeatures)));
            }
            if (p.empty()) continue;
            const double peak = *std::max_element(p.begin(), p.end());
            double denominator = 0.0;
            for (auto& score : p) { score = std::exp(score - peak); denominator += score; }
            for (auto& score : p) score /= denominator;
            std::vector<double> dy(kFeatures), projected(kFeatures);
            // Loss consumes the output projection's FP16 output, and the
            // output projection consumes SDPA's FP16 output.
            for (uint64_t d = 0; d < kFeatures; ++d) {
                double weighted = 0.0;
                for (size_t j = 0; j < p.size(); ++j)
                    weighted += p[j] * value2D2(2, ko[row] + j, d);
                projected[d] = __half2float(__float2half(static_cast<float>(weighted)));
                dy[d] = 2.0 * projected[d];
            }
            for (uint64_t i = 0; i < kFeatures; ++i)
                for (uint64_t o = 0; o < kFeatures; ++o)
                    r.dw["output_weights"][i * kFeatures + o] += projected[i] * dy[o];
            std::vector<double> dp(p.size());
            double mean = 0.0;
            for (size_t j = 0; j < p.size(); ++j) {
                const uint64_t ki = ko[row] + j;
                double dprob = 0.0;
                for (uint64_t d = 0; d < kFeatures; ++d) {
                    dprob += dy[d] * value2D2(2, ki, d);
                    r.dv[ki * kFeatures + d] += p[j] * dy[d];
                }
                dp[j] = dprob;
                mean += p[j] * dprob;
            }
            for (size_t j = 0; j < p.size(); ++j) {
                const uint64_t ki = ko[row] + j;
                const double ds = p[j] * (dp[j] - mean) /
                    std::sqrt(static_cast<double>(kFeatures));
                for (uint64_t d = 0; d < kFeatures; ++d) {
                    r.dq[qi * kFeatures + d] += ds * value2D2(1, ki, d);
                    r.dk[ki * kFeatures + d] += ds * value2D2(0, qi, d);
                }
            }
        }
    }
    for (uint64_t t = 0; t < qo.back(); ++t)
        for (uint64_t i = 0; i < kFeatures; ++i)
            for (uint64_t o = 0; o < kFeatures; ++o)
                r.dw["query_weights"][i*kFeatures+o] += value2D2(0,t,i)*r.dq[t*kFeatures+o];
    for (uint64_t t = 0; t < ko.back(); ++t)
        for (uint64_t i = 0; i < kFeatures; ++i)
            for (uint64_t o = 0; o < kFeatures; ++o) {
                r.dw["key_weights"][i*kFeatures+o] += value2D2(1,t,i)*r.dk[t*kFeatures+o];
                r.dw["value_weights"][i*kFeatures+o] += value2D2(2,t,i)*r.dv[t*kFeatures+o];
            }
    return r;
}

void checkGradient2D3(const std::vector<float>& got, const std::vector<double>& reference,
                      uint64_t count, const char* name, double tolerance) {
    ASSERT_GE(got.size(), count);
    ASSERT_GE(reference.size(), count);
    bool nonzero = false;
    for (uint64_t i = 0; i < count; ++i) {
        ASSERT_TRUE(std::isfinite(got[i])) << name << " i=" << i;
        const double expected = reference[i] * Impl::Loss::getLossScalingFactor();
        EXPECT_NEAR(got[i], expected, tolerance + std::abs(expected) * 0.2)
            << name << " i=" << i;
        nonzero |= std::abs(expected) > 1e-5;
    }
    EXPECT_TRUE(nonzero) << name << " reference should not be identically zero";
}
} // namespace

TEST(AttentionNeutral2D3, SavedCompositeB2B7RealLossConnectedBackwardAndProjectionGradients) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA required for connected composite backward";
    const auto archive = std::filesystem::temp_directory_path() /
        ("attention_neutral_2d3_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::remove_all(archive);
    {
        Api::Network network("attention_neutral_2d3_saved");
        auto q = Api::RaggedNetworkInput::Builder().network(network).name("q")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
            .trailingDimensions({kFeatures}).maxValuesPerRow(kQMax).build();
        auto k = Api::RaggedNetworkInput::Builder().network(network).name("k")
            .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
            .trailingDimensions({kFeatures}).maxValuesPerRow(kKVMax).build();
        auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
            .valuesDataType(DataType::FP16).trailingDimensions({kFeatures}).partition(k).build();
        auto labels = Api::RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP16).trailingDimensions({kFeatures}).partition(q).build();
        auto qr = Api::GradientRivet::Builder().network(network).tensor(q.getValues()).build();
        auto kr = Api::GradientRivet::Builder().network(network).tensor(k.getValues()).build();
        auto vr = Api::GradientRivet::Builder().network(network).tensor(v.getValues()).build();
        auto sgd = Api::Sgd::Builder().initialLearningRate(0.1f)
            .decay(0.0f).momentum(0.0f).build();
        auto attention = Api::Attention::Builder().network(network)
            .queryInput(q.withValues(qr.getFeatureOutput().value()))
            .keyInput(k.withValues(kr.getFeatureOutput().value()))
            .valueInput(v.withValues(vr.getFeatureOutput().value()))
            .numHeads(1).headDim(kFeatures).valueDim(kFeatures).outputFeatures(kFeatures)
            .hasBias(false).weightsDataType(DataType::FP16).computeDataType(DataType::FP32)
            .outputDataType(DataType::FP16).optimizer(sgd).build();
        ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
        auto loss = Api::RaggedCustomLoss::Builder().network(network)
            .predictions(*attention.getRaggedFeatureOutput()).labels(labels)
            .lossExpression(squaredLoss2D3(false)).gradientExpression(squaredLoss2D3(true)).build();
        (void)Api::NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(loss.getRaggedRawLoss().getValues()).dataType(DataType::FP16).build();
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Api::Network network("attention_neutral_2d3_saved");
        ASSERT_NO_THROW(network.load(archive.string()));
        auto attention = findAttention2D2(network);
        ASSERT_NE(attention, nullptr);
        ASSERT_EQ(attention->architectureJson().at("version"), "2.1.0");
        auto before = network.architectureJson();
        auto findInput = [&](const std::string& name) -> Api::RaggedTensor {
            for (const auto& input : network.getExternalRaggedNetworkInputs())
                if (input.name == name) return input.raggedTensor;
            throw std::runtime_error("Missing ragged input " + name);
        };
        const auto q = findInput("q"), k = findInput("k"), v = findInput("v"), labels = findInput("labels");
        ASSERT_FALSE(q.sharesPartitionWith(k));
        ASSERT_TRUE(k.sharesPartitionWith(v));
        ASSERT_TRUE(labels.sharesPartitionWith(q));
        std::vector<Event> events;
        auto placed = network.place(b, events, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto physical = std::dynamic_pointer_cast<Impl::Attention>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(attention->getId()));
        ASSERT_NE(physical, nullptr);
        const auto& native = physical->getNativeExecutionPlans().at(0, Impl::kPrimaryDynamicExpressionVariant);
        ASSERT_TRUE(native.hasRetainedForward());
        ASSERT_TRUE(native.hasBackward());
        ASSERT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (std::vector<uint64_t>{b * kQMax, kFeatures}));
        setIdentityProjections2D2(physical);
        std::map<std::string, std::vector<float>> weightsBefore;
        for (const char* name : {"query_weights", "key_weights", "value_weights", "output_weights"}) {
            auto p = physical->getParameter(name);
            ASSERT_NE(p, nullptr);
            ASSERT_TRUE(p->getStorage().has_value());
            ASSERT_TRUE(p->hasOptimizer());
            ASSERT_NE(p->getOptimizer(), nullptr);
            ASSERT_TRUE(p->getOptimizer()->getWeightsGradient().has_value());
            EXPECT_EQ(native.backward->getFinalOutputs().at(std::string(name) + "_grad"),
                      p->getOptimizer()->getWeightsGradient().value());
            weightsBefore[name] = copyNumeric2D3(*p->getStorage());
        }
        std::map<std::string, Impl::Tensor> outputs;
        std::map<std::string, Event> ready;
        auto done = placed->submitBatch(0, makeTrainingBatch2D3(b,q,k,v,labels),
                                        outputs, ready, false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        const auto& error = physical->getErrorOutputs();
        ASSERT_GE(error.size(), 3u);
        for (uint32_t port = 0; port < 3; ++port)
            ASSERT_TRUE(error[port].has_value()) << "differentiable port=" << port;
        const auto ref = cpuBackward2D3(b);
        checkGradient2D3(copyNumeric2D3(*error[0]), ref.dq,
                         offsets2D2(b, true, 0).back() * kFeatures, "dQ", 0.00015);
        checkGradient2D3(copyNumeric2D3(*error[1]), ref.dk,
                         offsets2D2(b, false, 0).back() * kFeatures, "dK", 0.0002);
        checkGradient2D3(copyNumeric2D3(*error[2]), ref.dv,
                         offsets2D2(b, false, 0).back() * kFeatures, "dV", 0.012);
        bool anyWeightUpdated = false;
        for (const char* name : {"query_weights", "key_weights", "value_weights", "output_weights"}) {
            auto p = physical->getParameter(name);
            const auto gradient = copyNumeric2D3(*p->getOptimizer()->getWeightsGradient());
            const auto& reference = ref.dw.at(name);
            checkGradient2D3(gradient, reference, kFeatures * kFeatures, name,
                             (std::string(name) == "query_weights" || std::string(name) == "key_weights")
                                 ? 0.00012 : 0.022);
            const auto after = copyNumeric2D3(*p->getStorage());
            ASSERT_EQ(after.size(), weightsBefore[name].size());
            for (size_t i = 0; i < after.size(); ++i)
                anyWeightUpdated |= std::abs(after[i] - weightsBefore[name][i]) > 1e-5f;
        }
        EXPECT_TRUE(anyWeightUpdated) << "RAW loss must update at least one trainable projection";
        EXPECT_EQ(network.architectureJson(), before);
    }
    std::filesystem::remove_all(archive);
}

// 2D-4A: GQA must preserve the independent Q and KV row partitions,
// including each KV head being shared by multiple Q heads. Exercise both
// cross-attention and top-left causal masking against a separate CPU model.
namespace {
constexpr uint32_t kGqaHeads2D4 = 4;
constexpr uint32_t kKvHeads2D4 = 2;
constexpr uint32_t kHeadWidth2D4 = 8;

std::vector<uint64_t> kvOffsets2D4(uint32_t b, uint32_t pattern, bool causal) {
    return causal ? offsets2D2(b, true, pattern) : offsets2D2(b, false, pattern);
}

Batch makeGqaBatch2D4(uint32_t b, uint32_t pattern, bool causal,
                      const Api::RaggedTensor& q, const Api::RaggedTensor& k,
                      const Api::RaggedTensor& v) {
    const auto qo = offsets2D2(b, true, pattern);
    const auto ko = kvOffsets2D4(b, pattern, causal);
    Impl::RowPartitionRuntime qp(
        physicalOffsets2D2(DataType::UINT32, qo), q.getDescriptorForPlacement(b).getRowPartition());
    Impl::RowPartitionRuntime kp(
        physicalOffsets2D2(DataType::UINT64, ko), k.getDescriptorForPlacement(b).getRowPartition());
    qp.setHostOffsets(qo);
    kp.setHostOffsets(ko);
    Batch batch;
    batch.insert("q", Impl::RaggedTensor(values2D2(b, kQMax, 0, qo), qp));
    batch.insert("k", Impl::RaggedTensor(values2D2(b, kKVMax, 1, ko), kp));
    batch.insert("v", Impl::RaggedTensor(values2D2(b, kKVMax, 2, ko), kp));
    batch.setValidExampleCount(b);
    return batch;
}

void setGqaWeights2D4(const std::shared_ptr<Impl::Attention>& physical) {
    ASSERT_NE(physical, nullptr);
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Stream stream = physical->getStreams().at(0);
    for (const char* name : {"query_weights", "key_weights", "value_weights", "output_weights"}) {
        auto parameter = physical->getParameter(name);
        ASSERT_NE(parameter, nullptr);
        ASSERT_TRUE(parameter->getStorage().has_value());
        Impl::Tensor device = parameter->getStorage().value();
        const bool kv = std::string(name) == "key_weights" || std::string(name) == "value_weights";
        const uint64_t cols = kv ? kKvHeads2D4 * kHeadWidth2D4 : kGqaHeads2D4 * kHeadWidth2D4;
        ASSERT_EQ(device.getDimensions(), (std::vector<uint64_t>{kFeatures, cols}));
        Impl::Tensor host(cpu, Impl::TensorDescriptor(DataType::FP16, {kFeatures, cols}));
        auto* values = host.getMemPtr<half>();
        for (uint64_t r = 0; r < kFeatures; ++r)
            for (uint64_t c = 0; c < cols; ++c)
                values[r * cols + c] = __float2half(r == c ? 1.0f : 0.0f);
        device.copyFromAsync(host, stream);
        stream.synchronize();
    }
}

void checkGqaReference2D4(const Impl::RaggedTensor& result, uint32_t b,
                          uint32_t pattern, bool causal) {
    const auto qo = offsets2D2(b, true, pattern);
    const auto ko = kvOffsets2D4(b, pattern, causal);
    EXPECT_EQ(result.getValues().getDimensions(), (std::vector<uint64_t>{b * kQMax, kFeatures}));
    ASSERT_TRUE(result.getHostOffsetsIfAvailable().has_value());
    EXPECT_EQ(*result.getHostOffsetsIfAvailable(), qo);
    ASSERT_EQ(result.getValues().getDataType(), DataType::FP16);
    const auto* output = result.getValues().getMemPtr<half>();
    uint64_t compared = 0;
    for (uint32_t row = 0; row < b; ++row) {
        for (uint64_t qi = qo[row]; qi < qo[row + 1]; ++qi) {
            const uint64_t localQuery = qi - qo[row];
            for (uint32_t head = 0; head < kGqaHeads2D4; ++head) {
                const uint64_t keyValueHead = head / (kGqaHeads2D4 / kKvHeads2D4);
                std::vector<double> weights;
                std::vector<uint64_t> keys;
                for (uint64_t ki = ko[row]; ki < ko[row + 1]; ++ki) {
                    if (causal && (ki - ko[row]) > localQuery) continue;
                    double score = 0.0;
                    for (uint64_t d = 0; d < kHeadWidth2D4; ++d) {
                        score += static_cast<double>(value2D2(0, qi, head * kHeadWidth2D4 + d)) *
                            value2D2(1, ki, keyValueHead * kHeadWidth2D4 + d);
                    }
                    keys.push_back(ki);
                    weights.push_back(score / std::sqrt(static_cast<double>(kHeadWidth2D4)));
                }
                ASSERT_FALSE(weights.empty()) << "row=" << row << " qi=" << qi;
                const double peak = *std::max_element(weights.begin(), weights.end());
                double denom = 0.0;
                for (auto& w : weights) { w = std::exp(w - peak); denom += w; }
                for (uint64_t d = 0; d < kHeadWidth2D4; ++d) {
                    double expected = 0.0;
                    for (size_t i = 0; i < weights.size(); ++i) {
                        expected += weights[i] / denom *
                            value2D2(2, keys[i], keyValueHead * kHeadWidth2D4 + d);
                    }
                    const uint64_t outFeature = head * kHeadWidth2D4 + d;
                    const float got = __half2float(output[qi * kFeatures + outFeature]);
                    ASSERT_TRUE(std::isfinite(got)) << "row=" << row << " qi=" << qi << " head=" << head;
                    EXPECT_NEAR(got, expected, 0.065f)
                        << "row=" << row << " qi=" << qi << " head=" << head << " d=" << d;
                    ++compared;
                }
            }
        }
    }
    EXPECT_GT(compared, 0u);
}
} // namespace

TEST(AttentionNeutral2D4, GqaCrossAndCausalB2B7SavedNeutralCompositeMatchCpu) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA required for 2D-4 GQA/causal numerical gate";
    for (bool causal : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "causal=" << causal);
        const auto archive = std::filesystem::temp_directory_path() /
            ("attention_neutral_2d4a_" + std::to_string(causal) + "_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::remove_all(archive);
        {
            Api::Network network("attention_neutral_2d4a_saved");
            const auto q = Api::RaggedNetworkInput::Builder().network(network).name("q")
                .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
                .trailingDimensions({kFeatures}).maxValuesPerRow(kQMax).build();
            const auto k = Api::RaggedNetworkInput::Builder().network(network).name("k")
                .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
                .trailingDimensions({kFeatures}).maxValuesPerRow(kKVMax).build();
            const auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
                .valuesDataType(DataType::FP16).trailingDimensions({kFeatures}).partition(k).build();
            Api::Attention::Builder builder;
            builder.network(network).queryInput(q).keyInput(k).valueInput(v)
                .numHeads(kGqaHeads2D4).numKeyValueHeads(kKvHeads2D4)
                .headDim(kHeadWidth2D4).valueDim(kHeadWidth2D4)
                .outputFeatures(kFeatures).hasBias(false);
            if (causal) builder.causal();
            const auto attention = builder.build();
            ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
            ASSERT_TRUE(attention.getRaggedFeatureOutput()->sharesPartitionWith(q));
            ASSERT_FALSE(attention.getRaggedFeatureOutput()->sharesPartitionWith(k));
            (void)Api::RaggedNetworkOutput::Builder().network(network).name("out")
                .inputTensor(*attention.getRaggedFeatureOutput()).build();
            network.save(archive.string(), true);
        }
        for (uint32_t b : {2u, 7u}) {
            SCOPED_TRACE(::testing::Message() << "B=" << b);
            Api::Network network("attention_neutral_2d4a_saved");
            ASSERT_NO_THROW(network.load(archive.string()));
            auto attention = findAttention2D2(network);
            ASSERT_NE(attention, nullptr);
            ASSERT_EQ(attention->architectureJson().at("version"), "2.1.0");
            EXPECT_EQ(attention->getNumHeads(), kGqaHeads2D4);
            EXPECT_EQ(attention->getNumKeyValueHeads(), kKvHeads2D4);
            EXPECT_EQ(attention->getMaskKind(), causal ? Impl::AttentionMaskKind::CausalTopLeft
                                                     : Impl::AttentionMaskKind::None);
            const auto architectureBefore = network.architectureJson();
            std::vector<Event> events;
            auto placed = network.place(b, events, true);
            ASSERT_NE(placed, nullptr);
            for (auto& event : events) event.synchronize();
            auto physical = std::dynamic_pointer_cast<Impl::Attention>(
                placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(attention->getId()));
            ASSERT_NE(physical, nullptr);
            setGqaWeights2D4(physical);
            for (uint32_t pattern : {0u, 1u}) {
                SCOPED_TRACE(::testing::Message() << "pattern=" << pattern);
                const auto result = placed->inferLogical(makeGqaBatch2D4(
                    b, pattern, causal, *attention->getRaggedQueryInput(),
                    *attention->getRaggedKeyInput(), *attention->getRaggedValueInput()));
                ASSERT_TRUE(result.contains("out"));
                ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(result.at("out")));
                checkGqaReference2D4(std::get<Impl::RaggedTensor>(result.at("out")), b, pattern, causal);
            }
            EXPECT_EQ(network.architectureJson(), architectureBefore);
        }
        std::filesystem::remove_all(archive);
    }
}


// 2D-4B: test both mixed directions numerically. The real ragged side has
// canonical [B+1] offsets, whereas the dense side's synthetic uniform partition
// must be built at stamp time using *this* placement's B. A dense query must
// produce a dense [B, Mq, F] output; ragged query retains Q's partition.
namespace {
Impl::Tensor denseMixedValues2D4B(uint32_t b, uint64_t tokensPerRow, uint32_t which) {
    const Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor data(cpu, Impl::TensorDescriptor(DataType::FP16, {b, tokensPerRow, kFeatures}));
    auto* values = data.getMemPtr<half>();
    for (uint64_t token = 0; token < b * tokensPerRow; ++token)
        for (uint64_t feature = 0; feature < kFeatures; ++feature)
            values[token * kFeatures + feature] = __float2half(value2D2(which, token, feature));
    return data;
}

Batch mixedBatch2D4B(uint32_t b, uint32_t pattern, bool queryRagged,
                    const Api::Attention& attention) {
    Batch batch;
    if (queryRagged) {
        const auto qo = offsets2D2(b, true, pattern);
        const auto query = *attention.getRaggedQueryInput();
        Impl::RowPartitionRuntime partition(
            physicalOffsets2D2(DataType::UINT32, qo),
            query.getDescriptorForPlacement(b).getRowPartition());
        partition.setHostOffsets(qo);
        batch.insert("q", Impl::RaggedTensor(values2D2(b, kQMax, 0, qo), partition));
        batch.insert("k", denseMixedValues2D4B(b, kKVMax, 1));
        batch.insert("v", denseMixedValues2D4B(b, kKVMax, 2));
    } else {
        const auto ko = offsets2D2(b, false, pattern);
        const auto key = *attention.getRaggedKeyInput();
        Impl::RowPartitionRuntime partition(
            physicalOffsets2D2(DataType::UINT64, ko),
            key.getDescriptorForPlacement(b).getRowPartition());
        partition.setHostOffsets(ko);
        batch.insert("q", denseMixedValues2D4B(b, kQMax, 0));
        batch.insert("k", Impl::RaggedTensor(values2D2(b, kKVMax, 1, ko), partition));
        batch.insert("v", Impl::RaggedTensor(values2D2(b, kKVMax, 2, ko), partition));
    }
    batch.setValidExampleCount(b);
    return batch;
}

void checkMixedCpu2D4B(const Api::InferenceOutputValue& result,
                      uint32_t b, uint32_t pattern, bool queryRagged) {
    const auto qo = queryRagged ? offsets2D2(b, true, pattern) : std::vector<uint64_t>();
    const auto ko = queryRagged ? std::vector<uint64_t>() : offsets2D2(b, false, pattern);
    Impl::Tensor outputValues;
    if (queryRagged) {
        ASSERT_TRUE(std::holds_alternative<Impl::RaggedTensor>(result));
        const auto& ragged = std::get<Impl::RaggedTensor>(result);
        ASSERT_TRUE(ragged.getHostOffsetsIfAvailable().has_value());
        EXPECT_EQ(*ragged.getHostOffsetsIfAvailable(), qo);
        outputValues = ragged.getValues();
        EXPECT_EQ(outputValues.getDimensions(), (std::vector<uint64_t>{b * kQMax, kFeatures}));
    } else {
        ASSERT_TRUE(std::holds_alternative<Impl::Tensor>(result));
        outputValues = std::get<Impl::Tensor>(result);
        EXPECT_EQ(outputValues.getDimensions(), (std::vector<uint64_t>{b, kQMax, kFeatures}));
    }
    ASSERT_TRUE(outputValues.isInitialized());
    ASSERT_EQ(outputValues.getDataType(), DataType::FP16);
    const auto* actual = outputValues.getMemPtr<half>();
    uint64_t comparisons = 0;
    for (uint32_t row = 0; row < b; ++row) {
        const uint64_t qLen = queryRagged ? qo[row + 1] - qo[row] : kQMax;
        const uint64_t kvLen = queryRagged ? kKVMax : ko[row + 1] - ko[row];
        for (uint64_t localQ = 0; localQ < qLen; ++localQ) {
            const uint64_t qToken = queryRagged ? qo[row] + localQ : row * kQMax + localQ;
            std::vector<double> weights;
            for (uint64_t localK = 0; localK < kvLen; ++localK) {
                const uint64_t kToken = queryRagged ? row * kKVMax + localK : ko[row] + localK;
                double dot = 0.0;
                for (uint64_t d = 0; d < kFeatures; ++d)
                    dot += static_cast<double>(value2D2(0, qToken, d)) * value2D2(1, kToken, d);
                weights.push_back(dot / std::sqrt(static_cast<double>(kFeatures)));
            }
            ASSERT_FALSE(weights.empty());
            const double peak = *std::max_element(weights.begin(), weights.end());
            double denom = 0.0;
            for (auto& w : weights) { w = std::exp(w - peak); denom += w; }
            for (uint64_t d = 0; d < kFeatures; ++d) {
                double expected = 0.0;
                for (uint64_t localK = 0; localK < kvLen; ++localK) {
                    const uint64_t vToken = queryRagged ? row * kKVMax + localK : ko[row] + localK;
                    expected += weights[localK] / denom * value2D2(2, vToken, d);
                }
                const float observed = __half2float(actual[qToken * kFeatures + d]);
                ASSERT_TRUE(std::isfinite(observed)) << "row=" << row << " query=" << localQ;
                EXPECT_NEAR(observed, expected, 0.065f)
                    << "row=" << row << " query=" << localQ << " feature=" << d;
                ++comparisons;
            }
        }
    }
    EXPECT_GT(comparisons, 0u);
}
} // namespace

void runMixedDomains2D4B(bool queryRagged) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA required for mixed-domain neutral Attention forward";
    {
        SCOPED_TRACE(::testing::Message() << "queryRagged=" << queryRagged);
        const auto archive = std::filesystem::temp_directory_path() /
            ("attention_neutral_2d4b_" + std::to_string(queryRagged) + "_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::remove_all(archive);
        {
            Api::Network network("attention_neutral_2d4b_saved");
            Api::Attention::Builder builder;
            builder.network(network).numHeads(1).headDim(kFeatures).valueDim(kFeatures)
                .outputFeatures(kFeatures).hasBias(false);
            if (queryRagged) {
                const auto q = Api::RaggedNetworkInput::Builder().network(network).name("q")
                    .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT32)
                    .trailingDimensions({kFeatures}).maxValuesPerRow(kQMax).build();
                const auto k = Api::NetworkInput::Builder().network(network).name("k")
                    .dataType(DataType::FP16).dimensions({kKVMax, kFeatures}).build();
                const auto v = Api::NetworkInput::Builder().network(network).name("v")
                    .dataType(DataType::FP16).dimensions({kKVMax, kFeatures}).build();
                builder.queryInput(q).keyInput(*k.getFeatureOutput()).valueInput(*v.getFeatureOutput());
                const auto attention = builder.build();
                ASSERT_TRUE(attention.getRaggedFeatureOutput().has_value());
                EXPECT_TRUE(attention.getRaggedFeatureOutput()->sharesPartitionWith(q));
                (void)Api::RaggedNetworkOutput::Builder().network(network).name("out")
                    .inputTensor(*attention.getRaggedFeatureOutput()).build();
            } else {
                const auto q = Api::NetworkInput::Builder().network(network).name("q")
                    .dataType(DataType::FP16).dimensions({kQMax, kFeatures}).build();
                const auto k = Api::RaggedNetworkInput::Builder().network(network).name("k")
                    .valuesDataType(DataType::FP16).offsetsDataType(DataType::UINT64)
                    .trailingDimensions({kFeatures}).maxValuesPerRow(kKVMax).build();
                const auto v = Api::RaggedNetworkInput::Builder().network(network).name("v")
                    .valuesDataType(DataType::FP16).trailingDimensions({kFeatures}).partition(k).build();
                builder.queryInput(*q.getFeatureOutput()).keyInput(k).valueInput(v);
                const auto attention = builder.build();
                ASSERT_FALSE(attention.getRaggedFeatureOutput().has_value());
                ASSERT_TRUE(attention.getFeatureOutput().has_value());
                (void)Api::NetworkOutput::Builder().network(network).name("out")
                    .inputTensor(*attention.getFeatureOutput()).dataType(DataType::FP16).build();
            }
            network.save(archive.string(), true);
        }
        for (uint32_t b : {2u, 7u}) {
            SCOPED_TRACE(::testing::Message() << "B=" << b);
            Api::Network network("attention_neutral_2d4b_saved");
            ASSERT_NO_THROW(network.load(archive.string()));
            auto attention = findAttention2D2(network);
            ASSERT_NE(attention, nullptr);
            ASSERT_EQ(attention->architectureJson().at("version"), "2.1.0");
            EXPECT_EQ(attention->getQueryRagged(), queryRagged);
            EXPECT_EQ(attention->getKeyValueRagged(), !queryRagged);
            const auto architectureBefore = network.architectureJson();
            // Validate original external boundaries and composite references before
            // trying to stamp: the first failed check identifies the geometry owner.
            for (const auto& boundary : network.getExternalRaggedNetworkInputs()) {
                SCOPED_TRACE(::testing::Message() << "ragged input=" << boundary.name);
                ASSERT_NO_THROW((void)boundary.raggedTensor.getDescriptorForPlacement(b));
            }
            for (const auto& boundary : network.getExternalRaggedNetworkOutputs()) {
                SCOPED_TRACE(::testing::Message() << "ragged output=" << boundary.name);
                ASSERT_NO_THROW((void)boundary.raggedTensor.getDescriptorForPlacement(b));
            }
            if (queryRagged) {
                ASSERT_TRUE(attention->getRaggedQueryInput().has_value());
                ASSERT_TRUE(attention->getRaggedFeatureOutput().has_value());
                EXPECT_EQ(attention->getRaggedQueryInput()->getValuesDimensions(),
                          (std::vector<uint64_t>{kQMax, kFeatures}));
                EXPECT_EQ(attention->getRaggedFeatureOutput()->getValuesDimensions(),
                          (std::vector<uint64_t>{kQMax, kFeatures}));
                ASSERT_NO_THROW((void)attention->getRaggedQueryInput()->getDescriptorForPlacement(b))
                    << "query input descriptor";
                ASSERT_NO_THROW((void)attention->getRaggedFeatureOutput()->getDescriptorForPlacement(b))
                    << "query output descriptor";
            } else {
                ASSERT_TRUE(attention->getRaggedKeyInput().has_value());
                ASSERT_TRUE(attention->getRaggedValueInput().has_value());
                EXPECT_EQ(attention->getRaggedKeyInput()->getValuesDimensions(),
                          (std::vector<uint64_t>{kKVMax, kFeatures}));
                EXPECT_EQ(attention->getRaggedValueInput()->getValuesDimensions(),
                          (std::vector<uint64_t>{kKVMax, kFeatures}));
                ASSERT_NO_THROW((void)attention->getRaggedKeyInput()->getDescriptorForPlacement(b))
                    << "KV key descriptor";
                ASSERT_NO_THROW((void)attention->getRaggedValueInput()->getDescriptorForPlacement(b))
                    << "KV value descriptor";
            }
            std::vector<Event> events;
            std::shared_ptr<Api::PlacedNetwork> placed;
            ASSERT_NO_THROW(placed = network.place(b, events, true))
                << "mixed neutral composite placement failed, queryRagged=" << queryRagged
                << " B=" << b;
            ASSERT_NE(placed, nullptr);
            for (auto& event : events) event.synchronize();
            auto physical = std::dynamic_pointer_cast<Impl::Attention>(
                placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(attention->getId()));
            ASSERT_NE(physical, nullptr);
            setIdentityProjections2D2(physical);
            for (uint32_t pattern : {0u, 1u}) {
                SCOPED_TRACE(::testing::Message() << "pattern=" << pattern);
                // Constructing the batch and submitting it are distinct stages;
                // identify which stage first creates invalid ragged geometry.
                Batch inputBatch;
                ASSERT_NO_THROW(inputBatch = mixedBatch2D4B(b, pattern, queryRagged, *attention))
                    << "mixed input batch construction failed";
                std::map<std::string, Api::InferenceOutputValue> output;
                ASSERT_NO_THROW(output = placed->inferLogical(inputBatch))
                    << "mixed composite inference failed, queryRagged=" << queryRagged
                    << " B=" << b << " pattern=" << pattern;
                ASSERT_TRUE(output.contains("out"));
                checkMixedCpu2D4B(output.at("out"), b, pattern, queryRagged);
            }
            EXPECT_EQ(network.architectureJson(), architectureBefore);
        }
        std::filesystem::remove_all(archive);
    }
}

// The two directions have different physical output domains. Keep each gate
// independently identifiable: the generic gtest C++ exception wrapper from the
// previous combined case obscured which direction failed.
TEST(AttentionNeutral2D4, RaggedQueryDenseKvB2B7SavedCompositeMatchCpu) {
    runMixedDomains2D4B(true);
}

TEST(AttentionNeutral2D4, DenseQueryRaggedKvB2B7SavedCompositeMatchCpu) {
    runMixedDomains2D4B(false);
}
