#include "DeepLearning/Api/Layers/Utility/LayerNorm.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
constexpr uint32_t kWidth1V = 3;
constexpr uint32_t kBound1V = 4;

filesystem::path archivePath1V(const string& tag) {
    return filesystem::temp_directory_path() /
        (tag + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()) + ".tar");
}

RaggedTensor input1V(Network& network, const string& name = "tokens") {
    return RaggedNetworkInput::Builder().network(network).name(name)
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({kWidth1V}).maxValuesPerRow(kBound1V).build();
}

template<class T>
shared_ptr<T> layer1V(Network& network) {
    shared_ptr<T> result;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        if (auto candidate = dynamic_pointer_cast<T>(network.getLayer(i))) {
            if (result) throw runtime_error("Expected one LayerNorm");
            result = candidate;
        }
    }
    return result;
}

vector<uint64_t> offsets1V(uint32_t b) {
    vector<uint64_t> ends(b + 1, 0);
    for (uint32_t i = 0; i < b; ++i)
        ends[i + 1] = ends[i] + (i % 4 == 0 ? 0u : i % 4 == 1 ? 2u : i % 4 == 2 ? 1u : 3u);
    return ends;
}

float value1V(uint64_t token, uint32_t channel) {
    return 0.13f * static_cast<float>(token) +
        (channel == 0 ? 0.25f : channel == 1 ? -0.40f : 0.70f) +
        0.023f * static_cast<float>((token + channel) % 3);
}

float target1V(uint64_t token, uint32_t channel) {
    return static_cast<float>(0.07 * static_cast<double>(token % 5)) +
        (channel == 0 ? 0.4f : channel == 1 ? -0.3f : 0.2f);
}

Batch batch1V(const RaggedTensor& tokens, uint32_t b, const optional<RaggedTensor>& labels = nullopt) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    const auto desc = tokens.getDescriptorForPlacement(b);
    Impl::Tensor values(cpu, desc.getValuesDescriptor());
    const auto ends = offsets1V(b);
    for (uint64_t t = 0; t < static_cast<uint64_t>(b) * kBound1V; ++t)
        for (uint32_t c = 0; c < kWidth1V; ++c)
            values.getMemPtr<float>()[t * kWidth1V + c] =
                t < ends.back() ? value1V(t, c) : numeric_limits<float>::quiet_NaN();
    Impl::Tensor offsets(cpu, desc.getOffsetsDescriptor());
    for (uint32_t i = 0; i <= b; ++i) offsets.getMemPtr<uint64_t>()[i] = ends[i];
    Impl::RowPartitionRuntime partition(offsets, desc.getRowPartition());
    partition.setHostOffsets(ends);
    Batch batch;
    batch.insert("tokens", Impl::RaggedTensor(values, partition));
    if (labels) {
        Impl::Tensor targets(cpu, labels->getDescriptorForPlacement(b).getValuesDescriptor());
        for (uint64_t t = 0; t < static_cast<uint64_t>(b) * kBound1V; ++t)
            for (uint32_t c = 0; c < kWidth1V; ++c)
                targets.getMemPtr<float>()[t * kWidth1V + c] =
                    t < ends.back() ? target1V(t, c) : numeric_limits<float>::quiet_NaN();
        batch.insert("labels", Impl::RaggedTensor(targets, partition));
    }
    batch.setValidExampleCount(b);
    return batch;
}

vector<float> download1V(Impl::Tensor source, uint64_t count) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = source.clone(cpu);
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    host.downloadSection(source, stream, 0, 0, count * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + count);
}

// CPU objective of one active token. Uses population variance as cuDNN LN does.
float tokenLoss1V(const array<float,3>& x, const array<float,3>& gamma,
                  const array<float,3>& beta, uint64_t token) {
    float mean = (x[0] + x[1] + x[2]) / 3.0f;
    float var = 0;
    for (float v : x) var += (v - mean) * (v - mean) / 3.0f;
    const float inv = 1.0f / sqrtf(var + 1.0e-5f);
    float loss = 0;
    for (uint32_t c = 0; c < 3; ++c) {
        const float d = gamma[c] * (x[c] - mean) * inv + beta[c] - target1V(token, c);
        loss += d * d;
    }
    return loss;
}

} // namespace

TEST(RaggedLayerNormNeutral1V, SemanticSchemaMemoryAndLegacyReconstruction) {
    Network network("layer_norm_1v_schema");
    const RaggedTensor input = input1V(network);
    LayerNorm layer = LayerNorm::Builder().network(network).featureInput(input).build();
    ASSERT_TRUE(layer.getRaggedFeatureOutput().has_value());
    const auto out = layer.getRaggedFeatureOutput().value();
    EXPECT_TRUE(out.sharesPartitionWith(input));
    EXPECT_EQ(out.getValuesDimensions(), (vector<uint64_t>{kBound1V, kWidth1V}));
    EXPECT_EQ(out.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_NE(input.getValues().getOriginalId(), out.getValues().getOriginalId());
    EXPECT_EQ(layer.getRaggedPartitionRequirementForInput(input.getOffsets()),
              Impl::RaggedPartitionRequirement::HOST_EXTENT);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2u, 7u}) {
        EXPECT_EQ(layer.getOutputTensorBytes(b), b * kBound1V * kWidth1V * sizeof(float));
        EXPECT_EQ(layer.getNonFirstInstanceMemRequirementInBytes(b, cpu), layer.getOutputTensorBytes(b));
        EXPECT_EQ(layer.getFirstInstanceMemRequirementInBytes(b, cpu),
                  layer.getOutputTensorBytes(b) + 2u * kWidth1V * sizeof(float));
    }
    EXPECT_ANY_THROW((void)layer.getOutputTensorBytes(0));
    const json record = layer.architectureJson();
    ASSERT_EQ(record.at("version"), "1.1.0");
    for (auto key : {"ragged_inputs", "ragged_outputs"}) {
        const json& r = record.at(key).at(0);
        EXPECT_FALSE(r.contains("geometry_mode"));
        EXPECT_FALSE(r.contains("batch_size"));
        EXPECT_FALSE(r.contains("max_total_values"));
    }
    shared_ptr<thor_file::TarReader> reader;
    auto reject = [&](const json& data) { EXPECT_ANY_THROW(LayerNorm::deserialize(reader, data, &network)); };
    json bad = record; bad["version"] = "1.0.0"; reject(bad);
    bad = record; bad["extra_runtime_batch_size"] = 7; reject(bad);
    bad = record; bad["ragged_inputs"][0]["batch_size"] = 2; reject(bad);
    bad = record; bad["ragged_outputs"][0]["max_total_values"] = 28; reject(bad);
    bad = record; bad["ragged_inputs"][0]["max_values_per_row"] = 5; reject(bad);
    bad = record; bad["ragged_inputs"][0]["values"]["dimensions"] = vector<uint64_t>{5,3}; reject(bad);
    bad = record; bad["ragged_outputs"][0]["offsets"]["id"] = input.getValues().getOriginalId(); reject(bad);
    bad = record; bad["ragged_outputs"][0]["values"]["data_type"] = DataType::FP16; reject(bad);
    bad = record; bad["inputs"][0]["data_type"] = DataType::FP16; reject(bad);
    bad = record; bad["outputs"][0]["dimensions"] = vector<uint64_t>{4,4}; reject(bad);
    bad = record; bad["normalized_shape"] = vector<uint64_t>{4}; reject(bad);
    bad = record; bad["parameters"]["weights"]["shape"] = vector<uint64_t>{4}; reject(bad);

    auto historical = [](json ref) {
        ref.erase("layer_ragged_reference_version");
        ref.erase("geometry_mode");
        ref["batch_size"] = 91;      // obsolete execution hint, not semantic M
        ref["max_total_values"] = 999;
        ref["values"]["dimensions"][0] = 123;
        ref["offsets"]["dimensions"] = vector<uint64_t>{92};
        return ref;
    };
    json historicalLayer = record;
    historicalLayer["version"] = "1.0.0";
    historicalLayer["ragged_inputs"][0] = historical(historicalLayer["ragged_inputs"][0]);
    historicalLayer["ragged_outputs"][0] = historical(historicalLayer["ragged_outputs"][0]);
    historicalLayer["inputs"][0]["dimensions"][0] = 123;
    historicalLayer["outputs"][0]["dimensions"][0] = 123;
    ASSERT_NO_THROW(LayerNorm::deserialize(reader, historicalLayer, &network));
    auto recovered = dynamic_pointer_cast<LayerNorm>(network.getLayer(network.getNumLayers()-1));
    ASSERT_NE(recovered, nullptr);
    ASSERT_TRUE(recovered->getRaggedFeatureOutput().has_value());
    for (uint32_t b : {2u,7u})
        EXPECT_EQ(recovered->getRaggedFeatureOutput()->getDescriptorForPlacement(b).getMaxTotalValues(), b*kBound1V);
    bad = historicalLayer; bad["ragged_outputs"][0]["offsets"]["id"] = input.getValues().getOriginalId(); reject(bad);

    Network denseNetwork("layer_norm_1v_dense");
    LayerNorm dense = LayerNorm::Builder().network(denseNetwork)
        .featureInput(Tensor(DataType::FP32, {2,3})).build();
    EXPECT_EQ(dense.architectureJson().at("version"), "1.0.0");
}

TEST(RaggedLayerNormNeutral1V, SavedInferenceAtB2AndB7NormalizesOnlyActiveValues) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const auto archive = archivePath1V("layer_norm_1v_infer");
    {
        Network network("layer_norm_1v_infer");
        const auto tokens = input1V(network);
        const auto layer = LayerNorm::Builder().network(network).featureInput(tokens).build();
        (void)RaggedNetworkOutput::Builder().network(network).name("normalized")
            .inputTensor(layer.getRaggedFeatureOutput().value()).build();
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network network("layer_norm_1v_infer");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto tokens = network.getExternalRaggedNetworkInputs().front().raggedTensor;
        auto layer = layer1V<LayerNorm>(network);
        ASSERT_NE(layer, nullptr);
        EXPECT_TRUE(layer->getRaggedFeatureOutput()->sharesPartitionWith(tokens));
        const json before = network.architectureJson();
        vector<Event> events;
        shared_ptr<PlacedNetwork> placed;
        ASSERT_NO_THROW(placed = network.place(b, events, true));
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        auto& stamp = placed->getStampedNetwork(0);
        // HOST_EXTENT is authoritative host metadata routed through the packed
        // values carrier. It does not allocate DEVICE_ACTIVE_COUNT [1] or
        // DEVICE_OFFSETS [B+1] inputs (see StampedNetwork::RowPartitionPhysicalization).
        const auto requirements = stamp.getExternalRowPartitionRequirementsForTest(tokens.getRowPartitionId());
        ASSERT_TRUE(requirements.has_value());
        EXPECT_EQ(requirements.value(), Impl::RaggedPartitionRequirement::HOST_EXTENT);
        EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(tokens.getRowPartitionId()), nullptr);
        EXPECT_EQ(stamp.getManagedPartitionActiveCountInputForTest(tokens.getRowPartitionId()), nullptr);
        auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
            stamp.getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{b * kBound1V, kWidth1V}));
        EXPECT_EQ(network.architectureJson(), before);
        const auto results = placed->inferLogical(batch1V(tokens, b));
        ASSERT_TRUE(results.contains("normalized"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(results.at("normalized")));
        const auto result = get<Impl::RaggedTensor>(results.at("normalized"));
        const auto ends = result.getHostOffsetsIfAvailable();
        ASSERT_TRUE(ends.has_value());
        EXPECT_EQ(ends.value(), offsets1V(b));
        const float* y = result.getValues().getMemPtr<float>();
        for (uint64_t token = 0; token < ends->back(); ++token) {
            array<float,3> x;
            for (uint32_t c = 0; c < 3; ++c) x[c] = value1V(token,c);
            float mean = (x[0]+x[1]+x[2])/3.0f;
            float var = 0;
            for (float v : x) var += (v - mean)*(v - mean)/3.0f;
            for (uint32_t c = 0; c < 3; ++c)
                EXPECT_NEAR(y[token*3+c], (x[c]-mean)/sqrtf(var+1.0e-5f), 2e-3f)
                    << "token " << token << " channel " << c;
        }
    }
    filesystem::remove_all(archive);
}

// LN-F: training must survive save/load and independent B=2/B=7 placement.
// The GradientRivet below retains the externally undriven input gradient for
// finite differences; it is a test-only sink, not a model placement mechanism.
TEST(RaggedLayerNormNeutral1V, SavedTrainingAtB2AndB7MatchesFiniteDifferenceInputGradientAndUpdatesAffineState) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const auto archive = archivePath1V("layer_norm_1v_train");
    {
        Network network("layer_norm_1v_train");
        (void)Sgd::Builder().network(network).initialLearningRate(0.01f)
            .decay(0.0f).momentum(0.0f).build();
        const auto tokens = input1V(network);
        const auto labels = RaggedNetworkInput::Builder().network(network).name("labels")
            .valuesDataType(DataType::FP32).trailingDimensions({kWidth1V}).partition(tokens).build();
        const auto gradientRivet = GradientRivet::Builder().network(network)
            .tensor(tokens.getValues()).build();
        const auto layer = LayerNorm::Builder().network(network)
            .featureInput(tokens.withValues(gradientRivet.getFeatureOutput().value())).build();
        auto pred = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
        auto label = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
        auto delta = pred - label;
        const auto loss = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "loss", (delta*delta).withOutputDType(DataType::FP32)}})));
        const auto grad = Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
                "predictions_grad", (delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()))
                    .withOutputDType(DataType::FP32)}})));
        const auto raw = RaggedCustomLoss::Builder().network(network)
            .predictions(layer.getRaggedFeatureOutput().value()).labels(labels)
            .lossExpression(loss).gradientExpression(grad).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        network.save(archive.string(), true);
    }
    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network network("layer_norm_1v_train");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto external = network.getExternalRaggedNetworkInputs();
        auto tokenIt = find_if(external.begin(), external.end(),
            [](const auto& entry){ return entry.name == "tokens"; });
        auto labelIt = find_if(external.begin(), external.end(),
            [](const auto& entry){ return entry.name == "labels"; });
        ASSERT_NE(tokenIt, external.end());
        ASSERT_NE(labelIt, external.end());
        const auto tokens = tokenIt->raggedTensor;
        const auto labels = labelIt->raggedTensor;
        auto layer = layer1V<LayerNorm>(network);
        ASSERT_NE(layer, nullptr);
        const json before = network.architectureJson();
        vector<Event> events;
        auto placed = network.place(b, events, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : events) event.synchronize();
        EXPECT_EQ(network.architectureJson(), before);
        auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{static_cast<uint64_t>(b) * kBound1V, kWidth1V}));
        const auto weights = physical->getParameterStorage("weights");
        const auto biases = physical->getParameterStorage("biases");
        const auto initialWeights = download1V(weights, 3);
        const auto initialBiases = download1V(biases, 3);
        const auto ends = offsets1V(b);
        ASSERT_GT(ends.back(), 0u);
        ASSERT_LT(ends.back(), static_cast<uint64_t>(b) * kBound1V)
            << "The test needs NaN-poisoned inactive storage to detect tail pollution";
        map<string, Impl::Tensor> results;
        map<string, Event> ready;
        auto completed = placed->submitBatch(0, batch1V(tokens, b, labels), results, ready, false);
        completed.synchronize();
        placed->synchronize();
        ASSERT_TRUE(results.contains("raw_loss"));
        const auto activeLossValues = download1V(results.at("raw_loss"), ends.back() * kWidth1V);
        for (float value : activeLossValues) EXPECT_TRUE(isfinite(value));
        const auto errors = physical->getErrorOutputs();
        ASSERT_EQ(errors.size(), 2u);
        ASSERT_TRUE(errors[0].has_value()) << "LayerNorm active values backward was pruned";
        EXPECT_FALSE(errors[1].has_value()) << "HOST_EXTENT carrier has no gradient";
        const auto actualDx = download1V(errors[0].value(), ends.back() * 3);
        const array<float,3> gamma = {initialWeights[0], initialWeights[1], initialWeights[2]};
        const array<float,3> beta = {initialBiases[0], initialBiases[1], initialBiases[2]};
        const float gradientScale = Impl::Loss::getLossScalingFactor();
        for (uint64_t token = 0; token < ends.back(); ++token) {
            array<float,3> x = {value1V(token,0), value1V(token,1), value1V(token,2)};
            for (uint32_t c = 0; c < 3; ++c) {
                constexpr float h = 0.001f;
                auto upper = x;
                auto lower = x;
                upper[c] += h;
                lower[c] -= h;
                const float numerical = (tokenLoss1V(upper, gamma, beta, token) -
                                         tokenLoss1V(lower, gamma, beta, token)) / (2.0f*h);
                EXPECT_NEAR(actualDx[token*3+c] / gradientScale, numerical, 3e-2f)
                    << "token " << token << " channel " << c;
            }
        }
        const auto finalWeights = download1V(weights, 3);
        const auto finalBiases = download1V(biases, 3);
        bool changedWeight = false, changedBias = false;
        // Directional parameter-gradient check against a central finite difference
        // of the full active-token loss, independent of the optimizer's batch
        // reduction scale. Do not infer a particular learning-rate normalization.
        auto activeLoss = [&](const array<float,3>& g, const array<float,3>& be) {
            float total = 0;
            for (uint64_t token = 0; token < ends.back(); ++token) {
                const array<float,3> x = {value1V(token,0), value1V(token,1), value1V(token,2)};
                total += tokenLoss1V(x, g, be, token);
            }
            return total;
        };
        for (uint32_t c = 0; c < 3; ++c) {
            constexpr float h = 0.001f;
            auto gp = gamma, gm = gamma, bp = beta, bm = beta;
            gp[c] += h; gm[c] -= h;
            bp[c] += h; bm[c] -= h;
            const float numericWeightGrad = (activeLoss(gp, beta) - activeLoss(gm, beta)) / (2*h);
            const float numericBiasGrad = (activeLoss(gamma, bp) - activeLoss(gamma, bm)) / (2*h);
            if (fabs(numericWeightGrad) > 0.03f)
                EXPECT_LT((finalWeights[c] - initialWeights[c]) * numericWeightGrad, 0.0f)
                    << "affine weight gradient direction, c=" << c;
            if (fabs(numericBiasGrad) > 0.03f)
                EXPECT_LT((finalBiases[c] - initialBiases[c]) * numericBiasGrad, 0.0f)
                    << "affine bias gradient direction, c=" << c;
            changedWeight |= fabs(finalWeights[c] - initialWeights[c]) > 1e-7f;
            changedBias |= fabs(finalBiases[c] - initialBiases[c]) > 1e-7f;
            EXPECT_TRUE(isfinite(finalWeights[c]));
            EXPECT_TRUE(isfinite(finalBiases[c]));
        }
        EXPECT_TRUE(changedWeight);
        EXPECT_TRUE(changedBias);
    }
    filesystem::remove_all(archive);
}
