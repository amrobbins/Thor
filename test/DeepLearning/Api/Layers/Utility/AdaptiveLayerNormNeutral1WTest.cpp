#include "DeepLearning/Api/Layers/Utility/AdaptiveLayerNorm.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Common/Stream.h"
#include "gtest/gtest.h"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <map>
#include <variant>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;
namespace {
constexpr uint32_t kM1W = 4;
constexpr uint32_t kC1W = 3;

RaggedTensor tokens1W(Network& n) {
    return RaggedNetworkInput::Builder().network(n).name("tokens")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT64)
        .trailingDimensions({kC1W}).maxValuesPerRow(kM1W).build();
}

struct AdaptiveFixture1W {
    RaggedTensor tokens;
    AdaptiveLayerNorm layer;
};

AdaptiveFixture1W make1W(Network& n) {
    const auto tokens = tokens1W(n);
    const auto scale = NetworkInput::Builder().network(n).name("scale")
        .dimensions({kC1W}).dataType(DataType::FP32).build();
    const auto bias = NetworkInput::Builder().network(n).name("bias")
        .dimensions({kC1W}).dataType(DataType::FP32).build();
    const auto layer = AdaptiveLayerNorm::Builder().network(n).featureInput(tokens)
        .scaleInput(scale.getFeatureOutput().value())
        .biasInput(bias.getFeatureOutput().value()).build();
    (void)RaggedNetworkOutput::Builder().network(n).name("normalized")
        .inputTensor(layer.getRaggedFeatureOutput().value()).build();
    return {tokens, layer};
}

// Keep all three externally supplied differentiable inputs live through
// backward. Without these test-only rivets the network may legitimately prune
// gradients at its input boundary before this test can inspect them.
AdaptiveFixture1W makeTraining1W(Network& n) {
    const auto tokens = tokens1W(n);
    const auto scale = NetworkInput::Builder().network(n).name("scale")
        .dimensions({kC1W}).dataType(DataType::FP32).build();
    const auto bias = NetworkInput::Builder().network(n).name("bias")
        .dimensions({kC1W}).dataType(DataType::FP32).build();
    const auto valuesRivet = GradientRivet::Builder().network(n)
        .tensor(tokens.getValues()).build();
    const auto scaleRivet = GradientRivet::Builder().network(n)
        .tensor(scale.getFeatureOutput().value()).build();
    const auto biasRivet = GradientRivet::Builder().network(n)
        .tensor(bias.getFeatureOutput().value()).build();
    const auto layer = AdaptiveLayerNorm::Builder().network(n)
        .featureInput(tokens.withValues(valuesRivet.getFeatureOutput().value()))
        .scaleInput(scaleRivet.getFeatureOutput().value())
        .biasInput(biasRivet.getFeatureOutput().value()).build();
    return {tokens, layer};
}

class AdaptiveProbe1W : public AdaptiveLayerNorm {
 public:
    explicit AdaptiveProbe1W(const AdaptiveLayerNorm& original) : AdaptiveLayerNorm(original) {}
    using AdaptiveLayerNorm::stamp;
};

vector<uint64_t> rowEnds1W(uint32_t b) {
    vector<uint64_t> ends(b + 1, 0);
    for (uint32_t r = 0; r < b; ++r)
        ends[r+1] = ends[r] + (r % 4 == 0 ? 0u : r % 4 == 1 ? 2u : r % 4 == 2 ? 1u : 3u);
    return ends;
}

float value1W(uint64_t t, uint32_t c) {
    return 0.21f * t + (c == 0 ? -0.5f : c == 1 ? 0.2f : 0.9f) + 0.021f * ((t + c) % 3);
}
float scale1W(uint32_t r, uint32_t c) { return 0.8f + 0.13f * r + 0.05f * c; }
float bias1W(uint32_t r, uint32_t c) { return -0.12f + 0.08f * r - 0.04f * c; }

Batch batch1W(const RaggedTensor& tokens, uint32_t b) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    const auto d = tokens.getDescriptorForPlacement(b);
    Impl::Tensor values(cpu, d.getValuesDescriptor());
    const auto ends = rowEnds1W(b);
    for (uint64_t t = 0; t < b*kM1W; ++t)
        for (uint32_t c = 0; c < kC1W; ++c)
            values.getMemPtr<float>()[t*kC1W+c] =
                t < ends.back() ? value1W(t,c) : numeric_limits<float>::quiet_NaN();
    Impl::Tensor offsets(cpu, d.getOffsetsDescriptor());
    for (uint32_t r = 0; r <= b; ++r) offsets.getMemPtr<uint64_t>()[r] = ends[r];
    Impl::RowPartitionRuntime partition(offsets, d.getRowPartition());
    partition.setHostOffsets(ends);
    Impl::Tensor scale(cpu, Impl::TensorDescriptor(DataType::FP32, {b,kC1W}));
    Impl::Tensor bias(cpu, Impl::TensorDescriptor(DataType::FP32, {b,kC1W}));
    for (uint32_t r = 0; r < b; ++r)
        for (uint32_t c = 0; c < kC1W; ++c) {
            scale.getMemPtr<float>()[r*kC1W+c] = scale1W(r,c);
            bias.getMemPtr<float>()[r*kC1W+c] = bias1W(r,c);
        }
    Batch result;
    result.insert("tokens", Impl::RaggedTensor(values, partition));
    result.insert("scale", scale);
    result.insert("bias", bias);
    result.setValidExampleCount(b);
    return result;
}

shared_ptr<AdaptiveLayerNorm> find1W(Network& n) {
    for (uint32_t i = 0; i < n.getNumLayers(); ++i) {
        auto value = dynamic_pointer_cast<AdaptiveLayerNorm>(n.getLayer(i));
        if (value) return value;
    }
    return {};
}
filesystem::path archive1W(const char* name) {
    return filesystem::temp_directory_path() /
        (string(name) + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()) + ".tar");
}

float target1W(uint64_t t, uint32_t c) { return 0.16f * (t % 5) + 0.10f * c; }

Batch trainingBatch1W(const RaggedTensor& tokens, const RaggedTensor& labels, uint32_t b) {
    Batch result = batch1W(tokens,b);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor targets(cpu, labels.getDescriptorForPlacement(b).getValuesDescriptor());
    const auto ends = rowEnds1W(b);
    for (uint64_t t = 0; t < b*kM1W; ++t)
        for (uint32_t c = 0; c < kC1W; ++c)
            targets.getMemPtr<float>()[t*kC1W+c] =
                t < ends.back() ? target1W(t,c) : numeric_limits<float>::quiet_NaN();
    result.insert("labels", result.getRaggedTensor("tokens").withValues(targets));
    return result;
}

vector<float> download1W(Impl::Tensor source, uint64_t size) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = source.clone(cpu);
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    host.downloadSection(source,stream,0,0,size*sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(),host.getMemPtr<float>()+size);
}

float tokenLoss1W(const array<float,kC1W>& x, const array<float,kC1W>& scale,
                  const array<float,kC1W>& bias, uint64_t t) {
    float mean = 0;
    for (float v : x) mean += v/kC1W;
    float var = 0;
    for (float v : x) var += (v-mean)*(v-mean)/kC1W;
    float total = 0;
    for (uint32_t c = 0; c < kC1W; ++c) {
        float delta = ((x[c]-mean)/sqrtf(var+1.0e-5f))*scale[c] + bias[c] - target1W(t,c);
        total += delta*delta;
    }
    return total;
}

float normalizedValue1W(uint64_t t, uint32_t c) {
    array<float,kC1W> x;
    for (uint32_t k = 0; k < kC1W; ++k) x[k] = value1W(t,k);
    float mean = 0;
    for (float v : x) mean += v / kC1W;
    float var = 0;
    for (float v : x) var += (v-mean)*(v-mean)/kC1W;
    return (x[c]-mean) / sqrtf(var + 1.0e-5f);
}

} // namespace

TEST(RaggedAdaptiveLayerNormNeutral1W, SemanticSchemaMemoryAndLegacy) {
    Network n("adaptive_1w_schema");
    const auto fixture = make1W(n);
    const auto out = fixture.layer.getRaggedFeatureOutput().value();
    EXPECT_TRUE(out.sharesPartitionWith(fixture.tokens));
    EXPECT_NE(out.getValues().getOriginalId(), fixture.tokens.getValues().getOriginalId());
    EXPECT_EQ(fixture.layer.getFeatureInputs().size(), 4u);
    EXPECT_EQ(fixture.layer.getFeatureInputs()[1], fixture.tokens.getRowPartitionToken());
    EXPECT_EQ(fixture.layer.getRaggedPartitionRequirementForInput(fixture.tokens.getRowPartitionToken()),
              Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2u,7u}) {
        EXPECT_EQ(fixture.layer.getOutputTensorBytes(b), b*kM1W*kC1W*sizeof(float));
        EXPECT_EQ(fixture.layer.getFirstInstanceMemRequirementInBytes(b,cpu), fixture.layer.getOutputTensorBytes(b));
        EXPECT_EQ(fixture.layer.getNonFirstInstanceMemRequirementInBytes(b,cpu), fixture.layer.getOutputTensorBytes(b));
    }
    EXPECT_ANY_THROW((void)fixture.layer.getOutputTensorBytes(0));
    const json j = fixture.layer.architectureJson();
    ASSERT_EQ(j.at("version"), "1.1.0");
    for (const char* key : {"ragged_feature_input", "ragged_feature_output"}) {
        const json& record = j.at(key);
        EXPECT_FALSE(record.contains("geometry_mode"));
        EXPECT_FALSE(record.contains("batch_size"));
        EXPECT_FALSE(record.contains("max_total_values"));
        EXPECT_EQ(record.at("max_values_per_row"), kM1W);
    }
    auto reject = [&](json jBad) { EXPECT_ANY_THROW(AdaptiveLayerNorm::deserialize(jBad, &n)); };
    json bad = j; bad["version"] = "1.0.0"; reject(bad);
    bad = j; bad["some_placement_field"] = 7; reject(bad);
    bad = j; bad["ragged_feature_input"]["batch_size"] = 2; reject(bad);
    bad = j; bad["ragged_feature_output"]["max_total_values"] = 28; reject(bad);
    bad = j; bad["ragged_feature_input"]["max_values_per_row"] = 5; reject(bad);
    bad = j; bad["ragged_feature_input"]["offsets"]["id"] = fixture.tokens.getValues().getOriginalId(); reject(bad);
    bad = j; bad["ragged_feature_input"]["values"]["dimensions"] = vector<uint64_t>{5,kC1W}; reject(bad);
    bad = j; bad["ragged_feature_output"]["values"]["data_type"] = DataType::BF16; reject(bad);
    bad = j; bad["ragged_feature_output"]["offsets"]["id"] = fixture.tokens.getValues().getOriginalId(); reject(bad);
    bad = j; bad["inputs"][1]["port"] = "bias_input"; reject(bad);
    bad = j; bad["inputs"][2]["dimensions"] = vector<uint64_t>{4}; reject(bad);
    bad = j; bad["normalized_shape"] = vector<uint64_t>{4}; reject(bad);
    bad = j; bad["outputs"][0]["id"] = fixture.tokens.getValues().getOriginalId(); reject(bad);

    auto historical = [](json ref) {
        ref.erase("layer_ragged_reference_version");
        ref.erase("geometry_mode");
        ref["batch_size"] = 91;      // obsolete execution hint, not semantic M
        ref["max_total_values"] = 999;
        ref["values"]["dimensions"][0] = 123;
        ref["offsets"]["dimensions"] = vector<uint64_t>{92};
        return ref;
    };
    json historicalLayer = j;
    historicalLayer["version"] = "1.0.0";
    historicalLayer["ragged_feature_input"] = historical(historicalLayer["ragged_feature_input"]);
    historicalLayer["ragged_feature_output"] = historical(historicalLayer["ragged_feature_output"]);
    historicalLayer["inputs"][0]["dimensions"][0] = 123;
    historicalLayer["outputs"][0]["dimensions"][0] = 123;
    ASSERT_NO_THROW(AdaptiveLayerNorm::deserialize(historicalLayer, &n));
    auto recovered = dynamic_pointer_cast<AdaptiveLayerNorm>(n.getLayer(n.getNumLayers()-1));
    ASSERT_NE(recovered, nullptr);
    ASSERT_TRUE(recovered->getRaggedFeatureOutput().has_value());
    for (uint32_t b : {2u,7u})
        EXPECT_EQ(recovered->getRaggedFeatureOutput()->getDescriptorForPlacement(b).getMaxTotalValues(), b*kM1W);
    bad = historicalLayer; bad["ragged_feature_input"]["max_values_per_row"] = 8; reject(bad);

    Network dense("adaptive_1w_dense");
    const auto ordinary = AdaptiveLayerNorm::Builder().network(dense)
        .featureInput(Tensor(DataType::FP32, {8,32})).scaleInput(Tensor(DataType::FP32, {32}))
        .biasInput(Tensor(DataType::FP32, {32})).build();
    EXPECT_EQ(ordinary.architectureJson().at("version"), "1.0.0");
}

TEST(RaggedAdaptiveLayerNormNeutral1W, ValuesOffsetsScaleAndBiasCanDriveIndependentStamp) {
    Network n("adaptive_1w_drivers");
    const auto fixture = make1W(n);
    AdaptiveProbe1W probe(fixture.layer);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2u,7u}) {
        const auto desc = fixture.tokens.getDescriptorForPlacement(b);
        const auto physicalInputs = fixture.layer.getFeatureInputs();
        const vector<vector<uint64_t>> shapes = {
            desc.getValuesDescriptor().getDimensions(), desc.getOffsetsDescriptor().getDimensions(),
            {b,kC1W}, {b,kC1W}};
        for (uint32_t port = 0; port < 4; ++port) {
            const DataType dtype = port == 1 ? fixture.tokens.getOffsetsDataType() : DataType::FP32;
            auto driver = make_shared<Impl::NetworkInput>(cpu, dtype, shapes[port]);
            auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
                probe.stamp(cpu,driver,nullptr,physicalInputs[port],true));
            ASSERT_NE(physical,nullptr) << "B=" << b << " input port=" << port;
        }
        auto wrongScale = make_shared<Impl::NetworkInput>(cpu,DataType::FP32,vector<uint64_t>{b,4});
        EXPECT_ANY_THROW((void)probe.stamp(cpu,wrongScale,nullptr,physicalInputs[2],true));
        auto wrongOffsets = make_shared<Impl::NetworkInput>(cpu,DataType::UINT64,vector<uint64_t>{1});
        EXPECT_ANY_THROW((void)probe.stamp(cpu,wrongOffsets,nullptr,physicalInputs[1],true));
    }
}

TEST(RaggedAdaptiveLayerNormNeutral1W, SavedB2AndB7InferenceRespectsNonuniformConditioningAndEmptyRows) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    const auto archive = archive1W("adaptive_1w_infer");
    {
        Network n("adaptive_1w_infer");
        (void)make1W(n);
        n.save(archive.string(), true);
    }
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network n("adaptive_1w_infer");
        ASSERT_NO_THROW(n.load(archive.string()));
        const auto tokens = n.getExternalRaggedNetworkInputs().front().raggedTensor;
        const auto layer = find1W(n);
        ASSERT_NE(layer,nullptr);
        EXPECT_TRUE(layer->getRaggedFeatureOutput()->sharesPartitionWith(tokens));
        const json before = n.architectureJson();
        vector<Event> events;
        shared_ptr<PlacedNetwork> placed;
        ASSERT_NO_THROW(placed = n.place(b,events,true));
        ASSERT_NE(placed,nullptr);
        for (auto& e : events) e.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        const auto requirement = stamped.getExternalRowPartitionRequirementsForTest(tokens.getRowPartitionId());
        ASSERT_TRUE(requirement.has_value());
        EXPECT_EQ(requirement.value(), Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);
        EXPECT_NE(stamped.getManagedPartitionOffsetsInputForTest(tokens.getRowPartitionId()),nullptr);
        auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical,nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b*kM1W,kC1W}));
        EXPECT_EQ(n.architectureJson(),before);
        const auto result = placed->inferLogical(batch1W(tokens,b));
        ASSERT_TRUE(result.contains("normalized"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(result.at("normalized")));
        const auto output = get<Impl::RaggedTensor>(result.at("normalized"));
        const auto ends = output.getHostOffsetsIfAvailable();
        ASSERT_TRUE(ends.has_value());
        EXPECT_EQ(ends.value(),rowEnds1W(b));
        const float* y = output.getValues().getMemPtr<float>();
        for (uint32_t row = 0; row < b; ++row)
            for (uint64_t t = (*ends)[row]; t < (*ends)[row+1]; ++t)
                for (uint32_t c = 0; c < kC1W; ++c)
                    EXPECT_NEAR(y[t*kC1W+c],normalizedValue1W(t,c)*scale1W(row,c)+bias1W(row,c),3e-3f)
                        << "row=" << row << " token=" << t << " channel=" << c;
    }
    filesystem::remove_all(archive);
}

// LN-G: independently reload one saved architecture at B=2 and B=7 and
// validate values plus per-example scale/bias derivatives on the active prefix.
TEST(RaggedAdaptiveLayerNormNeutral1W, SavedBackwardToValuesScaleAndBias) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    Network n("adaptive_1w_training_future");
    const auto fixture = makeTraining1W(n);
    const auto labels = RaggedNetworkInput::Builder().network(n).name("labels")
        .valuesDataType(DataType::FP32)
        .trailingDimensions({kC1W}).partition(fixture.tokens).build();
    const auto pred = Impl::Expression::input("predictions", DataType::FP32,DataType::FP32);
    const auto gold = Impl::Expression::input("labels", DataType::FP32,DataType::FP32);
    const auto delta = pred-gold;
    const auto loss = Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            "loss",(delta*delta).withOutputDType(DataType::FP32)}})));
    const auto grad = Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            "predictions_grad",(delta*Impl::Expression(2.0*Impl::Loss::getLossScalingFactor()))
                 .withOutputDType(DataType::FP32)}})));
    const auto raw = RaggedCustomLoss::Builder().network(n)
        .predictions(fixture.layer.getRaggedFeatureOutput().value())
        .labels(labels).lossExpression(loss).gradientExpression(grad).build();
    (void)NetworkOutput::Builder().network(n).name("raw_loss")
        .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
    const auto archive = archive1W("adaptive_1w_training_future");
    n.save(archive.string(),true);
    for (uint32_t b : {2u,7u}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network reloaded("adaptive_1w_training_future");
        ASSERT_NO_THROW(reloaded.load(archive.string()));
        const json before = reloaded.architectureJson();
        vector<Event> events;
        auto placed = reloaded.place(b,events,false);
        ASSERT_NE(placed,nullptr);
        for (auto& e : events) e.synchronize();
        auto layer = find1W(reloaded);
        ASSERT_NE(layer,nullptr);
        auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical,nullptr);
        const auto external = reloaded.getExternalRaggedNetworkInputs();
        const auto tokensIt = find_if(external.begin(),external.end(),[](const auto& e){return e.name=="tokens";});
        const auto labelsIt = find_if(external.begin(),external.end(),[](const auto& e){return e.name=="labels";});
        ASSERT_NE(tokensIt,external.end());
        ASSERT_NE(labelsIt,external.end());
        map<string,Impl::Tensor> results;
        map<string,Event> ready;
        auto completed = placed->submitBatch(0,trainingBatch1W(tokensIt->raggedTensor,labelsIt->raggedTensor,b),
                                             results,ready,false);
        completed.synchronize();
        placed->synchronize();
        ASSERT_TRUE(results.contains("raw_loss"));
        EXPECT_EQ(reloaded.architectureJson(),before);
        const auto ends = rowEnds1W(b);
        ASSERT_GT(ends.back(),0u);
        ASSERT_LT(ends.back(),static_cast<uint64_t>(b)*kM1W)
            << "Poisoned inactive storage must be present in this test";
        const auto activeLoss = download1W(results.at("raw_loss"),ends.back()*kC1W);
        for (float v : activeLoss) EXPECT_TRUE(isfinite(v));
        // All four physical inputs must preserve their distinct gradient
        // obligations. Offsets are non-differentiable, values and the two dense
        // conditioning ports have nonzero gradients on active rows only.
        const auto errors = physical->getErrorOutputs();
        ASSERT_EQ(errors.size(),4u);
        ASSERT_TRUE(errors[0].has_value());
        EXPECT_FALSE(errors[1].has_value());
        ASSERT_TRUE(errors[2].has_value());
        ASSERT_TRUE(errors[3].has_value());
        const auto dx = download1W(errors[0].value(),b*kM1W*kC1W);
        const auto dscale = download1W(errors[2].value(),b*kC1W);
        const auto dbias = download1W(errors[3].value(),b*kC1W);
        const float factor = Impl::Loss::getLossScalingFactor();
        constexpr float h = 0.001f;
        for (uint32_t r = 0; r < b; ++r) {
            array<float,kC1W> scale, bias;
            for (uint32_t c = 0; c < kC1W; ++c) {
                scale[c] = scale1W(r,c);
                bias[c] = bias1W(r,c);
            }
            for (uint64_t t = ends[r]; t < ends[r+1]; ++t) {
                array<float,kC1W> x;
                for (uint32_t c = 0; c < kC1W; ++c) x[c] = value1W(t,c);
                for (uint32_t c = 0; c < kC1W; ++c) {
                    auto upper=x, lower=x;
                    upper[c]+=h; lower[c]-=h;
                    const float numerical = (tokenLoss1W(upper,scale,bias,t)-
                                             tokenLoss1W(lower,scale,bias,t))/(2*h);
                    EXPECT_NEAR(dx[t*kC1W+c]/factor,numerical,3e-2f)
                        << "B=" << b << " row=" << r << " token=" << t << " channel=" << c;
                }
            }
            for (uint32_t c = 0; c < kC1W; ++c) {
                auto scalePlus=scale,scaleMinus=scale,biasPlus=bias,biasMinus=bias;
                scalePlus[c]+=h; scaleMinus[c]-=h;
                biasPlus[c]+=h; biasMinus[c]-=h;
                float numericalScale = 0, numericalBias = 0;
                for (uint64_t t = ends[r]; t < ends[r+1]; ++t) {
                    array<float,kC1W> x;
                    for (uint32_t k = 0; k < kC1W; ++k) x[k] = value1W(t,k);
                    numericalScale += (tokenLoss1W(x,scalePlus,bias,t)-
                                       tokenLoss1W(x,scaleMinus,bias,t))/(2*h);
                    numericalBias += (tokenLoss1W(x,scale,biasPlus,t)-
                                      tokenLoss1W(x,scale,biasMinus,t))/(2*h);
                }
                EXPECT_NEAR(dscale[r*kC1W+c]/factor,numericalScale,3e-2f);
                EXPECT_NEAR(dbias[r*kC1W+c]/factor,numericalBias,3e-2f);
            }
        }
    }
    filesystem::remove_all(archive);
}
