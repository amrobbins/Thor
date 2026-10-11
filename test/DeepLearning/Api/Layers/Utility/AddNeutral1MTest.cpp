#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/Add.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/Event.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"
#include <algorithm>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
RaggedTensor input(Network& n, const string& name = "tokens") {
    return RaggedNetworkInput::Builder().network(n).name(name).valuesDataType(DataType::FP32)
        .offsetsDataType(DataType::UINT32).trailingDimensions({2}).maxValuesPerRow(4).build();
}

Add* onlyAdd(Network& n) {
    for (uint32_t i = 0; i < n.getNumLayers(); ++i) {
        auto add = dynamic_pointer_cast<Add>(n.getLayer(i));
        if (add) return add.get();
    }
    return nullptr;
}

// Deliberately uses two different partition-preserving computation branches;
// they must not collapse into a single values port.
Add branchedAdd(Network& n, RaggedTensor p, bool reversePorts = false, bool preserveInputGradients = false) {
    auto left = Relu::Builder().network(n).featureInput(p).build();
    auto right = Sigmoid::Builder().network(n).featureInput(p).build();
    EXPECT_TRUE(left->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(right->getRaggedFeatureOutput().has_value());
    RaggedTensor l = left->getRaggedFeatureOutput().value();
    RaggedTensor r = right->getRaggedFeatureOutput().value();
    if (preserveInputGradients) {
        // The external input is intentionally a backward-pruning boundary.
        // Independently rivet the two values edges so that neither Add error
        // output is removed before the RAW terminal loss runs backward.
        // These are test-only identity layers, not extra Add operands.
        const auto leftRivet = GradientRivet::Builder().network(n).tensor(l.getValues()).build();
        const auto rightRivet = GradientRivet::Builder().network(n).tensor(r.getValues()).build();
        EXPECT_TRUE(leftRivet.architectureJson().is_object());
        EXPECT_TRUE(rightRivet.architectureJson().is_object());
        l = l.withValues(leftRivet.getFeatureOutput().value());
        r = r.withValues(rightRivet.getFeatureOutput().value());
    }
    return reversePorts ? Add::Builder().network(n).left(r).right(l).build()
                        : Add::Builder().network(n).left(l).right(r).build();
}

Batch batch(RaggedTensor logical, uint32_t b) {
    const auto desc = logical.getDescriptorForPlacement(b);
    const uint64_t capacity = desc.getMaxTotalValues();
    vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t i = 1; i <= b; ++i)
        offsets[i] = offsets[i-1] + (i == 1 || i % 3 == 0 ? 0 : 1 + i % 2);
    const uint64_t active = offsets.back();
    EXPECT_LT(active, capacity);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor data(cpu, desc.getValuesDescriptor());
    for (uint64_t i = 0; i < capacity; ++i)
        for (int c = 0; c < 2; ++c)
            data.getMemPtr<float>()[i * 2 + c] = i < active
                ? (0.3F * static_cast<float>(i) - 0.8F + static_cast<float>(c) * 0.45F)
                : numeric_limits<float>::quiet_NaN();
    Impl::Tensor ends(cpu, desc.getOffsetsDescriptor());
    for (uint32_t i = 0; i <= b; ++i) ends.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(offsets[i]);
    Impl::RowPartitionRuntime p(ends, desc.getRowPartition());
    p.setHostOffsets(offsets);
    Batch result;
    result.insert("tokens", Impl::RaggedTensor(data, p));
    return result;
}

void expectReject(Network& network, json j) {
    EXPECT_ANY_THROW(Add::deserialize(j, &network));
}

Impl::DynamicExpression squaredError(bool gradient) {
    const auto prediction = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    const auto label = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    const auto difference = prediction - label;
    auto result = gradient
        ? difference * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
        : difference * difference;
    result = result.withOutputDType(DataType::FP32);
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(
            Impl::Expression::outputs({{gradient ? "predictions_grad" : "loss", result}})));
}

float tokenValue(uint64_t token, uint32_t channel) {
    return 0.3f * static_cast<float>(token) - 0.8f + 0.45f * static_cast<float>(channel);
}

float expectedSum(uint64_t token, uint32_t channel) {
    const float x = tokenValue(token, channel);
    return max(0.0f, x) + 1.0f / (1.0f + exp(-x));
}

// Distinct signs and magnitudes ensure both value-port gradients are exercised.
float expectedDifference(uint64_t token, uint32_t channel) {
    const float magnitude = 0.12f + 0.07f * static_cast<float>(token % 4) + 0.11f * channel;
    return ((token + channel) % 2 == 0 ? 1.0f : -1.0f) * magnitude;
}

vector<float> downloadActiveElements(const Impl::Tensor& source, uint64_t elements) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = source.clone(cpu);
    Impl::Tensor sourceHandle = source;
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    // Only transfer the defined active packed prefix, never inactive gradients.
    host.downloadSection(sourceHandle, stream, 0, 0, elements * sizeof(float));
    stream.synchronize();
    const float* values = host.getMemPtr<float>();
    return vector<float>(values, values + elements);
}

shared_ptr<RaggedCustomLoss> onlyRawLoss(Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto loss = dynamic_pointer_cast<RaggedCustomLoss>(network.getLayer(i));
        if (loss) return loss;
    }
    return nullptr;
}
} // namespace

TEST(RaggedAddNeutral1M, SemanticCompatibilityStrictReferencesAndLegacy) {
    Network n("add_m_geometry");
    RaggedTensor p = input(n);
    auto a = branchedAdd(n, p);
    ASSERT_TRUE(a.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(a.getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{4,2}));
    EXPECT_EQ(a.getRaggedFeatureOutput()->getRowPartitionToken(), p.getRowPartitionToken());
    const auto inputs = a.getFeatureInputs();
    ASSERT_EQ(inputs.size(), 3U);
    EXPECT_NE(inputs[0], inputs[1]);
    EXPECT_EQ(inputs[2], p.getRowPartitionToken());
    const json j = a.architectureJson();
    EXPECT_EQ(j.at("version"), "1.1.0");
    for (const char* key : {"ragged_left", "ragged_right", "ragged_output"}) {
        const json& r = j.at(key);
        EXPECT_FALSE(r.contains("geometry_mode"));
        EXPECT_EQ(r.at("max_values_per_row"), 4);
        EXPECT_EQ(r.at("values").at("dimensions"), (vector<uint64_t>{4,2}));
        EXPECT_EQ(r.at("offsets").at("dimensions"), (vector<uint64_t>{1}));
        EXPECT_FALSE(r.contains("batch_size"));
        EXPECT_FALSE(r.contains("max_total_values"));
    }
    EXPECT_EQ(j.at("ragged_left").at("offsets"), j.at("ragged_right").at("offsets"));
    EXPECT_EQ(j.at("ragged_left").at("offsets"), j.at("ragged_output").at("offsets"));
    EXPECT_EQ(j.at("ragged_output").at("values"), j.at("feature_output"));
    EXPECT_EQ(a.getOutputTensorBytes(2), 2U * 4U * 2U * sizeof(float));
    EXPECT_EQ(a.getOutputTensorBytes(7), 7U * 4U * 2U * sizeof(float));
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    // Output and two differentiable inputs are B-scaled. [1] structural token charged once.
    EXPECT_EQ(static_cast<const Layer&>(a).getFirstInstanceMemRequirementInBytes(2, cpu),
              3U * 2U * 4U * 2U * sizeof(float) + sizeof(uint32_t));
    EXPECT_EQ(static_cast<const Layer&>(a).getFirstInstanceMemRequirementInBytes(7, cpu),
              3U * 7U * 4U * 2U * sizeof(float) + sizeof(uint32_t));

    json bad = j; bad["version"] = "7.0.0"; expectReject(n, bad);
    bad = j; bad["ragged_right"]["max_values_per_row"] = 5; expectReject(n, bad);
    bad = j; bad["ragged_right"]["values"]["dimensions"] = vector<uint64_t>{4,3}; expectReject(n, bad);
    bad = j; bad["ragged_right"]["values"]["data_type"] = DataType::BF16; expectReject(n, bad);
    bad = j; bad["ragged_right"]["offsets"]["id"] = p.getValues().getOriginalId(); expectReject(n, bad);
    bad = j; bad["ragged_right"]["geometry_mode"] = "fixed"; expectReject(n, bad);
    bad = j; bad["ragged_left"]["batch_size"] = 7; expectReject(n, bad);
    bad = j; bad["ragged_output"]["offsets"]["id"] = p.getValues().getOriginalId(); expectReject(n, bad);
    bad = j; bad["ragged_output"]["values"]["id"] = j.at("ragged_left").at("values").at("id"); expectReject(n, bad);
    bad = j; bad["ragged_output"]["values"]["dimensions"] = vector<uint64_t>{4,3}; expectReject(n, bad);
    bad = j; bad["ragged_output"]["offsets"]["dimensions"] = vector<uint64_t>{2}; expectReject(n, bad);
    bad = j; bad["ragged_output"]["offsets"]["data_type"] = DataType::UINT64; expectReject(n, bad);
    bad = j; bad["feature_output"]["dimensions"] = vector<uint64_t>{4,3};
    bad["ragged_output"]["values"]["dimensions"] = vector<uint64_t>{4,3}; expectReject(n, bad);
    bad = j; bad["feature_output"]["data_type"] = DataType::BF16;
    bad["ragged_output"]["values"]["data_type"] = DataType::BF16; expectReject(n, bad);

    RaggedTensor different = input(n, "different");
    EXPECT_THROW((void)Add::Builder().network(n).left(p).right(different).build(), invalid_argument);
    RaggedTensor wrongM = RaggedTensor::logical(Tensor(DataType::FP32, {5,2}), p.getRowPartitionToken(), 5);
    EXPECT_THROW((void)Add::Builder().network(n).left(p).right(wrongM).build(), invalid_argument);
    RaggedTensor wrongShape = p.withValues(Tensor(DataType::FP32, {4,3}));
    EXPECT_THROW((void)Add::Builder().network(n).left(p).right(wrongShape).build(), invalid_argument);
    RaggedTensor wrongDType = p.withValues(Tensor(DataType::BF16, {4,2}));
    EXPECT_THROW((void)Add::Builder().network(n).left(p).right(wrongDType).build(), invalid_argument);
    EXPECT_THROW((void)Add::Builder().network(n).left(p).right(Tensor(DataType::FP32, {4,2})).build(), invalid_argument);

    Network fixedNet("add_fixed_legacy");
    auto fixed = RaggedNetworkInput::Builder().network(fixedNet).name("fixed")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(4).build();
    auto fixedBranches = branchedAdd(fixedNet, fixed);
    json old = fixedBranches.architectureJson();
    ASSERT_EQ(old.at("version"), "1.1.0");
    // Simulate an archived 1.0.0 graph whose values and offsets carried
    // physical dimensions. Its B/total hints are deliberately inconsistent;
    // authoritative semantic M=4 comes from the already-reconstructed graph.
    old["version"] = "1.0.0";
    for (const char* key : {"ragged_left", "ragged_right", "ragged_output"}) {
        auto& ref = old[key];
        ref.erase("geometry_mode");
        ref.erase("layer_ragged_reference_version");
        ref.erase("max_values_per_row");
        ref["batch_size"] = 91;
        ref["max_total_values"] = 987;
        ref["values"]["dimensions"][0] = 987;
        ref["offsets"]["dimensions"] = vector<uint64_t>{92};
    }
    old["feature_output"]["dimensions"][0] = 987;
    ASSERT_NO_THROW(Add::deserialize(old, &fixedNet));
    auto loadedAdd = dynamic_pointer_cast<Add>(fixedNet.getLayer(fixedNet.getNumLayers() - 1));
    ASSERT_NE(loadedAdd, nullptr);
    ASSERT_TRUE(loadedAdd->getRaggedFeatureOutput().has_value());
    const auto recovered = loadedAdd->getRaggedFeatureOutput().value();
    EXPECT_EQ(recovered.getValuesDimensions(), (vector<uint64_t>{4, 2}));
    EXPECT_EQ(recovered.getOffsetsDimensions(), (vector<uint64_t>{1}));
    for (uint64_t b : {2U, 7U}) {
        EXPECT_EQ(recovered.getDescriptorForPlacement(b).getValuesDimensions(),
                  (vector<uint64_t>{4 * b, 2}));
    }
    json forged = old;
    forged["ragged_right"]["values"]["dimensions"][1] = 3;
    EXPECT_ANY_THROW(Add::deserialize(forged, &fixedNet));

    Network dense("add_dense_legacy");
    auto dl = NetworkInput::Builder().network(dense).name("left")
        .dimensions({2}).dataType(DataType::FP32).build();
    auto dr = NetworkInput::Builder().network(dense).name("right")
        .dimensions({2}).dataType(DataType::FP32).build();
    auto denseAdd = Add::Builder().network(dense).left(dl.getFeatureOutput().value())
        .right(dr.getFeatureOutput().value()).build();
    const json denseJson = denseAdd.architectureJson();
    EXPECT_EQ(denseJson.at("version"), "1.0.0");
    EXPECT_NO_THROW(Add::deserialize(denseJson, &dense));
    json wrongVersion = denseJson;
    wrongVersion["version"] = "1.1.0";
    expectReject(dense, wrongVersion);
}

TEST(RaggedAddNeutral1M, OneSavedArchitectureLoadsAtTwoAndSevenAndExecutesActivePrefix) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "CUDA unavailable";
    const filesystem::path archive = filesystem::temp_directory_path() /
        ("thor_add_1m_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    uint64_t leftId = 0, rightId = 0, outputId = 0, partitionId = 0;
    {
        Network n("add_1m_placement");
        auto p = input(n);
        auto a = branchedAdd(n, p);
        const auto ports = a.getFeatureInputs();
        ASSERT_EQ(ports.size(), 3U);
        leftId = ports[0].getOriginalId(); rightId = ports[1].getOriginalId();
        outputId = a.getFeatureOutputs().at(0).getOriginalId();
        partitionId = p.getRowPartitionToken().getOriginalId();
        (void)RaggedNetworkOutput::Builder().network(n).name("sum").inputTensor(a.getRaggedFeatureOutput().value()).build();
        n.save(archive.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(b);
        Network n("add_1m_placement");
        ASSERT_NO_THROW(n.load(archive.string()));
        const auto external = n.getExternalRaggedNetworkInputs();
        ASSERT_EQ(external.size(), 1U);
        const auto p = external.front().raggedTensor;
        auto* a = onlyAdd(n);
        ASSERT_NE(a, nullptr);
        const auto ports = a->getFeatureInputs();
        ASSERT_EQ(ports.size(), 3U);
        EXPECT_EQ(ports[0].getOriginalId(), leftId);
        EXPECT_EQ(ports[1].getOriginalId(), rightId);
        EXPECT_NE(ports[0], ports[1]);
        EXPECT_EQ(ports[2].getOriginalId(), partitionId);
        EXPECT_EQ(a->getFeatureOutputs()[0].getOriginalId(), outputId);
        const json before = n.architectureJson();
        vector<Event> events;
        auto placed = n.place(b, events, true);
        ASSERT_NE(placed, nullptr);
        for (auto& e : events) e.synchronize();
        auto& stamp = placed->getStampedNetwork(0);
        EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(p.getRowPartitionId()), nullptr);
        auto active = stamp.getManagedPartitionActiveCountInputForTest(p.getRowPartitionId());
        ASSERT_NE(active, nullptr);
        ASSERT_TRUE(active->getFeatureOutput().has_value());
        EXPECT_EQ(active->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        auto in = stamp.getNamedInput("tokens.values");
        ASSERT_NE(in, nullptr);
        ASSERT_TRUE(in->getFeatureOutput().has_value());
        EXPECT_EQ(in->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b * 4U, 2}));
        auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(stamp.getPhysicalLayerFromApiLayer(a->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b * 4U, 2}));
        EXPECT_EQ(n.architectureJson(), before);
        const auto outputs = placed->inferLogical(batch(p, b));
        ASSERT_TRUE(outputs.contains("sum"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at("sum")));
        const auto out = get<Impl::RaggedTensor>(outputs.at("sum"));
        auto offsets = out.getHostOffsetsIfAvailable();
        ASSERT_TRUE(offsets.has_value());
        EXPECT_EQ(offsets->at(0), 0U);
        EXPECT_EQ(offsets->at(1), 0U);
        EXPECT_LT(offsets->back(), b * 4U);
        const float* data = out.getValues().getMemPtr<float>();
        for (uint64_t t = 0; t < offsets->back(); ++t)
            for (int c = 0; c < 2; ++c) {
                const float x = 0.3F * static_cast<float>(t) - 0.8F + static_cast<float>(c) * 0.45F;
                EXPECT_NEAR(data[2 * t + c], max(x, 0.0F) + 1.0F / (1.0F + exp(-x)), 4e-3F);
            }
    }
    filesystem::remove_all(archive);
}

// Both values ports must be stampable in either order. The partition token is
// shared and can independently be selected as the physical stamp predecessor.
TEST(RaggedAddNeutral1M, SwappedValuePortOrderStillPhysicalizesNeutralCapacity) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "CUDA unavailable";
    Network n("add_1m_reverse_stamp");
    const RaggedTensor p = input(n);
    Add a = branchedAdd(n, p, true);
    ASSERT_EQ(a.getFeatureInputs().size(), 3U);
    EXPECT_NE(a.getFeatureInputs()[0], a.getFeatureInputs()[1]);
    (void)RaggedNetworkOutput::Builder().network(n).name("sum")
        .inputTensor(a.getRaggedFeatureOutput().value()).build();
    vector<Event> events;
    auto placed = n.place(2, events, true);
    ASSERT_NE(placed, nullptr);
    for (auto& e : events) e.synchronize();
    auto& stamp = placed->getStampedNetwork(0);
    auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
        stamp.getPhysicalLayerFromApiLayer(a.getId()));
    ASSERT_NE(physical, nullptr);
    ASSERT_TRUE(physical->getFeatureOutput().has_value());
    EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{8,2}));
    const auto result = placed->inferLogical(batch(p, 2));
    ASSERT_TRUE(result.contains("sum"));
    const auto& output = get<Impl::RaggedTensor>(result.at("sum"));
    auto offsets = output.getHostOffsetsIfAvailable();
    ASSERT_TRUE(offsets.has_value());
    for (uint64_t t = 0; t < offsets->back(); ++t) {
        for (int c = 0; c < 2; ++c) {
            const float x = 0.3F * static_cast<float>(t) - 0.8F + static_cast<float>(c) * 0.45F;
            EXPECT_NEAR(output.getValues().getMemPtr<float>()[2*t + c],
                        max(x, 0.0F) + 1.0F/(1.0F + exp(-x)), 4e-3F);
        }
    }
}


TEST(RaggedAddNeutral1M, SavedNeutralAddBackpropagatesToBothValuePortsAtTwoAndSeven) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";

    const filesystem::path archive = filesystem::temp_directory_path() /
        ("thor_add_1m_backward_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    uint64_t leftId = 0, rightId = 0, sumId = 0, partitionId = 0;
    {
        Network network("add_1m_backward");
        const RaggedTensor tokens = input(network);
        const RaggedTensor targets = RaggedNetworkInput::Builder().network(network).name("targets")
            .valuesDataType(DataType::FP32).trailingDimensions({2}).partition(tokens).build();
        const Add add = branchedAdd(network, tokens, /*reversePorts=*/false,
                                    /*preserveInputGradients=*/true);
        const auto ports = add.getFeatureInputs();
        ASSERT_EQ(ports.size(), 3u);
        EXPECT_NE(ports[0], ports[1]);
        EXPECT_EQ(ports[2], tokens.getRowPartitionToken());
        ASSERT_TRUE(add.getRaggedFeatureOutput().has_value());
        EXPECT_TRUE(add.getRaggedFeatureOutput()->sharesPartitionWith(targets));
        const auto raw = RaggedCustomLoss::Builder().network(network)
            .predictions(add.getRaggedFeatureOutput().value()).labels(targets)
            .lossExpression(squaredError(false)).gradientExpression(squaredError(true)).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        leftId = ports[0].getOriginalId();
        rightId = ports[1].getOriginalId();
        sumId = add.getFeatureOutputs()[0].getOriginalId();
        partitionId = tokens.getRowPartitionToken().getOriginalId();
        EXPECT_EQ(add.architectureJson().at("version"), "1.1.0");
        EXPECT_EQ(raw.architectureJson().at("version"), "1.1.0");
        network.save(archive.string(), true);
    }

    for (uint32_t b : {2u, 7u}) {
        SCOPED_TRACE(::testing::Message() << "placement B=" << b);
        Network network("add_1m_backward");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto external = network.getExternalRaggedNetworkInputs();
        ASSERT_EQ(external.size(), 2u);
        const auto tokenIt = find_if(external.begin(), external.end(),
            [](const auto& candidate) { return candidate.name == "tokens"; });
        const auto targetIt = find_if(external.begin(), external.end(),
            [](const auto& candidate) { return candidate.name == "targets"; });
        ASSERT_NE(tokenIt, external.end());
        ASSERT_NE(targetIt, external.end());
        const RaggedTensor tokens = tokenIt->raggedTensor;
        const RaggedTensor targets = targetIt->raggedTensor;
        EXPECT_TRUE(tokens.sharesPartitionWith(targets));
        EXPECT_EQ(tokens.getRowPartitionToken().getOriginalId(), partitionId);
        auto* add = onlyAdd(network);
        const auto raw = onlyRawLoss(network);
        ASSERT_NE(add, nullptr);
        ASSERT_NE(raw, nullptr);
        const auto ports = add->getFeatureInputs();
        ASSERT_EQ(ports.size(), 3u);
        EXPECT_EQ(ports[0].getOriginalId(), leftId);
        EXPECT_EQ(ports[1].getOriginalId(), rightId);
        EXPECT_NE(ports[0], ports[1]);
        EXPECT_EQ(ports[2].getOriginalId(), partitionId);
        EXPECT_EQ(add->getFeatureOutputs()[0].getOriginalId(), sumId);
        EXPECT_EQ(raw->getRaggedPredictions().getValues(), add->getFeatureOutputs()[0]);
        EXPECT_EQ(add->architectureJson().at("version"), "1.1.0");
        EXPECT_EQ(raw->architectureJson().at("version"), "1.1.0");
        EXPECT_EQ(add->architectureJson().dump().find("\"batch_size\""), string::npos);
        EXPECT_EQ(add->architectureJson().dump().find("\"max_total_values\""), string::npos);

        const json before = network.architectureJson();
        vector<Event> initialization;
        auto placed = network.place(b, initialization, /*inferenceOnly=*/false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : initialization) event.synchronize();
        auto& stamp = placed->getStampedNetwork(0);
        const uint64_t capacity = static_cast<uint64_t>(b) * 4;
        EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(tokens.getRowPartitionId()), nullptr);
        auto activeCarrier = stamp.getManagedPartitionActiveCountInputForTest(tokens.getRowPartitionId());
        ASSERT_NE(activeCarrier, nullptr);
        ASSERT_TRUE(activeCarrier->getFeatureOutput().has_value());
        EXPECT_EQ(activeCarrier->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        auto physical = dynamic_pointer_cast<Impl::RaggedCustomLayer>(
            stamp.getPhysicalLayerFromApiLayer(add->getId()));
        auto physicalLoss = dynamic_pointer_cast<Impl::RaggedCustomLoss>(
            stamp.getPhysicalLayerFromApiLayer(raw->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_NE(physicalLoss, nullptr);
        EXPECT_EQ(physicalLoss->getLogicalBatchSize(), b);
        EXPECT_EQ(physicalLoss->getMaxTotalValues(), capacity);
        const auto inputs = physical->getFeatureInputs();
        ASSERT_EQ(inputs.size(), 3u);
        ASSERT_TRUE(inputs[0].has_value());
        ASSERT_TRUE(inputs[1].has_value());
        ASSERT_TRUE(inputs[2].has_value());
        EXPECT_NE(inputs[0].value(), inputs[1].value());
        EXPECT_EQ(inputs[0]->getDimensions(), (vector<uint64_t>{capacity, 2}));
        EXPECT_EQ(inputs[1]->getDimensions(), (vector<uint64_t>{capacity, 2}));
        EXPECT_EQ(inputs[2]->getDimensions(), (vector<uint64_t>{1}));
        const auto sums = physical->getFeatureOutputs();
        ASSERT_EQ(sums.size(), 1u);
        ASSERT_TRUE(sums[0].has_value());
        EXPECT_EQ(sums[0]->getDimensions(), (vector<uint64_t>{capacity, 2}));
        EXPECT_EQ(network.architectureJson(), before);

        vector<uint64_t> offsets(b + 1, 0);
        for (uint32_t row = 0; row < b; ++row)
            offsets[row + 1] = offsets[row] + (row == 0 || row % 3 == 0 ? 0 : (row % 2 == 0 ? 1 : 2));
        const uint64_t active = offsets.back();
        ASSERT_GT(active, 0u);
        ASSERT_LT(active, capacity);
        const auto desc = tokens.getDescriptorForPlacement(b);
        const auto targetDesc = targets.getDescriptorForPlacement(b);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor values(cpu, desc.getValuesDescriptor());
        Impl::Tensor labels(cpu, targetDesc.getValuesDescriptor());
        Impl::Tensor ends(cpu, desc.getOffsetsDescriptor());
        for (uint64_t token = 0; token < capacity; ++token) {
            for (uint32_t channel = 0; channel < 2; ++channel) {
                const uint64_t i = 2 * token + channel;
                values.getMemPtr<float>()[i] = token < active ? tokenValue(token, channel)
                    : numeric_limits<float>::quiet_NaN();
                labels.getMemPtr<float>()[i] = token < active
                    ? expectedSum(token, channel) - expectedDifference(token, channel)
                    : numeric_limits<float>::quiet_NaN();
            }
        }
        for (uint32_t row = 0; row <= b; ++row)
            ends.getMemPtr<uint32_t>()[row] = static_cast<uint32_t>(offsets[row]);
        Impl::RowPartitionRuntime partition(ends, desc.getRowPartition());
        partition.setHostOffsets(offsets);
        Batch trainingBatch;
        trainingBatch.insert("tokens", Impl::RaggedTensor(values, partition));
        trainingBatch.insert("targets", labels);
        trainingBatch.setValidExampleCount(b);
        map<string, Impl::Tensor> results;
        map<string, Event> ready;
        auto done = placed->submitBatch(0, trainingBatch, results, ready, false);
        done.synchronize();
        ASSERT_TRUE(ready.contains("raw_loss"));
        ready.at("raw_loss").synchronize();
        placed->synchronize();
        ASSERT_TRUE(results.contains("raw_loss"));
        EXPECT_EQ(results.at("raw_loss").getDimensions(), (vector<uint64_t>{capacity, 2}));

        const auto gradients = physical->getErrorOutputs();
        ASSERT_EQ(gradients.size(), 3u) << "Add has two values ports and one structural carrier";
        ASSERT_TRUE(gradients[0].has_value()) << "missing left values gradient";
        ASSERT_TRUE(gradients[1].has_value()) << "missing right values gradient";
        EXPECT_EQ(gradients[0]->getDataType(), DataType::FP32);
        EXPECT_EQ(gradients[1]->getDataType(), DataType::FP32);
        EXPECT_EQ(gradients[0]->getDimensions(), (vector<uint64_t>{capacity, 2}));
        EXPECT_EQ(gradients[1]->getDimensions(), (vector<uint64_t>{capacity, 2}));
        ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
        const vector<float> output = downloadActiveElements(sums[0].value(), active * 2);
        const vector<float> losses = downloadActiveElements(results.at("raw_loss"), active * 2);
        const vector<float> upstream = downloadActiveElements(physicalLoss->getErrorOutput().value(), active * 2);
        const vector<float> left = downloadActiveElements(gradients[0].value(), active * 2);
        const vector<float> right = downloadActiveElements(gradients[1].value(), active * 2);
        bool sawPositive = false, sawNegative = false, sawDifferentMagnitude = false;
        for (uint64_t token = 0; token < active; ++token) {
            for (uint32_t channel = 0; channel < 2; ++channel) {
                const uint64_t i = token * 2 + channel;
                const float delta = expectedDifference(token, channel);
                const float expectedGradient = 2.0f * delta * Impl::Loss::getLossScalingFactor();
                sawPositive |= expectedGradient > 0.0f;
                sawNegative |= expectedGradient < 0.0f;
                sawDifferentMagnitude |= abs(delta) > 0.25f;
                EXPECT_NEAR(output[i], expectedSum(token, channel), 2e-4f) << "Add forward " << i;
                EXPECT_NEAR(losses[i], delta * delta, 2e-4f) << "RAW loss " << i;
                EXPECT_NEAR(upstream[i], expectedGradient, 5e-3f) << "RAW dL/dsum " << i;
                EXPECT_NEAR(left[i], expectedGradient, 6e-3f) << "left Add gradient " << i;
                EXPECT_NEAR(right[i], expectedGradient, 6e-3f) << "right Add gradient " << i;
            }
        }
        EXPECT_TRUE(sawPositive);
        EXPECT_TRUE(sawNegative);
        EXPECT_TRUE(sawDifferentMagnitude);
        EXPECT_EQ(network.architectureJson(), before);
        // Inactive storage is unspecified: never read its gradients.
    }
    filesystem::remove_all(archive);
}
