#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <memory>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {

Impl::DynamicExpression squaredDifference(bool gradient) {
    Impl::Expression p = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    Impl::Expression y = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    Impl::Expression d = p - y;
    Impl::Expression value = gradient
        ? (d * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()))
        : (d * d);
    value = value.withOutputDType(DataType::FP32);
    auto definition = Impl::ExpressionDefinition::fromOutputs(
        Impl::Expression::outputs({{gradient ? "predictions_grad" : "loss", value}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

RaggedTensor makeInput(Network& net, const string& name, const optional<RaggedTensor>& partition = nullopt,
                       uint64_t m = 4, vector<uint64_t> trailing = {1}, bool fixed = false) {
    RaggedNetworkInput::Builder builder;
    builder.network(net).name(name).valuesDataType(DataType::FP32).trailingDimensions(trailing);
    if (partition) builder.partition(partition.value());
    else {
        builder.offsetsDataType(DataType::UINT32).maxValuesPerRow(m);
        if (fixed) builder.batchSize(2).maxTotalValues(8);
    }
    return builder.build();
}

RaggedCustomLoss makeRawLoss(Network& net, RaggedTensor p, RaggedTensor labels,
                             optional<RaggedTensor> secondary = nullopt,
                             optional<RaggedTensor> weights = nullopt) {
    RaggedCustomLoss::Builder builder;
    builder.network(net).predictions(p).labels(labels)
        .lossExpression(squaredDifference(false)).gradientExpression(squaredDifference(true));
    if (secondary) builder.secondaryInput(secondary.value(), "secondary", "secondary_grad");
    if (weights) builder.exampleWeights(weights.value());
    return builder.build();
}

shared_ptr<RaggedCustomLoss> findRaw(Network& network) {
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto layer = dynamic_pointer_cast<RaggedCustomLoss>(network.getLayer(i));
        if (layer) return layer;
    }
    return nullptr;
}

Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);

vector<float> download(const Impl::Tensor& source) {
    Impl::Tensor host = source.clone(cpu);
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    host.copyFromAsync(source, stream);
    stream.synchronize();
    const float* values = host.getMemPtr<float>();
    return vector<float>(values, values + host.getTotalNumElements());
}

void sync(vector<Event>& events) {
    for (Event& e : events) e.synchronize();
}

void runRawTerminalLoss(Network& network, const RaggedTensor& p, uint32_t b) {
    vector<Event> init;
    shared_ptr<PlacedNetwork> placed = network.place(b, init, false);
    ASSERT_NE(placed, nullptr);
    sync(init);
    auto& stamp = placed->getStampedNetwork(0);
    const uint64_t capacity = static_cast<uint64_t>(b) * p.getMaxValuesPerRow();
    auto implLoss = dynamic_pointer_cast<Impl::RaggedCustomLoss>(
        stamp.getPhysicalLayerFromApiLayer(findRaw(network)->getId()));
    ASSERT_NE(implLoss, nullptr);
    EXPECT_EQ(implLoss->getLogicalBatchSize(), b);
    EXPECT_EQ(implLoss->getMaxTotalValues(), capacity);
    EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(p.getRowPartitionId()), nullptr);
    auto active = stamp.getManagedPartitionActiveCountInputForTest(p.getRowPartitionId());
    ASSERT_NE(active, nullptr);
    ASSERT_TRUE(active->getFeatureOutput().has_value());
    EXPECT_EQ(active->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
    ASSERT_TRUE(implLoss->getOffsetsInput().has_value());
    EXPECT_EQ(implLoss->getOffsetsInput()->getDimensions(), (vector<uint64_t>{1}));
    auto predictionInput = stamp.getNamedInput("predictions.values");
    ASSERT_NE(predictionInput, nullptr);
    ASSERT_TRUE(predictionInput->getFeatureOutput().has_value());
    EXPECT_EQ(predictionInput->getFeatureOutput()->getDimensions(), (vector<uint64_t>{capacity, 1}));

    vector<uint64_t> offsets(b + 1, 0);
    for (uint32_t row = 1; row <= b; ++row)
        offsets[row] = offsets[row - 1] + (row == 1 || row % 3 == 0 ? 0 : (row % 2 == 0 ? 2 : 1));
    const uint64_t activeCount = offsets.back();
    ASSERT_LT(activeCount, capacity);

    Impl::RaggedTensorDescriptor desc = p.getDescriptorForPlacement(b);
    Impl::Tensor preds(cpu, desc.getValuesDescriptor());
    Impl::Tensor targets(cpu, desc.getValuesDescriptor());
    Impl::Tensor off(cpu, desc.getOffsetsDescriptor());
    for (uint64_t i = 0; i < capacity; ++i) {
        preds.getMemPtr<float>()[i] = i < activeCount ? (0.25f + i * 0.3f) : numeric_limits<float>::quiet_NaN();
        targets.getMemPtr<float>()[i] = i < activeCount ? (-0.1f + i * 0.07f) : numeric_limits<float>::quiet_NaN();
    }
    for (uint32_t i = 0; i <= b; ++i) off.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(offsets[i]);
    Impl::RowPartitionRuntime partition(off, desc.getRowPartition());
    partition.setHostOffsets(offsets);

    Batch batch;
    batch.insert("predictions", Impl::RaggedTensor(preds, partition));
    batch.insert("labels", targets);
    batch.setValidExampleCount(b);
    map<string, Impl::Tensor> results;
    map<string, Event> outputReady;
    Event done = placed->submitBatch(0, batch, results, outputReady, false);
    done.synchronize();
    outputReady.at("raw_loss").synchronize();
    placed->synchronize();

    ASSERT_TRUE(results.contains("raw_loss"));
    EXPECT_EQ(results.at("raw_loss").getDimensions(), (vector<uint64_t>{capacity, 1}));
    const vector<float> rawValues = download(results.at("raw_loss"));
    ASSERT_TRUE(implLoss->getErrorOutput().has_value());
    const vector<float> gradients = download(implLoss->getErrorOutput().value());
    ASSERT_EQ(gradients.size(), capacity);
    for (uint64_t i = 0; i < activeCount; ++i) {
        const float d = preds.getMemPtr<float>()[i] - targets.getMemPtr<float>()[i];
        EXPECT_NEAR(rawValues[i], d * d, 5e-4f) << "active value " << i;
        EXPECT_NEAR(gradients[i], 2.0f * d * Impl::Loss::getLossScalingFactor(), 5e-3f)
            << "active gradient " << i;
    }
}

}  // namespace

TEST(RaggedCustomLossNeutralL1A, SemanticReferencesRejectForgedGeometryAndPreserveLegacy) {
    Network network("raw_neutral_json");
    RaggedTensor p = makeInput(network, "predictions");
    RaggedTensor labels = makeInput(network, "labels", p);
    RaggedTensor secondary = makeInput(network, "secondary", p);
    RaggedTensor weights = makeInput(network, "weights", p);
    auto loss = makeRawLoss(network, p, labels, secondary, weights);
    const json j = loss.architectureJson();
    EXPECT_EQ(j.at("version").get<string>(), "1.1.0");
    for (const char* key : {"ragged_predictions", "ragged_labels", "ragged_raw_loss",
                            "ragged_example_weights", "ragged_secondary_input"}) {
        EXPECT_FALSE(j.at(key).contains("batch_size"));
        EXPECT_FALSE(j.at(key).contains("max_total_values"));
        EXPECT_EQ(j.at(key).at("max_values_per_row").get<uint64_t>(), 4U);
        EXPECT_EQ(j.at(key).at("offsets").at("id"), j.at("ragged_predictions").at("offsets").at("id"));
    }
    EXPECT_EQ(loss.getRaggedRawLoss().getValuesDimensions(), (vector<uint64_t>{4, 1}));
    EXPECT_EQ(loss.getOutputTensorBytes(2), 8U * sizeof(float));
    EXPECT_EQ(loss.getOutputTensorBytes(7), 28U * sizeof(float));
    // RAW output and primary/secondary gradient buffers each scale once by B;
    // weights and managed active-count are not allocated again by the loss.
    EXPECT_EQ(loss.getFirstInstanceMemRequirementInBytes(2, cpu), 3U * 8U * sizeof(float));
    EXPECT_EQ(loss.getFirstInstanceMemRequirementInBytes(7, cpu), 3U * 28U * sizeof(float));

    auto rejects = [&](json changed) { EXPECT_ANY_THROW(RaggedCustomLoss::deserialize(changed, &network)); };
    json bad = j; bad["version"] = "1.0.0"; rejects(bad);
    bad = j; bad["ragged_labels"]["max_values_per_row"] = 5; rejects(bad);
    bad = j; bad["ragged_labels"]["offsets"]["id"] = p.getValues().getId(); rejects(bad);
    bad = j; bad["ragged_predictions"]["values"]["dimensions"] = vector<uint64_t>{4, 2}; rejects(bad);
    bad = j; bad["ragged_example_weights"]["values"]["id"] = p.getValues().getId(); rejects(bad);
    bad = j; bad["ragged_secondary_inputs"][0]["tensor"]["values"]["id"] = p.getValues().getId(); rejects(bad);
    bad = j; bad["ragged_raw_loss"]["values"]["dimensions"] = vector<uint64_t>{8, 1}; rejects(bad);
    bad = j; bad["ragged_raw_loss"]["offsets"]["id"] = p.getValues().getId(); rejects(bad);
    bad = j; bad["ragged_raw_loss"]["values"]["id"] = p.getValues().getId(); rejects(bad);
    bad = j; bad["ragged_labels"]["geometry_mode"] = "fixed"; rejects(bad);
    bad = j; bad["ragged_predictions"]["batch_size"] = 7; rejects(bad);

    Network other("raw_neutral_rejections");
    RaggedTensor q = makeInput(other, "predictions");
    RaggedTensor l = makeInput(other, "labels", q);
    RaggedTensor different = makeInput(other, "other");
    EXPECT_THROW((void)makeRawLoss(other, q, different), invalid_argument);
    RaggedTensor fixed = makeInput(other, "fixed", nullopt, 4, {1}, true);
    EXPECT_THROW((void)makeRawLoss(other, q, fixed), invalid_argument);
    RaggedTensor wrongM = RaggedTensor::placementNeutral(
        Tensor(DataType::FP32, {5, 1}), q.getRowPartitionToken(), 5);
    EXPECT_THROW((void)makeRawLoss(other, q, wrongM), invalid_argument);
    RaggedTensor wrongTrailing = makeInput(other, "wrong_trailing", q, 4, {2});
    EXPECT_THROW((void)makeRawLoss(other, q, wrongTrailing), invalid_argument);
    RaggedTensor wrongWeights = makeInput(other, "bad_weights", q, 4, {2});
    EXPECT_THROW((void)makeRawLoss(other, q, l, nullopt, wrongWeights), invalid_argument);
    EXPECT_THROW((void)makeRawLoss(other, q, l, different), invalid_argument);

    Network legacy("raw_fixed_legacy");
    RaggedTensor fp = makeInput(legacy, "predictions", nullopt, 4, {1}, true);
    RaggedTensor fl = makeInput(legacy, "labels", fp);
    auto fixedLoss = makeRawLoss(legacy, fp, fl);
    json old = fixedLoss.architectureJson();
    EXPECT_EQ(old.at("version").get<string>(), "1.0.0");
    EXPECT_EQ(old.at("ragged_predictions").at("batch_size").get<uint64_t>(), 2U);
    EXPECT_NO_THROW(RaggedCustomLoss::deserialize(old, &legacy));
}

TEST(RaggedCustomLossNeutralL1A, SameSavedRawTerminalLossTrainsAtTwoAndSeven) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    const string name = "raw_l1a_terminal";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() / ("thor_l1a_" + to_string(nonce));
    filesystem::remove_all(archive);
    {
        Network net(name);
        RaggedTensor p = makeInput(net, "predictions");
        RaggedTensor labels = makeInput(net, "labels", p);
        auto loss = makeRawLoss(net, p, labels);
        NetworkOutput::Builder().network(net).name("raw_loss")
            .inputTensor(loss.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        net.save(archive.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        Network loaded(name);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        auto inputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(), 2U);
        auto primaryIt = find_if(inputs.begin(), inputs.end(), [](const auto& ref) {
            return ref.name == "predictions";
        });
        ASSERT_NE(primaryIt, inputs.end());
        const RaggedTensor p = primaryIt->raggedTensor;
        ASSERT_TRUE(p.isPlacementNeutral());
        auto raw = findRaw(loaded);
        ASSERT_NE(raw, nullptr);
        EXPECT_EQ(raw->architectureJson().at("version").get<string>(), "1.1.0");
        EXPECT_EQ(raw->getRaggedRawLoss().getValuesDimensions(), (vector<uint64_t>{4, 1}));
        const json before = loaded.architectureJson();
        runRawTerminalLoss(loaded, p, b);
        EXPECT_EQ(loaded.architectureJson(), before);
    }
    filesystem::remove_all(archive);
}
