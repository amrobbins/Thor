#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedLossShaper.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <memory>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
RaggedTensor rawInput1T(Network& n) {
    RaggedNetworkInput::Builder b;
    b.network(n).name("loss_values").valuesDataType(DataType::FP32)
        .offsetsDataType(DataType::UINT32).trailingDimensions({2}).maxValuesPerRow(4);
    return b.build();
}
RaggedLossShaper shaper1T(Network& n, const RaggedTensor& input, bool batch) {
    RaggedLossShaper::Builder builder;
    builder.network(n).lossInput(input);
    if (batch) builder.reportsBatchLoss();
    else builder.reportsPerExampleLoss();
    return builder.build();
}
class Stamp1T : public RaggedLossShaper {
public:
    explicit Stamp1T(const RaggedLossShaper& layer) : RaggedLossShaper(layer) {}
    using RaggedLossShaper::stamp;
};
Impl::TensorPlacement cpu1T(Impl::TensorPlacement::MemDevices::CPU);
vector<float> download1T(const Impl::Tensor& device) {
    Impl::Tensor host = device.clone(cpu1T);
    Stream stream = Stream::getNextDownloadStream(device.getPlacement().getDeviceNum());
    host.copyFromAsync(device, stream);
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + host.getTotalNumElements());
}
void reject1T(Network& network, const json& record) {
    EXPECT_ANY_THROW(RaggedLossShaper::deserialize(record, &network));
}

void savedReporting1T(bool batchOutput) {
    const string name = batchOutput ? "loss_shaper_1t_batch" : "loss_shaper_1t_per_example";
    const auto archive = filesystem::temp_directory_path() /
        (name + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    uint64_t inputTokenId = 0, outputId = 0;
    {
        Network n(name);
        auto input = rawInput1T(n);
        auto shaper = shaper1T(n, input, batchOutput);
        inputTokenId = input.getRowPartitionToken().getOriginalId();
        outputId = shaper.getLossOutput().getOriginalId();
        (void)NetworkOutput::Builder().network(n).name("report")
            .inputTensor(shaper.getLossOutput()).dataType(DataType::FP32).build();
        n.save(archive.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b << " batch=" << batchOutput);
        Network n(name);
        ASSERT_NO_THROW(n.load(archive.string()));
        auto external = n.getExternalRaggedNetworkInputs();
        ASSERT_EQ(external.size(), 1U);
        auto source = external.front().raggedTensor;
        EXPECT_EQ(source.getRowPartitionToken().getOriginalId(), inputTokenId);
        shared_ptr<RaggedLossShaper> layer;
        for (uint32_t i = 0; i < n.getNumLayers(); ++i)
            if (auto candidate = dynamic_pointer_cast<RaggedLossShaper>(n.getLayer(i))) layer = candidate;
        ASSERT_NE(layer, nullptr);
        EXPECT_EQ(layer->getLossOutput().getOriginalId(), outputId);
        EXPECT_EQ(layer->architectureJson().at("version"), "1.1.0");
        const json before = n.architectureJson();
        vector<Event> init;
        auto placed = n.place(b, init, true);
        ASSERT_NE(placed, nullptr);
        for (auto& e : init) e.synchronize();
        auto& stamp = placed->getStampedNetwork(0);
        auto physical = dynamic_pointer_cast<Impl::RaggedLossShaper>(
            stamp.getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical, nullptr);
        EXPECT_EQ(physical->getLogicalBatchSize(), b);
        EXPECT_EQ(physical->getMaxTotalValues(), 4ULL * b);
        ASSERT_TRUE(physical->getOffsetsInput().has_value());
        EXPECT_EQ(physical->getOffsetsInput()->getDimensions(), (vector<uint64_t>{b + 1U}));
        EXPECT_NE(stamp.getManagedPartitionOffsetsInputForTest(source.getRowPartitionId()), nullptr);
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{batchOutput ? 1ULL : static_cast<uint64_t>(b), 1}));
        EXPECT_EQ(n.architectureJson(), before);

        const auto desc = source.getDescriptorForPlacement(b);
        Impl::Tensor values(cpu1T, desc.getValuesDescriptor());
        Impl::Tensor offsets(cpu1T, desc.getOffsetsDescriptor());
        vector<uint64_t> partition(b + 1, 0);
        vector<float> rowSums(b, 0.0f);
        for (uint32_t row = 0; row < b; ++row) {
            const uint32_t width = row % 5 == 0 ? 0U : row % 5 == 1 ? 2U :
                                   row % 5 == 2 ? 4U : row % 5 == 3 ? 1U : 3U;
            partition[row + 1] = partition[row] + width;
        }
        for (uint32_t row = 0; row <= b; ++row)
            offsets.getMemPtr<uint32_t>()[row] = static_cast<uint32_t>(partition[row]);
        for (uint64_t t = 0; t < 4ULL * b; ++t)
            for (uint32_t f = 0; f < 2; ++f)
                values.getMemPtr<float>()[t * 2 + f] = t < partition.back()
                    ? (0.25f + 0.13f * static_cast<float>(t) - 0.08f * f)
                    : numeric_limits<float>::quiet_NaN();
        for (uint32_t row = 0; row < b; ++row)
            for (uint64_t t = partition[row]; t < partition[row + 1]; ++t)
                for (uint32_t f = 0; f < 2; ++f)
                    rowSums[row] += values.getMemPtr<float>()[t * 2 + f];
        Impl::RowPartitionRuntime runtime(offsets, desc.getRowPartition());
        runtime.setHostOffsets(partition);
        Batch batch;
        batch.insert("loss_values", Impl::RaggedTensor(values, runtime));
        batch.setValidExampleCount(b);
        map<string, Impl::Tensor> results;
        map<string, Event> ready;
        auto done = placed->submitBatch(0, batch, results, ready, false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(results.contains("report"));
        const vector<float> reported = download1T(results.at("report"));
        const vector<float> expected = batchOutput
            ? vector<float>{static_cast<float>(accumulate(rowSums.begin(), rowSums.end(), 0.0f) / b)}
            : rowSums;
        ASSERT_EQ(reported.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(reported[i], expected[i], 3e-3f) << "row " << i;

        // A second submission on the same placed model must handle a genuinely
        // all-empty partition, without reading NaNs from inactive capacity.
        vector<uint64_t> empty(b + 1, 0);
        for (uint32_t row = 0; row <= b; ++row) offsets.getMemPtr<uint32_t>()[row] = 0;
        for (uint64_t t = 0; t < 8ULL * b; ++t)
            values.getMemPtr<float>()[t] = numeric_limits<float>::quiet_NaN();
        Impl::RowPartitionRuntime emptyRuntime(offsets, desc.getRowPartition());
        emptyRuntime.setHostOffsets(empty);
        Batch emptyBatch;
        emptyBatch.insert("loss_values", Impl::RaggedTensor(values, emptyRuntime));
        emptyBatch.setValidExampleCount(b);
        results.clear();
        ready.clear();
        done = placed->submitBatch(0, emptyBatch, results, ready, false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(results.contains("report"));
        const auto zeros = download1T(results.at("report"));
        ASSERT_EQ(zeros.size(), expected.size());
        for (float value : zeros) EXPECT_EQ(value, 0.0f);
    }
    filesystem::remove_all(archive);
}
}  // namespace

TEST(RaggedLossShaperNeutral1T, SemanticReferencesModesAndMemory) {
    Network n("loss_shaper_1t_semantics");
    const auto input = rawInput1T(n);
    const auto batch = shaper1T(n, input, true);
    const auto perExample = shaper1T(n, input, false);
    for (const auto& layer : {batch, perExample}) {
        EXPECT_EQ(layer.getRaggedPartitionRequirementForInput(input.getOffsets()),
                  Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);
        const json record = layer.architectureJson();
        EXPECT_EQ(record.at("version"), "1.1.0");
        EXPECT_EQ(record.at("ragged_loss_input").at("max_values_per_row"), 4);
        EXPECT_EQ(record.at("ragged_loss_input").at("offsets").at("dimensions"), (vector<uint64_t>{1}));
        EXPECT_FALSE(record.at("ragged_loss_input").contains("batch_size"));
        EXPECT_FALSE(record.at("ragged_loss_input").contains("max_total_values"));
        EXPECT_NO_THROW(RaggedLossShaper::deserialize(record, &n));
    }
    for (uint32_t b : {2U, 7U}) {
        EXPECT_EQ(perExample.getOutputTensorBytes(b), b * sizeof(float));
        EXPECT_EQ(perExample.getFirstInstanceMemRequirementInBytes(b, cpu1T), b * sizeof(float));
        EXPECT_EQ(batch.getOutputTensorBytes(b), sizeof(float));
    }
    EXPECT_THROW((void)perExample.getOutputTensorBytes(0), exception);
    EXPECT_THROW((void)batch.getOutputTensorBytes(0), exception);
    Network old("loss_shaper_1t_legacy");
    const auto fixed = rawInput1T(old);
    const auto legacy = shaper1T(old, fixed, false);
    EXPECT_EQ(legacy.architectureJson().at("version"), "1.1.0");
    EXPECT_NO_THROW(RaggedLossShaper::deserialize(legacy.architectureJson(), &old));
    EXPECT_EQ(legacy.getOutputTensorBytes(7), 7U * sizeof(float));
}

TEST(RaggedLossShaperNeutral1T, StrictReferencesAndModes) {
    Network n("loss_shaper_1t_corrupt");
    const auto input = rawInput1T(n);
    const auto layer = shaper1T(n, input, false);
    const json j = layer.architectureJson();
    json bad = j;
    bad["version"] = "1.0.0";
    EXPECT_NO_THROW(RaggedLossShaper::deserialize(bad, &n)); // Recognized historical input grammar.
    bad = j; bad["version"] = "7.0.0"; reject1T(n, bad);
    bad = j; bad["batch_size"] = 7; reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["max_total_values"] = 28; reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["max_values_per_row"] = 5; reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["values"]["id"] = input.getOffsets().getOriginalId(); reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["values"]["data_type"] = DataType::FP16; reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["values"]["dimensions"] = vector<uint64_t>{5, 2}; reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["offsets"]["dimensions"] = vector<uint64_t>{3}; reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["geometry_mode"] = "fixed"; reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["offsets"]["data_type"] = DataType::UINT64; reject1T(n, bad);
    bad = j; bad["ragged_loss_input"]["extra"] = 42; reject1T(n, bad);
    bad = j; bad["loss_output"]["dimensions"] = vector<uint64_t>{2}; reject1T(n, bad);
    bad = j; bad["loss_output"]["id"] = input.getValues().getOriginalId(); reject1T(n, bad);
    bad = j; bad["loss_output"]["data_type"] = DataType::FP16; reject1T(n, bad);
    bad = j; bad["loss_output"]["batch_size"] = 7; reject1T(n, bad);
    bad = j; bad["loss_shape"] = Loss::LossShape::RAW; reject1T(n, bad);
    bad = j; bad["factory"] = "invalid"; reject1T(n, bad);
}

TEST(RaggedLossShaperNeutral1T, HistoricalPackedInputUsesGraphMAndRuntimeBatch) {
    Network network("loss_shaper_1t_historical_grammar");
    const auto input = rawInput1T(network);
    const auto shaper = shaper1T(network, input, false);
    json historical = shaper.architectureJson();
    historical["version"] = "1.0.0";
    auto& reference = historical["ragged_loss_input"];
    reference.erase("layer_ragged_reference_version");
    reference.erase("geometry_mode");
    reference.erase("max_values_per_row");
    reference["batch_size"] = 41;
    reference["max_total_values"] = 999;
    reference["values"]["dimensions"] = vector<uint64_t>{999, 2};
    reference["offsets"]["dimensions"] = vector<uint64_t>{42};

    ASSERT_NO_THROW(RaggedLossShaper::deserialize(historical, &network));
    const auto loaded = dynamic_pointer_cast<RaggedLossShaper>(network.getLayer(network.getNumLayers() - 1));
    ASSERT_NE(loaded, nullptr);
    EXPECT_TRUE(loaded->getRaggedLossInput().sharesPartitionWith(input));
    EXPECT_EQ(loaded->getRaggedLossInput().getValuesDimensions(), (vector<uint64_t>{4, 2}));
    for (uint32_t batch : {2U, 7U}) {
        EXPECT_EQ(loaded->getRaggedLossInput().getDescriptorForPlacement(batch).getValuesDimensions(),
                  (vector<uint64_t>{batch * 4U, 2}));
        EXPECT_EQ(loaded->getOutputTensorBytes(batch), batch * sizeof(float));
    }
    json bad = historical;
    bad["ragged_loss_input"]["max_values_per_row"] = 5;
    EXPECT_ANY_THROW(RaggedLossShaper::deserialize(bad, &network));
    bad = historical;
    bad["ragged_loss_input"]["values"]["dimensions"] = vector<uint64_t>{999, 3};
    EXPECT_ANY_THROW(RaggedLossShaper::deserialize(bad, &network));
    bad = historical;
    bad["loss_output"]["id"] = input.getValues().getOriginalId();
    EXPECT_ANY_THROW(RaggedLossShaper::deserialize(bad, &network));
}

TEST(RaggedLossShaperNeutral1T, ValuesAndOffsetsMayDriveStamp) {
    Network n("loss_shaper_1t_stamp");
    auto input = rawInput1T(n);
    Stamp1T probe(shaper1T(n, input, false));
    for (uint32_t b : {2U, 7U}) {
        auto desc = input.getDescriptorForPlacement(b);
        auto valuesDriver = make_shared<Impl::NetworkInput>(cpu1T, DataType::FP32,
                                                             desc.getValuesDescriptor().getDimensions());
        auto offsetsDriver = make_shared<Impl::NetworkInput>(cpu1T, DataType::UINT32,
                                                              desc.getOffsetsDescriptor().getDimensions());
        auto fromValues = dynamic_pointer_cast<Impl::RaggedLossShaper>(
            probe.stamp(cpu1T, valuesDriver, nullptr, input.getValues(), true));
        auto fromOffsets = dynamic_pointer_cast<Impl::RaggedLossShaper>(
            probe.stamp(cpu1T, offsetsDriver, nullptr, input.getOffsets(), true));
        ASSERT_NE(fromValues, nullptr);
        ASSERT_NE(fromOffsets, nullptr);
        EXPECT_EQ(fromValues->getLogicalBatchSize(), b);
        EXPECT_EQ(fromOffsets->getMaxTotalValues(), 4U * b);
        auto badOffsets = make_shared<Impl::NetworkInput>(cpu1T, DataType::UINT32,
                                                           vector<uint64_t>{1});
        EXPECT_ANY_THROW((void)probe.stamp(cpu1T, badOffsets, nullptr, input.getOffsets(), true));
        auto badValues = make_shared<Impl::NetworkInput>(cpu1T, DataType::FP32,
                                                          vector<uint64_t>{b * 4U, 3});
        EXPECT_ANY_THROW((void)probe.stamp(cpu1T, badValues, nullptr, input.getValues(), true));
    }
}

TEST(RaggedLossShaperNeutral1T, SavedBatchReportingAtTwoAndSeven) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    savedReporting1T(true);
}
TEST(RaggedLossShaperNeutral1T, SavedPerExampleReportingAtTwoAndSeven) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    savedReporting1T(false);
}

TEST(RaggedLossShaperNeutral1T, ReportingBranchDoesNotPruneRawTerminalGradient) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA unavailable";
    const string name = "loss_shaper_1t_raw_training";
    const auto archive = filesystem::temp_directory_path() /
        (name + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    auto lossExpression = [](bool gradient) {
        auto p = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
        auto l = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
        auto d = p - l;
        auto result = gradient
            ? d * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
            : d * d;
        return Impl::DynamicExpression::fromExpressionDefinition(
            Impl::ExpressionDefinition::fromOutputs(
                Impl::Expression::outputs({{gradient ? "predictions_grad" : "loss",
                                            result.withOutputDType(DataType::FP32)}})));
    };
    {
        Network n(name);
        auto predictions = rawInput1T(n);
        auto labels = RaggedNetworkInput::Builder().network(n).name("labels")
            .partition(predictions).valuesDataType(DataType::FP32).trailingDimensions({2}).build();
        auto raw = RaggedCustomLoss::Builder().network(n).predictions(predictions).labels(labels)
            .lossExpression(lossExpression(false)).gradientExpression(lossExpression(true)).build();
        auto report = shaper1T(n, raw.getRaggedRawLoss(), false);
        (void)NetworkOutput::Builder().network(n).name("raw_loss")
            .inputTensor(raw.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        (void)NetworkOutput::Builder().network(n).name("report")
            .inputTensor(report.getLossOutput()).dataType(DataType::FP32).build();
        n.save(archive.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network n(name);
        ASSERT_NO_THROW(n.load(archive.string()));
        RaggedTensor predictions;
        RaggedTensor labels;
        for (const auto& entry : n.getExternalRaggedNetworkInputs()) {
            if (entry.name == "loss_values") predictions = entry.raggedTensor;
            else if (entry.name == "labels") labels = entry.raggedTensor;
        }
        ASSERT_TRUE(predictions.isInitialized());
        ASSERT_TRUE(labels.isInitialized());
        shared_ptr<RaggedCustomLoss> raw;
        for (uint32_t i = 0; i < n.getNumLayers(); ++i)
            if (auto candidate = dynamic_pointer_cast<RaggedCustomLoss>(n.getLayer(i))) raw = candidate;
        ASSERT_NE(raw, nullptr);
        vector<Event> init;
        auto placed = n.place(b, init, false);
        ASSERT_NE(placed, nullptr);
        for (auto& e : init) e.synchronize();
        auto& stamp = placed->getStampedNetwork(0);
        auto physical = dynamic_pointer_cast<Impl::RaggedCustomLoss>(
            stamp.getPhysicalLayerFromApiLayer(raw->getId()));
        ASSERT_NE(physical, nullptr);
        // The RAW loss has two forward consumers, so its physical output
        // must route through TensorFanout. Observe this through the public
        // physical layer API, not StampedNetwork's protected driver maps.
        const auto nextPhysical = physical->getNextLayer();
        ASSERT_TRUE(nextPhysical.has_value());
        EXPECT_NE(dynamic_cast<Impl::TensorFanout*>(nextPhysical.value()), nullptr);
        // Semantic loss ownership must survive that routing and remain active.
        EXPECT_TRUE(physical->isTrainingActive());
        const auto desc = predictions.getDescriptorForPlacement(b);
        Impl::Tensor packed(cpu1T, desc.getValuesDescriptor());
        Impl::Tensor targets(cpu1T, desc.getValuesDescriptor());
        Impl::Tensor offsets(cpu1T, desc.getOffsetsDescriptor());
        vector<uint64_t> rows(b + 1, 0);
        for (uint32_t row = 0; row < b; ++row)
            rows[row + 1] = rows[row] + (row % 3 == 0 ? 0U : 2U);
        for (uint32_t row = 0; row <= b; ++row)
            offsets.getMemPtr<uint32_t>()[row] = static_cast<uint32_t>(rows[row]);
        for (uint64_t t = 0; t < 4ULL * b; ++t)
            for (uint32_t f = 0; f < 2; ++f) {
                packed.getMemPtr<float>()[t * 2 + f] = t < rows.back()
                    ? (0.4f + 0.1f * static_cast<float>(t) - 0.12f * f)
                    : numeric_limits<float>::quiet_NaN();
                targets.getMemPtr<float>()[t * 2 + f] = t < rows.back()
                    ? (-0.2f + 0.02f * static_cast<float>(t))
                    : numeric_limits<float>::quiet_NaN();
            }
        Impl::RowPartitionRuntime partition(offsets, desc.getRowPartition());
        partition.setHostOffsets(rows);
        Batch batch;
        batch.insert("loss_values", Impl::RaggedTensor(packed, partition));
        batch.insert("labels", targets);
        batch.setValidExampleCount(b);
        map<string, Impl::Tensor> outputs;
        map<string, Event> ready;
        auto done = placed->submitBatch(0, batch, outputs, ready, false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(outputs.contains("raw_loss"));
        ASSERT_TRUE(outputs.contains("report"));
        ASSERT_TRUE(physical->getErrorOutput().has_value()) << "RAW gradient was pruned";
        const auto gradient = download1T(physical->getErrorOutput().value());
        for (uint64_t t = 0; t < rows.back(); ++t)
            for (uint32_t f = 0; f < 2; ++f) {
                const uint64_t index = t * 2 + f;
                const float d = packed.getMemPtr<float>()[index] - targets.getMemPtr<float>()[index];
                EXPECT_NEAR(gradient[index], 2.0f * d * Impl::Loss::getLossScalingFactor(), 6e-3f);
            }
    }
    filesystem::remove_all(archive);
}
