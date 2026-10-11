#include "DeepLearning/Api/Layers/Utility/RaggedToPaddedDense.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedToPaddedDense.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Event.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
RaggedTensor oneOInput(Network& n) {
    return RaggedNetworkInput::Builder().network(n).name("source")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(4).build();
}

void oneOReject(Network& n, const json& j) {
    EXPECT_ANY_THROW(RaggedToPaddedDense::deserialize(j, &n));
}

Impl::DynamicExpression oneOSquaredError(bool derivative) {
    auto predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    auto labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    auto delta = predictions - labels;
    auto result = derivative
        ? delta * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
        : delta * delta;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(
            Impl::Expression::outputs({{derivative ? "predictions_grad" : "loss",
                                       result.withOutputDType(DataType::FP32)}})));
}

float oneOValue(uint64_t token, uint32_t feature) {
    return 0.4f + static_cast<float>(token) * 0.13f - static_cast<float>(feature) * 0.23f;
}
float oneODelta(uint32_t row, uint32_t column, uint32_t feature) {
    const float magnitude = 0.11f + 0.07f * static_cast<float>((row + 2 * column + feature) % 5);
    return ((row + column + feature) % 2 == 0 ? 1.0f : -1.0f) * magnitude;
}
vector<float> oneODownload(const Impl::Tensor& source, uint64_t count) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = source.clone(cpu);
    Impl::Tensor handle = source;
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    host.downloadSection(handle, stream, 0, 0, count * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + count);
}

class OneOStampProbe : public RaggedToPaddedDense {
public:
    explicit OneOStampProbe(const RaggedToPaddedDense& value) : RaggedToPaddedDense(value) {}
    using RaggedToPaddedDense::stamp;
};
} // namespace

TEST(RaggedToPaddedDenseNeutral1O, SemanticShapeSchemaAndMemoryAtTwoAndSeven) {
    Network network("adapter_1o_unplaced");
    const RaggedTensor source = oneOInput(network);
    const auto layer = RaggedToPaddedDense::Builder().network(network)
        .featureInput(source).paddingValue(-7.5).build();
    EXPECT_EQ(layer.getPaddedFeatureOutput().getDimensions(), (vector<uint64_t>{4, 2}));
    EXPECT_EQ(layer.getPaddedFeatureOutput().getDataType(), DataType::FP32);
    ASSERT_EQ(layer.getFeatureInputs().size(), 2U);
    EXPECT_EQ(layer.getFeatureInputs()[0], source.getValues());
    EXPECT_EQ(layer.getFeatureInputs()[1], source.getRowPartitionToken());
    EXPECT_EQ(layer.getRaggedPartitionRequirementForInput(source.getRowPartitionToken()),
              Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);
    const json record = layer.architectureJson();
    EXPECT_EQ(record.at("version"), "1.1.0");
    EXPECT_EQ(record.at("padding_value"), -7.5);
    const json& r = record.at("ragged_feature_input");
    EXPECT_FALSE(r.contains("geometry_mode"));
    EXPECT_EQ(r.at("max_values_per_row"), 4);
    EXPECT_EQ(r.at("offsets").at("dimensions"), (vector<uint64_t>{1}));
    EXPECT_FALSE(r.contains("batch_size"));
    EXPECT_FALSE(r.contains("max_total_values"));
    EXPECT_NO_THROW(RaggedToPaddedDense::deserialize(record, &network));

    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        const uint64_t packedBytes = static_cast<uint64_t>(b) * 4 * 2 * sizeof(float);
        const uint64_t offsetsBytes = (static_cast<uint64_t>(b) + 1) * sizeof(uint32_t);
        EXPECT_EQ(layer.getOutputTensorBytes(b), packedBytes);
        EXPECT_EQ(layer.getFirstInstanceMemRequirementInBytes(b, cpu),
                  packedBytes + offsetsBytes + packedBytes);
    }
    EXPECT_ANY_THROW((void)layer.getOutputTensorBytes(0));
    EXPECT_ANY_THROW((void)layer.getFirstInstanceMemRequirementInBytes(0, cpu));

    Network fixed("adapter_1o_fixed");
    const auto legacy = RaggedNetworkInput::Builder().network(fixed).name("old")
        .valuesDataType(DataType::FP32).trailingDimensions({2})
        .maxValuesPerRow(4).build();
    const auto oldLayer = RaggedToPaddedDense::Builder().network(fixed)
        .featureInput(legacy).paddingValue(-3.0).build();
    EXPECT_EQ(oldLayer.architectureJson().at("version"), "1.1.0");
    EXPECT_NO_THROW(RaggedToPaddedDense::deserialize(oldLayer.architectureJson(), &fixed));
    EXPECT_EQ(oldLayer.getOutputTensorBytes(7), 7U * 4U * 2U * sizeof(float));

    // A historical 1.0.0 archive has physical hints, not a fixed-batch model.
    json historical = oldLayer.architectureJson();
    historical["version"] = "1.0.0";
    auto& ref = historical["ragged_feature_input"];
    ref.erase("layer_ragged_reference_version");
    ref.erase("max_values_per_row");
    ref["batch_size"] = 2U;
    ref["max_total_values"] = 8U;
    ref["values"]["dimensions"][0] = 8U;
    ref["offsets"]["dimensions"][0] = 3U;
    EXPECT_NO_THROW(RaggedToPaddedDense::deserialize(historical, &fixed));
}

TEST(RaggedToPaddedDenseNeutral1O, RejectsMalformedSemanticReferences) {
    Network n("adapter_1o_schema");
    const auto source = oneOInput(n);
    const auto layer = RaggedToPaddedDense::Builder().network(n).featureInput(source).build();
    const json j = layer.architectureJson();
    json bad = j; bad["version"] = "2.0.0"; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["values"]["id"] = source.getOffsets().getOriginalId(); oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["values"]["data_type"] = DataType::BF16; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["values"]["dimensions"] = vector<uint64_t>{4, 3}; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["offsets"]["id"] = source.getValues().getOriginalId(); oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["offsets"]["data_type"] = DataType::UINT64; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["offsets"]["dimensions"] = vector<uint64_t>{3}; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["max_values_per_row"] = 5; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["batch_size"] = 2; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["max_total_values"] = 8; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["geometry_mode"] = "fixed"; oneOReject(n, bad);
    bad = j; bad["ragged_feature_input"]["extra"] = 1; oneOReject(n, bad);
    bad = j; bad["feature_output"]["dimensions"] = vector<uint64_t>{3, 2}; oneOReject(n, bad);
    bad = j; bad["feature_output"]["data_type"] = DataType::BF16; oneOReject(n, bad);
    bad = j; bad["feature_output"]["id"] = source.getValues().getOriginalId(); oneOReject(n, bad);
    bad = j; bad["feature_output"]["batch_size"] = 7; oneOReject(n, bad);
    bad = j; bad["batch_size"] = 7; oneOReject(n, bad);
    bad = j; bad["factory"] = "unsupported"; oneOReject(n, bad);
}

TEST(RaggedToPaddedDenseNeutral1O, NeutralSubgraphCloneRemapsCanonicalPartition) {
    Network sourceNetwork("adapter_1o_clone_source");
    const RaggedTensor source = oneOInput(sourceNetwork);
    const auto adapter = RaggedToPaddedDense::Builder().network(sourceNetwork)
        .featureInput(source).paddingValue(-7.5).build();
    (void)NetworkOutput::Builder().network(sourceNetwork).name("padded")
        .inputTensor(adapter.getPaddedFeatureOutput()).dataType(DataType::FP32).build();

    Network destination("adapter_1o_clone_target");
    const RaggedTensor target = oneOInput(destination);
    ApiTensorRemap remap;
    remap.map(source.getValues(), target.getValues());
    remap.map(source.getRowPartitionToken(), target.getRowPartitionToken());
    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    const auto result = destination.cloneSubgraphInto(sourceNetwork, {"padded"}, remap, options);
    ASSERT_EQ(result.outputTensorsByName.size(), 1U);
    EXPECT_EQ(result.outputTensorsByName.at("padded").getDimensions(), (vector<uint64_t>{4, 2}));
    EXPECT_EQ(result.outputTensorsByName.at("padded").getDataType(), DataType::FP32);
    EXPECT_EQ(result.clonedTensorBySourceOriginalId.at(source.getRowPartitionToken().getOriginalId()),
              target.getRowPartitionToken());
}

TEST(RaggedToPaddedDenseNeutral1O, ValuesAndOffsetsCanEachDriveStamp) {
    Network n("adapter_1o_drivers");
    const auto source = oneOInput(n);
    const auto logical = RaggedToPaddedDense::Builder().network(n).featureInput(source).build();
    OneOStampProbe probe(logical);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        const auto descriptor = source.getDescriptorForPlacement(b);
        auto valuesDriver = make_shared<Impl::NetworkInput>(cpu, DataType::FP32,
                                                             descriptor.getValuesDescriptor().getDimensions());
        auto fromValues = dynamic_pointer_cast<Impl::RaggedToPaddedDense>(
            probe.stamp(cpu, valuesDriver, nullptr, source.getValues(), true));
        ASSERT_NE(fromValues, nullptr);
        auto offsetsDriver = make_shared<Impl::NetworkInput>(cpu, source.getOffsetsDataType(),
                                                              descriptor.getOffsetsDescriptor().getDimensions());
        auto fromOffsets = dynamic_pointer_cast<Impl::RaggedToPaddedDense>(
            probe.stamp(cpu, offsetsDriver, nullptr, source.getOffsets(), true));
        ASSERT_NE(fromOffsets, nullptr);
        auto wrongValues = make_shared<Impl::NetworkInput>(cpu, DataType::FP32,
                                                            vector<uint64_t>{b * 4U, 3});
        EXPECT_ANY_THROW((void)probe.stamp(cpu, wrongValues, nullptr, source.getValues(), true));
        auto wrongOffsets = make_shared<Impl::NetworkInput>(cpu, DataType::UINT32,
                                                             vector<uint64_t>{1});
        EXPECT_ANY_THROW((void)probe.stamp(cpu, wrongOffsets, nullptr, source.getOffsets(), true));
    }
}

TEST(RaggedToPaddedDenseNeutral1O, SavedForwardAndBackwardAtTwoAndSeven) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const filesystem::path path = filesystem::temp_directory_path() /
        ("adapter_1o_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(path);
    uint64_t partitionId = 0, valuesId = 0, outputId = 0;
    {
        Network n("adapter_1o_training");
        const auto source = oneOInput(n);
        partitionId = source.getRowPartitionToken().getOriginalId();
        const auto rivet = GradientRivet::Builder().network(n).tensor(source.getValues()).build();
        const auto adapter = RaggedToPaddedDense::Builder().network(n)
            .featureInput(source.withValues(rivet.getFeatureOutput().value()))
            .paddingValue(-7.5).build();
        valuesId = adapter.getFeatureInputs()[0].getOriginalId();
        outputId = adapter.getPaddedFeatureOutput().getOriginalId();
        const auto labels = NetworkInput::Builder().network(n).name("targets")
            .dataType(DataType::FP32).dimensions({4, 2}).build();
        const auto loss = CustomLoss::Builder().network(n)
            .predictions(adapter.getPaddedFeatureOutput()).labels(labels.getFeatureOutput().value())
            .lossExpression(oneOSquaredError(false)).gradientExpression(oneOSquaredError(true))
            .reportsRawLoss().build();
        (void)NetworkOutput::Builder().network(n).name("padded")
            .inputTensor(adapter.getPaddedFeatureOutput()).dataType(DataType::FP32).build();
        (void)NetworkOutput::Builder().network(n).name("raw_loss")
            .inputTensor(loss.getRawLoss()).dataType(DataType::FP32).build();
        n.save(path.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network n("adapter_1o_training");
        ASSERT_NO_THROW(n.load(path.string()));
        auto external = n.getExternalRaggedNetworkInputs();
        ASSERT_EQ(external.size(), 1U);
        const auto source = external.front().raggedTensor;
        EXPECT_EQ(source.getRowPartitionToken().getOriginalId(), partitionId);
        const RaggedToPaddedDense* layer = nullptr;
        for (uint32_t i = 0; i < n.getNumLayers(); ++i) {
            auto candidate = dynamic_pointer_cast<RaggedToPaddedDense>(n.getLayer(i));
            if (candidate) layer = candidate.get();
        }
        ASSERT_NE(layer, nullptr);
        EXPECT_EQ(layer->getFeatureInputs()[0].getOriginalId(), valuesId);
        EXPECT_EQ(layer->getFeatureInputs()[1], source.getRowPartitionToken());
        EXPECT_EQ(layer->getPaddedFeatureOutput().getOriginalId(), outputId);
        const json before = n.architectureJson();
        vector<Event> init;
        const auto placed = n.place(b, init, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : init) event.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        auto physical = dynamic_pointer_cast<Impl::RaggedToPaddedDense>(
            stamped.getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical, nullptr);
        const auto ports = physical->getFeatureInputs();
        ASSERT_EQ(ports.size(), 2U);
        ASSERT_TRUE(ports[0].has_value());
        ASSERT_TRUE(ports[1].has_value());
        EXPECT_EQ(ports[0]->getDimensions(), (vector<uint64_t>{b * 4U, 2}));
        EXPECT_EQ(ports[1]->getDimensions(), (vector<uint64_t>{b + 1U}));
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b, 4, 2}));
        EXPECT_NE(stamped.getManagedPartitionOffsetsInputForTest(source.getRowPartitionId()), nullptr);
        EXPECT_EQ(n.architectureJson(), before);

        vector<uint64_t> offsets(b + 1, 0);
        for (uint32_t row = 0; row < b; ++row) {
            const uint32_t length = row % 3 == 0 ? 0 : (row % 3 == 1 ? 2 : 4);
            offsets[row + 1] = offsets[row] + length;
        }
        const uint64_t active = offsets.back();
        ASSERT_GT(active, 0U);
        ASSERT_LT(active, static_cast<uint64_t>(b) * 4);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor off(cpu, source.getDescriptorForPlacement(b).getOffsetsDescriptor());
        for (uint32_t row = 0; row <= b; ++row)
            off.getMemPtr<uint32_t>()[row] = static_cast<uint32_t>(offsets[row]);
        Impl::RowPartitionRuntime partition(off, source.getDescriptorForPlacement(b).getRowPartition());
        partition.setHostOffsets(offsets);
        Impl::Tensor values(cpu, source.getDescriptorForPlacement(b).getValuesDescriptor());
        for (uint64_t token = 0; token < b * 4ULL; ++token)
            for (uint32_t f = 0; f < 2; ++f)
                values.getMemPtr<float>()[token * 2 + f] = token < active
                    ? oneOValue(token, f) : numeric_limits<float>::quiet_NaN();
        Impl::Tensor labels(cpu, Impl::TensorDescriptor(DataType::FP32, {b, 4, 2}));
        for (uint32_t row = 0; row < b; ++row)
            for (uint32_t column = 0; column < 4; ++column)
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint64_t token = offsets[row] + column;
                    const bool exists = column < offsets[row + 1] - offsets[row];
                    const float prediction = exists ? oneOValue(token, f) : -7.5f;
                    // Padded rows receive a non-zero upstream derivative.
                    // It must not leak into any active packed values gradient.
                    labels.getMemPtr<float>()[(row * 4 + column) * 2 + f] =
                        prediction - oneODelta(row, column, f);
                }
        Batch batch;
        batch.insert("source", Impl::RaggedTensor(values, partition));
        batch.insert("targets", labels);
        batch.setValidExampleCount(b);
        map<string, Impl::Tensor> results;
        map<string, Event> ready;
        auto done = placed->submitBatch(0, batch, results, ready, false);
        done.synchronize();
        placed->synchronize();
        ASSERT_TRUE(results.contains("padded"));
        ASSERT_TRUE(results.contains("raw_loss"));
        const auto predicted = oneODownload(results.at("padded"), static_cast<uint64_t>(b) * 8);
        const auto losses = oneODownload(results.at("raw_loss"), static_cast<uint64_t>(b) * 8);
        const auto grads = physical->getErrorOutputs();
        ASSERT_EQ(grads.size(), 2U);
        ASSERT_TRUE(grads[0].has_value()) << "values backward path was pruned";
        EXPECT_FALSE(grads[1].has_value()) << "offsets are structural, not differentiable";
        const auto activeGrad = oneODownload(grads[0].value(), active * 2);
        for (uint32_t row = 0; row < b; ++row)
            for (uint32_t column = 0; column < 4; ++column)
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint64_t index = (row * 4 + column) * 2 + f;
                    const bool exists = column < offsets[row + 1] - offsets[row];
                    const float prediction = exists ? oneOValue(offsets[row] + column, f) : -7.5f;
                    EXPECT_NEAR(predicted[index], prediction, 3e-4f) << "row " << row << " column " << column;
                    const float delta = oneODelta(row, column, f);
                    EXPECT_NEAR(losses[index], delta * delta, 4e-4f);
                    if (exists) {
                        const uint64_t packed = (offsets[row] + column) * 2 + f;
                        EXPECT_NEAR(activeGrad[packed], 2.0f * delta * Impl::Loss::getLossScalingFactor(), 6e-3f)
                            << "active gradient row " << row << " column " << column;
                    }
                }
    }
    filesystem::remove_all(path);
}
