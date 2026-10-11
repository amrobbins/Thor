#include "DeepLearning/Api/Layers/Utility/RaggedSequenceSlice.h"
#include "DeepLearning/Api/Layers/Utility/RaggedToPaddedDense.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Loss/CustomLoss.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedSequenceSlice.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Common/Event.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
RaggedTensor source1R(Network& network, const string& name = "source") {
    return RaggedNetworkInput::Builder().network(network).name(name)
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(5).build();
}
Impl::DynamicExpression squared1R(bool derivative) {
    auto predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    auto labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    auto d = predictions - labels;
    auto v = derivative ? d * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()) : d * d;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            derivative ? "predictions_grad" : "loss", v.withOutputDType(DataType::FP32)}})));
}
float value1R(uint64_t token, uint32_t feature) {
    return 0.31f + 0.12f * static_cast<float>(token) - 0.17f * static_cast<float>(feature);
}
float delta1R(uint32_t row, uint32_t column, uint32_t f) {
    return ((row + column + f) % 2 == 0 ? 1.0f : -1.0f) *
        (0.1f + 0.04f * static_cast<float>((row + column * 3 + f) % 6));
}
vector<float> download1R(const Impl::Tensor& gpu, uint64_t n) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = gpu.clone(cpu);
    Impl::Tensor handle = gpu;
    Stream stream = Stream::getNextDownloadStream(gpu.getPlacement().getDeviceNum());
    host.downloadSection(handle, stream, 0, 0, n * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + n);
}
class SliceStampProbe1R : public RaggedSequenceSlice {
public:
    explicit SliceStampProbe1R(const RaggedSequenceSlice& layer) : RaggedSequenceSlice(layer) {}
    using RaggedSequenceSlice::stamp;
};
void reject1R(Network& network, const json& j) {
    EXPECT_ANY_THROW(RaggedSequenceSlice::deserialize(j, &network));
}
} // namespace

// Simulate historical packed dimensions and deliberately wrong obsolete
// physical capacity metadata without changing semantic tensor identities.
static void toHistoricalPackedReference(nlohmann::json& ref) {
    const uint64_t m = ref.at("max_values_per_row").get<uint64_t>();
    ref.erase("layer_ragged_reference_version");
    ref.erase("geometry_mode");
    ref.erase("max_values_per_row");
    ref["batch_size"] = 999U;
    ref["max_total_values"] = 123456U;
    ref["values"]["dimensions"][0] = m * 3U;
    ref["offsets"]["dimensions"][0] = 4U;
}

TEST(RaggedSequenceSliceNeutral1R, BoundsPartitionMemoryAndLegacy) {
    Network n("slice_1r_bounds");
    const auto source = source1R(n);
    const auto slice = RaggedSequenceSlice::Builder().network(n).featureInput(source)
        .start(1).length(3).build();
    const auto output = slice.getRaggedFeatureOutput();
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{3, 2}));
    EXPECT_EQ(output.getOffsetsDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(output.getMaxValuesPerRow(), 3U);
    EXPECT_FALSE(output.sharesPartitionWith(source));
    EXPECT_NE(output.getRowPartitionToken(), source.getRowPartitionToken());
    EXPECT_EQ(slice.getRaggedPartitionRequirementForInput(source.getRowPartitionToken()),
              Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);
    const json j = slice.architectureJson();
    EXPECT_EQ(j.at("version"), "1.1.0");
    for (const char* field : {"ragged_input", "ragged_output"}) {
        EXPECT_FALSE(j.at(field).contains("batch_size"));
        EXPECT_FALSE(j.at(field).contains("max_total_values"));
        EXPECT_EQ(j.at(field).at("offsets").at("dimensions"), (vector<uint64_t>{1}));
    }
    EXPECT_NO_THROW(RaggedSequenceSlice::deserialize(j, &n));
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        EXPECT_EQ(slice.getOutputTensorBytes(b), b * 3U * 2U * sizeof(float));
        EXPECT_EQ(slice.getFirstInstanceMemRequirementInBytes(b, cpu),
                  b * 3U * 2U * sizeof(float) + (b + 1U) * sizeof(uint32_t));
    }
    EXPECT_ANY_THROW((void)slice.getOutputTensorBytes(0));
    const auto empty = RaggedSequenceSlice::Builder().network(n).featureInput(source)
        .start(numeric_limits<uint64_t>::max()).length(numeric_limits<uint64_t>::max()).build();
    EXPECT_EQ(empty.getRaggedFeatureOutput().getMaxValuesPerRow(), 1U);
    EXPECT_ANY_THROW((void)RaggedSequenceSlice::Builder().network(n)
        .featureInput(source).start(0).length(0).build());
    // Historical 1.0.0 packed references are exercised independently in
    // SliceHistoricalPackedGrammarDerivesNewPartitionM below.
}

TEST(RaggedSequenceSliceNeutral1R, RejectsCorruptReferencesAndModeSwitch) {
    Network n("slice_1r_bad_json");
    const auto source = source1R(n);
    const auto slice = RaggedSequenceSlice::Builder().network(n).featureInput(source).start(1).length(3).build();
    const json record = slice.architectureJson();
    json bad = record; bad["version"] = "2.0.0"; reject1R(n, bad);
    bad = record; bad["factory"] = "bogus"; reject1R(n, bad);
    bad = record; bad["batch_size"] = 7; reject1R(n, bad);
    bad = record; bad["ragged_input"]["max_total_values"] = 35; reject1R(n, bad);
    bad = record; bad["ragged_output"]["batch_size"] = 2; reject1R(n, bad);
    bad = record; bad["ragged_input"]["values"]["data_type"] = DataType::FP16; reject1R(n, bad);
    bad = record; bad["ragged_input"]["values"]["dimensions"] = vector<uint64_t>{5, 3}; reject1R(n, bad);
    bad = record; bad["ragged_input"]["offsets"]["id"] = source.getValues().getOriginalId(); reject1R(n, bad);
    bad = record; bad["ragged_output"]["values"]["id"] = source.getValues().getOriginalId(); reject1R(n, bad);
    bad = record; bad["ragged_output"]["offsets"]["id"] = source.getOffsets().getOriginalId(); reject1R(n, bad);
    bad = record; bad["ragged_output"]["values"]["dimensions"] = vector<uint64_t>{4, 2}; reject1R(n, bad);
    bad = record; bad["ragged_output"]["offsets"]["data_type"] = DataType::UINT64; reject1R(n, bad);
    bad = record; bad["ragged_output"]["max_values_per_row"] = 2; reject1R(n, bad);
    bad = record; bad["start"] = 5; reject1R(n, bad);
    bad = record; bad["length"] = 0; reject1R(n, bad);
    bad = record; bad["ragged_output"]["extra"] = 17; reject1R(n, bad);
}

TEST(RaggedSequenceSliceNeutral1R, ValuesAndOffsetsBothDriveStamp) {
    Network n("slice_1r_drivers");
    const auto source = source1R(n);
    const auto slice = RaggedSequenceSlice::Builder().network(n).featureInput(source).start(1).length(3).build();
    SliceStampProbe1R probe(slice);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        auto desc = source.getDescriptorForPlacement(b);
        auto driverValues = make_shared<Impl::NetworkInput>(cpu, DataType::FP32,
                                                             desc.getValuesDescriptor().getDimensions());
        auto driverOffsets = make_shared<Impl::NetworkInput>(cpu, DataType::UINT32,
                                                              desc.getOffsetsDescriptor().getDimensions());
        auto fromValues = dynamic_pointer_cast<Impl::RaggedSequenceSlice>(
            probe.stamp(cpu, driverValues, nullptr, source.getValues(), true));
        auto fromOffsets = dynamic_pointer_cast<Impl::RaggedSequenceSlice>(
            probe.stamp(cpu, driverOffsets, nullptr, source.getOffsets(), true));
        ASSERT_NE(fromValues, nullptr);
        ASSERT_NE(fromOffsets, nullptr);
        auto wrong = make_shared<Impl::NetworkInput>(cpu, DataType::FP32, vector<uint64_t>{b * 5U, 3});
        EXPECT_ANY_THROW((void)probe.stamp(cpu, wrong, nullptr, source.getValues(), true));
        auto wrongOffsets = make_shared<Impl::NetworkInput>(cpu, DataType::UINT32, vector<uint64_t>{1});
        EXPECT_ANY_THROW((void)probe.stamp(cpu, wrongOffsets, nullptr, source.getOffsets(), true));
    }
}

TEST(RaggedSequenceSliceNeutral1R, NeutralSubgraphCloneMakesNewPartition) {
    Network src("slice_1r_clone_source");
    const auto input = source1R(src);
    const auto slice = RaggedSequenceSlice::Builder().network(src).featureInput(input).start(1).length(3).build();
    (void)RaggedNetworkOutput::Builder().network(src).name("sliced")
        .inputTensor(slice.getRaggedFeatureOutput()).build();
    Network dst("slice_1r_clone_destination");
    const auto newInput = source1R(dst);
    ApiTensorRemap remap;
    remap.map(input.getValues(), newInput.getValues());
    remap.map(input.getRowPartitionToken(), newInput.getRowPartitionToken());
    ApiSubgraphCloneOptions options; options.inferenceOnly = true;
    const auto clone = dst.cloneSubgraphInto(src, {"__thor_ragged_output.sliced.values"}, remap, options);
    ASSERT_EQ(clone.outputTensorsByName.size(), 1U);
    EXPECT_EQ(clone.outputTensorsByName.at("__thor_ragged_output.sliced.values").getDimensions(), (vector<uint64_t>{3, 2}));
    auto clonedToken = clone.clonedTensorBySourceOriginalId.at(
        slice.getRaggedFeatureOutput().getRowPartitionToken().getOriginalId());
    EXPECT_NE(clonedToken, newInput.getRowPartitionToken());
    EXPECT_EQ(clonedToken.getDimensions(), (vector<uint64_t>{1}));
}

TEST(RaggedSequenceSliceNeutral1R, SavedForwardAndBackwardAtTwoAndSeven) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    const filesystem::path path = filesystem::temp_directory_path() /
        ("slice_1r_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(path);
    uint64_t inputTokenId = 0, outputTokenId = 0, outputValuesId = 0;
    {
        Network n("slice_1r_training");
        const auto source = source1R(n);
        inputTokenId = source.getRowPartitionToken().getOriginalId();
        const auto rivet = GradientRivet::Builder().network(n).tensor(source.getValues()).build();
        const auto slice = RaggedSequenceSlice::Builder().network(n)
            .featureInput(source.withValues(rivet.getFeatureOutput().value()))
            .start(1).length(3).build();
        const auto out = slice.getRaggedFeatureOutput();
        outputTokenId = out.getRowPartitionToken().getOriginalId();
        outputValuesId = out.getValues().getOriginalId();
        ASSERT_FALSE(out.sharesPartitionWith(source));
        const auto pad = RaggedToPaddedDense::Builder().network(n).featureInput(out)
            .paddingValue(-7.0).build();
        const auto labels = NetworkInput::Builder().network(n).name("targets")
            .dataType(DataType::FP32).dimensions({3, 2}).build();
        const auto loss = CustomLoss::Builder().network(n)
            .predictions(pad.getPaddedFeatureOutput()).labels(labels.getFeatureOutput().value())
            .lossExpression(squared1R(false)).gradientExpression(squared1R(true))
            .reportsRawLoss().build();
        (void)NetworkOutput::Builder().network(n).name("sliced_padded")
            .inputTensor(pad.getPaddedFeatureOutput()).dataType(DataType::FP32).build();
        (void)NetworkOutput::Builder().network(n).name("raw_loss")
            .inputTensor(loss.getRawLoss()).dataType(DataType::FP32).build();
        n.save(path.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network n("slice_1r_training");
        ASSERT_NO_THROW(n.load(path.string()));
        const auto inputs = n.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(), 1U);
        const auto source = inputs.front().raggedTensor;
        EXPECT_EQ(source.getRowPartitionToken().getOriginalId(), inputTokenId);
        shared_ptr<RaggedSequenceSlice> slice;
        for (uint32_t i = 0; i < n.getNumLayers(); ++i)
            if (auto candidate = dynamic_pointer_cast<RaggedSequenceSlice>(n.getLayer(i))) slice = candidate;
        ASSERT_NE(slice, nullptr);
        const auto out = slice->getRaggedFeatureOutput();
        EXPECT_EQ(out.getRowPartitionToken().getOriginalId(), outputTokenId);
        EXPECT_EQ(out.getValues().getOriginalId(), outputValuesId);
        EXPECT_NE(out.getRowPartitionToken(), source.getRowPartitionToken());
        const json before = n.architectureJson();
        vector<Event> init;
        auto placed = n.place(b, init, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : init) event.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        auto physical = dynamic_pointer_cast<Impl::RaggedSequenceSlice>(
            stamped.getPhysicalLayerFromApiLayer(slice->getId()));
        ASSERT_NE(physical, nullptr);
        const auto ports = physical->getFeatureInputs();
        ASSERT_EQ(ports.size(), 3U);
        for (const auto& port : ports) ASSERT_TRUE(port.has_value());
        EXPECT_EQ(ports[0]->getDimensions(), (vector<uint64_t>{b * 5U, 2}));
        EXPECT_EQ(ports[1]->getDimensions(), (vector<uint64_t>{b + 1U}));
        EXPECT_EQ(ports[2]->getDimensions(), (vector<uint64_t>{b + 1U}));
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b * 3U, 2}));
        EXPECT_EQ(n.architectureJson(), before);
        vector<uint64_t> offsets(b + 1, 0), slicedOffsets(b + 1, 0);
        for (uint32_t row = 0; row < b; ++row) {
            const uint32_t count = (row % 4 == 0) ? 5U : (row % 4 == 1) ? 2U :
                                   (row % 4 == 2) ? 0U : 4U;
            offsets[row + 1] = offsets[row] + count;
            slicedOffsets[row + 1] = slicedOffsets[row] +
                (count > 1 ? min<uint32_t>(3, count - 1) : 0);
        }
        const uint64_t active = offsets.back();
        const uint64_t slicedActive = slicedOffsets.back();
        ASSERT_GT(active, slicedActive);
        ASSERT_LT(active, static_cast<uint64_t>(b) * 5);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor off(cpu, source.getDescriptorForPlacement(b).getOffsetsDescriptor());
        for (uint32_t row = 0; row <= b; ++row)
            off.getMemPtr<uint32_t>()[row] = static_cast<uint32_t>(offsets[row]);
        Impl::RowPartitionRuntime partition(off, source.getDescriptorForPlacement(b).getRowPartition());
        partition.setHostOffsets(offsets);
        Impl::Tensor values(cpu, source.getDescriptorForPlacement(b).getValuesDescriptor());
        for (uint64_t token = 0; token < b * 5ULL; ++token)
            for (uint32_t f = 0; f < 2; ++f)
                values.getMemPtr<float>()[token * 2 + f] = token < active
                    ? value1R(token, f) : numeric_limits<float>::quiet_NaN();
        Impl::Tensor labels(cpu, Impl::TensorDescriptor(DataType::FP32, {b, 3, 2}));
        for (uint32_t row = 0; row < b; ++row)
            for (uint32_t col = 0; col < 3; ++col)
                for (uint32_t f = 0; f < 2; ++f) {
                    const bool selected = col < slicedOffsets[row + 1] - slicedOffsets[row];
                    const float prediction = selected ? value1R(offsets[row] + 1 + col, f) : -7.0f;
                    labels.getMemPtr<float>()[(row * 3 + col) * 2 + f] =
                        prediction - delta1R(row, col, f);
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
        ASSERT_TRUE(results.contains("sliced_padded"));
        const auto predicted = download1R(results.at("sliced_padded"), static_cast<uint64_t>(b) * 6);
        const auto grads = physical->getErrorOutputs();
        ASSERT_EQ(grads.size(), 3U);
        ASSERT_TRUE(grads[0].has_value()) << "slice values backward path was pruned";
        EXPECT_FALSE(grads[1].has_value());
        EXPECT_FALSE(grads[2].has_value());
        const auto sourceGrad = download1R(grads[0].value(), active * 2);
        for (uint32_t row = 0; row < b; ++row) {
            const uint32_t length = static_cast<uint32_t>(offsets[row + 1] - offsets[row]);
            for (uint32_t col = 0; col < 3; ++col)
                for (uint32_t f = 0; f < 2; ++f) {
                    const bool selected = col < slicedOffsets[row + 1] - slicedOffsets[row];
                    const float expected = selected ? value1R(offsets[row] + 1 + col, f) : -7.0f;
                    EXPECT_NEAR(predicted[(row * 3 + col) * 2 + f], expected, 3e-4f);
                }
            for (uint32_t col = 0; col < length; ++col)
                for (uint32_t f = 0; f < 2; ++f) {
                    const bool selected = col >= 1 && col < 4;
                    const float expected = selected ? 2.0f * delta1R(row, col - 1, f) *
                        Impl::Loss::getLossScalingFactor() : 0.0f;
                    EXPECT_NEAR(sourceGrad[(offsets[row] + col) * 2 + f], expected, 7e-3f)
                        << "source gradient row=" << row << " col=" << col;
                }
        }
    }
    filesystem::remove_all(path);
}

TEST(RaggedSequenceSliceNeutral1R, SliceHistoricalPackedGrammarDerivesNewPartitionM) {
    Network n("slice_old_reader");
    const auto source = source1R(n);
    const auto op = RaggedSequenceSlice::Builder().network(n).featureInput(source).start(1).length(3).build();
    json old = op.architectureJson();
    old["version"] = "1.0.0";
    toHistoricalPackedReference(old["ragged_input"]);
    toHistoricalPackedReference(old["ragged_output"]);
    ASSERT_NO_THROW(RaggedSequenceSlice::deserialize(old, &n));
    for (uint64_t b : {2U, 7U}) {
        const auto descriptor = op.getRaggedFeatureOutput().getDescriptorForPlacement(b);
        EXPECT_EQ(descriptor.getValuesDimensions(), (vector<uint64_t>{b * 3U, 2U}));
        EXPECT_EQ(descriptor.getOffsetsDescriptor().getDimensions(), (vector<uint64_t>{b + 1U}));
    }
    old["ragged_output"]["values"]["dimensions"][1] = 17U;
    EXPECT_ANY_THROW(RaggedSequenceSlice::deserialize(old, &n));
}
