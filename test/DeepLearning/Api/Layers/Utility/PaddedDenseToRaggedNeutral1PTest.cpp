#include "DeepLearning/Api/Layers/Utility/PaddedDenseToRagged.h"
#include "DeepLearning/Api/Layers/Utility/RaggedToPaddedDense.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/PaddedDenseToRagged.h"
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
RaggedTensor partition1P(Network& n, const string& name = "source") {
    return RaggedNetworkInput::Builder().network(n).name(name)
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(4).build();
}

Impl::DynamicExpression squared1P(bool gradient) {
    auto p = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    auto l = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    auto diff = p - l;
    auto value = gradient ? diff * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor()) : diff * diff;
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{
            gradient ? "predictions_grad" : "loss", value.withOutputDType(DataType::FP32)}})));
}

float packedValue(uint64_t token, uint32_t f) {
    return 0.3f + 0.17f * static_cast<float>(token) - 0.11f * static_cast<float>(f);
}
float directValue(uint32_t row, uint32_t col, uint32_t f) {
    return 0.4f + 0.23f * static_cast<float>(row) + 0.07f * static_cast<float>(col) - 0.2f * static_cast<float>(f);
}
float upstream1P(uint64_t token, uint32_t f) {
    float magnitude = 0.11f + 0.05f * static_cast<float>((token + 2 * f) % 5);
    return ((token + f) % 2 == 0 ? 1.0f : -1.0f) * magnitude;
}

vector<float> download1P(const Impl::Tensor& device, uint64_t count) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = device.clone(cpu);
    Impl::Tensor handle = device;
    Stream stream = Stream::getNextDownloadStream(device.getPlacement().getDeviceNum());
    host.downloadSection(handle, stream, 0, 0, count * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + count);
}
class Stamp1P : public PaddedDenseToRagged {
public:
    explicit Stamp1P(const PaddedDenseToRagged& source) : PaddedDenseToRagged(source) {}
    using PaddedDenseToRagged::stamp;
};
void reject1P(Network& n, const json& j) { EXPECT_ANY_THROW(PaddedDenseToRagged::deserialize(j, &n)); }

// Saves once, then reloads and trains at two independently chosen physical B.
// The direct case has W=6, M=4; the composed adapter case has W=M=4.
void savedExecution1P(bool roundTrip) {
    const string name = roundTrip ? "adapter_1p_round_trip" : "adapter_1p_direct";
    const auto archive = filesystem::temp_directory_path() /
        (name + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    uint64_t outputId = 0, partitionId = 0;
    const uint32_t width = roundTrip ? 4 : 6;
    {
        Network n(name);
        const auto source = partition1P(n);
        const auto targets = RaggedNetworkInput::Builder().network(n).name("targets")
            .partition(source).valuesDataType(DataType::FP32).trailingDimensions({2}).build();
        Tensor dense;
        if (roundTrip) {
            // Preserve the upstream source-values gradient through the ragged->dense
            // adapter; the partition carrier itself remains structural.
            const auto sourceRivet = GradientRivet::Builder().network(n)
                .tensor(source.getValues()).build();
            dense = RaggedToPaddedDense::Builder().network(n)
                .featureInput(source.withValues(sourceRivet.getFeatureOutput().value()))
                .paddingValue(-5.0).build().getPaddedFeatureOutput();
        } else {
            // PaddedDenseToRagged reads only the source partition. The source
            // values are deliberately unused data, but must have a forward
            // consumer for graph validation (a structural input is not one).
            (void)Stub::Builder().network(n).inputTensor(source.getValues()).build();
            dense = NetworkInput::Builder().network(n).name("padded")
                .dataType(DataType::FP32).dimensions({width, 2}).build().getFeatureOutput().value();
        }
        const auto rivet = GradientRivet::Builder().network(n).tensor(dense).build();
        const auto pack = PaddedDenseToRagged::Builder().network(n)
            .featureInput(rivet.getFeatureOutput().value()).partitionInput(source).build();
        const RaggedTensor result = pack.getRaggedFeatureOutput();
        ASSERT_TRUE(result.sharesPartitionWith(source));
        EXPECT_EQ(result.getValuesDimensions(), (vector<uint64_t>{4, 2}));
        outputId = result.getValues().getOriginalId();
        partitionId = result.getRowPartitionToken().getOriginalId();
        EXPECT_EQ(pack.architectureJson().at("version"), "1.1.0");
        const auto loss = RaggedCustomLoss::Builder().network(n).predictions(result).labels(targets)
            .lossExpression(squared1P(false)).gradientExpression(squared1P(true)).build();
        (void)NetworkOutput::Builder().network(n).name("raw_loss")
            .inputTensor(loss.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        n.save(archive.string(), true);
    }
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "roundTrip=" << roundTrip << " B=" << b);
        Network n(name);
        ASSERT_NO_THROW(n.load(archive.string()));
        auto inputs = n.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(), 2U);
        RaggedTensor source, targets;
        for (const auto& item : inputs) {
            if (item.name == "source") source = item.raggedTensor;
            if (item.name == "targets") targets = item.raggedTensor;
        }
        ASSERT_TRUE(source.isInitialized());
        ASSERT_TRUE(targets.isInitialized());
        ASSERT_TRUE(source.sharesPartitionWith(targets));
        EXPECT_EQ(source.getRowPartitionToken().getOriginalId(), partitionId);
        shared_ptr<PaddedDenseToRagged> logical;
        shared_ptr<RaggedToPaddedDense> unpack;
        for (uint32_t i = 0; i < n.getNumLayers(); ++i) {
            if (auto p = dynamic_pointer_cast<PaddedDenseToRagged>(n.getLayer(i))) logical = p;
            if (auto p = dynamic_pointer_cast<RaggedToPaddedDense>(n.getLayer(i))) unpack = p;
        }
        ASSERT_NE(logical, nullptr);
        EXPECT_EQ(logical->getRaggedFeatureOutput().getValues().getOriginalId(), outputId);
        EXPECT_EQ(logical->getRaggedFeatureOutput().getRowPartitionToken(), source.getRowPartitionToken());
        EXPECT_EQ(static_cast<bool>(unpack), roundTrip);
        const json before = n.architectureJson();
        vector<Event> initializers;
        auto placed = n.place(b, initializers, false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : initializers) event.synchronize();
        auto& stamped = placed->getStampedNetwork(0);
        auto physical = dynamic_pointer_cast<Impl::PaddedDenseToRagged>(
            stamped.getPhysicalLayerFromApiLayer(logical->getId()));
        ASSERT_NE(physical, nullptr);
        const auto physInputs = physical->getFeatureInputs();
        ASSERT_EQ(physInputs.size(), 2U);
        ASSERT_TRUE(physInputs[0].has_value());
        ASSERT_TRUE(physInputs[1].has_value());
        EXPECT_EQ(physInputs[0]->getDimensions(), (vector<uint64_t>{b, width, 2}));
        EXPECT_EQ(physInputs[1]->getDimensions(), (vector<uint64_t>{b + 1U}));
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b * 4U, 2}));
        EXPECT_NE(stamped.getManagedPartitionOffsetsInputForTest(source.getRowPartitionId()), nullptr);
        EXPECT_EQ(n.architectureJson(), before);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        const auto descriptor = source.getDescriptorForPlacement(b);
        vector<uint64_t> offsets(b + 1U, 0);
        for (uint32_t row = 0; row < b; ++row)
            offsets[row + 1] = offsets[row] + (row % 3 == 0 ? 0 : row % 3 == 1 ? 2 : 4);
        const uint64_t active = offsets.back();
        ASSERT_GT(active, 0U);
        ASSERT_LT(active, static_cast<uint64_t>(b) * 4);
        Impl::Tensor off(cpu, descriptor.getOffsetsDescriptor());
        for (uint32_t row = 0; row <= b; ++row) off.getMemPtr<uint32_t>()[row] = static_cast<uint32_t>(offsets[row]);
        Impl::RowPartitionRuntime partition(off, descriptor.getRowPartition());
        partition.setHostOffsets(offsets);
        Impl::Tensor sourceValues(cpu, descriptor.getValuesDescriptor());
        Impl::Tensor labels(cpu, targets.getDescriptorForPlacement(b).getValuesDescriptor());
        for (uint64_t token = 0; token < static_cast<uint64_t>(b) * 4; ++token)
            for (uint32_t f = 0; f < 2; ++f) {
                sourceValues.getMemPtr<float>()[token * 2 + f] = token < active
                    ? packedValue(token, f) : numeric_limits<float>::quiet_NaN();
                labels.getMemPtr<float>()[token * 2 + f] = numeric_limits<float>::quiet_NaN();
            }
        Impl::Tensor dense(cpu, Impl::TensorDescriptor(DataType::FP32, {b, width, 2}));
        for (uint32_t row = 0; row < b; ++row)
            for (uint32_t col = 0; col < width; ++col)
                for (uint32_t f = 0; f < 2; ++f)
                    dense.getMemPtr<float>()[(row * width + col) * 2 + f] = directValue(row, col, f);
        for (uint32_t row = 0; row < b; ++row)
            for (uint32_t col = 0; col < offsets[row + 1] - offsets[row]; ++col)
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint64_t token = offsets[row] + col;
                    const float p = roundTrip ? packedValue(token, f) : directValue(row, col, f);
                    labels.getMemPtr<float>()[token * 2 + f] = p - upstream1P(token, f);
                }
        Batch batch;
        batch.insert("source", Impl::RaggedTensor(sourceValues, partition));
        batch.insert("targets", labels);
        if (!roundTrip) batch.insert("padded", dense);
        batch.setValidExampleCount(b);
        map<string, Impl::Tensor> results;
        map<string, Event> ready;
        auto event = placed->submitBatch(0, batch, results, ready, false);
        event.synchronize();
        placed->synchronize();
        ASSERT_TRUE(results.contains("raw_loss"));
        const auto activeOutput = download1P(physical->getFeatureOutput().value(), active * 2);
        const auto activeLoss = download1P(results.at("raw_loss"), active * 2);
        const auto grads = physical->getErrorOutputs();
        ASSERT_EQ(grads.size(), 2U);
        ASSERT_TRUE(grads[0].has_value()) << "padded dense backward path unexpectedly pruned";
        EXPECT_FALSE(grads[1].has_value()) << "partition carrier must not be differentiated";
        EXPECT_EQ(grads[0]->getDimensions(), (vector<uint64_t>{b, width, 2}));
        const auto denseGrads = download1P(grads[0].value(), static_cast<uint64_t>(b) * width * 2);
        vector<float> sourceGrads;
        if (roundTrip) {
            ASSERT_NE(unpack, nullptr);
            auto physicalUnpack = dynamic_pointer_cast<Impl::RaggedToPaddedDense>(
                stamped.getPhysicalLayerFromApiLayer(unpack->getId()));
            ASSERT_NE(physicalUnpack, nullptr);
            const auto originalGrads = physicalUnpack->getErrorOutputs();
            ASSERT_EQ(originalGrads.size(), 2U);
            ASSERT_TRUE(originalGrads[0].has_value());
            EXPECT_FALSE(originalGrads[1].has_value());
            sourceGrads = download1P(originalGrads[0].value(), active * 2);
        }
        for (uint32_t row = 0; row < b; ++row)
            for (uint32_t col = 0; col < width; ++col)
                for (uint32_t f = 0; f < 2; ++f) {
                    const bool exists = col < offsets[row + 1] - offsets[row];
                    const uint64_t token = offsets[row] + col;
                    const float expected = exists ?
                        2.0f * upstream1P(token, f) * Impl::Loss::getLossScalingFactor() : 0.0f;
                    EXPECT_NEAR(denseGrads[(row * width + col) * 2 + f], expected, 7e-3f)
                        << "row=" << row << " col=" << col << " f=" << f;
                    if (exists) {
                        const float actual = roundTrip ? packedValue(token, f) : directValue(row, col, f);
                        EXPECT_NEAR(activeOutput[token * 2 + f], actual, 4e-4f);
                        EXPECT_NEAR(activeLoss[token * 2 + f],
                                    upstream1P(token, f) * upstream1P(token, f), 4e-4f);
                        if (roundTrip) EXPECT_NEAR(sourceGrads[token * 2 + f], expected, 7e-3f);
                    }
                }
        EXPECT_EQ(n.architectureJson(), before);
    }
    filesystem::remove_all(archive);
}
}  // namespace

TEST(PaddedDenseToRaggedNeutral1P, SemanticShapeAndMemoryAccounting) {
    Network n("adapter_1p_schema");
    const auto source = partition1P(n);
    const Tensor dense = NetworkInput::Builder().network(n).name("dense")
        .dataType(DataType::FP32).dimensions({6, 2}).build().getFeatureOutput().value();
    const auto layer = PaddedDenseToRagged::Builder().network(n)
        .featureInput(dense).partitionInput(source).build();
    const auto output = layer.getRaggedFeatureOutput();
    EXPECT_TRUE(output.sharesPartitionWith(source));
    EXPECT_EQ(output.getRowPartitionToken(), source.getRowPartitionToken());
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{4, 2}));
    EXPECT_NE(output.getValues(), source.getValues());
    ASSERT_EQ(layer.getFeatureInputs().size(), 2U);
    EXPECT_EQ(layer.getFeatureInputs()[0], dense);
    EXPECT_EQ(layer.getFeatureInputs()[1], source.getRowPartitionToken());
    EXPECT_EQ(layer.getRaggedPartitionRequirementForInput(source.getRowPartitionToken()),
              Impl::RaggedPartitionRequirement::DEVICE_OFFSETS);
    const json j = layer.architectureJson();
    EXPECT_EQ(j.at("version"), "1.1.0");
    EXPECT_FALSE(j.at("partition_input").contains("geometry_mode"));
    EXPECT_FALSE(j.at("ragged_feature_output").contains("geometry_mode"));
    EXPECT_FALSE(j.at("ragged_feature_output").contains("batch_size"));
    EXPECT_FALSE(j.at("ragged_feature_output").contains("max_total_values"));
    EXPECT_EQ(j.at("ragged_feature_output").at("offsets"), j.at("partition_input").at("offsets"));
    EXPECT_NO_THROW(PaddedDenseToRagged::deserialize(j, &n));
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        const auto bytes = static_cast<uint64_t>(b) * 4 * 2 * sizeof(float);
        EXPECT_EQ(layer.getOutputTensorBytes(b), bytes);
        EXPECT_EQ(layer.getFirstInstanceMemRequirementInBytes(b, cpu), bytes);
    }
    EXPECT_ANY_THROW((void)layer.getOutputTensorBytes(0));
    Network fixed("adapter_1p_fixed");
    const auto old = RaggedNetworkInput::Builder().network(fixed).name("fixed")
        .valuesDataType(DataType::FP32).trailingDimensions({2})
        .maxValuesPerRow(4).build();
    const auto oldDense = NetworkInput::Builder().network(fixed).name("padded")
        .dataType(DataType::FP32).dimensions({4, 2}).build().getFeatureOutput().value();
    const auto legacy = PaddedDenseToRagged::Builder().network(fixed)
        .featureInput(oldDense).partitionInput(old).build();
    EXPECT_EQ(legacy.architectureJson().at("version"), "1.1.0");
    EXPECT_EQ(legacy.getRaggedFeatureOutput().getValuesDimensions(), (vector<uint64_t>{4, 2}));
    EXPECT_EQ(legacy.getOutputTensorBytes(7), 7U * 4U * 2U * sizeof(float));
    EXPECT_NO_THROW(PaddedDenseToRagged::deserialize(legacy.architectureJson(), &fixed));

    // Exercise the actual historical packed-reference grammar separately from
    // the canonical builder: archived B and capacity must not determine M.
    json historical = legacy.architectureJson();
    historical["version"] = "1.0.0";
    for (const char* field : {"partition_input", "ragged_feature_output"}) {
        auto& ref = historical[field];
        ref.erase("layer_ragged_reference_version");
        ref.erase("max_values_per_row");
        ref["batch_size"] = 2U;
        ref["max_total_values"] = 8U;
        ref["values"]["dimensions"][0] = 8U;
        ref["offsets"]["dimensions"][0] = 3U;
    }
    EXPECT_NO_THROW(PaddedDenseToRagged::deserialize(historical, &fixed));
}

TEST(PaddedDenseToRaggedNeutral1P, RejectsMalformedReferencesAndNarrowDense) {
    Network n("adapter_1p_bad");
    const auto source = partition1P(n);
    EXPECT_ANY_THROW((void)PaddedDenseToRagged::Builder().network(n)
        .featureInput(Tensor(DataType::FP32, {3, 2})).partitionInput(source).build());
    const Tensor dense = NetworkInput::Builder().network(n).name("padded")
        .dataType(DataType::FP32).dimensions({6, 2}).build().getFeatureOutput().value();
    const auto layer = PaddedDenseToRagged::Builder().network(n)
        .featureInput(dense).partitionInput(source).build();
    const json j = layer.architectureJson();
    json bad = j; bad["version"] = "2.0.0"; reject1P(n, bad);
    bad = j; bad["batch_size"] = 7; reject1P(n, bad);
    bad = j; bad["feature_input"]["dimensions"] = vector<uint64_t>{3, 2}; reject1P(n, bad);
    bad = j; bad["feature_input"]["id"] = source.getValues().getOriginalId(); reject1P(n, bad);
    bad = j; bad["feature_input"]["data_type"] = DataType::FP16; reject1P(n, bad);
    bad = j; bad["partition_input"]["values"]["id"] = source.getOffsets().getOriginalId(); reject1P(n, bad);
    bad = j; bad["partition_input"]["offsets"]["id"] = source.getValues().getOriginalId(); reject1P(n, bad);
    bad = j; bad["partition_input"]["offsets"]["dimensions"] = vector<uint64_t>{3}; reject1P(n, bad);
    bad = j; bad["partition_input"]["max_values_per_row"] = 5; reject1P(n, bad);
    bad = j; bad["partition_input"]["batch_size"] = 2; reject1P(n, bad);
    bad = j; bad["partition_input"]["max_total_values"] = 8; reject1P(n, bad);
    bad = j; bad["partition_input"]["extra"] = 8; reject1P(n, bad);
    bad = j; bad["ragged_feature_output"]["offsets"]["id"] = source.getValues().getOriginalId(); reject1P(n, bad);
    bad = j; bad["ragged_feature_output"]["offsets"]["data_type"] = DataType::UINT64; reject1P(n, bad);
    bad = j; bad["ragged_feature_output"]["values"]["dimensions"] = vector<uint64_t>{8, 2}; reject1P(n, bad);
    bad = j; bad["ragged_feature_output"]["values"]["data_type"] = DataType::BF16; reject1P(n, bad);
    bad = j; bad["ragged_feature_output"]["values"]["id"] = source.getValues().getOriginalId(); reject1P(n, bad);
    bad = j; bad["ragged_feature_output"]["max_values_per_row"] = 3; reject1P(n, bad);
    bad = j; bad["ragged_feature_output"]["batch_size"] = 7; reject1P(n, bad);
    bad = j; bad["ragged_feature_output"]["extra"] = 3; reject1P(n, bad);
    bad = j; bad["factory"] = "broken"; reject1P(n, bad);
}

TEST(PaddedDenseToRaggedNeutral1P, DenseAndOffsetsCanIndependentlyDriveStamp) {
    Network n("adapter_1p_drivers");
    const auto source = partition1P(n);
    const auto layer = PaddedDenseToRagged::Builder().network(n)
        .featureInput(Tensor(DataType::FP32, {6, 2})).partitionInput(source).build();
    Stamp1P probe(layer);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        auto dense = make_shared<Impl::NetworkInput>(cpu, DataType::FP32, vector<uint64_t>{b, 6, 2});
        ASSERT_NE(dynamic_pointer_cast<Impl::PaddedDenseToRagged>(
            probe.stamp(cpu, dense, nullptr, layer.getDenseFeatureInput(), true)), nullptr);
        auto offsets = make_shared<Impl::NetworkInput>(cpu, DataType::UINT32, vector<uint64_t>{b + 1U});
        ASSERT_NE(dynamic_pointer_cast<Impl::PaddedDenseToRagged>(
            probe.stamp(cpu, offsets, nullptr, source.getOffsets(), true)), nullptr);
        auto invalidDense = make_shared<Impl::NetworkInput>(cpu, DataType::FP32, vector<uint64_t>{b, 3, 2});
        EXPECT_ANY_THROW((void)probe.stamp(cpu, invalidDense, nullptr, layer.getDenseFeatureInput(), true));
        auto invalidOffsets = make_shared<Impl::NetworkInput>(cpu, DataType::UINT32, vector<uint64_t>{1});
        EXPECT_ANY_THROW((void)probe.stamp(cpu, invalidOffsets, nullptr, source.getOffsets(), true));
    }
}

TEST(PaddedDenseToRaggedNeutral1P, SavedDirectPackingAndBackwardAtTwoAndSeven) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    savedExecution1P(false);
}

TEST(PaddedDenseToRaggedNeutral1P, SavedRoundTripAndBackwardAtTwoAndSeven) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP() << "CUDA device unavailable";
    savedExecution1P(true);
}

TEST(PaddedDenseToRaggedNeutral1P, NeutralRoundTripSubgraphCloneRetainsCanonicalPartition) {
    Network original("adapter_1p_clone_source");
    const auto source = partition1P(original);
    const Tensor padded = RaggedToPaddedDense::Builder().network(original)
        .featureInput(source).build().getPaddedFeatureOutput();
    const RaggedTensor result = PaddedDenseToRagged::Builder().network(original)
        .featureInput(padded).partitionInput(source).build().getRaggedFeatureOutput();
    (void)RaggedNetworkOutput::Builder().network(original).name("restored").inputTensor(result).build();

    Network destination("adapter_1p_clone_destination");
    const auto replacement = partition1P(destination);
    ApiTensorRemap remap;
    remap.map(source.getValues(), replacement.getValues());
    remap.map(source.getRowPartitionToken(), replacement.getRowPartitionToken());
    ApiSubgraphCloneOptions options;
    options.inferenceOnly = true;
    const auto cloned = destination.cloneSubgraphInto(
        original, {"__thor_ragged_output.restored.values"}, remap, options);
    ASSERT_EQ(cloned.outputTensorsByName.size(), 1U);
    EXPECT_EQ(cloned.outputTensorsByName.at("__thor_ragged_output.restored.values").getDimensions(),
              (vector<uint64_t>{4, 2}));
    EXPECT_EQ(cloned.clonedTensorBySourceOriginalId.at(result.getRowPartitionToken().getOriginalId()),
              replacement.getRowPartitionToken());
}
