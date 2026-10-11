#include "DeepLearning/Api/Layers/Utility/Concatenate.h"
#include "DeepLearning/Api/Layers/Loss/RaggedCustomLoss.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Layers/Loss/RaggedCustomLoss.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/Utility/RaggedConcatenate.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/Event.h"
#include "cuda_runtime.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <limits>
#include <string>
#include <tuple>
#include <variant>
#include <utility>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Impl = ThorImplementation;

namespace {
RaggedTensor owner(Network& n, vector<uint64_t> trailing = {2, 3}) {
    return RaggedNetworkInput::Builder().network(n).name("left")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions(trailing).maxValuesPerRow(4).build();
}
RaggedTensor sibling(Network& n, const RaggedTensor& source,
                     string name = "right", vector<uint64_t> trailing = {2, 5}) {
    return RaggedNetworkInput::Builder().network(n).name(name)
        .partition(source).valuesDataType(DataType::FP32).trailingDimensions(trailing).build();
}
Concatenate joined(Network& n, const RaggedTensor& first, const RaggedTensor& second,
                   uint32_t axis) {
    return Concatenate::Builder().network(n).featureInput(first)
        .featureInput(second).concatenationAxis(axis).build();
}
void reject(Network& n, const json& j) {
    EXPECT_ANY_THROW(Concatenate::deserialize(j, &n));
}
} // namespace

TEST(RaggedConcatenateNeutral1N, TrailingAxesAndCanonicalPartition) {
    Network n("concat_neutral_builder");
    RaggedTensor a = owner(n);
    RaggedTensor b = sibling(n, a);
    const auto layer = joined(n, a, b, 1);
    const auto output = layer.getRaggedFeatureOutput();
    ASSERT_TRUE(output.has_value());
    EXPECT_TRUE(output->sharesPartitionWith(a));
    EXPECT_EQ(output->getRowPartitionToken(), a.getRowPartitionToken());
    EXPECT_EQ(output->getValuesDimensions(), (vector<uint64_t>{4, 2, 8}));
    EXPECT_EQ(output->getDescriptorForPlacement(2).getMaxTotalValues(), 8U);
    EXPECT_EQ(output->getDescriptorForPlacement(7).getMaxTotalValues(), 28U);
    ASSERT_EQ(layer.getFeatureInputs().size(), 3U);
    EXPECT_NE(layer.getFeatureInputs()[0], layer.getFeatureInputs()[1]);
    EXPECT_EQ(layer.getFeatureInputs()[2], a.getRowPartitionToken());
    EXPECT_EQ(layer.getRaggedPartitionRequirementForInput(a.getRowPartitionToken()),
              Impl::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);

    RaggedTensor c = sibling(n, a, "third", {5, 3});
    auto axis0 = joined(n, a, c, 0);
    ASSERT_TRUE(axis0.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(axis0.getRaggedFeatureOutput()->getValuesDimensions(),
              (vector<uint64_t>{4, 7, 3}));
    RaggedTensor fourth = sibling(n, a, "fourth", {2, 1});
    auto three = Concatenate::Builder().network(n).featureInput(a)
        .featureInput(b).featureInput(fourth).concatenationAxis(1).build();
    EXPECT_EQ(three.getFeatureInputs().size(), 4U); // N values plus one structural carrier
    EXPECT_EQ(three.getRaggedFeatureOutput()->getValuesDimensions(),
              (vector<uint64_t>{4, 2, 9}));
    EXPECT_EQ(three.architectureJson().at("ragged_inputs").size(), 3U);

    const json j = layer.architectureJson();
    EXPECT_EQ(j.at("version"), "1.1.0");
    EXPECT_EQ(j.at("concatenation_axis"), 1);
    EXPECT_EQ(j.at("ragged_inputs").size(), 2U);
    for (const auto& r : j.at("ragged_inputs")) {
        EXPECT_FALSE(r.contains("geometry_mode"));
        EXPECT_EQ(r.at("max_values_per_row"), 4);
        EXPECT_EQ(r.at("offsets").at("dimensions"), (vector<uint64_t>{1}));
        EXPECT_FALSE(r.contains("batch_size"));
        EXPECT_FALSE(r.contains("max_total_values"));
    }
    EXPECT_EQ(j.at("ragged_output").at("values"), j.at("outputs").at(0));
    EXPECT_EQ(j.at("ragged_output").at("offsets"), j.at("ragged_inputs").at(0).at("offsets"));
    EXPECT_NO_THROW(Concatenate::deserialize(j, &n));
}

TEST(RaggedConcatenateNeutral1N, RejectsIncompatibleModesAndInputs) {
    Network n("concat_neutral_reject");
    auto a = owner(n);
    auto b = sibling(n, a);
    auto tryJoin = [&](const RaggedTensor& lhs, const RaggedTensor& rhs, uint32_t axis = 1) {
        EXPECT_ANY_THROW((void)joined(n, lhs, rhs, axis));
    };
    tryJoin(a, a);
    tryJoin(a, a.withValues(Tensor(DataType::BF16, {4, 2, 5})));
    tryJoin(a, a.withValues(Tensor(DataType::FP32, {4, 3, 5})));
    tryJoin(a, a.withValues(a.getValues())); // separate wrapper, identical values tensor identity
    EXPECT_NO_THROW((void)joined(n, a, a.withValues(Tensor(DataType::FP32, {4, 2, 3})), 1));
    // An independently allocated tensor is a valid distinct input, even with identical shape.
    tryJoin(a, RaggedTensor::logical(Tensor(DataType::FP32, {5, 2, 5}),
                                               a.getRowPartitionToken(), 5));
    tryJoin(a, RaggedTensor::logical(Tensor(DataType::FP32, {4, 2, 5}),
                                               Tensor(DataType::UINT32, {1}), 4));
    tryJoin(a, RaggedTensor::logical(Tensor(DataType::FP32, {4, 2, 5}),
                                               Tensor(DataType::UINT64, {1}), 4));
    tryJoin(a, a.withValues(Tensor(DataType::FP32, {4, 3}))); // different rank
    tryJoin(a, b, 2);
    // A packed [B*M,...] values shape and [B+1] offsets cannot be constructed
    // as a logical RaggedTensor. Reject it at the logical boundary.
    EXPECT_ANY_THROW((void)RaggedTensor(Tensor(DataType::FP32, {8, 2, 5}),
                                        Tensor(DataType::UINT32, {3}), 4));
    EXPECT_ANY_THROW((void)Concatenate::Builder().network(n).featureInput(a)
                         .featureInput(Tensor(DataType::FP32, {4, 2, 5}))
                         .concatenationAxis(1).build());
    // Valid heterogenous widths are deliberately not rejected.
    EXPECT_NO_THROW((void)joined(n, a, b, 1));
}

TEST(RaggedConcatenateNeutral1N, StrictOneOneReferencesAndLegacyCompatibility) {
    Network n("concat_neutral_schema");
    auto a = owner(n);
    auto b = sibling(n, a);
    auto layer = joined(n, a, b, 1);
    const json j = layer.architectureJson();
    json bad = j;
    bad["version"] = "1.0.0"; reject(n, bad);
    bad = j; bad["version"] = "8.0.0"; reject(n, bad);
    bad = j; bad["use_ragged"] = false; reject(n, bad);
    bad = j; bad["ragged_inputs"][1]["values"]["id"] = a.getValues().getOriginalId(); reject(n, bad);
    bad = j; bad["ragged_inputs"][1]["values"]["dimensions"] = vector<uint64_t>{4, 2, 7}; reject(n, bad);
    bad = j; bad["ragged_inputs"][1]["values"]["data_type"] = DataType::BF16; reject(n, bad);
    bad = j; bad["ragged_inputs"][1]["offsets"]["dimensions"] = vector<uint64_t>{2}; reject(n, bad);
    bad = j; bad["ragged_inputs"][1]["offsets"]["id"] = b.getValues().getOriginalId(); reject(n, bad);
    bad = j; bad["ragged_inputs"][1]["max_values_per_row"] = 5; reject(n, bad);
    bad = j; bad["ragged_inputs"][0]["batch_size"] = 2; reject(n, bad);
    bad = j; bad["ragged_inputs"][0]["max_total_values"] = 8; reject(n, bad);
    bad = j; bad["ragged_inputs"][0]["geometry_mode"] = "fixed"; reject(n, bad);
    bad = j; bad["ragged_inputs"][1]["layer_ragged_reference_version"] = "2.0.0"; reject(n, bad);
    bad = j; bad["ragged_output"]["offsets"]["id"] = b.getValues().getOriginalId(); reject(n, bad);
    bad = j; bad["ragged_output"]["offsets"]["data_type"] = DataType::UINT64; reject(n, bad);
    bad = j; bad["ragged_output"]["values"]["id"] = b.getValues().getOriginalId(); reject(n, bad);
    bad = j; bad["ragged_output"]["values"]["dimensions"] = vector<uint64_t>{4, 2, 9}; reject(n, bad);
    bad = j; bad["ragged_output"]["max_values_per_row"] = 3; reject(n, bad);
    bad = j; bad["outputs"][0]["dimensions"] = vector<uint64_t>{4, 2, 9};
    bad["ragged_output"]["values"]["dimensions"] = vector<uint64_t>{4, 2, 9}; reject(n, bad);
    bad = j; bad["outputs"][0]["data_type"] = DataType::BF16;
    bad["ragged_output"]["values"]["data_type"] = DataType::BF16; reject(n, bad);
    bad = j; bad["concatenation_axis"] = 0; reject(n, bad); // other trailing axis cannot sum to output
    bad = j; bad["ragged_inputs"].push_back(j["ragged_inputs"][0]); reject(n, bad);
    bad = j; bad["inputs"] = json::array(); reject(n, bad);
    bad = j; bad["ragged_output"]["foo"] = 1; reject(n, bad);
    bad = j; bad["ragged_output"]["geometry_mode"] = "fixed"; reject(n, bad);
    bad = j; bad["ragged_output"]["batch_size"] = 2; reject(n, bad);

    auto historical = [](json ref) {
        ref.erase("layer_ragged_reference_version");
        ref.erase("geometry_mode");
        ref["batch_size"] = 91;      // obsolete execution hint, not semantic M
        ref["max_total_values"] = 999;
        ref["values"]["dimensions"][0] = 123;
        ref["offsets"]["dimensions"] = vector<uint64_t>{92};
        return ref;
    };
    json old = j;
    old["version"] = "1.0.0";
    for (auto& input : old["ragged_inputs"]) input = historical(input);
    old["ragged_output"] = historical(old["ragged_output"]);
    old["outputs"][0]["dimensions"][0] = 123;
    ASSERT_NO_THROW(Concatenate::deserialize(old, &n));
    auto recovered = dynamic_pointer_cast<Concatenate>(n.getLayer(n.getNumLayers() - 1));
    ASSERT_NE(recovered, nullptr);
    ASSERT_TRUE(recovered->getRaggedFeatureOutput().has_value());
    for (uint32_t b : {2u, 7u}) {
        EXPECT_EQ(recovered->getRaggedFeatureOutput()->getDescriptorForPlacement(b).getMaxTotalValues(), b * 4u);
    }
    bad = old; bad["version"] = "1.1.0"; reject(n, bad);
    bad = old; bad["ragged_inputs"][1]["max_values_per_row"] = 8; reject(n, bad);
    bad = old; bad["ragged_output"]["offsets"]["id"] = b.getValues().getOriginalId(); reject(n, bad);

    Network denseNetwork("concat_dense_compat");
    auto d1 = NetworkInput::Builder().network(denseNetwork).name("x")
        .dimensions({2}).dataType(DataType::FP32).build();
    auto d2 = NetworkInput::Builder().network(denseNetwork).name("y")
        .dimensions({3}).dataType(DataType::FP32).build();
    auto dense = Concatenate::Builder().network(denseNetwork)
        .featureInput(d1.getFeatureOutput().value()).featureInput(d2.getFeatureOutput().value())
        .concatenationAxis(0).build();
    const json dj = dense.architectureJson();
    EXPECT_EQ(dj.at("version"), "1.0.0");
    EXPECT_NO_THROW(Concatenate::deserialize(dj, &denseNetwork));
    bad = dj; bad["version"] = "1.1.0"; reject(denseNetwork, bad);
}

TEST(RaggedConcatenateNeutral1N, MemoryAccountingForTwoAndSeven) {
    Network n("concat_neutral_memory");
    auto a = owner(n);
    auto b = sibling(n, a);
    const auto layer = joined(n, a, b, 1);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t B : {2U, 7U}) {
        const uint64_t left = B * 4U * 2U * 3U * sizeof(float);
        const uint64_t right = B * 4U * 2U * 5U * sizeof(float);
        const uint64_t output = B * 4U * 2U * 8U * sizeof(float);
        EXPECT_EQ(layer.getOutputTensorBytes(B), output);
        EXPECT_EQ(static_cast<const Layer&>(layer).getFirstInstanceMemRequirementInBytes(B, cpu),
                  left + right + output + sizeof(uint32_t));
    }
    // A logical input with M=4 gets B*M physical capacity at placement, not at construction.
    Network fixedN("concat_fixed_memory");
    auto fixedA = RaggedNetworkInput::Builder().network(fixedN).name("fixed_left")
        .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(4).build();
    auto fixedB = RaggedNetworkInput::Builder().network(fixedN).name("fixed_right")
        .valuesDataType(DataType::FP32).trailingDimensions({3}).partition(fixedA).build();
    const auto fixed = joined(fixedN, fixedA, fixedB, 0);
    EXPECT_EQ(fixed.getOutputTensorBytes(2), 8U * 5U * sizeof(float));
    EXPECT_EQ(static_cast<const Layer&>(fixed).getFirstInstanceMemRequirementInBytes(2, cpu),
              8U * (2U + 3U + 5U) * sizeof(float) + sizeof(uint32_t));
    Network denseN("concat_dense_memory");
    auto dense = Concatenate::Builder().network(denseN)
        .featureInput(Tensor(DataType::FP32, {2})).featureInput(Tensor(DataType::FP32, {3}))
        .concatenationAxis(0).build();
    EXPECT_EQ(dense.getOutputTensorBytes(7), 7U * 5U * sizeof(float));
    EXPECT_EQ(static_cast<const Layer&>(dense).getFirstInstanceMemRequirementInBytes(7, cpu),
              7U * (2U + 5U) * sizeof(float));
}

TEST(RaggedConcatenateNeutral1N, SavedArchitectureReconstructsUnplaced) {
    const auto dir = filesystem::temp_directory_path() /
        ("concat_neutral_1n1_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(dir);
    json original;
    uint64_t outputOriginalId = 0;
    {
        Network n("concatenate_1n_unplaced");
        auto a = owner(n);
        auto b = sibling(n, a);
        auto concat = joined(n, a, b, 1);
        auto out = concat.getRaggedFeatureOutput();
        ASSERT_TRUE(out.has_value());
        (void)RaggedNetworkOutput::Builder().network(n).name("joined").inputTensor(out.value()).build();
        original = concat.architectureJson();
        outputOriginalId = out->getValues().getOriginalId();
        n.save(dir.string(), true); // no place(), no GPU execution
    }
    {
        Network loaded("concatenate_1n_unplaced");
        ASSERT_NO_THROW(loaded.load(dir.string()));
        const Concatenate* concat = nullptr;
        for (uint32_t i = 0; i < loaded.getNumLayers(); ++i) {
            auto candidate = dynamic_pointer_cast<Concatenate>(loaded.getLayer(i));
            if (candidate) concat = candidate.get();
        }
        ASSERT_NE(concat, nullptr);
        ASSERT_TRUE(concat->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(concat->getRaggedFeatureOutput()->getValuesDimensions(),
                  (vector<uint64_t>{4, 2, 8}));
        EXPECT_EQ(concat->getRaggedFeatureOutput()->getValues().getOriginalId(), outputOriginalId);
        const auto inputs = concat->getFeatureInputs();
        EXPECT_NE(inputs[0], inputs[1]);
        EXPECT_EQ(inputs[2], concat->getRaggedFeatureOutput()->getRowPartitionToken());
        json reconstructed = concat->architectureJson();
        EXPECT_EQ(reconstructed.at("version"), "1.1.0");
        EXPECT_EQ(reconstructed.at("ragged_inputs").size(), original.at("ragged_inputs").size());
        EXPECT_EQ(reconstructed.at("ragged_output").at("values").at("dimensions"),
                  original.at("ragged_output").at("values").at("dimensions"));
        // Live IDs intentionally differ from saved original IDs on rehydration.
        EXPECT_NE(reconstructed.at("outputs").at(0).at("id"), original.at("outputs").at(0).at("id"));
    }
    filesystem::remove_all(dir);
}

// The neutral artifact has no physical batch geometry. One saved 3-input
// architecture must independently place and forward at two distinct B values.
TEST(RaggedConcatenateNeutral1N, SavedThreeInputForwardAtTwoAndSeven) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";

    const filesystem::path archive = filesystem::temp_directory_path() /
        ("concat_neutral_1n2_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    uint64_t leftId = 0, rightId = 0, thirdId = 0, outputId = 0, partitionId = 0;
    {
        Network n("concatenate_1n2_forward");
        const RaggedTensor left = owner(n, {2, 3});
        const RaggedTensor right = sibling(n, left, "right", {1, 3});
        const RaggedTensor third = sibling(n, left, "third", {3, 3});
        const auto layer = Concatenate::Builder().network(n)
            .featureInput(left).featureInput(right).featureInput(third)
            .concatenationAxis(0).build();
        ASSERT_TRUE(layer.getRaggedFeatureOutput().has_value());
        EXPECT_EQ(layer.architectureJson().at("version"), "1.1.0");
        EXPECT_EQ(layer.getRaggedFeatureOutput()->getValuesDimensions(),
                  (vector<uint64_t>{4, 6, 3}));
        const auto ports = layer.getFeatureInputs();
        ASSERT_EQ(ports.size(), 4U);
        leftId = ports[0].getOriginalId();
        rightId = ports[1].getOriginalId();
        thirdId = ports[2].getOriginalId();
        outputId = layer.getFeatureOutputs().at(0).getOriginalId();
        partitionId = left.getRowPartitionToken().getOriginalId();
        EXPECT_EQ(ports[3].getOriginalId(), partitionId);
        (void)RaggedNetworkOutput::Builder().network(n).name("joined")
            .inputTensor(layer.getRaggedFeatureOutput().value()).build();
        n.save(archive.string(), true);
    }

    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "placement B=" << b);
        Network n("concatenate_1n2_forward");
        ASSERT_NO_THROW(n.load(archive.string()));
        const auto external = n.getExternalRaggedNetworkInputs();
        ASSERT_EQ(external.size(), 3U);
        auto findInput = [&](const string& name) -> RaggedTensor {
            for (const auto& item : external)
                if (item.name == name) return item.raggedTensor;
            ADD_FAILURE() << "Missing saved ragged input: " << name;
            return external.front().raggedTensor;
        };
        const auto left = findInput("left");
        const auto right = findInput("right");
        const auto third = findInput("third");
        EXPECT_TRUE(left.sharesPartitionWith(right));
        EXPECT_TRUE(left.sharesPartitionWith(third));
        EXPECT_EQ(left.getRowPartitionToken().getOriginalId(), partitionId);
        const Concatenate* layer = nullptr;
        for (uint32_t i = 0; i < n.getNumLayers(); ++i) {
            const auto candidate = dynamic_pointer_cast<Concatenate>(n.getLayer(i));
            if (candidate) layer = candidate.get();
        }
        ASSERT_NE(layer, nullptr);
        ASSERT_EQ(layer->getFeatureInputs().size(), 4U);
        EXPECT_EQ(layer->getFeatureInputs()[0].getOriginalId(), leftId);
        EXPECT_EQ(layer->getFeatureInputs()[1].getOriginalId(), rightId);
        EXPECT_EQ(layer->getFeatureInputs()[2].getOriginalId(), thirdId);
        EXPECT_EQ(layer->getFeatureInputs()[3].getOriginalId(), partitionId);
        EXPECT_EQ(layer->getFeatureOutputs()[0].getOriginalId(), outputId);
        const json architectureBefore = n.architectureJson();
        vector<Event> initEvents;
        const auto placed = n.place(b, initEvents, true);
        ASSERT_NE(placed, nullptr);
        for (Event& event : initEvents) event.synchronize();
        auto& stamp = placed->getStampedNetwork(0);
        EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(left.getRowPartitionId()), nullptr);
        const auto activeCount = stamp.getManagedPartitionActiveCountInputForTest(left.getRowPartitionId());
        ASSERT_NE(activeCount, nullptr);
        ASSERT_TRUE(activeCount->getFeatureOutput().has_value());
        EXPECT_EQ(activeCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(activeCount->getFeatureOutput()->getDataType(), DataType::UINT32);
        for (const auto& [name, width] : vector<pair<string, uint32_t>>{
                 {"left", 2}, {"right", 1}, {"third", 3}}) {
            auto physicalInput = stamp.getNamedInput(name + ".values");
            ASSERT_NE(physicalInput, nullptr);
            ASSERT_TRUE(physicalInput->getFeatureOutput().has_value());
            EXPECT_EQ(physicalInput->getFeatureOutput()->getDimensions(),
                      (vector<uint64_t>{b * 4U, width, 3}));
        }
        auto physical = dynamic_pointer_cast<Impl::RaggedConcatenate>(
            stamp.getPhysicalLayerFromApiLayer(layer->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_EQ(physical->getFeatureInputs().size(), 4U);
        for (size_t port = 0; port < 3; ++port)
            ASSERT_TRUE(physical->getFeatureInputs()[port].has_value());
        ASSERT_TRUE(physical->getFeatureInputs()[3].has_value());
        EXPECT_EQ(physical->getFeatureInputs()[3]->getDimensions(), (vector<uint64_t>{1}));
        ASSERT_TRUE(physical->getFeatureOutput().has_value());
        EXPECT_EQ(physical->getFeatureOutput()->getDimensions(), (vector<uint64_t>{b * 4U, 6, 3}));
        EXPECT_EQ(n.architectureJson(), architectureBefore);

        vector<uint64_t> offsets(b + 1, 0);
        for (uint32_t i = 1; i <= b; ++i)
            offsets[i] = offsets[i - 1] + (i == 1 || i % 3 == 0 ? 0 : 1 + i % 2);
        const uint64_t active = offsets.back();
        ASSERT_LT(active, b * 4U);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor offsetData(cpu, left.getDescriptorForPlacement(b).getOffsetsDescriptor());
        for (uint32_t i = 0; i <= b; ++i)
            offsetData.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(offsets[i]);
        Impl::RowPartitionRuntime partition(offsetData,
            left.getDescriptorForPlacement(b).getRowPartition());
        partition.setHostOffsets(offsets);
        Batch batch;
        for (const auto& [name, logical, width, tag] :
             vector<tuple<string, RaggedTensor, uint32_t, uint32_t>>{
                 {"left", left, 2, 1}, {"right", right, 1, 2}, {"third", third, 3, 3}}) {
            const auto desc = logical.getDescriptorForPlacement(b);
            Impl::Tensor data(cpu, desc.getValuesDescriptor());
            float* values = data.getMemPtr<float>();
            for (uint64_t t = 0; t < b * 4U; ++t)
                for (uint32_t row = 0; row < width; ++row)
                    for (uint32_t c = 0; c < 3; ++c)
                        values[t * width * 3 + row * 3 + c] = t < active
                            ? static_cast<float>(100 * tag + 10 * row + c) + static_cast<float>(t) * 0.25f
                            : numeric_limits<float>::quiet_NaN();
            batch.insert(name, Impl::RaggedTensor(data, partition));
        }
        const auto outputs = placed->inferLogical(batch);
        ASSERT_TRUE(outputs.contains("joined"));
        ASSERT_TRUE(holds_alternative<Impl::RaggedTensor>(outputs.at("joined")));
        const auto output = get<Impl::RaggedTensor>(outputs.at("joined"));
        const auto hostOffsets = output.getHostOffsetsIfAvailable();
        ASSERT_TRUE(hostOffsets.has_value());
        EXPECT_EQ(hostOffsets.value(), offsets);
        EXPECT_EQ(output.getValues().getDimensions(), (vector<uint64_t>{b * 4U, 6, 3}));
        const float* data = output.getValues().getMemPtr<float>();
        for (uint64_t t = 0; t < active; ++t) {
            uint32_t outputRow = 0;
            for (const auto& [rows, tag] : vector<pair<uint32_t, uint32_t>>{{2, 1}, {1, 2}, {3, 3}})
                for (uint32_t row = 0; row < rows; ++row, ++outputRow)
                    for (uint32_t c = 0; c < 3; ++c)
                        EXPECT_NEAR(data[t * 18 + outputRow * 3 + c],
                            static_cast<float>(100 * tag + 10 * row + c) + static_cast<float>(t) * 0.25f,
                            1e-5f);
        }
    }
    filesystem::remove_all(archive);
}


// Probe the existing protected stamp boundary without changing the API or
// introducing a second placement context. The physical drivers are ordinary
// already-materialized descriptors and the existing managed fanout metadata.
namespace {
class ConcatenateStampProbe : public Concatenate {
public:
    explicit ConcatenateStampProbe(const Concatenate& layer) : Concatenate(layer) {}
    using Concatenate::stamp;
};
} // namespace

TEST(RaggedConcatenateNeutral1N, StampsFromEachValueAndManagedPartitionCarrier) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";
    Network n("concat_neutral_stamp_drivers");
    const RaggedTensor first = owner(n, {2, 3});
    const RaggedTensor second = sibling(n, first, "right", {1, 3});
    const RaggedTensor third = sibling(n, first, "third", {3, 3});
    const Concatenate logical = Concatenate::Builder().network(n).featureInput(first)
        .featureInput(second).featureInput(third).concatenationAxis(0).build();
    ConcatenateStampProbe probe(logical);
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(b);
        // Whichever values operand is the first traversed connection must
        // recover the same B from *its own* physical trailing shape.
        for (const RaggedTensor& input : {first, second, third}) {
            const auto placed = input.getDescriptorForPlacement(b);
            auto driver = make_shared<Impl::NetworkInput>(cpu, input.getValuesDataType(),
                                                          placed.getValuesDescriptor().getDimensions());
            const auto stamped = dynamic_pointer_cast<Impl::RaggedConcatenate>(
                probe.stamp(cpu, driver, nullptr, input.getValues(), true));
            ASSERT_NE(stamped, nullptr);
            EXPECT_EQ(stamped->getType(), "RaggedConcatenate");
            // The validation must not merely divide any unrelated capacity by M.
            auto wrong = make_shared<Impl::NetworkInput>(cpu, input.getValuesDataType(),
                vector<uint64_t>{b * 4U, 99, 3});
            EXPECT_ANY_THROW((void)probe.stamp(cpu, wrong, nullptr, input.getValues(), true));
        }
        // The structural driver is a managed DEVICE_ACTIVE_COUNT fanout.
        // Its [1] carrier has no B encoded in its physical TensorDescriptor;
        // the authoritative runtime RowPartitionDescriptor supplies B.
        auto carrier = make_shared<Impl::TensorFanout>();
        Impl::Tensor scalar(cpu, Impl::TensorDescriptor(DataType::UINT32, {1}));
        Stream stream(0);
        (void)carrier->connectToPreviousLayer(nullptr, scalar, stream, false);
        carrier->setRowPartitionDescriptorForStamp(first.getDescriptorForPlacement(b).getRowPartition());
        const auto stamped = dynamic_pointer_cast<Impl::RaggedConcatenate>(
            probe.stamp(cpu, carrier, nullptr, first.getRowPartitionToken(), true));
        ASSERT_NE(stamped, nullptr);
        EXPECT_EQ(stamped->getType(), "RaggedConcatenate");
    }
}


// Compare every differentiable values port with its own slice of the RAW
// squared-error derivative. No inactive-capacity gradient is ever read.
namespace {
Impl::DynamicExpression concatenateSquaredError(bool gradient) {
    const auto predictions = Impl::Expression::input("predictions", DataType::FP32, DataType::FP32);
    const auto labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    const auto difference = predictions - labels;
    auto result = gradient
        ? difference * Impl::Expression(2.0 * Impl::Loss::getLossScalingFactor())
        : difference * difference;
    result = result.withOutputDType(DataType::FP32);
    return Impl::DynamicExpression::fromExpressionDefinition(
        Impl::ExpressionDefinition::fromOutputs(
            Impl::Expression::outputs({{gradient ? "predictions_grad" : "loss", result}})));
}

float concatPortValue(uint64_t token, uint32_t port, uint32_t row, uint32_t column) {
    return 0.16f * static_cast<float>(token) + 0.41f * static_cast<float>(port + 1)
        + 0.13f * static_cast<float>(row) - 0.23f * static_cast<float>(column);
}

float concatUpstreamDifference(uint64_t token, uint32_t row, uint32_t column) {
    const float magnitude = 0.11f + 0.07f * static_cast<float>((token + 2 * row) % 5)
        + 0.035f * static_cast<float>(row) + 0.055f * static_cast<float>(column);
    return ((token + row + column) % 2 == 0 ? 1.0f : -1.0f) * magnitude;
}

vector<float> concatDownloadActive(const Impl::Tensor& source, uint64_t elements) {
    Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
    Impl::Tensor host = source.clone(cpu);
    Impl::Tensor deviceHandle = source;
    Stream stream = Stream::getNextDownloadStream(source.getPlacement().getDeviceNum());
    host.downloadSection(deviceHandle, stream, 0, 0, elements * sizeof(float));
    stream.synchronize();
    return vector<float>(host.getMemPtr<float>(), host.getMemPtr<float>() + elements);
}
} // namespace

TEST(RaggedConcatenateNeutral1N, SavedThreeInputBackpropagatesEveryValuesPortAtTwoAndSeven) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "CUDA device unavailable";

    const filesystem::path archive = filesystem::temp_directory_path() /
        ("concat_neutral_1n3_backward_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
    filesystem::remove_all(archive);
    vector<uint64_t> portOriginalIds;
    uint64_t partitionOriginalId = 0, outputOriginalId = 0;
    {
        Network network("concatenate_1n3_backward");
        const RaggedTensor first = owner(network, {2, 3});
        const RaggedTensor second = sibling(network, first, "right", {1, 3});
        const RaggedTensor third = sibling(network, first, "third", {3, 3});
        const RaggedTensor targets = sibling(network, first, "targets", {6, 3});

        // Each rivet preserves a separate differentiable input port across
        // graph pruning, while keeping the one canonical partition untouched.
        vector<RaggedTensor> operands;
        for (const RaggedTensor& input : {first, second, third}) {
            const auto rivet = GradientRivet::Builder().network(network)
                .tensor(input.getValues()).build();
            ASSERT_TRUE(rivet.getFeatureOutput().has_value());
            operands.push_back(input.withValues(rivet.getFeatureOutput().value()));
        }
        const auto concat = Concatenate::Builder().network(network)
            .featureInput(operands[0]).featureInput(operands[1])
            .featureInput(operands[2]).concatenationAxis(0).build();
        const auto ports = concat.getFeatureInputs();
        ASSERT_EQ(ports.size(), 4U);
        for (uint32_t i = 0; i < 3; ++i) {
            portOriginalIds.push_back(ports[i].getOriginalId());
            for (uint32_t j = 0; j < i; ++j) EXPECT_NE(ports[i], ports[j]);
        }
        ASSERT_TRUE(concat.getRaggedFeatureOutput().has_value());
        EXPECT_TRUE(concat.getRaggedFeatureOutput()->sharesPartitionWith(targets));
        EXPECT_EQ(concat.getRaggedFeatureOutput()->getValuesDimensions(),
                  (vector<uint64_t>{4, 6, 3}));
        const auto loss = RaggedCustomLoss::Builder().network(network)
            .predictions(concat.getRaggedFeatureOutput().value()).labels(targets)
            .lossExpression(concatenateSquaredError(false))
            .gradientExpression(concatenateSquaredError(true)).build();
        (void)NetworkOutput::Builder().network(network).name("raw_loss")
            .inputTensor(loss.getRaggedRawLoss().getValues()).dataType(DataType::FP32).build();
        partitionOriginalId = first.getRowPartitionToken().getOriginalId();
        outputOriginalId = concat.getFeatureOutputs().at(0).getOriginalId();
        EXPECT_EQ(ports[3].getOriginalId(), partitionOriginalId);
        EXPECT_EQ(concat.architectureJson().at("version"), "1.1.0");
        EXPECT_EQ(loss.architectureJson().at("version"), "1.1.0");
        network.save(archive.string(), true);
    }

    for (uint32_t b : {2U, 7U}) {
        SCOPED_TRACE(::testing::Message() << "B=" << b);
        Network network("concatenate_1n3_backward");
        ASSERT_NO_THROW(network.load(archive.string()));
        const auto external = network.getExternalRaggedNetworkInputs();
        ASSERT_EQ(external.size(), 4U);
        auto findInput = [&](const string& name) -> RaggedTensor {
            for (const auto& item : external)
                if (item.name == name) return item.raggedTensor;
            ADD_FAILURE() << "Missing external ragged input " << name;
            return external.front().raggedTensor;
        };
        const RaggedTensor first = findInput("left");
        const RaggedTensor second = findInput("right");
        const RaggedTensor third = findInput("third");
        const RaggedTensor targets = findInput("targets");
        for (const auto& input : {second, third, targets})
            EXPECT_TRUE(input.sharesPartitionWith(first));
        EXPECT_EQ(first.getRowPartitionToken().getOriginalId(), partitionOriginalId);

        const Concatenate* concat = nullptr;
        shared_ptr<RaggedCustomLoss> rawLoss;
        for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
            if (auto candidate = dynamic_pointer_cast<Concatenate>(network.getLayer(i)))
                concat = candidate.get();
            if (auto candidate = dynamic_pointer_cast<RaggedCustomLoss>(network.getLayer(i)))
                rawLoss = candidate;
        }
        ASSERT_NE(concat, nullptr);
        ASSERT_NE(rawLoss, nullptr);
        const auto ports = concat->getFeatureInputs();
        ASSERT_EQ(ports.size(), 4U);
        for (uint32_t i = 0; i < 3; ++i)
            EXPECT_EQ(ports[i].getOriginalId(), portOriginalIds[i]);
        EXPECT_EQ(ports[3].getOriginalId(), partitionOriginalId);
        EXPECT_EQ(concat->getFeatureOutputs()[0].getOriginalId(), outputOriginalId);
        EXPECT_EQ(rawLoss->getRaggedPredictions().getValues(), concat->getFeatureOutputs()[0]);
        EXPECT_EQ(concat->architectureJson().at("version"), "1.1.0");
        EXPECT_EQ(concat->architectureJson().dump().find("\"batch_size\""), string::npos);
        EXPECT_EQ(concat->architectureJson().dump().find("\"max_total_values\""), string::npos);
        const json architectureBefore = network.architectureJson();

        vector<Event> initialization;
        auto placed = network.place(b, initialization, /*inferenceOnly=*/false);
        ASSERT_NE(placed, nullptr);
        for (auto& event : initialization) event.synchronize();
        auto& stamp = placed->getStampedNetwork(0);
        auto physical = dynamic_pointer_cast<Impl::RaggedConcatenate>(
            stamp.getPhysicalLayerFromApiLayer(concat->getId()));
        auto physicalLoss = dynamic_pointer_cast<Impl::RaggedCustomLoss>(
            stamp.getPhysicalLayerFromApiLayer(rawLoss->getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_NE(physicalLoss, nullptr);
        const uint64_t capacity = static_cast<uint64_t>(b) * 4;
        EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(first.getRowPartitionId()), nullptr);
        auto activeCarrier = stamp.getManagedPartitionActiveCountInputForTest(first.getRowPartitionId());
        ASSERT_NE(activeCarrier, nullptr);
        ASSERT_TRUE(activeCarrier->getFeatureOutput().has_value());
        EXPECT_EQ(activeCarrier->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(physicalLoss->getLogicalBatchSize(), b);
        EXPECT_EQ(physicalLoss->getMaxTotalValues(), capacity);
        const auto physicalInputs = physical->getFeatureInputs();
        ASSERT_EQ(physicalInputs.size(), 4U);
        for (uint32_t i = 0; i < 3; ++i) {
            ASSERT_TRUE(physicalInputs[i].has_value());
            EXPECT_EQ(physicalInputs[i]->getDimensions(),
                      (vector<uint64_t>{capacity, vector<uint64_t>{2, 1, 3}[i], 3}));
        }
        ASSERT_TRUE(physicalInputs[3].has_value());
        EXPECT_EQ(physicalInputs[3]->getDimensions(), (vector<uint64_t>{1}));
        const auto outputs = physical->getFeatureOutputs();
        ASSERT_EQ(outputs.size(), 1U);
        ASSERT_TRUE(outputs[0].has_value());
        EXPECT_EQ(outputs[0]->getDimensions(), (vector<uint64_t>{capacity, 6, 3}));
        EXPECT_EQ(network.architectureJson(), architectureBefore);

        vector<uint64_t> offsets(b + 1, 0);
        for (uint32_t row = 0; row < b; ++row)
            offsets[row + 1] = offsets[row] + (row == 0 || row % 3 == 0 ? 0 : 1 + row % 2);
        const uint64_t active = offsets.back();
        ASSERT_GT(active, 0U);
        ASSERT_LT(active, capacity);
        Impl::TensorPlacement cpu(Impl::TensorPlacement::MemDevices::CPU);
        Impl::Tensor offsetTensor(cpu, first.getDescriptorForPlacement(b).getOffsetsDescriptor());
        for (uint32_t i = 0; i <= b; ++i)
            offsetTensor.getMemPtr<uint32_t>()[i] = static_cast<uint32_t>(offsets[i]);
        Impl::RowPartitionRuntime partition(offsetTensor, first.getDescriptorForPlacement(b).getRowPartition());
        partition.setHostOffsets(offsets);

        Batch trainingBatch;
        const vector<pair<string, RaggedTensor>> namedInputs = {
            {"left", first}, {"right", second}, {"third", third}};
        const vector<uint32_t> widths = {2, 1, 3};
        for (uint32_t port = 0; port < namedInputs.size(); ++port) {
            const auto& [name, logical] = namedInputs[port];
            Impl::Tensor tensor(cpu, logical.getDescriptorForPlacement(b).getValuesDescriptor());
            for (uint64_t token = 0; token < capacity; ++token)
                for (uint32_t row = 0; row < widths[port]; ++row)
                    for (uint32_t column = 0; column < 3; ++column)
                        tensor.getMemPtr<float>()[token * widths[port] * 3 + row * 3 + column] = token < active
                            ? concatPortValue(token, port, row, column)
                            : numeric_limits<float>::quiet_NaN();
            trainingBatch.insert(name, Impl::RaggedTensor(tensor, partition));
        }
        Impl::Tensor labels(cpu, targets.getDescriptorForPlacement(b).getValuesDescriptor());
        for (uint64_t token = 0; token < capacity; ++token) {
            uint32_t outputRow = 0;
            for (uint32_t port = 0; port < widths.size(); ++port)
                for (uint32_t row = 0; row < widths[port]; ++row, ++outputRow)
                    for (uint32_t column = 0; column < 3; ++column)
                        labels.getMemPtr<float>()[token * 18 + outputRow * 3 + column] = token < active
                            ? concatPortValue(token, port, row, column)
                                - concatUpstreamDifference(token, outputRow, column)
                            : numeric_limits<float>::quiet_NaN();
        }
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
        EXPECT_EQ(results.at("raw_loss").getDimensions(), (vector<uint64_t>{capacity, 6, 3}));

        const auto gradients = physical->getErrorOutputs();
        ASSERT_EQ(gradients.size(), 4U);
        for (uint32_t port = 0; port < 3; ++port) {
            ASSERT_TRUE(gradients[port].has_value()) << "missing differentiable port " << port;
            EXPECT_EQ(gradients[port]->getDataType(), DataType::FP32);
            EXPECT_EQ(gradients[port]->getDimensions(),
                      (vector<uint64_t>{capacity, widths[port], 3}));
        }
        EXPECT_FALSE(gradients[3].has_value()) << "structural carrier cannot receive a values gradient";
        ASSERT_TRUE(physicalLoss->getErrorOutput().has_value());
        const auto actualOutput = concatDownloadActive(outputs[0].value(), active * 18);
        const auto actualLoss = concatDownloadActive(results.at("raw_loss"), active * 18);
        const auto upstream = concatDownloadActive(physicalLoss->getErrorOutput().value(), active * 18);
        vector<vector<float>> portGradients;
        for (uint32_t port = 0; port < 3; ++port)
            portGradients.push_back(concatDownloadActive(gradients[port].value(),
                                                       active * widths[port] * 3));
        bool sawPositive = false, sawNegative = false, sawDifferentMagnitudes = false;
        for (uint64_t token = 0; token < active; ++token) {
            uint32_t outputRow = 0;
            for (uint32_t port = 0; port < 3; ++port)
                for (uint32_t row = 0; row < widths[port]; ++row, ++outputRow)
                    for (uint32_t col = 0; col < 3; ++col) {
                        const uint64_t outputIndex = token * 18 + outputRow * 3 + col;
                        const uint64_t portIndex = token * widths[port] * 3 + row * 3 + col;
                        const float delta = concatUpstreamDifference(token, outputRow, col);
                        const float expectedGradient = 2.0f * delta * Impl::Loss::getLossScalingFactor();
                        sawPositive |= expectedGradient > 0.0f;
                        sawNegative |= expectedGradient < 0.0f;
                        sawDifferentMagnitudes |= abs(delta) > 0.3f;
                        EXPECT_NEAR(actualOutput[outputIndex], concatPortValue(token, port, row, col), 2e-4f)
                            << "concatenate forward, token " << token << ", port " << port;
                        EXPECT_NEAR(actualLoss[outputIndex], delta * delta, 2e-4f)
                            << "RAW loss at " << outputIndex;
                        EXPECT_NEAR(upstream[outputIndex], expectedGradient, 5e-3f)
                            << "RAW upstream at " << outputIndex;
                        EXPECT_NEAR(portGradients[port][portIndex], expectedGradient, 6e-3f)
                            << "split derivative token " << token << " port " << port
                            << " row " << row << " col " << col;
                        EXPECT_NEAR(portGradients[port][portIndex], upstream[outputIndex], 6e-3f)
                            << "port slice must equal upstream slice";
                    }
        }
        EXPECT_TRUE(sawPositive);
        EXPECT_TRUE(sawNegative);
        EXPECT_TRUE(sawDifferentMagnitudes);
        EXPECT_EQ(network.architectureJson(), architectureBefore);
        // All downloads above are strictly limited to the active packed prefix.
    }
    filesystem::remove_all(archive);
}
