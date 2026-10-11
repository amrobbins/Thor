#include "DeepLearning/Api/Layers/Activations/BilinearGlu.h"
#include "DeepLearning/Api/Layers/Activations/Geglu.h"
#include "DeepLearning/Api/Layers/Activations/Glu.h"
#include "DeepLearning/Api/Layers/Activations/Reglu.h"
#include "DeepLearning/Api/Layers/Activations/Swiglu.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/Event.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
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

namespace {

struct G2Case {
    const char* name;
    const char* layerType;
};
constexpr array<G2Case, 5> g2Cases = {{{"glu", "Glu"}, {"reglu", "Reglu"}, {"geglu", "Geglu"},
                                       {"swiglu", "Swiglu"}, {"bilinear_glu", "BilinearGlu"}}};

shared_ptr<Activation> buildG2(const string& name, Network& network, RaggedTensor input) {
    if (name == "glu") return Glu::Builder().network(network).featureInput(input).build();
    if (name == "reglu") return Reglu::Builder().network(network).featureInput(input).build();
    if (name == "geglu") return Geglu::Builder().network(network).featureInput(input).build();
    if (name == "swiglu") return Swiglu::Builder().network(network).featureInput(input).build();
    if (name == "bilinear_glu") return BilinearGlu::Builder().network(network).featureInput(input).build();
    throw runtime_error("Unknown GLU variant");
}

shared_ptr<Activation> buildDenseG2(const string& name, Network& network, Tensor input) {
    if (name == "glu") return Glu::Builder().network(network).featureInput(input).build();
    if (name == "reglu") return Reglu::Builder().network(network).featureInput(input).build();
    if (name == "geglu") return Geglu::Builder().network(network).featureInput(input).build();
    if (name == "swiglu") return Swiglu::Builder().network(network).featureInput(input).build();
    if (name == "bilinear_glu") return BilinearGlu::Builder().network(network).featureInput(input).build();
    throw runtime_error("Unknown GLU variant");
}

shared_ptr<Activation> onlyG2(Network& network, const string& type) {
    shared_ptr<Activation> found;
    int count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto candidate = dynamic_pointer_cast<Activation>(network.getLayer(i));
        if (candidate && candidate->getLayerType() == type) { found = candidate; ++count; }
    }
    EXPECT_EQ(count, 1) << type;
    return found;
}

void syncEvents(vector<Event>& events) {
    for (auto& event : events) event.synchronize();
    events.clear();
}

float channel(uint64_t token, uint32_t channel) {
    switch (channel) {
        case 0: return 0.5F + 0.125F * static_cast<float>(token);
        case 1: return -1.25F + 0.2F * static_cast<float>(token);
        case 2: return -1.0F + 0.1F * static_cast<float>(token);
        default: return 0.7F - 0.15F * static_cast<float>(token);
    }
}

float reference(const string& kind, float value, float gate) {
    if (kind == "glu") return value / (1.0F + exp(-gate));
    if (kind == "reglu") return value * max(gate, 0.0F);
    if (kind == "geglu") return value * gate * 0.5F * erfc(-gate / sqrt(2.0F));
    if (kind == "swiglu") return value * gate / (1.0F + exp(-gate));
    return value * gate;
}

Batch makeG2Batch(const RaggedTensor& logicalInput, uint32_t batchSize) {
    const auto descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    const uint64_t capacity = descriptor.getMaxTotalValues();
    vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
    for (uint32_t row = 1; row <= batchSize; ++row) {
        const uint64_t length = row == 1 ? 0 : 1 + ((row - 2) % 2);
        hostOffsets[row] = hostOffsets[row - 1] + length;
    }
    const uint64_t active = hostOffsets.back();
    THOR_THROW_IF_FALSE(active < capacity);

    ThorImplementation::TensorPlacement cpu(ThorImplementation::TensorPlacement::MemDevices::CPU);
    ThorImplementation::Tensor values(cpu, descriptor.getValuesDescriptor());
    float* data = values.getMemPtr<float>();
    for (uint64_t token = 0; token < capacity; ++token)
        for (uint32_t c = 0; c < 4; ++c)
            data[token * 4 + c] = token < active ? channel(token, c) : numeric_limits<float>::quiet_NaN();

    ThorImplementation::Tensor offsets(cpu, descriptor.getOffsetsDescriptor());
    uint32_t* offsetData = offsets.getMemPtr<uint32_t>();
    for (size_t i = 0; i < hostOffsets.size(); ++i) offsetData[i] = static_cast<uint32_t>(hostOffsets[i]);
    ThorImplementation::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(hostOffsets);
    Batch batch;
    batch.insert("tokens", ThorImplementation::RaggedTensor(values, partition));
    return batch;
}

filesystem::path temporaryG2Archive(const string& name) {
    return filesystem::temp_directory_path() /
        ("thor_glu_g2_" + name + "_" + to_string(chrono::steady_clock::now().time_since_epoch().count()));
}

} // namespace

TEST(GatedLinearUnitsNeutralG2, FiveVariantsSerializeSemanticShapeAndRejectCorruptedReferences) {
    constexpr uint64_t M = 4;
    for (const auto& variant : g2Cases) {
        SCOPED_TRACE(variant.name);
        Network network(string("g2_geometry_") + variant.name);
        RaggedTensor input = RaggedNetworkInput::Builder().network(network).name("tokens")
            .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
            .trailingDimensions({4}).maxValuesPerRow(M).build();
        auto activation = buildG2(variant.name, network, input);
        ASSERT_NE(activation, nullptr);
        ASSERT_TRUE(activation->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(activation->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{M, 2}));
        EXPECT_EQ(activation->getRaggedFeatureOutput()->getRowPartitionToken(), input.getRowPartitionToken());
        const json j = activation->architectureJson();
        EXPECT_EQ(j.at("version").get<string>(), "1.1.0");
        EXPECT_EQ(j.at("layer_type").get<string>(), variant.name);
        EXPECT_EQ(j.at("feature_input").at("id").get<uint64_t>(), input.getValues().getOriginalId());
        EXPECT_EQ(j.at("feature_output").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{M, 2}));
        const json& in = j.at("ragged_feature_input");
        const json& out = j.at("ragged_feature_output");
        EXPECT_FALSE(in.contains("geometry_mode"));
        EXPECT_EQ(in.at("max_values_per_row").get<uint64_t>(), M);
        EXPECT_EQ(in.at("values").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{M, 4}));
        EXPECT_EQ(in.at("offsets").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{1}));
        EXPECT_EQ(out.at("values").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{M, 2}));
        EXPECT_EQ(out.at("offsets").at("id").get<uint64_t>(), in.at("offsets").at("id").get<uint64_t>());
        EXPECT_EQ(out.at("values").at("id").get<uint64_t>(), j.at("feature_output").at("id").get<uint64_t>());
        EXPECT_FALSE(in.contains("batch_size"));
        EXPECT_FALSE(in.contains("max_total_values"));
        EXPECT_FALSE(out.contains("batch_size"));
        EXPECT_FALSE(out.contains("max_total_values"));

        json invalid = j;
        // 1.0.0 is a supported historical input grammar, not an invalid
        // alternate model geometry. Reject a genuinely unsupported version.
        invalid["version"] = "9.9.9";
        EXPECT_THROW(Activation::deserialize(invalid, &network), runtime_error);
        invalid = j;
        invalid["ragged_feature_input"]["geometry_mode"] = "fixed";
        EXPECT_ANY_THROW(Activation::deserialize(invalid, &network));
        invalid = j;
        invalid["ragged_feature_input"]["max_values_per_row"] = M + 1;
        EXPECT_ANY_THROW(Activation::deserialize(invalid, &network));
        invalid = j;
        invalid["ragged_feature_output"]["offsets"]["id"] = input.getValues().getOriginalId();
        EXPECT_THROW(Activation::deserialize(invalid, &network), runtime_error);
        invalid = j;
        invalid["ragged_feature_output"]["values"]["id"] = input.getValues().getOriginalId();
        EXPECT_THROW(Activation::deserialize(invalid, &network), runtime_error);
        invalid = j;
        invalid["feature_input"]["id"] = j.at("feature_output").at("id");
        EXPECT_THROW(Activation::deserialize(invalid, &network), runtime_error);
        invalid = j;
        invalid["feature_output"]["dimensions"] = vector<uint64_t>{M, 3};
        invalid["ragged_feature_output"]["values"]["dimensions"] = vector<uint64_t>{M, 3};
        EXPECT_THROW(Activation::deserialize(invalid, &network), runtime_error);
        invalid = j;
        invalid["feature_output"]["data_type"] = DataType::FP16;
        invalid["ragged_feature_output"]["values"]["data_type"] = DataType::FP16;
        EXPECT_THROW(Activation::deserialize(invalid, &network), runtime_error);

        Network oddNetwork(string("g2_odd_") + variant.name);
        RaggedTensor odd = RaggedNetworkInput::Builder().network(oddNetwork).name("odd_tokens")
            .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
            .trailingDimensions({5}).maxValuesPerRow(M).build();
        EXPECT_ANY_THROW((void)buildG2(variant.name, oddNetwork, odd));
    }
}

TEST(GatedLinearUnitsNeutralG2, FiveVariantsLoadAtTwoAndSevenAndExecuteSplitGate) {
    constexpr uint64_t M = 4;
    for (const auto& variant : g2Cases) {
        SCOPED_TRACE(variant.name);
        const string networkName = string("g2_neutral_") + variant.name;
        const filesystem::path archive = temporaryG2Archive(variant.name);
        filesystem::remove_all(archive);
        uint64_t savedInputId = 0, savedOutputId = 0, savedPartitionId = 0;
        {
            Network network(networkName);
            RaggedTensor input = RaggedNetworkInput::Builder().network(network).name("tokens")
                .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
                .trailingDimensions({4}).maxValuesPerRow(M).build();
            auto activation = buildG2(variant.name, network, input);
            ASSERT_NE(activation, nullptr);
            ASSERT_TRUE(activation->getRaggedFeatureOutput().has_value());
            (void)RaggedNetworkOutput::Builder().network(network).name("activated")
                .inputTensor(activation->getRaggedFeatureOutput().value()).build();
            savedInputId = input.getValues().getOriginalId();
            savedOutputId = activation->getFeatureOutput()->getOriginalId();
            savedPartitionId = input.getRowPartitionToken().getOriginalId();
            network.save(archive.string(), true);
        }
        for (const uint32_t B : {2U, 7U}) {
            SCOPED_TRACE(B);
            Network loaded(networkName);
            ASSERT_NO_THROW(loaded.load(archive.string()));
            const auto inputs = loaded.getExternalRaggedNetworkInputs();
            ASSERT_EQ(inputs.size(), 1U);
            const RaggedTensor logicalInput = inputs.front().raggedTensor;
            EXPECT_EQ(logicalInput.getValues().getOriginalId(), savedInputId);
            EXPECT_EQ(logicalInput.getRowPartitionToken().getOriginalId(), savedPartitionId);
            auto activation = onlyG2(loaded, variant.layerType);
            ASSERT_NE(activation, nullptr);
            ASSERT_TRUE(activation->getRaggedFeatureOutput().has_value());
            EXPECT_EQ(activation->getFeatureOutput()->getOriginalId(), savedOutputId);
            EXPECT_EQ(activation->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{M, 2}));
            EXPECT_EQ(activation->getRaggedFeatureOutput()->getRowPartitionToken(), logicalInput.getRowPartitionToken());
            const json architectureBefore = loaded.architectureJson();
            vector<Event> events;
            auto placed = loaded.place(B, events, true);
            ASSERT_NE(placed, nullptr);
            syncEvents(events);
            auto& stamped = placed->getStampedNetwork(0);
            const auto requirement = stamped.getExternalRowPartitionRequirementsForTest(logicalInput.getRowPartitionId());
            ASSERT_TRUE(requirement.has_value());
            EXPECT_TRUE(ThorImplementation::hasRaggedPartitionRequirement(
                requirement.value(), ThorImplementation::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT));
            auto managed = stamped.getManagedPartitionActiveCountInputForTest(logicalInput.getRowPartitionId());
            ASSERT_NE(managed, nullptr);
            ASSERT_TRUE(managed->getFeatureOutput().has_value());
            EXPECT_EQ(managed->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
            EXPECT_EQ(stamped.getManagedPartitionOffsetsInputForTest(logicalInput.getRowPartitionId()), nullptr);
            auto physicalInput = stamped.getNamedInput("tokens.values");
            ASSERT_NE(physicalInput, nullptr);
            ASSERT_TRUE(physicalInput->getFeatureOutput().has_value());
            EXPECT_EQ(physicalInput->getFeatureOutput()->getDimensions(), (vector<uint64_t>{B * M, 4}));
            auto physicalGLU = dynamic_pointer_cast<ThorImplementation::RaggedCustomLayer>(
                stamped.getPhysicalLayerFromApiLayer(activation->getId()));
            ASSERT_NE(physicalGLU, nullptr);
            ASSERT_TRUE(physicalGLU->getFeatureOutput().has_value());
            EXPECT_EQ(physicalGLU->getFeatureOutput()->getDimensions(), (vector<uint64_t>{B * M, 2}));
            EXPECT_EQ(loaded.architectureJson(), architectureBefore);

            const map<string, InferenceOutputValue> outputs = placed->inferLogical(makeG2Batch(logicalInput, B));
            ASSERT_TRUE(outputs.contains("activated"));
            ASSERT_TRUE(holds_alternative<ThorImplementation::RaggedTensor>(outputs.at("activated")));
            const auto output = get<ThorImplementation::RaggedTensor>(outputs.at("activated"));
            EXPECT_EQ(output.getValues().getDimensions(), (vector<uint64_t>{B * M, 2}));
            auto offsets = output.getHostOffsetsIfAvailable();
            ASSERT_TRUE(offsets.has_value());
            EXPECT_EQ(offsets->at(0), 0U);
            EXPECT_EQ(offsets->at(1), 0U);
            const uint64_t active = offsets->back();
            EXPECT_LT(active, B * M);
            const float* result = output.getValues().getMemPtr<float>();
            for (uint64_t token = 0; token < active; ++token)
                for (uint32_t c = 0; c < 2; ++c)
                    EXPECT_NEAR(result[token * 2 + c],
                                reference(variant.name, channel(token, c), channel(token, c + 2)), 4e-3F)
                        << "token=" << token << " channel=" << c;
        }
        filesystem::remove_all(archive);
    }
}

TEST(GatedLinearUnitsNeutralG2, AllFivePreserveHistoricalDenseAndFixedVersionOne) {
    for (const auto& variant : g2Cases) {
        SCOPED_TRACE(variant.name);
        const string name = string("g2_compat_") + variant.name;
        const filesystem::path archive = temporaryG2Archive(variant.name);
        filesystem::remove_all(archive);
        {
            Network network(name);
            RaggedTensor input = RaggedNetworkInput::Builder().network(network).name("tokens")
                .valuesDataType(DataType::FP32).offsetsDataType(DataType::UINT32)
                .trailingDimensions({4}).maxValuesPerRow(4).build();
            auto activation = buildG2(variant.name, network, input);
            ASSERT_NE(activation, nullptr);
            (void)RaggedNetworkOutput::Builder().network(network).name("activated")
                .inputTensor(activation->getRaggedFeatureOutput().value()).build();
            EXPECT_EQ(activation->architectureJson().at("version").get<string>(), "1.1.0");
            EXPECT_FALSE(activation->architectureJson().at("ragged_feature_input").contains("batch_size"));
            network.save(archive.string(), true);
        }
        {
            Network loaded(name);
            ASSERT_NO_THROW(loaded.load(archive.string()));
            auto activation = onlyG2(loaded, variant.layerType);
            ASSERT_NE(activation, nullptr);
            ASSERT_TRUE(activation->getRaggedFeatureInput().has_value());
            EXPECT_EQ(activation->architectureJson().at("version").get<string>(), "1.1.0");
            json invalid = activation->architectureJson();
            invalid["version"] = "0.0.0";
            EXPECT_THROW(Activation::deserialize(invalid, &loaded), runtime_error);
        }
        filesystem::remove_all(archive);
        const string denseName = name + "_dense";
        const filesystem::path denseArchive = temporaryG2Archive(denseName);
        filesystem::remove_all(denseArchive);
        {
            Network dense(denseName);
            NetworkInput input = NetworkInput::Builder().network(dense).name("features")
                .dimensions({4}).dataType(DataType::FP32).build();
            auto activation = buildDenseG2(variant.name, dense, input.getFeatureOutput().value());
            ASSERT_NE(activation, nullptr);
            (void)NetworkOutput::Builder().network(dense).name("activated")
                .inputTensor(activation->getFeatureOutput().value()).dataType(DataType::FP32).build();
            const json j = activation->architectureJson();
            EXPECT_EQ(j.at("version").get<string>(), "1.0.0");
            EXPECT_FALSE(j.value("use_ragged", false));
            EXPECT_EQ(activation->getFeatureOutput()->getDimensions(), (vector<uint64_t>{2}));
            dense.save(denseArchive.string(), true);
        }
        {
            Network loaded(denseName);
            ASSERT_NO_THROW(loaded.load(denseArchive.string()));
            auto activation = onlyG2(loaded, variant.layerType);
            ASSERT_NE(activation, nullptr);
            EXPECT_FALSE(activation->getUseRagged());
            EXPECT_EQ(activation->architectureJson().at("version").get<string>(), "1.0.0");
            json invalid = activation->architectureJson();
            invalid["version"] = "1.1.0";
            EXPECT_THROW(Activation::deserialize(invalid, &loaded), runtime_error);
        }
        filesystem::remove_all(denseArchive);
    }
}
