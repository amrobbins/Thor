#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Activations/Gelu.h"
#include "DeepLearning/Api/Layers/Activations/Sigmoid.h"
#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "DeepLearning/Api/Layers/Activations/SoftSign.h"
#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "Utilities/Common/Event.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {

ThorImplementation::TensorPlacement cpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

template <typename ActivationType>
shared_ptr<ActivationType> buildActivation(Network& network, RaggedTensor input) {
    typename ActivationType::Builder builder;
    builder.network(network);
    builder.featureInput(input);
    return dynamic_pointer_cast<ActivationType>(builder.build());
}

template <typename ActivationType>
shared_ptr<ActivationType> findOnlyActivation(Network& network) {
    shared_ptr<ActivationType> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto candidate = dynamic_pointer_cast<ActivationType>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1U);
    return found;
}

template <typename ActivationType>
class TestableOrdinaryActivation : public ActivationType {
   public:
    void initializeNeutralForTest(RaggedTensor input) {
        this->initializeStandaloneActivation(input);
        this->initialized = true;
    }

    uint64_t memoryBytes(uint32_t batchSize) const {
        return this->getExpressionBackedActivationMemRequirementInBytes(batchSize);
    }
};

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

float neutralInputValue(uint64_t elementIndex) {
    return 0.5F * static_cast<float>(static_cast<int>(elementIndex % 9) - 4);
}

Batch makeNeutralActivationBatch(const string& inputName, const RaggedTensor& logicalInput, uint32_t batchSize) {
    const ThorImplementation::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    constexpr uint64_t elementsPerValue = 2;
    THOR_THROW_IF_FALSE(logicalInput.getTrailingDimensions() == (vector<uint64_t>{elementsPerValue}));

    vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
    uint64_t activeValues = 0;
    for (uint32_t row = 0; row < batchSize; ++row) {
        const uint64_t rowLength = row == 0 ? 0 : 1 + ((row - 1) % 2);
        activeValues += rowLength;
        hostOffsets[row + 1] = activeValues;
    }
    THOR_THROW_IF_FALSE(activeValues <= descriptor.getMaxTotalValues());

    ThorImplementation::Tensor values(cpuPlacement, descriptor.getValuesDescriptor());
    float* valuesPtr = values.getMemPtr<float>();
    for (uint64_t value = 0; value < descriptor.getMaxTotalValues(); ++value) {
        for (uint64_t element = 0; element < elementsPerValue; ++element) {
            const uint64_t i = value * elementsPerValue + element;
            valuesPtr[i] = value < activeValues ? neutralInputValue(i) : numeric_limits<float>::quiet_NaN();
        }
    }

    ThorImplementation::Tensor offsets(cpuPlacement, descriptor.getOffsetsDescriptor());
    THOR_THROW_IF_FALSE(logicalInput.getOffsetsDataType() == DataType::UINT32);
    uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
    for (size_t i = 0; i < hostOffsets.size(); ++i)
        offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);

    ThorImplementation::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(hostOffsets);
    Batch batch;
    batch.insert(inputName, ThorImplementation::RaggedTensor(values, partition));
    return batch;
}

template <typename ActivationType>
void expectNeutralSemanticJson(const string& label) {
    SCOPED_TRACE(label);
    constexpr uint64_t maxValuesPerRow = 4;
    Network network("neutral_" + label + "_json");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({2})
                             .maxValuesPerRow(maxValuesPerRow)
                             .build();
    ASSERT_TRUE(input.isPlacementNeutral());

    auto activation = buildActivation<ActivationType>(network, input);
    ASSERT_NE(activation, nullptr);
    ASSERT_TRUE(activation->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(activation->getRaggedFeatureOutput()->getRowPartitionToken(), input.getRowPartitionToken());
    EXPECT_EQ(activation->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{maxValuesPerRow, 2}));

    TestableOrdinaryActivation<ActivationType> testable;
    testable.initializeNeutralForTest(input);
    const uint64_t logicalValuesBytes = maxValuesPerRow * 2U * sizeof(float);
    EXPECT_EQ(testable.memoryBytes(2), 2U * logicalValuesBytes * 2U + sizeof(uint32_t));
    EXPECT_EQ(testable.memoryBytes(7), 7U * logicalValuesBytes * 2U + sizeof(uint32_t));

    const json architecture = activation->architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    const json& inputJson = architecture.at("ragged_feature_input");
    const json& outputJson = architecture.at("ragged_feature_output");
    EXPECT_EQ(inputJson.at("layer_ragged_reference_version").get<string>(), "1.0.0");
    EXPECT_EQ(inputJson.at("geometry_mode").get<string>(), "placement_neutral");
    EXPECT_EQ(inputJson.at("max_values_per_row").get<uint64_t>(), maxValuesPerRow);
    EXPECT_FALSE(inputJson.contains("batch_size"));
    EXPECT_FALSE(inputJson.contains("max_total_values"));
    EXPECT_FALSE(outputJson.contains("batch_size"));
    EXPECT_FALSE(outputJson.contains("max_total_values"));
    EXPECT_EQ(inputJson.at("values").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{maxValuesPerRow, 2}));
    EXPECT_EQ(inputJson.at("offsets").at("dimensions").get<vector<uint64_t>>(), (vector<uint64_t>{1}));
    EXPECT_EQ(outputJson.at("offsets").at("id").get<uint64_t>(), inputJson.at("offsets").at("id").get<uint64_t>());
    EXPECT_EQ(outputJson.at("values").at("id").get<uint64_t>(),
              architecture.at("feature_output").at("id").get<uint64_t>());
}

template <typename ActivationType>
void expectNeutralSaveLoadAndExecute(const string& label, const function<float(float)>& expectedValue) {
    SCOPED_TRACE(label);
    constexpr uint64_t maxValuesPerRow = 4;
    const string networkName = "neutral_" + label + "_round_trip";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_" + label + "_neutral_" + to_string(nonce));
    filesystem::remove_all(archive);

    uint64_t savedValuesOriginalId = 0;
    uint64_t savedPartitionOriginalId = 0;
    uint64_t savedOutputOriginalId = 0;
    {
        Network network(networkName);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .maxValuesPerRow(maxValuesPerRow)
                                 .build();
        auto activation = buildActivation<ActivationType>(network, input);
        ASSERT_NE(activation, nullptr);
        ASSERT_TRUE(activation->getRaggedFeatureOutput().has_value());
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("activated")
            .inputTensor(activation->getRaggedFeatureOutput().value())
            .build();

        savedValuesOriginalId = input.getValues().getOriginalId();
        savedPartitionOriginalId = input.getRowPartitionToken().getOriginalId();
        savedOutputOriginalId = activation->getFeatureOutput()->getOriginalId();
        const json layerJson = activation->architectureJson();
        EXPECT_EQ(layerJson.at("version").get<string>(), "1.1.0");
        EXPECT_FALSE(layerJson.at("ragged_feature_input").contains("batch_size"));
        EXPECT_FALSE(layerJson.at("ragged_feature_input").contains("max_total_values"));
        network.save(archive.string(), true);
    }

    auto loadPlaceAndRun = [&](uint32_t placementBatchSize) {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        const auto inputs = loaded.getExternalRaggedNetworkInputs();
        ASSERT_EQ(inputs.size(), 1U);
        const RaggedTensor loadedInput = inputs.front().raggedTensor;
        ASSERT_TRUE(loadedInput.isPlacementNeutral());
        EXPECT_EQ(loadedInput.getValues().getOriginalId(), savedValuesOriginalId);
        EXPECT_EQ(loadedInput.getRowPartitionToken().getOriginalId(), savedPartitionOriginalId);

        auto loadedActivation = findOnlyActivation<ActivationType>(loaded);
        ASSERT_NE(loadedActivation, nullptr);
        ASSERT_TRUE(loadedActivation->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedActivation->getFeatureInput().value(), loadedInput.getValues());
        EXPECT_EQ(loadedActivation->getFeatureOutput()->getOriginalId(), savedOutputOriginalId);
        EXPECT_EQ(loadedActivation->getRaggedFeatureOutput()->getRowPartitionToken(), loadedInput.getRowPartitionToken());

        const json architectureBeforePlacement = loaded.architectureJson();
        vector<Event> initDoneEvents;
        shared_ptr<PlacedNetwork> placed = loaded.place(placementBatchSize, initDoneEvents, true);
        ASSERT_NE(placed, nullptr);
        synchronizeEvents(initDoneEvents);

        auto& stamped = placed->getStampedNetwork(0);
        const auto requirements = stamped.getExternalRowPartitionRequirementsForTest(loadedInput.getRowPartitionId());
        ASSERT_TRUE(requirements.has_value());
        EXPECT_TRUE(ThorImplementation::hasRaggedPartitionRequirement(
            requirements.value(), ThorImplementation::RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT));
        auto activeCount = stamped.getManagedPartitionActiveCountInputForTest(loadedInput.getRowPartitionId());
        ASSERT_NE(activeCount, nullptr);
        ASSERT_TRUE(activeCount->getFeatureOutput().has_value());
        EXPECT_EQ(activeCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
        EXPECT_EQ(stamped.getManagedPartitionOffsetsInputForTest(loadedInput.getRowPartitionId()), nullptr);

        const uint64_t expectedCapacity = static_cast<uint64_t>(placementBatchSize) * maxValuesPerRow;
        auto physicalInput = stamped.getNamedInput("tokens.values");
        ASSERT_NE(physicalInput, nullptr);
        ASSERT_TRUE(physicalInput->getFeatureOutput().has_value());
        EXPECT_EQ(physicalInput->getFeatureOutput()->getDimensions(), (vector<uint64_t>{expectedCapacity, 2}));
        auto physicalActivation = dynamic_pointer_cast<ThorImplementation::RaggedCustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(loadedActivation->getId()));
        ASSERT_NE(physicalActivation, nullptr);
        ASSERT_TRUE(physicalActivation->getFeatureOutput().has_value());
        EXPECT_EQ(physicalActivation->getFeatureOutput()->getDimensions(), (vector<uint64_t>{expectedCapacity, 2}));
        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        const map<string, InferenceOutputValue> outputs =
            placed->inferLogical(makeNeutralActivationBatch("tokens", loadedInput, placementBatchSize));
        ASSERT_TRUE(outputs.contains("activated"));
        ASSERT_TRUE(holds_alternative<ThorImplementation::RaggedTensor>(outputs.at("activated")));
        const ThorImplementation::RaggedTensor activated =
            get<ThorImplementation::RaggedTensor>(outputs.at("activated"));
        EXPECT_EQ(activated.getValues().getDimensions(), (vector<uint64_t>{expectedCapacity, 2}));
        EXPECT_EQ(activated.getOffsets().getDimensions(),
                  (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));
        const auto hostOffsets = activated.getHostOffsetsIfAvailable();
        ASSERT_TRUE(hostOffsets.has_value());
        ASSERT_GE(hostOffsets->size(), 2U);
        EXPECT_EQ(hostOffsets->at(0), 0U);
        EXPECT_EQ(hostOffsets->at(1), 0U) << "first row must remain empty";

        const uint64_t activeValues = hostOffsets->back();
        const float* values = activated.getValues().getMemPtr<float>();
        for (uint64_t value = 0; value < activeValues; ++value) {
            for (uint64_t element = 0; element < 2; ++element) {
                const uint64_t i = value * 2 + element;
                EXPECT_NEAR(values[i], expectedValue(neutralInputValue(i)), 2e-3F)
                    << label << " mismatch at active element " << i;
            }
        }
    };

    try {
        loadPlaceAndRun(2);
        loadPlaceAndRun(7);
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}

template <typename ActivationType>
void expectFixedRaggedCompatibility(const string& label) {
    SCOPED_TRACE(label);
    const string networkName = "fixed_" + label + "_compatibility";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_" + label + "_fixed_" + to_string(nonce));
    filesystem::remove_all(archive);

    {
        Network network(networkName);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .batchSize(3)
                                 .maxTotalValues(9)
                                 .maxValuesPerRow(4)
                                 .build();
        auto activation = buildActivation<ActivationType>(network, input);
        ASSERT_NE(activation, nullptr);
        ASSERT_TRUE(activation->getRaggedFeatureOutput().has_value());
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("activated")
            .inputTensor(activation->getRaggedFeatureOutput().value())
            .build();
        const json architecture = activation->architectureJson();
        EXPECT_EQ(architecture.at("version").get<string>(), "1.0.0");
        EXPECT_EQ(architecture.at("ragged_feature_input").at("batch_size").get<uint64_t>(), 3U);
        EXPECT_EQ(architecture.at("ragged_feature_input").at("max_total_values").get<uint64_t>(), 9U);
        EXPECT_FALSE(architecture.at("ragged_feature_input").contains("layer_ragged_reference_version"));
        network.save(archive.string(), true);
    }

    try {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        auto activation = findOnlyActivation<ActivationType>(loaded);
        ASSERT_NE(activation, nullptr);
        ASSERT_TRUE(activation->getRaggedFeatureInput().has_value());
        EXPECT_FALSE(activation->getRaggedFeatureInput()->isPlacementNeutral());
        EXPECT_EQ(activation->architectureJson().at("version").template get<string>(), "1.0.0");
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}

float sigmoidExpected(float x) {
    return x >= 0.0F ? 1.0F / (1.0F + exp(-x)) : exp(x) / (1.0F + exp(x));
}

float geluExpected(float x) {
    return x * 0.5F * erfc(-x / sqrt(2.0F));
}

float softPlusExpected(float x) {
    return max(x, 0.0F) + log1p(exp(-fabs(x)));
}

}  // namespace

TEST(Activations, OrdinaryPlacementNeutralRaggedActivationsSerializeSemanticGeometry) {
    expectNeutralSemanticJson<Sigmoid>("sigmoid");
    expectNeutralSemanticJson<Tanh>("tanh");
    expectNeutralSemanticJson<Gelu>("gelu");
    expectNeutralSemanticJson<Swish>("swish");
    expectNeutralSemanticJson<SoftPlus>("soft_plus");
    expectNeutralSemanticJson<SoftSign>("soft_sign");

    Network network("ordinary_activation_invalid_neutral_json");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({2})
                             .maxValuesPerRow(4)
                             .build();
    auto sigmoid = buildActivation<Sigmoid>(network, input);
    ASSERT_NE(sigmoid, nullptr);
    const json architecture = sigmoid->architectureJson();

    json wrongVersion = architecture;
    wrongVersion["version"] = "1.0.0";
    EXPECT_THROW(Activation::deserialize(wrongVersion, &network), runtime_error);

    json wrongGeometry = architecture;
    wrongGeometry["ragged_feature_input"]["geometry_mode"] = "fixed";
    EXPECT_THROW(Activation::deserialize(wrongGeometry, &network), runtime_error);

    json wrongPartition = architecture;
    wrongPartition["ragged_feature_output"]["offsets"]["id"] = input.getValues().getId();
    EXPECT_THROW(Activation::deserialize(wrongPartition, &network), runtime_error);

    json wrongOutputValues = architecture;
    wrongOutputValues["ragged_feature_output"]["values"]["id"] = input.getValues().getId();
    EXPECT_THROW(Activation::deserialize(wrongOutputValues, &network), runtime_error);
}

TEST(Activations, OrdinaryPlacementNeutralRaggedActivationsSaveLoadPlaceAndExecute) {
    expectNeutralSaveLoadAndExecute<Sigmoid>("sigmoid", sigmoidExpected);
    expectNeutralSaveLoadAndExecute<Tanh>("tanh", [](float x) { return tanh(x); });
    expectNeutralSaveLoadAndExecute<Gelu>("gelu", geluExpected);
    expectNeutralSaveLoadAndExecute<Swish>("swish", [](float x) { return x * sigmoidExpected(x); });
    expectNeutralSaveLoadAndExecute<SoftPlus>("soft_plus", softPlusExpected);
    expectNeutralSaveLoadAndExecute<SoftSign>("soft_sign", [](float x) { return x / (fabs(x) + 1.0F); });
}

TEST(Activations, OrdinaryFixedRaggedActivationsRemainVersionOneAndLoad) {
    expectFixedRaggedCompatibility<Sigmoid>("sigmoid");
    expectFixedRaggedCompatibility<Tanh>("tanh");
    expectFixedRaggedCompatibility<Gelu>("gelu");
    expectFixedRaggedCompatibility<Swish>("swish");
    expectFixedRaggedCompatibility<SoftPlus>("soft_plus");
    expectFixedRaggedCompatibility<SoftSign>("soft_sign");
}
