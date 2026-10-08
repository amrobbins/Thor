#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Layers/Activations/Elu.h"
#include "DeepLearning/Api/Layers/Activations/Exponential.h"
#include "DeepLearning/Api/Layers/Activations/HardSigmoid.h"
#include "DeepLearning/Api/Layers/Activations/HardSwish.h"
#include "DeepLearning/Api/Layers/Activations/HardTanh.h"
#include "DeepLearning/Api/Layers/Activations/Mish.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Relu6.h"
#include "DeepLearning/Api/Layers/Activations/Selu.h"
#include "DeepLearning/Api/Layers/Activations/Threshold.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
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
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
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

struct ActivationCase {
    string label;
    string layerType;
};

const array<ActivationCase, 9> cases = {{{"relu6", "Relu6"},
                                         {"elu", "Elu"},
                                         {"selu", "Selu"},
                                         {"exponential", "Exponential"},
                                         {"hard_sigmoid", "HardSigmoid"},
                                         {"hard_swish", "HardSwish"},
                                         {"hard_tanh", "HardTanh"},
                                         {"mish", "Mish"},
                                         {"threshold", "Threshold"}}};

shared_ptr<Activation> buildActivation(const string& label, Network& network, RaggedTensor input) {
    if (label == "relu6") return dynamic_pointer_cast<Activation>(Relu6::Builder().network(network).featureInput(input).build());
    if (label == "elu") {
        Elu::Builder builder;
        builder.network(network).featureInput(input).alpha(1.75F);
        return dynamic_pointer_cast<Activation>(builder.build());
    }
    if (label == "selu") return dynamic_pointer_cast<Activation>(Selu::Builder().network(network).featureInput(input).build());
    if (label == "exponential")
        return dynamic_pointer_cast<Activation>(Exponential::Builder().network(network).featureInput(input).build());
    if (label == "hard_sigmoid")
        return dynamic_pointer_cast<Activation>(HardSigmoid::Builder().network(network).featureInput(input).build());
    if (label == "hard_swish")
        return dynamic_pointer_cast<Activation>(HardSwish::Builder().network(network).featureInput(input).build());
    if (label == "hard_tanh") {
        HardTanh::Builder builder;
        builder.network(network).featureInput(input).minValue(-0.75).maxValue(0.625);
        return dynamic_pointer_cast<Activation>(builder.build());
    }
    if (label == "mish") return dynamic_pointer_cast<Activation>(Mish::Builder().network(network).featureInput(input).build());
    if (label == "threshold") {
        Threshold::Builder builder;
        builder.network(network).featureInput(input).threshold(0.25).value(-2.0);
        return dynamic_pointer_cast<Activation>(builder.build());
    }
    throw runtime_error("Unknown K3 activation label: " + label);
}

shared_ptr<Activation> buildActivation(const string& label, Network& network, Tensor input) {
    if (label == "relu6") return dynamic_pointer_cast<Activation>(Relu6::Builder().network(network).featureInput(input).build());
    if (label == "elu") {
        Elu::Builder builder;
        builder.network(network).featureInput(input).alpha(1.75F);
        return dynamic_pointer_cast<Activation>(builder.build());
    }
    if (label == "selu") return dynamic_pointer_cast<Activation>(Selu::Builder().network(network).featureInput(input).build());
    if (label == "exponential")
        return dynamic_pointer_cast<Activation>(Exponential::Builder().network(network).featureInput(input).build());
    if (label == "hard_sigmoid")
        return dynamic_pointer_cast<Activation>(HardSigmoid::Builder().network(network).featureInput(input).build());
    if (label == "hard_swish")
        return dynamic_pointer_cast<Activation>(HardSwish::Builder().network(network).featureInput(input).build());
    if (label == "hard_tanh") {
        HardTanh::Builder builder;
        builder.network(network).featureInput(input).minValue(-0.75).maxValue(0.625);
        return dynamic_pointer_cast<Activation>(builder.build());
    }
    if (label == "mish") return dynamic_pointer_cast<Activation>(Mish::Builder().network(network).featureInput(input).build());
    if (label == "threshold") {
        Threshold::Builder builder;
        builder.network(network).featureInput(input).threshold(0.25).value(-2.0);
        return dynamic_pointer_cast<Activation>(builder.build());
    }
    throw runtime_error("Unknown K3 activation label: " + label);
}

shared_ptr<Activation> findOnlyActivation(Network& network, const string& layerType) {
    shared_ptr<Activation> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto candidate = dynamic_pointer_cast<Activation>(network.getLayer(i));
        if (candidate != nullptr && candidate->getLayerType() == layerType) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1U) << layerType;
    return found;
}

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events) event.synchronize();
    events.clear();
}

float inputValue(uint64_t elementIndex) {
    static constexpr array<float, 10> values = {-8.0F, -4.0F, -1.5F, -0.5F, 0.0F,
                                                 0.25F, 1.0F, 3.0F, 7.0F, 0.75F};
    return values[elementIndex % values.size()];
}

float expectedValue(const string& label, float x) {
    if (label == "relu6") return min(max(x, 0.0F), 6.0F);
    if (label == "elu") return x >= 0.0F ? x : 1.75F * expm1(x);
    if (label == "selu") return x >= 0.0F ? 1.05070098F * x : 1.758099326F * expm1(x);
    if (label == "exponential") return exp(x);
    if (label == "hard_sigmoid") return min(max(0.2F * x + 0.5F, 0.0F), 1.0F);
    if (label == "hard_swish") return x * min(max(x + 3.0F, 0.0F), 6.0F) / 6.0F;
    if (label == "hard_tanh") return min(max(x, -0.75F), 0.625F);
    if (label == "mish") {
        const float softplus = max(x, 0.0F) + log1p(exp(-fabs(x)));
        return x * tanh(softplus);
    }
    if (label == "threshold") return x > 0.25F ? x : -2.0F;
    throw runtime_error("Unknown K3 activation label: " + label);
}

void expectParameters(const string& label, const json& architecture) {
    if (label == "elu") {
        EXPECT_FLOAT_EQ(architecture.at("alpha").get<float>(), 1.75F);
    } else if (label == "hard_tanh") {
        EXPECT_DOUBLE_EQ(architecture.at("min_value").get<double>(), -0.75);
        EXPECT_DOUBLE_EQ(architecture.at("max_value").get<double>(), 0.625);
    } else if (label == "threshold") {
        EXPECT_DOUBLE_EQ(architecture.at("threshold").get<double>(), 0.25);
        EXPECT_DOUBLE_EQ(architecture.at("value").get<double>(), -2.0);
    }
}

Batch makeBatch(const string& inputName, const RaggedTensor& logicalInput, uint32_t batchSize) {
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
            valuesPtr[i] = value < activeValues ? inputValue(i) : numeric_limits<float>::quiet_NaN();
        }
    }

    ThorImplementation::Tensor offsets(cpuPlacement, descriptor.getOffsetsDescriptor());
    THOR_THROW_IF_FALSE(logicalInput.getOffsetsDataType() == DataType::UINT32);
    uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
    for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);

    ThorImplementation::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(hostOffsets);
    Batch batch;
    batch.insert(inputName, ThorImplementation::RaggedTensor(values, partition));
    return batch;
}

void expectNeutralSemanticJsonAndMalformed(const ActivationCase& activationCase) {
    SCOPED_TRACE(activationCase.label);
    constexpr uint64_t maxValuesPerRow = 4;
    Network network("k3_neutral_" + activationCase.label + "_json");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({2})
                             .maxValuesPerRow(maxValuesPerRow)
                             .build();
    ASSERT_TRUE(input.isPlacementNeutral());

    auto activation = buildActivation(activationCase.label, network, input);
    ASSERT_NE(activation, nullptr);
    ASSERT_TRUE(activation->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(activation->getRaggedFeatureOutput()->getRowPartitionToken(), input.getRowPartitionToken());
    EXPECT_EQ(activation->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{maxValuesPerRow, 2}));

    const json architecture = activation->architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    expectParameters(activationCase.label, architecture);
    const json& inputJson = architecture.at("ragged_feature_input");
    const json& outputJson = architecture.at("ragged_feature_output");
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
    EXPECT_EQ(outputJson.at("values").at("id").get<uint64_t>(), architecture.at("feature_output").at("id").get<uint64_t>());

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

void expectNeutralSaveLoadPlaceAndExecute(const ActivationCase& activationCase) {
    SCOPED_TRACE(activationCase.label);
    constexpr uint64_t maxValuesPerRow = 4;
    const string networkName = "k3_neutral_" + activationCase.label + "_round_trip";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_k3_" + activationCase.label + "_neutral_" + to_string(nonce));
    filesystem::remove_all(archive);

    uint64_t savedValuesOriginalId = 0;
    uint64_t savedPartitionOriginalId = 0;
    uint64_t savedOutputOriginalId = 0;
    json savedLayerArchitecture;
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
        auto activation = buildActivation(activationCase.label, network, input);
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
        savedLayerArchitecture = activation->architectureJson();
        EXPECT_EQ(savedLayerArchitecture.at("version").get<string>(), "1.1.0");
        expectParameters(activationCase.label, savedLayerArchitecture);
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

        auto loadedActivation = findOnlyActivation(loaded, activationCase.layerType);
        ASSERT_NE(loadedActivation, nullptr);
        ASSERT_TRUE(loadedActivation->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedActivation->getFeatureInput().value(), loadedInput.getValues());
        EXPECT_EQ(loadedActivation->getFeatureOutput()->getOriginalId(), savedOutputOriginalId);
        EXPECT_EQ(loadedActivation->getRaggedFeatureOutput()->getRowPartitionToken(), loadedInput.getRowPartitionToken());
        expectParameters(activationCase.label, loadedActivation->architectureJson());

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

        const map<string, InferenceOutputValue> outputs = placed->inferLogical(makeBatch("tokens", loadedInput, placementBatchSize));
        ASSERT_TRUE(outputs.contains("activated"));
        ASSERT_TRUE(holds_alternative<ThorImplementation::RaggedTensor>(outputs.at("activated")));
        const ThorImplementation::RaggedTensor activated = get<ThorImplementation::RaggedTensor>(outputs.at("activated"));
        EXPECT_EQ(activated.getValues().getDimensions(), (vector<uint64_t>{expectedCapacity, 2}));
        EXPECT_EQ(activated.getOffsets().getDimensions(), (vector<uint64_t>{static_cast<uint64_t>(placementBatchSize) + 1}));
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
                const float expected = expectedValue(activationCase.label, inputValue(i));
                const float tolerance = 5e-4F * max(1.0F, fabs(expected));
                EXPECT_NEAR(values[i], expected, tolerance)
                    << activationCase.label << " mismatch at active element " << i;
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

void expectFixedRaggedCompatibility(const ActivationCase& activationCase) {
    SCOPED_TRACE(activationCase.label);
    const string networkName = "k3_fixed_" + activationCase.label + "_compatibility";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_k3_" + activationCase.label + "_fixed_" + to_string(nonce));
    filesystem::remove_all(archive);

    json savedArchitecture;
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
        auto activation = buildActivation(activationCase.label, network, input);
        ASSERT_NE(activation, nullptr);
        ASSERT_TRUE(activation->getRaggedFeatureOutput().has_value());
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("activated")
            .inputTensor(activation->getRaggedFeatureOutput().value())
            .build();
        savedArchitecture = activation->architectureJson();
        EXPECT_EQ(savedArchitecture.at("version").get<string>(), "1.0.0");
        EXPECT_EQ(savedArchitecture.at("ragged_feature_input").at("batch_size").get<uint64_t>(), 3U);
        EXPECT_EQ(savedArchitecture.at("ragged_feature_input").at("max_total_values").get<uint64_t>(), 9U);
        EXPECT_FALSE(savedArchitecture.at("ragged_feature_input").contains("layer_ragged_reference_version"));
        expectParameters(activationCase.label, savedArchitecture);
        network.save(archive.string(), true);
    }

    try {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        auto activation = findOnlyActivation(loaded, activationCase.layerType);
        ASSERT_NE(activation, nullptr);
        ASSERT_TRUE(activation->getRaggedFeatureInput().has_value());
        EXPECT_FALSE(activation->getRaggedFeatureInput()->isPlacementNeutral());
        EXPECT_EQ(activation->architectureJson().at("version").get<string>(), "1.0.0");
        expectParameters(activationCase.label, activation->architectureJson());
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}

void expectDenseCompatibility(const ActivationCase& activationCase) {
    SCOPED_TRACE(activationCase.label);
    const string networkName = "k3_dense_" + activationCase.label + "_compatibility";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_k3_" + activationCase.label + "_dense_" + to_string(nonce));
    filesystem::remove_all(archive);

    json savedArchitecture;
    {
        Network network(networkName);
        NetworkInput networkInput = NetworkInput::Builder()
                                        .network(network)
                                        .name("features")
                                        .dimensions({3})
                                        .dataType(DataType::FP32)
                                        .build();
        auto activation = buildActivation(activationCase.label, network, networkInput.getFeatureOutput().value());
        ASSERT_NE(activation, nullptr);
        (void)NetworkOutput::Builder()
            .network(network)
            .name("activated")
            .inputTensor(activation->getFeatureOutput().value())
            .dataType(DataType::FP32)
            .build();
        savedArchitecture = activation->architectureJson();
        EXPECT_EQ(savedArchitecture.at("version").get<string>(), "1.0.0");
        EXPECT_FALSE(savedArchitecture.value("use_ragged", false));
        expectParameters(activationCase.label, savedArchitecture);
        network.save(archive.string(), true);
    }

    try {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        auto activation = findOnlyActivation(loaded, activationCase.layerType);
        ASSERT_NE(activation, nullptr);
        EXPECT_FALSE(activation->getUseRagged());
        EXPECT_EQ(activation->architectureJson().at("version").get<string>(), "1.0.0");
        expectParameters(activationCase.label, activation->architectureJson());
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}

}  // namespace


// Test-only derived activation: exercises the *polymorphic* output contract
// without opting any of the five production GLUs into neutral serialization.
// No expression or physical execution is involved in this serialization test.
class G1PolymorphicOutputProbe final : public Relu {
   public:
    void initializeForTest(RaggedTensor input) {
        initializeStandaloneActivation(input);
        initialized = true;
    }

    void deserializeForTest(const json& architecture, Network* network) {
        deserializeStandaloneFields(architecture, network);
    }

   protected:
    Tensor standaloneOutputTensorForInput(const Tensor& input) const override {
        std::vector<uint64_t> dims = input.getDimensions();
        THOR_THROW_IF_FALSE(!dims.empty() && dims.back() >= 2 && (dims.back() % 2) == 0);
        dims.back() /= 2;
        return Tensor(input.getDataType(), dims);
    }
};

TEST(Activations, G1NeutralDeserializeAcceptsConcretePolymorphicOutput) {
    Network network("g1_neutral_polymorphic_output");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({6})
                             .maxValuesPerRow(4)
                             .build();
    G1PolymorphicOutputProbe source;
    source.initializeForTest(input);
    const json saved = source.architectureJson();
    EXPECT_EQ(saved.at("version").get<string>(), "1.1.0");
    EXPECT_EQ(saved.at("feature_output").at("dimensions").get<vector<uint64_t>>(),
              (vector<uint64_t>{4, 3}));

    G1PolymorphicOutputProbe reconstructed;
    ASSERT_NO_THROW(reconstructed.deserializeForTest(saved, &network));
    ASSERT_TRUE(reconstructed.getFeatureOutput().has_value());
    EXPECT_EQ(reconstructed.getFeatureOutput()->getOriginalId(),
              saved.at("feature_output").at("id").get<uint64_t>());
    EXPECT_EQ(reconstructed.getFeatureOutput()->getDataType(), DataType::FP32);
    EXPECT_EQ(reconstructed.getFeatureOutput()->getDimensions(), (vector<uint64_t>{4, 3}));
    ASSERT_TRUE(reconstructed.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(reconstructed.getRaggedFeatureOutput()->getValuesDimensions(),
              (vector<uint64_t>{4, 3}));
    EXPECT_EQ(reconstructed.getRaggedFeatureOutput()->getRowPartitionToken(), input.getRowPartitionToken());

    json forged = saved;
    forged["feature_output"]["dimensions"] = vector<uint64_t>{4, 4};
    forged["ragged_feature_output"]["values"]["dimensions"] = vector<uint64_t>{4, 4};
    G1PolymorphicOutputProbe rejected;
    EXPECT_THROW(rejected.deserializeForTest(forged, &network), runtime_error);
}

// Both serialized output descriptors are forged together, so their mutual
// consistency and partition IDs still look valid. The concrete activation's
// model-derived output contract must be checked independently of either JSON
// field; ordinary activations are shape preserving.
TEST(Activations, G1NeutralConcreteOutputRejectsPairedForgedDescriptors) {
    for (const ActivationCase& activationCase : cases) {
        SCOPED_TRACE(activationCase.label);
        Network network("g1_neutral_output_" + activationCase.label);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .maxValuesPerRow(4)
                                 .build();
        auto activation = buildActivation(activationCase.label, network, input);
        ASSERT_NE(activation, nullptr);
        const json original = activation->architectureJson();
        const auto outputId = original.at("feature_output").at("id").get<uint64_t>();
        const auto partitionId = original.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>();

        json forgedDtype = original;
        forgedDtype["feature_output"]["data_type"] = DataType::FP16;
        forgedDtype["ragged_feature_output"]["values"]["data_type"] = DataType::FP16;
        EXPECT_EQ(forgedDtype.at("feature_output").at("id").get<uint64_t>(), outputId);
        EXPECT_EQ(forgedDtype.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(), partitionId);
        EXPECT_THROW(Activation::deserialize(forgedDtype, &network), std::runtime_error);

        json forgedShape = original;
        forgedShape["feature_output"]["dimensions"] = std::vector<uint64_t>{4, 3};
        forgedShape["ragged_feature_output"]["values"]["dimensions"] = std::vector<uint64_t>{4, 3};
        EXPECT_EQ(forgedShape.at("feature_output").at("id").get<uint64_t>(), outputId);
        EXPECT_EQ(forgedShape.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>(), partitionId);
        EXPECT_THROW(Activation::deserialize(forgedShape, &network), std::runtime_error);
    }
}

TEST(Activations, K3OrdinaryPlacementNeutralRaggedActivationsSerializeSemanticGeometryAndParameters) {
    for (const ActivationCase& activationCase : cases) expectNeutralSemanticJsonAndMalformed(activationCase);
}

TEST(Activations, K3OrdinaryPlacementNeutralRaggedActivationsSaveLoadPlaceAndExecute) {
    for (const ActivationCase& activationCase : cases) expectNeutralSaveLoadPlaceAndExecute(activationCase);
}

TEST(Activations, K3OrdinaryFixedRaggedAndDenseActivationsRemainVersionOneAndLoad) {
    for (const ActivationCase& activationCase : cases) {
        expectFixedRaggedCompatibility(activationCase);
        expectDenseCompatibility(activationCase);
    }
}
