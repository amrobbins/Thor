#include <optional>
#include "DeepLearning/Api/Layers/Activations/Softmax.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "DeepLearning/Api/Data/Batch.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "Utilities/Expression/RaggedExpression.h"

#include "gtest/gtest.h"

#include <nlohmann/json.hpp>

#include <stdio.h>
#include <memory>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <variant>
#include <map>

using namespace Thor;
using namespace std;
using json = nlohmann::json;

namespace {
ThorImplementation::TensorPlacement softmaxCpuPlacement(ThorImplementation::TensorPlacement::MemDevices::CPU);

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

Batch makeNeutralSoftmaxBatch(const string& inputName, const RaggedTensor& logicalInput, uint32_t batchSize) {
    const ThorImplementation::RaggedTensorDescriptor descriptor = logicalInput.getDescriptorForPlacement(batchSize);
    const uint64_t capacity = descriptor.getMaxTotalValues();
    const uint64_t elementsPerValue = 2;
    THOR_THROW_IF_FALSE(logicalInput.getTrailingDimensions() == (vector<uint64_t>{elementsPerValue}));

    vector<uint64_t> hostOffsets(static_cast<size_t>(batchSize) + 1, 0);
    uint64_t activeValues = 0;
    for (uint32_t row = 0; row < batchSize; ++row) {
        const uint64_t rowLength = row == 0 ? 0 : 1 + ((row - 1) % 2);
        activeValues += rowLength;
        hostOffsets[row + 1] = activeValues;
    }
    THOR_THROW_IF_FALSE(activeValues <= capacity);

    ThorImplementation::Tensor values(softmaxCpuPlacement, descriptor.getValuesDescriptor());
    float* valuesPtr = values.getMemPtr<float>();
    for (uint64_t value = 0; value < capacity; ++value) {
        for (uint64_t element = 0; element < elementsPerValue; ++element) {
            const uint64_t i = value * elementsPerValue + element;
            valuesPtr[i] = value < activeValues
                ? ((i % 3 == 0) ? -static_cast<float>(i + 1) : static_cast<float>(i + 1))
                : -1000.0F;
        }
    }

    ThorImplementation::Tensor offsets(softmaxCpuPlacement, descriptor.getOffsetsDescriptor());
    THOR_THROW_IF_FALSE(logicalInput.getOffsetsDataType() == DataType::UINT32);
    uint32_t* offsetsPtr = offsets.getMemPtr<uint32_t>();
    for (size_t i = 0; i < hostOffsets.size(); ++i) offsetsPtr[i] = static_cast<uint32_t>(hostOffsets[i]);

    ThorImplementation::RowPartitionRuntime partition(offsets, descriptor.getRowPartition());
    partition.setHostOffsets(hostOffsets);
    Batch batch;
    batch.insert(inputName, ThorImplementation::RaggedTensor(values, partition));
    return batch;
}

class TestableSoftmax : public Softmax {
   public:
    void initializeNeutralForTest(RaggedTensor input) {
        initializeStandaloneActivation(input);
        initialized = true;
    }
    uint64_t memoryBytes(uint32_t batchSize) const { return getExpressionBackedActivationMemRequirementInBytes(batchSize); }
};

shared_ptr<Softmax> findOnlySoftmax(Network& network) {
    shared_ptr<Softmax> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        auto candidate = dynamic_pointer_cast<Softmax>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1U);
    return found;
}

} // namespace

TEST(Activations, PlacementNeutralSoftmaxSerializesSemanticGeometryAndRejectsMismatches) {
    constexpr uint64_t maxValuesPerRow = 4;
    Network network("placement_neutral_softmax_json");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({2})
                             .maxValuesPerRow(maxValuesPerRow)
                             .build();

    auto softmax = dynamic_pointer_cast<Softmax>(Softmax::Builder().network(network).featureInput(input).build());
    ASSERT_NE(softmax, nullptr);
    ASSERT_TRUE(softmax->getRaggedFeatureOutput().has_value());
    EXPECT_EQ(softmax->getRaggedFeatureOutput()->getRowPartitionToken(), input.getRowPartitionToken());
    EXPECT_EQ(softmax->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{maxValuesPerRow, 2}));

    TestableSoftmax testable;
    testable.initializeNeutralForTest(input);
    const uint64_t logicalValuesBytes = maxValuesPerRow * 2U * sizeof(float);
    EXPECT_EQ(testable.memoryBytes(2), 2U * logicalValuesBytes * 2U + sizeof(uint32_t));
    EXPECT_EQ(testable.memoryBytes(7), 7U * logicalValuesBytes * 2U + sizeof(uint32_t));

    const json architecture = softmax->architectureJson();
    EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    const json& inputJson = architecture.at("ragged_feature_input");
    const json& outputJson = architecture.at("ragged_feature_output");
    EXPECT_EQ(inputJson.at("layer_ragged_reference_version").get<string>(), "1.0.0");
    EXPECT_FALSE(inputJson.contains("geometry_mode"));
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

    json wrongVersion = architecture;
    wrongVersion["version"] = "0.0.0";
    EXPECT_THROW(Activation::deserialize(wrongVersion, &network), runtime_error);
    EXPECT_THROW(Softmax::deserialize(wrongVersion, &network), runtime_error);

    json wrongGeometry = architecture;
    wrongGeometry["ragged_feature_input"]["geometry_mode"] = "fixed";
    EXPECT_THROW(Activation::deserialize(wrongGeometry, &network), runtime_error);
    EXPECT_THROW(Softmax::deserialize(wrongGeometry, &network), runtime_error);

    json wrongPartition = architecture;
    wrongPartition["ragged_feature_output"]["offsets"]["id"] = input.getValues().getId();
    EXPECT_THROW(Activation::deserialize(wrongPartition, &network), runtime_error);
    EXPECT_THROW(Softmax::deserialize(wrongPartition, &network), runtime_error);

    json wrongOutputValues = architecture;
    wrongOutputValues["ragged_feature_output"]["values"]["id"] = input.getValues().getId();
    EXPECT_THROW(Activation::deserialize(wrongOutputValues, &network), runtime_error);
    EXPECT_THROW(Softmax::deserialize(wrongOutputValues, &network), runtime_error);

    json wrongInputValues = architecture;
    wrongInputValues["feature_input"]["id"] = architecture.at("feature_output").at("id");
    EXPECT_THROW(Activation::deserialize(wrongInputValues, &network), runtime_error);
    EXPECT_THROW(Softmax::deserialize(wrongInputValues, &network), runtime_error);
}

TEST(Activations, PlacementNeutralSoftmaxSaveLoadReplacesPlacementGeometryPerBatchAndExecutesEmptyRows) {
    constexpr uint64_t maxValuesPerRow = 4;
    const string networkName = "placement_neutral_softmax_round_trip";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_softmax_neutral_" + to_string(nonce));
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
        auto softmax = dynamic_pointer_cast<Softmax>(Softmax::Builder().network(network).featureInput(input).build());
        ASSERT_NE(softmax, nullptr);
        ASSERT_TRUE(softmax->getRaggedFeatureOutput().has_value());
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("activated")
            .inputTensor(softmax->getRaggedFeatureOutput().value())
            .build();

        savedValuesOriginalId = input.getValues().getOriginalId();
        savedPartitionOriginalId = input.getRowPartitionToken().getOriginalId();
        savedOutputOriginalId = softmax->getFeatureOutput()->getOriginalId();
        const json layerJson = softmax->architectureJson();
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
        EXPECT_EQ(loadedInput.getValues().getOriginalId(), savedValuesOriginalId);
        EXPECT_EQ(loadedInput.getRowPartitionToken().getOriginalId(), savedPartitionOriginalId);

        shared_ptr<Softmax> loadedSoftmax = findOnlySoftmax(loaded);
        ASSERT_NE(loadedSoftmax, nullptr);
        ASSERT_TRUE(loadedSoftmax->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedSoftmax->getFeatureInput().value(), loadedInput.getValues());
        EXPECT_EQ(loadedSoftmax->getFeatureOutput()->getOriginalId(), savedOutputOriginalId);
        EXPECT_EQ(loadedSoftmax->getRaggedFeatureOutput()->getRowPartitionToken(), loadedInput.getRowPartitionToken());
        EXPECT_EQ(loadedSoftmax->getRaggedFeatureOutput()->getValuesDimensions(),
                  (vector<uint64_t>{maxValuesPerRow, 2}));

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
        EXPECT_EQ(physicalInput->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{expectedCapacity, 2}));
        auto physicalSoftmax = dynamic_pointer_cast<ThorImplementation::RaggedCustomLayer>(
            stamped.getPhysicalLayerFromApiLayer(loadedSoftmax->getId()));
        ASSERT_NE(physicalSoftmax, nullptr);
        EXPECT_EQ(physicalSoftmax->getLayerType(), "RaggedCustomLayer");
        ASSERT_TRUE(physicalSoftmax->getFeatureOutput().has_value());
        EXPECT_EQ(physicalSoftmax->getFeatureOutput()->getDimensions(),
                  (vector<uint64_t>{expectedCapacity, 2}));
        EXPECT_EQ(loaded.architectureJson(), architectureBeforePlacement);

        const map<string, InferenceOutputValue> outputs =
            placed->inferLogical(makeNeutralSoftmaxBatch("tokens", loadedInput, placementBatchSize));
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
                const float original = (i % 3 == 0) ? -static_cast<float>(i + 1) : static_cast<float>(i + 1);
                const uint64_t start = value * 2;
                const float x0 = (start % 3 == 0) ? -static_cast<float>(start + 1) : static_cast<float>(start + 1);
                const float x1 = ((start + 1) % 3 == 0) ? -static_cast<float>(start + 2) : static_cast<float>(start + 2);
                const float maxX = std::max(x0, x1);
                const float denom = std::exp(x0 - maxX) + std::exp(x1 - maxX);
                const float expected = std::exp(original - maxX) / denom;
                EXPECT_NEAR(values[i], expected, 2e-3F);
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

TEST(Activations, FixedRaggedSoftmaxRemainsVersionOneAndLoads) {
    const string networkName = "fixed_ragged_softmax_compatibility";
    const auto nonce = chrono::steady_clock::now().time_since_epoch().count();
    const filesystem::path archive = filesystem::temp_directory_path() /
                                     ("thor_softmax_fixed_" + to_string(nonce));
    filesystem::remove_all(archive);

    {
        Network network(networkName);
        RaggedTensor input = RaggedNetworkInput::Builder()
                                 .network(network)
                                 .name("tokens")
                                 .valuesDataType(DataType::FP32)
                                 .offsetsDataType(DataType::UINT32)
                                 .trailingDimensions({2})
                                 .maxValuesPerRow(4)
                                 .build();
        auto softmax = dynamic_pointer_cast<Softmax>(Softmax::Builder().network(network).featureInput(input).build());
        ASSERT_NE(softmax, nullptr);
        ASSERT_TRUE(softmax->getRaggedFeatureOutput().has_value());
        (void)RaggedNetworkOutput::Builder()
            .network(network)
            .name("activated")
            .inputTensor(softmax->getRaggedFeatureOutput().value())
            .build();
        const json architecture = softmax->architectureJson();
        EXPECT_EQ(architecture.at("version").get<string>(), "1.1.0");
        EXPECT_EQ(architecture.at("ragged_feature_input").at("max_values_per_row").get<uint64_t>(), 4U);
        EXPECT_FALSE(architecture.at("ragged_feature_input").contains("batch_size"));
        EXPECT_FALSE(architecture.at("ragged_feature_input").contains("max_total_values"));
        EXPECT_EQ(architecture.at("ragged_feature_input").at("offsets").at("dimensions"), json::array({1}));
        network.save(archive.string(), true);
    }

    try {
        Network loaded(networkName);
        ASSERT_NO_THROW(loaded.load(archive.string()));
        shared_ptr<Softmax> softmax = findOnlySoftmax(loaded);
        ASSERT_NE(softmax, nullptr);
        ASSERT_TRUE(softmax->getRaggedFeatureInput().has_value());
        EXPECT_EQ(softmax->architectureJson().at("version").get<string>(), "1.1.0");
    } catch (...) {
        filesystem::remove_all(archive);
        throw;
    }
    filesystem::remove_all(archive);
}

TEST(Activations, PlacementNeutralSoftmaxRejectsFp8Input) {
    Network network("neutral_softmax_fp8");
    RaggedTensor input = RaggedNetworkInput::Builder()
        .network(network).name("tokens")
        .valuesDataType(DataType::FP8_E4M3).offsetsDataType(DataType::UINT32)
        .trailingDimensions({2}).maxValuesPerRow(4).build();
    EXPECT_THROW(Softmax::Builder().network(network).featureInput(input).build(), invalid_argument);
}

TEST(Activations, SoftmaxBuilds) {
    srand(time(nullptr));

    Network network("testNetwork");

    vector<uint64_t> dimensions;
    int numDimensions = 1 + rand() % 6;
    for (int i = 0; i < numDimensions; ++i)
        dimensions.push_back(1 + (rand() % 1000));

    DataType dataType = rand() % 2 ? DataType::FP32 : DataType::FP16;

    Tensor featureInput(dataType, dimensions);
    Softmax::Builder softmaxBuilder;
    softmaxBuilder.network(network);
    softmaxBuilder.featureInput(featureInput);
    shared_ptr<Softmax> softmax = dynamic_pointer_cast<Softmax>(softmaxBuilder.build());

    ASSERT_TRUE(softmax->isInitialized());

    std::optional<Tensor> actualInput = softmax->getFeatureInput();
    ASSERT_TRUE(actualInput.has_value());
    ASSERT_EQ(actualInput.value().getDataType(), dataType);
    ASSERT_EQ(actualInput.value().getDimensions(), dimensions);

    std::optional<Tensor> actualOutput = softmax->getFeatureOutput();
    ASSERT_TRUE(actualOutput.has_value());
    ASSERT_EQ(actualOutput.value().getDataType(), dataType);
    ASSERT_EQ(actualOutput.value().getDimensions(), dimensions);

    shared_ptr<Layer> cloneLayer = softmax->clone();
    Softmax *clone = dynamic_cast<Softmax *>(cloneLayer.get());
    assert(clone != nullptr);

    ASSERT_TRUE(clone->isInitialized());

    std::optional<Tensor> cloneInput = clone->getFeatureInput();
    ASSERT_TRUE(cloneInput.has_value());
    ASSERT_EQ(cloneInput.value().getDataType(), dataType);
    ASSERT_EQ(cloneInput.value().getDimensions(), dimensions);

    std::optional<Tensor> cloneOutput = clone->getFeatureOutput();
    ASSERT_TRUE(cloneOutput.has_value());
    ASSERT_EQ(cloneOutput.value().getDataType(), dataType);
    ASSERT_EQ(cloneOutput.value().getDimensions(), dimensions);

    ASSERT_NE(softmax->getId(), clone->getId());
    ASSERT_GT(softmax->getId(), 1u);
}


TEST(Activations, SoftmaxRejectsFp8StandaloneDenseAndRaggedInputs) {
    for (DataType dtype : {DataType::FP8_E4M3, DataType::FP8_E5M2}) {
        {
            Network network("denseFp8SoftmaxReject");
            Tensor input(dtype, {4, 7});
            Softmax::Builder builder;
            builder.network(network);
            EXPECT_THROW(builder.featureInput(input), std::invalid_argument);
        }
        {
            Network network("raggedFp8SoftmaxReject");
            RaggedTensor input = RaggedTensor::logical(
                Tensor(dtype, {4, 7}), Tensor(DataType::UINT64, {1}), 4);
            Softmax::Builder builder;
            builder.network(network);
            EXPECT_THROW(builder.featureInput(input), std::invalid_argument);
        }
    }
}

TEST(Activations, RaggedSoftmaxBuildCloneAndExpressionPreservePartitionWithOrdinaryFinalAxisSemantics) {
    Network network("raggedSoftmaxBuildCloneAndExpression");
    RaggedTensor featureInput = RaggedTensor::logical(
        Tensor(DataType::FP32, {8, 2, 3}), Tensor(DataType::UINT64, {1}), 8);

    shared_ptr<Softmax> softmax = dynamic_pointer_cast<Softmax>(
        Softmax::Builder().network(network).featureInput(featureInput).build());
    ASSERT_NE(softmax, nullptr);
    ASSERT_TRUE(softmax->isInitialized());
    EXPECT_TRUE(softmax->supportsRaggedStandalone());
    EXPECT_FALSE(softmax->supportsRaggedLearningLayerFusion());
    EXPECT_TRUE(softmax->getUseRagged());
    ASSERT_TRUE(softmax->getRaggedFeatureInput().has_value());
    ASSERT_TRUE(softmax->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(softmax->getRaggedFeatureInput()->sharesPartitionWith(featureInput));
    EXPECT_TRUE(softmax->getRaggedFeatureOutput()->sharesPartitionWith(featureInput));
    EXPECT_EQ(softmax->getRaggedFeatureOutput()->getRowPartitionToken(), featureInput.getRowPartitionToken());
    EXPECT_EQ(softmax->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{8, 2, 3}));

    shared_ptr<Layer> cloneLayer = softmax->clone();
    auto* clone = dynamic_cast<Softmax*>(cloneLayer.get());
    ASSERT_NE(clone, nullptr);
    ASSERT_TRUE(clone->getRaggedFeatureInput().has_value());
    ASSERT_TRUE(clone->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(clone->getRaggedFeatureInput()->sharesPartitionWith(featureInput));
    EXPECT_TRUE(clone->getRaggedFeatureOutput()->sharesPartitionWith(featureInput));
    EXPECT_NE(clone->getId(), softmax->getId());

    const json architecture = softmax->architectureJson();
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    EXPECT_EQ(architecture.at("ragged_feature_input").at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_feature_output").at("offsets").at("id").get<uint64_t>());

    ThorImplementation::RaggedTensorDescriptor descriptor(
        DataType::FP32, {2, 3}, 2, 8, DataType::UINT64);
    ThorImplementation::RaggedExpression inputExpression =
        ThorImplementation::RaggedExpression::input("tokens.values",
                                                     "tokens.active_count",
                                                     descriptor,
                                                     ThorImplementation::RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT);
    ThorImplementation::RaggedExpression outputExpression = softmax->toRaggedExpression(inputExpression);
    EXPECT_EQ(outputExpression.getDescriptor(), descriptor);
    EXPECT_TRUE(outputExpression.getOffsets().isSameLogicalNode(inputExpression.getOffsets()));

    const ThorImplementation::PhysicalExpression physical = outputExpression.getValues().expression();
    const ThorImplementation::ExprNode& extent = physical.nodes.at(physical.output_node);
    ASSERT_EQ(extent.op, ThorImplementation::ExprOp::RAGGED_VALUEWISE_EXTENT);
    ASSERT_LT(extent.lhs, physical.nodes.size());
    const ThorImplementation::ExprNode& ordinarySoftmax = physical.nodes.at(extent.lhs);
    ASSERT_EQ(ordinarySoftmax.op, ThorImplementation::ExprOp::SOFTMAX);
    EXPECT_EQ(ordinarySoftmax.softmax_algorithm, CUDNN_SOFTMAX_ACCURATE);
    EXPECT_EQ(ordinarySoftmax.softmax_mode, CUDNN_SOFTMAX_MODE_CHANNEL);
}


TEST(Activations, RaggedSoftmaxStandaloneMaterializesManagedActiveCountWithoutFullOffsets) {
    constexpr uint32_t batchSize = 3;
    Network network("raggedSoftmaxManagedActiveCount");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT64)
                             .trailingDimensions({5})
                             .maxValuesPerRow(5)
                             .build();

    auto softmax = dynamic_pointer_cast<Softmax>(
        Softmax::Builder().network(network).featureInput(input).build());
    ASSERT_NE(softmax, nullptr);
    ASSERT_TRUE(softmax->getRaggedFeatureOutput().has_value());
    EXPECT_TRUE(softmax->getRaggedFeatureOutput()->sharesPartitionWith(input));
    RaggedNetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(softmax->getRaggedFeatureOutput().value())
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true);
    ASSERT_NE(placed, nullptr);
    for (Event& event : initDoneEvents) event.synchronize();

    const auto& stamp = placed->getStampedNetwork(0);
    EXPECT_EQ(stamp.getManagedPartitionOffsetsInputForTest(input.getRowPartitionId()), nullptr);
    auto activeCount = stamp.getManagedPartitionActiveCountInputForTest(input.getRowPartitionId());
    ASSERT_NE(activeCount, nullptr);
    ASSERT_TRUE(activeCount->getFeatureOutput().has_value());
    EXPECT_EQ(activeCount->getFeatureOutput()->getDimensions(), (vector<uint64_t>{1}));
    EXPECT_EQ(activeCount->getFeatureOutput()->getDataType(), DataType::UINT64);
}

TEST(Activations, SoftmaxSerializeDeserialize) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    DataType dataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1;
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Softmax::Builder softmaxBuilder = Softmax::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput().value());
    shared_ptr<Softmax> softmax = dynamic_pointer_cast<Softmax>(softmaxBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(softmax->getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(softmax->isInitialized());

    Tensor featureInput = softmax->getFeatureInput().value();
    Tensor featureOutput = softmax->getFeatureOutput().value();
    assert(featureInput == networkInput.getFeatureOutput());

    ASSERT_TRUE(softmax->getFeatureOutput().has_value());
    ASSERT_EQ(softmax->getFeatureOutput().value(), featureOutput);

    ASSERT_TRUE(softmax->getFeatureInput().has_value());
    assert(softmax->getFeatureInput().value() == featureInput);

    ASSERT_EQ(featureInput.getDataType(), dataType);
    ASSERT_EQ(featureInput.getDimensions(), inputDimensions);

    ASSERT_EQ(featureOutput.getDataType(), dataType);
    ASSERT_EQ(featureOutput.getDimensions(), inputDimensions);

    // Now stamp the network and test serialization
    Stream stream(0);
    uint32_t batchSize = 1 + (rand() % 16);
    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> initialPlacedNetwork = initialNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(initialPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    // Fetch the layer from the network
    ASSERT_EQ(initialPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = initialPlacedNetwork->getStampedNetwork(0);

    thor_file::TarWriter archiveWriter("testModel");

    json softmaxJ = softmax->serialize(archiveWriter, stream);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Ensure polymorphism is properly wired and that we get the same result when serializing from the base class
    Layer *layer = softmax.get();
    json fromLayerJ = layer->serialize(archiveWriter, stream);
    ASSERT_EQ(softmaxJ, fromLayerJ);

    ASSERT_EQ(softmaxJ["factory"], "activation");
    ASSERT_EQ(softmaxJ["version"], "1.0.0");
    ASSERT_EQ(softmaxJ["layer_type"], "softmax");

    EXPECT_TRUE(softmaxJ.contains("feature_input"));
    EXPECT_TRUE(softmaxJ.contains("feature_output"));

    const auto &input = softmaxJ.at("feature_input");
    ASSERT_TRUE(input.is_object());
    ASSERT_TRUE(input.at("data_type").is_string());
    string dataTypeString = dataType == DataType::FP16 ? "fp16" : "fp32";
    EXPECT_EQ(input.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(input.at("dimensions").is_array());
    ASSERT_EQ(input.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(input.at("id").is_number_integer());

    const auto &output = softmaxJ.at("feature_output");
    ASSERT_TRUE(output.is_object());
    ASSERT_TRUE(output.at("data_type").is_string());
    EXPECT_EQ(output.at("data_type").get<string>(), dataTypeString);
    ASSERT_TRUE(output.at("dimensions").is_array());
    ASSERT_EQ(output.at("dimensions").get<vector<uint64_t>>(), inputDimensions);
    ASSERT_TRUE(output.at("id").is_number_integer());

    //     printf("%s\n", networkInputJ.dump(4).c_str());
    //     printf("%s\n", softmaxJ.dump(4).c_str());
    //     printf("%s\n", networkOutputJ.dump(4).c_str());

    ////////////////////////////
    // Deserialize
    ////////////////////////////
    // Verify that the layer gets added to the network and that its weights are set to the correct values
    Network newNetwork("newNetwork");

    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Softmax::deserialize(softmaxJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    batchSize = 1 + (rand() % 16);
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &newStamp = newPlacedNetwork->getStampedNetwork(0);

    ASSERT_EQ(newStamp.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::CustomLayer> stampedSoftmax = dynamic_pointer_cast<ThorImplementation::CustomLayer>(newStamp.getTrainableLayer(0));
    ASSERT_NE(stampedSoftmax, nullptr);
    ASSERT_EQ(stampedSoftmax->getLayerType(), "CustomLayer<Softmax>");

    vector<shared_ptr<ThorImplementation::NetworkInput>> inputLayers = newStamp.getInputs();
    ASSERT_EQ(inputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkInput> stampedInput = dynamic_pointer_cast<ThorImplementation::NetworkInput>(inputLayers[0]);
    ASSERT_NE(inputLayers[0], nullptr);

    vector<shared_ptr<ThorImplementation::NetworkOutput>> outputLayers = newStamp.getOutputs();
    ASSERT_EQ(outputLayers.size(), 1U);
    shared_ptr<ThorImplementation::NetworkOutput> stampedOutput = dynamic_pointer_cast<ThorImplementation::NetworkOutput>(outputLayers[0]);
    ASSERT_NE(outputLayers[0], nullptr);

    ASSERT_TRUE(stampedInput->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedSoftmax->getFeatureOutput().has_value());
    ASSERT_TRUE(stampedOutput->getFeatureOutput().has_value());
    ASSERT_EQ(stampedInput->getFeatureOutput().value(), stampedSoftmax->getFeatureInput().value());
    ASSERT_EQ(stampedSoftmax->getFeatureOutput().value(), stampedOutput->getFeatureInput().value());

    filesystem::remove("/tmp/testModel.thor.tar");
}

TEST(Activations, SoftmaxRegistered) {
    srand(time(nullptr));

    Network initialNetwork("initialNetwork");
    DataType dataType = rand() % 2 ? DataType::FP16 : DataType::FP32;
    vector<uint64_t> inputDimensions;
    uint32_t numDimensions = 1;
    for (uint32_t i = 0; i < numDimensions; ++i)
        inputDimensions.push_back(1 + (rand() % 5));

    NetworkInput networkInput =
        NetworkInput::Builder().network(initialNetwork).name("testInput").dimensions(inputDimensions).dataType(dataType).build();

    Softmax::Builder softmaxBuilder = Softmax::Builder().network(initialNetwork).featureInput(networkInput.getFeatureOutput().value());
    shared_ptr<Softmax> softmax = dynamic_pointer_cast<Softmax>(softmaxBuilder.build());

    NetworkOutput networkOutput = NetworkOutput::Builder()
                                      .network(initialNetwork)
                                      .name("testOutput")
                                      .inputTensor(softmax->getFeatureOutput().value())
                                      .dataType(dataType)
                                      .build();

    ASSERT_TRUE(softmax->isInitialized());

    thor_file::TarWriter archiveWriter("testModel");

    Stream stream(0);
    json networkInputJ = networkInput.serialize(archiveWriter, stream);
    json softmaxJ = softmax->serialize(archiveWriter, stream);
    json networkOutputJ = networkOutput.serialize(archiveWriter, stream);

    // Test that it is registered with Activation to deserialize
    Network newNetwork("newNetwork");
    NetworkInput::deserialize(networkInputJ, &newNetwork);
    Activation::deserialize(softmaxJ, &newNetwork);
    NetworkOutput::deserialize(networkOutputJ, &newNetwork);

    vector<Event> initDoneEvents;
    uint32_t batchSize = 1 + (rand() % 16);
    shared_ptr<PlacedNetwork> newPlacedNetwork = newNetwork.place(batchSize, initDoneEvents);
    ASSERT_TRUE(newPlacedNetwork != nullptr);
    for (uint32_t i = 0; i < initDoneEvents.size(); ++i) {
        stream.waitEvent(initDoneEvents[i]);
    }
    initDoneEvents.clear();

    ASSERT_EQ(newPlacedNetwork->getNumStamps(), 1UL);
    ThorImplementation::StampedNetwork &stampedNetwork = newPlacedNetwork->getStampedNetwork(0);

    ASSERT_EQ(stampedNetwork.getNumTrainableLayers(), 1UL);
    shared_ptr<ThorImplementation::CustomLayer> stampedSoftmax = dynamic_pointer_cast<ThorImplementation::CustomLayer>(stampedNetwork.getTrainableLayer(0));
    ASSERT_NE(stampedSoftmax, nullptr);
    ASSERT_EQ(stampedSoftmax->getLayerType(), "CustomLayer<Softmax>");
    filesystem::remove("/tmp/testModel.thor.tar");
}
