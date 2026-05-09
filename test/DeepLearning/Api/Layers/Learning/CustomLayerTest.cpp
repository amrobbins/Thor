#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "Utilities/Expression/Expression.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::TensorDescriptor::DataType;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t tensorNumel(const Impl::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t d : tensor.getDimensions())
        numel *= d;
    return numel;
}

void synchronizeEvents(vector<Event>& events) {
    for (Event& event : events)
        event.synchronize();
    events.clear();
}

void writeCpuTensor(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);
    ASSERT_EQ(tensorNumel(tensor), values.size());

    auto* ptr = static_cast<float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

vector<float> readCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);

    vector<float> values(tensorNumel(tensor));
    const auto* ptr = static_cast<const float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];
    return values;
}

void expectAllClose(const vector<float>& actual, const vector<float>& expected, float atol = 1e-5f, float rtol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        EXPECT_LE(diff, tol) << "mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

std::filesystem::path makeUniqueTestArchiveDir(const std::string& testName) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::filesystem::path dir = std::filesystem::temp_directory_path() / (testName + "_" + std::to_string(now));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

template <typename LayerT>
std::shared_ptr<LayerT> findOnlyLayerOfType(Api::Network& network) {
    std::shared_ptr<LayerT> found;
    uint32_t count = 0;
    for (uint32_t i = 0; i < network.getNumLayers(); ++i) {
        std::shared_ptr<LayerT> candidate = std::dynamic_pointer_cast<LayerT>(network.getLayer(i));
        if (candidate != nullptr) {
            found = candidate;
            ++count;
        }
    }
    EXPECT_EQ(count, 1u);
    return found;
}

Impl::DynamicExpression makeSerializableAffineExpression() {
    Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
    Impl::Expression y = x * 3.0f + 2.0f;
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"y", y}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

struct PlacedCustomLayerFixture {
    std::shared_ptr<Api::PlacedNetwork> placedNetwork;
    ThorImplementation::StampedNetwork* stampedNetwork = nullptr;
    std::shared_ptr<Impl::NetworkInput> physicalInput;
    std::shared_ptr<Impl::NetworkOutput> physicalOutput;
    std::shared_ptr<Impl::CustomLayer> physicalCustomLayer;
};

PlacedCustomLayerFixture placeSingleCustomLayerNetwork(Api::Network& network,
                                                       const Api::NetworkInput& apiInput,
                                                       const Api::NetworkOutput& apiOutput,
                                                       const Api::CustomLayer& apiCustomLayer,
                                                       uint32_t batchSize,
                                                       bool inferenceOnly) {
    vector<Event> initDoneEvents;
    PlacedCustomLayerFixture fixture;
    fixture.placedNetwork = network.place(batchSize, initDoneEvents, inferenceOnly);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(fixture.placedNetwork, nullptr);
    fixture.stampedNetwork = &fixture.placedNetwork->getStampedNetwork(0);

    fixture.physicalInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiInput.getId()));
    fixture.physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiOutput.getId()));
    fixture.physicalCustomLayer =
        dynamic_pointer_cast<Impl::CustomLayer>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiCustomLayer.getId()));

    EXPECT_NE(fixture.physicalInput, nullptr);
    EXPECT_NE(fixture.physicalOutput, nullptr);
    EXPECT_NE(fixture.physicalCustomLayer, nullptr);
    return fixture;
}

vector<float> runForward(Impl::NetworkInput& physicalInput,
                         Impl::NetworkOutput& physicalOutput,
                         Impl::Tensor& featureInHost,
                         uint32_t batchSize) {
    physicalInput.forward(featureInHost, false, batchSize);
    Event featureOutReadyEvent = physicalOutput.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    return readCpuTensor(physicalOutput.getFeatureOutput().value());
}

}  // namespace

TEST(CustomLayerApi, SerializableExpressionDefinitionSaveLoadRoundTripPreservesExpressionAndRuns) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numFeatures = 3;
    const DataType dataType = DataType::FP32;
    const vector<float> inputValues = {1.0f, -2.0f, 0.5f, -1.0f, 3.0f, 2.0f};
    const vector<float> expectedValues = {5.0f, -4.0f, 3.5f, -1.0f, 11.0f, 8.0f};

    const std::string networkName = "custom_layer_serializable_round_trip";
    std::filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input =
            Api::NetworkInput::Builder().network(network).name("input").dimensions({numFeatures}).dataType(dataType).build();
        Api::CustomLayer customLayer = Api::CustomLayer::Builder()
                                           .network(network)
                                           .expression(makeSerializableAffineExpression())
                                           .inputNames({"x"})
                                           .outputNames({"y"})
                                           .inputInterface({{"x", input.getFeatureOutput().value()}})
                                           .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(customLayer.getOutput("y"))
                                        .dataType(dataType)
                                        .build();

        const nlohmann::json beforeSaveJson = customLayer.architectureJson();
        EXPECT_EQ(beforeSaveJson.at("layer_type").get<string>(), "custom_layer");
        EXPECT_EQ(beforeSaveJson.at("input_names").get<vector<string>>(), (vector<string>{"x"}));
        EXPECT_EQ(beforeSaveJson.at("output_names").get<vector<string>>(), (vector<string>{"y"}));
        ASSERT_FALSE(beforeSaveJson.at("expression").is_null());

        network.save(archiveDir.string(), true);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        ASSERT_EQ(loadedNetwork.getNumLayers(), 3u);
        std::shared_ptr<Api::NetworkInput> loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        std::shared_ptr<Api::CustomLayer> loadedCustomLayer = findOnlyLayerOfType<Api::CustomLayer>(loadedNetwork);
        std::shared_ptr<Api::NetworkOutput> loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedCustomLayer, nullptr);
        ASSERT_NE(loadedOutput, nullptr);

        const nlohmann::json loadedJson = loadedCustomLayer->architectureJson();
        EXPECT_EQ(loadedJson.at("layer_type").get<string>(), "custom_layer");
        EXPECT_EQ(loadedJson.at("input_names").get<vector<string>>(), (vector<string>{"x"}));
        EXPECT_EQ(loadedJson.at("output_names").get<vector<string>>(), (vector<string>{"y"}));
        ASSERT_FALSE(loadedJson.at("expression").is_null());

        PlacedCustomLayerFixture fixture =
            placeSingleCustomLayerNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedCustomLayer, batchSize, true);
        ASSERT_EQ(fixture.stampedNetwork->getNumTrainableLayers(), 1u);
        EXPECT_EQ(fixture.physicalCustomLayer->getLayerType(), "CustomLayer<CustomLayer>");
        EXPECT_TRUE(fixture.physicalCustomLayer->listParameters().empty());

        Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numFeatures}));
        writeCpuTensor(featureInHost, inputValues);

        const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
        expectAllClose(actual, expectedValues);
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}
