#include "DeepLearning/Api/BatchValidity.h"
#include "DeepLearning/Api/Layers/Learning/CustomLayer.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Training/PhaseGraphConnector.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/RaggedCustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "Utilities/Expression/CudaKernelExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/ComputeTopology/MachineEvaluator.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

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

void writeDeviceTensor(Impl::Tensor tensor, const vector<float>& values, Stream& stream) {
    Impl::Tensor host = tensor.clone(cpuPlacement);
    writeCpuTensor(host, values);
    tensor.copyFromAsync(host, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
}

vector<float> readDeviceTensor(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor host = tensor.clone(cpuPlacement);
    host.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return readCpuTensor(host);
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

Impl::DynamicExpression makeSerializableLogicalMaskExpression() {
    Impl::Expression labels = Impl::Expression::input("labels", DataType::FP32, DataType::FP32);
    Impl::Expression initial = Impl::Expression::input("initial_daily_units", DataType::FP32, DataType::FP32);
    Impl::Expression zero = Impl::Expression::constantScalar(0.0);
    Impl::Expression two = Impl::Expression::constantScalar(2.0);

    Impl::Expression positive = labels > zero;
    Impl::Expression initialPositive = initial > zero;
    Impl::Expression spike = labels > (initial * two);

    Impl::Expression notPositive = positive.logicalNot();
    Impl::Expression positiveAndInitial = positive.logicalAnd(initialPositive);
    Impl::Expression positiveOrInitial = positive.logicalOr(initialPositive);
    Impl::Expression positiveAndNotInitial = positive.logicalAnd(initialPositive.logicalNot());
    Impl::Expression typicalPositive = positive.logicalAnd(spike.logicalNot());

    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({
        {"not_positive", notPositive},
        {"positive_and_initial", positiveAndInitial},
        {"positive_or_initial", positiveOrInitial},
        {"positive_and_not_initial", positiveAndNotInitial},
        {"not_positive_fp32", notPositive.cast(DataType::FP32)},
        {"positive_and_initial_fp32", positiveAndInitial.cast(DataType::FP32)},
        {"positive_or_initial_fp32", positiveOrInitial.cast(DataType::FP32)},
        {"positive_and_not_initial_fp32", positiveAndNotInitial.cast(DataType::FP32)},
        {"zero_mask", (labels == zero).cast(DataType::FP32)},
        {"typical_positive_mask", typicalPositive.cast(DataType::FP32)},
        {"spike_mask", spike.cast(DataType::FP32)},
    }));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

Impl::DynamicExpression makeSerializableConditionalCudaExpression() {
    auto kernel = Impl::CudaKernelExpression::builder("custom_layer_conditional_scale")
                      .source(R"cuda(
extern "C" __global__
void custom_layer_conditional_scale_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 2.0f * x[i];
    }
}
)cuda")
                      .entry("custom_layer_conditional_scale_kernel")
                      .input("x", DataType::FP32)
                      .outputLike("y", DataType::FP32, "x")
                      .scalar("n", DataType::INT64, Impl::CudaKernelExpression::DimExpr::numel("y"))
                      .launchGrid1D(Impl::CudaKernelExpression::DimExpr::numel("y"), 128)
                      .build();

    Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
    Impl::Expression predicate = x.reduce_sum().greaterThan(Impl::Expression::constantScalar(0.0));
    Impl::Outputs cudaBranch = kernel.apply({{"x", x}});
    Impl::Outputs ordinaryBranch = Impl::Expression::outputs({{"y", x - Impl::Expression::constantScalar(5.0)}});
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Outputs::conditional(predicate, cudaBranch, ordinaryBranch));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

Impl::DynamicExpression makeSerializableValidityMaskedExpression() {
    Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
    Impl::Expression validity =
        Impl::Expression::input(Api::BATCH_VALIDITY_MASK_NAME, DataType::FP32, DataType::FP32);
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"y", x * validity}}));
    return Impl::DynamicExpression::fromExpressionDefinition(definition);
}

Impl::DynamicExpression makeRuntimeOnlyAffineExpression() {
    return Impl::DynamicExpression(
        {"x"},
        {"y"},
        [](const Impl::DynamicExpression::TensorMap& inputs,
           const Impl::DynamicExpression::TensorMap& outputs,
           Stream& stream) -> Impl::DynamicExpressionBuild {
            Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
            Impl::Expression y = x + 1.0f;
            Impl::Outputs expressionOutputs = Impl::Expression::outputs({{"y", y}});
            return Impl::DynamicExpressionBuild{
                .equation = std::make_shared<Impl::FusedEquation>(
                    Impl::FusedEquation::compile(expressionOutputs.physicalOutputs(), stream.getGpuNum())),
                .stamp_inputs = inputs,
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = {},
                .serialized_definition = nullptr,
            };
        });
}

Impl::DynamicExpression makeBatchDependentConditionalCudaExpression() {
    return Impl::DynamicExpression(
        {"x"},
        {"y"},
        [](const Impl::DynamicExpression::TensorMap& inputs,
           const Impl::DynamicExpression::TensorMap& outputs,
           Stream& stream) -> Impl::DynamicExpressionBuild {
            const std::vector<uint64_t> dimensions = inputs.at("x").getDimensions();
            if (dimensions.size() != 2) {
                throw std::runtime_error("Batch-dependent conditional CUDA expression expects rank-2 input.");
            }

            auto kernel = Impl::CudaKernelExpression::builder("batch_dependent_conditional_scale")
                              .source(R"cuda(
extern "C" __global__
void batch_dependent_conditional_scale_kernel(const float* x, float* y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 2.0f * x[i];
}
)cuda")
                              .entry("batch_dependent_conditional_scale_kernel")
                              .input("x", DataType::FP32)
                              .output("y",
                                      DataType::FP32,
                                      {Impl::CudaKernelExpression::DimExpr::constant(dimensions[0]),
                                       Impl::CudaKernelExpression::DimExpr::constant(dimensions[1])})
                              .scalar("n", DataType::INT64, Impl::CudaKernelExpression::DimExpr::numel("y"))
                              .launchGrid1D(Impl::CudaKernelExpression::DimExpr::numel("y"), 128)
                              .build();

            Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
            Impl::Expression predicate = x.reduce_sum().greaterThan(Impl::Expression::constantScalar(0.0));
            Impl::Outputs conditional = Impl::Outputs::conditional(
                predicate,
                kernel.apply({{"x", x}}),
                Impl::Expression::outputs({{"y", x}}));
            Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(conditional);
            auto serializedDefinition = std::make_shared<Impl::ExpressionDefinition>(std::move(definition));
            return Impl::DynamicExpressionBuild{
                .equation = std::make_shared<Impl::FusedEquation>(
                    Impl::FusedEquation::compile(serializedDefinition->outputs, stream.getGpuNum())),
                .stamp_inputs = inputs,
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = {},
                .serialized_definition = serializedDefinition,
            };
        });
}

Impl::DynamicExpression makeBatchLiteralTailExpression() {
    return Impl::DynamicExpression(
        {"sequence"},
        {"tail"},
        [](const Impl::DynamicExpression::TensorMap& inputs,
           const Impl::DynamicExpression::TensorMap& outputs,
           Stream& stream) -> Impl::DynamicExpressionBuild {
            const Impl::Tensor& sequenceTensor = inputs.at("sequence");
            const vector<uint64_t> dimensions = sequenceTensor.getDimensions();
            if (dimensions.size() != 3 || dimensions[1] < 3)
                throw runtime_error("Tail expression expects [batch, sequence>=3, channels].");
            const vector<uint64_t> strides = sequenceTensor.getStridesElements();
            Impl::Expression sequence = Impl::Expression::input("sequence");
            Impl::Expression tail = sequence.stridedView(
                {dimensions[0], 3, dimensions[2]}, strides, (dimensions[1] - 3) * strides[1]);
            Impl::ExpressionDefinition definition =
                Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"tail", tail}}));
            auto serializedDefinition = std::make_shared<Impl::ExpressionDefinition>(std::move(definition));
            return Impl::DynamicExpressionBuild{
                .equation = std::make_shared<Impl::FusedEquation>(
                    Impl::FusedEquation::compile(serializedDefinition->outputs, stream.getGpuNum())),
                .stamp_inputs = inputs,
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = {},
                .serialized_definition = serializedDefinition,
            };
        });
}

Impl::DynamicExpression makeBatchProductReshapeExpression() {
    return Impl::DynamicExpression(
        {"x"},
        {"y"},
        [](const Impl::DynamicExpression::TensorMap& inputs,
           const Impl::DynamicExpression::TensorMap& outputs,
           Stream& stream) -> Impl::DynamicExpressionBuild {
            const vector<uint64_t> dimensions = inputs.at("x").getDimensions();
            if (dimensions.size() != 3)
                throw runtime_error("Batch-product reshape expression expects [batch, sequence, channels].");

            Impl::Expression x = Impl::Expression::input("x");
            Impl::Expression flat = x.reshape({dimensions[0] * dimensions[1], dimensions[2]});
            Impl::Expression y = flat.reshape(dimensions);
            Impl::ExpressionDefinition definition =
                Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"y", y}}));
            auto serializedDefinition = std::make_shared<Impl::ExpressionDefinition>(std::move(definition));
            return Impl::DynamicExpressionBuild{
                .equation = std::make_shared<Impl::FusedEquation>(
                    Impl::FusedEquation::compile(serializedDefinition->outputs, stream.getGpuNum())),
                .stamp_inputs = inputs,
                .tensor_scalar_inputs = {},
                .preallocated_outputs = outputs,
                .requested_output_shapes = {},
                .pre_forward_hook = {},
                .serialized_definition = serializedDefinition,
            };
        });
}

bool jsonContainsArrayField(const nlohmann::json& value,
                            const std::string& field,
                            const std::vector<uint64_t>& expected) {
    if (value.is_object()) {
        auto fieldIt = value.find(field);
        if (fieldIt != value.end() && fieldIt->is_array() && fieldIt->get<std::vector<uint64_t>>() == expected)
            return true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (jsonContainsArrayField(it.value(), field, expected))
                return true;
        }
    } else if (value.is_array()) {
        for (const auto& child : value) {
            if (jsonContainsArrayField(child, field, expected))
                return true;
        }
    }
    return false;
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


TEST(CustomLayerApi, RaggedMultipleApplicationsSaveLoadPreservesIndependentPartitions) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Ragged CustomLayer persistence test requires a GPU";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t capacity = 8;
    const string networkName = "custom_layer_ragged_multiple_applications_round_trip";
    const filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::RaggedTensor first = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("first")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({4})
                                      .maxTotalValues(capacity)
                                      .batchSize(batchSize)
                                      .build();
        Api::RaggedTensor second = Api::RaggedNetworkInput::Builder()
                                       .network(network)
                                       .name("second")
                                       .valuesDataType(DataType::FP32)
                                       .offsetsDataType(DataType::UINT32)
                                       .trailingDimensions({4})
                                       .maxTotalValues(capacity)
                                       .batchSize(batchSize)
                                       .build();
        ASSERT_NE(first.getOffsets(), second.getOffsets());

        Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
        Impl::ExpressionDefinition definition =
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"y", x.gelu()}}));
        Api::CustomLayer custom = Api::CustomLayer::Builder()
                                      .network(network)
                                      .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
                                      .inputNames({"x"})
                                      .outputNames({"y"})
                                      .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", first}})
                                      .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", second}})
                                      .build();
        ASSERT_TRUE(custom.getUseRagged());
        ASSERT_EQ(custom.getRaggedInputInterface(0).at("x").getOffsets(), first.getOffsets());
        ASSERT_EQ(custom.getRaggedInputInterface(1).at("x").getOffsets(), second.getOffsets());
        ASSERT_EQ(custom.getRaggedOutput("y", 0).getOffsets(), first.getOffsets());
        ASSERT_EQ(custom.getRaggedOutput("y", 1).getOffsets(), second.getOffsets());

        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("first_out")
            .inputTensor(custom.getRaggedOutput("y", 0))
            .build();
        (void)Api::RaggedNetworkOutput::Builder()
            .network(network)
            .name("second_out")
            .inputTensor(custom.getRaggedOutput("y", 1))
            .build();

        const nlohmann::json beforeSaveJson = custom.architectureJson();
        EXPECT_EQ(beforeSaveJson.at("layer_type").get<string>(), "custom_layer");
        EXPECT_TRUE(beforeSaveJson.at("use_ragged").get<bool>());
        ASSERT_EQ(beforeSaveJson.at("ragged_input_interfaces").size(), 2u);
        ASSERT_EQ(beforeSaveJson.at("ragged_output_interfaces").size(), 2u);
        network.save(archiveDir.string(), true);

        Api::Network loaded(networkName);
        loaded.load(archiveDir.string());
        std::shared_ptr<Api::CustomLayer> loadedCustom = findOnlyLayerOfType<Api::CustomLayer>(loaded);
        ASSERT_NE(loadedCustom, nullptr);
        ASSERT_TRUE(loadedCustom->getUseRagged());
        ASSERT_EQ(loadedCustom->getRaggedInputInterface(0).at("x").getOffsets(),
                  loadedCustom->getRaggedOutput("y", 0).getOffsets());
        ASSERT_EQ(loadedCustom->getRaggedInputInterface(1).at("x").getOffsets(),
                  loadedCustom->getRaggedOutput("y", 1).getOffsets());
        EXPECT_NE(loadedCustom->getRaggedInputInterface(0).at("x").getOffsets(),
                  loadedCustom->getRaggedInputInterface(1).at("x").getOffsets());
        const nlohmann::json loadedJson = loadedCustom->architectureJson();
        EXPECT_TRUE(loadedJson.at("use_ragged").get<bool>());
        ASSERT_EQ(loadedJson.at("ragged_input_interfaces").size(), 2u);
        ASSERT_EQ(loadedJson.at("ragged_output_interfaces").size(), 2u);

        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed;
        ASSERT_NO_THROW(placed = loaded.place(batchSize, initDoneEvents, true));
        synchronizeEvents(initDoneEvents);
        ASSERT_NE(placed, nullptr);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }
    filesystem::remove_all(archiveDir);
}

TEST(CustomLayerApi, RaggedTrainablePlacedSaveLoadRestoresParameterState) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Ragged CustomLayer state persistence test requires a GPU";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t capacity = 8;
    constexpr uint64_t width = 4;
    const string networkName = "custom_layer_ragged_parameter_state_round_trip";
    const filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);
    const vector<float> savedScale = {2.0f, 3.0f, 4.0f, 5.0f};

    try {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT32)
                                      .trailingDimensions({width})
                                      .maxTotalValues(capacity)
                                      .batchSize(batchSize)
                                      .build();

        Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
        Impl::Expression scale = Impl::Expression::input("scale", DataType::FP32, DataType::FP32);
        Impl::ExpressionDefinition definition =
            Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"y", x * scale}}));
        auto initializer = Api::UniformRandom::Builder().minValue(1.0f).maxValue(1.0f).build();
        auto scaleParameter = std::make_shared<Api::ParameterSpecification>(
            Api::ParameterSpecification::Builder()
                .name("scale")
                .shape({width})
                .dtype(DataType::FP32)
                .initializer(initializer)
                .trainable(true)
                .build());

        Api::CustomLayer custom = Api::CustomLayer::Builder()
                                      .network(network)
                                      .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
                                      .inputNames({"x"})
                                      .outputNames({"y"})
                                      .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", input}})
                                      .parameter(scaleParameter)
                                      .build();
        (void)Api::RaggedNetworkOutput::Builder().network(network).name("result").inputTensor(custom.getRaggedOutput("y")).build();

        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed = network.place(batchSize, initDoneEvents, true);
        synchronizeEvents(initDoneEvents);
        ASSERT_NE(placed, nullptr);
        auto physical = std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(
            placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(custom.getId()));
        ASSERT_NE(physical, nullptr);
        ASSERT_NE(physical->getParameter("scale"), nullptr);
        ASSERT_TRUE(physical->getParameter("scale")->getStorage().has_value());
        Stream stream = physical->getStreams()[0];
        writeDeviceTensor(physical->getParameter("scale")->getStorage().value(), savedScale, stream);
        expectAllClose(readDeviceTensor(physical->getParameter("scale")->getStorage().value(), stream), savedScale);

        placed->save(archiveDir.string(), true, false);

        Api::Network loaded(networkName);
        loaded.load(archiveDir.string());
        std::shared_ptr<Api::CustomLayer> loadedCustom = findOnlyLayerOfType<Api::CustomLayer>(loaded);
        ASSERT_NE(loadedCustom, nullptr);
        ASSERT_TRUE(loadedCustom->getUseRagged());
        vector<Event> loadedInitEvents;
        shared_ptr<Api::PlacedNetwork> loadedPlaced = loaded.place(batchSize, loadedInitEvents, true);
        synchronizeEvents(loadedInitEvents);
        ASSERT_NE(loadedPlaced, nullptr);
        auto loadedPhysical = std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(
            loadedPlaced->getStampedNetwork(0).getPhysicalLayerFromApiLayer(loadedCustom->getId()));
        ASSERT_NE(loadedPhysical, nullptr);
        ASSERT_NE(loadedPhysical->getParameter("scale"), nullptr);
        ASSERT_TRUE(loadedPhysical->getParameter("scale")->getStorage().has_value());
        Stream loadedStream = loadedPhysical->getStreams()[0];
        expectAllClose(readDeviceTensor(loadedPhysical->getParameter("scale")->getStorage().value(), loadedStream), savedScale);
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }
    filesystem::remove_all(archiveDir);
}

}  // namespace

TEST(CustomLayerApi, RuntimeOnlyExpressionIsRejectedDuringConstruction) {
    Api::Network network("custom_layer_runtime_only_rejected");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3}).dataType(DataType::FP32).build();

    try {
        (void)Api::CustomLayer::Builder()
            .network(network)
            .expression(makeRuntimeOnlyAffineExpression())
            .inputNames({"x"})
            .outputNames({"y"})
            .inputInterface({{"x", input.getFeatureOutput().value()}})
            .build();
        FAIL() << "Expected non-serializable CustomLayer construction to fail.";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("CustomLayer construction rejected"), std::string::npos);
        EXPECT_NE(message.find("did not provide an ExpressionDefinition"), std::string::npos);
    }
    EXPECT_EQ(network.getNumLayers(), 1u);
}

TEST(CustomLayerApi, BatchDependentNestedCudaExpressionIsNotGeneralizedDuringSerializationAnalysis) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "CustomLayer conditional CUDA serialization-analysis test requires a GPU";

    Api::Network network("custom_layer_batch_dependent_nested_cuda_rejected");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({3}).dataType(DataType::FP32).build();

    try {
        (void)Api::CustomLayer::Builder()
            .network(network)
            .expression(makeBatchDependentConditionalCudaExpression())
            .inputNames({"x"})
            .outputNames({"y"})
            .inputInterface({{"x", input.getFeatureOutput().value()}})
            .build();
        FAIL() << "Expected batch-dependent nested CUDA expression to require an explicit symbolic definition.";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("batch-dependent CUDA-kernel expressions require an explicitly supplied symbolic ExpressionDefinition"),
                  std::string::npos);
    }
    EXPECT_EQ(network.getNumLayers(), 1u);
}

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
                                           .requiresFullBatch()
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
        EXPECT_TRUE(customLayer.requiresFullBatch());
        EXPECT_TRUE(beforeSaveJson.at("requires_full_batch").get<bool>());

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
        EXPECT_TRUE(loadedCustomLayer->requiresFullBatch());
        EXPECT_TRUE(loadedJson.at("requires_full_batch").get<bool>());

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


TEST(CustomLayerApi, LogicalCompositionInfersShapesAndDTypesInsideCustomLayer) {
    constexpr uint32_t numFeatures = 3;
    Api::Network network("custom_layer_logical_composition_shape_dtype");
    Api::NetworkInput labels =
        Api::NetworkInput::Builder().network(network).name("labels_input").dimensions({numFeatures}).dataType(DataType::FP32).build();
    Api::NetworkInput initial = Api::NetworkInput::Builder()
                                    .network(network)
                                    .name("initial_input")
                                    .dimensions({numFeatures})
                                    .dataType(DataType::FP32)
                                    .build();

    const vector<string> outputNames = {
        "not_positive",
        "positive_and_initial",
        "positive_or_initial",
        "positive_and_not_initial",
        "not_positive_fp32",
        "positive_and_initial_fp32",
        "positive_or_initial_fp32",
        "positive_and_not_initial_fp32",
        "zero_mask",
        "typical_positive_mask",
        "spike_mask",
    };

    Api::CustomLayer custom = Api::CustomLayer::Builder()
                                  .network(network)
                                  .expression(makeSerializableLogicalMaskExpression())
                                  .inputNames({"labels", "initial_daily_units"})
                                  .outputNames(outputNames)
                                  .inputInterface({{"labels", labels.getFeatureOutput().value()},
                                                   {"initial_daily_units", initial.getFeatureOutput().value()}})
                                  .build();

    const vector<string> booleanOutputNames = {
        "not_positive", "positive_and_initial", "positive_or_initial", "positive_and_not_initial"};
    for (const string& name : booleanOutputNames) {
        EXPECT_EQ(custom.getOutput(name).getDimensions(), (vector<uint64_t>{numFeatures})) << name;
        EXPECT_EQ(custom.getOutput(name).getDataType(), DataType::BOOLEAN) << name;
    }
    const vector<string> fp32OutputNames = {"not_positive_fp32",
                                             "positive_and_initial_fp32",
                                             "positive_or_initial_fp32",
                                             "positive_and_not_initial_fp32",
                                             "zero_mask",
                                             "typical_positive_mask",
                                             "spike_mask"};
    for (const string& name : fp32OutputNames) {
        EXPECT_EQ(custom.getOutput(name).getDimensions(), (vector<uint64_t>{numFeatures})) << name;
        EXPECT_EQ(custom.getOutput(name).getDataType(), DataType::FP32) << name;
    }

    const nlohmann::json json = custom.architectureJson();
    ASSERT_FALSE(json.at("expression").is_null());
    EXPECT_NE(json.at("expression").dump().find("logical_not"), string::npos);
    EXPECT_NE(json.at("expression").dump().find("logical_and"), string::npos);
    EXPECT_NE(json.at("expression").dump().find("logical_or"), string::npos);
}

TEST(CustomLayerApi, LogicalCompositionSaveLoadExecutesExpectedMasks) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Logical CustomLayer execution test requires a GPU";

    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numFeatures = 3;
    const vector<float> labelsValues = {0.0f, 1.0f, 5.0f, -1.0f, 3.0f, 2.0f};
    const vector<float> initialValues = {1.0f, 1.0f, 2.0f, 1.0f, 10.0f, 0.0f};

    const vector<pair<string, vector<float>>> expectedOutputs = {
        {"not_positive_fp32", {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}},
        {"positive_and_initial_fp32", {0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f}},
        {"positive_or_initial_fp32", {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}},
        {"positive_and_not_initial_fp32", {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}},
        {"zero_mask", {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
        {"typical_positive_mask", {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}},
        {"spike_mask", {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}},
    };

    const string networkName = "custom_layer_logical_composition_save_load";
    const filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput labels =
            Api::NetworkInput::Builder().network(network).name("labels_input").dimensions({numFeatures}).dataType(DataType::FP32).build();
        Api::NetworkInput initial = Api::NetworkInput::Builder()
                                        .network(network)
                                        .name("initial_input")
                                        .dimensions({numFeatures})
                                        .dataType(DataType::FP32)
                                        .build();

        const vector<string> outputNames = {
            "not_positive",
            "positive_and_initial",
            "positive_or_initial",
            "positive_and_not_initial",
            "not_positive_fp32",
            "positive_and_initial_fp32",
            "positive_or_initial_fp32",
            "positive_and_not_initial_fp32",
            "zero_mask",
            "typical_positive_mask",
            "spike_mask",
        };
        Api::CustomLayer custom = Api::CustomLayer::Builder()
                                      .network(network)
                                      .expression(makeSerializableLogicalMaskExpression())
                                      .inputNames({"labels", "initial_daily_units"})
                                      .outputNames(outputNames)
                                      .inputInterface({{"labels", labels.getFeatureOutput().value()},
                                                       {"initial_daily_units", initial.getFeatureOutput().value()}})
                                      .build();

        for (const string& name : outputNames) {
            Api::Tensor customOutput = custom.getOutput(name);
            (void)Api::NetworkOutput::Builder()
                .network(network)
                .name(name)
                .inputTensor(customOutput)
                .dataType(customOutput.getDataType())
                .build();
        }

        network.save(archiveDir.string(), true);

        Api::Network loaded(networkName);
        loaded.load(archiveDir.string());

        shared_ptr<Api::NetworkInput> loadedLabels;
        shared_ptr<Api::NetworkInput> loadedInitial;
        shared_ptr<Api::CustomLayer> loadedCustom;
        unordered_map<string, shared_ptr<Api::NetworkOutput>> loadedOutputs;
        for (uint32_t i = 0; i < loaded.getNumLayers(); ++i) {
            shared_ptr<Api::Layer> layer = loaded.getLayer(i);
            if (auto input = dynamic_pointer_cast<Api::NetworkInput>(layer)) {
                if (input->getName() == "labels_input")
                    loadedLabels = input;
                else if (input->getName() == "initial_input")
                    loadedInitial = input;
            } else if (auto customLayer = dynamic_pointer_cast<Api::CustomLayer>(layer)) {
                loadedCustom = customLayer;
            } else if (auto output = dynamic_pointer_cast<Api::NetworkOutput>(layer)) {
                loadedOutputs.emplace(output->getName(), output);
            }
        }

        ASSERT_NE(loadedLabels, nullptr);
        ASSERT_NE(loadedInitial, nullptr);
        ASSERT_NE(loadedCustom, nullptr);
        for (const auto& [name, expected] : expectedOutputs) {
            (void)expected;
            ASSERT_TRUE(loadedOutputs.contains(name)) << name;
        }

        const nlohmann::json loadedJson = loadedCustom->architectureJson();
        ASSERT_FALSE(loadedJson.at("expression").is_null());
        EXPECT_NE(loadedJson.at("expression").dump().find("logical_not"), string::npos);
        EXPECT_NE(loadedJson.at("expression").dump().find("logical_and"), string::npos);
        EXPECT_NE(loadedJson.at("expression").dump().find("logical_or"), string::npos);

        vector<Event> initDoneEvents;
        shared_ptr<Api::PlacedNetwork> placed = loaded.place(batchSize, initDoneEvents, true);
        synchronizeEvents(initDoneEvents);
        ASSERT_NE(placed, nullptr);
        auto& stamped = placed->getStampedNetwork(0);

        auto physicalLabels = dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(loadedLabels->getId()));
        auto physicalInitial = dynamic_pointer_cast<Impl::NetworkInput>(stamped.getPhysicalLayerFromApiLayer(loadedInitial->getId()));
        ASSERT_NE(physicalLabels, nullptr);
        ASSERT_NE(physicalInitial, nullptr);

        unordered_map<string, shared_ptr<Impl::NetworkOutput>> physicalOutputs;
        for (const auto& [name, apiOutput] : loadedOutputs) {
            auto output = dynamic_pointer_cast<Impl::NetworkOutput>(stamped.getPhysicalLayerFromApiLayer(apiOutput->getId()));
            ASSERT_NE(output, nullptr) << name;
            physicalOutputs.emplace(name, output);
        }

        Impl::Tensor labelsHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, numFeatures}));
        Impl::Tensor initialHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, numFeatures}));
        writeCpuTensor(labelsHost, labelsValues);
        writeCpuTensor(initialHost, initialValues);

        physicalLabels->forward(labelsHost, false, batchSize);
        physicalInitial->forward(initialHost, false, batchSize);

        for (const auto& [name, expected] : expectedOutputs) {
            auto& output = physicalOutputs.at(name);
            output->getOutputReadyEvent().synchronize();
            expectAllClose(readCpuTensor(output->getFeatureOutput().value()), expected);
        }
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }
    filesystem::remove_all(archiveDir);
}

TEST(CustomLayerApi, ConditionalCudaExpressionSaveLoadRoundTripRunsBothBranches) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "CustomLayer conditional CUDA round-trip test requires a GPU";

    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numFeatures = 3;
    const DataType dataType = DataType::FP32;
    const std::string networkName = "custom_layer_conditional_cuda_round_trip";
    std::filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);
    std::filesystem::path keyPath = archiveDir.parent_path() / (archiveDir.filename().string() + "_cuda_keys.json");
    std::filesystem::remove(keyPath);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input =
            Api::NetworkInput::Builder().network(network).name("input").dimensions({numFeatures}).dataType(dataType).build();
        Api::CustomLayer customLayer = Api::CustomLayer::Builder()
                                           .network(network)
                                           .expression(makeSerializableConditionalCudaExpression())
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
        ASSERT_TRUE(beforeSaveJson.at("expression").contains("conditional"));
        ASSERT_TRUE(beforeSaveJson.at("expression").at("conditional").at("then_branch").contains("cuda_kernels"));
        EXPECT_TRUE(network.hasCudaKernelExpressions());

        network.captureCudaKernelSaveKeysToFile(keyPath.string());
        network.save(archiveDir.string(), true);

        std::ifstream keyFile(keyPath);
        ASSERT_TRUE(keyFile.good());
        nlohmann::json capturedKeys;
        keyFile >> capturedKeys;
        ASSERT_EQ(capturedKeys.at("status").get<std::string>(), "complete");
        ASSERT_EQ(capturedKeys.at("keys").size(), 1u);
        const std::string signingPublicKey = capturedKeys.at("keys").at(0).at("signing_public_key").get<std::string>();
        const std::string sourceDecryptionKey = capturedKeys.at("keys").at(0).at("source_decryption_key").get<std::string>();
        ASSERT_FALSE(signingPublicKey.empty());
        ASSERT_FALSE(sourceDecryptionKey.empty());

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string(), true, signingPublicKey, sourceDecryptionKey);

        std::shared_ptr<Api::NetworkInput> loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        std::shared_ptr<Api::CustomLayer> loadedCustomLayer = findOnlyLayerOfType<Api::CustomLayer>(loadedNetwork);
        std::shared_ptr<Api::NetworkOutput> loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedCustomLayer, nullptr);
        ASSERT_NE(loadedOutput, nullptr);
        EXPECT_TRUE(loadedNetwork.hasCudaKernelExpressions());

        const nlohmann::json loadedJson = loadedCustomLayer->architectureJson();
        ASSERT_TRUE(loadedJson.at("expression").contains("conditional"));
        ASSERT_TRUE(loadedJson.at("expression").at("conditional").at("then_branch").contains("cuda_kernels"));

        PlacedCustomLayerFixture fixture =
            placeSingleCustomLayerNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedCustomLayer, batchSize, true);

        Impl::Tensor positiveInput(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numFeatures}));
        writeCpuTensor(positiveInput, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        expectAllClose(runForward(*fixture.physicalInput, *fixture.physicalOutput, positiveInput, batchSize),
                       {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f});

        Impl::Tensor negativeInput(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numFeatures}));
        writeCpuTensor(negativeInput, {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f});
        expectAllClose(runForward(*fixture.physicalInput, *fixture.physicalOutput, negativeInput, batchSize),
                       {-6.0f, -7.0f, -8.0f, -9.0f, -10.0f, -11.0f});
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        std::filesystem::remove(keyPath);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
    std::filesystem::remove(keyPath);
}

TEST(CustomLayerApi, BatchProductReshapeIsGeneralizedWithInferDimension) {
    constexpr uint32_t batchSize = 4;
    constexpr uint32_t sequence = 5;
    constexpr uint32_t channels = 2;
    auto phaseNetwork = std::make_shared<Api::Network>("custom_layer_batch_product_reshape_phase");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(*phaseNetwork)
                                  .name("x")
                                  .dimensions({sequence, channels})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::CustomLayer reshape = Api::CustomLayer::Builder()
                                   .network(*phaseNetwork)
                                   .expression(makeBatchProductReshapeExpression())
                                   .inputNames({"x"})
                                   .outputNames({"y"})
                                   .inputInterface({{"x", input.getFeatureOutput().value()}})
                                   .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(*phaseNetwork)
                                    .name("y")
                                    .inputTensor(reshape.getOutput("y"))
                                    .dataType(DataType::FP32)
                                    .build();

    const nlohmann::json customArchitecture = reshape.architectureJson();
    ASSERT_TRUE(jsonContainsArrayField(
        customArchitecture.at("expression"),
        "reshape_dims",
        {std::numeric_limits<uint64_t>::max(), channels}));
    ASSERT_TRUE(jsonContainsArrayField(
        customArchitecture.at("expression"),
        "reshape_dims",
        {std::numeric_limits<uint64_t>::max(), sequence, channels}));

    Api::ComposedPhaseGraph composed = Api::buildComposedPhaseGraphByName(
        {{"phase", phaseNetwork, true}},
        Api::PhaseGraphComposeOptions{"custom_layer_batch_product_reshape_composed"});
    ASSERT_NE(composed.network, nullptr);

    std::shared_ptr<Api::NetworkInput> composedInput = findOnlyLayerOfType<Api::NetworkInput>(*composed.network);
    std::shared_ptr<Api::CustomLayer> composedReshape = findOnlyLayerOfType<Api::CustomLayer>(*composed.network);
    std::shared_ptr<Api::NetworkOutput> composedOutput = findOnlyLayerOfType<Api::NetworkOutput>(*composed.network);
    ASSERT_NE(composedInput, nullptr);
    ASSERT_NE(composedReshape, nullptr);
    ASSERT_NE(composedOutput, nullptr);

    PlacedCustomLayerFixture fixture = placeSingleCustomLayerNetwork(
        *composed.network, *composedInput, *composedOutput, *composedReshape, batchSize, true);
    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, sequence, channels}));
    vector<float> values(batchSize * sequence * channels);
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(i) - 7.0f;
    writeCpuTensor(featureInHost, values);

    expectAllClose(runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize), values);
}

TEST(CustomLayerApi, BatchLiteralShapeIsGeneralizedBeforePhaseCompositionAtRuntimeBatchFour) {
    constexpr uint32_t batchSize = 4;
    auto phaseNetwork = std::make_shared<Api::Network>("custom_layer_batch_literal_phase");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(*phaseNetwork)
                                  .name("sequence")
                                  .dimensions({6, 2})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::CustomLayer tail = Api::CustomLayer::Builder()
                                .network(*phaseNetwork)
                                .expression(makeBatchLiteralTailExpression())
                                .inputNames({"sequence"})
                                .outputNames({"tail"})
                                .inputInterface({{"sequence", input.getFeatureOutput().value()}})
                                .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(*phaseNetwork)
                                    .name("tail")
                                    .inputTensor(tail.getOutput("tail"))
                                    .dataType(DataType::FP32)
                                    .build();

    // API output inference probes physical batches 1 and 2, then phase composition
    // serializes a generalized definition rather than retaining either concrete probe.
    const nlohmann::json customArchitecture = tail.architectureJson();
    ASSERT_TRUE(jsonContainsArrayField(customArchitecture.at("expression"), "view_dims", {0, 3, 2}));

    Api::ComposedPhaseGraph composed = Api::buildComposedPhaseGraphByName(
        {{"phase", phaseNetwork, true}},
        Api::PhaseGraphComposeOptions{"custom_layer_batch_literal_composed"});
    ASSERT_NE(composed.network, nullptr);

    std::shared_ptr<Api::NetworkInput> composedInput = findOnlyLayerOfType<Api::NetworkInput>(*composed.network);
    std::shared_ptr<Api::CustomLayer> composedTail = findOnlyLayerOfType<Api::CustomLayer>(*composed.network);
    std::shared_ptr<Api::NetworkOutput> composedOutput = findOnlyLayerOfType<Api::NetworkOutput>(*composed.network);
    ASSERT_NE(composedInput, nullptr);
    ASSERT_NE(composedTail, nullptr);
    ASSERT_NE(composedOutput, nullptr);

    PlacedCustomLayerFixture fixture = placeSingleCustomLayerNetwork(
        *composed.network, *composedInput, *composedOutput, *composedTail, batchSize, true);
    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchSize, 6, 2}));
    vector<float> values(batchSize * 6 * 2);
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(i);
    writeCpuTensor(featureInHost, values);

    vector<float> expected;
    expected.reserve(batchSize * 3 * 2);
    for (uint64_t batch = 0; batch < batchSize; ++batch) {
        const uint64_t base = batch * 6 * 2;
        for (uint64_t i = 3 * 2; i < 6 * 2; ++i)
            expected.push_back(values[base + i]);
    }
    expectAllClose(runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize), expected);
}


TEST(CustomLayerApi, OptionalValidityMaskIsAvailableInsideExpressionAndNotExposedAsAFeaturePort) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "CustomLayer validity-mask execution test requires a GPU";

    constexpr uint32_t batchCapacity = 4;
    constexpr uint32_t validExampleCount = 2;
    Api::Network network("custom_layer_validity_mask");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("x")
                                  .dimensions({2})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::CustomLayer customLayer = Api::CustomLayer::Builder()
                                       .network(network)
                                       .expression(makeSerializableValidityMaskedExpression())
                                       .usesBatchValidity()
                                       .inputInterface({{"x", input.getFeatureOutput().value()}})
                                       .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("y")
                                    .inputTensor(customLayer.getOutput("y"))
                                    .dataType(DataType::FP32)
                                    .build();

    EXPECT_TRUE(customLayer.usesBatchValidity());
    EXPECT_EQ(customLayer.getInputNames(), (vector<string>{"x"}));
    EXPECT_TRUE(customLayer.architectureJson().at("uses_batch_validity").get<bool>());
    EXPECT_FALSE(customLayer.architectureJson().contains("uses_batch_validity_mask"));

    PlacedCustomLayerFixture fixture =
        placeSingleCustomLayerNetwork(network, input, output, customLayer, batchCapacity, /*inferenceOnly=*/true);
    Impl::Tensor inputCpu(cpuPlacement, Impl::TensorDescriptor(DataType::FP32, {batchCapacity, 2}));
    writeCpuTensor(inputCpu, {1.0f, 2.0f, 3.0f, 4.0f, 50.0f, 60.0f, 70.0f, 80.0f});

    const vector<float> actual =
        runForward(*fixture.physicalInput, *fixture.physicalOutput, inputCpu, validExampleCount);
    expectAllClose(actual, {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(CustomLayerApi, BatchValidityDeclarationMustMatchExpressionContract) {
    Api::Network network("custom_layer_validity_mask_contract");
    Api::Tensor input(DataType::FP32, {2});

    EXPECT_THROW((void)Api::CustomLayer::Builder()
                     .network(network)
                     .expression(makeSerializableValidityMaskedExpression())
                     .inputInterface({{"x", input}})
                     .build(),
                 std::runtime_error);

    EXPECT_THROW((void)Api::CustomLayer::Builder()
                     .network(network)
                     .expression(makeSerializableAffineExpression())
                     .usesBatchValidity()
                     .inputInterface({{"x", input}})
                     .build(),
                 std::runtime_error);
}
TEST(CustomLayerApi, FullBatchRequirementIsSerializedAndPropagatedToPhysicalLayer) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "CustomLayer placement test requires a GPU";

    Api::Network network("custom_layer_full_batch_requirement");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("x")
                                  .dimensions({2})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::CustomLayer customLayer = Api::CustomLayer::Builder()
                                       .network(network)
                                       .expression(makeSerializableAffineExpression())
                                       .requiresFullBatch()
                                       .inputInterface({{"x", input.getFeatureOutput().value()}})
                                       .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("y")
                                    .inputTensor(customLayer.getOutput("y"))
                                    .dataType(DataType::FP32)
                                    .build();

    EXPECT_TRUE(customLayer.requiresFullBatch());
    EXPECT_TRUE(customLayer.architectureJson().at("requires_full_batch").get<bool>());
    PlacedCustomLayerFixture fixture =
        placeSingleCustomLayerNetwork(network, input, output, customLayer, /*batchSize=*/4, /*inferenceOnly=*/true);
    ASSERT_NE(fixture.physicalCustomLayer, nullptr);
    EXPECT_FALSE(fixture.physicalCustomLayer->supportsPartialBatches());

    const std::vector<ThorImplementation::PartialBatchIncompatibility> incompatibilities =
        fixture.placedNetwork->getPartialBatchIncompatibilities();
    ASSERT_EQ(incompatibilities.size(), 1u);
    EXPECT_EQ(incompatibilities.front().layerId, customLayer.getId());
    EXPECT_EQ(incompatibilities.front().layerType, "CustomLayer");
}

TEST(CustomLayerApi, PartialBatchCompatibilityReportsEveryIncompatibleLayer) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "CustomLayer placement test requires a GPU";

    Api::Network network("custom_layer_all_partial_batch_incompatibilities");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("x")
                                  .dimensions({2})
                                  .dataType(DataType::FP32)
                                  .build();
    Api::CustomLayer first = Api::CustomLayer::Builder()
                                 .network(network)
                                 .expression(makeSerializableAffineExpression())
                                 .requiresFullBatch()
                                 .inputInterface({{"x", input.getFeatureOutput().value()}})
                                 .build();
    Api::CustomLayer second = Api::CustomLayer::Builder()
                                  .network(network)
                                  .expression(makeSerializableAffineExpression())
                                  .requiresFullBatch()
                                  .inputInterface({{"x", first.getOutput("y")}})
                                  .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("y")
                                    .inputTensor(second.getOutput("y"))
                                    .dataType(DataType::FP32)
                                    .build();

    PlacedCustomLayerFixture fixture =
        placeSingleCustomLayerNetwork(network, input, output, second, /*batchSize=*/4, /*inferenceOnly=*/true);
    const std::vector<ThorImplementation::PartialBatchIncompatibility> incompatibilities =
        fixture.placedNetwork->getPartialBatchIncompatibilities();
    ASSERT_EQ(incompatibilities.size(), 2u);
    std::set<uint64_t> incompatibleIds;
    for (const auto& incompatibility : incompatibilities) {
        EXPECT_EQ(incompatibility.layerType, "CustomLayer");
        incompatibleIds.insert(incompatibility.layerId);
    }
    EXPECT_EQ(incompatibleIds, (std::set<uint64_t>{first.getId(), second.getId()}));
}

TEST(CustomLayerApi, BatchValidityAndFullBatchRequirementAreMutuallyExclusive) {
    Api::Network network("custom_layer_conflicting_partial_batch_contract");
    Api::Tensor input(DataType::FP32, {2});

    EXPECT_ANY_THROW((void)Api::CustomLayer::Builder()
                         .network(network)
                         .expression(makeSerializableValidityMaskedExpression())
                         .usesBatchValidity()
                         .requiresFullBatch()
                         .inputInterface({{"x", input}})
                         .build());
    EXPECT_ANY_THROW((void)Api::CustomLayer::Builder()
                         .network(network)
                         .expression(makeSerializableAffineExpression())
                         .requiresFullBatch()
                         .usesBatchValidity()
                         .inputInterface({{"x", input}})
                         .build());
}


TEST(CustomLayerApi, RaggedSingleInputPreservesPartitionAndPlaces) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Ragged CustomLayer placement test requires a GPU";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t capacity = 8;
    Api::Network network("custom_layer_ragged_single_input");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({4})
                                  .maxTotalValues(capacity)
                                  .batchSize(batchSize)
                                  .build();

    Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"y", x.gelu()}}));
    Api::CustomLayer custom = Api::CustomLayer::Builder()
                                  .network(network)
                                  .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
                                  .inputNames({"x"})
                                  .outputNames({"y"})
                                  .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", input}})
                                  .build();

    ASSERT_TRUE(custom.getUseRagged());
    ASSERT_EQ(custom.getFeatureInputs().size(), 2u);
    EXPECT_EQ(custom.getFeatureInputs()[0], input.getValues());
    EXPECT_EQ(custom.getFeatureInputs()[1], input.getOffsets());
    Api::RaggedTensor output = custom.getRaggedOutput("y");
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    EXPECT_EQ(output.getValuesDimensions(), (vector<uint64_t>{capacity, 4}));

    (void)Api::RaggedNetworkOutput::Builder().network(network).name("result").inputTensor(output).build();
    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(batchSize, initDoneEvents, true));
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
}

TEST(CustomLayerApi, RaggedCompositePointwiseExpressionStampsAsSingleRaggedCustomLayer) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Ragged CustomLayer placement test requires a GPU";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t capacity = 8;
    Api::Network network("custom_layer_ragged_composite_pointwise");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({4})
                                  .maxTotalValues(capacity)
                                  .batchSize(batchSize)
                                  .build();

    Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
    Impl::Expression logged = x.log1p();
    Impl::Expression activity = -((-logged).expm1());
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"activity", activity}}));
    Api::CustomLayer custom = Api::CustomLayer::Builder()
                                  .network(network)
                                  .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
                                  .inputNames({"x"})
                                  .outputNames({"activity"})
                                  .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", input}})
                                  .build();

    Api::RaggedTensor output = custom.getRaggedOutput("activity");
    EXPECT_EQ(output.getOffsets(), input.getOffsets());
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("result").inputTensor(output).build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(batchSize, initDoneEvents, true));
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);

    auto physical = std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(custom.getId()));
    ASSERT_NE(physical, nullptr);
    EXPECT_EQ(physical->getLayerType(), "RaggedCustomLayer");
}

TEST(CustomLayerApi, RaggedMultiInputMultiOutputCanChangeTrailingShapeWithoutExposingOffsetsToExpression) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Ragged CustomLayer placement test requires a GPU";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t capacity = 8;
    Api::Network network("custom_layer_ragged_multi_io");
    Api::RaggedTensor lhs = Api::RaggedNetworkInput::Builder()
                                .network(network)
                                .name("lhs")
                                .valuesDataType(DataType::FP32)
                                .offsetsDataType(DataType::UINT32)
                                .trailingDimensions({4})
                                .maxTotalValues(capacity)
                                .batchSize(batchSize)
                                .build();
    Api::NetworkInput rhsValues = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("rhs_values")
                                      .dimensions({capacity, 4})
                                      .dataType(DataType::FP32)
                                      .dimensionsIncludeBatch(true)
                                      .build();
    Api::RaggedTensor rhs(rhsValues.getFeatureOutput().value(), lhs.getOffsets());

    Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
    Impl::Expression y = Impl::Expression::input("y", DataType::FP32, DataType::FP32);
    Impl::Expression wide = x + y;
    Impl::Expression narrow = wide.stridedView({capacity, 2}, {4, 1}, 1);
    Impl::ExpressionDefinition definition = Impl::ExpressionDefinition::fromOutputs(
        Impl::Expression::outputs({{"wide", wide}, {"narrow", narrow}}));

    Api::CustomLayer custom = Api::CustomLayer::Builder()
                                  .network(network)
                                  .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
                                  .inputNames({"x", "y"})
                                  .outputNames({"wide", "narrow"})
                                  .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", lhs}, {"y", rhs}})
                                  .build();

    ASSERT_TRUE(custom.getUseRagged());
    EXPECT_EQ(custom.getFeatureInputs().size(), 3u);
    Api::RaggedTensor wideOutput = custom.getRaggedOutput("wide");
    Api::RaggedTensor narrowOutput = custom.getRaggedOutput("narrow");
    EXPECT_EQ(wideOutput.getOffsets(), lhs.getOffsets());
    EXPECT_EQ(narrowOutput.getOffsets(), lhs.getOffsets());
    EXPECT_EQ(wideOutput.getValuesDimensions(), (vector<uint64_t>{capacity, 4}));
    EXPECT_EQ(narrowOutput.getValuesDimensions(), (vector<uint64_t>{capacity, 2}));

    (void)Api::RaggedNetworkOutput::Builder().network(network).name("wide_out").inputTensor(wideOutput).build();
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("narrow_out").inputTensor(narrowOutput).build();
    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(batchSize, initDoneEvents, true));
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
}

TEST(CustomLayerApi, RaggedInputsMustShareExactPartitionAndDenseRaggedBuildersCannotMix) {
    Api::Network network("custom_layer_ragged_partition_validation");
    Api::RaggedTensor lhs = Api::RaggedNetworkInput::Builder()
                                .network(network)
                                .name("lhs")
                                .valuesDataType(DataType::FP32)
                                .trailingDimensions({4})
                                .maxTotalValues(8)
                                .batchSize(3)
                                .build();
    Api::RaggedTensor rhs = Api::RaggedNetworkInput::Builder()
                                .network(network)
                                .name("rhs")
                                .valuesDataType(DataType::FP32)
                                .trailingDimensions({4})
                                .maxTotalValues(8)
                                .batchSize(3)
                                .build();
    Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
    Impl::Expression y = Impl::Expression::input("y", DataType::FP32, DataType::FP32);
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"z", x + y}}));

    EXPECT_THROW((Api::CustomLayer::Builder()
                      .network(network)
                      .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
                      .inputNames({"x", "y"})
                      .outputNames({"z"})
                      .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", lhs}, {"y", rhs}})
                      .build()),
                 std::invalid_argument);

    Api::CustomLayer::Builder mixedBuilder;
    mixedBuilder.network(network)
        .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
        .inputNames({"x", "y"})
        .outputNames({"z"})
        .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", lhs}, {"y", lhs}});
    EXPECT_ANY_THROW(mixedBuilder.inputInterface(Api::CustomLayer::TensorMap{{"x", lhs.getValues()}, {"y", lhs.getValues()}}));
}

TEST(CustomLayerApi, RaggedTrainableParameterStampsIntoRaggedCustomLayer) {
    if (MachineEvaluator::instance().getNumGpus() == 0)
        GTEST_SKIP() << "Ragged trainable CustomLayer placement test requires a GPU";

    constexpr uint32_t batchSize = 3;
    constexpr uint64_t capacity = 8;
    constexpr uint64_t width = 4;
    Api::Network network("custom_layer_ragged_trainable_parameter");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({width})
                                  .maxTotalValues(capacity)
                                  .batchSize(batchSize)
                                  .build();

    Impl::Expression x = Impl::Expression::input("x", DataType::FP32, DataType::FP32);
    Impl::Expression scale = Impl::Expression::input("scale", DataType::FP32, DataType::FP32);
    Impl::ExpressionDefinition definition =
        Impl::ExpressionDefinition::fromOutputs(Impl::Expression::outputs({{"y", x * scale}}));

    auto initializer = Api::UniformRandom::Builder().minValue(1.0f).maxValue(1.0f).build();
    std::shared_ptr<Api::Optimizer> optimizer = Api::Sgd::Builder().initialLearningRate(0.01f).build();
    auto scaleParameter = std::make_shared<Api::ParameterSpecification>(
        Api::ParameterSpecification::Builder()
            .name("scale")
            .shape({width})
            .dtype(DataType::FP32)
            .initializer(initializer)
            .trainable(true)
            .optimizer(optimizer)
            .build());

    Api::CustomLayer custom = Api::CustomLayer::Builder()
                                  .network(network)
                                  .expression(Impl::DynamicExpression::fromExpressionDefinition(definition))
                                  .inputNames({"x"})
                                  .outputNames({"y"})
                                  .inputInterface(Api::CustomLayer::RaggedTensorMap{{"x", input}})
                                  .parameter(scaleParameter)
                                  .build();
    Api::RaggedTensor output = custom.getRaggedOutput("y");
    (void)Api::RaggedNetworkOutput::Builder().network(network).name("result").inputTensor(output).build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placed;
    ASSERT_NO_THROW(placed = network.place(batchSize, initDoneEvents, true));
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);

    auto physical = std::dynamic_pointer_cast<Impl::RaggedCustomLayer>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(custom.getId()));
    ASSERT_NE(physical, nullptr);
    EXPECT_EQ(physical->listParameters(), (vector<string>{"scale"}));
    ASSERT_NE(physical->getParameter("scale"), nullptr);
    ASSERT_TRUE(physical->getParameter("scale")->getStorage().has_value());
    EXPECT_EQ(physical->getParameter("scale")->getStorage()->getDimensions(), (vector<uint64_t>{width}));
}
