#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::TensorDescriptor::DataType;

namespace {

Impl::TensorPlacement cpuPlacement(Impl::TensorPlacement::MemDevices::CPU);
Impl::TensorPlacement gpuPlacement(Impl::TensorPlacement::MemDevices::GPU, 0);

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
    ASSERT_EQ(tensorNumel(tensor), values.size());

    switch (tensor.getDataType()) {
        case DataType::FP16: {
            auto* ptr = static_cast<half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2half(values[i]);
            break;
        }
        case DataType::FP32: {
            auto* ptr = static_cast<float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = values[i];
            break;
        }
        default:
            FAIL() << "Unsupported tensor dtype in test writeCpuTensor.";
    }
}

vector<float> readCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);

    vector<float> values(tensorNumel(tensor));
    switch (tensor.getDataType()) {
        case DataType::FP16: {
            const auto* ptr = static_cast<const half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = __half2float(ptr[i]);
            break;
        }
        case DataType::FP32: {
            const auto* ptr = static_cast<const float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = ptr[i];
            break;
        }
        default:
            ADD_FAILURE() << "Unsupported tensor dtype in test readCpuTensor.";
            break;
    }
    return values;
}

Impl::Tensor copyTensorToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpuTensor = tensor.clone(cpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return cpuTensor;
}

void expectAllClose(
    const vector<float>& actual, const vector<float>& expected, float atol = 2e-2f, float rtol = 2e-2f, const string& what = "") {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        EXPECT_LE(diff, tol) << what << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

vector<float> fullyConnectedReference(const vector<float>& input,
                                      const vector<float>& weights,
                                      const vector<float>& biases,
                                      uint64_t batchSize,
                                      uint64_t numInputFeatures,
                                      uint64_t numOutputFeatures,
                                      bool hasBias) {
    vector<float> output(batchSize * numOutputFeatures, 0.0f);
    for (uint64_t b = 0; b < batchSize; ++b) {
        for (uint64_t o = 0; o < numOutputFeatures; ++o) {
            float acc = hasBias ? biases[o] : 0.0f;
            for (uint64_t i = 0; i < numInputFeatures; ++i)
                acc += input[b * numInputFeatures + i] * weights[i * numOutputFeatures + o];
            output[b * numOutputFeatures + o] = acc;
        }
    }
    return output;
}

vector<float> fullyConnectedBackwardErrorReference(const vector<float>& errorInput,
                                                   const vector<float>& weights,
                                                   uint64_t batchSize,
                                                   uint64_t numInputFeatures,
                                                   uint64_t numOutputFeatures) {
    vector<float> errorOutput(batchSize * numInputFeatures, 0.0f);
    for (uint64_t b = 0; b < batchSize; ++b) {
        for (uint64_t i = 0; i < numInputFeatures; ++i) {
            float acc = 0.0f;
            for (uint64_t o = 0; o < numOutputFeatures; ++o)
                acc += errorInput[b * numOutputFeatures + o] * weights[i * numOutputFeatures + o];
            errorOutput[b * numInputFeatures + i] = acc;
        }
    }
    return errorOutput;
}

vector<float> fullyConnectedWeightGradReference(const vector<float>& input,
                                                const vector<float>& errorInput,
                                                uint64_t batchSize,
                                                uint64_t numInputFeatures,
                                                uint64_t numOutputFeatures) {
    vector<float> gradWeights(numInputFeatures * numOutputFeatures, 0.0f);
    for (uint64_t i = 0; i < numInputFeatures; ++i) {
        for (uint64_t o = 0; o < numOutputFeatures; ++o) {
            float acc = 0.0f;
            for (uint64_t b = 0; b < batchSize; ++b)
                acc += input[b * numInputFeatures + i] * errorInput[b * numOutputFeatures + o];
            gradWeights[i * numOutputFeatures + o] = acc;
        }
    }
    return gradWeights;
}

vector<float> fullyConnectedBiasGradReference(const vector<float>& errorInput, uint64_t batchSize, uint64_t numOutputFeatures) {
    vector<float> gradBiases(numOutputFeatures, 0.0f);
    for (uint64_t o = 0; o < numOutputFeatures; ++o) {
        float acc = 0.0f;
        for (uint64_t b = 0; b < batchSize; ++b)
            acc += errorInput[b * numOutputFeatures + o];
        gradBiases[o] = acc;
    }
    return gradBiases;
}

vector<float> sgdUpdatedReference(const vector<float>& initial, const vector<float>& rawGradient, uint64_t batchSize, float lr) {
    const float step = lr / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    vector<float> updated(initial.size());
    for (uint64_t i = 0; i < initial.size(); ++i)
        updated[i] = initial[i] - step * rawGradient[i];
    return updated;
}

void setParameterTensor(const shared_ptr<Impl::PhysicalParameter>& parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().isPresent());
    Impl::Tensor deviceTensor = parameter->getStorage();
    Impl::Tensor cpuTensor = deviceTensor.clone(cpuPlacement);
    writeCpuTensor(cpuTensor, values);
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

struct PlacedFullyConnectedFixture {
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    Impl::StampedNetwork* stampedNetwork = nullptr;
    shared_ptr<Impl::NetworkInput> physicalInput;
    shared_ptr<Impl::NetworkOutput> physicalOutput;
    shared_ptr<Impl::CustomLayer> physicalFc;
};

PlacedFullyConnectedFixture placeSingleFullyConnectedNetwork(Api::Network& network,
                                                             const Api::NetworkInput& apiInput,
                                                             const Api::NetworkOutput& apiOutput,
                                                             const Api::FullyConnected& apiFc,
                                                             uint32_t batchSize,
                                                             bool inferenceOnly) {
    vector<Event> initDoneEvents;
    PlacedFullyConnectedFixture fixture;
    fixture.placedNetwork = network.place(batchSize, initDoneEvents, inferenceOnly);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(fixture.placedNetwork, nullptr);
    fixture.stampedNetwork = &fixture.placedNetwork->getStampedNetwork(0);

    fixture.physicalInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiInput.getId()));
    fixture.physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiOutput.getId()));
    fixture.physicalFc = dynamic_pointer_cast<Impl::CustomLayer>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiFc.getId()));

    EXPECT_NE(fixture.physicalInput, nullptr);
    EXPECT_NE(fixture.physicalOutput, nullptr);
    EXPECT_NE(fixture.physicalFc, nullptr);
    return fixture;
}

vector<float> runForward(Impl::NetworkInput& physicalInput,
                         Impl::NetworkOutput& physicalOutput,
                         Impl::Tensor& featureInHost,
                         uint32_t batchSize) {
    physicalInput.forward(featureInHost, false, batchSize);
    Event featureOutReadyEvent = physicalOutput.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    return readCpuTensor(physicalOutput.getFeatureOutput());
}

}  // namespace

TEST(FullyConnectedApi, BuilderCreatesParameterSpecsOutputsAndConnectionTypes) {
    Api::Network network("testNetwork");
    Api::Tensor featureInput0(DataType::FP16, {4});
    Api::Tensor featureInput1(DataType::FP16, {4});

    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(featureInput0)
                                 .featureInput(featureInput1)
                                 .numOutputFeatures(3)
                                 .hasBias(true)
                                 .weightsDataType(DataType::FP32)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();

    ASSERT_TRUE(fc.isInitialized());
    ASSERT_EQ(fc.getFeatureInputs().size(), 2u);
    ASSERT_EQ(fc.getFeatureOutputs().size(), 2u);
    EXPECT_EQ(fc.getFeatureOutput(featureInput0), fc.getFeatureOutputs()[0]);
    EXPECT_EQ(fc.getFeatureOutput(featureInput1), fc.getFeatureOutputs()[1]);
    EXPECT_EQ(fc.getFeatureInput(fc.getFeatureOutputs()[0]), featureInput0);
    EXPECT_EQ(fc.getFeatureInput(fc.getFeatureOutputs()[1]), featureInput1);
    EXPECT_EQ(fc.getConnectionType(featureInput0), 0);
    EXPECT_EQ(fc.getConnectionType(featureInput1), 1);
    EXPECT_EQ(fc.getConnectionType(fc.getFeatureOutputs()[0]), 0);
    EXPECT_EQ(fc.getConnectionType(fc.getFeatureOutputs()[1]), 1);

    EXPECT_EQ(fc.listParameters(), (vector<string>{"weights", "biases"}));
    EXPECT_EQ(fc.getParameterBytes(), static_cast<uint64_t>((4u * 3u + 3u) * Api::Tensor::getBytesPerElement(DataType::FP32)));

    const nlohmann::json j = fc.architectureJson();
    EXPECT_EQ(j.at("layer_type").get<string>(), "fully_connected");
    EXPECT_EQ(j.at("weights_data_type").get<DataType>(), DataType::FP32);
    EXPECT_EQ(j.at("compute_data_type").get<DataType>(), DataType::FP32);
    EXPECT_EQ(j.at("output_data_type").get<DataType>(), DataType::FP32);
    ASSERT_TRUE(j.contains("parameters"));
    ASSERT_TRUE(j.at("parameters").contains("weights"));
    ASSERT_TRUE(j.at("parameters").contains("biases"));
    EXPECT_EQ(j.at("parameters").at("weights").at("shape").get<vector<uint64_t>>(), (vector<uint64_t>{4, 3}));
    EXPECT_EQ(j.at("parameters").at("biases").at("shape").get<vector<uint64_t>>(), (vector<uint64_t>{3}));
}

TEST(FullyConnectedApi, StampsAsPhysicalCustomLayerAndAllocatesParameters) {
    constexpr uint32_t batchSize = 4;
    Api::Network network("testNetwork");

    Api::NetworkInput input = Api::NetworkInput::Builder().network(network).name("input").dimensions({5}).dataType(DataType::FP16).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input.getFeatureOutput())
                                 .numOutputFeatures(2)
                                 .hasBias(true)
                                 .weightsDataType(DataType::FP32)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput()).dataType(DataType::FP32).build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);

    ASSERT_EQ(fixture.stampedNetwork->getNumTrainableLayers(), 1u);
    EXPECT_EQ(fixture.physicalFc->getLayerType(), "CustomLayer<FullyConnected>");
    EXPECT_EQ(fixture.physicalFc->listParameters(), (vector<string>{"weights", "biases"}));

    Impl::Tensor weights = fixture.physicalFc->getParameter("weights")->getStorage();
    Impl::Tensor biases = fixture.physicalFc->getParameter("biases")->getStorage();
    EXPECT_EQ(weights.getDimensions(), (vector<uint64_t>{5, 2}));
    EXPECT_EQ(biases.getDimensions(), (vector<uint64_t>{2}));
    EXPECT_EQ(weights.getDataType(), DataType::FP32);
    EXPECT_EQ(biases.getDataType(), DataType::FP32);
}

TEST(FullyConnectedApi, ForwardNumericalWithBias) {
    constexpr uint32_t batchSize = 3;
    constexpr uint32_t numInputFeatures = 4;
    constexpr uint32_t numOutputFeatures = 3;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.0f, 4.0f, -0.5f, 0.25f, -3.0f, 1.5f, 2.0f};
    const vector<float> weightValues = {0.5f, -1.0f, 2.0f, -0.25f, 0.75f, 1.5f, 1.25f, -2.0f, -0.5f, 0.0f, 1.0f, -1.5f};
    const vector<float> biasValues = {0.25f, -0.5f, 1.0f};

    Api::Network network("testNetwork");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(dataType).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input.getFeatureOutput())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(true)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput()).dataType(dataType).build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);
    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
    setParameterTensor(fixture.physicalFc->getParameter("biases"), biasValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numInputFeatures}));
    writeCpuTensor(featureInHost, inputValues);

    const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
    const vector<float> expected =
        fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
    expectAllClose(actual, expected);
}

TEST(FullyConnectedApi, ForwardNumericalWithoutBias) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 4;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = {2.0f, -1.0f, 0.25f, -3.0f, 4.0f, 1.5f};
    const vector<float> weightValues = {1.0f, -2.0f, 0.5f, 0.0f, -1.5f, 0.25f, 2.0f, -0.75f, 0.5f, 1.25f, -1.0f, 3.0f};

    Api::Network network("testNetwork");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(dataType).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input.getFeatureOutput())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(false)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput()).dataType(dataType).build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);
    ASSERT_EQ(fixture.physicalFc->listParameters(), (vector<string>{"weights"}));

    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numInputFeatures}));
    writeCpuTensor(featureInHost, inputValues);

    const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
    const vector<float> expected =
        fullyConnectedReference(inputValues, weightValues, {}, batchSize, numInputFeatures, numOutputFeatures, false);
    expectAllClose(actual, expected);
}

TEST(FullyConnectedApi, ForwardFlattensHigherRankFeatureInput) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t flattenedFeatures = 4;
    constexpr uint32_t numOutputFeatures = 2;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = {1.0f, 2.0f, -1.0f, 0.5f, -2.0f, 1.5f, 0.25f, 3.0f};
    const vector<float> weightValues = {0.5f, 1.0f, -1.0f, 0.25f, 2.0f, -0.5f, 1.5f, 0.75f};

    Api::Network network("testNetwork");
    Api::NetworkInput input = Api::NetworkInput::Builder().network(network).name("input").dimensions({2, 2}).dataType(dataType).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input.getFeatureOutput())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(false)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput()).dataType(dataType).build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);
    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, 2, 2}));
    writeCpuTensor(featureInHost, inputValues);

    const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
    const vector<float> expected =
        fullyConnectedReference(inputValues, weightValues, {}, batchSize, flattenedFeatures, numOutputFeatures, false);
    expectAllClose(actual, expected);
}

TEST(FullyConnectedApi, ForwardAppliesEpilogueAfterMatmulBiasAndActivation) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numInputFeatures = 2;
    constexpr uint32_t numOutputFeatures = 2;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValues = {1.0f, 2.0f, -1.0f, 0.5f};
    const vector<float> weightValues = {0.5f, -1.0f, 2.0f, 0.25f};
    const vector<float> biasValues = {0.25f, -0.5f};

    auto epilogueInput = Api::FullyConnected::epilogueInput(DataType::FP32, DataType::FP32);
    auto epilogue = epilogueInput * Impl::Expression::constantScalar(2.0) + Impl::Expression::constantScalar(1.0);
    std::shared_ptr<Api::Activation> relu = Api::Relu::Builder().build();

    Api::Network network("testNetwork");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(dataType).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input.getFeatureOutput())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(true)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .activation(relu)
                                 .epilogue(epilogue)
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput()).dataType(dataType).build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);
    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
    setParameterTensor(fixture.physicalFc->getParameter("biases"), biasValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numInputFeatures}));
    writeCpuTensor(featureInHost, inputValues);

    vector<float> expected =
        fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
    for (float& value : expected) {
        value = std::max(0.0f, value);
        value = value * 2.0f + 1.0f;
    }

    const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
    expectAllClose(actual, expected, 1e-5f, 1e-5f);
}

TEST(FullyConnectedApi, ForwardHonorsExplicitInputWeightComputeAndOutputDtypes) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 2;

    const vector<float> inputValues = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.0f};
    const vector<float> weightValues = {0.5f, -1.0f, 2.0f, -0.25f, 0.75f, 1.5f};

    Api::Network network("testNetwork");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(DataType::FP16).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input.getFeatureOutput())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(false)
                                 .weightsDataType(DataType::FP16)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput()).dataType(DataType::FP32).build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);
    EXPECT_EQ(fixture.physicalFc->getParameter("weights")->getStorage().get().getDataType(), DataType::FP16);
    EXPECT_EQ(fixture.physicalOutput->getFeatureOutput().get().getDataType(), DataType::FP32);

    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::FP16, {batchSize, numInputFeatures}));
    writeCpuTensor(featureInHost, inputValues);

    const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
    const vector<float> expected =
        fullyConnectedReference(inputValues, weightValues, {}, batchSize, numInputFeatures, numOutputFeatures, false);
    expectAllClose(actual, expected, 3e-2f, 3e-2f);
}

TEST(FullyConnectedApi, BackwardNumericalWithSgdUpdate) {
    constexpr uint32_t batchSize = 4;
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 2;
    constexpr float learningRate = 0.1f;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValues = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.0f, -0.25f, 1.25f, -3.0f, 0.75f, -0.5f, 1.5f};
    const vector<float> weightValues = {0.5f, -1.0f, 2.0f, -0.25f, 0.75f, 1.5f};
    const vector<float> biasValues = {0.25f, -0.5f};
    const vector<float> errorInputValues = {1.0f, -0.5f, 0.25f, 2.0f, -1.0f, 1.5f, 0.75f, -0.25f};

    Api::Network network("testNetwork");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(dataType).build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput()).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(inputRivet.getFeatureOutput())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(true)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput())
                                    .dataType(dataType)
                                    .build();
    shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder().network(network).initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();
    (void)sgd;

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, false);
    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
    setParameterTensor(fixture.physicalFc->getParameter("biases"), biasValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numInputFeatures}));
    writeCpuTensor(featureInHost, inputValues);
    const vector<float> actualForward = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);

    ASSERT_GT(fixture.physicalFc->getErrorInputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorInputs()[0].isPresent());
    ASSERT_GT(fixture.physicalFc->getErrorOutputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorOutputs()[0].isPresent());
    ASSERT_TRUE(fixture.physicalFc->getGradientUpdateStream().isPresent());

    Impl::Tensor fcErrorInput = fixture.physicalFc->getErrorInputs()[0];
    Impl::Tensor fcErrorInputHost = fcErrorInput.clone(cpuPlacement);
    writeCpuTensor(fcErrorInputHost, errorInputValues);
    fcErrorInput.copyFromAsync(fcErrorInputHost, stream);
    fixture.physicalFc->backward(fcErrorInput, batchSize);

    Stream gradientStream = fixture.physicalFc->getGradientUpdateStream();
    Impl::Tensor errorOutputHost = copyTensorToCpu(fixture.physicalFc->getErrorOutputs()[0], stream);
    Impl::Tensor weightsAfterHost = copyTensorToCpu(fixture.physicalFc->getParameter("weights")->getStorage(), gradientStream);
    Impl::Tensor biasesAfterHost = copyTensorToCpu(fixture.physicalFc->getParameter("biases")->getStorage(), gradientStream);
    Impl::Tensor weightsGradHost =
        copyTensorToCpu(fixture.physicalFc->getParameter("weights")->getOptimizer()->getWeightsGradient().get(), gradientStream);
    Impl::Tensor biasesGradHost =
        copyTensorToCpu(fixture.physicalFc->getParameter("biases")->getOptimizer()->getWeightsGradient().get(), gradientStream);

    stream.synchronize();
    gradientStream.synchronize();

    const vector<float> expectedForward =
        fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
    const vector<float> expectedErrorOutput =
        fullyConnectedBackwardErrorReference(errorInputValues, weightValues, batchSize, numInputFeatures, numOutputFeatures);
    const vector<float> expectedWeightsGrad =
        fullyConnectedWeightGradReference(inputValues, errorInputValues, batchSize, numInputFeatures, numOutputFeatures);
    const vector<float> expectedBiasesGrad = fullyConnectedBiasGradReference(errorInputValues, batchSize, numOutputFeatures);
    const vector<float> expectedWeightsAfter = sgdUpdatedReference(weightValues, expectedWeightsGrad, batchSize, learningRate);
    const vector<float> expectedBiasesAfter = sgdUpdatedReference(biasValues, expectedBiasesGrad, batchSize, learningRate);

    expectAllClose(actualForward, expectedForward, 1e-5f, 1e-5f, "feature out");
    expectAllClose(readCpuTensor(errorOutputHost), expectedErrorOutput, 1e-5f, 1e-5f, "error out");
    expectAllClose(readCpuTensor(weightsGradHost), expectedWeightsGrad, 1e-5f, 1e-5f, "weights grad");
    expectAllClose(readCpuTensor(biasesGradHost), expectedBiasesGrad, 1e-5f, 1e-5f, "biases grad");
    expectAllClose(readCpuTensor(weightsAfterHost), expectedWeightsAfter, 1e-5f, 1e-5f, "weights after");
    expectAllClose(readCpuTensor(biasesAfterHost), expectedBiasesAfter, 1e-5f, 1e-5f, "biases after");
}
