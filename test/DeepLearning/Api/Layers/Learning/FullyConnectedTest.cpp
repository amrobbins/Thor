#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Layers/Learning/FullyConnected.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Adam.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedFullyConnected.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include <regex>
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "DeepLearning/Api/Layers/Activations/Relu.h"

using namespace std;
namespace Api = Thor;
namespace Impl = ThorImplementation;
using DataType = Impl::DataType;

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
        case DataType::BF16: {
            auto* ptr = static_cast<__nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                ptr[i] = __float2bfloat16(values[i]);
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
        case DataType::BF16: {
            const auto* ptr = static_cast<const __nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i)
                values[i] = __bfloat162float(ptr[i]);
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

float roundToDataType(float value, DataType dataType) {
    switch (dataType) {
        case DataType::FP16:
            return __half2float(__float2half(value));
        case DataType::BF16:
            return __bfloat162float(__float2bfloat16(value));
        case DataType::FP32:
            return value;
        default:
            throw std::invalid_argument("Unsupported dtype in roundToDataType test helper.");
    }
}

vector<float> roundToDataType(const vector<float>& values, DataType dataType) {
    vector<float> rounded(values.size());
    for (uint64_t i = 0; i < values.size(); ++i)
        rounded[i] = roundToDataType(values[i], dataType);
    return rounded;
}

Impl::Tensor copyTensorToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpuTensor = tensor.clone(cpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return cpuTensor;
}

vector<uint8_t> readTensorBytes(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpuTensor = copyTensorToCpu(tensor, stream);
    vector<uint8_t> bytes(cpuTensor.getArraySizeInBytes());
    std::memcpy(bytes.data(), cpuTensor.getMemPtr(), bytes.size());
    return bytes;
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
    ASSERT_TRUE(parameter->getStorage().has_value());
    Impl::Tensor deviceTensor = parameter->getStorage().value();
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
    return readCpuTensor(physicalOutput.getFeatureOutput().value());
}

struct PlacedRaggedFullyConnectedFixture {
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    Impl::StampedNetwork* stampedNetwork = nullptr;
    shared_ptr<Impl::NetworkInput> physicalValuesInput;
    shared_ptr<Impl::NetworkOutput> physicalOutput;
    shared_ptr<Impl::RaggedFullyConnected> physicalFc;
};

PlacedRaggedFullyConnectedFixture placeSingleRaggedFullyConnectedNetwork(Api::Network& network,
                                                                         const Api::NetworkOutput& apiOutput,
                                                                         const Api::FullyConnected& apiFc,
                                                                         const string& raggedInputName,
                                                                         uint32_t batchSize,
                                                                         bool inferenceOnly) {
    vector<Event> initDoneEvents;
    PlacedRaggedFullyConnectedFixture fixture;
    fixture.placedNetwork = network.place(batchSize, initDoneEvents, inferenceOnly);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(fixture.placedNetwork, nullptr);
    fixture.stampedNetwork = &fixture.placedNetwork->getStampedNetwork(0);
    fixture.physicalValuesInput = fixture.stampedNetwork->getNamedInput(raggedInputName + ".values");
    fixture.physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiOutput.getId()));
    fixture.physicalFc =
        dynamic_pointer_cast<Impl::RaggedFullyConnected>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiFc.getId()));
    EXPECT_NE(fixture.physicalValuesInput, nullptr);
    EXPECT_NE(fixture.physicalOutput, nullptr);
    EXPECT_NE(fixture.physicalFc, nullptr);
    return fixture;
}

struct FullyConnectedAdamPassReference {
    vector<float> featureOut;
    vector<float> errorOut;

    vector<float> weightsGrad;
    vector<float> weightsM;
    vector<float> weightsV;
    vector<float> weightsAfter;

    vector<float> biasesGrad;
    vector<float> biasesM;
    vector<float> biasesV;
    vector<float> biasesAfter;
};

vector<FullyConnectedAdamPassReference> computeFullyConnectedAdamReferenceSequence(const vector<vector<float>>& inputValuesByPass,
                                                                                   const vector<vector<float>>& errorInputValuesByPass,
                                                                                   const vector<float>& initialWeightValues,
                                                                                   const vector<float>& initialBiasValues,
                                                                                   uint64_t batchSize,
                                                                                   uint64_t numInputFeatures,
                                                                                   uint64_t numOutputFeatures,
                                                                                   bool hasBias,
                                                                                   float lossScalingFactor,
                                                                                   float alpha,
                                                                                   float beta1,
                                                                                   float beta2,
                                                                                   float epsilon) {
    EXPECT_EQ(inputValuesByPass.size(), errorInputValuesByPass.size());

    vector<FullyConnectedAdamPassReference> refs;
    refs.reserve(inputValuesByPass.size());

    vector<float> weights = initialWeightValues;
    vector<float> weightsM(weights.size(), 0.0f);
    vector<float> weightsV(weights.size(), 0.0f);

    vector<float> biases = initialBiasValues;
    vector<float> biasesM(hasBias ? biases.size() : 0, 0.0f);
    vector<float> biasesV(hasBias ? biases.size() : 0, 0.0f);

    const float scale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

    for (uint64_t pass = 0; pass < inputValuesByPass.size(); ++pass) {
        FullyConnectedAdamPassReference ref;

        ref.featureOut =
            fullyConnectedReference(inputValuesByPass[pass], weights, biases, batchSize, numInputFeatures, numOutputFeatures, hasBias);

        ref.errorOut =
            fullyConnectedBackwardErrorReference(errorInputValuesByPass[pass], weights, batchSize, numInputFeatures, numOutputFeatures);

        ref.weightsGrad = fullyConnectedWeightGradReference(
            inputValuesByPass[pass], errorInputValuesByPass[pass], batchSize, numInputFeatures, numOutputFeatures);

        const uint64_t t = pass + 1;
        const double alphaT64 =
            static_cast<double>(alpha) * sqrt(1.0 - pow(static_cast<double>(beta2), t)) / (1.0 - pow(static_cast<double>(beta1), t));
        const float alphaT = static_cast<float>(alphaT64);

        ref.weightsM.resize(weights.size());
        ref.weightsV.resize(weights.size());
        ref.weightsAfter.resize(weights.size());

        for (uint64_t i = 0; i < weights.size(); ++i) {
            const float g = ref.weightsGrad[i] * scale;

            weightsM[i] = beta1 * weightsM[i] + (1.0f - beta1) * g;
            weightsV[i] = beta2 * weightsV[i] + (1.0f - beta2) * g * g;
            weights[i] = weights[i] - alphaT * weightsM[i] / (sqrt(weightsV[i]) + epsilon);

            ref.weightsM[i] = weightsM[i];
            ref.weightsV[i] = weightsV[i];
            ref.weightsAfter[i] = weights[i];
        }

        if (hasBias) {
            ref.biasesGrad = fullyConnectedBiasGradReference(errorInputValuesByPass[pass], batchSize, numOutputFeatures);

            ref.biasesM.resize(biases.size());
            ref.biasesV.resize(biases.size());
            ref.biasesAfter.resize(biases.size());

            for (uint64_t i = 0; i < biases.size(); ++i) {
                const float g = ref.biasesGrad[i] * scale;

                biasesM[i] = beta1 * biasesM[i] + (1.0f - beta1) * g;
                biasesV[i] = beta2 * biasesV[i] + (1.0f - beta2) * g * g;
                biases[i] = biases[i] - alphaT * biasesM[i] / (sqrt(biasesV[i]) + epsilon);

                ref.biasesM[i] = biasesM[i];
                ref.biasesV[i] = biasesV[i];
                ref.biasesAfter[i] = biases[i];
            }
        }

        refs.push_back(std::move(ref));
    }

    return refs;
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

float geluReference(float x) { return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f))); }

float cublasLtGeluReference(float x) {
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(kSqrtTwoOverPi * (x + 0.044715f * x * x * x)));
}

vector<float> applyGeluThenTestEpilogue(const vector<float>& values) {
    vector<float> out(values.size());
    for (uint64_t i = 0; i < values.size(); ++i)
        out[i] = 2.0f * geluReference(values[i]) + 1.0f;
    return out;
}

vector<float> applyCublasLtGeluThenTestEpilogue(const vector<float>& values) {
    vector<float> out(values.size());
    for (uint64_t i = 0; i < values.size(); ++i)
        out[i] = 2.0f * cublasLtGeluReference(values[i]) + 1.0f;
    return out;
}

}  // namespace

TEST(FullyConnectedApi, RaggedBuilderPreservesRowPartitionAndUsesTokenWiseOutputShape) {
    Api::Network network("ragged_fc_builder_preserves_partition");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .offsetsDataType(DataType::UINT32)
                                  .trailingDimensions({4})
                                  .maxTotalValues(66)
                                  .batchSize(2)
                                  .build();

    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input)
                                 .numOutputFeatures(3)
                                 .hasBias(true)
                                 .weightsDataType(DataType::FP32)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();

    ASSERT_TRUE(fc.getUseRagged());
    ASSERT_TRUE(fc.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(fc.getRaggedFeatureOutput().has_value());
    EXPECT_EQ(fc.getRaggedFeatureInput()->getValues(), input.getValues());
    EXPECT_EQ(fc.getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(fc.getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{66, 3}));

    const auto architecture = fc.architectureJson();
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    ASSERT_EQ(architecture.at("ragged_inputs").size(), 1u);
    ASSERT_EQ(architecture.at("ragged_outputs").size(), 1u);
    EXPECT_EQ(architecture.at("ragged_outputs").at(0).at("offsets").at("id").get<uint64_t>(), input.getOffsets().getId());
}

TEST(FullyConnectedApi, RaggedBuilderUsesRegularDefaultActivationPattern) {
    Api::Network network("ragged_fc_default_activation");
    Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                  .network(network)
                                  .name("tokens")
                                  .valuesDataType(DataType::FP32)
                                  .trailingDimensions({4})
                                  .maxTotalValues(66)
                                  .batchSize(2)
                                  .build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input)
                                 .numOutputFeatures(3)
                                 .hasBias(false)
                                 .weightsDataType(DataType::FP32)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .build();
    ASSERT_TRUE(fc.getRaggedFeatureOutput().has_value());
    EXPECT_FALSE(fc.architectureJson().at("activation").is_null());
}

TEST(FullyConnectedApi, RaggedArchitectureSaveLoadRoundTripPreservesRowPartition) {
    const std::string networkName = "ragged_fc_arch_round_trip";
    std::filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::RaggedTensor input = Api::RaggedNetworkInput::Builder()
                                      .network(network)
                                      .name("tokens")
                                      .valuesDataType(DataType::FP32)
                                      .offsetsDataType(DataType::UINT64)
                                      .trailingDimensions({4})
                                      .maxTotalValues(66)
                                      .batchSize(2)
                                      .build();
        Api::FullyConnected fc = Api::FullyConnected::Builder()
                                     .network(network)
                                     .featureInput(input)
                                     .numOutputFeatures(3)
                                     .hasBias(true)
                                     .weightsDataType(DataType::FP32)
                                     .computeDataType(DataType::FP32)
                                     .outputDataType(DataType::FP32)
                                     .noActivation()
                                     .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(fc.getRaggedFeatureOutput()->getValues())
                                        .dataType(DataType::FP32)
                                        .build();
        (void)output;

        network.save(archiveDir.string(), true);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());
        std::shared_ptr<Api::FullyConnected> loadedFc = findOnlyLayerOfType<Api::FullyConnected>(loadedNetwork);
        ASSERT_NE(loadedFc, nullptr);
        ASSERT_TRUE(loadedFc->getUseRagged());
        ASSERT_TRUE(loadedFc->getRaggedFeatureInput().has_value());
        ASSERT_TRUE(loadedFc->getRaggedFeatureOutput().has_value());
        EXPECT_EQ(loadedFc->getRaggedFeatureInput()->getBatchSize(), 2u);
        EXPECT_EQ(loadedFc->getRaggedFeatureInput()->getMaxTotalValues(), 66u);
        EXPECT_EQ(loadedFc->getRaggedFeatureInput()->getOffsetsDataType(), DataType::UINT64);
        EXPECT_EQ(loadedFc->getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{66, 3}));
        EXPECT_EQ(loadedFc->getRaggedFeatureOutput()->getOffsets(), loadedFc->getRaggedFeatureInput()->getOffsets());
        EXPECT_TRUE(loadedFc->architectureJson().at("use_ragged").get<bool>());
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(FullyConnectedApi, RaggedForwardBackwardUsesCapacityBucketAndIgnoresInvalidTail) {
    constexpr uint32_t logicalBatchSize = 2;
    constexpr uint64_t fullRows = 66;
    constexpr uint64_t activeRows = 31;
    constexpr uint64_t inputFeatures = 3;
    constexpr uint64_t outputFeatures = 2;
    const DataType dataType = DataType::FP32;

    Api::Network network("ragged_fc_forward_backward_bucketed");
    Api::RaggedTensor networkInput = Api::RaggedNetworkInput::Builder()
                                         .network(network)
                                         .name("tokens")
                                         .valuesDataType(dataType)
                                         .offsetsDataType(DataType::UINT32)
                                         .trailingDimensions({inputFeatures})
                                         .maxTotalValues(fullRows)
                                         .batchSize(logicalBatchSize)
                                         .build();
    Api::GradientRivet inputRivet =
        Api::GradientRivet::Builder().network(network).tensor(networkInput.getValues()).build();
    Api::RaggedTensor raggedInput(inputRivet.getFeatureOutput().value(), networkInput.getOffsets());
    // This test observes only the packed values path. The row partition is intentionally
    // not exposed as an output, so consume that otherwise-dangling structural tensor explicitly.
    (void)Api::Stub::Builder().network(network).inputTensor(networkInput.getOffsets()).build();

    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(raggedInput)
                                 .numOutputFeatures(outputFeatures)
                                 .hasBias(true)
                                 .weightsDataType(dataType)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(dataType)
                                 .noActivation()
                                 .build();
    ASSERT_TRUE(fc.getRaggedFeatureOutput().has_value());
    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(network).tensor(fc.getRaggedFeatureOutput()->getValues()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();
    shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder()
                                   .network(network)
                                   .initialLearningRate(0.001f)
                                   .decay(0.0f)
                                   .momentum(0.0f)
                                   .build();
    (void)sgd;

    PlacedRaggedFullyConnectedFixture fixture =
        placeSingleRaggedFullyConnectedNetwork(network, output, fc, "tokens", logicalBatchSize, false);
    ASSERT_EQ(fixture.physicalFc->selectedCapacityRows(7), 8u);
    ASSERT_EQ(fixture.physicalFc->selectedCapacityRows(9), 16u);
    ASSERT_EQ(fixture.physicalFc->selectedCapacityRows(activeRows), 32u);
    ASSERT_EQ(fixture.physicalFc->selectedCapacityRows(33), 66u);

    vector<float> weights = {0.5f, -0.25f, 1.0f, 0.75f, -0.5f, 0.125f};
    vector<float> biases = {0.2f, -0.3f};
    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weights, stream);
    setParameterTensor(fixture.physicalFc->getParameter("biases"), biases, stream);
    stream.synchronize();

    vector<float> inputValues(fullRows * inputFeatures, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < inputFeatures; ++col) {
            inputValues[row * inputFeatures + col] = static_cast<float>(static_cast<int>(row % 7) - 3) * 0.1f + static_cast<float>(col) * 0.25f;
        }
    }
    for (uint64_t row = activeRows; row < fullRows; ++row) {
        for (uint64_t col = 0; col < inputFeatures; ++col) {
            inputValues[row * inputFeatures + col] = std::numeric_limits<float>::quiet_NaN();
        }
    }
    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {fullRows, inputFeatures}));
    writeCpuTensor(featureInHost, inputValues);
    featureInHost.setRaggedActiveRows(activeRows);

    const vector<float> actualForward =
        runForward(*fixture.physicalValuesInput, *fixture.physicalOutput, featureInHost, logicalBatchSize);
    vector<float> validInput(inputValues.begin(), inputValues.begin() + activeRows * inputFeatures);
    const vector<float> expectedForward =
        fullyConnectedReference(validInput, weights, biases, activeRows, inputFeatures, outputFeatures, true);
    expectAllClose(vector<float>(actualForward.begin(), actualForward.begin() + activeRows * outputFeatures),
                   expectedForward,
                   2e-4f,
                   2e-4f,
                   "ragged feature out");
    for (uint64_t i = activeRows * outputFeatures; i < actualForward.size(); ++i) {
        EXPECT_EQ(actualForward[i], 0.0f) << "invalid ragged FC output tail must be canonical zero";
    }

    ASSERT_GT(fixture.physicalFc->getErrorInputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorInputs()[0].has_value());
    ASSERT_GT(fixture.physicalFc->getErrorOutputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorOutputs()[0].has_value());

    vector<float> dYValues(fullRows * outputFeatures, 0.0f);
    for (uint64_t row = 0; row < activeRows; ++row) {
        for (uint64_t col = 0; col < outputFeatures; ++col) {
            dYValues[row * outputFeatures + col] = static_cast<float>(static_cast<int>((row + 2 * col) % 9) - 4) * 0.2f;
        }
    }
    for (uint64_t row = activeRows; row < fullRows; ++row) {
        for (uint64_t col = 0; col < outputFeatures; ++col) dYValues[row * outputFeatures + col] = -2000.0f - row - col;
    }

    Impl::Tensor fcErrorInput = fixture.physicalFc->getErrorInputs()[0].value();
    Impl::Tensor fcErrorInputHost = fcErrorInput.clone(cpuPlacement);
    writeCpuTensor(fcErrorInputHost, dYValues);
    fcErrorInput.copyFromAsync(fcErrorInputHost, stream);
    fixture.physicalFc->backward(fcErrorInput, logicalBatchSize);

    ASSERT_TRUE(fixture.physicalFc->getGradientUpdateStream().has_value());
    Stream gradientStream = fixture.physicalFc->getGradientUpdateStream().value();
    gradientStream.synchronize();
    stream.synchronize();

    const vector<float> validDY(dYValues.begin(), dYValues.begin() + activeRows * outputFeatures);
    const vector<float> expectedDX =
        fullyConnectedBackwardErrorReference(validDY, weights, activeRows, inputFeatures, outputFeatures);
    const vector<float> expectedDW =
        fullyConnectedWeightGradReference(validInput, validDY, activeRows, inputFeatures, outputFeatures);
    const vector<float> expectedDB = fullyConnectedBiasGradReference(validDY, activeRows, outputFeatures);

    Impl::Tensor dXHost = copyTensorToCpu(fixture.physicalFc->getErrorOutputs()[0].value(), stream);
    const vector<float> actualDX = readCpuTensor(dXHost);
    expectAllClose(vector<float>(actualDX.begin(), actualDX.begin() + activeRows * inputFeatures),
                   expectedDX,
                   2e-4f,
                   2e-4f,
                   "ragged dX");
    for (uint64_t i = activeRows * inputFeatures; i < actualDX.size(); ++i) EXPECT_EQ(actualDX[i], 0.0f);

    // Expression-backed FullyConnected follows the ordinary CustomLayer optimizer path. For a
    // single-input/single-application layer with SGD, CustomLayer fuses the parameter-gradient
    // expression directly into the optimizer update and intentionally does not allocate the
    // optimizer-owned dense gradient tensors. Verify dW/db through the resulting SGD update
    // rather than requiring the legacy materialized-gradient path.
    const auto weightsGradient = fixture.physicalFc->getParameter("weights")->getOptimizer()->getWeightsGradient();
    const auto biasesGradient = fixture.physicalFc->getParameter("biases")->getOptimizer()->getWeightsGradient();
    EXPECT_FALSE(weightsGradient.has_value());
    EXPECT_FALSE(biasesGradient.has_value());

    constexpr float learningRate = 0.001f;
    const float sgdStep =
        learningRate / (static_cast<float>(logicalBatchSize) * Impl::Loss::getLossScalingFactor());
    vector<float> expectedUpdatedWeights = weights;
    vector<float> expectedUpdatedBiases = biases;
    for (uint64_t i = 0; i < expectedUpdatedWeights.size(); ++i) expectedUpdatedWeights[i] -= sgdStep * expectedDW[i];
    for (uint64_t i = 0; i < expectedUpdatedBiases.size(); ++i) expectedUpdatedBiases[i] -= sgdStep * expectedDB[i];

    const vector<float> actualUpdatedWeights = readCpuTensor(
        copyTensorToCpu(fixture.physicalFc->getParameter("weights")->getStorage().value(), gradientStream));
    const vector<float> actualUpdatedBiases = readCpuTensor(
        copyTensorToCpu(fixture.physicalFc->getParameter("biases")->getStorage().value(), gradientStream));
    expectAllClose(actualUpdatedWeights, expectedUpdatedWeights, 2e-4f, 2e-4f, "ragged fused-SGD dW update");
    expectAllClose(actualUpdatedBiases, expectedUpdatedBiases, 2e-4f, 2e-4f, "ragged fused-SGD db update");

    // A training-mode CustomLayer executes one forward per forward/backward cycle. Validate the
    // >32-row/full-capacity bucket only after completing the 31-row backward pass above; issuing
    // two forwards before backward would intentionally reuse the first application state. Reset
    // the parameters because SGD updated them during that backward pass.
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weights, stream);
    setParameterTensor(fixture.physicalFc->getParameter("biases"), biases, stream);
    stream.synchronize();

    constexpr uint64_t aboveSmallBucketRows = 33;
    vector<float> aboveSmallBucketInput(fullRows * inputFeatures, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t row = 0; row < aboveSmallBucketRows; ++row) {
        for (uint64_t col = 0; col < inputFeatures; ++col) {
            aboveSmallBucketInput[row * inputFeatures + col] =
                static_cast<float>(static_cast<int>(row % 5) - 2) * 0.15f + static_cast<float>(col) * 0.2f;
        }
    }
    Impl::Tensor aboveSmallBucketHost(cpuPlacement, Impl::TensorDescriptor(dataType, {fullRows, inputFeatures}));
    writeCpuTensor(aboveSmallBucketHost, aboveSmallBucketInput);
    aboveSmallBucketHost.setRaggedActiveRows(aboveSmallBucketRows);
    const vector<float> actualAboveSmallBucket =
        runForward(*fixture.physicalValuesInput, *fixture.physicalOutput, aboveSmallBucketHost, logicalBatchSize);
    vector<float> validAboveSmallBucketInput(
        aboveSmallBucketInput.begin(), aboveSmallBucketInput.begin() + aboveSmallBucketRows * inputFeatures);
    const vector<float> expectedAboveSmallBucket =
        fullyConnectedReference(validAboveSmallBucketInput,
                                weights,
                                biases,
                                aboveSmallBucketRows,
                                inputFeatures,
                                outputFeatures,
                                true);
    expectAllClose(vector<float>(actualAboveSmallBucket.begin(),
                                 actualAboveSmallBucket.begin() + aboveSmallBucketRows * outputFeatures),
                   expectedAboveSmallBucket,
                   2e-4f,
                   2e-4f,
                   "ragged feature out full bucket");
    for (uint64_t i = aboveSmallBucketRows * outputFeatures; i < actualAboveSmallBucket.size(); ++i) {
        EXPECT_EQ(actualAboveSmallBucket[i], 0.0f);
    }
}

TEST(FullyConnectedApi, BuilderCreatesParameterSpecsOutputsAndConnectionTypes) {
    Api::Network network("testNetwork");
    // Keep this topology/metadata test on a directly supported dtype plan. Mixed
    // operand rejection is covered separately by
    // BuilderRejectsUnsupportedMixedInputAndWeightDtypesInsteadOfFallingBack.
    Api::Tensor featureInput0(DataType::FP32, {4});
    Api::Tensor featureInput1(DataType::FP32, {4});

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

    // This test verifies stamping and parameter allocation, not implicit dtype
    // adaptation. Use the same FP32 storage dtype for the input and weights.
    Api::NetworkInput input = Api::NetworkInput::Builder().network(network).name("input").dimensions({5}).dataType(DataType::FP32).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input.getFeatureOutput().value())
                                 .numOutputFeatures(2)
                                 .hasBias(true)
                                 .weightsDataType(DataType::FP32)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(fc.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);

    ASSERT_EQ(fixture.stampedNetwork->getNumTrainableLayers(), 1u);
    EXPECT_TRUE(std::regex_match(fixture.physicalFc->getLayerType(), std::regex(R"(CustomLayer<FullyConnected#[0-9]+>)")));
    EXPECT_EQ(fixture.physicalFc->listParameters(), (vector<string>{"weights", "biases"}));

    Impl::Tensor weights = fixture.physicalFc->getParameter("weights")->getStorage().value();
    Impl::Tensor biases = fixture.physicalFc->getParameter("biases")->getStorage().value();
    EXPECT_EQ(weights.getDimensions(), (vector<uint64_t>{5, 2}));
    EXPECT_EQ(biases.getDimensions(), (vector<uint64_t>{2}));
    EXPECT_EQ(weights.getDataType(), DataType::FP32);
    EXPECT_EQ(biases.getDataType(), DataType::FP32);
}

TEST(FullyConnectedApi, DefaultsToGeluActivationWhenActivationIsOmitted) {
    Api::Network network("testNetwork");
    Api::Tensor featureInput(DataType::FP32, {4});

    Api::FullyConnected fc =
        Api::FullyConnected::Builder().network(network).featureInput(featureInput).numOutputFeatures(3).hasBias(true).build();

    const nlohmann::json j = fc.architectureJson();
    ASSERT_TRUE(j.contains("activation"));
    ASSERT_FALSE(j.at("activation").is_null());
    EXPECT_EQ(j.at("activation").at("layer_type").get<string>(), "gelu");
}

TEST(FullyConnectedApi, Fp32StorageDefaultsToFp32ComputeAndAllowsExplicitTf32) {
    Api::Network network("testNetwork");
    Api::Tensor featureInput(DataType::FP32, {4});

    Api::FullyConnected defaultFc =
        Api::FullyConnected::Builder().network(network).featureInput(featureInput).numOutputFeatures(3).hasBias(false).noActivation().build();
    EXPECT_EQ(defaultFc.getWeightsDataType(), DataType::FP32);
    EXPECT_EQ(defaultFc.getComputeDataType(), DataType::FP32);
    EXPECT_EQ(defaultFc.getOutputDataType(), DataType::FP32);
    EXPECT_EQ(defaultFc.architectureJson().at("compute_data_type").get<DataType>(), DataType::FP32);

    Api::FullyConnected tf32Fc = Api::FullyConnected::Builder()
                                     .network(network)
                                     .featureInput(featureInput)
                                     .numOutputFeatures(3)
                                     .hasBias(false)
                                     .computeDataType(DataType::TF32)
                                     .noActivation()
                                     .build();
    EXPECT_EQ(tf32Fc.getComputeDataType(), DataType::TF32);
    EXPECT_EQ(tf32Fc.architectureJson().at("compute_data_type").get<DataType>(), DataType::TF32);
}

TEST(FullyConnectedApi, ArchitectureSaveLoadRoundTripPreservesGeluActivationEpilogueParametersAndRuns) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 2;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValues = {1.0f, -2.0f, 0.5f, -1.0f, 3.0f, 2.0f};
    const vector<float> weightValues = {0.5f, -1.0f, 1.5f, 0.25f, -0.75f, 2.0f};
    const vector<float> biasValues = {0.1f, -0.2f};

    const std::string networkName = "fc_arch_round_trip";
    std::filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input =
            Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(dataType).build();

        Impl::Expression epilogueInput = Api::FullyConnected::epilogueInput(dataType, dataType);
        Impl::Expression epilogue = epilogueInput * 2.0f + 1.0f;

        Api::FullyConnected fc = Api::FullyConnected::Builder()
                                     .network(network)
                                     .featureInput(input.getFeatureOutput().value())
                                     .numOutputFeatures(numOutputFeatures)
                                     .hasBias(true)
                                     .weightsDataType(dataType)
                                     .computeDataType(dataType)
                                     .outputDataType(dataType)
                                     .epilogue(epilogue)
                                     .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(fc.getFeatureOutput().value())
                                        .dataType(dataType)
                                        .build();

        network.save(archiveDir.string(), true);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        ASSERT_EQ(loadedNetwork.getNumLayers(), 3u);
        std::shared_ptr<Api::NetworkInput> loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        std::shared_ptr<Api::FullyConnected> loadedFc = findOnlyLayerOfType<Api::FullyConnected>(loadedNetwork);
        std::shared_ptr<Api::NetworkOutput> loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedFc, nullptr);
        ASSERT_NE(loadedOutput, nullptr);

        const nlohmann::json j = loadedFc->architectureJson();
        EXPECT_EQ(j.at("layer_type").get<string>(), "fully_connected");
        EXPECT_EQ(j.at("num_output_features").get<uint32_t>(), numOutputFeatures);
        EXPECT_TRUE(j.at("has_bias").get<bool>());
        EXPECT_EQ(j.at("weights_data_type").get<DataType>(), dataType);
        EXPECT_EQ(j.at("compute_data_type").get<DataType>(), dataType);
        EXPECT_EQ(j.at("output_data_type").get<DataType>(), dataType);
        ASSERT_FALSE(j.at("activation").is_null());
        EXPECT_EQ(j.at("activation").at("layer_type").get<string>(), "gelu");
        ASSERT_FALSE(j.at("epilogue").is_null());
        ASSERT_TRUE(j.at("parameters").contains("weights"));
        ASSERT_TRUE(j.at("parameters").contains("biases"));
        EXPECT_EQ(j.at("parameters").at("weights").at("shape").get<vector<uint64_t>>(),
                  (vector<uint64_t>{numInputFeatures, numOutputFeatures}));
        EXPECT_EQ(j.at("parameters").at("biases").at("shape").get<vector<uint64_t>>(), (vector<uint64_t>{numOutputFeatures}));

        PlacedFullyConnectedFixture fixture =
            placeSingleFullyConnectedNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedFc, batchSize, true);
        Stream stream = fixture.physicalFc->getStreams()[0];
        setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
        setParameterTensor(fixture.physicalFc->getParameter("biases"), biasValues, stream);
        stream.synchronize();

        Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numInputFeatures}));
        writeCpuTensor(featureInHost, inputValues);

        const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
        const vector<float> affine =
            fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
        // This FC graph lowers matmul+bias+GELU to the cuBLASLt GELU epilogue,
        // whose documented GELU is the tanh-polynomial approximation. Keep the
        // layer activation's exact-GELU semantics for unfused/backward paths,
        // but validate this fused forward path against the math it actually runs.
        const vector<float> expected = applyCublasLtGeluThenTestEpilogue(affine);
        expectAllClose(actual, expected, 3e-4f, 3e-4f, "loaded FC cuBLASLt Gelu+epilogue output");
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(FullyConnectedApi, PlacedSaveLoadRoundTripRestoresParameterStorageAndRuns) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 2;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValues = {2.0f, -1.0f, 0.25f, -0.5f, 1.5f, 3.0f};
    const vector<float> weightValues = {0.25f, -0.5f, 1.25f, 0.75f, -1.5f, 2.0f};
    const vector<float> biasValues = {0.5f, -1.0f};

    const std::string networkName = "fc_state_round_trip";
    std::filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input =
            Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(dataType).build();
        Api::FullyConnected fc = Api::FullyConnected::Builder()
                                     .network(network)
                                     .featureInput(input.getFeatureOutput().value())
                                     .numOutputFeatures(numOutputFeatures)
                                     .hasBias(true)
                                     .weightsDataType(dataType)
                                     .computeDataType(dataType)
                                     .outputDataType(dataType)
                                     .noActivation()
                                     .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(fc.getFeatureOutput().value())
                                        .dataType(dataType)
                                        .build();

        PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);
        Stream stream = fixture.physicalFc->getStreams()[0];
        setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
        setParameterTensor(fixture.physicalFc->getParameter("biases"), biasValues, stream);
        stream.synchronize();

        fixture.placedNetwork->save(archiveDir.string(), true, false);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());
        std::shared_ptr<Api::NetworkInput> loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        std::shared_ptr<Api::FullyConnected> loadedFc = findOnlyLayerOfType<Api::FullyConnected>(loadedNetwork);
        std::shared_ptr<Api::NetworkOutput> loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedFc, nullptr);
        ASSERT_NE(loadedOutput, nullptr);

        PlacedFullyConnectedFixture loadedFixture =
            placeSingleFullyConnectedNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedFc, batchSize, true);

        Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numInputFeatures}));
        writeCpuTensor(featureInHost, inputValues);

        const vector<float> actual = runForward(*loadedFixture.physicalInput, *loadedFixture.physicalOutput, featureInHost, batchSize);
        const vector<float> expected =
            fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
        expectAllClose(actual, expected, 2e-4f, 2e-4f, "loaded FC restored parameter output");
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
}

TEST(FullyConnectedApi, PlacedSaveLoadRoundTripRestoresBf16WeightBytesAndExecution) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 2;

    const vector<float> inputValues = {2.0f, -1.0f, 0.25f, -0.5f, 1.5f, 3.0f};
    const vector<float> weightValues = {0.25f, -0.5f, 1.25f, 0.75f, -1.5f, 2.0f};
    const vector<float> biasValues = {0.5f, -1.0f};

    const std::string networkName = "fc_bf16_state_round_trip";
    std::filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input = Api::NetworkInput::Builder()
                                      .network(network)
                                      .name("input")
                                      .dimensions({numInputFeatures})
                                      .dataType(DataType::BF16)
                                      .build();
        Api::FullyConnected fc = Api::FullyConnected::Builder()
                                     .network(network)
                                     .featureInput(input.getFeatureOutput().value())
                                     .numOutputFeatures(numOutputFeatures)
                                     .hasBias(true)
                                     .weightsDataType(DataType::BF16)
                                     .computeDataType(DataType::BF16)
                                     .outputDataType(DataType::FP32)
                                     .noActivation()
                                     .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(fc.getFeatureOutput().value())
                                        .dataType(DataType::FP32)
                                        .build();

        PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);
        Stream stream = fixture.physicalFc->getStreams()[0];
        std::shared_ptr<Impl::PhysicalParameter> sourceWeights = fixture.physicalFc->getParameter("weights");
        std::shared_ptr<Impl::PhysicalParameter> sourceBiases = fixture.physicalFc->getParameter("biases");
        setParameterTensor(sourceWeights, weightValues, stream);
        setParameterTensor(sourceBiases, biasValues, stream);
        stream.synchronize();

        const vector<uint8_t> sourceWeightBytes = readTensorBytes(sourceWeights->getStorage().value(), stream);
        const vector<uint8_t> sourceBiasBytes = readTensorBytes(sourceBiases->getStorage().value(), stream);

        Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::BF16, {batchSize, numInputFeatures}));
        writeCpuTensor(featureInHost, inputValues);
        const vector<float> expected =
            fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
        const vector<float> sourceOutput = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
        expectAllClose(sourceOutput, expected, 6e-2f, 6e-2f, "source FC BF16 parameter output");

        fixture.placedNetwork->save(archiveDir.string(), true, false);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());
        std::shared_ptr<Api::NetworkInput> loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        std::shared_ptr<Api::FullyConnected> loadedFc = findOnlyLayerOfType<Api::FullyConnected>(loadedNetwork);
        std::shared_ptr<Api::NetworkOutput> loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedFc, nullptr);
        ASSERT_NE(loadedOutput, nullptr);
        EXPECT_EQ(loadedInput->getFeatureOutput()->getDataType(), DataType::BF16);
        EXPECT_EQ(loadedFc->getWeightsDataType(), DataType::BF16);
        EXPECT_EQ(loadedFc->getComputeDataType(), DataType::BF16);
        EXPECT_EQ(loadedFc->getOutputDataType(), DataType::FP32);

        PlacedFullyConnectedFixture loadedFixture =
            placeSingleFullyConnectedNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedFc, batchSize, true);
        Stream loadedStream = loadedFixture.physicalFc->getStreams()[0];
        std::shared_ptr<Impl::PhysicalParameter> loadedWeights = loadedFixture.physicalFc->getParameter("weights");
        std::shared_ptr<Impl::PhysicalParameter> loadedBiases = loadedFixture.physicalFc->getParameter("biases");
        ASSERT_EQ(loadedWeights->getStorage()->getDataType(), DataType::BF16);
        ASSERT_EQ(loadedBiases->getStorage()->getDataType(), DataType::FP32);
        EXPECT_EQ(readTensorBytes(loadedWeights->getStorage().value(), loadedStream), sourceWeightBytes);
        EXPECT_EQ(readTensorBytes(loadedBiases->getStorage().value(), loadedStream), sourceBiasBytes);

        const vector<float> actual =
            runForward(*loadedFixture.physicalInput, *loadedFixture.physicalOutput, featureInHost, batchSize);
        expectAllClose(actual, sourceOutput, 6e-2f, 6e-2f, "loaded FC BF16 round-trip output");
        expectAllClose(actual, expected, 6e-2f, 6e-2f, "loaded FC BF16 restored parameter output");
    } catch (...) {
        std::filesystem::remove_all(archiveDir);
        throw;
    }
    std::filesystem::remove_all(archiveDir);
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
                                 .featureInput(input.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(true)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput().value()).dataType(dataType).build();

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
                                 .featureInput(input.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(false)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput().value()).dataType(dataType).build();

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
                                 .featureInput(input.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(false)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput().value()).dataType(dataType).build();

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

TEST(FullyConnectedApi, BackwardFlattensHigherRankFeatureInputWithoutMaterializedFlatten) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t flattenedFeatures = 4;
    constexpr uint32_t numOutputFeatures = 2;
    constexpr float learningRate = 0.1f;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValues = {1.0f, 2.0f, -1.0f, 0.5f, -2.0f, 1.5f, 0.25f, 3.0f};
    const vector<float> weightValues = {0.5f, -1.0f, 2.0f, -0.25f, 0.75f, 1.5f, -1.25f, 0.5f};
    const vector<float> errorInputValues = {1.0f, -0.5f, 0.25f, 2.0f};

    Api::Network network("testNetwork");
    Api::NetworkInput input = Api::NetworkInput::Builder().network(network).name("input").dimensions({2, 2}).dataType(dataType).build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(inputRivet.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(false)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();
    shared_ptr<Api::Sgd> sgd = Api::Sgd::Builder().network(network).initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();
    (void)sgd;

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, false);
    ASSERT_EQ(fixture.physicalFc->listParameters(), (vector<string>{"weights"}));

    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, 2, 2}));
    writeCpuTensor(featureInHost, inputValues);
    const vector<float> actualForward = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);

    ASSERT_GT(fixture.physicalFc->getErrorInputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorInputs()[0].has_value());
    ASSERT_GT(fixture.physicalFc->getErrorOutputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorOutputs()[0].has_value());
    EXPECT_EQ(fixture.physicalFc->getErrorOutputs()[0].value().getDimensions(), (vector<uint64_t>{batchSize, 2, 2}));
    ASSERT_TRUE(fixture.physicalFc->getGradientUpdateStream().has_value());

    Impl::Tensor fcErrorInput = fixture.physicalFc->getErrorInputs()[0].value();
    Impl::Tensor fcErrorInputHost = fcErrorInput.clone(cpuPlacement);
    writeCpuTensor(fcErrorInputHost, errorInputValues);
    fcErrorInput.copyFromAsync(fcErrorInputHost, stream);
    fixture.physicalFc->backward(fcErrorInput, batchSize);

    Stream gradientStream = fixture.physicalFc->getGradientUpdateStream().value();
    Impl::Tensor errorOutputHost = copyTensorToCpu(fixture.physicalFc->getErrorOutputs()[0].value(), stream);
    Impl::Tensor weightsAfterHost = copyTensorToCpu(fixture.physicalFc->getParameter("weights")->getStorage().value(), gradientStream);
    EXPECT_FALSE(fixture.physicalFc->getParameter("weights")->getOptimizer()->getWeightsGradient().has_value())
        << "Fused FullyConnected CustomLayer update should not allocate a dense weights gradient tensor.";

    stream.synchronize();
    gradientStream.synchronize();

    const vector<float> expectedForward =
        fullyConnectedReference(inputValues, weightValues, {}, batchSize, flattenedFeatures, numOutputFeatures, false);
    const vector<float> expectedErrorOutput =
        fullyConnectedBackwardErrorReference(errorInputValues, weightValues, batchSize, flattenedFeatures, numOutputFeatures);
    const vector<float> expectedWeightsGrad =
        fullyConnectedWeightGradReference(inputValues, errorInputValues, batchSize, flattenedFeatures, numOutputFeatures);
    const vector<float> expectedWeightsAfter = sgdUpdatedReference(weightValues, expectedWeightsGrad, batchSize, learningRate);

    expectAllClose(actualForward, expectedForward, 1e-5f, 1e-5f, "feature out");
    expectAllClose(readCpuTensor(errorOutputHost), expectedErrorOutput, 1e-5f, 1e-5f, "error out");
    expectAllClose(readCpuTensor(weightsAfterHost), expectedWeightsAfter, 1e-5f, 1e-5f, "weights after");
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
                                 .featureInput(input.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(true)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .activation(relu)
                                 .epilogue(epilogue)
                                 .build();
    Api::NetworkOutput output =
        Api::NetworkOutput::Builder().network(network).name("output").inputTensor(fc.getFeatureOutput().value()).dataType(dataType).build();

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
                                 .featureInput(input.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(false)
                                 .weightsDataType(DataType::FP16)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(fc.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, true);
    EXPECT_EQ(fixture.physicalFc->getParameter("weights")->getStorage().value().getDataType(), DataType::FP16);
    EXPECT_EQ(fixture.physicalOutput->getFeatureOutput().value().getDataType(), DataType::FP32);

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

TEST(FullyConnectedApi, BuilderRejectsUnsupportedMixedInputAndWeightDtypesInsteadOfFallingBack) {
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 2;

    Api::Network fp32InputNetwork("fc_rejects_fp32_input_bf16_weights");
    Api::NetworkInput fp32Input = Api::NetworkInput::Builder()
                                      .network(fp32InputNetwork)
                                      .name("input")
                                      .dimensions({numInputFeatures})
                                      .dataType(DataType::FP32)
                                      .build();

    EXPECT_THROW(
        (void)Api::FullyConnected::Builder()
            .network(fp32InputNetwork)
            .featureInput(fp32Input.getFeatureOutput().value())
            .numOutputFeatures(numOutputFeatures)
            .hasBias(false)
            .weightsDataType(DataType::BF16)
            .computeDataType(DataType::FP32)
            .outputDataType(DataType::FP32)
            .noActivation()
            .build(),
        std::invalid_argument);

    Api::Network bf16InputNetwork("fc_rejects_bf16_input_fp32_weights");
    Api::NetworkInput bf16Input = Api::NetworkInput::Builder()
                                      .network(bf16InputNetwork)
                                      .name("input")
                                      .dimensions({numInputFeatures})
                                      .dataType(DataType::BF16)
                                      .build();

    EXPECT_THROW(
        (void)Api::FullyConnected::Builder()
            .network(bf16InputNetwork)
            .featureInput(bf16Input.getFeatureOutput().value())
            .numOutputFeatures(numOutputFeatures)
            .hasBias(false)
            .weightsDataType(DataType::FP32)
            .computeDataType(DataType::FP32)
            .outputDataType(DataType::FP32)
            .noActivation()
            .build(),
        std::invalid_argument);
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
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(inputRivet.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(true)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
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
    ASSERT_TRUE(fixture.physicalFc->getErrorInputs()[0].has_value());
    ASSERT_GT(fixture.physicalFc->getErrorOutputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(fixture.physicalFc->getGradientUpdateStream().has_value());

    Impl::Tensor fcErrorInput = fixture.physicalFc->getErrorInputs()[0].value();
    Impl::Tensor fcErrorInputHost = fcErrorInput.clone(cpuPlacement);
    writeCpuTensor(fcErrorInputHost, errorInputValues);
    fcErrorInput.copyFromAsync(fcErrorInputHost, stream);
    fixture.physicalFc->backward(fcErrorInput, batchSize);

    Stream gradientStream = fixture.physicalFc->getGradientUpdateStream().value();
    Impl::Tensor errorOutputHost = copyTensorToCpu(fixture.physicalFc->getErrorOutputs()[0].value(), stream);
    Impl::Tensor weightsAfterHost = copyTensorToCpu(fixture.physicalFc->getParameter("weights")->getStorage().value(), gradientStream);
    Impl::Tensor biasesAfterHost = copyTensorToCpu(fixture.physicalFc->getParameter("biases")->getStorage().value(), gradientStream);
    EXPECT_FALSE(fixture.physicalFc->getParameter("weights")->getOptimizer()->getWeightsGradient().has_value())
        << "Fused FullyConnected CustomLayer update should not allocate a dense weights gradient tensor.";
    EXPECT_FALSE(fixture.physicalFc->getParameter("biases")->getOptimizer()->getWeightsGradient().has_value())
        << "Fused FullyConnected CustomLayer update should not allocate a dense biases gradient tensor.";

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
    expectAllClose(readCpuTensor(weightsAfterHost), expectedWeightsAfter, 1e-5f, 1e-5f, "weights after");
    expectAllClose(readCpuTensor(biasesAfterHost), expectedBiasesAfter, 1e-5f, 1e-5f, "biases after");
}

void runLowPrecisionInputsFp32OutputSgd(DataType operandDataType) {
    constexpr uint32_t batchSize = 4;
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 2;
    constexpr float learningRate = 0.1f;

    const vector<float> inputValues = {
        1.0f, -2.0f, 0.5f,
        3.0f, -1.5f, 2.0f,
        -0.25f, 1.25f, -3.0f,
        0.75f, -0.5f, 1.5f,
    };
    const vector<float> weightValues = {
        0.5f, -1.0f,
        2.0f, -0.25f,
        0.75f, 1.5f,
    };
    const vector<float> biasValues = {0.25f, -0.5f};
    const vector<float> errorInputValues = {
        0.1234f, -0.5678f,
        0.2345f, 1.2345f,
        -0.3456f, 0.7891f,
        0.4567f, -0.1123f,
    };

    Api::Network network(operandDataType == DataType::BF16 ? "fcBf16InputsFp32OutputSgd" : "fcFp16InputsFp32OutputSgd");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({numInputFeatures})
                                  .dataType(operandDataType)
                                  .build();
    Api::GradientRivet inputRivet =
        Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(inputRivet.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(true)
                                 .weightsDataType(operandDataType)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .noActivation()
                                 .build();
    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();
    shared_ptr<Api::Sgd> sgd =
        Api::Sgd::Builder().network(network).initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();
    (void)sgd;

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, false);
    ASSERT_NE(fixture.physicalFc->getParameter("weights"), nullptr);
    ASSERT_NE(fixture.physicalFc->getParameter("biases"), nullptr);
    ASSERT_TRUE(fixture.physicalFc->getParameter("weights")->getStorage().has_value());
    ASSERT_TRUE(fixture.physicalFc->getParameter("biases")->getStorage().has_value());
    ASSERT_NE(fixture.physicalFc->getParameter("weights")->getOptimizer(), nullptr);
    ASSERT_NE(fixture.physicalFc->getParameter("biases")->getOptimizer(), nullptr);
    ASSERT_TRUE(fixture.physicalOutput->getFeatureOutput().has_value());
    ASSERT_EQ(fixture.physicalFc->getErrorInputs().size(), 1u);
    ASSERT_TRUE(fixture.physicalFc->getErrorInputs()[0].has_value());
    ASSERT_EQ(fixture.physicalFc->getErrorOutputs().size(), 1u);
    ASSERT_TRUE(fixture.physicalFc->getErrorOutputs()[0].has_value());
    ASSERT_EQ(fixture.physicalFc->getParameter("weights")->getStorage().value().getDataType(), operandDataType);
    ASSERT_EQ(fixture.physicalFc->getParameter("biases")->getStorage().value().getDataType(), DataType::FP32);
    ASSERT_EQ(fixture.physicalOutput->getFeatureOutput().value().getDataType(), DataType::FP32);
    ASSERT_EQ(fixture.physicalFc->getErrorInputs()[0].value().getDataType(), DataType::FP32);
    ASSERT_EQ(fixture.physicalFc->getErrorOutputs()[0].value().getDataType(), operandDataType);

    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), weightValues, stream);
    setParameterTensor(fixture.physicalFc->getParameter("biases"), biasValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(operandDataType, {batchSize, numInputFeatures}));
    writeCpuTensor(featureInHost, inputValues);
    const vector<float> actualForward = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);

    Impl::Tensor errorInput = fixture.physicalFc->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuTensor(errorInputHost, errorInputValues);
    errorInput.copyFromAsync(errorInputHost, stream);
    fixture.physicalFc->backward(errorInput, batchSize);

    ASSERT_TRUE(fixture.physicalFc->getGradientUpdateStream().has_value());
    Stream gradientStream = fixture.physicalFc->getGradientUpdateStream().value();
    Impl::Tensor errorOutputHost = copyTensorToCpu(fixture.physicalFc->getErrorOutputs()[0].value(), stream);
    Impl::Tensor weightsAfterHost =
        copyTensorToCpu(fixture.physicalFc->getParameter("weights")->getStorage().value(), gradientStream);
    Impl::Tensor biasesAfterHost =
        copyTensorToCpu(fixture.physicalFc->getParameter("biases")->getStorage().value(), gradientStream);
    stream.synchronize();
    gradientStream.synchronize();

    const vector<float> effectiveInputs = roundToDataType(inputValues, operandDataType);
    const vector<float> effectiveWeights = roundToDataType(weightValues, operandDataType);
    const vector<float> effectiveMatrixGradient = roundToDataType(errorInputValues, operandDataType);

    const vector<float> expectedForward = fullyConnectedReference(
        effectiveInputs, effectiveWeights, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
    const vector<float> expectedErrorOutput = roundToDataType(
        fullyConnectedBackwardErrorReference(
            effectiveMatrixGradient, effectiveWeights, batchSize, numInputFeatures, numOutputFeatures),
        operandDataType);
    const vector<float> expectedWeightGradient = roundToDataType(
        fullyConnectedWeightGradReference(
            effectiveInputs, effectiveMatrixGradient, batchSize, numInputFeatures, numOutputFeatures),
        operandDataType);
    const vector<float> expectedBiasGradient =
        fullyConnectedBiasGradReference(errorInputValues, batchSize, numOutputFeatures);
    const vector<float> expectedWeightsAfter = roundToDataType(
        sgdUpdatedReference(effectiveWeights, expectedWeightGradient, batchSize, learningRate), operandDataType);
    const vector<float> expectedBiasesAfter =
        sgdUpdatedReference(biasValues, expectedBiasGradient, batchSize, learningRate);

    const float tolerance = operandDataType == DataType::BF16 ? 4e-2f : 8e-3f;
    expectAllClose(actualForward, expectedForward, tolerance, tolerance, "low-precision/fp32 feature out");
    expectAllClose(readCpuTensor(errorOutputHost), expectedErrorOutput, tolerance, tolerance,
                   "low-precision/fp32 error out");
    expectAllClose(readCpuTensor(weightsAfterHost), expectedWeightsAfter, tolerance, tolerance,
                   "low-precision/fp32 weights after SGD");
    expectAllClose(readCpuTensor(biasesAfterHost), expectedBiasesAfter, 2e-4f, 2e-4f,
                   "low-precision/fp32 biases after SGD");

    EXPECT_FALSE(fixture.physicalFc->getParameter("weights")->getOptimizer()->getWeightsGradient().has_value());
    EXPECT_FALSE(fixture.physicalFc->getParameter("biases")->getOptimizer()->getWeightsGradient().has_value());
}

TEST(FullyConnectedApi, Bf16InputsAndWeightsFp32OutputTrainWithSgd) {
    runLowPrecisionInputsFp32OutputSgd(DataType::BF16);
}

TEST(FullyConnectedApi, Fp16InputsAndWeightsFp32OutputTrainWithSgd) {
    runLowPrecisionInputsFp32OutputSgd(DataType::FP16);
}

TEST(FullyConnectedApi, Bf16InputsAndWeightsFp32OutputTrainWithAdamAndFp32State) {
    constexpr uint32_t batchSize = 3;
    constexpr uint32_t numInputFeatures = 4;
    constexpr uint32_t numOutputFeatures = 3;
    constexpr float alpha = 0.001f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1e-7f;

    const vector<float> inputValues = {
        1.0f, -2.0f, 0.5f, 0.25f,
        3.0f, -1.5f, 2.0f, -0.75f,
        -0.25f, 1.25f, -3.0f, 0.5f,
    };
    const vector<float> initialWeightValues = {
        0.25f, -0.5f, 0.75f,
        1.0f, -0.25f, 0.5f,
        -0.75f, 0.25f, -0.5f,
        0.5f, 1.0f, -0.25f,
    };
    const vector<float> initialBiasValues = {0.125f, -0.25f, 0.5f};
    const vector<float> errorInputValues = {
        0.1234f, -0.5678f, 1.2345f,
        -0.2345f, 0.7891f, -1.1123f,
        0.4567f, 0.3456f, -0.6789f,
    };

    shared_ptr<Api::Adam> weightsAdam =
        Api::Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    shared_ptr<Api::Adam> biasesAdam =
        Api::Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();

    Api::Network network("fcBf16InputsFp32OutputAdam");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({numInputFeatures})
                                  .dataType(DataType::BF16)
                                  .build();
    Api::GradientRivet inputRivet =
        Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(inputRivet.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(true)
                                 .weightsDataType(DataType::BF16)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(DataType::FP32)
                                 .weightsOptimizer(weightsAdam)
                                 .biasesOptimizer(biasesAdam)
                                 .noActivation()
                                 .build();
    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(DataType::FP32)
                                    .build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, false);
    ASSERT_NE(fixture.physicalFc->getParameter("weights"), nullptr);
    ASSERT_NE(fixture.physicalFc->getParameter("biases"), nullptr);
    ASSERT_NE(fixture.physicalFc->getParameter("weights")->getOptimizer(), nullptr);
    ASSERT_NE(fixture.physicalFc->getParameter("biases")->getOptimizer(), nullptr);
    ASSERT_EQ(fixture.physicalFc->getErrorInputs().size(), 1u);
    ASSERT_TRUE(fixture.physicalFc->getErrorInputs()[0].has_value());
    ASSERT_EQ(fixture.physicalFc->getErrorOutputs().size(), 1u);
    ASSERT_TRUE(fixture.physicalFc->getErrorOutputs()[0].has_value());
    ASSERT_EQ(fixture.physicalFc->getErrorInputs()[0].value().getDataType(), DataType::FP32);
    ASSERT_EQ(fixture.physicalFc->getErrorOutputs()[0].value().getDataType(), DataType::BF16);
    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), initialWeightValues, stream);
    setParameterTensor(fixture.physicalFc->getParameter("biases"), initialBiasValues, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(DataType::BF16, {batchSize, numInputFeatures}));
    writeCpuTensor(featureInHost, inputValues);
    const vector<float> actualForward = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);

    Impl::Tensor errorInput = fixture.physicalFc->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuTensor(errorInputHost, errorInputValues);
    errorInput.copyFromAsync(errorInputHost, stream);
    fixture.physicalFc->backward(errorInput, batchSize);

    ASSERT_TRUE(fixture.physicalFc->getGradientUpdateStream().has_value());
    Stream gradientStream = fixture.physicalFc->getGradientUpdateStream().value();
    shared_ptr<Impl::Optimizer> physicalWeightsAdam = fixture.physicalFc->getParameter("weights")->getOptimizer();
    shared_ptr<Impl::Optimizer> physicalBiasesAdam = fixture.physicalFc->getParameter("biases")->getOptimizer();
    Impl::Tensor weightsAfterHost =
        copyTensorToCpu(fixture.physicalFc->getParameter("weights")->getStorage().value(), gradientStream);
    Impl::Tensor biasesAfterHost =
        copyTensorToCpu(fixture.physicalFc->getParameter("biases")->getStorage().value(), gradientStream);
    Impl::Tensor weightsMHost = copyTensorToCpu(physicalWeightsAdam->getOptimizerParameterTensor("m"), gradientStream);
    Impl::Tensor weightsVHost = copyTensorToCpu(physicalWeightsAdam->getOptimizerParameterTensor("v"), gradientStream);
    Impl::Tensor biasesMHost = copyTensorToCpu(physicalBiasesAdam->getOptimizerParameterTensor("m"), gradientStream);
    Impl::Tensor biasesVHost = copyTensorToCpu(physicalBiasesAdam->getOptimizerParameterTensor("v"), gradientStream);
    stream.synchronize();
    gradientStream.synchronize();

    EXPECT_EQ(physicalWeightsAdam->getOptimizerParameterTensor("m").getDataType(), DataType::FP32);
    EXPECT_EQ(physicalWeightsAdam->getOptimizerParameterTensor("v").getDataType(), DataType::FP32);
    EXPECT_EQ(physicalBiasesAdam->getOptimizerParameterTensor("m").getDataType(), DataType::FP32);
    EXPECT_EQ(physicalBiasesAdam->getOptimizerParameterTensor("v").getDataType(), DataType::FP32);

    const vector<float> effectiveInputs = roundToDataType(inputValues, DataType::BF16);
    const vector<float> effectiveWeights = roundToDataType(initialWeightValues, DataType::BF16);
    const vector<float> effectiveMatrixGradient = roundToDataType(errorInputValues, DataType::BF16);
    const vector<float> expectedForward = fullyConnectedReference(
        effectiveInputs, effectiveWeights, initialBiasValues, batchSize, numInputFeatures, numOutputFeatures, true);
    const vector<float> rawWeightGradient = roundToDataType(
        fullyConnectedWeightGradReference(
            effectiveInputs, effectiveMatrixGradient, batchSize, numInputFeatures, numOutputFeatures),
        DataType::BF16);
    const vector<float> rawBiasGradient =
        fullyConnectedBiasGradReference(errorInputValues, batchSize, numOutputFeatures);

    const float scale = 1.0f / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    const float alphaT = alpha * std::sqrt(1.0f - beta2) / (1.0f - beta1);
    vector<float> expectedWeightsM(rawWeightGradient.size());
    vector<float> expectedWeightsV(rawWeightGradient.size());
    vector<float> expectedWeightsAfter(rawWeightGradient.size());
    for (uint64_t i = 0; i < rawWeightGradient.size(); ++i) {
        const float g = rawWeightGradient[i] * scale;
        expectedWeightsM[i] = (1.0f - beta1) * g;
        expectedWeightsV[i] = (1.0f - beta2) * g * g;
        expectedWeightsAfter[i] = roundToDataType(
            effectiveWeights[i] - alphaT * expectedWeightsM[i] / (std::sqrt(expectedWeightsV[i]) + epsilon),
            DataType::BF16);
    }
    vector<float> expectedBiasesM(rawBiasGradient.size());
    vector<float> expectedBiasesV(rawBiasGradient.size());
    vector<float> expectedBiasesAfter(rawBiasGradient.size());
    for (uint64_t i = 0; i < rawBiasGradient.size(); ++i) {
        const float g = rawBiasGradient[i] * scale;
        expectedBiasesM[i] = (1.0f - beta1) * g;
        expectedBiasesV[i] = (1.0f - beta2) * g * g;
        expectedBiasesAfter[i] =
            initialBiasValues[i] - alphaT * expectedBiasesM[i] / (std::sqrt(expectedBiasesV[i]) + epsilon);
    }

    expectAllClose(actualForward, expectedForward, 4e-2f, 4e-2f, "bf16/fp32 Adam feature out");
    expectAllClose(readCpuTensor(weightsMHost), expectedWeightsM, 4e-3f, 4e-3f, "bf16/fp32 Adam weights m");
    expectAllClose(readCpuTensor(weightsVHost), expectedWeightsV, 4e-3f, 4e-3f, "bf16/fp32 Adam weights v");
    expectAllClose(readCpuTensor(weightsAfterHost), expectedWeightsAfter, 4e-2f, 4e-2f,
                   "bf16/fp32 Adam weights after");
    expectAllClose(readCpuTensor(biasesMHost), expectedBiasesM, 2e-4f, 2e-4f, "bf16/fp32 Adam biases m");
    expectAllClose(readCpuTensor(biasesVHost), expectedBiasesV, 2e-4f, 2e-4f, "bf16/fp32 Adam biases v");
    expectAllClose(readCpuTensor(biasesAfterHost), expectedBiasesAfter, 2e-4f, 2e-4f,
                   "bf16/fp32 Adam biases after");

    EXPECT_FALSE(physicalWeightsAdam->getWeightsGradient().has_value());
    EXPECT_FALSE(physicalBiasesAdam->getWeightsGradient().has_value());
}

void runFullyConnectedAdamThreePasses(bool hasBias) {
    constexpr uint32_t batchSize = 3;
    constexpr uint32_t numInputFeatures = 4;
    constexpr uint32_t numOutputFeatures = 3;

    constexpr float alpha = 0.001f;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    constexpr float epsilon = 1e-7f;

    const DataType dataType = DataType::FP32;

    const vector<float> initialWeightValues = {
        0.25f,
        -0.50f,
        0.75f,
        1.00f,
        -0.25f,
        0.50f,
        -0.75f,
        0.30f,
        -0.60f,
        0.40f,
        0.90f,
        -0.20f,
    };

    const vector<float> initialBiasValues = hasBias ? vector<float>{0.10f, -0.20f, 0.30f} : vector<float>{};

    const vector<vector<float>> inputValuesByPass = {
        {
            1.00f,
            -2.00f,
            0.50f,
            0.25f,
            3.00f,
            -1.50f,
            2.00f,
            -0.75f,
            -0.25f,
            1.25f,
            -3.00f,
            0.50f,
        },
        {
            -1.00f,
            0.75f,
            2.25f,
            -0.50f,
            0.50f,
            -2.50f,
            1.00f,
            1.50f,
            2.00f,
            0.25f,
            -1.25f,
            -0.75f,
        },
        {
            1.50f,
            0.50f,
            -0.25f,
            2.00f,
            -2.00f,
            1.75f,
            0.75f,
            -1.50f,
            0.25f,
            -0.50f,
            1.25f,
            3.00f,
        },
    };

    const vector<vector<float>> errorInputValuesByPass = {
        {
            0.50f,
            -1.00f,
            1.50f,
            -0.25f,
            0.75f,
            -1.25f,
            1.00f,
            0.25f,
            -0.50f,
        },
        {
            -1.50f,
            0.50f,
            0.25f,
            1.25f,
            -0.75f,
            1.00f,
            0.50f,
            1.50f,
            -1.00f,
        },
        {
            0.75f,
            1.25f,
            -0.25f,
            -1.00f,
            0.50f,
            1.75f,
            1.50f,
            -1.25f,
            0.25f,
        },
    };

    shared_ptr<Api::Adam> weightsAdam = Api::Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build();
    shared_ptr<Api::Adam> biasesAdam =
        hasBias ? Api::Adam::Builder().alpha(alpha).beta1(beta1).beta2(beta2).epsilon(epsilon).build() : nullptr;

    Api::Network network(hasBias ? "fullyConnectedAdamThreePassesWithBias" : "fullyConnectedAdamThreePassesWithoutBias");

    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(dataType).build();

    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();

    Api::FullyConnected::Builder fcBuilder;
    fcBuilder.network(network)
        .featureInput(inputRivet.getFeatureOutput().value())
        .numOutputFeatures(numOutputFeatures)
        .hasBias(hasBias)
        .weightsDataType(dataType)
        .computeDataType(DataType::FP32)
        .outputDataType(dataType)
        .weightsOptimizer(weightsAdam)
        .noActivation();

    if (hasBias)
        fcBuilder.biasesOptimizer(biasesAdam);

    Api::FullyConnected fc = fcBuilder.build();

    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput().value()).build();

    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    PlacedFullyConnectedFixture fixture = placeSingleFullyConnectedNetwork(network, input, output, fc, batchSize, false);

    ASSERT_NE(fixture.physicalFc->getParameter("weights"), nullptr);
    ASSERT_NE(fixture.physicalFc->getParameter("weights")->getOptimizer(), nullptr);
    ASSERT_TRUE(fixture.physicalFc->getParameter("weights")->getStorage().has_value());

    if (hasBias) {
        ASSERT_NE(fixture.physicalFc->getParameter("biases"), nullptr);
        ASSERT_NE(fixture.physicalFc->getParameter("biases")->getOptimizer(), nullptr);
        ASSERT_TRUE(fixture.physicalFc->getParameter("biases")->getStorage().has_value());
    }

    Stream stream = fixture.physicalFc->getStreams()[0];
    setParameterTensor(fixture.physicalFc->getParameter("weights"), initialWeightValues, stream);
    if (hasBias)
        setParameterTensor(fixture.physicalFc->getParameter("biases"), initialBiasValues, stream);
    stream.synchronize();

    const float lossScalingFactor = Impl::Loss::getLossScalingFactor();
    const vector<FullyConnectedAdamPassReference> references = computeFullyConnectedAdamReferenceSequence(inputValuesByPass,
                                                                                                          errorInputValuesByPass,
                                                                                                          initialWeightValues,
                                                                                                          initialBiasValues,
                                                                                                          batchSize,
                                                                                                          numInputFeatures,
                                                                                                          numOutputFeatures,
                                                                                                          hasBias,
                                                                                                          lossScalingFactor,
                                                                                                          alpha,
                                                                                                          beta1,
                                                                                                          beta2,
                                                                                                          epsilon);

    ASSERT_EQ(references.size(), inputValuesByPass.size());

    ASSERT_GT(fixture.physicalFc->getErrorInputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorInputs()[0].has_value());
    ASSERT_GT(fixture.physicalFc->getErrorOutputs().size(), 0u);
    ASSERT_TRUE(fixture.physicalFc->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(fixture.physicalFc->getGradientUpdateStream().has_value());

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numInputFeatures}));
    Impl::Tensor fcErrorInput = fixture.physicalFc->getErrorInputs()[0].value();
    Impl::Tensor fcErrorInputHost = fcErrorInput.clone(cpuPlacement);

    shared_ptr<Impl::Optimizer> physicalWeightsAdam = fixture.physicalFc->getParameter("weights")->getOptimizer();
    shared_ptr<Impl::Optimizer> physicalBiasesAdam = hasBias ? fixture.physicalFc->getParameter("biases")->getOptimizer() : nullptr;

    for (uint64_t pass = 0; pass < inputValuesByPass.size(); ++pass) {
        SCOPED_TRACE(::testing::Message() << "pass=" << pass << " hasBias=" << hasBias);

        writeCpuTensor(featureInHost, inputValuesByPass[pass]);

        const vector<float> actualFeatureOut = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);

        writeCpuTensor(fcErrorInputHost, errorInputValuesByPass[pass]);
        fcErrorInput.copyFromAsync(fcErrorInputHost, stream);
        fixture.physicalFc->backward(fcErrorInput, batchSize);

        Stream gradientStream = fixture.physicalFc->getGradientUpdateStream().value();

        Impl::Tensor errorOutputHost = copyTensorToCpu(fixture.physicalFc->getErrorOutputs()[0].value(), stream);
        EXPECT_FALSE(physicalWeightsAdam->getWeightsGradient().has_value())
            << "Fused FullyConnected Adam weights update should not allocate a dense gradient tensor.";
        Impl::Tensor weightsAfterHost = copyTensorToCpu(fixture.physicalFc->getParameter("weights")->getStorage().value(), gradientStream);
        Impl::Tensor weightsMHost = copyTensorToCpu(physicalWeightsAdam->getOptimizerParameterTensor("m"), gradientStream);
        Impl::Tensor weightsVHost = copyTensorToCpu(physicalWeightsAdam->getOptimizerParameterTensor("v"), gradientStream);

        stream.synchronize();
        gradientStream.synchronize();

        const FullyConnectedAdamPassReference& reference = references[pass];

        expectAllClose(actualFeatureOut, reference.featureOut, 2e-4f, 2e-4f, "feature out");
        expectAllClose(readCpuTensor(errorOutputHost), reference.errorOut, 2e-4f, 2e-4f, "error out");
        expectAllClose(readCpuTensor(weightsMHost), reference.weightsM, 2e-4f, 2e-4f, "weights m");
        expectAllClose(readCpuTensor(weightsVHost), reference.weightsV, 2e-4f, 2e-4f, "weights v");
        expectAllClose(readCpuTensor(weightsAfterHost), reference.weightsAfter, 2e-4f, 2e-4f, "weights after");

        if (hasBias) {
            ASSERT_NE(physicalBiasesAdam, nullptr);

            EXPECT_FALSE(physicalBiasesAdam->getWeightsGradient().has_value())
                << "Fused FullyConnected Adam biases update should not allocate a dense gradient tensor.";
            Impl::Tensor biasesAfterHost =
                copyTensorToCpu(fixture.physicalFc->getParameter("biases")->getStorage().value(), gradientStream);
            Impl::Tensor biasesMHost = copyTensorToCpu(physicalBiasesAdam->getOptimizerParameterTensor("m"), gradientStream);
            Impl::Tensor biasesVHost = copyTensorToCpu(physicalBiasesAdam->getOptimizerParameterTensor("v"), gradientStream);

            gradientStream.synchronize();

            expectAllClose(readCpuTensor(biasesMHost), reference.biasesM, 2e-4f, 2e-4f, "biases m");
            expectAllClose(readCpuTensor(biasesVHost), reference.biasesV, 2e-4f, 2e-4f, "biases v");
            expectAllClose(readCpuTensor(biasesAfterHost), reference.biasesAfter, 2e-4f, 2e-4f, "biases after");
        }
    }
}

TEST(FullyConnectedApi, AdamThreePassesForwardBackwardAndUpdateWithBias) { runFullyConnectedAdamThreePasses(true); }

TEST(FullyConnectedApi, AdamThreePassesForwardBackwardAndUpdateWithoutBias) { runFullyConnectedAdamThreePasses(false); }

TEST(FullyConnectedApi, MultiInputEpilogueRunsForwardBackwardResidualAddAndUpdatesWeights) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t numInputFeatures = 3;
    constexpr uint32_t numOutputFeatures = 2;
    constexpr float learningRate = 0.1f;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = {
        1.0f, -2.0f, 0.5f,
        -1.5f, 0.25f, 2.0f,
    };
    const vector<float> residualValues = {
        0.25f, -0.50f,
        1.25f, 0.75f,
    };
    const vector<float> upstreamErrors = {
        0.5f, -1.0f,
        1.5f, -0.25f,
    };
    const vector<float> initialWeights = {
        0.25f, -0.50f,
        0.75f, 1.00f,
        -0.25f, 0.50f,
    };

    shared_ptr<Api::Sgd> weightsSgd = Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();

    Api::Network network("fullyConnectedMultiInputEpilogueForwardBackward");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({numInputFeatures}).dataType(dataType).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({numOutputFeatures}).dataType(dataType).build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::GradientRivet residualRivet = Api::GradientRivet::Builder().network(network).tensor(residual.getFeatureOutput().value()).build();

    Impl::Expression fcOutput = Api::FullyConnected::epilogueInput(DataType::FP32, dataType);
    Impl::Expression residualInput = Api::FullyConnected::epilogueAuxInput("residual", DataType::FP32, dataType);
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(inputRivet.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .hasBias(false)
                                 .weightsDataType(dataType)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(dataType)
                                 .weightsOptimizer(weightsSgd)
                                 .noActivation()
                                 .epilogueInput("residual", residualRivet.getFeatureOutput().value())
                                 .epilogue(fcOutput + residualInput)
                                 .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placedNetwork, nullptr);
    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(input.getId()));
    auto physicalResidual = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(residual.getId()));
    auto physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalFc = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(fc.getId()));
    ASSERT_NE(physicalInput, nullptr);
    ASSERT_NE(physicalResidual, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalFc, nullptr);
    ASSERT_TRUE(physicalFc->getGradientUpdateStream().has_value());

    Stream stream = physicalFc->getStreams()[0];
    Stream gradientStream = physicalFc->getGradientUpdateStream().value();
    setParameterTensor(physicalFc->getParameter("weights"), initialWeights, stream);
    stream.synchronize();

    Impl::Tensor inputHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numInputFeatures}));
    Impl::Tensor residualHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, numOutputFeatures}));
    writeCpuTensor(inputHost, inputValues);
    writeCpuTensor(residualHost, residualValues);

    physicalInput->forward(inputHost, false, batchSize);
    physicalResidual->forward(residualHost, false, batchSize);
    Event outputReady = physicalOutput->getOutputReadyEvent();
    outputReady.synchronize();

    vector<float> expectedForward = fullyConnectedReference(
        inputValues, initialWeights, {}, batchSize, numInputFeatures, numOutputFeatures, false);
    for (uint64_t i = 0; i < expectedForward.size(); ++i) {
        expectedForward[i] += residualValues[i];
    }
    expectAllClose(readCpuTensor(physicalOutput->getFeatureOutput().value()), expectedForward, 8e-2f, 8e-2f,
                   "fully connected residual epilogue forward");

    ASSERT_EQ(physicalFc->getErrorInputs().size(), 1u);
    ASSERT_TRUE(physicalFc->getErrorInputs()[0].has_value());
    ASSERT_EQ(physicalFc->getErrorOutputs().size(), 2u)
        << "Multi-input epilogue backward must produce gradients for the primary feature input and auxiliary residual input.";
    ASSERT_TRUE(physicalFc->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(physicalFc->getErrorOutputs()[1].has_value());

    Impl::Tensor errorInput = physicalFc->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
    writeCpuTensor(errorInputHost, upstreamErrors);
    errorInput.copyFromAsync(errorInputHost, stream);
    physicalFc->backward(errorInput, batchSize);

    Impl::Tensor primaryErrorOutputHost = copyTensorToCpu(physicalFc->getErrorOutputs()[0].value(), stream);
    Impl::Tensor residualErrorOutputHost = copyTensorToCpu(physicalFc->getErrorOutputs()[1].value(), stream);
    Impl::Tensor weightsAfterHost = copyTensorToCpu(physicalFc->getParameter("weights")->getStorage().value(), gradientStream);
    stream.synchronize();
    gradientStream.synchronize();

    const vector<float> expectedPrimaryError = fullyConnectedBackwardErrorReference(
        upstreamErrors, initialWeights, batchSize, numInputFeatures, numOutputFeatures);
    const vector<float> expectedWeightsGrad = fullyConnectedWeightGradReference(
        inputValues, upstreamErrors, batchSize, numInputFeatures, numOutputFeatures);
    const vector<float> expectedWeightsAfter = sgdUpdatedReference(initialWeights, expectedWeightsGrad, batchSize, learningRate);

    expectAllClose(readCpuTensor(primaryErrorOutputHost), expectedPrimaryError, 8e-2f, 8e-2f,
                   "fully connected residual epilogue primary error out");
    expectAllClose(readCpuTensor(residualErrorOutputHost), upstreamErrors, 8e-2f, 8e-2f,
                   "fully connected residual epilogue auxiliary residual error out");
    expectAllClose(readCpuTensor(weightsAfterHost), expectedWeightsAfter, 8e-2f, 8e-2f,
                   "fully connected residual epilogue weights after");
}

TEST(FullyConnectedApi, PrefixPreservingRank3ResidualEpiloguePlacesAndKeepsPublicShape) {
    constexpr uint32_t batchSize = 3;
    constexpr uint64_t sequenceLength = 100;
    constexpr uint64_t numInputFeatures = 128;
    constexpr uint64_t numOutputFeatures = 1;
    const DataType dataType = DataType::BF16;

    Api::Network network("prefixPreservingRank3ResidualEpilogue");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({sequenceLength, numInputFeatures})
                                  .dataType(dataType)
                                  .build();
    Api::NetworkInput residual = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("residual")
                                     .dimensions({sequenceLength, numOutputFeatures})
                                     .dataType(dataType)
                                     .build();

    Impl::Expression projected = Api::FullyConnected::epilogueInput(DataType::FP32, dataType);
    Impl::Expression residualInput = Api::FullyConnected::epilogueAuxInput("residual", DataType::FP32, dataType);
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(input.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .preserveInputPrefixDimensions(true)
                                 .hasBias(true)
                                 .weightsDataType(dataType)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(dataType)
                                 .noActivation()
                                 .epilogueInput("residual", residual.getFeatureOutput().value())
                                 .epilogue(projected + residualInput)
                                 .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(fc.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    ASSERT_NO_THROW(placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/true));
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placedNetwork, nullptr);

    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalFc = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(fc.getId()));
    auto physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    ASSERT_NE(physicalFc, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_TRUE(physicalOutput->getFeatureOutput().has_value());
    EXPECT_EQ(physicalOutput->getFeatureOutput()->getDimensions(),
              (vector<uint64_t>{batchSize, sequenceLength, numOutputFeatures}));
}

TEST(FullyConnectedApi, PrefixPreservingRank3ResidualEpilogueTrainingBuildsFoldedBackward) {
    constexpr uint32_t batchSize = 3;
    constexpr uint64_t sequenceLength = 53;
    constexpr uint64_t numInputFeatures = 256;
    constexpr uint64_t numOutputFeatures = 128;
    const DataType dataType = DataType::BF16;

    shared_ptr<Api::Sgd> optimizer =
        Api::Sgd::Builder().initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();

    Api::Network network("prefixPreservingRank3ResidualEpilogueTraining");
    Api::NetworkInput input = Api::NetworkInput::Builder()
                                  .network(network)
                                  .name("input")
                                  .dimensions({sequenceLength, numInputFeatures})
                                  .dataType(dataType)
                                  .build();
    Api::NetworkInput residual = Api::NetworkInput::Builder()
                                     .network(network)
                                     .name("residual")
                                     .dimensions({sequenceLength, numOutputFeatures})
                                     .dataType(dataType)
                                     .build();
    Api::GradientRivet inputRivet =
        Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::GradientRivet residualRivet =
        Api::GradientRivet::Builder().network(network).tensor(residual.getFeatureOutput().value()).build();

    Impl::Expression projected = Api::FullyConnected::epilogueInput(DataType::FP32, dataType);
    Impl::Expression residualInput = Api::FullyConnected::epilogueAuxInput("residual", DataType::FP32, dataType);
    Api::FullyConnected fc = Api::FullyConnected::Builder()
                                 .network(network)
                                 .featureInput(inputRivet.getFeatureOutput().value())
                                 .numOutputFeatures(numOutputFeatures)
                                 .preserveInputPrefixDimensions(true)
                                 .hasBias(true)
                                 .weightsDataType(dataType)
                                 .computeDataType(DataType::FP32)
                                 .outputDataType(dataType)
                                 .weightsOptimizer(optimizer)
                                 .biasesOptimizer(optimizer)
                                 .noActivation()
                                 .epilogueInput("residual", residualRivet.getFeatureOutput().value())
                                 .epilogue(projected + residualInput)
                                 .build();
    Api::GradientRivet outputRivet =
        Api::GradientRivet::Builder().network(network).tensor(fc.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    ASSERT_NO_THROW(placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false));
    synchronizeEvents(initDoneEvents);
    ASSERT_NE(placedNetwork, nullptr);

    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalFc = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(fc.getId()));
    auto physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    ASSERT_NE(physicalFc, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_TRUE(physicalOutput->getFeatureOutput().has_value());
    EXPECT_EQ(physicalOutput->getFeatureOutput()->getDimensions(),
              (vector<uint64_t>{batchSize, sequenceLength, numOutputFeatures}));

    ASSERT_EQ(physicalFc->getErrorOutputs().size(), 2u);
    ASSERT_TRUE(physicalFc->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(physicalFc->getErrorOutputs()[1].has_value());
    EXPECT_EQ(physicalFc->getErrorOutputs()[0]->getDimensions(),
              (vector<uint64_t>{batchSize, sequenceLength, numInputFeatures}));
    EXPECT_EQ(physicalFc->getErrorOutputs()[1]->getDimensions(),
              (vector<uint64_t>{batchSize, sequenceLength, numOutputFeatures}));
}

