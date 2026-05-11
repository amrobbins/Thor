#include "DeepLearning/Api/Layers/Learning/Convolution3d.h"
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
using json = nlohmann::json;

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
            FAIL() << "Unsupported tensor dtype in writeCpuTensor.";
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
            ADD_FAILURE() << "Unsupported tensor dtype in readCpuTensor.";
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
    const vector<float>& actual, const vector<float>& expected, float atol = 8e-2f, float rtol = 8e-2f, const string& what = "") {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        EXPECT_LE(diff, tol) << what << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

void setParameterTensor(const shared_ptr<Impl::PhysicalParameter>& parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());
    Impl::Tensor deviceTensor = parameter->getStorage().value();
    Impl::Tensor cpuTensor = deviceTensor.clone(cpuPlacement);
    writeCpuTensor(cpuTensor, values);
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

uint32_t convOutputDim(uint32_t input, uint32_t stride, uint32_t filter, uint32_t padding) {
    return 1 + (((input + 2 * padding) - filter) / stride);
}

uint64_t ncdhwIndex(uint64_t n, uint64_t c, uint64_t d, uint64_t h, uint64_t w, uint64_t C, uint64_t D, uint64_t H, uint64_t W) {
    return (((n * C + c) * D + d) * H + h) * W + w;
}

uint64_t kcdhwIndex(uint64_t k, uint64_t c, uint64_t z, uint64_t r, uint64_t s, uint64_t C, uint64_t Z, uint64_t R, uint64_t S) {
    return (((k * C + c) * Z + z) * R + r) * S + s;
}

uint64_t convolutionFilterIndex(
    uint64_t k, uint64_t c, uint64_t z, uint64_t r, uint64_t s, uint64_t C, uint64_t Z, uint64_t R, uint64_t S) {
    // Match cuDNN's deep-learning convention: convolution layers use cross-correlation,
    // so the filter is addressed in natural KCDHW order.
    return kcdhwIndex(k, c, z, r, s, C, Z, R, S);
}

vector<float> conv3dForwardReference(const vector<float>& input,
                                     const vector<float>& weights,
                                     const vector<float>& biases,
                                     uint64_t N,
                                     uint64_t C,
                                     uint64_t D,
                                     uint64_t H,
                                     uint64_t W,
                                     uint64_t K,
                                     uint64_t Z,
                                     uint64_t R,
                                     uint64_t S,
                                     uint64_t strideD,
                                     uint64_t strideH,
                                     uint64_t strideW,
                                     uint64_t padD,
                                     uint64_t padH,
                                     uint64_t padW,
                                     bool hasBias) {
    const uint64_t OD = convOutputDim(D, strideD, Z, padD);
    const uint64_t OH = convOutputDim(H, strideH, R, padH);
    const uint64_t OW = convOutputDim(W, strideW, S, padW);
    vector<float> output(N * K * OD * OH * OW, 0.0f);

    for (uint64_t n = 0; n < N; ++n) {
        for (uint64_t k = 0; k < K; ++k) {
            for (uint64_t od = 0; od < OD; ++od) {
                for (uint64_t oh = 0; oh < OH; ++oh) {
                    for (uint64_t ow = 0; ow < OW; ++ow) {
                        float acc = hasBias ? biases[k] : 0.0f;
                        for (uint64_t c = 0; c < C; ++c) {
                            for (uint64_t z = 0; z < Z; ++z) {
                                const int64_t id = static_cast<int64_t>(od * strideD + z) - static_cast<int64_t>(padD);
                                if (id < 0 || id >= static_cast<int64_t>(D))
                                    continue;
                                for (uint64_t r = 0; r < R; ++r) {
                                    const int64_t ih = static_cast<int64_t>(oh * strideH + r) - static_cast<int64_t>(padH);
                                    if (ih < 0 || ih >= static_cast<int64_t>(H))
                                        continue;
                                    for (uint64_t s = 0; s < S; ++s) {
                                        const int64_t iw = static_cast<int64_t>(ow * strideW + s) - static_cast<int64_t>(padW);
                                        if (iw < 0 || iw >= static_cast<int64_t>(W))
                                            continue;
                                        acc += input[ncdhwIndex(n, c, id, ih, iw, C, D, H, W)] *
                                               weights[convolutionFilterIndex(k, c, z, r, s, C, Z, R, S)];
                                    }
                                }
                            }
                        }
                        output[ncdhwIndex(n, k, od, oh, ow, K, OD, OH, OW)] = acc;
                    }
                }
            }
        }
    }
    return output;
}

vector<float> conv3dErrorReference(const vector<float>& errorInput,
                                   const vector<float>& weights,
                                   uint64_t N,
                                   uint64_t C,
                                   uint64_t D,
                                   uint64_t H,
                                   uint64_t W,
                                   uint64_t K,
                                   uint64_t Z,
                                   uint64_t R,
                                   uint64_t S,
                                   uint64_t strideD,
                                   uint64_t strideH,
                                   uint64_t strideW,
                                   uint64_t padD,
                                   uint64_t padH,
                                   uint64_t padW) {
    const uint64_t OD = convOutputDim(D, strideD, Z, padD);
    const uint64_t OH = convOutputDim(H, strideH, R, padH);
    const uint64_t OW = convOutputDim(W, strideW, S, padW);
    vector<float> errorOutput(N * C * D * H * W, 0.0f);

    for (uint64_t n = 0; n < N; ++n) {
        for (uint64_t k = 0; k < K; ++k) {
            for (uint64_t od = 0; od < OD; ++od) {
                for (uint64_t oh = 0; oh < OH; ++oh) {
                    for (uint64_t ow = 0; ow < OW; ++ow) {
                        const float dy = errorInput[ncdhwIndex(n, k, od, oh, ow, K, OD, OH, OW)];
                        for (uint64_t c = 0; c < C; ++c) {
                            for (uint64_t z = 0; z < Z; ++z) {
                                const int64_t id = static_cast<int64_t>(od * strideD + z) - static_cast<int64_t>(padD);
                                if (id < 0 || id >= static_cast<int64_t>(D))
                                    continue;
                                for (uint64_t r = 0; r < R; ++r) {
                                    const int64_t ih = static_cast<int64_t>(oh * strideH + r) - static_cast<int64_t>(padH);
                                    if (ih < 0 || ih >= static_cast<int64_t>(H))
                                        continue;
                                    for (uint64_t s = 0; s < S; ++s) {
                                        const int64_t iw = static_cast<int64_t>(ow * strideW + s) - static_cast<int64_t>(padW);
                                        if (iw < 0 || iw >= static_cast<int64_t>(W))
                                            continue;
                                        errorOutput[ncdhwIndex(n, c, id, ih, iw, C, D, H, W)] +=
                                            dy * weights[convolutionFilterIndex(k, c, z, r, s, C, Z, R, S)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return errorOutput;
}

vector<float> conv3dWeightGradReference(const vector<float>& input,
                                        const vector<float>& errorInput,
                                        uint64_t N,
                                        uint64_t C,
                                        uint64_t D,
                                        uint64_t H,
                                        uint64_t W,
                                        uint64_t K,
                                        uint64_t Z,
                                        uint64_t R,
                                        uint64_t S,
                                        uint64_t strideD,
                                        uint64_t strideH,
                                        uint64_t strideW,
                                        uint64_t padD,
                                        uint64_t padH,
                                        uint64_t padW) {
    const uint64_t OD = convOutputDim(D, strideD, Z, padD);
    const uint64_t OH = convOutputDim(H, strideH, R, padH);
    const uint64_t OW = convOutputDim(W, strideW, S, padW);
    vector<float> grad(K * C * Z * R * S, 0.0f);

    for (uint64_t k = 0; k < K; ++k) {
        for (uint64_t c = 0; c < C; ++c) {
            for (uint64_t z = 0; z < Z; ++z) {
                for (uint64_t r = 0; r < R; ++r) {
                    for (uint64_t s = 0; s < S; ++s) {
                        float acc = 0.0f;
                        for (uint64_t n = 0; n < N; ++n) {
                            for (uint64_t od = 0; od < OD; ++od) {
                                const int64_t id = static_cast<int64_t>(od * strideD + z) - static_cast<int64_t>(padD);
                                if (id < 0 || id >= static_cast<int64_t>(D))
                                    continue;
                                for (uint64_t oh = 0; oh < OH; ++oh) {
                                    const int64_t ih = static_cast<int64_t>(oh * strideH + r) - static_cast<int64_t>(padH);
                                    if (ih < 0 || ih >= static_cast<int64_t>(H))
                                        continue;
                                    for (uint64_t ow = 0; ow < OW; ++ow) {
                                        const int64_t iw = static_cast<int64_t>(ow * strideW + s) - static_cast<int64_t>(padW);
                                        if (iw < 0 || iw >= static_cast<int64_t>(W))
                                            continue;
                                        acc += input[ncdhwIndex(n, c, id, ih, iw, C, D, H, W)] *
                                               errorInput[ncdhwIndex(n, k, od, oh, ow, K, OD, OH, OW)];
                                    }
                                }
                            }
                        }
                        grad[kcdhwIndex(k, c, z, r, s, C, Z, R, S)] = acc;
                    }
                }
            }
        }
    }
    return grad;
}

vector<float> conv3dBiasGradReference(const vector<float>& errorInput, uint64_t N, uint64_t K, uint64_t OD, uint64_t OH, uint64_t OW) {
    vector<float> grad(K, 0.0f);
    for (uint64_t n = 0; n < N; ++n)
        for (uint64_t k = 0; k < K; ++k)
            for (uint64_t od = 0; od < OD; ++od)
                for (uint64_t oh = 0; oh < OH; ++oh)
                    for (uint64_t ow = 0; ow < OW; ++ow)
                        grad[k] += errorInput[ncdhwIndex(n, k, od, oh, ow, K, OD, OH, OW)];
    return grad;
}

vector<float> sgdUpdatedReference(const vector<float>& initial, const vector<float>& rawGradient, uint64_t batchSize, float lr) {
    const float step = lr / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    vector<float> updated(initial.size());
    for (uint64_t i = 0; i < initial.size(); ++i)
        updated[i] = initial[i] - step * rawGradient[i];
    return updated;
}

vector<float> makeDeterministicValues(uint64_t count, int seed, float scale = 0.125f) {
    vector<float> values(count);
    for (uint64_t i = 0; i < count; ++i) {
        const int centered = static_cast<int>((i * 37 + seed * 19) % 17) - 8;
        const int small = static_cast<int>((i * 11 + seed * 7) % 5) - 2;
        values[i] = static_cast<float>(centered) * scale + static_cast<float>(small) * 0.03125f;
    }
    return values;
}

vector<vector<float>> makeDeterministicPassValues(uint64_t count, int firstSeed, uint32_t numPasses) {
    vector<vector<float>> values;
    values.reserve(numPasses);
    for (uint32_t pass = 0; pass < numPasses; ++pass)
        values.push_back(makeDeterministicValues(count, firstSeed + static_cast<int>(pass)));
    return values;
}

struct PlacedConvolution3dFixture {
    shared_ptr<Api::PlacedNetwork> placedNetwork;
    Impl::StampedNetwork* stampedNetwork = nullptr;
    shared_ptr<Impl::NetworkInput> physicalInput;
    shared_ptr<Impl::NetworkOutput> physicalOutput;
    shared_ptr<Impl::CustomLayer> physicalConvolution;
};

PlacedConvolution3dFixture placeSingleConvolution3dNetwork(Api::Network& network,
                                                           const Api::NetworkInput& apiInput,
                                                           const Api::NetworkOutput& apiOutput,
                                                           const Api::Convolution3d& apiConvolution,
                                                           uint32_t batchSize,
                                                           bool inferenceOnly) {
    vector<Event> initDoneEvents;
    PlacedConvolution3dFixture fixture;
    fixture.placedNetwork = network.place(batchSize, initDoneEvents, inferenceOnly);
    synchronizeEvents(initDoneEvents);
    EXPECT_NE(fixture.placedNetwork, nullptr);
    fixture.stampedNetwork = &fixture.placedNetwork->getStampedNetwork(0);

    fixture.physicalInput =
        dynamic_pointer_cast<Impl::NetworkInput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiInput.getId()));
    fixture.physicalOutput =
        dynamic_pointer_cast<Impl::NetworkOutput>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiOutput.getId()));
    fixture.physicalConvolution =
        dynamic_pointer_cast<Impl::CustomLayer>(fixture.stampedNetwork->getPhysicalLayerFromApiLayer(apiConvolution.getId()));

    EXPECT_NE(fixture.physicalInput, nullptr);
    EXPECT_NE(fixture.physicalOutput, nullptr);
    EXPECT_NE(fixture.physicalConvolution, nullptr);
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

}  // namespace

TEST(Convolution3dApi, DefaultsToGeluAndExplicitNoActivationShape) {
    Api::Network defaultNetwork("conv3dDefaults");
    Api::NetworkInput defaultInput =
        Api::NetworkInput::Builder().network(defaultNetwork).name("input").dimensions({2, 5, 6, 7}).dataType(DataType::FP16).build();
    Api::Convolution3d defaultConv = Api::Convolution3d::Builder()
                                         .network(defaultNetwork)
                                         .featureInput(defaultInput.getFeatureOutput().value())
                                         .numOutputChannels(3)
                                         .filterDepth(3)
                                         .filterHeight(3)
                                         .filterWidth(3)
                                         .depthPadding(1)
                                         .verticalPadding(1)
                                         .horizontalPadding(1)
                                         .build();

    ASSERT_TRUE(defaultConv.isInitialized());
    EXPECT_EQ(defaultConv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{3, 5, 6, 7}));
    const json defaultJson = defaultConv.architectureJson();
    ASSERT_TRUE(defaultJson.contains("activation"));
    ASSERT_FALSE(defaultJson.at("activation").is_null());
    EXPECT_EQ(defaultJson.at("activation").at("layer_type").get<string>(), "gelu");

    Api::Network explicitNetwork("conv3dExplicit");
    Api::NetworkInput explicitInput =
        Api::NetworkInput::Builder().network(explicitNetwork).name("input").dimensions({2, 5, 6, 7}).dataType(DataType::FP16).build();
    Api::Convolution3d explicitConv = Api::Convolution3d::Builder()
                                          .network(explicitNetwork)
                                          .featureInput(explicitInput.getFeatureOutput().value())
                                          .numOutputChannels(4)
                                          .filterDepth(2)
                                          .filterHeight(3)
                                          .filterWidth(2)
                                          .depthStride(2)
                                          .verticalStride(2)
                                          .horizontalStride(3)
                                          .depthPadding(0)
                                          .verticalPadding(1)
                                          .horizontalPadding(0)
                                          .hasBias(true)
                                          .noActivation()
                                          .build();

    EXPECT_EQ(explicitConv.getFeatureOutput().value().getDimensions(), (vector<uint64_t>{4, 2, 3, 2}));
    const json explicitJson = explicitConv.architectureJson();
    ASSERT_TRUE(explicitJson.contains("activation"));
    EXPECT_TRUE(explicitJson.at("activation").is_null());
    EXPECT_TRUE(explicitJson.at("has_bias").get<bool>());
}

TEST(Convolution3dApi, StampsAsPhysicalCustomLayerAllocatesParametersAndSerializesOptimizers) {
    constexpr uint32_t batchSize = 2;
    Api::Network network("conv3dStamp");

    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({8, 5, 5, 5}).dataType(DataType::FP16).build();
    shared_ptr<Api::Sgd> weightsSgd = Api::Sgd::Builder().initialLearningRate(0.01f).decay(0.0f).momentum(0.0f).build();
    shared_ptr<Api::Sgd> biasesSgd = Api::Sgd::Builder().initialLearningRate(0.02f).decay(0.0f).momentum(0.0f).build();
    Api::Convolution3d conv = Api::Convolution3d::Builder()
                                  .network(network)
                                  .featureInput(input.getFeatureOutput().value())
                                  .numOutputChannels(8)
                                  .filterDepth(2)
                                  .filterHeight(3)
                                  .filterWidth(2)
                                  .depthPadding(0)
                                  .verticalPadding(0)
                                  .horizontalPadding(0)
                                  .hasBias(true)
                                  .weightsOptimizer(weightsSgd)
                                  .biasesOptimizer(biasesSgd)
                                  .noActivation()
                                  .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(conv.getFeatureOutput().value())
                                    .dataType(DataType::FP16)
                                    .build();

    const json j = conv.architectureJson();
    ASSERT_TRUE(j.contains("parameters"));
    ASSERT_TRUE(j.at("parameters").contains("weights"));
    ASSERT_TRUE(j.at("parameters").contains("biases"));
    ASSERT_TRUE(j.at("parameters").at("weights").contains("optimizer_override"));
    ASSERT_TRUE(j.at("parameters").at("biases").contains("optimizer_override"));

    PlacedConvolution3dFixture fixture = placeSingleConvolution3dNetwork(network, input, output, conv, batchSize, false);
    ASSERT_EQ(fixture.stampedNetwork->getNumTrainableLayers(), 1u);
    EXPECT_EQ(fixture.physicalConvolution->getLayerType(), "CustomLayer<Convolution3d>");
    EXPECT_EQ(fixture.physicalConvolution->listParameters(), (vector<string>{"weights", "biases"}));

    Impl::Tensor weights = fixture.physicalConvolution->getParameter("weights")->getStorage().value();
    Impl::Tensor biases = fixture.physicalConvolution->getParameter("biases")->getStorage().value();
    EXPECT_EQ(weights.getDimensions(), (vector<uint64_t>{8, 8, 2, 3, 2}));
    EXPECT_EQ(biases.getDimensions(), (vector<uint64_t>{8}));
    EXPECT_EQ(weights.getDataType(), DataType::FP16);
    EXPECT_EQ(biases.getDataType(), DataType::FP16);
    EXPECT_TRUE(fixture.physicalConvolution->getParameter("weights")->hasOptimizer());
    EXPECT_TRUE(fixture.physicalConvolution->getParameter("biases")->hasOptimizer());
}

TEST(Convolution3dApi, ThreePassForwardBackwardWithSgdUpdatesWeightsAndBiases) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t C = 8;
    constexpr uint32_t D = 3;
    constexpr uint32_t H = 3;
    constexpr uint32_t W = 3;
    constexpr uint32_t K = 8;
    constexpr uint32_t Z = 2;
    constexpr uint32_t R = 2;
    constexpr uint32_t S = 2;
    constexpr uint32_t strideD = 1;
    constexpr uint32_t strideH = 1;
    constexpr uint32_t strideW = 1;
    constexpr uint32_t padD = 0;
    constexpr uint32_t padH = 0;
    constexpr uint32_t padW = 0;
    constexpr uint32_t OD = 2;
    constexpr uint32_t OH = 2;
    constexpr uint32_t OW = 2;
    constexpr float learningRate = 0.1f;
    const DataType dataType = DataType::FP16;

    vector<float> currentWeights = makeDeterministicValues(K * C * Z * R * S, 3);
    vector<float> currentBiases = makeDeterministicValues(K, 5);

    const vector<vector<float>> inputsByPass = makeDeterministicPassValues(batchSize * C * D * H * W, 11, 3);
    const vector<vector<float>> errorsByPass = makeDeterministicPassValues(batchSize * K * OD * OH * OW, 23, 3);

    shared_ptr<Api::Sgd> weightsSgd = Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();
    shared_ptr<Api::Sgd> biasesSgd = Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();

    Api::Network network("conv3dThreePasses");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({C, D, H, W}).dataType(dataType).build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::Convolution3d conv = Api::Convolution3d::Builder()
                                  .network(network)
                                  .featureInput(inputRivet.getFeatureOutput().value())
                                  .numOutputChannels(K)
                                  .filterDepth(Z)
                                  .filterHeight(R)
                                  .filterWidth(S)
                                  .depthStride(strideD)
                                  .verticalStride(strideH)
                                  .horizontalStride(strideW)
                                  .depthPadding(padD)
                                  .verticalPadding(padH)
                                  .horizontalPadding(padW)
                                  .hasBias(true)
                                  .weightsOptimizer(weightsSgd)
                                  .biasesOptimizer(biasesSgd)
                                  .noActivation()
                                  .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(conv.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    PlacedConvolution3dFixture fixture = placeSingleConvolution3dNetwork(network, input, output, conv, batchSize, false);
    ASSERT_TRUE(fixture.physicalConvolution->getGradientUpdateStream().has_value());
    Stream stream = fixture.physicalConvolution->getStreams()[0];
    Stream gradientStream = fixture.physicalConvolution->getGradientUpdateStream().value();

    setParameterTensor(fixture.physicalConvolution->getParameter("weights"), currentWeights, stream);
    setParameterTensor(fixture.physicalConvolution->getParameter("biases"), currentBiases, stream);
    stream.synchronize();

    Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, C, D, H, W}));

    for (uint32_t pass = 0; pass < inputsByPass.size(); ++pass) {
        writeCpuTensor(featureInHost, inputsByPass[pass]);
        const vector<float> actualForward = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
        const vector<float> expectedForward = conv3dForwardReference(inputsByPass[pass],
                                                                     currentWeights,
                                                                     currentBiases,
                                                                     batchSize,
                                                                     C,
                                                                     D,
                                                                     H,
                                                                     W,
                                                                     K,
                                                                     Z,
                                                                     R,
                                                                     S,
                                                                     strideD,
                                                                     strideH,
                                                                     strideW,
                                                                     padD,
                                                                     padH,
                                                                     padW,
                                                                     true);
        expectAllClose(actualForward, expectedForward, 1e-1f, 1e-1f, "pass " + to_string(pass) + " feature out");

        ASSERT_GT(fixture.physicalConvolution->getErrorInputs().size(), 0u);
        ASSERT_TRUE(fixture.physicalConvolution->getErrorInputs()[0].has_value());
        ASSERT_GT(fixture.physicalConvolution->getErrorOutputs().size(), 0u);
        ASSERT_TRUE(fixture.physicalConvolution->getErrorOutputs()[0].has_value());

        Impl::Tensor errorInput = fixture.physicalConvolution->getErrorInputs()[0].value();
        Impl::Tensor errorInputHost = errorInput.clone(cpuPlacement);
        writeCpuTensor(errorInputHost, errorsByPass[pass]);
        errorInput.copyFromAsync(errorInputHost, stream);
        fixture.physicalConvolution->backward(errorInput, batchSize);

        Impl::Tensor errorOutputHost = copyTensorToCpu(fixture.physicalConvolution->getErrorOutputs()[0].value(), stream);
        Impl::Tensor weightsGradHost = copyTensorToCpu(
            fixture.physicalConvolution->getParameter("weights")->getOptimizer()->getWeightsGradient().value(), gradientStream);
        Impl::Tensor biasesGradHost = copyTensorToCpu(
            fixture.physicalConvolution->getParameter("biases")->getOptimizer()->getWeightsGradient().value(), gradientStream);
        Impl::Tensor weightsAfterHost =
            copyTensorToCpu(fixture.physicalConvolution->getParameter("weights")->getStorage().value(), gradientStream);
        Impl::Tensor biasesAfterHost =
            copyTensorToCpu(fixture.physicalConvolution->getParameter("biases")->getStorage().value(), gradientStream);
        stream.synchronize();
        gradientStream.synchronize();

        const vector<float> expectedErrorOut = conv3dErrorReference(
            errorsByPass[pass], currentWeights, batchSize, C, D, H, W, K, Z, R, S, strideD, strideH, strideW, padD, padH, padW);
        const vector<float> expectedWeightsGrad = conv3dWeightGradReference(
            inputsByPass[pass], errorsByPass[pass], batchSize, C, D, H, W, K, Z, R, S, strideD, strideH, strideW, padD, padH, padW);
        const vector<float> expectedBiasesGrad = conv3dBiasGradReference(errorsByPass[pass], batchSize, K, OD, OH, OW);
        currentWeights = sgdUpdatedReference(currentWeights, expectedWeightsGrad, batchSize, learningRate);
        currentBiases = sgdUpdatedReference(currentBiases, expectedBiasesGrad, batchSize, learningRate);

        expectAllClose(readCpuTensor(errorOutputHost), expectedErrorOut, 1e-1f, 1e-1f, "pass " + to_string(pass) + " error out");
        expectAllClose(readCpuTensor(weightsGradHost), expectedWeightsGrad, 1e-1f, 1e-1f, "pass " + to_string(pass) + " weights grad");
        expectAllClose(readCpuTensor(biasesGradHost), expectedBiasesGrad, 1e-1f, 1e-1f, "pass " + to_string(pass) + " biases grad");
        expectAllClose(readCpuTensor(weightsAfterHost), currentWeights, 1e-1f, 1e-1f, "pass " + to_string(pass) + " weights after");
        expectAllClose(readCpuTensor(biasesAfterHost), currentBiases, 1e-1f, 1e-1f, "pass " + to_string(pass) + " biases after");
    }
}

TEST(Convolution3dApi, ArchitectureSaveLoadRoundTripPreservesConfigurationAndDeserializedLayerRunsForward) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t C = 8;
    constexpr uint32_t D = 3;
    constexpr uint32_t H = 3;
    constexpr uint32_t W = 3;
    constexpr uint32_t K = 8;
    constexpr uint32_t Z = 2;
    constexpr uint32_t R = 2;
    constexpr uint32_t S = 2;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = makeDeterministicValues(batchSize * C * D * H * W, 31);
    const vector<float> weightValues = makeDeterministicValues(K * C * Z * R * S, 37);
    const vector<float> biasValues = makeDeterministicValues(K, 41);

    const string networkName = "conv3d_arch_round_trip";
    filesystem::path archiveDir = makeUniqueTestArchiveDir(networkName);

    try {
        Api::Network network(networkName);
        Api::NetworkInput input =
            Api::NetworkInput::Builder().network(network).name("input").dimensions({C, D, H, W}).dataType(dataType).build();
        Api::Convolution3d conv = Api::Convolution3d::Builder()
                                      .network(network)
                                      .featureInput(input.getFeatureOutput().value())
                                      .numOutputChannels(K)
                                      .filterDepth(Z)
                                      .filterHeight(R)
                                      .filterWidth(S)
                                      .depthPadding(0)
                                      .verticalPadding(0)
                                      .horizontalPadding(0)
                                      .hasBias(true)
                                      .noActivation()
                                      .build();
        Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                        .network(network)
                                        .name("output")
                                        .inputTensor(conv.getFeatureOutput().value())
                                        .dataType(dataType)
                                        .build();

        network.save(archiveDir.string(), true);

        Api::Network loadedNetwork(networkName);
        loadedNetwork.load(archiveDir.string());

        shared_ptr<Api::NetworkInput> loadedInput = findOnlyLayerOfType<Api::NetworkInput>(loadedNetwork);
        shared_ptr<Api::Convolution3d> loadedConv = findOnlyLayerOfType<Api::Convolution3d>(loadedNetwork);
        shared_ptr<Api::NetworkOutput> loadedOutput = findOnlyLayerOfType<Api::NetworkOutput>(loadedNetwork);
        ASSERT_NE(loadedInput, nullptr);
        ASSERT_NE(loadedConv, nullptr);
        ASSERT_NE(loadedOutput, nullptr);

        const json j = loadedConv->architectureJson();
        EXPECT_EQ(j.at("layer_type").get<string>(), "convolution_3d");
        EXPECT_EQ(j.at("filter_depth").get<uint32_t>(), Z);
        EXPECT_EQ(j.at("filter_height").get<uint32_t>(), R);
        EXPECT_EQ(j.at("filter_width").get<uint32_t>(), S);
        EXPECT_EQ(j.at("num_output_channels").get<uint32_t>(), K);
        EXPECT_TRUE(j.at("has_bias").get<bool>());
        EXPECT_TRUE(j.at("activation").is_null());

        PlacedConvolution3dFixture fixture =
            placeSingleConvolution3dNetwork(loadedNetwork, *loadedInput, *loadedOutput, *loadedConv, batchSize, true);
        Stream stream = fixture.physicalConvolution->getStreams()[0];
        setParameterTensor(fixture.physicalConvolution->getParameter("weights"), weightValues, stream);
        setParameterTensor(fixture.physicalConvolution->getParameter("biases"), biasValues, stream);
        stream.synchronize();

        Impl::Tensor featureInHost(cpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, C, D, H, W}));
        writeCpuTensor(featureInHost, inputValues);

        const vector<float> actual = runForward(*fixture.physicalInput, *fixture.physicalOutput, featureInHost, batchSize);
        const vector<float> expected =
            conv3dForwardReference(inputValues, weightValues, biasValues, batchSize, C, D, H, W, K, Z, R, S, 1, 1, 1, 0, 0, 0, true);
        expectAllClose(actual, expected, 1e-1f, 1e-1f, "loaded conv3d output");
    } catch (...) {
        filesystem::remove_all(archiveDir);
        throw;
    }
    filesystem::remove_all(archiveDir);
}
