#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"

#include "gtest/gtest.h"

#include <cmath>
#include <memory>

using namespace Thor;
using namespace std;
using json = nlohmann::json;
namespace Api = Thor;
namespace Impl = ThorImplementation;

namespace {

Impl::TensorPlacement rmsCpuPlacement(Impl::TensorPlacement::MemDevices::CPU);

uint64_t rmsTensorNumel(const Impl::Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t dim : tensor.getDimensions()) {
        numel *= dim;
    }
    return numel;
}

void rmsSynchronizeEvents(vector<Event>& events) {
    for (Event& event : events) {
        event.synchronize();
    }
    events.clear();
}

void rmsWriteCpuTensor(Impl::Tensor& tensor, const vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), rmsCpuPlacement);
    ASSERT_EQ(rmsTensorNumel(tensor), values.size());
    ASSERT_EQ(tensor.getDataType(), Impl::DataType::FP32);
    auto* ptr = static_cast<float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i) {
        ptr[i] = values[i];
    }
}

vector<float> rmsReadCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), rmsCpuPlacement);
    EXPECT_EQ(tensor.getDataType(), Impl::DataType::FP32);
    vector<float> values(rmsTensorNumel(tensor));
    const auto* ptr = static_cast<const float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i) {
        values[i] = ptr[i];
    }
    return values;
}

Impl::Tensor rmsCopyTensorToCpu(const Impl::Tensor& tensor, Stream& stream) {
    Impl::Tensor cpuTensor = tensor.clone(rmsCpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return cpuTensor;
}

void rmsExpectAllClose(const vector<float>& actual,
                       const vector<float>& expected,
                       float atol = 2e-4f,
                       float rtol = 2e-4f,
                       const string& what = "") {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        EXPECT_LE(diff, tol) << what << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

void rmsSetParameterTensor(const shared_ptr<Impl::PhysicalParameter>& parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().has_value());
    Impl::Tensor deviceTensor = parameter->getStorage().value();
    Impl::Tensor cpuTensor = deviceTensor.clone(rmsCpuPlacement);
    rmsWriteCpuTensor(cpuTensor, values);
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

vector<float> rmsNormForwardReference(const vector<float>& input,
                                      const vector<float>& weights,
                                      const vector<float>& residual,
                                      uint64_t batchSize,
                                      uint64_t hidden,
                                      float epsilon) {
    vector<float> output(batchSize * hidden, 0.0f);
    for (uint64_t b = 0; b < batchSize; ++b) {
        float meanSquare = 0.0f;
        for (uint64_t h = 0; h < hidden; ++h) {
            const float x = input[b * hidden + h];
            meanSquare += x * x;
        }
        meanSquare /= static_cast<float>(hidden);
        const float invRms = 1.0f / sqrtf(meanSquare + epsilon);
        for (uint64_t h = 0; h < hidden; ++h) {
            output[b * hidden + h] = input[b * hidden + h] * invRms * weights[h] + residual[b * hidden + h];
        }
    }
    return output;
}

vector<float> rmsNormInputGradientReference(const vector<float>& input,
                                            const vector<float>& weights,
                                            const vector<float>& upstream,
                                            uint64_t batchSize,
                                            uint64_t hidden,
                                            float epsilon) {
    vector<float> dx(batchSize * hidden, 0.0f);
    for (uint64_t b = 0; b < batchSize; ++b) {
        float meanSquare = 0.0f;
        for (uint64_t h = 0; h < hidden; ++h) {
            const float x = input[b * hidden + h];
            meanSquare += x * x;
        }
        meanSquare /= static_cast<float>(hidden);
        const float invRms = 1.0f / sqrtf(meanSquare + epsilon);
        const float invRmsCubed = invRms * invRms * invRms;
        float dot = 0.0f;
        for (uint64_t h = 0; h < hidden; ++h) {
            dot += upstream[b * hidden + h] * weights[h] * input[b * hidden + h];
        }
        for (uint64_t h = 0; h < hidden; ++h) {
            const float direct = upstream[b * hidden + h] * weights[h] * invRms;
            const float correction = input[b * hidden + h] * dot * invRmsCubed / static_cast<float>(hidden);
            dx[b * hidden + h] = direct - correction;
        }
    }
    return dx;
}

vector<float> rmsNormWeightGradientReference(const vector<float>& input,
                                             const vector<float>& upstream,
                                             uint64_t batchSize,
                                             uint64_t hidden,
                                             float epsilon) {
    vector<float> grad(hidden, 0.0f);
    for (uint64_t b = 0; b < batchSize; ++b) {
        float meanSquare = 0.0f;
        for (uint64_t h = 0; h < hidden; ++h) {
            const float x = input[b * hidden + h];
            meanSquare += x * x;
        }
        meanSquare /= static_cast<float>(hidden);
        const float invRms = 1.0f / sqrtf(meanSquare + epsilon);
        for (uint64_t h = 0; h < hidden; ++h) {
            grad[h] += upstream[b * hidden + h] * input[b * hidden + h] * invRms;
        }
    }
    return grad;
}

vector<float> rmsSgdUpdatedReference(const vector<float>& initial, const vector<float>& rawGradient, uint64_t batchSize, float lr) {
    const float step = lr / (static_cast<float>(batchSize) * Impl::Loss::getLossScalingFactor());
    vector<float> updated(initial.size());
    for (uint64_t i = 0; i < initial.size(); ++i) {
        updated[i] = initial[i] - step * rawGradient[i];
    }
    return updated;
}

}  // namespace

TEST(UtilityApiLayers, RMSNormDefaultsToLastFeatureDimension) {
    Network network("rms_norm_default_shape");
    Tensor input(DataType::FP16, {4, 8, 16});

    RMSNorm layer = RMSNorm::Builder().network(network).featureInput(input).build();

    ASSERT_TRUE(layer.isInitialized());
    ASSERT_EQ(layer.getNormalizedShape(), vector<uint64_t>({16}));
    ASSERT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-5);
    ASSERT_EQ(layer.getParameterDataType(), DataType::FP32);

    optional<Tensor> output = layer.getFeatureOutput();
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(output.value().getDimensions(), input.getDimensions());
    EXPECT_EQ(output.value().getDataType(), input.getDataType());
}

TEST(UtilityApiLayers, RMSNormAcceptsExplicitTrailingNormalizedShape) {
    Network network("rms_norm_explicit_shape");
    Tensor input(DataType::BF16, {2, 3, 4});

    RMSNorm layer = RMSNorm::Builder().network(network).featureInput(input).normalizedShape({3, 4}).epsilon(1.0e-4).build();

    EXPECT_EQ(layer.getNormalizedShape(), vector<uint64_t>({3, 4}));
    EXPECT_DOUBLE_EQ(layer.getEpsilon(), 1.0e-4);
    EXPECT_EQ(layer.getFeatureOutput().value().getDimensions(), input.getDimensions());
}

TEST(UtilityApiLayers, RMSNormRejectsBadNormalizedShape) {
    Network network("rms_norm_bad_shape");
    Tensor input(DataType::FP16, {2, 3, 4});

    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(input).normalizedShape({4, 3}).build(), std::invalid_argument);
    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(input).normalizedShape({0}).build(), std::invalid_argument);
    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(input).normalizedShape({}).build(), std::invalid_argument);
}

TEST(UtilityApiLayers, RMSNormRejectsUnsupportedDtypes) {
    Network network("rms_norm_bad_dtype");
    Tensor intInput(DataType::INT32, {2, 4});
    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(intInput).build(), std::invalid_argument);

    Tensor fpInput(DataType::FP16, {2, 4});
    EXPECT_THROW(RMSNorm::Builder().network(network).featureInput(fpInput).parameterDataType(DataType::FP16).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, RMSNormArchitectureJsonContainsWeightsOnlyAndNullableEpilogue) {
    Network network("rms_norm_architecture");
    Tensor input(DataType::FP32, {8, 32});

    RMSNorm layer = RMSNorm::Builder().network(network).featureInput(input).normalizedShape({32}).build();
    json arch = layer.architectureJson();

    EXPECT_EQ(arch.at("layer_type").get<string>(), "rms_norm");
    EXPECT_EQ(arch.at("normalized_shape").get<vector<uint64_t>>(), vector<uint64_t>({32}));
    EXPECT_TRUE(arch.at("parameters").contains("weights"));
    EXPECT_FALSE(arch.at("parameters").contains("biases"));
    ASSERT_TRUE(arch.contains("epilogue"));
    EXPECT_TRUE(arch.at("epilogue").is_null());
    EXPECT_FALSE(arch.contains("fused_activation"));
    EXPECT_FALSE(arch.contains("rht_amax"));
    EXPECT_FALSE(arch.contains("amax_output"));
}

TEST(UtilityApiLayers, RMSNormAcceptsSwishEpilogueAndSerializesExpression) {
    Network network("rms_norm_swish_epilogue");
    Tensor input(DataType::BF16, {8, 32});
    Swish swish;

    RMSNorm layer = RMSNorm::Builder()
                        .network(network)
                        .featureInput(input)
                        .normalizedShape({32})
                        .epilogue(swish.toExpression(RMSNorm::epilogueInput()))
                        .build();

    EXPECT_EQ(layer.getParameterDataType(), DataType::FP32);
    json arch = layer.architectureJson();
    ASSERT_TRUE(arch.contains("epilogue"));
    EXPECT_FALSE(arch.at("epilogue").is_null());
    EXPECT_EQ(layer.getFeatureOutput().value().getDimensions(), input.getDimensions());
}

TEST(UtilityApiLayers, RMSNormAcceptsBf16WeightsOnlyForSwishEpilogueFusionCandidate) {
    Network network("rms_norm_swish_epilogue_bf16_weights");
    Tensor input(DataType::BF16, {8, 32});
    Swish swish;

    RMSNorm layer = RMSNorm::Builder()
                        .network(network)
                        .featureInput(input)
                        .normalizedShape({32})
                        .parameterDataType(DataType::BF16)
                        .epilogue(swish)
                        .build();

    EXPECT_EQ(layer.getParameterDataType(), DataType::BF16);

    Network badNetwork("rms_norm_bf16_weights_without_swish_epilogue");
    EXPECT_THROW(RMSNorm::Builder().network(badNetwork).featureInput(input).parameterDataType(DataType::BF16).build(),
                 std::invalid_argument);

    Network badInputNetwork("rms_norm_swish_bf16_weights_bad_input");
    Tensor fp16Input(DataType::FP16, {8, 32});
    EXPECT_THROW(RMSNorm::Builder()
                     .network(badInputNetwork)
                     .featureInput(fp16Input)
                     .parameterDataType(DataType::BF16)
                     .epilogue(swish)
                     .build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, RMSNormMultiInputEpilogueRunsForwardBackwardResidualAddAndUpdatesWeights) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t hidden = 3;
    constexpr float epsilon = 1.0e-5f;
    constexpr float learningRate = 0.1f;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValues = {
        1.0f, -2.0f, 0.5f,
        -1.5f, 0.25f, 2.0f,
    };
    const vector<float> residualValues = {
        0.25f, -0.50f, 0.75f,
        1.25f, 0.75f, -1.0f,
    };
    const vector<float> upstreamErrors = {
        0.5f, -1.0f, 1.5f,
        -0.25f, 0.75f, -1.25f,
    };
    const vector<float> initialWeights = {1.0f, 0.5f, -0.25f};

    shared_ptr<Api::Sgd> weightsSgd = Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();

    Api::Network network("rmsNormMultiInputEpilogueForwardBackward");
    Api::NetworkInput input =
        Api::NetworkInput::Builder().network(network).name("input").dimensions({hidden}).dataType(dataType).build();
    Api::NetworkInput residual =
        Api::NetworkInput::Builder().network(network).name("residual").dimensions({hidden}).dataType(dataType).build();
    Api::GradientRivet inputRivet = Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
    Api::GradientRivet residualRivet = Api::GradientRivet::Builder().network(network).tensor(residual.getFeatureOutput().value()).build();

    Impl::Expression rmsOutput = Api::RMSNorm::epilogueInput(Impl::DataType::FP32, Impl::DataType::FP32);
    Impl::Expression residualInput = Api::RMSNorm::epilogueAuxInput("residual", Impl::DataType::FP32, Impl::DataType::FP32);
    Api::RMSNorm rmsNorm = Api::RMSNorm::Builder()
                               .network(network)
                               .featureInput(inputRivet.getFeatureOutput().value())
                               .normalizedShape({hidden})
                               .epsilon(epsilon)
                               .parameterDataType(dataType)
                               .weightsOptimizer(weightsSgd)
                               .epilogueInput("residual", residualRivet.getFeatureOutput().value())
                               .epilogue(rmsOutput + residualInput)
                               .build();
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder().network(network).tensor(rmsNorm.getFeatureOutput().value()).build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(batchSize, initDoneEvents, /*inferenceOnly=*/false);
    rmsSynchronizeEvents(initDoneEvents);
    ASSERT_NE(placedNetwork, nullptr);
    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalInput = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(input.getId()));
    auto physicalResidual = dynamic_pointer_cast<Impl::NetworkInput>(stampedNetwork.getPhysicalLayerFromApiLayer(residual.getId()));
    auto physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalRmsNorm = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(rmsNorm.getId()));
    ASSERT_NE(physicalInput, nullptr);
    ASSERT_NE(physicalResidual, nullptr);
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalRmsNorm, nullptr);
    ASSERT_TRUE(physicalRmsNorm->getGradientUpdateStream().has_value());

    Stream stream = physicalRmsNorm->getStreams()[0];
    Stream gradientStream = physicalRmsNorm->getGradientUpdateStream().value();
    rmsSetParameterTensor(physicalRmsNorm->getParameter("weights"), initialWeights, stream);
    stream.synchronize();

    Impl::Tensor inputHost(rmsCpuPlacement, Impl::TensorDescriptor(Impl::DataType::FP32, {batchSize, hidden}));
    Impl::Tensor residualHost(rmsCpuPlacement, Impl::TensorDescriptor(Impl::DataType::FP32, {batchSize, hidden}));
    rmsWriteCpuTensor(inputHost, inputValues);
    rmsWriteCpuTensor(residualHost, residualValues);

    physicalInput->forward(inputHost, false, batchSize);
    physicalResidual->forward(residualHost, false, batchSize);
    Event outputReady = physicalOutput->getOutputReadyEvent();
    outputReady.synchronize();

    const vector<float> expectedForward = rmsNormForwardReference(inputValues, initialWeights, residualValues, batchSize, hidden, epsilon);
    rmsExpectAllClose(rmsReadCpuTensor(physicalOutput->getFeatureOutput().value()), expectedForward, 3e-4f, 3e-4f,
                      "rmsnorm residual epilogue forward");

    ASSERT_EQ(physicalRmsNorm->getErrorInputs().size(), 1u);
    ASSERT_TRUE(physicalRmsNorm->getErrorInputs()[0].has_value());
    ASSERT_EQ(physicalRmsNorm->getErrorOutputs().size(), 2u)
        << "Multi-input epilogue backward must produce gradients for the primary RMSNorm input and auxiliary residual input.";
    ASSERT_TRUE(physicalRmsNorm->getErrorOutputs()[0].has_value());
    ASSERT_TRUE(physicalRmsNorm->getErrorOutputs()[1].has_value());

    Impl::Tensor errorInput = physicalRmsNorm->getErrorInputs()[0].value();
    Impl::Tensor errorInputHost = errorInput.clone(rmsCpuPlacement);
    rmsWriteCpuTensor(errorInputHost, upstreamErrors);
    errorInput.copyFromAsync(errorInputHost, stream);
    physicalRmsNorm->backward(errorInput, batchSize);

    Impl::Tensor primaryErrorOutputHost = rmsCopyTensorToCpu(physicalRmsNorm->getErrorOutputs()[0].value(), stream);
    Impl::Tensor residualErrorOutputHost = rmsCopyTensorToCpu(physicalRmsNorm->getErrorOutputs()[1].value(), stream);
    Impl::Tensor weightsAfterHost = rmsCopyTensorToCpu(physicalRmsNorm->getParameter("weights")->getStorage().value(), gradientStream);
    stream.synchronize();
    gradientStream.synchronize();

    const vector<float> expectedPrimaryError =
        rmsNormInputGradientReference(inputValues, initialWeights, upstreamErrors, batchSize, hidden, epsilon);
    const vector<float> expectedWeightsGrad = rmsNormWeightGradientReference(inputValues, upstreamErrors, batchSize, hidden, epsilon);
    const vector<float> expectedWeightsAfter = rmsSgdUpdatedReference(initialWeights, expectedWeightsGrad, batchSize, learningRate);

    rmsExpectAllClose(rmsReadCpuTensor(primaryErrorOutputHost), expectedPrimaryError, 3e-4f, 3e-4f,
                      "rmsnorm residual epilogue primary error out");
    rmsExpectAllClose(rmsReadCpuTensor(residualErrorOutputHost), upstreamErrors, 3e-4f, 3e-4f,
                      "rmsnorm residual epilogue auxiliary residual error out");
    rmsExpectAllClose(rmsReadCpuTensor(weightsAfterHost), expectedWeightsAfter, 3e-4f, 3e-4f,
                      "rmsnorm residual epilogue weights after");
}
