#include "DeepLearning/Api/Layers/Activations/Swish.h"
#include "DeepLearning/Api/Layers/Utility/RMSNorm.h"
#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/RaggedNetworkOutput.h"
#include "DeepLearning/Api/Layers/Utility/Stub.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Network/PlacedNetwork.h"
#include "DeepLearning/Api/Optimizers/Sgd.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/RaggedRMSNorm.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"
#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"
#include "test/DeepLearning/Api/Helpers/GradientRivet.h"
#include "test/DeepLearning/RaggedTestUtils.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"

#include "gtest/gtest.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <memory>
#include <limits>

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
    switch (tensor.getDataType()) {
        case Impl::DataType::FP32: {
            auto* ptr = static_cast<float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) ptr[i] = values[i];
            return;
        }
        case Impl::DataType::FP16: {
            auto* ptr = static_cast<__half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) ptr[i] = __float2half_rn(values[i]);
            return;
        }
        case Impl::DataType::BF16: {
            auto* ptr = static_cast<__nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) ptr[i] = __float2bfloat16_rn(values[i]);
            return;
        }
        default:
            FAIL() << "Unsupported RMSNorm test tensor dtype for CPU write.";
    }
}

vector<float> rmsReadCpuTensor(const Impl::Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), rmsCpuPlacement);
    vector<float> values(rmsTensorNumel(tensor));
    switch (tensor.getDataType()) {
        case Impl::DataType::FP32: {
            const auto* ptr = static_cast<const float*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) values[i] = ptr[i];
            break;
        }
        case Impl::DataType::FP16: {
            const auto* ptr = static_cast<const __half*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) values[i] = __half2float(ptr[i]);
            break;
        }
        case Impl::DataType::BF16: {
            const auto* ptr = static_cast<const __nv_bfloat16*>(tensor.getMemPtr());
            for (uint64_t i = 0; i < values.size(); ++i) values[i] = __bfloat162float(ptr[i]);
            break;
        }
        default:
            ADD_FAILURE() << "Unsupported RMSNorm test tensor dtype for CPU read.";
            return {};
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

    // Run a second, different batch through the same stamped training plan. The
    // saved cuDNN invVariance must be refreshed by this forward pass rather than
    // reusing the statistic retained for the first backward pass.
    const vector<float> secondInputValues = {
        -0.75f, 1.25f, 2.5f,
        3.0f, -1.0f, 0.5f,
    };
    const vector<float> secondResidualValues = {
        -0.5f, 0.25f, 1.0f,
        0.75f, -1.25f, 0.5f,
    };
    const vector<float> secondUpstreamErrors = {
        -1.0f, 0.25f, 0.75f,
        1.25f, -0.5f, 0.125f,
    };

    rmsWriteCpuTensor(inputHost, secondInputValues);
    rmsWriteCpuTensor(residualHost, secondResidualValues);
    physicalInput->forward(inputHost, false, batchSize);
    physicalResidual->forward(residualHost, false, batchSize);
    physicalOutput->getOutputReadyEvent().synchronize();

    const vector<float> expectedSecondForward =
        rmsNormForwardReference(secondInputValues, expectedWeightsAfter, secondResidualValues, batchSize, hidden, epsilon);
    rmsExpectAllClose(rmsReadCpuTensor(physicalOutput->getFeatureOutput().value()), expectedSecondForward, 3e-4f, 3e-4f,
                      "rmsnorm second forward");

    rmsWriteCpuTensor(errorInputHost, secondUpstreamErrors);
    errorInput.copyFromAsync(errorInputHost, stream);
    physicalRmsNorm->backward(errorInput, batchSize);

    Impl::Tensor secondPrimaryErrorOutputHost = rmsCopyTensorToCpu(physicalRmsNorm->getErrorOutputs()[0].value(), stream);
    Impl::Tensor secondWeightsAfterHost = rmsCopyTensorToCpu(physicalRmsNorm->getParameter("weights")->getStorage().value(), gradientStream);
    stream.synchronize();
    gradientStream.synchronize();

    const vector<float> expectedSecondPrimaryError =
        rmsNormInputGradientReference(secondInputValues, expectedWeightsAfter, secondUpstreamErrors, batchSize, hidden, epsilon);
    const vector<float> expectedSecondWeightsGrad =
        rmsNormWeightGradientReference(secondInputValues, secondUpstreamErrors, batchSize, hidden, epsilon);
    const vector<float> expectedSecondWeightsAfter =
        rmsSgdUpdatedReference(expectedWeightsAfter, expectedSecondWeightsGrad, batchSize, learningRate);

    rmsExpectAllClose(rmsReadCpuTensor(secondPrimaryErrorOutputHost), expectedSecondPrimaryError, 3e-4f, 3e-4f,
                      "rmsnorm second primary error out");
    rmsExpectAllClose(rmsReadCpuTensor(secondWeightsAfterHost), expectedSecondWeightsAfter, 3e-4f, 3e-4f,
                      "rmsnorm second weights after");
}

TEST(UtilityApiLayers, RMSNormDenseFp16AndBf16BackwardMatchReferenceAndUpdateFp32Weights) {
    constexpr uint32_t batchSize = 2;
    constexpr uint32_t hidden = 4;
    constexpr float epsilon = 1.0e-5f;
    constexpr float learningRate = 0.05f;
    const vector<float> inputValues = {
        1.0f, -2.25f, 0.5f, 3.0f,
        -1.5f, 0.25f, 2.0f, -0.75f,
    };
    const vector<float> upstreamErrors = {
        0.5f, -1.0f, 1.5f, -0.25f,
        0.75f, -1.25f, 0.375f, 1.0f,
    };
    const vector<float> initialWeights = {1.0f, 0.5f, -0.25f, 1.5f};
    const vector<float> zeroResidual(batchSize * hidden, 0.0f);

    for (const DataType dataType : {DataType::FP16, DataType::BF16}) {
        SCOPED_TRACE("dtype=" + std::to_string(static_cast<int>(dataType)));
        const float activationTolerance = dataType == DataType::FP16 ? 3.0e-3f : 1.5e-2f;
        const float weightsTolerance = dataType == DataType::FP16 ? 1.5e-3f : 6.0e-3f;

        shared_ptr<Api::Sgd> weightsSgd =
            Api::Sgd::Builder().initialLearningRate(learningRate).decay(0.0f).momentum(0.0f).build();
        Api::Network network("rms_norm_dense_low_precision_backward_" + std::to_string(static_cast<int>(dataType)));
        Api::NetworkInput input =
            Api::NetworkInput::Builder().network(network).name("input").dimensions({hidden}).dataType(dataType).build();
        Api::GradientRivet inputRivet =
            Api::GradientRivet::Builder().network(network).tensor(input.getFeatureOutput().value()).build();
        Api::RMSNorm rmsNorm = Api::RMSNorm::Builder()
                                   .network(network)
                                   .featureInput(inputRivet.getFeatureOutput().value())
                                   .normalizedShape({hidden})
                                   .epsilon(epsilon)
                                   .parameterDataType(DataType::FP32)
                                   .weightsOptimizer(weightsSgd)
                                   .build();
        Api::GradientRivet outputRivet =
            Api::GradientRivet::Builder().network(network).tensor(rmsNorm.getFeatureOutput().value()).build();
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
        auto physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
        auto physicalRmsNorm = dynamic_pointer_cast<Impl::CustomLayer>(stampedNetwork.getPhysicalLayerFromApiLayer(rmsNorm.getId()));
        ASSERT_NE(physicalInput, nullptr);
        ASSERT_NE(physicalOutput, nullptr);
        ASSERT_NE(physicalRmsNorm, nullptr);
        ASSERT_TRUE(physicalRmsNorm->getGradientUpdateStream().has_value());

        Stream stream = physicalRmsNorm->getStreams()[0];
        Stream gradientStream = physicalRmsNorm->getGradientUpdateStream().value();
        rmsSetParameterTensor(physicalRmsNorm->getParameter("weights"), initialWeights, stream);
        stream.synchronize();

        Impl::Tensor inputHost(rmsCpuPlacement, Impl::TensorDescriptor(dataType, {batchSize, hidden}));
        rmsWriteCpuTensor(inputHost, inputValues);
        const vector<float> quantizedInput = rmsReadCpuTensor(inputHost);
        physicalInput->forward(inputHost, false, batchSize);
        physicalOutput->getOutputReadyEvent().synchronize();

        const vector<float> expectedForward =
            rmsNormForwardReference(quantizedInput, initialWeights, zeroResidual, batchSize, hidden, epsilon);
        rmsExpectAllClose(rmsReadCpuTensor(physicalOutput->getFeatureOutput().value()),
                          expectedForward,
                          activationTolerance,
                          activationTolerance,
                          "low-precision RMSNorm forward");

        ASSERT_EQ(physicalRmsNorm->getErrorInputs().size(), 1u);
        ASSERT_TRUE(physicalRmsNorm->getErrorInputs()[0].has_value());
        ASSERT_EQ(physicalRmsNorm->getErrorOutputs().size(), 1u);
        ASSERT_TRUE(physicalRmsNorm->getErrorOutputs()[0].has_value());

        Impl::Tensor errorInput = physicalRmsNorm->getErrorInputs()[0].value();
        Impl::Tensor errorInputHost = errorInput.clone(rmsCpuPlacement);
        rmsWriteCpuTensor(errorInputHost, upstreamErrors);
        const vector<float> quantizedUpstream = rmsReadCpuTensor(errorInputHost);
        errorInput.copyFromAsync(errorInputHost, stream);
        physicalRmsNorm->backward(errorInput, batchSize);

        Impl::Tensor inputGradientHost = rmsCopyTensorToCpu(physicalRmsNorm->getErrorOutputs()[0].value(), stream);
        Impl::Tensor weightsAfterHost =
            rmsCopyTensorToCpu(physicalRmsNorm->getParameter("weights")->getStorage().value(), gradientStream);
        stream.synchronize();
        gradientStream.synchronize();

        const vector<float> expectedInputGradient =
            rmsNormInputGradientReference(quantizedInput, initialWeights, quantizedUpstream, batchSize, hidden, epsilon);
        const vector<float> expectedWeightsGradient =
            rmsNormWeightGradientReference(quantizedInput, quantizedUpstream, batchSize, hidden, epsilon);
        const vector<float> expectedWeightsAfter =
            rmsSgdUpdatedReference(initialWeights, expectedWeightsGradient, batchSize, learningRate);

        rmsExpectAllClose(rmsReadCpuTensor(inputGradientHost),
                          expectedInputGradient,
                          activationTolerance,
                          activationTolerance,
                          "low-precision RMSNorm dX");
        rmsExpectAllClose(rmsReadCpuTensor(weightsAfterHost),
                          expectedWeightsAfter,
                          weightsTolerance,
                          weightsTolerance,
                          "low-precision RMSNorm weights after");
    }
}


TEST(UtilityApiLayers, RMSNormAcceptsRaggedTensorPreservesPartitionAndUsesPackedCapacityMemoryAccounting) {
    constexpr uint64_t fullRows = 66;
    constexpr uint64_t hidden = 4;
    constexpr uint32_t logicalBatchSize = 2;

    Network network("rms_norm_ragged_build");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::FP32)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({hidden})
                             .maxTotalValues(fullRows)
                             .batchSize(logicalBatchSize)
                             .build();

    RMSNorm layer = RMSNorm::Builder().network(network).featureInput(input).build();
    ASSERT_TRUE(layer.getUseRagged());
    ASSERT_TRUE(layer.getRaggedFeatureInput().has_value());
    ASSERT_TRUE(layer.getRaggedFeatureOutput().has_value());
    ASSERT_EQ(layer.getFeatureInputs().size(), 2u);
    EXPECT_EQ(layer.getFeatureInputs()[0], input.getValues());
    EXPECT_EQ(layer.getFeatureInputs()[1], input.getOffsets());
    EXPECT_EQ(layer.getNormalizedShape(), (vector<uint64_t>{hidden}));
    EXPECT_EQ(layer.getRaggedFeatureOutput()->getValuesDimensions(), (vector<uint64_t>{fullRows, hidden}));
    EXPECT_EQ(layer.getRaggedFeatureOutput()->getOffsets(), input.getOffsets());
    EXPECT_EQ(layer.getOutputTensorBytes(logicalBatchSize), layer.getFeatureOutput().value().getTotalSizeInBytes());

    const json architecture = layer.architectureJson();
    EXPECT_TRUE(architecture.at("use_ragged").get<bool>());
    ASSERT_EQ(architecture.at("ragged_inputs").size(), 1u);
    ASSERT_EQ(architecture.at("ragged_outputs").size(), 1u);
    EXPECT_EQ(architecture.at("ragged_inputs").at(0).at("offsets").at("id").get<uint64_t>(),
              architecture.at("ragged_outputs").at(0).at("offsets").at("id").get<uint64_t>());

    Network badNetwork("rms_norm_ragged_reject_packed_row_normalization");
    RaggedTensor badInput = RaggedNetworkInput::Builder()
                                .network(badNetwork)
                                .name("tokens")
                                .valuesDataType(DataType::FP32)
                                .offsetsDataType(DataType::UINT32)
                                .trailingDimensions({hidden})
                                .maxTotalValues(fullRows)
                                .batchSize(logicalBatchSize)
                                .build();
    EXPECT_THROW(RMSNorm::Builder().network(badNetwork).featureInput(badInput).normalizedShape({fullRows, hidden}).build(),
                 std::invalid_argument);
}

TEST(UtilityApiLayers, RaggedRMSNormBf16PlacesWithFp32Scale) {
    constexpr uint32_t logicalBatchSize = 2;
    Network network("ragged_rms_norm_bf16_places");
    RaggedTensor input = RaggedNetworkInput::Builder()
                             .network(network)
                             .name("tokens")
                             .valuesDataType(DataType::BF16)
                             .offsetsDataType(DataType::UINT32)
                             .trailingDimensions({8})
                             .maxTotalValues(66)
                             .batchSize(logicalBatchSize)
                             .build();
    RMSNorm rmsNorm = RMSNorm::Builder().network(network).featureInput(input).normalizedShape({8}).build();
    ASSERT_TRUE(rmsNorm.getRaggedFeatureOutput().has_value());
    (void)RaggedNetworkOutput::Builder()
        .network(network)
        .name("output")
        .inputTensor(rmsNorm.getRaggedFeatureOutput().value())
        .build();

    vector<Event> initDoneEvents;
    shared_ptr<PlacedNetwork> placed = network.place(logicalBatchSize, initDoneEvents, /*inferenceOnly=*/true);
    rmsSynchronizeEvents(initDoneEvents);
    ASSERT_NE(placed, nullptr);
    auto physicalRmsNorm = dynamic_pointer_cast<Impl::RaggedRMSNorm>(
        placed->getStampedNetwork(0).getPhysicalLayerFromApiLayer(rmsNorm.getId()));
    ASSERT_NE(physicalRmsNorm, nullptr);
    const vector<uint64_t> capacityBuckets = Impl::makeRaggedRmsNormCapacityBuckets(66);
    EXPECT_EQ(Impl::chooseRaggedMatmulCapacityBucket(33, capacityBuckets), 64u);
}

TEST(UtilityApiLayers, RaggedRMSNormForwardBackwardUsesCapacityBucketsAndIgnoresInactiveStorage) {
    constexpr uint32_t logicalBatchSize = 2;
    constexpr uint64_t fullRows = 66;
    constexpr uint64_t hidden = 4;
    constexpr float epsilon = 1.0e-5f;
    constexpr float learningRate = 0.001f;
    const DataType dataType = DataType::FP32;

    Api::Network network("ragged_rms_norm_forward_backward_bucketed");
    Api::RaggedTensor networkInput = Api::RaggedNetworkInput::Builder()
                                         .network(network)
                                         .name("tokens")
                                         .valuesDataType(dataType)
                                         .offsetsDataType(DataType::UINT32)
                                         .trailingDimensions({hidden})
                                         .maxTotalValues(fullRows)
                                         .batchSize(logicalBatchSize)
                                         .build();
    Api::GradientRivet inputRivet =
        Api::GradientRivet::Builder().network(network).tensor(networkInput.getValues()).build();
    Api::RaggedTensor raggedInput(inputRivet.getFeatureOutput().value(), networkInput.getOffsets());

    shared_ptr<Api::Sgd> weightsSgd = Api::Sgd::Builder()
                                          .initialLearningRate(learningRate)
                                          .decay(0.0f)
                                          .momentum(0.0f)
                                          .build();
    Api::RMSNorm rmsNorm = Api::RMSNorm::Builder()
                               .network(network)
                               .featureInput(raggedInput)
                               .normalizedShape({hidden})
                               .epsilon(epsilon)
                               .parameterDataType(dataType)
                               .weightsOptimizer(weightsSgd)
                               .build();
    ASSERT_TRUE(rmsNorm.getRaggedFeatureOutput().has_value());
    Api::GradientRivet outputRivet = Api::GradientRivet::Builder()
                                         .network(network)
                                         .tensor(rmsNorm.getRaggedFeatureOutput()->getValues())
                                         .build();
    Api::NetworkOutput output = Api::NetworkOutput::Builder()
                                    .network(network)
                                    .name("output")
                                    .inputTensor(outputRivet.getFeatureOutput().value())
                                    .dataType(dataType)
                                    .build();

    vector<Event> initDoneEvents;
    shared_ptr<Api::PlacedNetwork> placedNetwork = network.place(logicalBatchSize, initDoneEvents, /*inferenceOnly=*/false);
    rmsSynchronizeEvents(initDoneEvents);
    ASSERT_NE(placedNetwork, nullptr);
    Impl::StampedNetwork& stampedNetwork = placedNetwork->getStampedNetwork(0);
    auto physicalOutput = dynamic_pointer_cast<Impl::NetworkOutput>(stampedNetwork.getPhysicalLayerFromApiLayer(output.getId()));
    auto physicalRmsNorm =
        dynamic_pointer_cast<Impl::RaggedRMSNorm>(stampedNetwork.getPhysicalLayerFromApiLayer(rmsNorm.getId()));
    ASSERT_NE(physicalOutput, nullptr);
    ASSERT_NE(physicalRmsNorm, nullptr);
    ASSERT_TRUE(physicalRmsNorm->getGradientUpdateStream().has_value());

    const vector<uint64_t> capacityBuckets = Impl::makeRaggedRmsNormCapacityBuckets(fullRows);
    auto selectedCapacityRows = [&](uint64_t activeRows) {
        return Impl::chooseRaggedMatmulCapacityBucket(activeRows, capacityBuckets);
    };
    EXPECT_EQ(selectedCapacityRows(7), 8u);
    EXPECT_EQ(selectedCapacityRows(9), 16u);
    EXPECT_EQ(selectedCapacityRows(31), 32u);
    EXPECT_EQ(selectedCapacityRows(33), 64u);
    EXPECT_EQ(selectedCapacityRows(66), 66u);

    const vector<float> initialWeights = {1.0f, 0.75f, -0.5f, 1.25f};
    Stream stream = physicalRmsNorm->getStreams()[0];
    Stream gradientStream = physicalRmsNorm->getGradientUpdateStream().value();

    ASSERT_EQ(physicalRmsNorm->getFeatureInputs().size(), 2u);
    ASSERT_TRUE(physicalRmsNorm->getFeatureInputs()[0].has_value());
    ASSERT_TRUE(physicalRmsNorm->getFeatureInputs()[1].has_value());
    ASSERT_EQ(physicalRmsNorm->getFeatureOutputs().size(), 1u);
    ASSERT_TRUE(physicalRmsNorm->getFeatureOutputs()[0].has_value());
    ASSERT_EQ(physicalRmsNorm->getErrorInputs().size(), 1u);
    ASSERT_TRUE(physicalRmsNorm->getErrorInputs()[0].has_value());
    ASSERT_EQ(physicalRmsNorm->getErrorOutputs().size(), 2u);
    ASSERT_TRUE(physicalRmsNorm->getErrorOutputs()[0].has_value());
    EXPECT_FALSE(physicalRmsNorm->getErrorOutputs()[1].has_value());

    Impl::Tensor packedInput = physicalRmsNorm->getFeatureInputs()[0].value();
    Impl::Tensor rowPartitionOffsets = physicalRmsNorm->getFeatureInputs()[1].value();
    Impl::Tensor rmsOutput = physicalRmsNorm->getFeatureOutputs()[0].value();
    Impl::Tensor errorInput = physicalRmsNorm->getErrorInputs()[0].value();
    Impl::Tensor dXTensor = physicalRmsNorm->getErrorOutputs()[0].value();

    for (const uint64_t activeRows : vector<uint64_t>{7, 9, 31, 33, fullRows}) {
        SCOPED_TRACE("activeRows=" + std::to_string(activeRows));
        const uint64_t selectedRows = selectedCapacityRows(activeRows);

        rmsSetParameterTensor(physicalRmsNorm->getParameter("weights"), initialWeights, stream);
        stream.synchronize();

        vector<float> inputValues(fullRows * hidden, 0.0f);
        for (uint64_t row = 0; row < activeRows; ++row) {
            for (uint64_t col = 0; col < hidden; ++col) {
                inputValues[row * hidden + col] =
                    static_cast<float>(static_cast<int>((row + 2 * col) % 11) - 5) * 0.2f +
                    static_cast<float>(col + 1) * 0.075f;
            }
        }
        ThorTest::poisonInactiveRows(inputValues, activeRows, hidden, ThorTest::RaggedInactivePoison::NaN);

        Impl::Tensor rowPartitionOffsetsHost(
            rmsCpuPlacement, Impl::TensorDescriptor(DataType::UINT32, {logicalBatchSize + 1}));
        rowPartitionOffsetsHost.getMemPtr<uint32_t>()[0] = 0;
        rowPartitionOffsetsHost.getMemPtr<uint32_t>()[1] = static_cast<uint32_t>(activeRows / 2);
        rowPartitionOffsetsHost.getMemPtr<uint32_t>()[2] = static_cast<uint32_t>(activeRows);
        rowPartitionOffsets.copyFromAsync(rowPartitionOffsetsHost, stream);
        Impl::RowPartitionRuntime rowPartition(
            rowPartitionOffsets,
            Impl::RowPartitionDescriptor(logicalBatchSize, fullRows, rowPartitionOffsets.getDataType()));
        rowPartition.setHostActiveValueCount(activeRows);

        Impl::Tensor packedInputHost(rmsCpuPlacement, Impl::TensorDescriptor(dataType, {fullRows, hidden}));
        rmsWriteCpuTensor(packedInputHost, inputValues);
        packedInput.copyFromAsync(packedInputHost, stream);

        Impl::Tensor poisonedOutputHost = rmsOutput.clone(rmsCpuPlacement);
        rmsWriteCpuTensor(poisonedOutputHost,
                          vector<float>(fullRows * hidden, std::numeric_limits<float>::quiet_NaN()));
        rmsOutput.copyFromAsync(poisonedOutputHost, stream);

        // Bypass RaggedNetworkInput so the packed consumer sees deliberately
        // poisoned inactive storage. Publish the offsets separately so host bucket
        // selection receives the authoritative logical extent.
        physicalRmsNorm->forward(packedInput, false, logicalBatchSize);
        physicalRmsNorm->forward(rowPartitionOffsets, false, logicalBatchSize);
        Event outputReady = physicalOutput->getOutputReadyEvent();
        outputReady.synchronize();

        const vector<float> actualForward = rmsReadCpuTensor(physicalOutput->getFeatureOutput().value());
        const vector<float> validInput(inputValues.begin(), inputValues.begin() + activeRows * hidden);
        const vector<float> zeroResidual(activeRows * hidden, 0.0f);
        const vector<float> expectedForward =
            rmsNormForwardReference(validInput, initialWeights, zeroResidual, activeRows, hidden, epsilon);
        rmsExpectAllClose(vector<float>(actualForward.begin(), actualForward.begin() + activeRows * hidden),
                          expectedForward,
                          3e-4f,
                          3e-4f,
                          "ragged rmsnorm forward active prefix");

        // Forward cuDNN consumes exactly the selected bucket: only [active,bucket)
        // is sanitized. Neither the consumer nor the producer touches [bucket,capacity).
        const vector<float> consumedInput = rmsReadCpuTensor(rmsCopyTensorToCpu(packedInput, stream));
        for (uint64_t row = activeRows; row < selectedRows; ++row) {
            for (uint64_t col = 0; col < hidden; ++col) {
                EXPECT_EQ(consumedInput[row * hidden + col], 0.0f)
                    << "RMSNorm forward bucket slack was not sanitized";
            }
        }
        for (uint64_t row = selectedRows; row < fullRows; ++row) {
            for (uint64_t col = 0; col < hidden; ++col) {
                EXPECT_TRUE(std::isnan(consumedInput[row * hidden + col]))
                    << "RMSNorm forward touched storage beyond its selected bucket";
            }
        }
        const vector<float> producedOutput = rmsReadCpuTensor(rmsCopyTensorToCpu(rmsOutput, stream));
        for (uint64_t row = selectedRows; row < fullRows; ++row) {
            for (uint64_t col = 0; col < hidden; ++col) {
                EXPECT_TRUE(std::isnan(producedOutput[row * hidden + col]))
                    << "RMSNorm producer canonicalized output beyond its selected bucket";
            }
        }

        vector<float> upstream(fullRows * hidden, 0.0f);
        for (uint64_t row = 0; row < activeRows; ++row) {
            for (uint64_t col = 0; col < hidden; ++col) {
                upstream[row * hidden + col] =
                    static_cast<float>(static_cast<int>((3 * row + col) % 9) - 4) * 0.125f;
            }
        }
        ThorTest::poisonInactiveRows(upstream, activeRows, hidden, ThorTest::RaggedInactivePoison::NaN);
        Impl::Tensor errorInputHost = errorInput.clone(rmsCpuPlacement);
        rmsWriteCpuTensor(errorInputHost, upstream);
        errorInput.copyFromAsync(errorInputHost, stream);

        Impl::Tensor poisonedDXHost = dXTensor.clone(rmsCpuPlacement);
        rmsWriteCpuTensor(poisonedDXHost,
                          vector<float>(fullRows * hidden, std::numeric_limits<float>::quiet_NaN()));
        dXTensor.copyFromAsync(poisonedDXHost, stream);

        physicalRmsNorm->backward(errorInput, logicalBatchSize);
        stream.synchronize();
        gradientStream.synchronize();

        const vector<float> validUpstream(upstream.begin(), upstream.begin() + activeRows * hidden);
        const vector<float> expectedDX =
            rmsNormInputGradientReference(validInput, initialWeights, validUpstream, activeRows, hidden, epsilon);
        const vector<float> expectedDScale =
            rmsNormWeightGradientReference(validInput, validUpstream, activeRows, hidden, epsilon);
        const vector<float> expectedWeightsAfter =
            rmsSgdUpdatedReference(initialWeights, expectedDScale, logicalBatchSize, learningRate);

        const vector<float> actualDX = rmsReadCpuTensor(rmsCopyTensorToCpu(dXTensor, stream));
        rmsExpectAllClose(vector<float>(actualDX.begin(), actualDX.begin() + activeRows * hidden),
                          expectedDX,
                          4e-4f,
                          4e-4f,
                          "ragged rmsnorm dX active prefix");

        const vector<float> consumedDY = rmsReadCpuTensor(rmsCopyTensorToCpu(errorInput, stream));
        for (uint64_t row = activeRows; row < selectedRows; ++row) {
            for (uint64_t col = 0; col < hidden; ++col) {
                EXPECT_EQ(consumedDY[row * hidden + col], 0.0f)
                    << "RMSNorm backward bucket slack was not sanitized";
            }
        }
        for (uint64_t row = selectedRows; row < fullRows; ++row) {
            for (uint64_t col = 0; col < hidden; ++col) {
                EXPECT_TRUE(std::isnan(consumedDY[row * hidden + col]))
                    << "RMSNorm backward touched dY storage beyond its selected bucket";
                EXPECT_TRUE(std::isnan(actualDX[row * hidden + col]))
                    << "RMSNorm producer canonicalized dX beyond its selected bucket";
            }
        }

        const vector<float> actualWeightsAfter = rmsReadCpuTensor(
            rmsCopyTensorToCpu(physicalRmsNorm->getParameter("weights")->getStorage().value(), gradientStream));
        rmsExpectAllClose(actualWeightsAfter,
                          expectedWeightsAfter,
                          4e-4f,
                          4e-4f,
                          "ragged rmsnorm dscale through fused SGD update");
    }
}
