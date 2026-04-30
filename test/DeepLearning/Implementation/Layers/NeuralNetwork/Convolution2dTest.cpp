#include "DeepLearning/Implementation/Layers/NeuralNetwork/Convolution2d.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkOutput.h"

#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "test/DeepLearning/Implementation/Layers/Helpers/GradientRivet.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"
#include "test/Utilities/TensorOperations/GpuConvolution/ConvolutionTestHelper.h"

using namespace std;
using namespace ThorImplementation;
using DataType = TensorDescriptor::DataType;

namespace {

TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

uint64_t tensorNumel(const Tensor& tensor) {
    uint64_t numel = 1;
    for (uint64_t d : tensor.getDimensions())
        numel *= d;
    return numel;
}

void writeCpuTensor(Tensor& tensor, const vector<float>& values) {
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

vector<float> readCpuTensor(const Tensor& tensor) {
    if (!(tensor.getPlacement() == cpuPlacement)) {
        ADD_FAILURE() << "Expected CPU tensor in readCpuTensor.";
        return {};
    }

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

Tensor copyTensorToCpu(const Tensor& tensor, Stream& stream) {
    Tensor cpuTensor = tensor.clone(cpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return cpuTensor;
}

void expectAllClose(
    const vector<float>& actual, const vector<float>& expected, float atol = 2e-2f, float rtol = 2e-2f, const string& paramName = "") {
    ASSERT_EQ(actual.size(), expected.size());
    uint32_t count = 0;
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        EXPECT_LE(diff, tol) << paramName << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
        if (diff > tol)
            count += 1;
        if (count == 10)
            break;
    }
}

vector<float> randomFloatVector(mt19937& rng, uint64_t size, float low, float high) {
    uniform_real_distribution<float> dist(low, high);
    vector<float> values(size);
    for (float& value : values)
        value = dist(rng);
    return values;
}

ConvolutionKernelRequirement makeConvolutionKernelRequirement(uint64_t batchSize,
                                                              uint32_t numInputChannels,
                                                              uint64_t inputH,
                                                              uint64_t inputW,
                                                              uint32_t numOutputChannels,
                                                              uint32_t filterH,
                                                              uint32_t filterW,
                                                              uint32_t strideH,
                                                              uint32_t strideW,
                                                              uint32_t padH,
                                                              uint32_t padW) {
    return ConvolutionKernelRequirement(MachineEvaluator::instance().getGpuType(0),
                                        filterW,
                                        filterH,
                                        strideW,
                                        strideH,
                                        padW,
                                        padH,
                                        numInputChannels,
                                        numOutputChannels,
                                        batchSize,
                                        inputW,
                                        inputH);
}

pair<uint64_t, uint64_t> convOutputSpatial(uint64_t inputH,
                                           uint64_t inputW,
                                           uint32_t filterH,
                                           uint32_t filterW,
                                           uint32_t strideH,
                                           uint32_t strideW,
                                           uint32_t padH,
                                           uint32_t padW) {
    const auto requirement = makeConvolutionKernelRequirement(1, 1, inputH, inputW, 1, filterH, filterW, strideH, strideW, padH, padW);
    return {static_cast<uint64_t>(requirement.getNumOutputRows()), static_cast<uint64_t>(requirement.getNumOutputColumns())};
}

Tensor makeCpuTensor(DataType dataType, const vector<uint64_t>& dims, const vector<float>& values) {
    Tensor tensor(cpuPlacement, TensorDescriptor(dataType, dims));
    writeCpuTensor(tensor, values);
    return tensor;
}

vector<float> conv2dForwardReference(const vector<float>& inputValues,
                                     const vector<float>& weightValues,
                                     const vector<float>& biasValues,
                                     uint64_t batchSize,
                                     uint32_t numInputChannels,
                                     uint64_t inputH,
                                     uint64_t inputW,
                                     uint32_t numOutputChannels,
                                     uint32_t filterH,
                                     uint32_t filterW,
                                     uint32_t strideH,
                                     uint32_t strideW,
                                     uint32_t padH,
                                     uint32_t padW,
                                     bool hasBias) {
    const auto requirement = makeConvolutionKernelRequirement(
        batchSize, numInputChannels, inputH, inputW, numOutputChannels, filterH, filterW, strideH, strideW, padH, padW);

    Tensor inputTensor = makeCpuTensor(DataType::FP16, {batchSize, numInputChannels, inputH, inputW}, inputValues);
    Tensor weightsTensor = makeCpuTensor(DataType::FP16, {numOutputChannels, numInputChannels, filterH, filterW}, weightValues);
    Tensor outputTensor(cpuPlacement,
                        TensorDescriptor(DataType::FP16,
                                         {batchSize,
                                          numOutputChannels,
                                          static_cast<uint64_t>(requirement.getNumOutputRows()),
                                          static_cast<uint64_t>(requirement.getNumOutputColumns())}));

    Optional<Tensor> biasTensor = Optional<Tensor>::empty();
    if (hasBias) {
        biasTensor = makeCpuTensor(DataType::FP16, {numOutputChannels}, biasValues);
    }

    ConvolutionTestHelper::cpuConvolutionForward(inputTensor, weightsTensor, biasTensor, outputTensor, requirement);
    return readCpuTensor(outputTensor);
}

vector<float> conv2dBackwardErrorReference(const vector<float>& errorInputValues,
                                           const vector<float>& weightValues,
                                           uint64_t batchSize,
                                           uint32_t numInputChannels,
                                           uint64_t inputH,
                                           uint64_t inputW,
                                           uint32_t numOutputChannels,
                                           uint32_t filterH,
                                           uint32_t filterW,
                                           uint32_t strideH,
                                           uint32_t strideW,
                                           uint32_t padH,
                                           uint32_t padW) {
    const auto requirement = makeConvolutionKernelRequirement(
        batchSize, numInputChannels, inputH, inputW, numOutputChannels, filterH, filterW, strideH, strideW, padH, padW);

    Tensor errorInputTensor = makeCpuTensor(DataType::FP16,
                                            {batchSize,
                                             numOutputChannels,
                                             static_cast<uint64_t>(requirement.getNumOutputRows()),
                                             static_cast<uint64_t>(requirement.getNumOutputColumns())},
                                            errorInputValues);
    Tensor weightsTensor = makeCpuTensor(DataType::FP16, {numOutputChannels, numInputChannels, filterH, filterW}, weightValues);
    Tensor errorOutputTensor(cpuPlacement, TensorDescriptor(DataType::FP16, {batchSize, numInputChannels, inputH, inputW}));

    ConvolutionTestHelper::cpuConvolutionBackwardData(errorInputTensor, weightsTensor, errorOutputTensor, requirement);
    return readCpuTensor(errorOutputTensor);
}

vector<float> conv2dWeightGradReference(const vector<float>& inputValues,
                                        const vector<float>& errorInputValues,
                                        uint64_t batchSize,
                                        uint32_t numInputChannels,
                                        uint64_t inputH,
                                        uint64_t inputW,
                                        uint32_t numOutputChannels,
                                        uint32_t filterH,
                                        uint32_t filterW,
                                        uint32_t strideH,
                                        uint32_t strideW,
                                        uint32_t padH,
                                        uint32_t padW) {
    const auto requirement = makeConvolutionKernelRequirement(
        batchSize, numInputChannels, inputH, inputW, numOutputChannels, filterH, filterW, strideH, strideW, padH, padW);

    Tensor inputTensor = makeCpuTensor(DataType::FP16, {batchSize, numInputChannels, inputH, inputW}, inputValues);
    Tensor errorInputTensor = makeCpuTensor(DataType::FP16,
                                            {batchSize,
                                             numOutputChannels,
                                             static_cast<uint64_t>(requirement.getNumOutputRows()),
                                             static_cast<uint64_t>(requirement.getNumOutputColumns())},
                                            errorInputValues);
    Tensor weightsGradTensor(cpuPlacement, TensorDescriptor(DataType::FP32, {numOutputChannels, numInputChannels, filterH, filterW}));

    ConvolutionTestHelper::cpuConvolutionBackwardFilter(inputTensor, errorInputTensor, weightsGradTensor, requirement, false);
    return readCpuTensor(weightsGradTensor);
}

vector<float> conv2dBiasGradReference(
    const vector<float>& errorInputValues, uint64_t batchSize, uint32_t numOutputChannels, uint64_t outputH, uint64_t outputW) {
    Tensor errorInputTensor = makeCpuTensor(DataType::FP16, {batchSize, numOutputChannels, outputH, outputW}, errorInputValues);
    Tensor biasGradTensor(cpuPlacement, TensorDescriptor(DataType::FP32, {numOutputChannels}));
    ConvolutionTestHelper::cpuConvolutionBackwardBias(errorInputTensor, biasGradTensor, false);
    return readCpuTensor(biasGradTensor);
}

void setParameterTensor(shared_ptr<PhysicalParameter> parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().isPresent());
    Tensor deviceTensor = parameter->getStorage();
    Tensor cpuTensor = deviceTensor.clone(cpuPlacement);
    writeCpuTensor(cpuTensor, values);
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

void attachAdam(Convolution2d& conv, bool hasBias) {
    conv.setOptimizer("weights", make_shared<Adam>(3000, 0.001f, 0.9f, 0.999f, 1e-7f));
    if (hasBias)
        conv.setOptimizer("biases", make_shared<Adam>(3001, 0.001f, 0.9f, 0.999f, 1e-7f));
}

vector<float> adamFirstMomentReference(const vector<float>& rawGradient, uint64_t batchSize, float lossScalingFactor, float beta1) {
    const float scale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);
    vector<float> m(rawGradient.size(), 0.0f);
    for (uint64_t i = 0; i < rawGradient.size(); ++i)
        m[i] = (1.0f - beta1) * (rawGradient[i] * scale);
    return m;
}

vector<float> adamFirstVelocityReference(const vector<float>& rawGradient, uint64_t batchSize, float lossScalingFactor, float beta2) {
    const float scale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);
    vector<float> v(rawGradient.size(), 0.0f);
    for (uint64_t i = 0; i < rawGradient.size(); ++i) {
        const float g = rawGradient[i] * scale;
        v[i] = (1.0f - beta2) * g * g;
    }
    return v;
}

vector<float> adamFirstUpdatedWeightsReference(const vector<float>& initialWeights,
                                               const vector<float>& rawGradient,
                                               uint64_t batchSize,
                                               float lossScalingFactor,
                                               float alpha,
                                               float beta1,
                                               float beta2,
                                               float epsilon) {
    const double alphaT = static_cast<double>(alpha) * sqrt(1.0 - static_cast<double>(beta2)) / (1.0 - static_cast<double>(beta1));
    const float scale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);
    vector<float> updated(initialWeights.size(), 0.0f);
    for (uint64_t i = 0; i < initialWeights.size(); ++i) {
        const float g = rawGradient[i] * scale;
        const float m = (1.0f - beta1) * g;
        const float v = (1.0f - beta2) * g * g;
        updated[i] = initialWeights[i] - static_cast<float>(alphaT) * m / (sqrt(v) + epsilon);
    }
    return updated;
}

struct Convolution2dDirectBackwardReference {
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

Convolution2dDirectBackwardReference computeConvolution2dDirectBackwardReference(const vector<float>& inputValues,
                                                                                 const vector<float>& weightValues,
                                                                                 const vector<float>& biasValues,
                                                                                 const vector<float>& errorInputValues,
                                                                                 uint64_t batchSize,
                                                                                 uint32_t numInputChannels,
                                                                                 uint64_t inputH,
                                                                                 uint64_t inputW,
                                                                                 uint32_t numOutputChannels,
                                                                                 uint32_t filterH,
                                                                                 uint32_t filterW,
                                                                                 uint32_t strideH,
                                                                                 uint32_t strideW,
                                                                                 uint32_t padH,
                                                                                 uint32_t padW,
                                                                                 bool hasBias,
                                                                                 float lossScalingFactor,
                                                                                 float alpha,
                                                                                 float beta1,
                                                                                 float beta2,
                                                                                 float epsilon) {
    Convolution2dDirectBackwardReference ref;
    const auto [outputH, outputW] = convOutputSpatial(inputH, inputW, filterH, filterW, strideH, strideW, padH, padW);

    ref.featureOut = conv2dForwardReference(inputValues,
                                            weightValues,
                                            biasValues,
                                            batchSize,
                                            numInputChannels,
                                            inputH,
                                            inputW,
                                            numOutputChannels,
                                            filterH,
                                            filterW,
                                            strideH,
                                            strideW,
                                            padH,
                                            padW,
                                            hasBias);
    ref.errorOut = conv2dBackwardErrorReference(errorInputValues,
                                                weightValues,
                                                batchSize,
                                                numInputChannels,
                                                inputH,
                                                inputW,
                                                numOutputChannels,
                                                filterH,
                                                filterW,
                                                strideH,
                                                strideW,
                                                padH,
                                                padW);
    ref.weightsGrad = conv2dWeightGradReference(inputValues,
                                                errorInputValues,
                                                batchSize,
                                                numInputChannels,
                                                inputH,
                                                inputW,
                                                numOutputChannels,
                                                filterH,
                                                filterW,
                                                strideH,
                                                strideW,
                                                padH,
                                                padW);
    ref.weightsM = adamFirstMomentReference(ref.weightsGrad, batchSize, lossScalingFactor, beta1);
    ref.weightsV = adamFirstVelocityReference(ref.weightsGrad, batchSize, lossScalingFactor, beta2);
    ref.weightsAfter =
        adamFirstUpdatedWeightsReference(weightValues, ref.weightsGrad, batchSize, lossScalingFactor, alpha, beta1, beta2, epsilon);

    if (hasBias) {
        ref.biasesGrad = conv2dBiasGradReference(errorInputValues, batchSize, numOutputChannels, outputH, outputW);
        ref.biasesM = adamFirstMomentReference(ref.biasesGrad, batchSize, lossScalingFactor, beta1);
        ref.biasesV = adamFirstVelocityReference(ref.biasesGrad, batchSize, lossScalingFactor, beta2);
        ref.biasesAfter =
            adamFirstUpdatedWeightsReference(biasValues, ref.biasesGrad, batchSize, lossScalingFactor, alpha, beta1, beta2, epsilon);
    }

    return ref;
}

vector<Convolution2dDirectBackwardReference> computeConvolution2dDirectBackwardReferenceSequence(
    const vector<vector<float>>& inputValuesByPass,
    const vector<vector<float>>& errorInputValuesByPass,
    const vector<float>& initialWeightValues,
    const vector<float>& initialBiasValues,
    uint64_t batchSize,
    uint32_t numInputChannels,
    uint64_t inputH,
    uint64_t inputW,
    uint32_t numOutputChannels,
    uint32_t filterH,
    uint32_t filterW,
    uint32_t strideH,
    uint32_t strideW,
    uint32_t padH,
    uint32_t padW,
    bool hasBias,
    float lossScalingFactor,
    float alpha,
    float beta1,
    float beta2,
    float epsilon) {
    EXPECT_EQ(inputValuesByPass.size(), errorInputValuesByPass.size());

    vector<Convolution2dDirectBackwardReference> refs;
    refs.reserve(inputValuesByPass.size());

    vector<float> weights = initialWeightValues;
    vector<float> weightsM(weights.size(), 0.0f);
    vector<float> weightsV(weights.size(), 0.0f);
    vector<float> biases = initialBiasValues;
    vector<float> biasesM(hasBias ? biases.size() : 0, 0.0f);
    vector<float> biasesV(hasBias ? biases.size() : 0, 0.0f);
    const float scale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);
    const auto [outputH, outputW] = convOutputSpatial(inputH, inputW, filterH, filterW, strideH, strideW, padH, padW);

    for (uint64_t pass = 0; pass < inputValuesByPass.size(); ++pass) {
        Convolution2dDirectBackwardReference ref;
        ref.featureOut = conv2dForwardReference(inputValuesByPass[pass],
                                                weights,
                                                biases,
                                                batchSize,
                                                numInputChannels,
                                                inputH,
                                                inputW,
                                                numOutputChannels,
                                                filterH,
                                                filterW,
                                                strideH,
                                                strideW,
                                                padH,
                                                padW,
                                                hasBias);
        ref.errorOut = conv2dBackwardErrorReference(errorInputValuesByPass[pass],
                                                    weights,
                                                    batchSize,
                                                    numInputChannels,
                                                    inputH,
                                                    inputW,
                                                    numOutputChannels,
                                                    filterH,
                                                    filterW,
                                                    strideH,
                                                    strideW,
                                                    padH,
                                                    padW);
        ref.weightsGrad = conv2dWeightGradReference(inputValuesByPass[pass],
                                                    errorInputValuesByPass[pass],
                                                    batchSize,
                                                    numInputChannels,
                                                    inputH,
                                                    inputW,
                                                    numOutputChannels,
                                                    filterH,
                                                    filterW,
                                                    strideH,
                                                    strideW,
                                                    padH,
                                                    padW);

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
            ref.biasesGrad = conv2dBiasGradReference(errorInputValuesByPass[pass], batchSize, numOutputChannels, outputH, outputW);
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

        refs.push_back(move(ref));
    }

    return refs;
}

void compileAndInitialize(NetworkInput& ni, Convolution2d& conv, NetworkOutput& no) {
    ni.compile();
    conv.compile();
    no.compile();
    ni.initialize();
    conv.initialize();
    no.initialize();
}

}  // namespace

TEST(Convolution2d, ParameterNamesAndShapesWithBias) {
    const uint64_t batchSize = 4;
    const uint32_t numInputChannels = 3;
    const uint64_t inputH = 7;
    const uint64_t inputW = 5;
    const uint32_t numOutputChannels = 6;
    const uint32_t filterH = 2;
    const uint32_t filterW = 3;
    const DataType dataType = DataType::FP16;

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputChannels, inputH, inputW});
    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    Convolution2d conv(filterW, filterH, 1, 1, 0, 0, numOutputChannels, true, Optional<DataType>::empty(), gpuPlacement, false);
    NetworkOutput no(cpuPlacement);

    attachAdam(conv, true);
    ni.connectToNextLayer(&conv);
    conv.connectToNextLayer(&no);
    compileAndInitialize(ni, conv, no);

    ASSERT_EQ(conv.listParameters(), (vector<string>{"weights", "biases"}));
    ASSERT_TRUE(conv.getParameter("weights")->getStorage().isPresent());
    ASSERT_TRUE(conv.getParameter("biases")->getStorage().isPresent());

    Tensor weights = conv.getParameter("weights")->getStorage();
    Tensor biases = conv.getParameter("biases")->getStorage();
    EXPECT_EQ(weights.getDimensions(), (vector<uint64_t>{numOutputChannels, numInputChannels, filterH, filterW}));
    EXPECT_EQ(biases.getDimensions(), (vector<uint64_t>{numOutputChannels}));
    EXPECT_EQ(weights.getDataType(), dataType);
    EXPECT_EQ(biases.getDataType(), dataType);
}

TEST(Convolution2d, DirectForwardConnectionNumericalWithBias) {
    const uint64_t batchSize = 2;
    const uint32_t numInputChannels = 2;
    const uint64_t inputH = 4;
    const uint64_t inputW = 5;
    const uint32_t numOutputChannels = 3;
    const uint32_t filterH = 2;
    const uint32_t filterW = 3;
    const uint32_t strideH = 1;
    const uint32_t strideW = 1;
    const uint32_t padH = 0;
    const uint32_t padW = 1;
    const DataType dataType = DataType::FP16;

    mt19937 rng(1234);
    const vector<float> inputValues = randomFloatVector(rng, batchSize * numInputChannels * inputH * inputW, -0.75f, 0.75f);
    const vector<float> weightValues = randomFloatVector(rng, numOutputChannels * numInputChannels * filterH * filterW, -0.5f, 0.5f);
    const vector<float> biasValues = randomFloatVector(rng, numOutputChannels, -0.25f, 0.25f);

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputChannels, inputH, inputW});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);
    writeCpuTensor(featureIn_h, inputValues);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    Convolution2d conv(
        filterW, filterH, strideW, strideH, padW, padH, numOutputChannels, true, Optional<DataType>::empty(), gpuPlacement, false);
    NetworkOutput no(cpuPlacement);

    attachAdam(conv, true);
    ni.connectToNextLayer(&conv);
    conv.connectToNextLayer(&no);
    compileAndInitialize(ni, conv, no);

    Stream stream = conv.getStreams()[0];
    setParameterTensor(conv.getParameter("weights"), weightValues, stream);
    setParameterTensor(conv.getParameter("biases"), biasValues, stream);

    ni.forward(featureIn_h, false, batchSize);

    Event featureOutReadyEvent = no.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    Tensor featureOut_h = no.getFeatureOutput();
    const vector<float> actual = readCpuTensor(featureOut_h);
    const vector<float> expected = conv2dForwardReference(inputValues,
                                                          weightValues,
                                                          biasValues,
                                                          batchSize,
                                                          numInputChannels,
                                                          inputH,
                                                          inputW,
                                                          numOutputChannels,
                                                          filterH,
                                                          filterW,
                                                          strideH,
                                                          strideW,
                                                          padH,
                                                          padW,
                                                          true);
    expectAllClose(actual, expected, 5e-2f, 5e-2f, "featureOut");
}

TEST(Convolution2d, DirectForwardConnectionNumericalWithoutBiasStrideAndPadding) {
    const uint64_t batchSize = 1;
    const uint32_t numInputChannels = 2;
    const uint64_t inputH = 5;
    const uint64_t inputW = 4;
    const uint32_t numOutputChannels = 2;
    const uint32_t filterH = 3;
    const uint32_t filterW = 2;
    const uint32_t strideH = 2;
    const uint32_t strideW = 1;
    const uint32_t padH = 1;
    const uint32_t padW = 0;
    const DataType dataType = DataType::FP16;

    mt19937 rng(5678);
    const vector<float> inputValues = randomFloatVector(rng, batchSize * numInputChannels * inputH * inputW, -0.75f, 0.75f);
    const vector<float> weightValues = randomFloatVector(rng, numOutputChannels * numInputChannels * filterH * filterW, -0.5f, 0.5f);

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputChannels, inputH, inputW});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);
    writeCpuTensor(featureIn_h, inputValues);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    Convolution2d conv(
        filterW, filterH, strideW, strideH, padW, padH, numOutputChannels, false, Optional<DataType>::empty(), gpuPlacement, false);
    NetworkOutput no(cpuPlacement);

    attachAdam(conv, false);
    ni.connectToNextLayer(&conv);
    conv.connectToNextLayer(&no);
    compileAndInitialize(ni, conv, no);

    ASSERT_EQ(conv.listParameters(), (vector<string>{"weights"}));

    Stream stream = conv.getStreams()[0];
    setParameterTensor(conv.getParameter("weights"), weightValues, stream);

    ni.forward(featureIn_h, false, batchSize);

    Event featureOutReadyEvent = no.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    Tensor featureOut_h = no.getFeatureOutput();
    const vector<float> actual = readCpuTensor(featureOut_h);
    const vector<float> expected = conv2dForwardReference(inputValues,
                                                          weightValues,
                                                          {},
                                                          batchSize,
                                                          numInputChannels,
                                                          inputH,
                                                          inputW,
                                                          numOutputChannels,
                                                          filterH,
                                                          filterW,
                                                          strideH,
                                                          strideW,
                                                          padH,
                                                          padW,
                                                          false);
    expectAllClose(actual, expected, 5e-2f, 5e-2f, "featureOut");
}

TEST(Convolution2d, DirectBackwardConnectionNumerical) {
    for (bool hasBias : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "hasBias=" << hasBias);

        const uint64_t batchSize = 4;
        const uint32_t numInputChannels = 2;
        const uint64_t inputH = 5;
        const uint64_t inputW = 4;
        const uint32_t numOutputChannels = 3;
        const uint32_t filterH = 3;
        const uint32_t filterW = 2;
        const uint32_t strideH = 2;
        const uint32_t strideW = 1;
        const uint32_t padH = 1;
        const uint32_t padW = 0;
        const DataType dataType = DataType::FP16;

        const auto [outputH, outputW] = convOutputSpatial(inputH, inputW, filterH, filterW, strideH, strideW, padH, padW);

        mt19937 rng(hasBias ? 9001 : 9002);
        const vector<float> inputValues = randomFloatVector(rng, batchSize * numInputChannels * inputH * inputW, -0.5f, 0.5f);
        const vector<float> weightValues = randomFloatVector(rng, numOutputChannels * numInputChannels * filterH * filterW, -0.5f, 0.5f);
        const vector<float> biasValues = hasBias ? randomFloatVector(rng, numOutputChannels, -0.25f, 0.25f) : vector<float>{};
        const vector<float> errorInputValues = randomFloatVector(rng, batchSize * numOutputChannels * outputH * outputW, -0.5f, 0.5f);

        TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputChannels, inputH, inputW});
        Tensor featureIn_h(cpuPlacement, featureInDescriptor);
        writeCpuTensor(featureIn_h, inputValues);

        NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
        GradientRivet gr1, gr2;
        Convolution2d conv(
            filterW, filterH, strideW, strideH, padW, padH, numOutputChannels, hasBias, Optional<DataType>::empty(), gpuPlacement, false);
        NetworkOutput no(cpuPlacement);

        shared_ptr<Adam> adamWeights = make_shared<Adam>(3000, 0.001f, 0.9f, 0.999f, 1e-7f);
        shared_ptr<Adam> adamBiases = hasBias ? make_shared<Adam>(3001, 0.001f, 0.9f, 0.999f, 1e-7f) : nullptr;
        conv.setOptimizer("weights", adamWeights);
        if (hasBias)
            conv.setOptimizer("biases", adamBiases);

        ni.connectToNextLayer(&gr1);
        gr1.connectToNextLayer(&conv);
        conv.connectToNextLayer(&gr2);
        gr2.connectToNextLayer(&no);

        ni.compile();
        gr1.compile();
        conv.compile();
        gr2.compile();
        no.compile();
        ni.initialize();
        gr1.initialize();
        conv.initialize();
        gr2.initialize();
        no.initialize();

        vector<string> expectedParameterNames{"weights"};
        if (hasBias)
            expectedParameterNames.push_back("biases");
        ASSERT_EQ(conv.listParameters(), expectedParameterNames);

        Stream stream = conv.getStreams()[0];
        setParameterTensor(conv.getParameter("weights"), weightValues, stream);
        if (hasBias)
            setParameterTensor(conv.getParameter("biases"), biasValues, stream);

        ni.forward(featureIn_h, false, batchSize);

        Event featureOutReadyEvent = no.getOutputReadyEvent();
        featureOutReadyEvent.synchronize();
        Tensor featureOut_h = no.getFeatureOutput();
        const vector<float> actualFeatureOut = readCpuTensor(featureOut_h);

        ASSERT_GT(conv.getErrorInputs().size(), 0);
        ASSERT_TRUE(conv.getErrorInputs()[0].isPresent());
        Tensor convErrorInput = conv.getErrorInputs()[0];
        ASSERT_GT(conv.getErrorOutputs().size(), 0);
        ASSERT_TRUE(conv.getErrorOutputs()[0].isPresent());
        Tensor convErrorOutput = conv.getErrorOutputs()[0];
        ASSERT_TRUE(conv.getGradientUpdateStream().isPresent());
        Stream gradientUpdateStream = conv.getGradientUpdateStream();

        Tensor convErrorInput_h = convErrorInput.clone(cpuPlacement);
        writeCpuTensor(convErrorInput_h, errorInputValues);
        convErrorInput.copyFromAsync(convErrorInput_h, stream);
        conv.backward(convErrorInput, batchSize);

        Tensor convErrorOutput_h = convErrorOutput.clone(cpuPlacement);
        convErrorOutput_h.copyFromAsync(convErrorOutput, stream);

        ASSERT_TRUE(adamWeights->getWeightsGradient().isPresent());
        Tensor weightsGrad_h = copyTensorToCpu(adamWeights->getWeightsGradient().get(), gradientUpdateStream);
        Tensor weightsAfter_h = copyTensorToCpu(conv.getParameter("weights")->getStorage(), gradientUpdateStream);
        Tensor weightsM_h = copyTensorToCpu(adamWeights->getOptimizerParameterTensor("m"), gradientUpdateStream);
        Tensor weightsV_h = copyTensorToCpu(adamWeights->getOptimizerParameterTensor("v"), gradientUpdateStream);

        Optional<Tensor> biasesGrad_h;
        Optional<Tensor> biasesAfter_h;
        Optional<Tensor> biasesM_h;
        Optional<Tensor> biasesV_h;
        if (hasBias) {
            ASSERT_TRUE(adamBiases != nullptr);
            ASSERT_TRUE(adamBiases->getWeightsGradient().isPresent());
            biasesGrad_h = copyTensorToCpu(adamBiases->getWeightsGradient().get(), gradientUpdateStream);
            biasesAfter_h = copyTensorToCpu(conv.getParameter("biases")->getStorage(), gradientUpdateStream);
            biasesM_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("m"), gradientUpdateStream);
            biasesV_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("v"), gradientUpdateStream);
        }

        stream.synchronize();
        gradientUpdateStream.synchronize();

        const vector<float> actualErrorOut = readCpuTensor(convErrorOutput_h);
        const vector<float> actualWeightsGrad = readCpuTensor(weightsGrad_h);
        const vector<float> actualWeightsAfter = readCpuTensor(weightsAfter_h);
        const vector<float> actualWeightsM = readCpuTensor(weightsM_h);
        const vector<float> actualWeightsV = readCpuTensor(weightsV_h);

        const float lossScalingFactor = Loss::getLossScalingFactor();
        const Convolution2dDirectBackwardReference reference = computeConvolution2dDirectBackwardReference(inputValues,
                                                                                                           weightValues,
                                                                                                           biasValues,
                                                                                                           errorInputValues,
                                                                                                           batchSize,
                                                                                                           numInputChannels,
                                                                                                           inputH,
                                                                                                           inputW,
                                                                                                           numOutputChannels,
                                                                                                           filterH,
                                                                                                           filterW,
                                                                                                           strideH,
                                                                                                           strideW,
                                                                                                           padH,
                                                                                                           padW,
                                                                                                           hasBias,
                                                                                                           lossScalingFactor,
                                                                                                           0.001f,
                                                                                                           0.9f,
                                                                                                           0.999f,
                                                                                                           1e-7f);

        expectAllClose(actualFeatureOut, reference.featureOut, 7e-2f, 7e-2f, "featureOut");
        expectAllClose(actualErrorOut, reference.errorOut, 1e-1f, 1e-1f, "errorOut");
        expectAllClose(actualWeightsGrad, reference.weightsGrad, 1e-1f, 1e-1f, "weightsGrad");
        expectAllClose(actualWeightsM, reference.weightsM, 1e-1f, 1e-1f, "weightsM");
        expectAllClose(actualWeightsV, reference.weightsV, 1e-1f, 1e-1f, "weightsV");
        expectAllClose(actualWeightsAfter, reference.weightsAfter, 1e-1f, 1e-1f, "weightsAfter");

        if (hasBias) {
            const vector<float> actualBiasesGrad = readCpuTensor(biasesGrad_h.get());
            const vector<float> actualBiasesAfter = readCpuTensor(biasesAfter_h.get());
            const vector<float> actualBiasesM = readCpuTensor(biasesM_h.get());
            const vector<float> actualBiasesV = readCpuTensor(biasesV_h.get());

            expectAllClose(actualBiasesGrad, reference.biasesGrad, 1e-1f, 1e-1f, "biasesGrad");
            expectAllClose(actualBiasesM, reference.biasesM, 1e-1f, 1e-1f, "biasesM");
            expectAllClose(actualBiasesV, reference.biasesV, 1e-1f, 1e-1f, "biasesV");
            expectAllClose(actualBiasesAfter, reference.biasesAfter, 1e-1f, 1e-1f, "biasesAfter");
        }
    }
}

TEST(Convolution2d, DirectBackwardConnectionNumericalThreePasses) {
    for (bool hasBias : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "hasBias=" << hasBias);

        const uint64_t batchSize = 3;
        const uint32_t numInputChannels = 2;
        const uint64_t inputH = 4;
        const uint64_t inputW = 5;
        const uint32_t numOutputChannels = 2;
        const uint32_t filterH = 2;
        const uint32_t filterW = 3;
        const uint32_t strideH = 1;
        const uint32_t strideW = 2;
        const uint32_t padH = 0;
        const uint32_t padW = 1;
        const uint64_t numPasses = 3;
        const DataType dataType = DataType::FP16;

        const auto [outputH, outputW] = convOutputSpatial(inputH, inputW, filterH, filterW, strideH, strideW, padH, padW);

        mt19937 rng(hasBias ? 4242 : 5252);
        const vector<float> initialWeightValues =
            randomFloatVector(rng, numOutputChannels * numInputChannels * filterH * filterW, -0.4f, 0.4f);
        const vector<float> initialBiasValues = hasBias ? randomFloatVector(rng, numOutputChannels, -0.2f, 0.2f) : vector<float>{};

        vector<vector<float>> inputValuesByPass(numPasses);
        vector<vector<float>> errorInputValuesByPass(numPasses);
        for (uint64_t pass = 0; pass < numPasses; ++pass) {
            inputValuesByPass[pass] = randomFloatVector(rng, batchSize * numInputChannels * inputH * inputW, -0.4f, 0.4f);
            errorInputValuesByPass[pass] = randomFloatVector(rng, batchSize * numOutputChannels * outputH * outputW, -0.4f, 0.4f);
        }

        const float lossScalingFactor = Loss::getLossScalingFactor();
        const vector<Convolution2dDirectBackwardReference> references =
            computeConvolution2dDirectBackwardReferenceSequence(inputValuesByPass,
                                                                errorInputValuesByPass,
                                                                initialWeightValues,
                                                                initialBiasValues,
                                                                batchSize,
                                                                numInputChannels,
                                                                inputH,
                                                                inputW,
                                                                numOutputChannels,
                                                                filterH,
                                                                filterW,
                                                                strideH,
                                                                strideW,
                                                                padH,
                                                                padW,
                                                                hasBias,
                                                                lossScalingFactor,
                                                                0.001f,
                                                                0.9f,
                                                                0.999f,
                                                                1e-7f);

        TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputChannels, inputH, inputW});
        Tensor featureIn_h(cpuPlacement, featureInDescriptor);

        NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
        GradientRivet gr1, gr2;
        Convolution2d conv(
            filterW, filterH, strideW, strideH, padW, padH, numOutputChannels, hasBias, Optional<DataType>::empty(), gpuPlacement, false);
        NetworkOutput no(cpuPlacement);

        shared_ptr<Adam> adamWeights = make_shared<Adam>(3000, 0.001f, 0.9f, 0.999f, 1e-7f);
        shared_ptr<Adam> adamBiases = hasBias ? make_shared<Adam>(3001, 0.001f, 0.9f, 0.999f, 1e-7f) : nullptr;
        conv.setOptimizer("weights", adamWeights);
        if (hasBias)
            conv.setOptimizer("biases", adamBiases);

        ni.connectToNextLayer(&gr1);
        gr1.connectToNextLayer(&conv);
        conv.connectToNextLayer(&gr2);
        gr2.connectToNextLayer(&no);

        ni.compile();
        gr1.compile();
        conv.compile();
        gr2.compile();
        no.compile();
        ni.initialize();
        gr1.initialize();
        conv.initialize();
        gr2.initialize();
        no.initialize();

        Stream stream = conv.getStreams()[0];
        setParameterTensor(conv.getParameter("weights"), initialWeightValues, stream);
        if (hasBias)
            setParameterTensor(conv.getParameter("biases"), initialBiasValues, stream);

        ASSERT_GT(conv.getErrorInputs().size(), 0);
        ASSERT_TRUE(conv.getErrorInputs()[0].isPresent());
        Tensor convErrorInput = conv.getErrorInputs()[0];
        ASSERT_GT(conv.getErrorOutputs().size(), 0);
        ASSERT_TRUE(conv.getErrorOutputs()[0].isPresent());
        Tensor convErrorOutput = conv.getErrorOutputs()[0];
        ASSERT_TRUE(conv.getGradientUpdateStream().isPresent());
        Stream gradientUpdateStream = conv.getGradientUpdateStream();
        Tensor convErrorInput_h = convErrorInput.clone(cpuPlacement);

        for (uint64_t pass = 0; pass < numPasses; ++pass) {
            SCOPED_TRACE(::testing::Message() << "pass=" << pass);

            writeCpuTensor(featureIn_h, inputValuesByPass[pass]);
            ni.forward(featureIn_h, false, batchSize);

            Event featureOutReadyEvent = no.getOutputReadyEvent();
            featureOutReadyEvent.synchronize();
            Tensor featureOut_h = no.getFeatureOutput();
            const vector<float> actualFeatureOut = readCpuTensor(featureOut_h);

            writeCpuTensor(convErrorInput_h, errorInputValuesByPass[pass]);
            convErrorInput.copyFromAsync(convErrorInput_h, stream);
            conv.backward(convErrorInput, batchSize);

            Tensor convErrorOutput_h = convErrorOutput.clone(cpuPlacement);
            convErrorOutput_h.copyFromAsync(convErrorOutput, stream);

            ASSERT_TRUE(adamWeights->getWeightsGradient().isPresent());
            Tensor weightsGrad_h = copyTensorToCpu(adamWeights->getWeightsGradient().get(), gradientUpdateStream);
            Tensor weightsAfter_h = copyTensorToCpu(conv.getParameter("weights")->getStorage(), gradientUpdateStream);
            Tensor weightsM_h = copyTensorToCpu(adamWeights->getOptimizerParameterTensor("m"), gradientUpdateStream);
            Tensor weightsV_h = copyTensorToCpu(adamWeights->getOptimizerParameterTensor("v"), gradientUpdateStream);

            Optional<Tensor> biasesGrad_h;
            Optional<Tensor> biasesAfter_h;
            Optional<Tensor> biasesM_h;
            Optional<Tensor> biasesV_h;
            if (hasBias) {
                ASSERT_TRUE(adamBiases != nullptr);
                ASSERT_TRUE(adamBiases->getWeightsGradient().isPresent());
                biasesGrad_h = copyTensorToCpu(adamBiases->getWeightsGradient().get(), gradientUpdateStream);
                biasesAfter_h = copyTensorToCpu(conv.getParameter("biases")->getStorage(), gradientUpdateStream);
                biasesM_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("m"), gradientUpdateStream);
                biasesV_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("v"), gradientUpdateStream);
            }

            stream.synchronize();
            gradientUpdateStream.synchronize();

            const Convolution2dDirectBackwardReference& reference = references[pass];
            const vector<float> actualErrorOut = readCpuTensor(convErrorOutput_h);
            const vector<float> actualWeightsGrad = readCpuTensor(weightsGrad_h);
            const vector<float> actualWeightsAfter = readCpuTensor(weightsAfter_h);
            const vector<float> actualWeightsM = readCpuTensor(weightsM_h);
            const vector<float> actualWeightsV = readCpuTensor(weightsV_h);

            expectAllClose(actualFeatureOut, reference.featureOut, 7e-2f, 7e-2f, "featureOut");
            expectAllClose(actualErrorOut, reference.errorOut, 1.25e-1f, 1.25e-1f, "errorOut");
            expectAllClose(actualWeightsGrad, reference.weightsGrad, 1.25e-1f, 1.25e-1f, "weightsGrad");
            expectAllClose(actualWeightsM, reference.weightsM, 1.25e-1f, 1.25e-1f, "weightsM");
            expectAllClose(actualWeightsV, reference.weightsV, 1.25e-1f, 1.25e-1f, "weightsV");
            expectAllClose(actualWeightsAfter, reference.weightsAfter, 1.25e-1f, 1.25e-1f, "weightsAfter");

            if (hasBias) {
                const vector<float> actualBiasesGrad = readCpuTensor(biasesGrad_h.get());
                const vector<float> actualBiasesAfter = readCpuTensor(biasesAfter_h.get());
                const vector<float> actualBiasesM = readCpuTensor(biasesM_h.get());
                const vector<float> actualBiasesV = readCpuTensor(biasesV_h.get());

                expectAllClose(actualBiasesGrad, reference.biasesGrad, 1.25e-1f, 1.25e-1f, "biasesGrad");
                expectAllClose(actualBiasesM, reference.biasesM, 1.25e-1f, 1.25e-1f, "biasesM");
                expectAllClose(actualBiasesV, reference.biasesV, 1.25e-1f, 1.25e-1f, "biasesV");
                expectAllClose(actualBiasesAfter, reference.biasesAfter, 1.25e-1f, 1.25e-1f, "biasesAfter");
            }
        }
    }
}
