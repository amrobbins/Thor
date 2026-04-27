#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"
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
#include <vector>

#include "test/DeepLearning/Implementation/Layers/Helpers/GradientRivet.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

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
    const vector<float>& actual, const vector<float>& expected, float atol = 2e-2f, float rtol = 2e-2f, string paramName = "") {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        if (!paramName.empty())
            paramName += " ";
        EXPECT_LE(diff, tol) << paramName << "mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
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

void setParameterTensor(shared_ptr<PhysicalParameter> parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().isPresent());
    Tensor deviceTensor = parameter->getStorage();
    Tensor cpuTensor = deviceTensor.clone(cpuPlacement);
    writeCpuTensor(cpuTensor, values);
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

void attachAdam(FullyConnected& fc, bool hasBias) {
    fc.setOptimizer("weights", make_shared<Adam>(2000, 0.001f, 0.9f, 0.999f, 1e-7f));
    if (hasBias)
        fc.setOptimizer("biases", make_shared<Adam>(2001, 0.001f, 0.9f, 0.999f, 1e-7f));
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

vector<float> randomFloatVector(mt19937& rng, uint64_t size, float low, float high) {
    uniform_real_distribution<float> dist(low, high);
    vector<float> values(size);
    for (float& value : values)
        value = dist(rng);
    return values;
}

struct FullyConnectedDirectBackwardReference {
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

FullyConnectedDirectBackwardReference computeFullyConnectedDirectBackwardReference(const vector<float>& inputValues,
                                                                                   const vector<float>& weightValues,
                                                                                   const vector<float>& biasValues,
                                                                                   const vector<float>& errorInputValues,
                                                                                   uint64_t batchSize,
                                                                                   uint64_t numInputFeatures,
                                                                                   uint64_t numOutputFeatures,
                                                                                   bool hasBias,
                                                                                   float lossScalingFactor,
                                                                                   float alpha,
                                                                                   float beta1,
                                                                                   float beta2,
                                                                                   float epsilon) {
    FullyConnectedDirectBackwardReference ref;
    ref.featureOut =
        fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, hasBias);
    ref.errorOut = fullyConnectedBackwardErrorReference(errorInputValues, weightValues, batchSize, numInputFeatures, numOutputFeatures);
    ref.weightsGrad = fullyConnectedWeightGradReference(inputValues, errorInputValues, batchSize, numInputFeatures, numOutputFeatures);
    ref.weightsM = adamFirstMomentReference(ref.weightsGrad, batchSize, lossScalingFactor, beta1);
    ref.weightsV = adamFirstVelocityReference(ref.weightsGrad, batchSize, lossScalingFactor, beta2);
    ref.weightsAfter =
        adamFirstUpdatedWeightsReference(weightValues, ref.weightsGrad, batchSize, lossScalingFactor, alpha, beta1, beta2, epsilon);

    if (hasBias) {
        ref.biasesGrad = fullyConnectedBiasGradReference(errorInputValues, batchSize, numOutputFeatures);
        ref.biasesM = adamFirstMomentReference(ref.biasesGrad, batchSize, lossScalingFactor, beta1);
        ref.biasesV = adamFirstVelocityReference(ref.biasesGrad, batchSize, lossScalingFactor, beta2);
        ref.biasesAfter =
            adamFirstUpdatedWeightsReference(biasValues, ref.biasesGrad, batchSize, lossScalingFactor, alpha, beta1, beta2, epsilon);
    }

    return ref;
}

vector<FullyConnectedDirectBackwardReference> computeFullyConnectedDirectBackwardReferenceSequence(
    const vector<vector<float>>& inputValuesByPass,
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

    vector<FullyConnectedDirectBackwardReference> refs;
    refs.reserve(inputValuesByPass.size());

    vector<float> weights = initialWeightValues;
    vector<float> weightsM(weights.size(), 0.0f);
    vector<float> weightsV(weights.size(), 0.0f);
    vector<float> biases = initialBiasValues;
    vector<float> biasesM(hasBias ? biases.size() : 0, 0.0f);
    vector<float> biasesV(hasBias ? biases.size() : 0, 0.0f);
    const float scale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);

    for (uint64_t pass = 0; pass < inputValuesByPass.size(); ++pass) {
        FullyConnectedDirectBackwardReference ref;
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

        refs.push_back(move(ref));
    }

    return refs;
}

void compileAndInitialize(NetworkInput& ni, FullyConnected& fc, NetworkOutput& no) {
    ni.compile();
    fc.compile();
    no.compile();
    ni.initialize();
    fc.initialize();
    no.initialize();
}

}  // namespace

TEST(FullyConnected, ParameterNamesAndShapesWithBias) {
    const uint64_t batchSize = 4;
    const uint32_t numInputFeatures = 3;
    const uint32_t numOutputFeatures = 2;
    const DataType dataType = DataType::FP16;

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputFeatures});
    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    FullyConnected fc(numOutputFeatures, true, Optional<DataType>::empty(), gpuPlacement, false);
    NetworkOutput no(cpuPlacement);

    attachAdam(fc, true);
    ni.connectToNextLayer(&fc);
    fc.connectToNextLayer(&no);
    compileAndInitialize(ni, fc, no);

    ASSERT_EQ(fc.listParameters(), (vector<string>{"weights", "biases"}));
    ASSERT_TRUE(fc.getParameter("weights")->getStorage().isPresent());
    ASSERT_TRUE(fc.getParameter("biases")->getStorage().isPresent());

    Tensor weights = fc.getParameter("weights")->getStorage();
    Tensor biases = fc.getParameter("biases")->getStorage();
    EXPECT_EQ(weights.getDimensions(), (vector<uint64_t>{numInputFeatures, numOutputFeatures}));
    EXPECT_EQ(biases.getDimensions(), (vector<uint64_t>{numOutputFeatures}));
    EXPECT_EQ(weights.getDataType(), dataType);
    EXPECT_EQ(biases.getDataType(), dataType);
}

TEST(FullyConnected, DirectForwardConnectionNumericalWithBias) {
    const uint64_t batchSize = 3;
    const uint32_t numInputFeatures = 4;
    const uint32_t numOutputFeatures = 3;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = {
        1.0f,
        -2.0f,
        0.5f,
        3.0f,
        -1.5f,
        2.0f,
        4.0f,
        -0.5f,
        0.25f,
        -3.0f,
        1.5f,
        2.0f,
    };
    const vector<float> weightValues = {
        0.5f,
        -1.0f,
        2.0f,
        -0.25f,
        0.75f,
        1.5f,
        1.25f,
        -2.0f,
        -0.5f,
        0.0f,
        1.0f,
        -1.5f,
    };
    const vector<float> biasValues = {0.25f, -0.5f, 1.0f};

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputFeatures});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);
    writeCpuTensor(featureIn_h, inputValues);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    FullyConnected fc(numOutputFeatures, true, Optional<DataType>::empty(), gpuPlacement, false);
    NetworkOutput no(cpuPlacement);

    attachAdam(fc, true);
    ni.connectToNextLayer(&fc);
    fc.connectToNextLayer(&no);
    compileAndInitialize(ni, fc, no);

    Stream stream = fc.getStreams()[0];
    setParameterTensor(fc.getParameter("weights"), weightValues, stream);
    setParameterTensor(fc.getParameter("biases"), biasValues, stream);

    ni.forward(featureIn_h, false, batchSize);

    Event featureOutReadyEvent = no.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    Tensor featureOut_h = no.getFeatureOutput();
    const vector<float> actual = readCpuTensor(featureOut_h);
    const vector<float> expected =
        fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
    expectAllClose(actual, expected);
}

TEST(FullyConnected, DirectForwardConnectionNumericalWithBiasRandom128x128) {
    const uint64_t batchSize = 5;
    const uint32_t numInputFeatures = 128;
    const uint32_t numOutputFeatures = 128;
    const DataType dataType = DataType::FP16;

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputFeatures});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    FullyConnected fc(numOutputFeatures, true, Optional<DataType>::empty(), gpuPlacement, false);
    NetworkOutput no(cpuPlacement);

    attachAdam(fc, true);
    ni.connectToNextLayer(&fc);
    fc.connectToNextLayer(&no);
    compileAndInitialize(ni, fc, no);

    Stream stream = fc.getStreams()[0];

    Tensor weights = fc.getParameter("weights")->getStorage();
    Tensor biases = fc.getParameter("biases")->getStorage();
    Tensor weightsCpu = weights.clone(cpuPlacement);
    Tensor biasesCpu = biases.clone(cpuPlacement);

    weightsCpu.fillRandom(-1.25, 1.25, stream);
    biasesCpu.fillRandom(-1.1, 1.1, stream);
    featureIn_h.fillRandom(-2.25, 2.25, stream);

    Event randomFillReady = stream.putEvent();
    randomFillReady.synchronize();

    const vector<float> inputValues = readCpuTensor(featureIn_h);
    const vector<float> weightValues = readCpuTensor(weightsCpu);
    const vector<float> biasValues = readCpuTensor(biasesCpu);

    weights.copyFromAsync(weightsCpu, stream);
    biases.copyFromAsync(biasesCpu, stream);

    ni.forward(featureIn_h, false, batchSize);

    Event featureOutReadyEvent = no.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    Tensor featureOut_h = no.getFeatureOutput();
    const vector<float> actual = readCpuTensor(featureOut_h);
    const vector<float> expected =
        fullyConnectedReference(inputValues, weightValues, biasValues, batchSize, numInputFeatures, numOutputFeatures, true);
    expectAllClose(actual, expected, 6e-2f, 6e-2f);
}

TEST(FullyConnected, DirectForwardConnectionNumericalWithoutBias) {
    const uint64_t batchSize = 2;
    const uint32_t numInputFeatures = 3;
    const uint32_t numOutputFeatures = 4;
    const DataType dataType = DataType::FP16;

    const vector<float> inputValues = {
        2.0f,
        -1.0f,
        0.25f,
        -3.0f,
        4.0f,
        1.5f,
    };
    const vector<float> weightValues = {
        1.0f,
        -2.0f,
        0.5f,
        0.0f,
        -1.5f,
        0.25f,
        2.0f,
        -0.75f,
        0.5f,
        1.25f,
        -1.0f,
        3.0f,
    };

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputFeatures});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);
    writeCpuTensor(featureIn_h, inputValues);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    FullyConnected fc(numOutputFeatures, false, Optional<DataType>::empty(), gpuPlacement, false);
    NetworkOutput no(cpuPlacement);

    attachAdam(fc, false);
    ni.connectToNextLayer(&fc);
    fc.connectToNextLayer(&no);
    compileAndInitialize(ni, fc, no);

    ASSERT_EQ(fc.listParameters(), (vector<string>{"weights"}));

    Stream stream = fc.getStreams()[0];
    setParameterTensor(fc.getParameter("weights"), weightValues, stream);

    ni.forward(featureIn_h, false, batchSize);

    Event featureOutReadyEvent = no.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    Tensor featureOut_h = no.getFeatureOutput();
    const vector<float> actual = readCpuTensor(featureOut_h);
    const vector<float> expected =
        fullyConnectedReference(inputValues, weightValues, {}, batchSize, numInputFeatures, numOutputFeatures, false);
    expectAllClose(actual, expected);
}

TEST(FullyConnected, DirectBackwardConnectionNumerical) {
    mt19937 rng(time(nullptr));
    const bool hasBias = bernoulli_distribution(0.5)(rng);

    const uint64_t batchSize = 32;
    const uint32_t numInputFeatures = 5;
    const uint32_t numOutputFeatures = 10;
    const DataType dataType = DataType::FP16;

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputFeatures});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);

    // Create layers.
    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    // So that backprop occurs rather than being pruned during connection,
    // since network input doesn't take an error in, and there is no loss in the network:
    GradientRivet gr1, gr2;
    FullyConnected fc(numOutputFeatures, hasBias, Optional<DataType>::empty(), gpuPlacement, false);
    NetworkOutput no(cpuPlacement);

    // Create optimizers.
    shared_ptr<Adam> adamWeights = make_shared<Adam>(2000, 0.001f, 0.9f, 0.999f, 1e-7f);
    shared_ptr<Adam> adamBiases = hasBias ? make_shared<Adam>(2001, 0.001f, 0.9f, 0.999f, 1e-7f) : nullptr;
    fc.setOptimizer("weights", adamWeights);
    if (hasBias)
        fc.setOptimizer("biases", adamBiases);

    // Connect layers.
    ni.connectToNextLayer(&gr1);
    gr1.connectToNextLayer(&fc);
    fc.connectToNextLayer(&gr2);
    gr2.connectToNextLayer(&no);

    // Compile then initialize layers (just internal state, not weights).
    ni.compile();
    gr1.compile();
    fc.compile();
    gr2.compile();
    no.compile();
    ni.initialize();
    gr1.initialize();
    fc.initialize();
    gr2.initialize();
    no.initialize();

    // Set weights and biases manually.
    const vector<string> parameterNames = fc.listParameters();
    vector<string> expectedParameterNames{"weights"};
    if (hasBias)
        expectedParameterNames.push_back("biases");
    ASSERT_EQ(parameterNames, expectedParameterNames);
    ASSERT_TRUE(fc.getParameter("weights")->getStorage().isPresent());
    if (hasBias)
        ASSERT_TRUE(fc.getParameter("biases")->getStorage().isPresent());

    Tensor weights = fc.getParameter("weights")->getStorage();
    Optional<Tensor> biases;
    if (hasBias)
        biases = fc.getParameter("biases")->getStorage();

    Stream stream = fc.getStreams()[0];
    Tensor weightsCpu = weights.clone(cpuPlacement);
    Optional<Tensor> biasesCpu;

    featureIn_h.fillRandom(-5.0, 5.0, stream);
    weightsCpu.fillRandom(-3.0, 3.0, stream);
    weights.copyFromAsync(weightsCpu, stream);

    if (hasBias) {
        assert(biases.isPresent());
        biasesCpu = biases.get().clone(cpuPlacement);
        biasesCpu.get().fillRandom(-3.0, 3.0, stream);
        biases.get().copyFromAsync(biasesCpu.get(), stream);
    }

    // Make the randomized host-side reference inputs stable before reading them.
    Event randomFillReady = stream.putEvent();
    randomFillReady.synchronize();

    const vector<float> inputValues = readCpuTensor(featureIn_h);
    const vector<float> weightValues = readCpuTensor(weightsCpu);
    const vector<float> biasValues = hasBias ? readCpuTensor(biasesCpu.get()) : vector<float>{};

    // Call forward at the network input.
    ni.forward(featureIn_h, false, batchSize);

    // Check that the network output has the right values.
    Event featureOutReadyEvent = no.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();
    Tensor featureOut_h = no.getFeatureOutput();
    const vector<float> actualFeatureOut = readCpuTensor(featureOut_h);

    // Now the backward direction.
    ASSERT_GT(fc.getErrorInputs().size(), 0);
    ASSERT_TRUE(fc.getErrorInputs()[0].isPresent());
    Tensor fcErrorInput = fc.getErrorInputs()[0];
    ASSERT_GT(fc.getErrorOutputs().size(), 0);
    ASSERT_TRUE(fc.getErrorOutputs()[0].isPresent());
    Tensor fcErrorOutput = fc.getErrorOutputs()[0];
    ASSERT_TRUE(fc.getGradientUpdateStream().isPresent());
    Stream gradientUpdateStream = fc.getGradientUpdateStream();

    Tensor fcErrorInput_h = fcErrorInput.clone(cpuPlacement);
    fcErrorInput_h.fillRandom(-3.0, 3.0, stream);

    Event errorInputFillReady = stream.putEvent();
    errorInputFillReady.synchronize();
    const vector<float> errorInputValues = readCpuTensor(fcErrorInput_h);

    fcErrorInput.copyFromAsync(fcErrorInput_h, stream);
    fc.backward(fcErrorInput, batchSize);

    Tensor fcErrorOutput_h = fcErrorOutput.clone(cpuPlacement);
    fcErrorOutput_h.copyFromAsync(fcErrorOutput, stream);

    ASSERT_TRUE(adamWeights->getWeightsGradient().isPresent());
    Tensor weightsGrad_h = copyTensorToCpu(adamWeights->getWeightsGradient().get(), gradientUpdateStream);
    Tensor weightsAfter_h = copyTensorToCpu(weights, gradientUpdateStream);
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
        biasesAfter_h = copyTensorToCpu(biases.get(), gradientUpdateStream);
        biasesM_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("m"), gradientUpdateStream);
        biasesV_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("v"), gradientUpdateStream);
    }

    stream.synchronize();
    gradientUpdateStream.synchronize();

    const vector<float> actualErrorOut = readCpuTensor(fcErrorOutput_h);
    const vector<float> actualWeightsGrad = readCpuTensor(weightsGrad_h);
    const vector<float> actualWeightsAfter = readCpuTensor(weightsAfter_h);
    const vector<float> actualWeightsM = readCpuTensor(weightsM_h);
    const vector<float> actualWeightsV = readCpuTensor(weightsV_h);

    const float lossScalingFactor = Loss::getLossScalingFactor();
    const FullyConnectedDirectBackwardReference reference = computeFullyConnectedDirectBackwardReference(inputValues,
                                                                                                         weightValues,
                                                                                                         biasValues,
                                                                                                         errorInputValues,
                                                                                                         batchSize,
                                                                                                         numInputFeatures,
                                                                                                         numOutputFeatures,
                                                                                                         hasBias,
                                                                                                         lossScalingFactor,
                                                                                                         0.001f,
                                                                                                         0.9f,
                                                                                                         0.999f,
                                                                                                         1e-7f);

    expectAllClose(actualFeatureOut, reference.featureOut, 6e-2f, 6e-2f, "actualFeatureOut");
    expectAllClose(actualErrorOut, reference.errorOut, 8e-2f, 8e-2f, "actualErrorOut");
    expectAllClose(actualWeightsGrad, reference.weightsGrad, 8e-2f, 8e-2f, "actualWeightsGrad");
    expectAllClose(actualWeightsM, reference.weightsM, 8e-2f, 8e-2f, "actualWeightsM");
    expectAllClose(actualWeightsV, reference.weightsV, 8e-2f, 8e-2f, "actualWeightsV");
    expectAllClose(actualWeightsAfter, reference.weightsAfter, 8e-2f, 8e-2f, "actualWeightsAfter");

    if (hasBias) {
        const vector<float> actualBiasesGrad = readCpuTensor(biasesGrad_h.get());
        const vector<float> actualBiasesAfter = readCpuTensor(biasesAfter_h.get());
        const vector<float> actualBiasesM = readCpuTensor(biasesM_h.get());
        const vector<float> actualBiasesV = readCpuTensor(biasesV_h.get());

        expectAllClose(actualBiasesGrad, reference.biasesGrad, 8e-2f, 8e-2f, "actualBiasesGrad");
        expectAllClose(actualBiasesM, reference.biasesM, 8e-2f, 8e-2f, "actualBiasesM");
        expectAllClose(actualBiasesV, reference.biasesV, 8e-2f, 8e-2f, "actualBiasesV");
        expectAllClose(actualBiasesAfter, reference.biasesAfter, 8e-2f, 8e-2f, "actualBiasesAfter");
    }
}

TEST(FullyConnected, DirectBackwardConnectionNumericalThreePasses) {
    mt19937 rng(time(nullptr));

    uint32_t NUM_TESTS = 5;
    for (uint32_t t = 0; t < NUM_TESTS; ++t) {
        const bool hasBias = bernoulli_distribution(0.5)(rng);
        const uint64_t batchSize = uniform_int_distribution<uint32_t>(1, 150)(rng);
        const uint32_t numInputFeatures = uniform_int_distribution<uint32_t>(1, 300)(rng);
        const uint32_t numOutputFeatures = uniform_int_distribution<uint32_t>(1, 300)(rng);
        const DataType dataType = DataType::FP16;
        const uint64_t numPasses = uniform_int_distribution<uint32_t>(3, 5)(rng);

        TensorDescriptor featureInDescriptor(dataType, {batchSize, numInputFeatures});
        Tensor featureIn_h(cpuPlacement, featureInDescriptor);

        // Create layers.
        NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
        // So that backprop occurs rather than being pruned during connection,
        // since network input doesn't take an error in, and there is no loss in the network:
        GradientRivet gr1, gr2;
        FullyConnected fc(numOutputFeatures, hasBias, Optional<DataType>::empty(), gpuPlacement, false);
        NetworkOutput no(cpuPlacement);

        // Create optimizers.
        shared_ptr<Adam> adamWeights = make_shared<Adam>(2000, 0.001f, 0.9f, 0.999f, 1e-7f);
        shared_ptr<Adam> adamBiases = hasBias ? make_shared<Adam>(2001, 0.001f, 0.9f, 0.999f, 1e-7f) : nullptr;
        fc.setOptimizer("weights", adamWeights);
        if (hasBias)
            fc.setOptimizer("biases", adamBiases);

        // Connect layers.
        ni.connectToNextLayer(&gr1);
        gr1.connectToNextLayer(&fc);
        fc.connectToNextLayer(&gr2);
        gr2.connectToNextLayer(&no);

        // Compile then initialize layers (just internal state, not weights).
        ni.compile();
        gr1.compile();
        fc.compile();
        gr2.compile();
        no.compile();
        ni.initialize();
        gr1.initialize();
        fc.initialize();
        gr2.initialize();
        no.initialize();

        // Set weights and biases manually.
        const vector<string> parameterNames = fc.listParameters();
        vector<string> expectedParameterNames{"weights"};
        if (hasBias)
            expectedParameterNames.push_back("biases");
        ASSERT_EQ(parameterNames, expectedParameterNames);
        ASSERT_TRUE(fc.getParameter("weights")->getStorage().isPresent());
        if (hasBias)
            ASSERT_TRUE(fc.getParameter("biases")->getStorage().isPresent());

        Tensor weights = fc.getParameter("weights")->getStorage();
        Optional<Tensor> biases;
        if (hasBias)
            biases = fc.getParameter("biases")->getStorage();

        Stream stream = fc.getStreams()[0];

        const vector<float> initialWeightValues =
            randomFloatVector(rng, static_cast<uint64_t>(numInputFeatures) * numOutputFeatures, -0.5f, 0.5f);
        const vector<float> initialBiasValues = hasBias ? randomFloatVector(rng, numOutputFeatures, -0.25f, 0.25f) : vector<float>{};
        setParameterTensor(fc.getParameter("weights"), initialWeightValues, stream);
        if (hasBias) {
            assert(biases.isPresent());
            setParameterTensor(fc.getParameter("biases"), initialBiasValues, stream);
        }

        vector<vector<float>> inputValuesByPass(numPasses);
        vector<vector<float>> errorInputValuesByPass(numPasses);
        for (uint64_t pass = 0; pass < numPasses; ++pass) {
            inputValuesByPass[pass] = randomFloatVector(rng, batchSize * numInputFeatures, -0.5f, 0.5f);
            errorInputValuesByPass[pass] = randomFloatVector(rng, batchSize * numOutputFeatures, -0.5f, 0.5f);
        }

        const float lossScalingFactor = Loss::getLossScalingFactor();
        const vector<FullyConnectedDirectBackwardReference> references =
            computeFullyConnectedDirectBackwardReferenceSequence(inputValuesByPass,
                                                                 errorInputValuesByPass,
                                                                 initialWeightValues,
                                                                 initialBiasValues,
                                                                 batchSize,
                                                                 numInputFeatures,
                                                                 numOutputFeatures,
                                                                 hasBias,
                                                                 lossScalingFactor,
                                                                 0.001f,
                                                                 0.9f,
                                                                 0.999f,
                                                                 1e-7f);

        ASSERT_EQ(references.size(), numPasses);

        // Now run three forward/backward passes in sequence.
        ASSERT_GT(fc.getErrorInputs().size(), 0);
        ASSERT_TRUE(fc.getErrorInputs()[0].isPresent());
        Tensor fcErrorInput = fc.getErrorInputs()[0];
        ASSERT_GT(fc.getErrorOutputs().size(), 0);
        ASSERT_TRUE(fc.getErrorOutputs()[0].isPresent());
        Tensor fcErrorOutput = fc.getErrorOutputs()[0];
        ASSERT_TRUE(fc.getGradientUpdateStream().isPresent());
        Stream gradientUpdateStream = fc.getGradientUpdateStream();
        Tensor fcErrorInput_h = fcErrorInput.clone(cpuPlacement);

        for (uint64_t pass = 0; pass < numPasses; ++pass) {
            SCOPED_TRACE(::testing::Message() << "pass=" << pass << " hasBias=" << hasBias << " batchSize=" << batchSize
                                              << " numInputFeatures=" << numInputFeatures << " numOutputFeatures=" << numOutputFeatures);

            writeCpuTensor(featureIn_h, inputValuesByPass[pass]);

            // Call forward at the network input.
            ni.forward(featureIn_h, false, batchSize);

            // Check that the network output has the right values.
            Event featureOutReadyEvent = no.getOutputReadyEvent();
            featureOutReadyEvent.synchronize();
            Tensor featureOut_h = no.getFeatureOutput();
            const vector<float> actualFeatureOut = readCpuTensor(featureOut_h);

            // Now the backward direction.
            writeCpuTensor(fcErrorInput_h, errorInputValuesByPass[pass]);
            fcErrorInput.copyFromAsync(fcErrorInput_h, stream);
            fc.backward(fcErrorInput, batchSize);

            Tensor fcErrorOutput_h = fcErrorOutput.clone(cpuPlacement);
            fcErrorOutput_h.copyFromAsync(fcErrorOutput, stream);

            ASSERT_TRUE(adamWeights->getWeightsGradient().isPresent());
            Tensor weightsGrad_h = copyTensorToCpu(adamWeights->getWeightsGradient().get(), gradientUpdateStream);
            Tensor weightsAfter_h = copyTensorToCpu(weights, gradientUpdateStream);
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
                biasesAfter_h = copyTensorToCpu(biases.get(), gradientUpdateStream);
                biasesM_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("m"), gradientUpdateStream);
                biasesV_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("v"), gradientUpdateStream);
            }

            stream.synchronize();
            gradientUpdateStream.synchronize();

            const vector<float> actualErrorOut = readCpuTensor(fcErrorOutput_h);
            const vector<float> actualWeightsGrad = readCpuTensor(weightsGrad_h);
            const vector<float> actualWeightsAfter = readCpuTensor(weightsAfter_h);
            const vector<float> actualWeightsM = readCpuTensor(weightsM_h);
            const vector<float> actualWeightsV = readCpuTensor(weightsV_h);
            const FullyConnectedDirectBackwardReference& reference = references[pass];

            expectAllClose(actualFeatureOut, reference.featureOut, 9e-2f, 9e-2f, "actualFeatureOut");
            expectAllClose(actualErrorOut, reference.errorOut, 1e-1f, 1e-1f, "actualErrorOut");
            expectAllClose(actualWeightsGrad, reference.weightsGrad, 1e-1f, 1e-1f, "actualWeightsGrad");
            expectAllClose(actualWeightsM, reference.weightsM, 1e-1f, 1e-1f, "actualWeightsM");
            expectAllClose(actualWeightsV, reference.weightsV, 1e-1f, 1e-1f, "actualWeightsV");
            expectAllClose(actualWeightsAfter, reference.weightsAfter, 1e-1f, 1e-1f, "actualWeightsAfter");

            if (hasBias) {
                const vector<float> actualBiasesGrad = readCpuTensor(biasesGrad_h.get());
                const vector<float> actualBiasesAfter = readCpuTensor(biasesAfter_h.get());
                const vector<float> actualBiasesM = readCpuTensor(biasesM_h.get());
                const vector<float> actualBiasesV = readCpuTensor(biasesV_h.get());

                expectAllClose(actualBiasesGrad, reference.biasesGrad, 1e-1f, 1e-1f, "actualBiasesGrad");
                expectAllClose(actualBiasesM, reference.biasesM, 1e-1f, 1e-1f, "actualBiasesM");
                expectAllClose(actualBiasesV, reference.biasesV, 1e-1f, 1e-1f, "actualBiasesV");
                expectAllClose(actualBiasesAfter, reference.biasesAfter, 1e-1f, 1e-1f, "actualBiasesAfter");
            }
        }
    }
}
