#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
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

uint64_t bnIndex(uint64_t n, uint64_t c, uint64_t h, uint64_t w, uint64_t numChannels, uint64_t height, uint64_t width) {
    return ((n * numChannels + c) * height + h) * width + w;
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
    const vector<float>& actual, const vector<float>& expected, float atol = 2e-4f, float rtol = 2e-4f, const string& paramName = "") {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * fabs(expected[i]);
        EXPECT_LE(diff, tol) << paramName << " mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

vector<float> randomFloatVector(mt19937& rng, uint64_t size, float low, float high) {
    uniform_real_distribution<float> dist(low, high);
    vector<float> values(size);
    for (float& value : values)
        value = dist(rng);
    return values;
}

void setParameterTensor(shared_ptr<PhysicalParameter> parameter, const vector<float>& values, Stream& stream) {
    ASSERT_NE(parameter, nullptr);
    ASSERT_TRUE(parameter->getStorage().isPresent());
    Tensor deviceTensor = parameter->getStorage();
    Tensor cpuTensor = deviceTensor.clone(cpuPlacement);
    writeCpuTensor(cpuTensor, values);
    deviceTensor.copyFromAsync(cpuTensor, stream);
}

void compileAndInitialize(NetworkInput& ni, BatchNormalization& bn, NetworkOutput& no) {
    ni.compile();
    bn.compile();
    no.compile();
    ni.initialize();
    bn.initialize();
    no.initialize();
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

struct BatchNormForwardReference {
    vector<float> featureOut;
    vector<float> batchMean;
    vector<float> batchVariance;
    vector<float> runningMeanAfter;
    vector<float> runningVarianceAfter;
    vector<float> xhat;
};

struct BatchNormBackwardReference {
    vector<float> errorOut;
    vector<float> weightsGrad;
    vector<float> biasesGrad;
};

struct BatchNormDirectBackwardReference {
    vector<float> featureOut;
    vector<float> runningMeanAfter;
    vector<float> runningVarianceAfter;
    vector<float> errorOut;
    vector<float> weightsGrad;
    vector<float> biasesGrad;
    vector<float> weightsM;
    vector<float> weightsV;
    vector<float> weightsAfter;
    vector<float> biasesM;
    vector<float> biasesV;
    vector<float> biasesAfter;
};

BatchNormForwardReference batchNormForwardTrainingReference(const vector<float>& inputValues,
                                                            const vector<float>& weightValues,
                                                            const vector<float>& biasValues,
                                                            const vector<float>& runningMeanBefore,
                                                            const vector<float>& runningVarianceBefore,
                                                            uint64_t batchSize,
                                                            uint64_t numChannels,
                                                            uint64_t height,
                                                            uint64_t width,
                                                            double exponentialRunningAverageFactor,
                                                            double epsilon) {
    const uint64_t reduceCount = batchSize * height * width;

    BatchNormForwardReference ref;
    ref.featureOut.resize(inputValues.size(), 0.0f);
    ref.batchMean.resize(numChannels, 0.0f);
    ref.batchVariance.resize(numChannels, 0.0f);
    ref.runningMeanAfter.resize(numChannels, 0.0f);
    ref.runningVarianceAfter.resize(numChannels, 0.0f);
    ref.xhat.resize(inputValues.size(), 0.0f);

    for (uint64_t c = 0; c < numChannels; ++c) {
        double mean = 0.0;
        for (uint64_t n = 0; n < batchSize; ++n) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    mean += inputValues[bnIndex(n, c, h, w, numChannels, height, width)];
                }
            }
        }
        mean /= static_cast<double>(reduceCount);
        ref.batchMean[c] = static_cast<float>(mean);

        double variance = 0.0;
        for (uint64_t n = 0; n < batchSize; ++n) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    const double centered = inputValues[bnIndex(n, c, h, w, numChannels, height, width)] - mean;
                    variance += centered * centered;
                }
            }
        }
        variance /= static_cast<double>(reduceCount);
        ref.batchVariance[c] = static_cast<float>(variance);

        const double invStd = 1.0 / sqrt(variance + epsilon);
        for (uint64_t n = 0; n < batchSize; ++n) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    const uint64_t idx = bnIndex(n, c, h, w, numChannels, height, width);
                    const float xhat = static_cast<float>((inputValues[idx] - mean) * invStd);
                    ref.xhat[idx] = xhat;
                    ref.featureOut[idx] = weightValues[c] * xhat + biasValues[c];
                }
            }
        }

        ref.runningMeanAfter[c] =
            static_cast<float>((1.0 - exponentialRunningAverageFactor) * runningMeanBefore[c] + exponentialRunningAverageFactor * mean);

        double varianceForRunning = variance;
        if (reduceCount > 1) {
            varianceForRunning *= static_cast<double>(reduceCount) / static_cast<double>(reduceCount - 1);
        }
        ref.runningVarianceAfter[c] = static_cast<float>((1.0 - exponentialRunningAverageFactor) * runningVarianceBefore[c] +
                                                         exponentialRunningAverageFactor * varianceForRunning);
    }

    return ref;
}

vector<float> batchNormForwardInferenceReference(const vector<float>& inputValues,
                                                 const vector<float>& weightValues,
                                                 const vector<float>& biasValues,
                                                 const vector<float>& runningMean,
                                                 const vector<float>& runningVariance,
                                                 uint64_t batchSize,
                                                 uint64_t numChannels,
                                                 uint64_t height,
                                                 uint64_t width,
                                                 double epsilon) {
    vector<float> featureOut(inputValues.size(), 0.0f);
    for (uint64_t n = 0; n < batchSize; ++n) {
        for (uint64_t c = 0; c < numChannels; ++c) {
            const double invStd = 1.0 / sqrt(static_cast<double>(runningVariance[c]) + epsilon);
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    const uint64_t idx = bnIndex(n, c, h, w, numChannels, height, width);
                    const float xhat = static_cast<float>((inputValues[idx] - runningMean[c]) * invStd);
                    featureOut[idx] = weightValues[c] * xhat + biasValues[c];
                }
            }
        }
    }
    return featureOut;
}

BatchNormBackwardReference batchNormBackwardReference(const vector<float>& inputValues,
                                                      const vector<float>& errorInputValues,
                                                      const vector<float>& weightValues,
                                                      const vector<float>& batchMean,
                                                      const vector<float>& batchVariance,
                                                      uint64_t batchSize,
                                                      uint64_t numChannels,
                                                      uint64_t height,
                                                      uint64_t width,
                                                      double epsilon) {
    const uint64_t reduceCount = batchSize * height * width;

    BatchNormBackwardReference ref;
    ref.errorOut.resize(inputValues.size(), 0.0f);
    ref.weightsGrad.resize(numChannels, 0.0f);
    ref.biasesGrad.resize(numChannels, 0.0f);

    for (uint64_t c = 0; c < numChannels; ++c) {
        const double mean = batchMean[c];
        const double variance = batchVariance[c];
        const double invStd = 1.0 / sqrt(variance + epsilon);

        double sumDy = 0.0;
        double sumDyXhat = 0.0;
        for (uint64_t n = 0; n < batchSize; ++n) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    const uint64_t idx = bnIndex(n, c, h, w, numChannels, height, width);
                    const double dy = errorInputValues[idx];
                    const double xhat = (inputValues[idx] - mean) * invStd;
                    sumDy += dy;
                    sumDyXhat += dy * xhat;
                }
            }
        }

        ref.biasesGrad[c] = static_cast<float>(sumDy);
        ref.weightsGrad[c] = static_cast<float>(sumDyXhat);

        for (uint64_t n = 0; n < batchSize; ++n) {
            for (uint64_t h = 0; h < height; ++h) {
                for (uint64_t w = 0; w < width; ++w) {
                    const uint64_t idx = bnIndex(n, c, h, w, numChannels, height, width);
                    const double dy = errorInputValues[idx];
                    const double xhat = (inputValues[idx] - mean) * invStd;
                    const double dx = (static_cast<double>(weightValues[c]) * invStd / static_cast<double>(reduceCount)) *
                                      (static_cast<double>(reduceCount) * dy - sumDy - xhat * sumDyXhat);
                    ref.errorOut[idx] = static_cast<float>(dx);
                }
            }
        }
    }

    return ref;
}

BatchNormDirectBackwardReference computeBatchNormDirectBackwardReference(const vector<float>& inputValues,
                                                                         const vector<float>& weightValues,
                                                                         const vector<float>& biasValues,
                                                                         const vector<float>& runningMeanBefore,
                                                                         const vector<float>& runningVarianceBefore,
                                                                         const vector<float>& errorInputValues,
                                                                         uint64_t batchSize,
                                                                         uint64_t numChannels,
                                                                         uint64_t height,
                                                                         uint64_t width,
                                                                         double exponentialRunningAverageFactor,
                                                                         double epsilon,
                                                                         float lossScalingFactor,
                                                                         float alpha,
                                                                         float beta1,
                                                                         float beta2,
                                                                         float adamEpsilon) {
    BatchNormDirectBackwardReference ref;
    const BatchNormForwardReference forward = batchNormForwardTrainingReference(inputValues,
                                                                                weightValues,
                                                                                biasValues,
                                                                                runningMeanBefore,
                                                                                runningVarianceBefore,
                                                                                batchSize,
                                                                                numChannels,
                                                                                height,
                                                                                width,
                                                                                exponentialRunningAverageFactor,
                                                                                epsilon);
    const BatchNormBackwardReference backward = batchNormBackwardReference(inputValues,
                                                                           errorInputValues,
                                                                           weightValues,
                                                                           forward.batchMean,
                                                                           forward.batchVariance,
                                                                           batchSize,
                                                                           numChannels,
                                                                           height,
                                                                           width,
                                                                           epsilon);

    ref.featureOut = forward.featureOut;
    ref.runningMeanAfter = forward.runningMeanAfter;
    ref.runningVarianceAfter = forward.runningVarianceAfter;
    ref.errorOut = backward.errorOut;
    ref.weightsGrad = backward.weightsGrad;
    ref.biasesGrad = backward.biasesGrad;
    ref.weightsM = adamFirstMomentReference(ref.weightsGrad, batchSize, lossScalingFactor, beta1);
    ref.weightsV = adamFirstVelocityReference(ref.weightsGrad, batchSize, lossScalingFactor, beta2);
    ref.weightsAfter =
        adamFirstUpdatedWeightsReference(weightValues, ref.weightsGrad, batchSize, lossScalingFactor, alpha, beta1, beta2, adamEpsilon);
    ref.biasesM = adamFirstMomentReference(ref.biasesGrad, batchSize, lossScalingFactor, beta1);
    ref.biasesV = adamFirstVelocityReference(ref.biasesGrad, batchSize, lossScalingFactor, beta2);
    ref.biasesAfter =
        adamFirstUpdatedWeightsReference(biasValues, ref.biasesGrad, batchSize, lossScalingFactor, alpha, beta1, beta2, adamEpsilon);
    return ref;
}

}  // namespace

TEST(BatchNormalization, ParameterNamesAndShapes) {
    const uint64_t batchSize = 4;
    const uint64_t numChannels = 5;
    const DataType dataType = DataType::FP32;

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numChannels});
    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    BatchNormalization bn(gpuPlacement, false, 0);
    NetworkOutput no(cpuPlacement);

    bn.setOptimizer("weights", make_shared<Adam>(4000, 0.001f, 0.9f, 0.999f, 1e-7f));
    bn.setOptimizer("biases", make_shared<Adam>(4001, 0.001f, 0.9f, 0.999f, 1e-7f));

    ni.connectToNextLayer(&bn);
    bn.connectToNextLayer(&no);
    compileAndInitialize(ni, bn, no);

    ASSERT_EQ(bn.listParameters(), (vector<string>{"weights", "biases", "running_mean", "running_variance"}));
    ASSERT_TRUE(bn.getParameter("weights")->getStorage().isPresent());
    ASSERT_TRUE(bn.getParameter("biases")->getStorage().isPresent());
    ASSERT_TRUE(bn.getParameter("running_mean")->getStorage().isPresent());
    ASSERT_TRUE(bn.getParameter("running_variance")->getStorage().isPresent());

    Tensor weights = bn.getParameter("weights")->getStorage();
    Tensor biases = bn.getParameter("biases")->getStorage();
    Tensor runningMean = bn.getParameter("running_mean")->getStorage();
    Tensor runningVariance = bn.getParameter("running_variance")->getStorage();

    EXPECT_EQ(weights.getDimensions(), (vector<uint64_t>{numChannels}));
    EXPECT_EQ(biases.getDimensions(), (vector<uint64_t>{numChannels}));
    EXPECT_EQ(runningMean.getDimensions(), (vector<uint64_t>{numChannels}));
    EXPECT_EQ(runningVariance.getDimensions(), (vector<uint64_t>{numChannels}));

    EXPECT_EQ(weights.getDataType(), DataType::FP32);
    EXPECT_EQ(biases.getDataType(), DataType::FP32);
    EXPECT_EQ(runningMean.getDataType(), DataType::FP32);
    EXPECT_EQ(runningVariance.getDataType(), DataType::FP32);
}

TEST(BatchNormalization, DirectForwardTrainingPerActivation2DNumerical) {
    const uint64_t batchSize = 4;
    const uint64_t numChannels = 3;
    const uint64_t height = 1;
    const uint64_t width = 1;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValues = {
        1.0f,
        -2.0f,
        0.5f,
        0.0f,
        1.0f,
        -1.5f,
        2.0f,
        -1.0f,
        3.0f,
        -3.0f,
        0.5f,
        -0.25f,
    };
    const vector<float> weightValues = {1.5f, -0.75f, 0.25f};
    const vector<float> biasValues = {0.2f, -0.3f, 1.1f};
    const vector<float> runningMeanBefore = {0.4f, -0.2f, 0.7f};
    const vector<float> runningVarianceBefore = {1.2f, 0.8f, 2.5f};

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numChannels});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);
    writeCpuTensor(featureIn_h, inputValues);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    BatchNormalization bn(gpuPlacement, false, 0);
    NetworkOutput no(cpuPlacement);

    bn.setOptimizer("weights", make_shared<Adam>(4000, 0.001f, 0.9f, 0.999f, 1e-7f));
    bn.setOptimizer("biases", make_shared<Adam>(4001, 0.001f, 0.9f, 0.999f, 1e-7f));
    ni.connectToNextLayer(&bn);
    bn.connectToNextLayer(&no);
    compileAndInitialize(ni, bn, no);

    Stream stream = bn.getStreams()[0];
    setParameterTensor(bn.getParameter("weights"), weightValues, stream);
    setParameterTensor(bn.getParameter("biases"), biasValues, stream);
    setParameterTensor(bn.getParameter("running_mean"), runningMeanBefore, stream);
    setParameterTensor(bn.getParameter("running_variance"), runningVarianceBefore, stream);

    ni.forward(featureIn_h, false, batchSize);

    Event featureOutReadyEvent = no.getOutputReadyEvent();
    featureOutReadyEvent.synchronize();

    const vector<float> actualFeatureOut = readCpuTensor(no.getFeatureOutput());
    const vector<float> actualRunningMean = readCpuTensor(copyTensorToCpu(bn.getParameter("running_mean")->getStorage(), stream));
    const vector<float> actualRunningVariance = readCpuTensor(copyTensorToCpu(bn.getParameter("running_variance")->getStorage(), stream));

    const double epsilon = bn.getEpsilon();
    const BatchNormForwardReference reference = batchNormForwardTrainingReference(inputValues,
                                                                                  weightValues,
                                                                                  biasValues,
                                                                                  runningMeanBefore,
                                                                                  runningVarianceBefore,
                                                                                  batchSize,
                                                                                  numChannels,
                                                                                  height,
                                                                                  width,
                                                                                  1.0,
                                                                                  epsilon);

    expectAllClose(actualFeatureOut, reference.featureOut, 2e-4f, 2e-4f, "featureOut");
    expectAllClose(actualRunningMean, reference.runningMeanAfter, 2e-4f, 2e-4f, "runningMean");
    expectAllClose(actualRunningVariance, reference.runningVarianceAfter, 2e-4f, 2e-4f, "runningVariance");
}

TEST(BatchNormalization, DirectForwardTrainingSpatialTwoPassesRunningStats) {
    const uint64_t batchSize = 2;
    const uint64_t numChannels = 2;
    const uint64_t height = 2;
    const uint64_t width = 3;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValuesPass1 = {
        1.0f,  0.5f,   -1.0f, 2.0f, -0.5f,  0.0f,  -1.0f, 1.5f,  0.25f, 0.5f, 2.5f,  -2.0f,
        0.75f, -1.25f, 0.5f,  1.0f, -0.75f, 1.25f, 2.0f,  -0.5f, -1.5f, 0.0f, 1.25f, 0.75f,
    };
    const vector<float> inputValuesPass2 = {
        -0.75f, 1.25f, 0.5f,  -1.5f,  0.25f, 2.0f,  1.5f, -0.25f, 0.75f, -0.5f, 1.0f,   -1.25f,
        -1.0f,  0.0f,  1.25f, -0.75f, 0.5f,  1.75f, 0.5f, -1.5f,  2.25f, 1.0f,  -0.25f, -0.5f,
    };
    const vector<float> weightValues = {1.25f, -0.8f};
    const vector<float> biasValues = {-0.1f, 0.35f};
    const vector<float> runningMeanBefore = {0.2f, -0.4f};
    const vector<float> runningVarianceBefore = {1.0f, 1.7f};

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numChannels, height, width});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    BatchNormalization bn(gpuPlacement, false, 0);
    NetworkOutput no(cpuPlacement);

    bn.setOptimizer("weights", make_shared<Adam>(4000, 0.001f, 0.9f, 0.999f, 1e-7f));
    bn.setOptimizer("biases", make_shared<Adam>(4001, 0.001f, 0.9f, 0.999f, 1e-7f));

    ni.connectToNextLayer(&bn);
    bn.connectToNextLayer(&no);
    compileAndInitialize(ni, bn, no);

    Stream stream = bn.getStreams()[0];
    setParameterTensor(bn.getParameter("weights"), weightValues, stream);
    setParameterTensor(bn.getParameter("biases"), biasValues, stream);
    setParameterTensor(bn.getParameter("running_mean"), runningMeanBefore, stream);
    setParameterTensor(bn.getParameter("running_variance"), runningVarianceBefore, stream);

    const double epsilon = bn.getEpsilon();

    writeCpuTensor(featureIn_h, inputValuesPass1);
    ni.forward(featureIn_h, false, batchSize);
    no.getOutputReadyEvent().synchronize();

    const vector<float> actualFeatureOutPass1 = readCpuTensor(no.getFeatureOutput());
    const vector<float> actualRunningMeanPass1 = readCpuTensor(copyTensorToCpu(bn.getParameter("running_mean")->getStorage(), stream));
    const vector<float> actualRunningVariancePass1 =
        readCpuTensor(copyTensorToCpu(bn.getParameter("running_variance")->getStorage(), stream));

    const BatchNormForwardReference refPass1 = batchNormForwardTrainingReference(inputValuesPass1,
                                                                                 weightValues,
                                                                                 biasValues,
                                                                                 runningMeanBefore,
                                                                                 runningVarianceBefore,
                                                                                 batchSize,
                                                                                 numChannels,
                                                                                 height,
                                                                                 width,
                                                                                 1.0,
                                                                                 epsilon);

    expectAllClose(actualFeatureOutPass1, refPass1.featureOut, 2e-4f, 2e-4f, "featureOut pass1");
    expectAllClose(actualRunningMeanPass1, refPass1.runningMeanAfter, 2e-4f, 2e-4f, "runningMean pass1");
    expectAllClose(actualRunningVariancePass1, refPass1.runningVarianceAfter, 2e-4f, 2e-4f, "runningVariance pass1");

    writeCpuTensor(featureIn_h, inputValuesPass2);
    ni.forward(featureIn_h, false, batchSize);
    no.getOutputReadyEvent().synchronize();

    const vector<float> actualFeatureOutPass2 = readCpuTensor(no.getFeatureOutput());
    const vector<float> actualRunningMeanPass2 = readCpuTensor(copyTensorToCpu(bn.getParameter("running_mean")->getStorage(), stream));
    const vector<float> actualRunningVariancePass2 =
        readCpuTensor(copyTensorToCpu(bn.getParameter("running_variance")->getStorage(), stream));

    const BatchNormForwardReference refPass2 = batchNormForwardTrainingReference(inputValuesPass2,
                                                                                 weightValues,
                                                                                 biasValues,
                                                                                 refPass1.runningMeanAfter,
                                                                                 refPass1.runningVarianceAfter,
                                                                                 batchSize,
                                                                                 numChannels,
                                                                                 height,
                                                                                 width,
                                                                                 0.5,
                                                                                 epsilon);

    expectAllClose(actualFeatureOutPass2, refPass2.featureOut, 2e-4f, 2e-4f, "featureOut pass2");
    expectAllClose(actualRunningMeanPass2, refPass2.runningMeanAfter, 2e-4f, 2e-4f, "runningMean pass2");
    expectAllClose(actualRunningVariancePass2, refPass2.runningVarianceAfter, 2e-4f, 2e-4f, "runningVariance pass2");
}

TEST(BatchNormalization, DirectForwardInferenceSpatialNumerical) {
    const uint64_t batchSize = 2;
    const uint64_t numChannels = 3;
    const uint64_t height = 2;
    const uint64_t width = 2;
    const DataType dataType = DataType::FP32;

    const vector<float> inputValues = {
        1.0f,   -1.0f, 0.5f,  2.0f,   -0.25f, 1.5f,  -1.5f, 0.75f, 2.5f,  -0.5f, 0.0f,  1.0f,
        -1.25f, 0.25f, 1.75f, -0.75f, 0.5f,   -2.0f, 1.0f,  0.25f, -1.0f, 0.5f,  -0.5f, 1.5f,
    };
    const vector<float> weightValues = {1.1f, -0.7f, 0.35f};
    const vector<float> biasValues = {0.15f, -0.2f, 0.8f};
    const vector<float> runningMean = {0.5f, -0.25f, 1.2f};
    const vector<float> runningVariance = {1.5f, 0.75f, 2.25f};

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numChannels, height, width});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);
    writeCpuTensor(featureIn_h, inputValues);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    BatchNormalization bn(gpuPlacement, true, 0);
    NetworkOutput no(cpuPlacement);

    bn.setOptimizer("weights", make_shared<Adam>(4000, 0.001f, 0.9f, 0.999f, 1e-7f));
    bn.setOptimizer("biases", make_shared<Adam>(4001, 0.001f, 0.9f, 0.999f, 1e-7f));

    ni.connectToNextLayer(&bn);
    bn.connectToNextLayer(&no);
    compileAndInitialize(ni, bn, no);

    Stream stream = bn.getStreams()[0];
    setParameterTensor(bn.getParameter("weights"), weightValues, stream);
    setParameterTensor(bn.getParameter("biases"), biasValues, stream);
    setParameterTensor(bn.getParameter("running_mean"), runningMean, stream);
    setParameterTensor(bn.getParameter("running_variance"), runningVariance, stream);

    ni.forward(featureIn_h, true, batchSize);
    no.getOutputReadyEvent().synchronize();

    const vector<float> actualFeatureOut = readCpuTensor(no.getFeatureOutput());
    const vector<float> actualRunningMean = readCpuTensor(copyTensorToCpu(bn.getParameter("running_mean")->getStorage(), stream));
    const vector<float> actualRunningVariance = readCpuTensor(copyTensorToCpu(bn.getParameter("running_variance")->getStorage(), stream));

    const vector<float> expectedFeatureOut = batchNormForwardInferenceReference(
        inputValues, weightValues, biasValues, runningMean, runningVariance, batchSize, numChannels, height, width, bn.getEpsilon());

    expectAllClose(actualFeatureOut, expectedFeatureOut, 2e-4f, 2e-4f, "featureOut");
    expectAllClose(actualRunningMean, runningMean, 2e-6f, 2e-6f, "runningMean");
    expectAllClose(actualRunningVariance, runningVariance, 2e-6f, 2e-6f, "runningVariance");
}

TEST(BatchNormalization, DirectBackwardSpatialNumerical) {
    const uint64_t batchSize = 3;
    const uint64_t numChannels = 2;
    const uint64_t height = 2;
    const uint64_t width = 2;
    const DataType dataType = DataType::FP32;

    mt19937 rng(7777);
    const vector<float> inputValues = randomFloatVector(rng, batchSize * numChannels * height * width, -1.25f, 1.25f);
    const vector<float> weightValues = randomFloatVector(rng, numChannels, -0.75f, 0.75f);
    const vector<float> biasValues = randomFloatVector(rng, numChannels, -0.5f, 0.5f);
    const vector<float> runningMeanBefore = randomFloatVector(rng, numChannels, -0.5f, 0.5f);
    const vector<float> runningVarianceBefore = randomFloatVector(rng, numChannels, 0.5f, 2.0f);
    const vector<float> errorInputValues = randomFloatVector(rng, batchSize * numChannels * height * width, -0.8f, 0.8f);

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numChannels, height, width});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);
    writeCpuTensor(featureIn_h, inputValues);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    GradientRivet gr1, gr2;
    BatchNormalization bn(gpuPlacement, false, 0);
    NetworkOutput no(cpuPlacement);

    shared_ptr<Adam> adamWeights = make_shared<Adam>(4000, 0.001f, 0.9f, 0.999f, 1e-7f);
    shared_ptr<Adam> adamBiases = make_shared<Adam>(4001, 0.001f, 0.9f, 0.999f, 1e-7f);
    bn.setOptimizer("weights", adamWeights);
    bn.setOptimizer("biases", adamBiases);

    ni.connectToNextLayer(&gr1);
    gr1.connectToNextLayer(&bn);
    bn.connectToNextLayer(&gr2);
    gr2.connectToNextLayer(&no);

    ni.compile();
    gr1.compile();
    bn.compile();
    gr2.compile();
    no.compile();
    ni.initialize();
    gr1.initialize();
    bn.initialize();
    gr2.initialize();
    no.initialize();

    Stream stream = bn.getStreams()[0];
    setParameterTensor(bn.getParameter("weights"), weightValues, stream);
    setParameterTensor(bn.getParameter("biases"), biasValues, stream);
    setParameterTensor(bn.getParameter("running_mean"), runningMeanBefore, stream);
    setParameterTensor(bn.getParameter("running_variance"), runningVarianceBefore, stream);

    ni.forward(featureIn_h, false, batchSize);
    no.getOutputReadyEvent().synchronize();

    const vector<float> actualFeatureOut = readCpuTensor(no.getFeatureOutput());
    const vector<float> actualRunningMean = readCpuTensor(copyTensorToCpu(bn.getParameter("running_mean")->getStorage(), stream));
    const vector<float> actualRunningVariance = readCpuTensor(copyTensorToCpu(bn.getParameter("running_variance")->getStorage(), stream));

    ASSERT_GT(bn.getErrorInputs().size(), 0);
    ASSERT_TRUE(bn.getErrorInputs()[0].isPresent());
    ASSERT_GT(bn.getErrorOutputs().size(), 0);
    ASSERT_TRUE(bn.getErrorOutputs()[0].isPresent());
    ASSERT_TRUE(bn.getGradientUpdateStream().isPresent());

    Tensor bnErrorInput = bn.getErrorInputs()[0];
    Tensor bnErrorOutput = bn.getErrorOutputs()[0];
    Stream gradientUpdateStream = bn.getGradientUpdateStream();

    Tensor bnErrorInput_h = bnErrorInput.clone(cpuPlacement);
    writeCpuTensor(bnErrorInput_h, errorInputValues);
    bnErrorInput.copyFromAsync(bnErrorInput_h, stream);
    bn.backward(bnErrorInput, batchSize);

    Tensor bnErrorOutput_h = copyTensorToCpu(bnErrorOutput, gradientUpdateStream);
    ASSERT_TRUE(adamWeights->getWeightsGradient().isPresent());
    ASSERT_TRUE(adamBiases->getWeightsGradient().isPresent());
    Tensor weightsGrad_h = copyTensorToCpu(adamWeights->getWeightsGradient().get(), gradientUpdateStream);
    Tensor biasesGrad_h = copyTensorToCpu(adamBiases->getWeightsGradient().get(), gradientUpdateStream);
    Tensor weightsAfter_h = copyTensorToCpu(bn.getParameter("weights")->getStorage(), gradientUpdateStream);
    Tensor biasesAfter_h = copyTensorToCpu(bn.getParameter("biases")->getStorage(), gradientUpdateStream);
    Tensor weightsM_h = copyTensorToCpu(adamWeights->getOptimizerParameterTensor("m"), gradientUpdateStream);
    Tensor weightsV_h = copyTensorToCpu(adamWeights->getOptimizerParameterTensor("v"), gradientUpdateStream);
    Tensor biasesM_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("m"), gradientUpdateStream);
    Tensor biasesV_h = copyTensorToCpu(adamBiases->getOptimizerParameterTensor("v"), gradientUpdateStream);

    const vector<float> actualErrorOut = readCpuTensor(bnErrorOutput_h);
    const vector<float> actualWeightsGrad = readCpuTensor(weightsGrad_h);
    const vector<float> actualBiasesGrad = readCpuTensor(biasesGrad_h);
    const vector<float> actualWeightsAfter = readCpuTensor(weightsAfter_h);
    const vector<float> actualBiasesAfter = readCpuTensor(biasesAfter_h);
    const vector<float> actualWeightsM = readCpuTensor(weightsM_h);
    const vector<float> actualWeightsV = readCpuTensor(weightsV_h);
    const vector<float> actualBiasesM = readCpuTensor(biasesM_h);
    const vector<float> actualBiasesV = readCpuTensor(biasesV_h);

    const float lossScalingFactor = Loss::getLossScalingFactor();
    const BatchNormDirectBackwardReference reference = computeBatchNormDirectBackwardReference(inputValues,
                                                                                               weightValues,
                                                                                               biasValues,
                                                                                               runningMeanBefore,
                                                                                               runningVarianceBefore,
                                                                                               errorInputValues,
                                                                                               batchSize,
                                                                                               numChannels,
                                                                                               height,
                                                                                               width,
                                                                                               1.0,
                                                                                               bn.getEpsilon(),
                                                                                               lossScalingFactor,
                                                                                               0.001f,
                                                                                               0.9f,
                                                                                               0.999f,
                                                                                               1e-7f);

    expectAllClose(actualFeatureOut, reference.featureOut, 3e-4f, 3e-4f, "featureOut");
    expectAllClose(actualRunningMean, reference.runningMeanAfter, 3e-4f, 3e-4f, "runningMean");
    expectAllClose(actualRunningVariance, reference.runningVarianceAfter, 3e-4f, 3e-4f, "runningVariance");
    expectAllClose(actualErrorOut, reference.errorOut, 7e-4f, 7e-4f, "errorOut");
    expectAllClose(actualWeightsGrad, reference.weightsGrad, 7e-4f, 7e-4f, "weightsGrad");
    expectAllClose(actualBiasesGrad, reference.biasesGrad, 7e-4f, 7e-4f, "biasesGrad");
    expectAllClose(actualWeightsM, reference.weightsM, 7e-4f, 7e-4f, "weightsM");
    expectAllClose(actualWeightsV, reference.weightsV, 7e-4f, 7e-4f, "weightsV");
    expectAllClose(actualBiasesM, reference.biasesM, 7e-4f, 7e-4f, "biasesM");
    expectAllClose(actualBiasesV, reference.biasesV, 7e-4f, 7e-4f, "biasesV");
    expectAllClose(actualWeightsAfter, reference.weightsAfter, 7e-4f, 7e-4f, "weightsAfter");
    expectAllClose(actualBiasesAfter, reference.biasesAfter, 7e-4f, 7e-4f, "biasesAfter");
}

TEST(BatchNormalization, DirectForwardTrainingSpatialRunningAverageWarmupAndClamp) {
    const uint64_t batchSize = 2;
    const uint64_t numChannels = 1;
    const uint64_t height = 1;
    const uint64_t width = 2;
    const DataType dataType = DataType::FP32;
    const double exponentialRunningAverageFactor = 0.25;

    // Each pass has 4 elements total (N * H * W = 4), so the running-variance
    // update uses the unbiased correction factor 4/3 internally.
    const vector<float> inputValuesPass1 = {1.0f, 3.0f, 1.0f, 3.0f};      // mean = 2,  var = 1
    const vector<float> inputValuesPass2 = {4.0f, 8.0f, 4.0f, 8.0f};      // mean = 6,  var = 4
    const vector<float> inputValuesPass3 = {-5.0f, -1.0f, -5.0f, -1.0f};  // mean = -3, var = 4
    const vector<float> inputValuesPass4 = {7.0f, 13.0f, 7.0f, 13.0f};    // mean = 10, var = 9
    const vector<float> inputValuesPass5 = {10.0f, 18.0f, 10.0f, 18.0f};  // mean = 14, var = 16

    const vector<float> weightValues = {1.0f};
    const vector<float> biasValues = {0.0f};
    const vector<float> runningMeanBefore = {0.5f};
    const vector<float> runningVarianceBefore = {2.0f};

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numChannels, height, width});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    BatchNormalization bn(gpuPlacement, false, 0.0, exponentialRunningAverageFactor);
    NetworkOutput no(cpuPlacement);

    bn.setOptimizer("weights", make_shared<Adam>(4000, 0.001f, 0.9f, 0.999f, 1e-7f));
    bn.setOptimizer("biases", make_shared<Adam>(4001, 0.001f, 0.9f, 0.999f, 1e-7f));

    ni.connectToNextLayer(&bn);
    bn.connectToNextLayer(&no);
    compileAndInitialize(ni, bn, no);

    Stream stream = bn.getStreams()[0];
    setParameterTensor(bn.getParameter("weights"), weightValues, stream);
    setParameterTensor(bn.getParameter("biases"), biasValues, stream);
    setParameterTensor(bn.getParameter("running_mean"), runningMeanBefore, stream);
    setParameterTensor(bn.getParameter("running_variance"), runningVarianceBefore, stream);

    vector<float> expectedRunningMean = runningMeanBefore;
    vector<float> expectedRunningVariance = runningVarianceBefore;

    auto runPass = [&](const vector<float>& inputValues, double expectedFactor, const string& label) {
        writeCpuTensor(featureIn_h, inputValues);
        ni.forward(featureIn_h, false, batchSize);
        no.getOutputReadyEvent().synchronize();

        const vector<float> actualRunningMean = readCpuTensor(copyTensorToCpu(bn.getParameter("running_mean")->getStorage(), stream));
        const vector<float> actualRunningVariance =
            readCpuTensor(copyTensorToCpu(bn.getParameter("running_variance")->getStorage(), stream));

        const BatchNormForwardReference ref = batchNormForwardTrainingReference(inputValues,
                                                                                weightValues,
                                                                                biasValues,
                                                                                expectedRunningMean,
                                                                                expectedRunningVariance,
                                                                                batchSize,
                                                                                numChannels,
                                                                                height,
                                                                                width,
                                                                                expectedFactor,
                                                                                bn.getEpsilon());

        expectAllClose(actualRunningMean, ref.runningMeanAfter, 2e-4f, 2e-4f, "runningMean " + label);
        expectAllClose(actualRunningVariance, ref.runningVarianceAfter, 2e-4f, 2e-4f, "runningVariance " + label);

        expectedRunningMean = ref.runningMeanAfter;
        expectedRunningVariance = ref.runningVarianceAfter;
    };

    runPass(inputValuesPass1, 1.0, "pass1");
    runPass(inputValuesPass2, 0.5, "pass2");
    runPass(inputValuesPass3, 1.0 / 3.0, "pass3");
    runPass(inputValuesPass4, 0.25, "pass4");

    // Pass 5 is the important one:
    // once the warmup factor reaches the configured floor (0.25 at pass 4),
    // the implementation should keep using 0.25 rather than continuing to 1/5.
    writeCpuTensor(featureIn_h, inputValuesPass5);
    ni.forward(featureIn_h, false, batchSize);
    no.getOutputReadyEvent().synchronize();

    const vector<float> actualRunningMeanPass5 = readCpuTensor(copyTensorToCpu(bn.getParameter("running_mean")->getStorage(), stream));
    const vector<float> actualRunningVariancePass5 =
        readCpuTensor(copyTensorToCpu(bn.getParameter("running_variance")->getStorage(), stream));

    const BatchNormForwardReference refPass5 = batchNormForwardTrainingReference(inputValuesPass5,
                                                                                 weightValues,
                                                                                 biasValues,
                                                                                 expectedRunningMean,
                                                                                 expectedRunningVariance,
                                                                                 batchSize,
                                                                                 numChannels,
                                                                                 height,
                                                                                 width,
                                                                                 0.25,
                                                                                 bn.getEpsilon());

    expectAllClose(actualRunningMeanPass5, refPass5.runningMeanAfter, 2e-4f, 2e-4f, "runningMean pass5");
    expectAllClose(actualRunningVariancePass5, refPass5.runningVarianceAfter, 2e-4f, 2e-4f, "runningVariance pass5");

    // And explicitly show it did NOT behave like 1/5.
    const BatchNormForwardReference wrongRefPass5 = batchNormForwardTrainingReference(inputValuesPass5,
                                                                                      weightValues,
                                                                                      biasValues,
                                                                                      expectedRunningMean,
                                                                                      expectedRunningVariance,
                                                                                      batchSize,
                                                                                      numChannels,
                                                                                      height,
                                                                                      width,
                                                                                      0.2,
                                                                                      bn.getEpsilon());

    EXPECT_GT(fabs(actualRunningMeanPass5[0] - wrongRefPass5.runningMeanAfter[0]), 1e-3f);
    EXPECT_GT(fabs(actualRunningVariancePass5[0] - wrongRefPass5.runningVarianceAfter[0]), 1e-3f);
}

vector<float> sgdUpdatedWeightsReference(const vector<float>& initialWeights,
                                         const vector<float>& rawGradient,
                                         uint64_t batchSize,
                                         float lossScalingFactor,
                                         float learningRate) {
    const float scale = 1.0f / (static_cast<float>(batchSize) * lossScalingFactor);
    vector<float> updated(initialWeights.size(), 0.0f);
    for (uint64_t i = 0; i < initialWeights.size(); ++i) {
        updated[i] = initialWeights[i] - learningRate * (rawGradient[i] * scale);
    }
    return updated;
}

struct BatchNormDirectBackwardSgdReference {
    vector<float> featureOut;
    vector<float> runningMeanAfter;
    vector<float> runningVarianceAfter;
    vector<float> errorOut;
    vector<float> weightsGrad;
    vector<float> biasesGrad;
    vector<float> weightsAfter;
    vector<float> biasesAfter;
};

BatchNormDirectBackwardSgdReference computeBatchNormDirectBackwardSgdReference(const vector<float>& inputValues,
                                                                               const vector<float>& weightValues,
                                                                               const vector<float>& biasValues,
                                                                               const vector<float>& runningMeanBefore,
                                                                               const vector<float>& runningVarianceBefore,
                                                                               const vector<float>& errorInputValues,
                                                                               uint64_t batchSize,
                                                                               uint64_t numChannels,
                                                                               uint64_t height,
                                                                               uint64_t width,
                                                                               double exponentialRunningAverageFactor,
                                                                               double epsilon,
                                                                               float lossScalingFactor,
                                                                               float learningRate) {
    BatchNormDirectBackwardSgdReference ref;
    const BatchNormForwardReference forward = batchNormForwardTrainingReference(inputValues,
                                                                                weightValues,
                                                                                biasValues,
                                                                                runningMeanBefore,
                                                                                runningVarianceBefore,
                                                                                batchSize,
                                                                                numChannels,
                                                                                height,
                                                                                width,
                                                                                exponentialRunningAverageFactor,
                                                                                epsilon);
    const BatchNormBackwardReference backward = batchNormBackwardReference(inputValues,
                                                                           errorInputValues,
                                                                           weightValues,
                                                                           forward.batchMean,
                                                                           forward.batchVariance,
                                                                           batchSize,
                                                                           numChannels,
                                                                           height,
                                                                           width,
                                                                           epsilon);

    ref.featureOut = forward.featureOut;
    ref.runningMeanAfter = forward.runningMeanAfter;
    ref.runningVarianceAfter = forward.runningVarianceAfter;
    ref.errorOut = backward.errorOut;
    ref.weightsGrad = backward.weightsGrad;
    ref.biasesGrad = backward.biasesGrad;
    ref.weightsAfter = sgdUpdatedWeightsReference(weightValues, ref.weightsGrad, batchSize, lossScalingFactor, learningRate);
    ref.biasesAfter = sgdUpdatedWeightsReference(biasValues, ref.biasesGrad, batchSize, lossScalingFactor, learningRate);
    return ref;
}

TEST(BatchNormalization, DirectBackwardSpatialNumericalWithSgd) {
    const uint64_t batchSize = 3;
    const uint64_t numChannels = 2;
    const uint64_t height = 2;
    const uint64_t width = 2;
    const DataType dataType = DataType::FP32;
    const float learningRate = 0.01f;

    mt19937 rng(8888);
    const vector<float> inputValues = randomFloatVector(rng, batchSize * numChannels * height * width, -1.0f, 1.0f);
    const vector<float> weightValues = randomFloatVector(rng, numChannels, -0.6f, 0.6f);
    const vector<float> biasValues = randomFloatVector(rng, numChannels, -0.4f, 0.4f);
    const vector<float> runningMeanBefore = randomFloatVector(rng, numChannels, -0.5f, 0.5f);
    const vector<float> runningVarianceBefore = randomFloatVector(rng, numChannels, 0.5f, 2.0f);
    const vector<float> errorInputValues = randomFloatVector(rng, batchSize * numChannels * height * width, -0.75f, 0.75f);

    TensorDescriptor featureInDescriptor(dataType, {batchSize, numChannels, height, width});
    Tensor featureIn_h(cpuPlacement, featureInDescriptor);
    writeCpuTensor(featureIn_h, inputValues);

    NetworkInput ni(gpuPlacement, dataType, featureInDescriptor.getDimensions());
    GradientRivet gr1, gr2;
    BatchNormalization bn(gpuPlacement, false, 0);
    NetworkOutput no(cpuPlacement);

    shared_ptr<Sgd> sgdWeights = make_shared<Sgd>(5000, learningRate, 0.0f, 0.0f, false);
    shared_ptr<Sgd> sgdBiases = make_shared<Sgd>(5001, learningRate, 0.0f, 0.0f, false);
    bn.setOptimizer("weights", sgdWeights);
    bn.setOptimizer("biases", sgdBiases);

    ni.connectToNextLayer(&gr1);
    gr1.connectToNextLayer(&bn);
    bn.connectToNextLayer(&gr2);
    gr2.connectToNextLayer(&no);

    ni.compile();
    gr1.compile();
    bn.compile();
    gr2.compile();
    no.compile();
    ni.initialize();
    gr1.initialize();
    bn.initialize();
    gr2.initialize();
    no.initialize();

    Stream stream = bn.getStreams()[0];
    setParameterTensor(bn.getParameter("weights"), weightValues, stream);
    setParameterTensor(bn.getParameter("biases"), biasValues, stream);
    setParameterTensor(bn.getParameter("running_mean"), runningMeanBefore, stream);
    setParameterTensor(bn.getParameter("running_variance"), runningVarianceBefore, stream);

    ni.forward(featureIn_h, false, batchSize);
    no.getOutputReadyEvent().synchronize();

    const vector<float> actualFeatureOut = readCpuTensor(no.getFeatureOutput());
    const vector<float> actualRunningMean = readCpuTensor(copyTensorToCpu(bn.getParameter("running_mean")->getStorage(), stream));
    const vector<float> actualRunningVariance = readCpuTensor(copyTensorToCpu(bn.getParameter("running_variance")->getStorage(), stream));

    ASSERT_GT(bn.getErrorInputs().size(), 0);
    ASSERT_TRUE(bn.getErrorInputs()[0].isPresent());
    ASSERT_GT(bn.getErrorOutputs().size(), 0);
    ASSERT_TRUE(bn.getErrorOutputs()[0].isPresent());
    ASSERT_TRUE(bn.getGradientUpdateStream().isPresent());

    Tensor bnErrorInput = bn.getErrorInputs()[0];
    Tensor bnErrorOutput = bn.getErrorOutputs()[0];
    Stream gradientUpdateStream = bn.getGradientUpdateStream();

    Tensor bnErrorInput_h = bnErrorInput.clone(cpuPlacement);
    writeCpuTensor(bnErrorInput_h, errorInputValues);
    bnErrorInput.copyFromAsync(bnErrorInput_h, stream);
    bn.backward(bnErrorInput, batchSize);

    Tensor bnErrorOutput_h = copyTensorToCpu(bnErrorOutput, gradientUpdateStream);
    ASSERT_TRUE(sgdWeights->getWeightsGradient().isPresent());
    ASSERT_TRUE(sgdBiases->getWeightsGradient().isPresent());
    Tensor weightsGrad_h = copyTensorToCpu(sgdWeights->getWeightsGradient().get(), gradientUpdateStream);
    Tensor biasesGrad_h = copyTensorToCpu(sgdBiases->getWeightsGradient().get(), gradientUpdateStream);
    Tensor weightsAfter_h = copyTensorToCpu(bn.getParameter("weights")->getStorage(), gradientUpdateStream);
    Tensor biasesAfter_h = copyTensorToCpu(bn.getParameter("biases")->getStorage(), gradientUpdateStream);

    const vector<float> actualErrorOut = readCpuTensor(bnErrorOutput_h);
    const vector<float> actualWeightsGrad = readCpuTensor(weightsGrad_h);
    const vector<float> actualBiasesGrad = readCpuTensor(biasesGrad_h);
    const vector<float> actualWeightsAfter = readCpuTensor(weightsAfter_h);
    const vector<float> actualBiasesAfter = readCpuTensor(biasesAfter_h);

    const float lossScalingFactor = Loss::getLossScalingFactor();
    const BatchNormDirectBackwardSgdReference reference = computeBatchNormDirectBackwardSgdReference(inputValues,
                                                                                                     weightValues,
                                                                                                     biasValues,
                                                                                                     runningMeanBefore,
                                                                                                     runningVarianceBefore,
                                                                                                     errorInputValues,
                                                                                                     batchSize,
                                                                                                     numChannels,
                                                                                                     height,
                                                                                                     width,
                                                                                                     1.0,
                                                                                                     bn.getEpsilon(),
                                                                                                     lossScalingFactor,
                                                                                                     learningRate);

    expectAllClose(actualFeatureOut, reference.featureOut, 7e-4f, 7e-4f, "featureOut");
    expectAllClose(actualRunningMean, reference.runningMeanAfter, 7e-4f, 7e-4f, "runningMean");
    expectAllClose(actualRunningVariance, reference.runningVarianceAfter, 7e-4f, 7e-4f, "runningVariance");
    expectAllClose(actualErrorOut, reference.errorOut, 9e-4f, 9e-4f, "errorOut");
    expectAllClose(actualWeightsGrad, reference.weightsGrad, 9e-4f, 9e-4f, "weightsGrad");
    expectAllClose(actualBiasesGrad, reference.biasesGrad, 9e-4f, 9e-4f, "biasesGrad");
    expectAllClose(actualWeightsAfter, reference.weightsAfter, 9e-4f, 9e-4f, "weightsAfter");
    expectAllClose(actualBiasesAfter, reference.biasesAfter, 9e-4f, 9e-4f, "biasesAfter");
}
