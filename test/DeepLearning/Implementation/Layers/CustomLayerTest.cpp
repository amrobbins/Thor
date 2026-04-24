#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Implementation/Layers/Utility/TensorFanout.h"

#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/Expression/FusedEquation.h"

#include "cuda_fp16.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"
#include "Helpers/GradientRivet.h"

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

void writeCpuTensor(Tensor& tensor, const std::vector<float>& values) {
    ASSERT_EQ(tensor.getPlacement(), cpuPlacement);
    ASSERT_EQ(tensorNumel(tensor), values.size());
    ASSERT_EQ(tensor.getDataType(), DataType::FP32);

    auto* ptr = static_cast<float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        ptr[i] = values[i];
}

std::vector<float> readCpuTensor(const Tensor& tensor) {
    EXPECT_EQ(tensor.getPlacement(), cpuPlacement);
    EXPECT_EQ(tensor.getDataType(), DataType::FP32);

    std::vector<float> values(tensorNumel(tensor));
    const auto* ptr = static_cast<const float*>(tensor.getMemPtr());
    for (uint64_t i = 0; i < values.size(); ++i)
        values[i] = ptr[i];
    return values;
}

Tensor copyTensorToCpu(const Tensor& tensor, Stream& stream) {
    Tensor cpuTensor = tensor.clone(cpuPlacement);
    cpuTensor.copyFromAsync(tensor, stream);
    Event copied = stream.putEvent();
    copied.synchronize();
    return cpuTensor;
}

void expectAllClose(const std::vector<float>& actual, const std::vector<float>& expected, float atol = 1e-5f, float rtol = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (uint64_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
        const float tol = atol + rtol * std::fabs(expected[i]);
        EXPECT_LE(diff, tol) << "Mismatch at index " << i << ": actual=" << actual[i] << ", expected=" << expected[i];
    }
}

void requireSameSize(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    if (a.size() != b.size()) {
        throw std::runtime_error(std::string("Mismatched vector sizes in ") + what);
    }
}

std::vector<float> scaleBiasForward(const std::vector<float>& input, const std::vector<float>& scale, const std::vector<float>& bias) {
    requireSameSize(scale, bias, "scaleBiasForward scale/bias");
    std::vector<float> out(input.size());
    for (uint64_t i = 0; i < input.size(); ++i) {
        const uint64_t f = i % scale.size();
        out[i] = input[i] * scale[f] + bias[f];
    }
    return out;
}

std::vector<float> scaleInputGradient(const std::vector<float>& outputGradient, const std::vector<float>& scale) {
    std::vector<float> inputGradient(outputGradient.size());
    for (uint64_t i = 0; i < outputGradient.size(); ++i) {
        inputGradient[i] = outputGradient[i] * scale[i % scale.size()];
    }
    return inputGradient;
}

std::vector<float> featureWiseScaleGradient(const std::vector<std::vector<float>>& inputs,
                                            const std::vector<std::vector<float>>& outputGradients,
                                            uint64_t features) {
    if (inputs.size() != outputGradients.size()) {
        throw std::runtime_error("Mismatched branch counts in featureWiseScaleGradient");
    }
    std::vector<float> gradient(features, 0.0f);
    for (uint64_t branch = 0; branch < inputs.size(); ++branch) {
        requireSameSize(inputs[branch], outputGradients[branch], "featureWiseScaleGradient input/gradient");
        for (uint64_t i = 0; i < inputs[branch].size(); ++i) {
            gradient[i % features] += inputs[branch][i] * outputGradients[branch][i];
        }
    }
    return gradient;
}

std::vector<float> featureWiseBiasGradient(const std::vector<std::vector<float>>& outputGradients, uint64_t features) {
    std::vector<float> gradient(features, 0.0f);
    for (const std::vector<float>& branchGradient : outputGradients) {
        for (uint64_t i = 0; i < branchGradient.size(); ++i) {
            gradient[i % features] += branchGradient[i];
        }
    }
    return gradient;
}

void subtractInPlace(std::vector<float>& lhs, const std::vector<float>& rhs) {
    requireSameSize(lhs, rhs, "subtractInPlace");
    for (uint64_t i = 0; i < lhs.size(); ++i) {
        lhs[i] -= rhs[i];
    }
}

void compileAndInitialize(const std::vector<Layer*>& layers) {
    for (Layer* layer : layers)
        layer->compile();
    for (Layer* layer : layers)
        layer->initialize();
}

void cleanupLayers(const std::vector<Layer*>& layers) {
    for (Layer* layer : layers)
        layer->cleanup();
}

class CountingPassthrough : public Layer {
   public:
    int forwardCalls = 0;
    int backwardCalls = 0;

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override {
        if (inputTensor.isPresent() && outputTensor.isPresent())
            outputTensor.get().copyFromAsync(inputTensor.get(), stream);
    }

    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override {
        (void)dataIn;
        if (errorIn.isPresent() && errorOut.isPresent())
            errorOut.get().copyFromAsync(errorIn.get(), stream);
    }

    void forward(Optional<Tensor> featureInput, bool validationPass, uint32_t batchSize = 0) override {
        ++forwardCalls;
        Layer::forward(featureInput, validationPass, batchSize);
    }

    void backward(Optional<Tensor> errorInput, uint32_t batchSize = 0) override {
        ++backwardCalls;
        Layer::backward(errorInput, batchSize);
    }
};

class ContextAwareBiasParameter : public Parameter {
   public:
    ContextAwareBiasParameter() : Parameter("bias", false) {}

    size_t seenFeatureInputCount = 0;
    bool sawLhs = false;
    bool sawRhs = false;
    std::vector<unsigned long> rhsDimensions;

    void createStorage(const StorageContext& context) override {
        seenFeatureInputCount = context.featureInputs.size();
        sawLhs = context.namedInputs.contains("lhs");
        sawRhs = context.namedInputs.contains("rhs");
        if (!sawLhs || !sawRhs)
            throw std::runtime_error("ContextAwareBiasParameter expected lhs and rhs named inputs.");

        const Tensor& rhs = context.namedInputs.at("rhs");
        rhsDimensions = rhs.getDimensions();
        storage = Tensor(rhs.getPlacement(), TensorDescriptor(rhs.getDataType(), {rhsDimensions.back()}));
    }

    void createStorage(const Tensor& inputTensor) override {
        throw std::runtime_error("ContextAwareBiasParameter should use the StorageContext overload, not the single-input overload.");
    }
};

class FixedVectorParameter : public Parameter {
   public:
    FixedVectorParameter(std::string name, std::vector<float> initialValues, bool trainable)
        : Parameter(std::move(name), trainable), initialValues(std::move(initialValues)) {}

    void createStorage(const StorageContext& context) override {
        const Tensor& primary = context.primaryInput;
        if (primary.getDataType() != DataType::FP32)
            throw std::runtime_error("FixedVectorParameter currently supports FP32 only.");

        storage =
            Tensor(primary.getPlacement(), TensorDescriptor(primary.getDataType(), {static_cast<unsigned long>(initialValues.size())}));

        Tensor init_h(cpuPlacement, TensorDescriptor(primary.getDataType(), {static_cast<unsigned long>(initialValues.size())}));
        writeCpuTensor(init_h, initialValues);

        Stream initStream = gradientUpdateStream.isPresent()
                                ? gradientUpdateStream.get()
                                : Stream::getMostRecentGradientUpdateStream(primary.getPlacement().getDeviceNum());
        storage.get().copyFromAsync(init_h, initStream);
        Event ready = initStream.putEvent();
        ready.synchronize();
    }

    void createStorage(const Tensor& inputTensor) override { createStorage(StorageContext(inputTensor)); }

   private:
    std::vector<float> initialValues;
};

DynamicExpression buildSingleInputSingleOutputExpression(const TensorPlacement& placement) {
    return DynamicExpression([placement](const DynamicExpression::TensorMap& inputs,
                                         const DynamicExpression::TensorMap& outputs,
                                         Stream& stream) -> DynamicExpressionBuild {
        (void)stream;
        auto x = Expression::input("feature_input");
        auto expressionOutputs = Expression::outputs({{"feature_output", x + 1.5f}});
        return DynamicExpressionBuild{
            std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,
            {},
            outputs,
            {}};
    });
}

DynamicExpression buildTwoInputTwoOutputExpression(const TensorPlacement& placement) {
    return DynamicExpression([placement](const DynamicExpression::TensorMap& inputs,
                                         const DynamicExpression::TensorMap& outputs,
                                         Stream& stream) -> DynamicExpressionBuild {
        (void)stream;
        auto lhs = Expression::input("lhs");
        auto rhs = Expression::input("rhs");
        auto expressionOutputs = Expression::outputs({
            {"sum", lhs + rhs},
            {"diff", lhs - rhs},
        });
        return DynamicExpressionBuild{
            std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,
            {},
            outputs,
            {}};
    });
}

DynamicExpression buildTwoInputParameterizedExpression(const TensorPlacement& placement) {
    return DynamicExpression([placement](const DynamicExpression::TensorMap& inputs,
                                         const DynamicExpression::TensorMap& outputs,
                                         Stream& stream) -> DynamicExpressionBuild {
        (void)stream;
        auto lhs = Expression::input("lhs");
        auto rhs = Expression::input("rhs");
        auto bias = Expression::input("bias");
        auto expressionOutputs = Expression::outputs({{"out", lhs + rhs + bias}});
        return DynamicExpressionBuild{
            std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,
            {},
            outputs,
            {}};
    });
}

DynamicExpression buildSharedScaleBiasTwoInputTwoOutputExpression(const TensorPlacement& placement) {
    return DynamicExpression([placement](const DynamicExpression::TensorMap& inputs,
                                         const DynamicExpression::TensorMap& outputs,
                                         Stream& stream) -> DynamicExpressionBuild {
        (void)stream;
        auto x = Expression::input("x");
        auto y = Expression::input("y");
        auto scale = Expression::input("scale");
        auto bias = Expression::input("bias");
        auto expressionOutputs = Expression::outputs({
            {"out_x", x * scale + bias},
            {"out_y", y * scale + bias},
        });
        return DynamicExpressionBuild{
            std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,
            {},
            outputs,
            {}};
    });
}

DynamicExpression buildThreeInputThreeOutputExpression(const TensorPlacement& placement) {
    return DynamicExpression([placement](const DynamicExpression::TensorMap& inputs,
                                         const DynamicExpression::TensorMap& outputs,
                                         Stream& stream) -> DynamicExpressionBuild {
        (void)stream;
        auto a = Expression::input("a");
        auto b = Expression::input("b");
        auto c = Expression::input("c");
        auto expressionOutputs = Expression::outputs({
            {"out_a", a + 1.0f},
            {"out_b", b + 2.0f},
            {"out_c", c + 3.0f},
        });
        return DynamicExpressionBuild{
            std::make_shared<FusedEquation>(FusedEquation::compile(expressionOutputs.physicalOutputs(), placement.getDeviceNum())),
            inputs,
            {},
            outputs,
            {}};
    });
}

TEST(CustomLayer, SingleInputSingleOutputForwardCompatibility) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor featureIn_h(cpuPlacement, descriptor);
    writeCpuTensor(featureIn_h, {1.0f, 2.0f, 3.0f, -4.0f, 0.5f, 7.0f});

    NetworkInput input(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    CustomLayer custom(buildSingleInputSingleOutputExpression(gpuPlacement), gpuPlacement, {}, true);
    CountingPassthrough sink;

    input.connectToNextLayer(&custom);
    custom.connectToNextLayer(&sink);
    compileAndInitialize({&input, &custom, &sink});

    input.forward(featureIn_h, false, batchSize);

    ASSERT_EQ(sink.forwardCalls, 1);
    ASSERT_TRUE(sink.getFeatureInput().isPresent());
    Tensor result_h = copyTensorToCpu(sink.getFeatureInput().get(), custom.getStreams()[0]);
    const std::vector<float> actual = readCpuTensor(result_h);
    const std::vector<float> expected{2.5f, 3.5f, 4.5f, -2.5f, 2.0f, 8.5f};
    expectAllClose(actual, expected);

    cleanupLayers({&input, &custom, &sink});
}

TEST(CustomLayer, MultiInputMultiOutputWaitsForAllInputsAndRoutesByPortIndex) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor lhs_h(cpuPlacement, descriptor);
    Tensor rhs_h(cpuPlacement, descriptor);
    writeCpuTensor(lhs_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(rhs_h, {10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f});

    NetworkInput lhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    CountingPassthrough lhsBridge;
    CountingPassthrough rhsBridge;
    CustomLayer custom(buildTwoInputTwoOutputExpression(gpuPlacement), {"lhs", "rhs"}, {"sum", "diff"}, gpuPlacement, {}, true);
    CountingPassthrough diffSink;
    CountingPassthrough sumSink;

    lhsIn.connectToNextLayer(&lhsBridge);
    rhsIn.connectToNextLayer(&rhsBridge);
    lhsBridge.connectToNextLayer(&custom, 0, 0);
    rhsBridge.connectToNextLayer(&custom, 0, 1);

    // Intentionally connect the outputs in reverse order to validate port-index routing.
    custom.connectToNextLayer(&diffSink, 1, 0);
    custom.connectToNextLayer(&sumSink, 0, 0);

    compileAndInitialize({&lhsIn, &rhsIn, &lhsBridge, &rhsBridge, &custom, &diffSink, &sumSink});

    lhsIn.forward(lhs_h, false, batchSize);
    ASSERT_EQ(sumSink.forwardCalls, 0);
    ASSERT_EQ(diffSink.forwardCalls, 0);

    rhsIn.forward(rhs_h, false, batchSize);
    ASSERT_EQ(sumSink.forwardCalls, 1);
    ASSERT_EQ(diffSink.forwardCalls, 1);

    Tensor sum_h = copyTensorToCpu(sumSink.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor diff_h = copyTensorToCpu(diffSink.getFeatureInput().get(), custom.getStreams()[0]);

    expectAllClose(readCpuTensor(sum_h), {11.0f, 22.0f, 33.0f, 5.0f, 7.0f, 9.0f});
    expectAllClose(readCpuTensor(diff_h), {-9.0f, -18.0f, -27.0f, 3.0f, 3.0f, 3.0f});

    cleanupLayers({&lhsIn, &rhsIn, &lhsBridge, &rhsBridge, &custom, &diffSink, &sumSink});
}

TEST(CustomLayer, MultiInputMultiOutputBackwardWaitsForAllOutputGradients) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor lhs_h(cpuPlacement, descriptor);
    Tensor rhs_h(cpuPlacement, descriptor);
    writeCpuTensor(lhs_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(rhs_h, {10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f});

    NetworkInput lhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    CountingPassthrough lhsBridge;
    CountingPassthrough rhsBridge;
    CustomLayer custom(buildTwoInputTwoOutputExpression(gpuPlacement), {"lhs", "rhs"}, {"sum", "diff"}, gpuPlacement, {}, false);
    CountingPassthrough sumSink;
    CountingPassthrough diffSink;
    GradientRivet gradientRivetLhs, gradientRivetRhs;

    lhsIn.connectToNextLayer(&gradientRivetLhs);
    gradientRivetLhs.connectToNextLayer(&lhsBridge);

    rhsIn.connectToNextLayer(&gradientRivetRhs);
    gradientRivetRhs.connectToNextLayer(&rhsBridge);

    lhsBridge.connectToNextLayer(&custom, 0, 0);
    rhsBridge.connectToNextLayer(&custom, 0, 1);
    custom.connectToNextLayer(&sumSink, 0, 0);
    custom.connectToNextLayer(&diffSink, 1, 0);

    compileAndInitialize({&lhsIn, &rhsIn, &lhsBridge, &rhsBridge, &custom, &sumSink, &diffSink, &gradientRivetLhs, &gradientRivetRhs});

    lhsIn.forward(lhs_h, false, batchSize);
    rhsIn.forward(rhs_h, false, batchSize);

    ASSERT_TRUE(sumSink.getErrorOutput().isPresent());
    ASSERT_TRUE(diffSink.getErrorOutput().isPresent());

    Tensor sumGrad_h(cpuPlacement, descriptor);
    Tensor diffGrad_h(cpuPlacement, descriptor);
    writeCpuTensor(sumGrad_h, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    writeCpuTensor(diffGrad_h, {2.0f, 2.0f, 2.0f, -1.0f, -1.0f, -1.0f});

    sumSink.getErrorOutput().get().copyFromAsync(sumGrad_h, custom.getStreams()[0]);
    diffSink.getErrorOutput().get().copyFromAsync(diffGrad_h, custom.getStreams()[0]);
    Event gradsReady = custom.getStreams()[0].putEvent();
    gradsReady.synchronize();

    sumSink.backward(sumSink.getErrorOutput(), batchSize);
    ASSERT_EQ(lhsBridge.backwardCalls, 0);
    ASSERT_EQ(rhsBridge.backwardCalls, 0);

    diffSink.backward(diffSink.getErrorOutput(), batchSize);
    ASSERT_EQ(lhsBridge.backwardCalls, 1);
    ASSERT_EQ(rhsBridge.backwardCalls, 1);

    ASSERT_EQ(custom.getErrorOutputs().size(), 2u);
    ASSERT_TRUE(custom.getErrorOutputs()[0].isPresent());
    ASSERT_TRUE(custom.getErrorOutputs()[1].isPresent());

    Tensor lhsGrad_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
    Tensor rhsGrad_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);

    // d(sum)/d(lhs)=1, d(diff)/d(lhs)=1 => lhs_grad = grad_sum + grad_diff
    expectAllClose(readCpuTensor(lhsGrad_h), {3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f});
    // d(sum)/d(rhs)=1, d(diff)/d(rhs)=-1 => rhs_grad = grad_sum - grad_diff
    expectAllClose(readCpuTensor(rhsGrad_h), {-1.0f, -1.0f, -1.0f, 2.0f, 2.0f, 2.0f});

    cleanupLayers({&lhsIn, &rhsIn, &lhsBridge, &rhsBridge, &custom, &sumSink, &diffSink});
}

TEST(CustomLayer, MultiInputMultiOutputTwoPassForwardBackwardCycleResetsState) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    Tensor lhsPass1_h(cpuPlacement, descriptor);
    Tensor rhsPass1_h(cpuPlacement, descriptor);
    writeCpuTensor(lhsPass1_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(rhsPass1_h, {10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f});

    Tensor lhsPass2_h(cpuPlacement, descriptor);
    Tensor rhsPass2_h(cpuPlacement, descriptor);
    writeCpuTensor(lhsPass2_h, {-1.0f, -2.0f, -3.0f, 8.0f, 9.0f, 10.0f});
    writeCpuTensor(rhsPass2_h, {5.0f, 4.0f, 3.0f, -2.0f, -4.0f, -6.0f});

    NetworkInput lhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet gradientRivetLhs, gradientRivetRhs;
    CountingPassthrough lhsBridge;
    CountingPassthrough rhsBridge;
    CustomLayer custom(buildTwoInputTwoOutputExpression(gpuPlacement), {"lhs", "rhs"}, {"sum", "diff"}, gpuPlacement, {}, false);
    CountingPassthrough sumSink;
    CountingPassthrough diffSink;

    lhsIn.connectToNextLayer(&gradientRivetLhs);
    gradientRivetLhs.connectToNextLayer(&lhsBridge);

    rhsIn.connectToNextLayer(&gradientRivetRhs);
    gradientRivetRhs.connectToNextLayer(&rhsBridge);

    lhsBridge.connectToNextLayer(&custom, 0, 0);
    rhsBridge.connectToNextLayer(&custom, 0, 1);
    custom.connectToNextLayer(&sumSink, 0, 0);
    custom.connectToNextLayer(&diffSink, 1, 0);

    compileAndInitialize({&lhsIn, &rhsIn, &gradientRivetLhs, &gradientRivetRhs, &lhsBridge, &rhsBridge, &custom, &sumSink, &diffSink});

    ASSERT_TRUE(sumSink.getErrorOutput().isPresent());
    ASSERT_TRUE(diffSink.getErrorOutput().isPresent());

    auto runPass = [&](const Tensor& lhs_h,
                       const Tensor& rhs_h,
                       const std::vector<float>& sumGradValues,
                       const std::vector<float>& diffGradValues,
                       const std::vector<float>& expectedSum,
                       const std::vector<float>& expectedDiff,
                       const std::vector<float>& expectedLhsGrad,
                       const std::vector<float>& expectedRhsGrad,
                       int expectedForwardCalls,
                       int expectedBackwardCalls) {
        lhsIn.forward(lhs_h, false, batchSize);
        ASSERT_EQ(sumSink.forwardCalls, expectedForwardCalls - 1);
        ASSERT_EQ(diffSink.forwardCalls, expectedForwardCalls - 1);

        rhsIn.forward(rhs_h, false, batchSize);
        ASSERT_EQ(sumSink.forwardCalls, expectedForwardCalls);
        ASSERT_EQ(diffSink.forwardCalls, expectedForwardCalls);

        Tensor sum_h = copyTensorToCpu(sumSink.getFeatureInput().get(), custom.getStreams()[0]);
        Tensor diff_h = copyTensorToCpu(diffSink.getFeatureInput().get(), custom.getStreams()[0]);
        expectAllClose(readCpuTensor(sum_h), expectedSum);
        expectAllClose(readCpuTensor(diff_h), expectedDiff);

        Tensor sumGrad_h(cpuPlacement, descriptor);
        Tensor diffGrad_h(cpuPlacement, descriptor);
        writeCpuTensor(sumGrad_h, sumGradValues);
        writeCpuTensor(diffGrad_h, diffGradValues);

        sumSink.getErrorOutput().get().copyFromAsync(sumGrad_h, custom.getStreams()[0]);
        diffSink.getErrorOutput().get().copyFromAsync(diffGrad_h, custom.getStreams()[0]);
        Event gradsReady = custom.getStreams()[0].putEvent();
        gradsReady.synchronize();

        sumSink.backward(sumSink.getErrorOutput(), batchSize);
        ASSERT_EQ(lhsBridge.backwardCalls, expectedBackwardCalls - 1);
        ASSERT_EQ(rhsBridge.backwardCalls, expectedBackwardCalls - 1);

        diffSink.backward(diffSink.getErrorOutput(), batchSize);
        ASSERT_EQ(lhsBridge.backwardCalls, expectedBackwardCalls);
        ASSERT_EQ(rhsBridge.backwardCalls, expectedBackwardCalls);

        ASSERT_EQ(custom.getErrorOutputs().size(), 2u);
        ASSERT_TRUE(custom.getErrorOutputs()[0].isPresent());
        ASSERT_TRUE(custom.getErrorOutputs()[1].isPresent());

        Tensor lhsGrad_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
        Tensor rhsGrad_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);
        expectAllClose(readCpuTensor(lhsGrad_h), expectedLhsGrad);
        expectAllClose(readCpuTensor(rhsGrad_h), expectedRhsGrad);
    };

    runPass(lhsPass1_h,
            rhsPass1_h,
            {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
            {2.0f, 2.0f, 2.0f, -1.0f, -1.0f, -1.0f},
            {11.0f, 22.0f, 33.0f, 5.0f, 7.0f, 9.0f},
            {-9.0f, -18.0f, -27.0f, 3.0f, 3.0f, 3.0f},
            {3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f},
            {-1.0f, -1.0f, -1.0f, 2.0f, 2.0f, 2.0f},
            1,
            1);

    runPass(lhsPass2_h,
            rhsPass2_h,
            {-2.0f, 0.5f, 1.5f, 3.0f, -4.0f, 2.0f},
            {0.25f, -1.0f, 2.5f, -3.0f, 1.0f, 4.0f},
            {4.0f, 2.0f, 0.0f, 6.0f, 5.0f, 4.0f},
            {-6.0f, -6.0f, -6.0f, 10.0f, 13.0f, 16.0f},
            {-1.75f, -0.5f, 4.0f, 0.0f, -3.0f, 6.0f},
            {-2.25f, 1.5f, -1.0f, 6.0f, -5.0f, -2.0f},
            2,
            2);

    cleanupLayers({&lhsIn, &rhsIn, &gradientRivetLhs, &gradientRivetRhs, &lhsBridge, &rhsBridge, &custom, &sumSink, &diffSink});
}

TEST(CustomLayer, ParameterStorageContextIncludesAllNamedInputs) {
    const uint64_t batchSize = 2;
    const uint64_t features = 4;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor lhs_h(cpuPlacement, descriptor);
    Tensor rhs_h(cpuPlacement, descriptor);
    writeCpuTensor(lhs_h, {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f});
    writeCpuTensor(rhs_h, {10.0f, 20.0f, 30.0f, 40.0f, 1.0f, 2.0f, 3.0f, 4.0f});

    auto bias = std::make_shared<ContextAwareBiasParameter>();

    NetworkInput lhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    CountingPassthrough lhsBridge;
    CountingPassthrough rhsBridge;
    CustomLayer custom(buildTwoInputParameterizedExpression(gpuPlacement), {"lhs", "rhs"}, {"out"}, gpuPlacement, {bias}, true);
    CountingPassthrough sink;

    lhsIn.connectToNextLayer(&lhsBridge);
    rhsIn.connectToNextLayer(&rhsBridge);
    lhsBridge.connectToNextLayer(&custom, 0, 0);
    rhsBridge.connectToNextLayer(&custom, 0, 1);
    custom.connectToNextLayer(&sink, 0, 0);

    compileAndInitialize({&lhsIn, &rhsIn, &lhsBridge, &rhsBridge, &custom, &sink});

    ASSERT_EQ(bias->seenFeatureInputCount, 2u);
    ASSERT_TRUE(bias->sawLhs);
    ASSERT_TRUE(bias->sawRhs);
    ASSERT_EQ(bias->rhsDimensions, descriptor.getDimensions());
    ASSERT_TRUE(bias->getStorage().isPresent());
    ASSERT_EQ(bias->getStorage().get().getDimensions(), (std::vector<unsigned long>{features}));

    cleanupLayers({&lhsIn, &rhsIn, &lhsBridge, &rhsBridge, &custom, &sink});
}

TEST(CustomLayer, MultiInputMultiOutputTwoPassForwardBackwardCycleResetsStateWithReversedSecondPassArrivalOrder) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    Tensor lhsPass1_h(cpuPlacement, descriptor);
    Tensor rhsPass1_h(cpuPlacement, descriptor);
    writeCpuTensor(lhsPass1_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(rhsPass1_h, {10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f});

    Tensor lhsPass2_h(cpuPlacement, descriptor);
    Tensor rhsPass2_h(cpuPlacement, descriptor);
    writeCpuTensor(lhsPass2_h, {-1.0f, -2.0f, -3.0f, 8.0f, 9.0f, 10.0f});
    writeCpuTensor(rhsPass2_h, {5.0f, 4.0f, 3.0f, -2.0f, -4.0f, -6.0f});

    NetworkInput lhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhsIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet gradientRivetLhs, gradientRivetRhs;
    CountingPassthrough lhsBridge;
    CountingPassthrough rhsBridge;
    CustomLayer custom(buildTwoInputTwoOutputExpression(gpuPlacement), {"lhs", "rhs"}, {"sum", "diff"}, gpuPlacement, {}, false);
    CountingPassthrough sumSink;
    CountingPassthrough diffSink;

    lhsIn.connectToNextLayer(&gradientRivetLhs);
    gradientRivetLhs.connectToNextLayer(&lhsBridge);

    rhsIn.connectToNextLayer(&gradientRivetRhs);
    gradientRivetRhs.connectToNextLayer(&rhsBridge);

    lhsBridge.connectToNextLayer(&custom, 0, 0);
    rhsBridge.connectToNextLayer(&custom, 0, 1);
    custom.connectToNextLayer(&sumSink, 0, 0);
    custom.connectToNextLayer(&diffSink, 1, 0);

    compileAndInitialize({&lhsIn, &rhsIn, &gradientRivetLhs, &gradientRivetRhs, &lhsBridge, &rhsBridge, &custom, &sumSink, &diffSink});

    ASSERT_TRUE(sumSink.getErrorOutput().isPresent());
    ASSERT_TRUE(diffSink.getErrorOutput().isPresent());

    // Pass 1 in the ordinary arrival order.
    lhsIn.forward(lhsPass1_h, false, batchSize);
    ASSERT_EQ(sumSink.forwardCalls, 0);
    ASSERT_EQ(diffSink.forwardCalls, 0);

    rhsIn.forward(rhsPass1_h, false, batchSize);
    ASSERT_EQ(sumSink.forwardCalls, 1);
    ASSERT_EQ(diffSink.forwardCalls, 1);

    Tensor sumPass1_h = copyTensorToCpu(sumSink.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor diffPass1_h = copyTensorToCpu(diffSink.getFeatureInput().get(), custom.getStreams()[0]);
    expectAllClose(readCpuTensor(sumPass1_h), {11.0f, 22.0f, 33.0f, 5.0f, 7.0f, 9.0f});
    expectAllClose(readCpuTensor(diffPass1_h), {-9.0f, -18.0f, -27.0f, 3.0f, 3.0f, 3.0f});

    Tensor sumGradPass1_h(cpuPlacement, descriptor);
    Tensor diffGradPass1_h(cpuPlacement, descriptor);
    writeCpuTensor(sumGradPass1_h, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    writeCpuTensor(diffGradPass1_h, {2.0f, 2.0f, 2.0f, -1.0f, -1.0f, -1.0f});

    sumSink.getErrorOutput().get().copyFromAsync(sumGradPass1_h, custom.getStreams()[0]);
    diffSink.getErrorOutput().get().copyFromAsync(diffGradPass1_h, custom.getStreams()[0]);
    Event gradsPass1Ready = custom.getStreams()[0].putEvent();
    gradsPass1Ready.synchronize();

    sumSink.backward(sumSink.getErrorOutput(), batchSize);
    ASSERT_EQ(lhsBridge.backwardCalls, 0);
    ASSERT_EQ(rhsBridge.backwardCalls, 0);

    diffSink.backward(diffSink.getErrorOutput(), batchSize);
    ASSERT_EQ(lhsBridge.backwardCalls, 1);
    ASSERT_EQ(rhsBridge.backwardCalls, 1);

    Tensor lhsGradPass1_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
    Tensor rhsGradPass1_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);
    expectAllClose(readCpuTensor(lhsGradPass1_h), {3.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f});
    expectAllClose(readCpuTensor(rhsGradPass1_h), {-1.0f, -1.0f, -1.0f, 2.0f, 2.0f, 2.0f});

    // Pass 2 reverses both forward and backward arrival order to stress the reset logic.
    rhsIn.forward(rhsPass2_h, false, batchSize);
    ASSERT_EQ(sumSink.forwardCalls, 1);
    ASSERT_EQ(diffSink.forwardCalls, 1);

    lhsIn.forward(lhsPass2_h, false, batchSize);
    ASSERT_EQ(sumSink.forwardCalls, 2);
    ASSERT_EQ(diffSink.forwardCalls, 2);

    Tensor sumPass2_h = copyTensorToCpu(sumSink.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor diffPass2_h = copyTensorToCpu(diffSink.getFeatureInput().get(), custom.getStreams()[0]);
    expectAllClose(readCpuTensor(sumPass2_h), {4.0f, 2.0f, 0.0f, 6.0f, 5.0f, 4.0f});
    expectAllClose(readCpuTensor(diffPass2_h), {-6.0f, -6.0f, -6.0f, 10.0f, 13.0f, 16.0f});

    Tensor sumGradPass2_h(cpuPlacement, descriptor);
    Tensor diffGradPass2_h(cpuPlacement, descriptor);
    writeCpuTensor(sumGradPass2_h, {-2.0f, 0.5f, 1.5f, 3.0f, -4.0f, 2.0f});
    writeCpuTensor(diffGradPass2_h, {0.25f, -1.0f, 2.5f, -3.0f, 1.0f, 4.0f});

    sumSink.getErrorOutput().get().copyFromAsync(sumGradPass2_h, custom.getStreams()[0]);
    diffSink.getErrorOutput().get().copyFromAsync(diffGradPass2_h, custom.getStreams()[0]);
    Event gradsPass2Ready = custom.getStreams()[0].putEvent();
    gradsPass2Ready.synchronize();

    diffSink.backward(diffSink.getErrorOutput(), batchSize);
    ASSERT_EQ(lhsBridge.backwardCalls, 1);
    ASSERT_EQ(rhsBridge.backwardCalls, 1);

    sumSink.backward(sumSink.getErrorOutput(), batchSize);
    ASSERT_EQ(lhsBridge.backwardCalls, 2);
    ASSERT_EQ(rhsBridge.backwardCalls, 2);

    Tensor lhsGradPass2_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
    Tensor rhsGradPass2_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);
    expectAllClose(readCpuTensor(lhsGradPass2_h), {-1.75f, -0.5f, 4.0f, 0.0f, -3.0f, 6.0f});
    expectAllClose(readCpuTensor(rhsGradPass2_h), {-2.25f, 1.5f, -1.0f, 6.0f, -5.0f, -2.0f});

    cleanupLayers({&lhsIn, &rhsIn, &gradientRivetLhs, &gradientRivetRhs, &lhsBridge, &rhsBridge, &custom, &sumSink, &diffSink});
}

TEST(CustomLayer, ThreeInputThreeOutputSupportsMultipleSameShapeConnectionsByDifferentLayers) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor a_h(cpuPlacement, descriptor);
    Tensor b_h(cpuPlacement, descriptor);
    Tensor c_h(cpuPlacement, descriptor);
    writeCpuTensor(a_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(b_h, {10.0f, 20.0f, 30.0f, -1.0f, -2.0f, -3.0f});
    writeCpuTensor(c_h, {7.0f, 8.0f, 9.0f, 0.5f, 1.5f, 2.5f});

    NetworkInput aIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput bIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput cIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    CountingPassthrough aBridge;
    CountingPassthrough bBridge;
    CountingPassthrough cBridge;
    CustomLayer custom(
        buildThreeInputThreeOutputExpression(gpuPlacement), {"a", "b", "c"}, {"out_a", "out_b", "out_c"}, gpuPlacement, {}, true);
    CountingPassthrough sinkB;
    CountingPassthrough sinkC;
    CountingPassthrough sinkA;

    aIn.connectToNextLayer(&aBridge);
    bIn.connectToNextLayer(&bBridge);
    cIn.connectToNextLayer(&cBridge);

    aBridge.connectToNextLayer(&custom, 0, 0);
    bBridge.connectToNextLayer(&custom, 0, 1);
    cBridge.connectToNextLayer(&custom, 0, 2);

    // Connect outputs in a different order than declaration to validate routing by port index.
    custom.connectToNextLayer(&sinkB, 1, 0);
    custom.connectToNextLayer(&sinkC, 2, 0);
    custom.connectToNextLayer(&sinkA, 0, 0);

    compileAndInitialize({&aIn, &bIn, &cIn, &aBridge, &bBridge, &cBridge, &custom, &sinkA, &sinkB, &sinkC});

    aIn.forward(a_h, false, batchSize);
    bIn.forward(b_h, false, batchSize);
    ASSERT_EQ(sinkA.forwardCalls, 0);
    ASSERT_EQ(sinkB.forwardCalls, 0);
    ASSERT_EQ(sinkC.forwardCalls, 0);

    cIn.forward(c_h, false, batchSize);
    ASSERT_EQ(sinkA.forwardCalls, 1);
    ASSERT_EQ(sinkB.forwardCalls, 1);
    ASSERT_EQ(sinkC.forwardCalls, 1);

    Tensor outA_h = copyTensorToCpu(sinkA.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor outB_h = copyTensorToCpu(sinkB.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor outC_h = copyTensorToCpu(sinkC.getFeatureInput().get(), custom.getStreams()[0]);

    expectAllClose(readCpuTensor(outA_h), {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    expectAllClose(readCpuTensor(outB_h), {12.0f, 22.0f, 32.0f, 1.0f, 0.0f, -1.0f});
    expectAllClose(readCpuTensor(outC_h), {10.0f, 11.0f, 12.0f, 3.5f, 4.5f, 5.5f});

    cleanupLayers({&aIn, &bIn, &cIn, &aBridge, &bBridge, &cBridge, &custom, &sinkA, &sinkB, &sinkC});
}

TEST(CustomLayer, SharedNamedParametersDriveMultipleInputAndOutputConnections) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor x_h(cpuPlacement, descriptor);
    Tensor y_h(cpuPlacement, descriptor);
    writeCpuTensor(x_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(y_h, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});

    auto scale = std::make_shared<FixedVectorParameter>("scale", std::vector<float>{2.0f, 3.0f, 4.0f}, false);
    auto bias = std::make_shared<FixedVectorParameter>("bias", std::vector<float>{10.0f, 20.0f, 30.0f}, false);

    NetworkInput xIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput yIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    CountingPassthrough xBridge;
    CountingPassthrough yBridge;
    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, true);
    CountingPassthrough outYSink;
    CountingPassthrough outXSink;

    xIn.connectToNextLayer(&xBridge);
    yIn.connectToNextLayer(&yBridge);
    xBridge.connectToNextLayer(&custom, 0, 0);
    yBridge.connectToNextLayer(&custom, 0, 1);
    custom.connectToNextLayer(&outYSink, 1, 0);
    custom.connectToNextLayer(&outXSink, 0, 0);

    compileAndInitialize({&xIn, &yIn, &xBridge, &yBridge, &custom, &outYSink, &outXSink});

    xIn.forward(x_h, false, batchSize);
    ASSERT_EQ(outXSink.forwardCalls, 0);
    ASSERT_EQ(outYSink.forwardCalls, 0);

    yIn.forward(y_h, false, batchSize);
    ASSERT_EQ(outXSink.forwardCalls, 1);
    ASSERT_EQ(outYSink.forwardCalls, 1);

    Tensor outX_h = copyTensorToCpu(outXSink.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor outY_h = copyTensorToCpu(outYSink.getFeatureInput().get(), custom.getStreams()[0]);

    expectAllClose(readCpuTensor(outX_h), {12.0f, 26.0f, 42.0f, 18.0f, 35.0f, 54.0f});
    expectAllClose(readCpuTensor(outY_h), {8.0f, 20.0f, 34.0f, 14.0f, 29.0f, 46.0f});

    cleanupLayers({&xIn, &yIn, &xBridge, &yBridge, &custom, &outYSink, &outXSink});
}

TEST(CustomLayer, SharedNamedParameterGradientsAccumulateAcrossMultipleConnections) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor x_h(cpuPlacement, descriptor);
    Tensor y_h(cpuPlacement, descriptor);
    writeCpuTensor(x_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(y_h, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});

    auto scale = std::make_shared<FixedVectorParameter>("scale", std::vector<float>{2.0f, 3.0f, 4.0f}, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", std::vector<float>{10.0f, 20.0f, 30.0f}, true);
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(101, 0.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(102, 0.0f, 0.0f, 0.0f, false)));

    NetworkInput xIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput yIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet gradientRivetX, gradientRivetY;
    CountingPassthrough xBridge;
    CountingPassthrough yBridge;
    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);
    CountingPassthrough outXSink;
    CountingPassthrough outYSink;

    xIn.connectToNextLayer(&gradientRivetX);
    gradientRivetX.connectToNextLayer(&xBridge);
    yIn.connectToNextLayer(&gradientRivetY);
    gradientRivetY.connectToNextLayer(&yBridge);

    xBridge.connectToNextLayer(&custom, 0, 0);
    yBridge.connectToNextLayer(&custom, 0, 1);
    custom.connectToNextLayer(&outXSink, 0, 0);
    custom.connectToNextLayer(&outYSink, 1, 0);

    compileAndInitialize({&xIn, &yIn, &gradientRivetX, &gradientRivetY, &xBridge, &yBridge, &custom, &outXSink, &outYSink});

    xIn.forward(x_h, false, batchSize);
    yIn.forward(y_h, false, batchSize);

    ASSERT_TRUE(outXSink.getErrorOutput().isPresent());
    ASSERT_TRUE(outYSink.getErrorOutput().isPresent());
    ASSERT_TRUE(custom.getGradientUpdateStream().isPresent());
    ASSERT_TRUE(scale->getOptimizer()->getWeightsGradient().isPresent());
    ASSERT_TRUE(bias->getOptimizer()->getWeightsGradient().isPresent());

    Tensor gradOutX_h(cpuPlacement, descriptor);
    Tensor gradOutY_h(cpuPlacement, descriptor);
    writeCpuTensor(gradOutX_h, {1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f});
    writeCpuTensor(gradOutY_h, {-2.0f, 3.0f, 1.0f, 0.0f, -1.0f, 2.0f});

    outXSink.getErrorOutput().get().copyFromAsync(gradOutX_h, custom.getStreams()[0]);
    outYSink.getErrorOutput().get().copyFromAsync(gradOutY_h, custom.getStreams()[0]);
    Event gradsReady = custom.getStreams()[0].putEvent();
    gradsReady.synchronize();

    outXSink.backward(outXSink.getErrorOutput(), batchSize);
    ASSERT_EQ(xBridge.backwardCalls, 0);
    ASSERT_EQ(yBridge.backwardCalls, 0);

    outYSink.backward(outYSink.getErrorOutput(), batchSize);
    ASSERT_EQ(xBridge.backwardCalls, 1);
    ASSERT_EQ(yBridge.backwardCalls, 1);

    Tensor xGrad_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
    Tensor yGrad_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);
    Tensor scaleGrad_h = copyTensorToCpu(scale->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());
    Tensor biasGrad_h = copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());

    expectAllClose(readCpuTensor(xGrad_h), {2.0f, 0.0f, -4.0f, 4.0f, 3.0f, 2.0f});
    expectAllClose(readCpuTensor(yGrad_h), {-4.0f, 9.0f, 4.0f, 0.0f, -3.0f, 8.0f});
    expectAllClose(readCpuTensor(scaleGrad_h), {11.0f, 2.0f, 9.0f});
    expectAllClose(readCpuTensor(biasGrad_h), {1.0f, 3.0f, 2.5f});

    cleanupLayers({&xIn, &yIn, &gradientRivetX, &gradientRivetY, &xBridge, &yBridge, &custom, &outXSink, &outYSink});
}

TEST(CustomLayer, SharedNamedParameterGradientsAccumulateAcrossMultipleConnectionsWithNonZeroSgdLearningRate) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor x_h(cpuPlacement, descriptor);
    Tensor y_h(cpuPlacement, descriptor);
    writeCpuTensor(x_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(y_h, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});

    auto scale = std::make_shared<FixedVectorParameter>("scale", std::vector<float>{2.0f, 3.0f, 4.0f}, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", std::vector<float>{10.0f, 20.0f, 30.0f}, true);
    // step = learningRate / (batchSize * Loss::lossScalingFactor) = 8 / (2 * 4) = 1
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(201, 8.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(202, 8.0f, 0.0f, 0.0f, false)));

    NetworkInput xIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput yIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet gradientRivetX, gradientRivetY;
    CountingPassthrough xBridge;
    CountingPassthrough yBridge;
    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);
    CountingPassthrough outXSink;
    CountingPassthrough outYSink;

    xIn.connectToNextLayer(&gradientRivetX);
    gradientRivetX.connectToNextLayer(&xBridge);
    yIn.connectToNextLayer(&gradientRivetY);
    gradientRivetY.connectToNextLayer(&yBridge);

    xBridge.connectToNextLayer(&custom, 0, 0);
    yBridge.connectToNextLayer(&custom, 0, 1);
    custom.connectToNextLayer(&outXSink, 0, 0);
    custom.connectToNextLayer(&outYSink, 1, 0);

    compileAndInitialize({&xIn, &yIn, &gradientRivetX, &gradientRivetY, &xBridge, &yBridge, &custom, &outXSink, &outYSink});

    xIn.forward(x_h, false, batchSize);
    yIn.forward(y_h, false, batchSize);

    ASSERT_TRUE(outXSink.getErrorOutput().isPresent());
    ASSERT_TRUE(outYSink.getErrorOutput().isPresent());
    ASSERT_TRUE(custom.getGradientUpdateStream().isPresent());
    ASSERT_TRUE(scale->getOptimizer()->getWeightsGradient().isPresent());
    ASSERT_TRUE(bias->getOptimizer()->getWeightsGradient().isPresent());
    ASSERT_TRUE(scale->getStorage().isPresent());
    ASSERT_TRUE(bias->getStorage().isPresent());

    Tensor gradOutX_h(cpuPlacement, descriptor);
    Tensor gradOutY_h(cpuPlacement, descriptor);
    writeCpuTensor(gradOutX_h, {1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f});
    writeCpuTensor(gradOutY_h, {-2.0f, 3.0f, 1.0f, 0.0f, -1.0f, 2.0f});

    outXSink.getErrorOutput().get().copyFromAsync(gradOutX_h, custom.getStreams()[0]);
    outYSink.getErrorOutput().get().copyFromAsync(gradOutY_h, custom.getStreams()[0]);
    Event gradsReady = custom.getStreams()[0].putEvent();
    gradsReady.synchronize();

    outXSink.backward(outXSink.getErrorOutput(), batchSize);
    ASSERT_EQ(xBridge.backwardCalls, 0);
    ASSERT_EQ(yBridge.backwardCalls, 0);

    outYSink.backward(outYSink.getErrorOutput(), batchSize);
    ASSERT_EQ(xBridge.backwardCalls, 1);
    ASSERT_EQ(yBridge.backwardCalls, 1);

    Tensor xGrad_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
    Tensor yGrad_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);
    Tensor scaleGrad_h = copyTensorToCpu(scale->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());
    Tensor biasGrad_h = copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());
    Tensor scaleWeights_h = copyTensorToCpu(scale->getStorage().get(), custom.getGradientUpdateStream().get());
    Tensor biasWeights_h = copyTensorToCpu(bias->getStorage().get(), custom.getGradientUpdateStream().get());

    expectAllClose(readCpuTensor(xGrad_h), {2.0f, 0.0f, -4.0f, 4.0f, 3.0f, 2.0f});
    expectAllClose(readCpuTensor(yGrad_h), {-4.0f, 9.0f, 4.0f, 0.0f, -3.0f, 8.0f});
    expectAllClose(readCpuTensor(scaleGrad_h), {11.0f, 2.0f, 9.0f});
    expectAllClose(readCpuTensor(biasGrad_h), {1.0f, 3.0f, 2.5f});
    expectAllClose(readCpuTensor(scaleWeights_h), {-9.0f, 1.0f, -5.0f});
    expectAllClose(readCpuTensor(biasWeights_h), {9.0f, 17.0f, 27.5f});

    cleanupLayers({&xIn, &yIn, &gradientRivetX, &gradientRivetY, &xBridge, &yBridge, &custom, &outXSink, &outYSink});
}

TEST(CustomLayer, SharedNamedParameterGradientsAccumulateAcrossMultipleConnectionsWithNonZeroSgdLearningRateAcrossTwoPasses) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});
    Tensor x_h(cpuPlacement, descriptor);
    Tensor y_h(cpuPlacement, descriptor);
    writeCpuTensor(x_h, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    writeCpuTensor(y_h, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});

    auto scale = std::make_shared<FixedVectorParameter>("scale", std::vector<float>{2.0f, 3.0f, 4.0f}, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", std::vector<float>{10.0f, 20.0f, 30.0f}, true);
    // step = learningRate / (batchSize * Loss::lossScalingFactor) = 8 / (2 * 4) = 1
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(301, 8.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(302, 8.0f, 0.0f, 0.0f, false)));

    NetworkInput xIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput yIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet gradientRivetX, gradientRivetY;
    CountingPassthrough xBridge;
    CountingPassthrough yBridge;
    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);
    CountingPassthrough outXSink;
    CountingPassthrough outYSink;

    xIn.connectToNextLayer(&gradientRivetX);
    gradientRivetX.connectToNextLayer(&xBridge);
    yIn.connectToNextLayer(&gradientRivetY);
    gradientRivetY.connectToNextLayer(&yBridge);

    xBridge.connectToNextLayer(&custom, 0, 0);
    yBridge.connectToNextLayer(&custom, 0, 1);
    custom.connectToNextLayer(&outXSink, 0, 0);
    custom.connectToNextLayer(&outYSink, 1, 0);

    compileAndInitialize({&xIn, &yIn, &gradientRivetX, &gradientRivetY, &xBridge, &yBridge, &custom, &outXSink, &outYSink});

    ASSERT_TRUE(outXSink.getErrorOutput().isPresent());
    ASSERT_TRUE(outYSink.getErrorOutput().isPresent());
    ASSERT_TRUE(custom.getGradientUpdateStream().isPresent());
    ASSERT_TRUE(scale->getOptimizer()->getWeightsGradient().isPresent());
    ASSERT_TRUE(bias->getOptimizer()->getWeightsGradient().isPresent());
    ASSERT_TRUE(scale->getStorage().isPresent());
    ASSERT_TRUE(bias->getStorage().isPresent());

    Tensor gradOutX_h(cpuPlacement, descriptor);
    Tensor gradOutY_h(cpuPlacement, descriptor);
    writeCpuTensor(gradOutX_h, {1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f});
    writeCpuTensor(gradOutY_h, {-2.0f, 3.0f, 1.0f, 0.0f, -1.0f, 2.0f});

    auto runPass = [&](const std::vector<float>& expectedOutX,
                       const std::vector<float>& expectedOutY,
                       const std::vector<float>& expectedXGrad,
                       const std::vector<float>& expectedYGrad,
                       const std::vector<float>& expectedScaleGrad,
                       const std::vector<float>& expectedBiasGrad,
                       const std::vector<float>& expectedScaleWeights,
                       const std::vector<float>& expectedBiasWeights,
                       int expectedForwardCalls,
                       int expectedBackwardCalls) {
        xIn.forward(x_h, false, batchSize);
        ASSERT_EQ(outXSink.forwardCalls, expectedForwardCalls - 1);
        ASSERT_EQ(outYSink.forwardCalls, expectedForwardCalls - 1);

        yIn.forward(y_h, false, batchSize);
        ASSERT_EQ(outXSink.forwardCalls, expectedForwardCalls);
        ASSERT_EQ(outYSink.forwardCalls, expectedForwardCalls);

        Tensor outX_h = copyTensorToCpu(outXSink.getFeatureInput().get(), custom.getStreams()[0]);
        Tensor outY_h = copyTensorToCpu(outYSink.getFeatureInput().get(), custom.getStreams()[0]);
        expectAllClose(readCpuTensor(outX_h), expectedOutX);
        expectAllClose(readCpuTensor(outY_h), expectedOutY);

        outXSink.getErrorOutput().get().copyFromAsync(gradOutX_h, custom.getStreams()[0]);
        outYSink.getErrorOutput().get().copyFromAsync(gradOutY_h, custom.getStreams()[0]);
        Event gradsReady = custom.getStreams()[0].putEvent();
        gradsReady.synchronize();

        outXSink.backward(outXSink.getErrorOutput(), batchSize);
        ASSERT_EQ(xBridge.backwardCalls, expectedBackwardCalls - 1);
        ASSERT_EQ(yBridge.backwardCalls, expectedBackwardCalls - 1);

        outYSink.backward(outYSink.getErrorOutput(), batchSize);
        ASSERT_EQ(xBridge.backwardCalls, expectedBackwardCalls);
        ASSERT_EQ(yBridge.backwardCalls, expectedBackwardCalls);

        Tensor xGrad_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
        Tensor yGrad_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);
        Tensor scaleGrad_h = copyTensorToCpu(scale->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());
        Tensor biasGrad_h = copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());
        Tensor scaleWeights_h = copyTensorToCpu(scale->getStorage().get(), custom.getGradientUpdateStream().get());
        Tensor biasWeights_h = copyTensorToCpu(bias->getStorage().get(), custom.getGradientUpdateStream().get());

        expectAllClose(readCpuTensor(xGrad_h), expectedXGrad);
        expectAllClose(readCpuTensor(yGrad_h), expectedYGrad);
        expectAllClose(readCpuTensor(scaleGrad_h), expectedScaleGrad);
        expectAllClose(readCpuTensor(biasGrad_h), expectedBiasGrad);
        expectAllClose(readCpuTensor(scaleWeights_h), expectedScaleWeights);
        expectAllClose(readCpuTensor(biasWeights_h), expectedBiasWeights);
    };

    runPass({12.0f, 26.0f, 42.0f, 18.0f, 35.0f, 54.0f},
            {8.0f, 20.0f, 34.0f, 14.0f, 29.0f, 46.0f},
            {2.0f, 0.0f, -4.0f, 4.0f, 3.0f, 2.0f},
            {-4.0f, 9.0f, 4.0f, 0.0f, -3.0f, 8.0f},
            {11.0f, 2.0f, 9.0f},
            {1.0f, 3.0f, 2.5f},
            {-9.0f, 1.0f, -5.0f},
            {9.0f, 17.0f, 27.5f},
            1,
            1);

    runPass({0.0f, 19.0f, 12.5f, -27.0f, 22.0f, -2.5f},
            {18.0f, 17.0f, 22.5f, -9.0f, 20.0f, 7.5f},
            {-9.0f, 0.0f, 5.0f, -18.0f, 1.0f, -2.5f},
            {18.0f, 3.0f, -5.0f, 0.0f, -1.0f, -10.0f},
            {11.0f, 2.0f, 9.0f},
            {1.0f, 3.0f, 2.5f},
            {-20.0f, -1.0f, -14.0f},
            {8.0f, 14.0f, 25.0f},
            2,
            2);

    cleanupLayers({&xIn, &yIn, &gradientRivetX, &gradientRivetY, &xBridge, &yBridge, &custom, &outXSink, &outYSink});
}

TEST(CustomLayer, MultiInterfaceTwoInputTwoOutputForwardWaitsIndependentlyAndRoutesByEncodedPort) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> lhs0Values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> rhs0Values{10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> lhs1Values{-1.0f, -2.0f, -3.0f, 8.0f, 9.0f, 10.0f};
    const std::vector<float> rhs1Values{5.0f, 4.0f, 3.0f, -2.0f, -4.0f, -6.0f};

    Tensor lhs0_h(cpuPlacement, descriptor);
    Tensor rhs0_h(cpuPlacement, descriptor);
    Tensor lhs1_h(cpuPlacement, descriptor);
    Tensor rhs1_h(cpuPlacement, descriptor);
    writeCpuTensor(lhs0_h, lhs0Values);
    writeCpuTensor(rhs0_h, rhs0Values);
    writeCpuTensor(lhs1_h, lhs1Values);
    writeCpuTensor(rhs1_h, rhs1Values);

    NetworkInput lhs0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput lhs1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    CountingPassthrough lhs0Bridge;
    CountingPassthrough rhs0Bridge;
    CountingPassthrough lhs1Bridge;
    CountingPassthrough rhs1Bridge;
    CustomLayer custom(buildTwoInputTwoOutputExpression(gpuPlacement), {"lhs", "rhs"}, {"sum", "diff"}, gpuPlacement, {}, true);
    CountingPassthrough sum0Sink;
    CountingPassthrough diff0Sink;
    CountingPassthrough sum1Sink;
    CountingPassthrough diff1Sink;

    lhs0In.connectToNextLayer(&lhs0Bridge);
    rhs0In.connectToNextLayer(&rhs0Bridge);
    lhs1In.connectToNextLayer(&lhs1Bridge);
    rhs1In.connectToNextLayer(&rhs1Bridge);

    // Application 0: encoded inputs 0=lhs, 1=rhs.
    lhs0Bridge.connectToNextLayer(&custom, 0, 0);
    rhs0Bridge.connectToNextLayer(&custom, 0, 1);
    // Application 1: encoded inputs 2=lhs, 3=rhs.
    lhs1Bridge.connectToNextLayer(&custom, 0, 2);
    rhs1Bridge.connectToNextLayer(&custom, 0, 3);

    // Encoded outputs: app0 sum/diff are 0/1, app1 sum/diff are 2/3.
    custom.connectToNextLayer(&sum0Sink, 0, 0);
    custom.connectToNextLayer(&diff0Sink, 1, 0);
    custom.connectToNextLayer(&sum1Sink, 2, 0);
    custom.connectToNextLayer(&diff1Sink, 3, 0);

    compileAndInitialize({&lhs0In,
                          &rhs0In,
                          &lhs1In,
                          &rhs1In,
                          &lhs0Bridge,
                          &rhs0Bridge,
                          &lhs1Bridge,
                          &rhs1Bridge,
                          &custom,
                          &sum0Sink,
                          &diff0Sink,
                          &sum1Sink,
                          &diff1Sink});

    lhs0In.forward(lhs0_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 0);
    ASSERT_EQ(diff0Sink.forwardCalls, 0);
    ASSERT_EQ(sum1Sink.forwardCalls, 0);
    ASSERT_EQ(diff1Sink.forwardCalls, 0);

    rhs1In.forward(rhs1_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 0);
    ASSERT_EQ(diff0Sink.forwardCalls, 0);
    ASSERT_EQ(sum1Sink.forwardCalls, 0);
    ASSERT_EQ(diff1Sink.forwardCalls, 0);

    rhs0In.forward(rhs0_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 1);
    ASSERT_EQ(diff0Sink.forwardCalls, 1);
    ASSERT_EQ(sum1Sink.forwardCalls, 0);
    ASSERT_EQ(diff1Sink.forwardCalls, 0);

    lhs1In.forward(lhs1_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 1);
    ASSERT_EQ(diff0Sink.forwardCalls, 1);
    ASSERT_EQ(sum1Sink.forwardCalls, 1);
    ASSERT_EQ(diff1Sink.forwardCalls, 1);

    Tensor sum0_h = copyTensorToCpu(sum0Sink.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor diff0_h = copyTensorToCpu(diff0Sink.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor sum1_h = copyTensorToCpu(sum1Sink.getFeatureInput().get(), custom.getStreams()[2]);
    Tensor diff1_h = copyTensorToCpu(diff1Sink.getFeatureInput().get(), custom.getStreams()[2]);

    expectAllClose(readCpuTensor(sum0_h), {11.0f, 22.0f, 33.0f, 5.0f, 7.0f, 9.0f});
    expectAllClose(readCpuTensor(diff0_h), {-9.0f, -18.0f, -27.0f, 3.0f, 3.0f, 3.0f});
    expectAllClose(readCpuTensor(sum1_h), {4.0f, 2.0f, 0.0f, 6.0f, 5.0f, 4.0f});
    expectAllClose(readCpuTensor(diff1_h), {-6.0f, -6.0f, -6.0f, 10.0f, 13.0f, 16.0f});

    cleanupLayers({&lhs0In,
                   &rhs0In,
                   &lhs1In,
                   &rhs1In,
                   &lhs0Bridge,
                   &rhs0Bridge,
                   &lhs1Bridge,
                   &rhs1Bridge,
                   &custom,
                   &sum0Sink,
                   &diff0Sink,
                   &sum1Sink,
                   &diff1Sink});
}

TEST(CustomLayer, MultiInterfaceSharedInputTensorMarksAliasedInterfacesAndFiresIndependently) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> sharedValues{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> rhs0Values{10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> rhs1Values{5.0f, 4.0f, 3.0f, -2.0f, -4.0f, -6.0f};

    Tensor shared_h(cpuPlacement, descriptor);
    Tensor rhs0_h(cpuPlacement, descriptor);
    Tensor rhs1_h(cpuPlacement, descriptor);
    writeCpuTensor(shared_h, sharedValues);
    writeCpuTensor(rhs0_h, rhs0Values);
    writeCpuTensor(rhs1_h, rhs1Values);

    NetworkInput sharedIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    TensorFanout sharedFanout;
    CountingPassthrough rhs0Bridge;
    CountingPassthrough rhs1Bridge;
    CustomLayer custom(buildTwoInputTwoOutputExpression(gpuPlacement), {"lhs", "rhs"}, {"sum", "diff"}, gpuPlacement, {}, true);
    CountingPassthrough sum0Sink;
    CountingPassthrough diff0Sink;
    CountingPassthrough sum1Sink;
    CountingPassthrough diff1Sink;

    sharedIn.connectToNextLayer(&sharedFanout);
    rhs0In.connectToNextLayer(&rhs0Bridge);
    rhs1In.connectToNextLayer(&rhs1Bridge);

    // The same physical tensor participates in lhs for two otherwise-distinct complete interfaces.
    sharedFanout.connectToNextLayer(&custom, 0, 0);
    rhs0Bridge.connectToNextLayer(&custom, 0, 1);
    sharedFanout.connectToNextLayer(&custom, 0, 2);
    rhs1Bridge.connectToNextLayer(&custom, 0, 3);

    custom.connectToNextLayer(&sum0Sink, 0, 0);
    custom.connectToNextLayer(&diff0Sink, 1, 0);
    custom.connectToNextLayer(&sum1Sink, 2, 0);
    custom.connectToNextLayer(&diff1Sink, 3, 0);

    compileAndInitialize(
        {&sharedIn, &rhs0In, &rhs1In, &sharedFanout, &rhs0Bridge, &rhs1Bridge, &custom, &sum0Sink, &diff0Sink, &sum1Sink, &diff1Sink});

    sharedIn.forward(shared_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 0);
    ASSERT_EQ(diff0Sink.forwardCalls, 0);
    ASSERT_EQ(sum1Sink.forwardCalls, 0);
    ASSERT_EQ(diff1Sink.forwardCalls, 0);

    rhs0In.forward(rhs0_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 1);
    ASSERT_EQ(diff0Sink.forwardCalls, 1);
    ASSERT_EQ(sum1Sink.forwardCalls, 0);
    ASSERT_EQ(diff1Sink.forwardCalls, 0);

    rhs1In.forward(rhs1_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 1);
    ASSERT_EQ(diff0Sink.forwardCalls, 1);
    ASSERT_EQ(sum1Sink.forwardCalls, 1);
    ASSERT_EQ(diff1Sink.forwardCalls, 1);

    Tensor sum0_h = copyTensorToCpu(sum0Sink.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor diff0_h = copyTensorToCpu(diff0Sink.getFeatureInput().get(), custom.getStreams()[0]);
    Tensor sum1_h = copyTensorToCpu(sum1Sink.getFeatureInput().get(), custom.getStreams()[2]);
    Tensor diff1_h = copyTensorToCpu(diff1Sink.getFeatureInput().get(), custom.getStreams()[2]);

    expectAllClose(readCpuTensor(sum0_h), {11.0f, 22.0f, 33.0f, 5.0f, 7.0f, 9.0f});
    expectAllClose(readCpuTensor(diff0_h), {-9.0f, -18.0f, -27.0f, 3.0f, 3.0f, 3.0f});
    expectAllClose(readCpuTensor(sum1_h), {6.0f, 6.0f, 6.0f, 2.0f, 1.0f, 0.0f});
    expectAllClose(readCpuTensor(diff1_h), {-4.0f, -2.0f, 0.0f, 6.0f, 9.0f, 12.0f});

    cleanupLayers(
        {&sharedIn, &rhs0In, &rhs1In, &sharedFanout, &rhs0Bridge, &rhs1Bridge, &custom, &sum0Sink, &diff0Sink, &sum1Sink, &diff1Sink});
}

TEST(CustomLayer, SharedParameterGradientsAccumulateAcrossTwoCompleteInterfaces) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> x0Values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> y0Values{-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> x1Values{2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f};
    const std::vector<float> y1Values{3.0f, 1.0f, -1.0f, 0.0f, 2.0f, 4.0f};

    const std::vector<float> gradOutX0Values{1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f};
    const std::vector<float> gradOutY0Values{-2.0f, 3.0f, 1.0f, 0.0f, -1.0f, 2.0f};
    const std::vector<float> gradOutX1Values{0.5f, -1.0f, 2.0f, -0.5f, 1.5f, -2.0f};
    const std::vector<float> gradOutY1Values{1.0f, 1.0f, 0.0f, -1.0f, 2.0f, -0.5f};

    Tensor x0_h(cpuPlacement, descriptor);
    Tensor y0_h(cpuPlacement, descriptor);
    Tensor x1_h(cpuPlacement, descriptor);
    Tensor y1_h(cpuPlacement, descriptor);
    writeCpuTensor(x0_h, x0Values);
    writeCpuTensor(y0_h, y0Values);
    writeCpuTensor(x1_h, x1Values);
    writeCpuTensor(y1_h, y1Values);

    Tensor gradOutX0_h(cpuPlacement, descriptor);
    Tensor gradOutY0_h(cpuPlacement, descriptor);
    Tensor gradOutX1_h(cpuPlacement, descriptor);
    Tensor gradOutY1_h(cpuPlacement, descriptor);
    writeCpuTensor(gradOutX0_h, gradOutX0Values);
    writeCpuTensor(gradOutY0_h, gradOutY0Values);
    writeCpuTensor(gradOutX1_h, gradOutX1Values);
    writeCpuTensor(gradOutY1_h, gradOutY1Values);

    auto scale = std::make_shared<FixedVectorParameter>("scale", std::vector<float>{2.0f, 3.0f, 4.0f}, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", std::vector<float>{10.0f, 20.0f, 30.0f}, true);
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(401, 0.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(402, 0.0f, 0.0f, 0.0f, false)));

    NetworkInput x0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput x1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet x0Rivet, y0Rivet, x1Rivet, y1Rivet;
    CountingPassthrough x0Bridge, y0Bridge, x1Bridge, y1Bridge;
    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);
    CountingPassthrough outX0Sink, outY0Sink, outX1Sink, outY1Sink;

    x0In.connectToNextLayer(&x0Rivet);
    x0Rivet.connectToNextLayer(&x0Bridge);
    y0In.connectToNextLayer(&y0Rivet);
    y0Rivet.connectToNextLayer(&y0Bridge);
    x1In.connectToNextLayer(&x1Rivet);
    x1Rivet.connectToNextLayer(&x1Bridge);
    y1In.connectToNextLayer(&y1Rivet);
    y1Rivet.connectToNextLayer(&y1Bridge);

    x0Bridge.connectToNextLayer(&custom, 0, 0);
    y0Bridge.connectToNextLayer(&custom, 0, 1);
    x1Bridge.connectToNextLayer(&custom, 0, 2);
    y1Bridge.connectToNextLayer(&custom, 0, 3);

    custom.connectToNextLayer(&outX0Sink, 0, 0);
    custom.connectToNextLayer(&outY0Sink, 1, 0);
    custom.connectToNextLayer(&outX1Sink, 2, 0);
    custom.connectToNextLayer(&outY1Sink, 3, 0);

    compileAndInitialize({&x0In,
                          &y0In,
                          &x1In,
                          &y1In,
                          &x0Rivet,
                          &y0Rivet,
                          &x1Rivet,
                          &y1Rivet,
                          &x0Bridge,
                          &y0Bridge,
                          &x1Bridge,
                          &y1Bridge,
                          &custom,
                          &outX0Sink,
                          &outY0Sink,
                          &outX1Sink,
                          &outY1Sink});

    x0In.forward(x0_h, false, batchSize);
    y0In.forward(y0_h, false, batchSize);
    x1In.forward(x1_h, false, batchSize);
    y1In.forward(y1_h, false, batchSize);

    outX0Sink.getErrorOutput().get().copyFromAsync(gradOutX0_h, custom.getStreams()[0]);
    outY0Sink.getErrorOutput().get().copyFromAsync(gradOutY0_h, custom.getStreams()[0]);
    outX1Sink.getErrorOutput().get().copyFromAsync(gradOutX1_h, custom.getStreams()[2]);
    outY1Sink.getErrorOutput().get().copyFromAsync(gradOutY1_h, custom.getStreams()[2]);
    Event app0GradsReady = custom.getStreams()[0].putEvent();
    Event app1GradsReady = custom.getStreams()[2].putEvent();
    app0GradsReady.synchronize();
    app1GradsReady.synchronize();

    outX0Sink.backward(outX0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 0);
    ASSERT_EQ(y0Bridge.backwardCalls, 0);

    outY0Sink.backward(outY0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 1);
    ASSERT_EQ(y0Bridge.backwardCalls, 1);
    ASSERT_EQ(x1Bridge.backwardCalls, 0);
    ASSERT_EQ(y1Bridge.backwardCalls, 0);

    outY1Sink.backward(outY1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 0);
    ASSERT_EQ(y1Bridge.backwardCalls, 0);

    outX1Sink.backward(outX1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 1);
    ASSERT_EQ(y1Bridge.backwardCalls, 1);

    const std::vector<float> scaleValues{2.0f, 3.0f, 4.0f};
    const std::vector<float> expectedScaleGrad = featureWiseScaleGradient(
        {x0Values, y0Values, x1Values, y1Values}, {gradOutX0Values, gradOutY0Values, gradOutX1Values, gradOutY1Values}, features);
    const std::vector<float> expectedBiasGrad =
        featureWiseBiasGradient({gradOutX0Values, gradOutY0Values, gradOutX1Values, gradOutY1Values}, features);

    ASSERT_EQ(custom.getErrorOutputs().size(), 4u);
    Tensor x0Grad_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
    Tensor y0Grad_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);
    Tensor x1Grad_h = copyTensorToCpu(custom.getErrorOutputs()[2].get(), custom.getStreams()[2]);
    Tensor y1Grad_h = copyTensorToCpu(custom.getErrorOutputs()[3].get(), custom.getStreams()[2]);
    Tensor scaleGrad_h = copyTensorToCpu(scale->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());
    Tensor biasGrad_h = copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());

    expectAllClose(readCpuTensor(x0Grad_h), scaleInputGradient(gradOutX0Values, scaleValues));
    expectAllClose(readCpuTensor(y0Grad_h), scaleInputGradient(gradOutY0Values, scaleValues));
    expectAllClose(readCpuTensor(x1Grad_h), scaleInputGradient(gradOutX1Values, scaleValues));
    expectAllClose(readCpuTensor(y1Grad_h), scaleInputGradient(gradOutY1Values, scaleValues));
    expectAllClose(readCpuTensor(scaleGrad_h), expectedScaleGrad);
    expectAllClose(readCpuTensor(biasGrad_h), expectedBiasGrad);

    cleanupLayers({&x0In,
                   &y0In,
                   &x1In,
                   &y1In,
                   &x0Rivet,
                   &y0Rivet,
                   &x1Rivet,
                   &y1Rivet,
                   &x0Bridge,
                   &y0Bridge,
                   &x1Bridge,
                   &y1Bridge,
                   &custom,
                   &outX0Sink,
                   &outY0Sink,
                   &outX1Sink,
                   &outY1Sink});
}

TEST(CustomLayer, MultiInterfaceSharedParameterForwardBackwardThreePassesResetBookkeepingAcrossBatches) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> x0Values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> y0Values{-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> x1Values{2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f};
    const std::vector<float> y1Values{3.0f, 1.0f, -1.0f, 0.0f, 2.0f, 4.0f};

    const std::vector<float> gradOutX0Values{1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f};
    const std::vector<float> gradOutY0Values{-2.0f, 3.0f, 1.0f, 0.0f, -1.0f, 2.0f};
    const std::vector<float> gradOutX1Values{0.5f, -1.0f, 2.0f, -0.5f, 1.5f, -2.0f};
    const std::vector<float> gradOutY1Values{1.0f, 1.0f, 0.0f, -1.0f, 2.0f, -0.5f};

    Tensor x0_h(cpuPlacement, descriptor);
    Tensor y0_h(cpuPlacement, descriptor);
    Tensor x1_h(cpuPlacement, descriptor);
    Tensor y1_h(cpuPlacement, descriptor);
    writeCpuTensor(x0_h, x0Values);
    writeCpuTensor(y0_h, y0Values);
    writeCpuTensor(x1_h, x1Values);
    writeCpuTensor(y1_h, y1Values);

    Tensor gradOutX0_h(cpuPlacement, descriptor);
    Tensor gradOutY0_h(cpuPlacement, descriptor);
    Tensor gradOutX1_h(cpuPlacement, descriptor);
    Tensor gradOutY1_h(cpuPlacement, descriptor);
    writeCpuTensor(gradOutX0_h, gradOutX0Values);
    writeCpuTensor(gradOutY0_h, gradOutY0Values);
    writeCpuTensor(gradOutX1_h, gradOutX1Values);
    writeCpuTensor(gradOutY1_h, gradOutY1Values);

    std::vector<float> expectedScale{2.0f, 3.0f, 4.0f};
    std::vector<float> expectedBias{10.0f, 20.0f, 30.0f};
    const std::vector<float> expectedScaleGrad = featureWiseScaleGradient(
        {x0Values, y0Values, x1Values, y1Values}, {gradOutX0Values, gradOutY0Values, gradOutX1Values, gradOutY1Values}, features);
    const std::vector<float> expectedBiasGrad =
        featureWiseBiasGradient({gradOutX0Values, gradOutY0Values, gradOutX1Values, gradOutY1Values}, features);

    auto scale = std::make_shared<FixedVectorParameter>("scale", expectedScale, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", expectedBias, true);
    // step = learningRate / (batchSize * Loss::lossScalingFactor) = 8 / (2 * 4) = 1.
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(501, 8.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(502, 8.0f, 0.0f, 0.0f, false)));

    NetworkInput x0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput x1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet x0Rivet, y0Rivet, x1Rivet, y1Rivet;
    CountingPassthrough x0Bridge, y0Bridge, x1Bridge, y1Bridge;
    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);
    CountingPassthrough outX0Sink, outY0Sink, outX1Sink, outY1Sink;

    x0In.connectToNextLayer(&x0Rivet);
    x0Rivet.connectToNextLayer(&x0Bridge);
    y0In.connectToNextLayer(&y0Rivet);
    y0Rivet.connectToNextLayer(&y0Bridge);
    x1In.connectToNextLayer(&x1Rivet);
    x1Rivet.connectToNextLayer(&x1Bridge);
    y1In.connectToNextLayer(&y1Rivet);
    y1Rivet.connectToNextLayer(&y1Bridge);

    x0Bridge.connectToNextLayer(&custom, 0, 0);
    y0Bridge.connectToNextLayer(&custom, 0, 1);
    x1Bridge.connectToNextLayer(&custom, 0, 2);
    y1Bridge.connectToNextLayer(&custom, 0, 3);

    custom.connectToNextLayer(&outX0Sink, 0, 0);
    custom.connectToNextLayer(&outY0Sink, 1, 0);
    custom.connectToNextLayer(&outX1Sink, 2, 0);
    custom.connectToNextLayer(&outY1Sink, 3, 0);

    compileAndInitialize({&x0In,
                          &y0In,
                          &x1In,
                          &y1In,
                          &x0Rivet,
                          &y0Rivet,
                          &x1Rivet,
                          &y1Rivet,
                          &x0Bridge,
                          &y0Bridge,
                          &x1Bridge,
                          &y1Bridge,
                          &custom,
                          &outX0Sink,
                          &outY0Sink,
                          &outX1Sink,
                          &outY1Sink});

    ASSERT_TRUE(custom.getGradientUpdateStream().isPresent());
    ASSERT_EQ(custom.getErrorOutputs().size(), 4u);

    auto copyGradientsToDevice = [&]() {
        outX0Sink.getErrorOutput().get().copyFromAsync(gradOutX0_h, custom.getStreams()[0]);
        outY0Sink.getErrorOutput().get().copyFromAsync(gradOutY0_h, custom.getStreams()[0]);
        outX1Sink.getErrorOutput().get().copyFromAsync(gradOutX1_h, custom.getStreams()[2]);
        outY1Sink.getErrorOutput().get().copyFromAsync(gradOutY1_h, custom.getStreams()[2]);
        Event app0GradsReady = custom.getStreams()[0].putEvent();
        Event app1GradsReady = custom.getStreams()[2].putEvent();
        app0GradsReady.synchronize();
        app1GradsReady.synchronize();
    };

    auto expectCurrentForwardOutputs = [&]() {
        Tensor outX0_h = copyTensorToCpu(outX0Sink.getFeatureInput().get(), custom.getStreams()[0]);
        Tensor outY0_h = copyTensorToCpu(outY0Sink.getFeatureInput().get(), custom.getStreams()[0]);
        Tensor outX1_h = copyTensorToCpu(outX1Sink.getFeatureInput().get(), custom.getStreams()[2]);
        Tensor outY1_h = copyTensorToCpu(outY1Sink.getFeatureInput().get(), custom.getStreams()[2]);

        expectAllClose(readCpuTensor(outX0_h), scaleBiasForward(x0Values, expectedScale, expectedBias));
        expectAllClose(readCpuTensor(outY0_h), scaleBiasForward(y0Values, expectedScale, expectedBias));
        expectAllClose(readCpuTensor(outX1_h), scaleBiasForward(x1Values, expectedScale, expectedBias));
        expectAllClose(readCpuTensor(outY1_h), scaleBiasForward(y1Values, expectedScale, expectedBias));
    };

    auto expectCurrentBackwardOutputsAndWeights = [&]() {
        Tensor x0Grad_h = copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0]);
        Tensor y0Grad_h = copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0]);
        Tensor x1Grad_h = copyTensorToCpu(custom.getErrorOutputs()[2].get(), custom.getStreams()[2]);
        Tensor y1Grad_h = copyTensorToCpu(custom.getErrorOutputs()[3].get(), custom.getStreams()[2]);
        Tensor scaleGrad_h = copyTensorToCpu(scale->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());
        Tensor biasGrad_h = copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get());

        expectAllClose(readCpuTensor(x0Grad_h), scaleInputGradient(gradOutX0Values, expectedScale));
        expectAllClose(readCpuTensor(y0Grad_h), scaleInputGradient(gradOutY0Values, expectedScale));
        expectAllClose(readCpuTensor(x1Grad_h), scaleInputGradient(gradOutX1Values, expectedScale));
        expectAllClose(readCpuTensor(y1Grad_h), scaleInputGradient(gradOutY1Values, expectedScale));
        expectAllClose(readCpuTensor(scaleGrad_h), expectedScaleGrad);
        expectAllClose(readCpuTensor(biasGrad_h), expectedBiasGrad);

        subtractInPlace(expectedScale, expectedScaleGrad);
        subtractInPlace(expectedBias, expectedBiasGrad);

        Tensor scaleWeights_h = copyTensorToCpu(scale->getStorage().get(), custom.getGradientUpdateStream().get());
        Tensor biasWeights_h = copyTensorToCpu(bias->getStorage().get(), custom.getGradientUpdateStream().get());
        expectAllClose(readCpuTensor(scaleWeights_h), expectedScale);
        expectAllClose(readCpuTensor(biasWeights_h), expectedBias);
    };

    // Pass 1: complete application 0 first, then application 1.
    x0In.forward(x0_h, false, batchSize);
    ASSERT_EQ(outX0Sink.forwardCalls, 0);
    y0In.forward(y0_h, false, batchSize);
    ASSERT_EQ(outX0Sink.forwardCalls, 1);
    ASSERT_EQ(outY0Sink.forwardCalls, 1);

    y1In.forward(y1_h, false, batchSize);
    ASSERT_EQ(outX1Sink.forwardCalls, 0);
    x1In.forward(x1_h, false, batchSize);
    ASSERT_EQ(outX1Sink.forwardCalls, 1);
    ASSERT_EQ(outY1Sink.forwardCalls, 1);
    expectCurrentForwardOutputs();
    copyGradientsToDevice();
    outX0Sink.backward(outX0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 0);
    outY0Sink.backward(outY0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 1);
    ASSERT_EQ(y0Bridge.backwardCalls, 1);
    outY1Sink.backward(outY1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 0);
    outX1Sink.backward(outX1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 1);
    ASSERT_EQ(y1Bridge.backwardCalls, 1);
    expectCurrentBackwardOutputsAndWeights();

    // Pass 2: interleave the two applications and complete application 1 first.
    y1In.forward(y1_h, false, batchSize);
    x0In.forward(x0_h, false, batchSize);
    ASSERT_EQ(outX0Sink.forwardCalls, 1);
    ASSERT_EQ(outX1Sink.forwardCalls, 1);
    x1In.forward(x1_h, false, batchSize);
    ASSERT_EQ(outX1Sink.forwardCalls, 2);
    ASSERT_EQ(outY1Sink.forwardCalls, 2);
    y0In.forward(y0_h, false, batchSize);
    ASSERT_EQ(outX0Sink.forwardCalls, 2);
    ASSERT_EQ(outY0Sink.forwardCalls, 2);
    expectCurrentForwardOutputs();
    copyGradientsToDevice();
    outX1Sink.backward(outX1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 1);
    outY1Sink.backward(outY1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 2);
    ASSERT_EQ(y1Bridge.backwardCalls, 2);
    outY0Sink.backward(outY0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 1);
    outX0Sink.backward(outX0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 2);
    ASSERT_EQ(y0Bridge.backwardCalls, 2);
    expectCurrentBackwardOutputsAndWeights();

    // Pass 3: complete application 1 first in forward, but application 0 first in backward.
    x1In.forward(x1_h, false, batchSize);
    y1In.forward(y1_h, false, batchSize);
    ASSERT_EQ(outX1Sink.forwardCalls, 3);
    ASSERT_EQ(outY1Sink.forwardCalls, 3);
    y0In.forward(y0_h, false, batchSize);
    ASSERT_EQ(outX0Sink.forwardCalls, 2);
    x0In.forward(x0_h, false, batchSize);
    ASSERT_EQ(outX0Sink.forwardCalls, 3);
    ASSERT_EQ(outY0Sink.forwardCalls, 3);
    expectCurrentForwardOutputs();
    copyGradientsToDevice();
    outY0Sink.backward(outY0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 2);
    outX0Sink.backward(outX0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 3);
    ASSERT_EQ(y0Bridge.backwardCalls, 3);
    outX1Sink.backward(outX1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 2);
    outY1Sink.backward(outY1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 3);
    ASSERT_EQ(y1Bridge.backwardCalls, 3);
    expectCurrentBackwardOutputsAndWeights();

    cleanupLayers({&x0In,
                   &y0In,
                   &x1In,
                   &y1In,
                   &x0Rivet,
                   &y0Rivet,
                   &x1Rivet,
                   &y1Rivet,
                   &x0Bridge,
                   &y0Bridge,
                   &x1Bridge,
                   &y1Bridge,
                   &custom,
                   &outX0Sink,
                   &outY0Sink,
                   &outX1Sink,
                   &outY1Sink});
}

TEST(CustomLayer, MultiInterfaceOutputConstructionRequiresCompleteInputInterface) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;
    (void)batchSize;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    NetworkInput lhs0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput lhs1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());

    CountingPassthrough lhs0Bridge;
    CountingPassthrough rhs0Bridge;
    CountingPassthrough lhs1Bridge;
    CountingPassthrough rhs1Bridge;

    CountingPassthrough sum1Sink;
    CountingPassthrough diff1Sink;

    CustomLayer custom(buildTwoInputTwoOutputExpression(gpuPlacement), {"lhs", "rhs"}, {"sum", "diff"}, gpuPlacement, {}, true);

    lhs0In.connectToNextLayer(&lhs0Bridge);
    rhs0In.connectToNextLayer(&rhs0Bridge);
    lhs1In.connectToNextLayer(&lhs1Bridge);
    rhs1In.connectToNextLayer(&rhs1Bridge);

    // Keep application indices dense. App 0 is fully connected and acts as the
    // reference interface for expression preparation/storage context.
    lhs0Bridge.connectToNextLayer(&custom, 0, 0);  // app 0, lhs
    rhs0Bridge.connectToNextLayer(&custom, 0, 1);  // app 0, rhs

    // Application 1 lhs is connected, but application 1 rhs is intentionally missing.
    lhs1Bridge.connectToNextLayer(&custom, 0, 2);  // app 1, lhs

    try {
        custom.connectToNextLayer(&sum1Sink, 2, 0);  // app 1, sum
        FAIL() << "Expected output construction for an incomplete input interface to fail.";
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        EXPECT_NE(message.find("application 1"), std::string::npos);
        EXPECT_NE(message.find("rhs"), std::string::npos);
    }

    // After completing the input interface, both app-1 outputs can be constructed.
    rhs1Bridge.connectToNextLayer(&custom, 0, 3);  // app 1, rhs

    EXPECT_NO_THROW(custom.connectToNextLayer(&sum1Sink, 2, 0));
    EXPECT_NO_THROW(custom.connectToNextLayer(&diff1Sink, 3, 0));
}

TEST(CustomLayer, MultiInterfaceThreeApplicationsForwardReadinessAndEncodedRouting) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> lhs0Values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> rhs0Values{10.0f, 20.0f, 30.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> lhs1Values{-1.0f, -2.0f, -3.0f, 8.0f, 9.0f, 10.0f};
    const std::vector<float> rhs1Values{5.0f, 4.0f, 3.0f, -2.0f, -4.0f, -6.0f};
    const std::vector<float> lhs2Values{0.5f, 1.5f, 2.5f, -3.5f, -4.5f, -5.5f};
    const std::vector<float> rhs2Values{-0.5f, 2.0f, -2.5f, 3.5f, 4.5f, -6.5f};

    Tensor lhs0_h(cpuPlacement, descriptor);
    Tensor rhs0_h(cpuPlacement, descriptor);
    Tensor lhs1_h(cpuPlacement, descriptor);
    Tensor rhs1_h(cpuPlacement, descriptor);
    Tensor lhs2_h(cpuPlacement, descriptor);
    Tensor rhs2_h(cpuPlacement, descriptor);
    writeCpuTensor(lhs0_h, lhs0Values);
    writeCpuTensor(rhs0_h, rhs0Values);
    writeCpuTensor(lhs1_h, lhs1Values);
    writeCpuTensor(rhs1_h, rhs1Values);
    writeCpuTensor(lhs2_h, lhs2Values);
    writeCpuTensor(rhs2_h, rhs2Values);

    NetworkInput lhs0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput lhs1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput lhs2In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput rhs2In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    CountingPassthrough lhs0Bridge, rhs0Bridge, lhs1Bridge, rhs1Bridge, lhs2Bridge, rhs2Bridge;
    CustomLayer custom(buildTwoInputTwoOutputExpression(gpuPlacement), {"lhs", "rhs"}, {"sum", "diff"}, gpuPlacement, {}, true);
    CountingPassthrough sum0Sink, diff0Sink, sum1Sink, diff1Sink, sum2Sink, diff2Sink;

    lhs0In.connectToNextLayer(&lhs0Bridge);
    rhs0In.connectToNextLayer(&rhs0Bridge);
    lhs1In.connectToNextLayer(&lhs1Bridge);
    rhs1In.connectToNextLayer(&rhs1Bridge);
    lhs2In.connectToNextLayer(&lhs2Bridge);
    rhs2In.connectToNextLayer(&rhs2Bridge);

    lhs0Bridge.connectToNextLayer(&custom, 0, 0);
    rhs0Bridge.connectToNextLayer(&custom, 0, 1);
    lhs1Bridge.connectToNextLayer(&custom, 0, 2);
    rhs1Bridge.connectToNextLayer(&custom, 0, 3);
    lhs2Bridge.connectToNextLayer(&custom, 0, 4);
    rhs2Bridge.connectToNextLayer(&custom, 0, 5);

    custom.connectToNextLayer(&sum0Sink, 0, 0);
    custom.connectToNextLayer(&diff0Sink, 1, 0);
    custom.connectToNextLayer(&sum1Sink, 2, 0);
    custom.connectToNextLayer(&diff1Sink, 3, 0);
    custom.connectToNextLayer(&sum2Sink, 4, 0);
    custom.connectToNextLayer(&diff2Sink, 5, 0);

    compileAndInitialize({&lhs0In,
                          &rhs0In,
                          &lhs1In,
                          &rhs1In,
                          &lhs2In,
                          &rhs2In,
                          &lhs0Bridge,
                          &rhs0Bridge,
                          &lhs1Bridge,
                          &rhs1Bridge,
                          &lhs2Bridge,
                          &rhs2Bridge,
                          &custom,
                          &sum0Sink,
                          &diff0Sink,
                          &sum1Sink,
                          &diff1Sink,
                          &sum2Sink,
                          &diff2Sink});

    auto add = [](const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> out(a.size());
        for (uint64_t i = 0; i < a.size(); ++i)
            out[i] = a[i] + b[i];
        return out;
    };
    auto sub = [](const std::vector<float>& a, const std::vector<float>& b) {
        std::vector<float> out(a.size());
        for (uint64_t i = 0; i < a.size(); ++i)
            out[i] = a[i] - b[i];
        return out;
    };

    // Interleave arrivals so app 2 completes first, then app 0, then app 1.
    lhs0In.forward(lhs0_h, false, batchSize);
    rhs2In.forward(rhs2_h, false, batchSize);
    lhs1In.forward(lhs1_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 0);
    ASSERT_EQ(sum1Sink.forwardCalls, 0);
    ASSERT_EQ(sum2Sink.forwardCalls, 0);

    lhs2In.forward(lhs2_h, false, batchSize);
    ASSERT_EQ(sum2Sink.forwardCalls, 1);
    ASSERT_EQ(diff2Sink.forwardCalls, 1);
    ASSERT_EQ(sum0Sink.forwardCalls, 0);
    ASSERT_EQ(sum1Sink.forwardCalls, 0);

    rhs0In.forward(rhs0_h, false, batchSize);
    ASSERT_EQ(sum0Sink.forwardCalls, 1);
    ASSERT_EQ(diff0Sink.forwardCalls, 1);
    ASSERT_EQ(sum1Sink.forwardCalls, 0);

    rhs1In.forward(rhs1_h, false, batchSize);
    ASSERT_EQ(sum1Sink.forwardCalls, 1);
    ASSERT_EQ(diff1Sink.forwardCalls, 1);

    expectAllClose(readCpuTensor(copyTensorToCpu(sum0Sink.getFeatureInput().get(), custom.getStreams()[0])), add(lhs0Values, rhs0Values));
    expectAllClose(readCpuTensor(copyTensorToCpu(diff0Sink.getFeatureInput().get(), custom.getStreams()[0])), sub(lhs0Values, rhs0Values));
    expectAllClose(readCpuTensor(copyTensorToCpu(sum1Sink.getFeatureInput().get(), custom.getStreams()[2])), add(lhs1Values, rhs1Values));
    expectAllClose(readCpuTensor(copyTensorToCpu(diff1Sink.getFeatureInput().get(), custom.getStreams()[2])), sub(lhs1Values, rhs1Values));
    expectAllClose(readCpuTensor(copyTensorToCpu(sum2Sink.getFeatureInput().get(), custom.getStreams()[4])), add(lhs2Values, rhs2Values));
    expectAllClose(readCpuTensor(copyTensorToCpu(diff2Sink.getFeatureInput().get(), custom.getStreams()[4])), sub(lhs2Values, rhs2Values));

    cleanupLayers({&lhs0In,
                   &rhs0In,
                   &lhs1In,
                   &rhs1In,
                   &lhs2In,
                   &rhs2In,
                   &lhs0Bridge,
                   &rhs0Bridge,
                   &lhs1Bridge,
                   &rhs1Bridge,
                   &lhs2Bridge,
                   &rhs2Bridge,
                   &custom,
                   &sum0Sink,
                   &diff0Sink,
                   &sum1Sink,
                   &diff1Sink,
                   &sum2Sink,
                   &diff2Sink});
}

TEST(CustomLayer, MultiInterfaceSparseDownstreamBackpropWaitsOnlyForConnectedGradientOutputs) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> x0Values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> y0Values{-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> x1Values{2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f};
    const std::vector<float> y1Values{3.0f, 1.0f, -1.0f, 0.0f, 2.0f, 4.0f};

    const std::vector<float> gradOutX0Values{1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f};
    const std::vector<float> gradOutY1Values{1.0f, 1.0f, 0.0f, -1.0f, 2.0f, -0.5f};

    Tensor x0_h(cpuPlacement, descriptor);
    Tensor y0_h(cpuPlacement, descriptor);
    Tensor x1_h(cpuPlacement, descriptor);
    Tensor y1_h(cpuPlacement, descriptor);
    Tensor gradOutX0_h(cpuPlacement, descriptor);
    Tensor gradOutY1_h(cpuPlacement, descriptor);
    writeCpuTensor(x0_h, x0Values);
    writeCpuTensor(y0_h, y0Values);
    writeCpuTensor(x1_h, x1Values);
    writeCpuTensor(y1_h, y1Values);
    writeCpuTensor(gradOutX0_h, gradOutX0Values);
    writeCpuTensor(gradOutY1_h, gradOutY1Values);

    const std::vector<float> scaleValues{2.0f, 3.0f, 4.0f};
    const std::vector<float> biasValues{10.0f, 20.0f, 30.0f};
    auto scale = std::make_shared<FixedVectorParameter>("scale", scaleValues, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", biasValues, true);
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(601, 0.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(602, 0.0f, 0.0f, 0.0f, false)));

    NetworkInput x0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput x1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet x0Rivet, y0Rivet, x1Rivet, y1Rivet;
    CountingPassthrough x0Bridge, y0Bridge, x1Bridge, y1Bridge;
    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);
    CountingPassthrough outX0Sink, outY0InferenceSink, outX1InferenceSink, outY1Sink;
    outY0InferenceSink.setConstructForInferenceOnly(true);
    outX1InferenceSink.setConstructForInferenceOnly(true);

    x0In.connectToNextLayer(&x0Rivet);
    x0Rivet.connectToNextLayer(&x0Bridge);
    y0In.connectToNextLayer(&y0Rivet);
    y0Rivet.connectToNextLayer(&y0Bridge);
    x1In.connectToNextLayer(&x1Rivet);
    x1Rivet.connectToNextLayer(&x1Bridge);
    y1In.connectToNextLayer(&y1Rivet);
    y1Rivet.connectToNextLayer(&y1Bridge);

    x0Bridge.connectToNextLayer(&custom, 0, 0);
    y0Bridge.connectToNextLayer(&custom, 0, 1);
    x1Bridge.connectToNextLayer(&custom, 0, 2);
    y1Bridge.connectToNextLayer(&custom, 0, 3);

    custom.connectToNextLayer(&outX0Sink, 0, 0);
    custom.connectToNextLayer(&outY0InferenceSink, 1, 0);
    custom.connectToNextLayer(&outX1InferenceSink, 2, 0);
    custom.connectToNextLayer(&outY1Sink, 3, 0);

    compileAndInitialize({&x0In,
                          &y0In,
                          &x1In,
                          &y1In,
                          &x0Rivet,
                          &y0Rivet,
                          &x1Rivet,
                          &y1Rivet,
                          &x0Bridge,
                          &y0Bridge,
                          &x1Bridge,
                          &y1Bridge,
                          &custom,
                          &outX0Sink,
                          &outY0InferenceSink,
                          &outX1InferenceSink,
                          &outY1Sink});

    ASSERT_TRUE(outX0Sink.getErrorOutput().isPresent());
    ASSERT_TRUE(outY0InferenceSink.getErrorOutput().isEmpty());
    ASSERT_TRUE(outX1InferenceSink.getErrorOutput().isEmpty());
    ASSERT_TRUE(outY1Sink.getErrorOutput().isPresent());

    x0In.forward(x0_h, false, batchSize);
    y0In.forward(y0_h, false, batchSize);
    x1In.forward(x1_h, false, batchSize);
    y1In.forward(y1_h, false, batchSize);

    outX0Sink.getErrorOutput().get().copyFromAsync(gradOutX0_h, custom.getStreams()[0]);
    outY1Sink.getErrorOutput().get().copyFromAsync(gradOutY1_h, custom.getStreams()[2]);
    Event app0GradReady = custom.getStreams()[0].putEvent();
    Event app1GradReady = custom.getStreams()[2].putEvent();
    app0GradReady.synchronize();
    app1GradReady.synchronize();

    outX0Sink.backward(outX0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 1);
    ASSERT_EQ(y0Bridge.backwardCalls, 1);
    ASSERT_EQ(x1Bridge.backwardCalls, 0);
    ASSERT_EQ(y1Bridge.backwardCalls, 0);

    outY1Sink.backward(outY1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 1);
    ASSERT_EQ(y1Bridge.backwardCalls, 1);

    const std::vector<float> zeroGradient(batchSize * features, 0.0f);
    expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0])),
                   scaleInputGradient(gradOutX0Values, scaleValues));
    expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0])), zeroGradient);
    expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[2].get(), custom.getStreams()[2])), zeroGradient);
    expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[3].get(), custom.getStreams()[2])),
                   scaleInputGradient(gradOutY1Values, scaleValues));

    const std::vector<float> expectedScaleGrad =
        featureWiseScaleGradient({x0Values, y1Values}, {gradOutX0Values, gradOutY1Values}, features);
    const std::vector<float> expectedBiasGrad = featureWiseBiasGradient({gradOutX0Values, gradOutY1Values}, features);

    expectAllClose(
        readCpuTensor(copyTensorToCpu(scale->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get())),
        expectedScaleGrad);
    expectAllClose(readCpuTensor(copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get())),
                   expectedBiasGrad);
    expectAllClose(readCpuTensor(copyTensorToCpu(scale->getStorage().get(), custom.getGradientUpdateStream().get())), scaleValues);
    expectAllClose(readCpuTensor(copyTensorToCpu(bias->getStorage().get(), custom.getGradientUpdateStream().get())), biasValues);

    cleanupLayers({&x0In,
                   &y0In,
                   &x1In,
                   &y1In,
                   &x0Rivet,
                   &y0Rivet,
                   &x1Rivet,
                   &y1Rivet,
                   &x0Bridge,
                   &y0Bridge,
                   &x1Bridge,
                   &y1Bridge,
                   &custom,
                   &outX0Sink,
                   &outY0InferenceSink,
                   &outX1InferenceSink,
                   &outY1Sink});
}

TEST(CustomLayer, MultiInterfaceNoBackpropApplicationDoesNotDelaySharedWeightUpdate) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> x0Values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> y0Values{-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> x1Values{2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f};
    const std::vector<float> y1Values{3.0f, 1.0f, -1.0f, 0.0f, 2.0f, 4.0f};
    const std::vector<float> gradOutX1Values{0.5f, -1.0f, 2.0f, -0.5f, 1.5f, -2.0f};
    const std::vector<float> gradOutY1Values{1.0f, 1.0f, 0.0f, -1.0f, 2.0f, -0.5f};

    Tensor x0_h(cpuPlacement, descriptor);
    Tensor y0_h(cpuPlacement, descriptor);
    Tensor x1_h(cpuPlacement, descriptor);
    Tensor y1_h(cpuPlacement, descriptor);
    Tensor gradOutX1_h(cpuPlacement, descriptor);
    Tensor gradOutY1_h(cpuPlacement, descriptor);
    writeCpuTensor(x0_h, x0Values);
    writeCpuTensor(y0_h, y0Values);
    writeCpuTensor(x1_h, x1Values);
    writeCpuTensor(y1_h, y1Values);
    writeCpuTensor(gradOutX1_h, gradOutX1Values);
    writeCpuTensor(gradOutY1_h, gradOutY1Values);

    const std::vector<float> initialScale{2.0f, 3.0f, 4.0f};
    const std::vector<float> initialBias{10.0f, 20.0f, 30.0f};
    auto scale = std::make_shared<FixedVectorParameter>("scale", initialScale, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", initialBias, true);
    // step = learningRate / (batchSize * Loss::lossScalingFactor) = 8 / (2 * 4) = 1.
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(701, 8.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(702, 8.0f, 0.0f, 0.0f, false)));

    NetworkInput x0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput x1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet x0Rivet, y0Rivet, x1Rivet, y1Rivet;
    CountingPassthrough x0Bridge, y0Bridge, x1Bridge, y1Bridge;
    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);
    CountingPassthrough outX0InferenceSink, outY0InferenceSink, outX1Sink, outY1Sink;
    outX0InferenceSink.setConstructForInferenceOnly(true);
    outY0InferenceSink.setConstructForInferenceOnly(true);

    x0In.connectToNextLayer(&x0Rivet);
    x0Rivet.connectToNextLayer(&x0Bridge);
    y0In.connectToNextLayer(&y0Rivet);
    y0Rivet.connectToNextLayer(&y0Bridge);
    x1In.connectToNextLayer(&x1Rivet);
    x1Rivet.connectToNextLayer(&x1Bridge);
    y1In.connectToNextLayer(&y1Rivet);
    y1Rivet.connectToNextLayer(&y1Bridge);

    x0Bridge.connectToNextLayer(&custom, 0, 0);
    y0Bridge.connectToNextLayer(&custom, 0, 1);
    x1Bridge.connectToNextLayer(&custom, 0, 2);
    y1Bridge.connectToNextLayer(&custom, 0, 3);

    custom.connectToNextLayer(&outX0InferenceSink, 0, 0);
    custom.connectToNextLayer(&outY0InferenceSink, 1, 0);
    custom.connectToNextLayer(&outX1Sink, 2, 0);
    custom.connectToNextLayer(&outY1Sink, 3, 0);

    compileAndInitialize({&x0In,
                          &y0In,
                          &x1In,
                          &y1In,
                          &x0Rivet,
                          &y0Rivet,
                          &x1Rivet,
                          &y1Rivet,
                          &x0Bridge,
                          &y0Bridge,
                          &x1Bridge,
                          &y1Bridge,
                          &custom,
                          &outX0InferenceSink,
                          &outY0InferenceSink,
                          &outX1Sink,
                          &outY1Sink});

    ASSERT_TRUE(outX0InferenceSink.getErrorOutput().isEmpty());
    ASSERT_TRUE(outY0InferenceSink.getErrorOutput().isEmpty());
    ASSERT_TRUE(outX1Sink.getErrorOutput().isPresent());
    ASSERT_TRUE(outY1Sink.getErrorOutput().isPresent());

    x0In.forward(x0_h, false, batchSize);
    y0In.forward(y0_h, false, batchSize);
    x1In.forward(x1_h, false, batchSize);
    y1In.forward(y1_h, false, batchSize);

    outX1Sink.getErrorOutput().get().copyFromAsync(gradOutX1_h, custom.getStreams()[2]);
    outY1Sink.getErrorOutput().get().copyFromAsync(gradOutY1_h, custom.getStreams()[2]);
    Event gradsReady = custom.getStreams()[2].putEvent();
    gradsReady.synchronize();

    outX1Sink.backward(outX1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 0);
    ASSERT_EQ(y1Bridge.backwardCalls, 0);

    outY1Sink.backward(outY1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x1Bridge.backwardCalls, 1);
    ASSERT_EQ(y1Bridge.backwardCalls, 1);
    ASSERT_EQ(x0Bridge.backwardCalls, 0);
    ASSERT_EQ(y0Bridge.backwardCalls, 0);

    const std::vector<float> expectedScaleGrad =
        featureWiseScaleGradient({x1Values, y1Values}, {gradOutX1Values, gradOutY1Values}, features);
    const std::vector<float> expectedBiasGrad = featureWiseBiasGradient({gradOutX1Values, gradOutY1Values}, features);
    std::vector<float> expectedScale = initialScale;
    std::vector<float> expectedBias = initialBias;
    subtractInPlace(expectedScale, expectedScaleGrad);
    subtractInPlace(expectedBias, expectedBiasGrad);

    expectAllClose(
        readCpuTensor(copyTensorToCpu(scale->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get())),
        expectedScaleGrad);
    expectAllClose(readCpuTensor(copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get())),
                   expectedBiasGrad);
    expectAllClose(readCpuTensor(copyTensorToCpu(scale->getStorage().get(), custom.getGradientUpdateStream().get())), expectedScale);
    expectAllClose(readCpuTensor(copyTensorToCpu(bias->getStorage().get(), custom.getGradientUpdateStream().get())), expectedBias);

    cleanupLayers({&x0In,
                   &y0In,
                   &x1In,
                   &y1In,
                   &x0Rivet,
                   &y0Rivet,
                   &x1Rivet,
                   &y1Rivet,
                   &x0Bridge,
                   &y0Bridge,
                   &x1Bridge,
                   &y1Bridge,
                   &custom,
                   &outX0InferenceSink,
                   &outY0InferenceSink,
                   &outX1Sink,
                   &outY1Sink});
}

TEST(CustomLayer, TrainingDisabledParameterDoesNotReceiveGradientOrUpdateButStillAffectsInputGradient) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> xValues{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> yValues{-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> gradOutXValues{1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f};
    const std::vector<float> gradOutYValues{-2.0f, 3.0f, 1.0f, 0.0f, -1.0f, 2.0f};

    Tensor x_h(cpuPlacement, descriptor);
    Tensor y_h(cpuPlacement, descriptor);
    Tensor gradOutX_h(cpuPlacement, descriptor);
    Tensor gradOutY_h(cpuPlacement, descriptor);
    writeCpuTensor(x_h, xValues);
    writeCpuTensor(y_h, yValues);
    writeCpuTensor(gradOutX_h, gradOutXValues);
    writeCpuTensor(gradOutY_h, gradOutYValues);

    const std::vector<float> scaleValues{2.0f, 3.0f, 4.0f};
    const std::vector<float> biasValues{10.0f, 20.0f, 30.0f};

    auto scale = std::make_shared<FixedVectorParameter>("scale", scaleValues, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", biasValues, true);

    // scale is trainable in principle, but frozen for this compiled graph.
    // If your setter has a different name, this is the one line to rename.
    scale->setTrainingEnabled(false);

    // step = learningRate / (batchSize * Loss::lossScalingFactor) = 8 / (2 * 4) = 1.
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(801, 8.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(802, 8.0f, 0.0f, 0.0f, false)));

    NetworkInput xIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput yIn(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    GradientRivet xRivet, yRivet;
    CountingPassthrough xBridge, yBridge;

    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);

    CountingPassthrough outXSink;
    CountingPassthrough outYSink;

    xIn.connectToNextLayer(&xRivet);
    xRivet.connectToNextLayer(&xBridge);
    yIn.connectToNextLayer(&yRivet);
    yRivet.connectToNextLayer(&yBridge);

    xBridge.connectToNextLayer(&custom, 0, 0);
    yBridge.connectToNextLayer(&custom, 0, 1);

    custom.connectToNextLayer(&outXSink, 0, 0);
    custom.connectToNextLayer(&outYSink, 1, 0);

    compileAndInitialize({&xIn, &yIn, &xRivet, &yRivet, &xBridge, &yBridge, &custom, &outXSink, &outYSink});

    ASSERT_TRUE(outXSink.getErrorOutput().isPresent());
    ASSERT_TRUE(outYSink.getErrorOutput().isPresent());
    ASSERT_TRUE(custom.getGradientUpdateStream().isPresent());
    ASSERT_TRUE(scale->getStorage().isPresent());
    ASSERT_TRUE(bias->getStorage().isPresent());

    xIn.forward(x_h, false, batchSize);
    yIn.forward(y_h, false, batchSize);

    outXSink.getErrorOutput().get().copyFromAsync(gradOutX_h, custom.getStreams()[0]);
    outYSink.getErrorOutput().get().copyFromAsync(gradOutY_h, custom.getStreams()[0]);
    Event gradsReady = custom.getStreams()[0].putEvent();
    gradsReady.synchronize();

    outXSink.backward(outXSink.getErrorOutput(), batchSize);
    ASSERT_EQ(xBridge.backwardCalls, 0);
    ASSERT_EQ(yBridge.backwardCalls, 0);

    outYSink.backward(outYSink.getErrorOutput(), batchSize);
    ASSERT_EQ(xBridge.backwardCalls, 1);
    ASSERT_EQ(yBridge.backwardCalls, 1);

    // Even though scale is frozen for training, it is still part of the forward expression
    // and must still be used as a constant when computing input gradients.
    expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0])),
                   scaleInputGradient(gradOutXValues, scaleValues));
    expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0])),
                   scaleInputGradient(gradOutYValues, scaleValues));

    const std::vector<float> expectedBiasGrad = featureWiseBiasGradient({gradOutXValues, gradOutYValues}, features);
    std::vector<float> expectedBias = biasValues;
    subtractInPlace(expectedBias, expectedBiasGrad);

    // bias participates in compiled training.
    ASSERT_TRUE(bias->getOptimizer()->getWeightsGradient().isPresent());
    expectAllClose(readCpuTensor(copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get())),
                   expectedBiasGrad);
    expectAllClose(readCpuTensor(copyTensorToCpu(bias->getStorage().get(), custom.getGradientUpdateStream().get())), expectedBias);

    // scale does not participate in compiled training. It must not be updated even though
    // it is mathematically needed for dx/dy.
    expectAllClose(readCpuTensor(copyTensorToCpu(scale->getStorage().get(), custom.getGradientUpdateStream().get())), scaleValues);

    cleanupLayers({&xIn, &yIn, &xRivet, &yRivet, &xBridge, &yBridge, &custom, &outXSink, &outYSink});
}

TEST(CustomLayer, MultiInterfaceSparseDownstreamBackpropResetsAcrossPassesAndReversedOrder) {
    const uint64_t batchSize = 2;
    const uint64_t features = 3;

    TensorDescriptor descriptor(DataType::FP32, {batchSize, features});

    const std::vector<float> x0Values{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> y0Values{-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> x1Values{2.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f};
    const std::vector<float> y1Values{3.0f, 1.0f, -1.0f, 0.0f, 2.0f, 4.0f};

    const std::vector<float> gradOutX0Values{1.0f, 0.0f, -1.0f, 2.0f, 1.0f, 0.5f};
    const std::vector<float> gradOutY1Values{1.0f, 1.0f, 0.0f, -1.0f, 2.0f, -0.5f};

    Tensor x0_h(cpuPlacement, descriptor);
    Tensor y0_h(cpuPlacement, descriptor);
    Tensor x1_h(cpuPlacement, descriptor);
    Tensor y1_h(cpuPlacement, descriptor);
    Tensor gradOutX0_h(cpuPlacement, descriptor);
    Tensor gradOutY1_h(cpuPlacement, descriptor);
    writeCpuTensor(x0_h, x0Values);
    writeCpuTensor(y0_h, y0Values);
    writeCpuTensor(x1_h, x1Values);
    writeCpuTensor(y1_h, y1Values);
    writeCpuTensor(gradOutX0_h, gradOutX0Values);
    writeCpuTensor(gradOutY1_h, gradOutY1Values);

    const std::vector<float> scaleValues{2.0f, 3.0f, 4.0f};
    const std::vector<float> biasValues{10.0f, 20.0f, 30.0f};

    auto scale = std::make_shared<FixedVectorParameter>("scale", scaleValues, true);
    auto bias = std::make_shared<FixedVectorParameter>("bias", biasValues, true);

    // Zero LR keeps weights fixed so each pass should produce the same gradients.
    // This makes stale gradient accumulation/reset bugs easy to detect.
    scale->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(901, 0.0f, 0.0f, 0.0f, false)));
    bias->setOptimizer(std::static_pointer_cast<Optimizer>(std::make_shared<Sgd>(902, 0.0f, 0.0f, 0.0f, false)));

    NetworkInput x0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y0In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput x1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());
    NetworkInput y1In(gpuPlacement, DataType::FP32, descriptor.getDimensions());

    GradientRivet x0Rivet, y0Rivet, x1Rivet, y1Rivet;
    CountingPassthrough x0Bridge, y0Bridge, x1Bridge, y1Bridge;

    CustomLayer custom(
        buildSharedScaleBiasTwoInputTwoOutputExpression(gpuPlacement), {"x", "y"}, {"out_x", "out_y"}, gpuPlacement, {scale, bias}, false);

    CountingPassthrough outX0Sink;
    CountingPassthrough outY0InferenceSink;
    CountingPassthrough outX1InferenceSink;
    CountingPassthrough outY1Sink;
    outY0InferenceSink.setConstructForInferenceOnly(true);
    outX1InferenceSink.setConstructForInferenceOnly(true);

    x0In.connectToNextLayer(&x0Rivet);
    x0Rivet.connectToNextLayer(&x0Bridge);
    y0In.connectToNextLayer(&y0Rivet);
    y0Rivet.connectToNextLayer(&y0Bridge);
    x1In.connectToNextLayer(&x1Rivet);
    x1Rivet.connectToNextLayer(&x1Bridge);
    y1In.connectToNextLayer(&y1Rivet);
    y1Rivet.connectToNextLayer(&y1Bridge);

    x0Bridge.connectToNextLayer(&custom, 0, 0);
    y0Bridge.connectToNextLayer(&custom, 0, 1);
    x1Bridge.connectToNextLayer(&custom, 0, 2);
    y1Bridge.connectToNextLayer(&custom, 0, 3);

    // app 0 only receives out_x gradient; app 1 only receives out_y gradient.
    custom.connectToNextLayer(&outX0Sink, 0, 0);
    custom.connectToNextLayer(&outY0InferenceSink, 1, 0);
    custom.connectToNextLayer(&outX1InferenceSink, 2, 0);
    custom.connectToNextLayer(&outY1Sink, 3, 0);

    compileAndInitialize({&x0In,
                          &y0In,
                          &x1In,
                          &y1In,
                          &x0Rivet,
                          &y0Rivet,
                          &x1Rivet,
                          &y1Rivet,
                          &x0Bridge,
                          &y0Bridge,
                          &x1Bridge,
                          &y1Bridge,
                          &custom,
                          &outX0Sink,
                          &outY0InferenceSink,
                          &outX1InferenceSink,
                          &outY1Sink});

    ASSERT_TRUE(outX0Sink.getErrorOutput().isPresent());
    ASSERT_TRUE(outY0InferenceSink.getErrorOutput().isEmpty());
    ASSERT_TRUE(outX1InferenceSink.getErrorOutput().isEmpty());
    ASSERT_TRUE(outY1Sink.getErrorOutput().isPresent());
    ASSERT_TRUE(custom.getGradientUpdateStream().isPresent());
    ASSERT_TRUE(scale->getOptimizer()->getWeightsGradient().isPresent());
    ASSERT_TRUE(bias->getOptimizer()->getWeightsGradient().isPresent());

    const std::vector<float> zeroGradient(batchSize * features, 0.0f);
    const std::vector<float> expectedScaleGrad =
        featureWiseScaleGradient({x0Values, y1Values}, {gradOutX0Values, gradOutY1Values}, features);
    const std::vector<float> expectedBiasGrad = featureWiseBiasGradient({gradOutX0Values, gradOutY1Values}, features);

    auto copySparseGradientsToDevice = [&]() {
        outX0Sink.getErrorOutput().get().copyFromAsync(gradOutX0_h, custom.getStreams()[0]);
        outY1Sink.getErrorOutput().get().copyFromAsync(gradOutY1_h, custom.getStreams()[2]);

        Event app0GradReady = custom.getStreams()[0].putEvent();
        Event app1GradReady = custom.getStreams()[2].putEvent();
        app0GradReady.synchronize();
        app1GradReady.synchronize();
    };

    auto expectSparseBackwardResults = [&]() {
        expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[0].get(), custom.getStreams()[0])),
                       scaleInputGradient(gradOutX0Values, scaleValues));
        expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[1].get(), custom.getStreams()[0])), zeroGradient);
        expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[2].get(), custom.getStreams()[2])), zeroGradient);
        expectAllClose(readCpuTensor(copyTensorToCpu(custom.getErrorOutputs()[3].get(), custom.getStreams()[2])),
                       scaleInputGradient(gradOutY1Values, scaleValues));

        expectAllClose(
            readCpuTensor(copyTensorToCpu(scale->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get())),
            expectedScaleGrad);
        expectAllClose(
            readCpuTensor(copyTensorToCpu(bias->getOptimizer()->getWeightsGradient().get(), custom.getGradientUpdateStream().get())),
            expectedBiasGrad);

        expectAllClose(readCpuTensor(copyTensorToCpu(scale->getStorage().get(), custom.getGradientUpdateStream().get())), scaleValues);
        expectAllClose(readCpuTensor(copyTensorToCpu(bias->getStorage().get(), custom.getGradientUpdateStream().get())), biasValues);
    };

    // Pass 1: app 0 completes first in forward and backward.
    x0In.forward(x0_h, false, batchSize);
    y0In.forward(y0_h, false, batchSize);
    ASSERT_EQ(outX0Sink.forwardCalls, 1);
    ASSERT_EQ(outY0InferenceSink.forwardCalls, 1);
    ASSERT_EQ(outX1InferenceSink.forwardCalls, 0);
    ASSERT_EQ(outY1Sink.forwardCalls, 0);

    x1In.forward(x1_h, false, batchSize);
    y1In.forward(y1_h, false, batchSize);
    ASSERT_EQ(outX1InferenceSink.forwardCalls, 1);
    ASSERT_EQ(outY1Sink.forwardCalls, 1);

    copySparseGradientsToDevice();

    outX0Sink.backward(outX0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 1);
    ASSERT_EQ(y0Bridge.backwardCalls, 1);
    ASSERT_EQ(x1Bridge.backwardCalls, 0);
    ASSERT_EQ(y1Bridge.backwardCalls, 0);

    outY1Sink.backward(outY1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 1);
    ASSERT_EQ(y0Bridge.backwardCalls, 1);
    ASSERT_EQ(x1Bridge.backwardCalls, 1);
    ASSERT_EQ(y1Bridge.backwardCalls, 1);

    expectSparseBackwardResults();

    // Pass 2: app 1 completes first in forward and backward. This stresses that
    // the sparse expected-gradient masks and clear-first/accumulate bookkeeping reset.
    y1In.forward(y1_h, false, batchSize);
    x1In.forward(x1_h, false, batchSize);
    ASSERT_EQ(outX1InferenceSink.forwardCalls, 2);
    ASSERT_EQ(outY1Sink.forwardCalls, 2);
    ASSERT_EQ(outX0Sink.forwardCalls, 1);
    ASSERT_EQ(outY0InferenceSink.forwardCalls, 1);

    y0In.forward(y0_h, false, batchSize);
    x0In.forward(x0_h, false, batchSize);
    ASSERT_EQ(outX0Sink.forwardCalls, 2);
    ASSERT_EQ(outY0InferenceSink.forwardCalls, 2);

    copySparseGradientsToDevice();

    outY1Sink.backward(outY1Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 1);
    ASSERT_EQ(y0Bridge.backwardCalls, 1);
    ASSERT_EQ(x1Bridge.backwardCalls, 2);
    ASSERT_EQ(y1Bridge.backwardCalls, 2);

    outX0Sink.backward(outX0Sink.getErrorOutput(), batchSize);
    ASSERT_EQ(x0Bridge.backwardCalls, 2);
    ASSERT_EQ(y0Bridge.backwardCalls, 2);
    ASSERT_EQ(x1Bridge.backwardCalls, 2);
    ASSERT_EQ(y1Bridge.backwardCalls, 2);

    expectSparseBackwardResults();

    cleanupLayers({&x0In,
                   &y0In,
                   &x1In,
                   &y1In,
                   &x0Rivet,
                   &y0Rivet,
                   &x1Rivet,
                   &y1Rivet,
                   &x0Bridge,
                   &y0Bridge,
                   &x1Bridge,
                   &y1Bridge,
                   &custom,
                   &outX0Sink,
                   &outY0InferenceSink,
                   &outX1InferenceSink,
                   &outY1Sink});
}

}  // namespace
