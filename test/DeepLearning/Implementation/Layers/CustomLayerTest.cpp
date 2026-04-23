#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/NetworkInput.h"

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

}  // namespace
