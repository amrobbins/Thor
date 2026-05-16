#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNorm.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

namespace {

CudnnRmsNormDescriptor makeDescriptor() {
    CudnnRmsNormDescriptor descriptor;
    descriptor.outerSize = 8;
    descriptor.normalizedFeatureCount = 16;
    descriptor.inputDataType = TensorDescriptor::DataType::FP16;
    descriptor.outputDataType = TensorDescriptor::DataType::FP16;
    descriptor.parameterDataType = TensorDescriptor::DataType::FP32;
    descriptor.computeDataType = TensorDescriptor::DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    return descriptor;
}

}  // namespace

TEST(CudnnRmsNormDescriptor, AcceptsFp16Bf16AndFp32IoWithFp32Parameters) {
    for (TensorDescriptor::DataType dtype : {TensorDescriptor::DataType::FP16,
                                             TensorDescriptor::DataType::BF16,
                                             TensorDescriptor::DataType::FP32}) {
        CudnnRmsNormDescriptor descriptor = makeDescriptor();
        descriptor.inputDataType = dtype;
        descriptor.outputDataType = dtype;
        EXPECT_NO_THROW(descriptor.validateForward());
        EXPECT_NO_THROW(descriptor.validateBackward());
    }
}

TEST(CudnnRmsNormDescriptor, RejectsEmptyOuterOrFeatureCount) {
    CudnnRmsNormDescriptor descriptor = makeDescriptor();
    descriptor.outerSize = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.normalizedFeatureCount = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnRmsNormDescriptor, RejectsUnsupportedIoDtype) {
    CudnnRmsNormDescriptor descriptor = makeDescriptor();
    descriptor.inputDataType = TensorDescriptor::DataType::FP8_E4M3;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.outputDataType = TensorDescriptor::DataType::INT32;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnRmsNormDescriptor, RejectsNonFp32ParametersOrCompute) {
    CudnnRmsNormDescriptor descriptor = makeDescriptor();
    descriptor.parameterDataType = TensorDescriptor::DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.computeDataType = TensorDescriptor::DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnRmsNormDescriptor, RejectsNonPositiveEpsilon) {
    CudnnRmsNormDescriptor descriptor = makeDescriptor();
    descriptor.epsilon = 0.0f;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.epsilon = -1.0e-5f;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnRmsNormDescriptor, AcceptsInferenceFusedSwishAndRejectsTrainingOrBackwardFusedSwish) {
    CudnnRmsNormDescriptor descriptor = makeDescriptor();
    descriptor.training = false;
    descriptor.inputDataType = TensorDescriptor::DataType::BF16;
    descriptor.outputDataType = TensorDescriptor::DataType::BF16;
    descriptor.parameterDataType = TensorDescriptor::DataType::BF16;
    descriptor.fusedActivation = CudnnRmsNormFusedActivation::SWISH;
    EXPECT_NO_THROW(descriptor.validateForward());
    EXPECT_THROW(descriptor.validateBackward(), std::invalid_argument);

    descriptor.training = true;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.training = false;
    descriptor.fusedActivation = CudnnRmsNormFusedActivation::SWISH;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnRmsNormDescriptor, ParsesFusedActivationAliases) {
    EXPECT_EQ(cudnnRmsNormFusedActivationFromString("none"), CudnnRmsNormFusedActivation::NONE);
    EXPECT_EQ(cudnnRmsNormFusedActivationFromString("swish"), CudnnRmsNormFusedActivation::SWISH);
    EXPECT_EQ(cudnnRmsNormFusedActivationFromString("silu"), CudnnRmsNormFusedActivation::SWISH);
    EXPECT_STREQ(toString(CudnnRmsNormFusedActivation::SWISH), "swish");
    EXPECT_THROW(cudnnRmsNormFusedActivationFromString("relu"), std::invalid_argument);
}
