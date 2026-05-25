#include "Utilities/TensorOperations/DeepLearning/CudnnLayerNorm.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

namespace {

CudnnLayerNormDescriptor makeDescriptor() {
    CudnnLayerNormDescriptor descriptor;
    descriptor.outerSize = 8;
    descriptor.normalizedFeatureCount = 16;
    descriptor.inputDataType = DataType::FP16;
    descriptor.outputDataType = DataType::FP16;
    descriptor.parameterDataType = DataType::FP32;
    descriptor.computeDataType = DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    return descriptor;
}

}  // namespace

TEST(CudnnLayerNormDescriptor, AcceptsFp16Bf16AndFp32IoWithFp32Parameters) {
    for (DataType dtype : {DataType::FP16,
                                             DataType::BF16,
                                             DataType::FP32}) {
        CudnnLayerNormDescriptor descriptor = makeDescriptor();
        descriptor.inputDataType = dtype;
        descriptor.outputDataType = dtype;
        EXPECT_NO_THROW(descriptor.validateForward());
        EXPECT_NO_THROW(descriptor.validateBackward());
    }
}

TEST(CudnnLayerNormDescriptor, RejectsEmptyOuterOrFeatureCount) {
    CudnnLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.outerSize = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.normalizedFeatureCount = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnLayerNormDescriptor, RejectsUnsupportedIoDtype) {
    CudnnLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.inputDataType = DataType::FP8_E4M3;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.outputDataType = DataType::INT32;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnLayerNormDescriptor, RejectsNonFp32ParametersOrCompute) {
    CudnnLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.parameterDataType = DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.computeDataType = DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnLayerNormDescriptor, RejectsNonPositiveEpsilon) {
    CudnnLayerNormDescriptor descriptor = makeDescriptor();
    descriptor.epsilon = 0.0f;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.epsilon = -1.0e-5f;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}
