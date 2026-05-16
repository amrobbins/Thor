#include "Utilities/TensorOperations/DeepLearning/CudnnInstanceNorm.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

namespace {

CudnnInstanceNormDescriptor makeDescriptor() {
    CudnnInstanceNormDescriptor descriptor;
    descriptor.batchSize = 8;
    descriptor.channelCount = 16;
    descriptor.spatialElementCount = 64;
    descriptor.inputDataType = TensorDescriptor::DataType::FP16;
    descriptor.outputDataType = TensorDescriptor::DataType::FP16;
    descriptor.parameterDataType = TensorDescriptor::DataType::FP32;
    descriptor.computeDataType = TensorDescriptor::DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.training = true;
    return descriptor;
}

}  // namespace

TEST(CudnnInstanceNormDescriptor, AcceptsFp16Bf16AndFp32IoWithFp32Parameters) {
    for (TensorDescriptor::DataType dtype : {TensorDescriptor::DataType::FP16,
                                             TensorDescriptor::DataType::BF16,
                                             TensorDescriptor::DataType::FP32}) {
        CudnnInstanceNormDescriptor descriptor = makeDescriptor();
        descriptor.inputDataType = dtype;
        descriptor.outputDataType = dtype;
        EXPECT_NO_THROW(descriptor.validateForward());
        EXPECT_NO_THROW(descriptor.validateBackward());
    }
}

TEST(CudnnInstanceNormDescriptor, RejectsEmptyShapeComponents) {
    CudnnInstanceNormDescriptor descriptor = makeDescriptor();
    descriptor.batchSize = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.channelCount = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.spatialElementCount = 0;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnInstanceNormDescriptor, RejectsUnsupportedIoDtype) {
    CudnnInstanceNormDescriptor descriptor = makeDescriptor();
    descriptor.inputDataType = TensorDescriptor::DataType::FP8_E4M3;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.outputDataType = TensorDescriptor::DataType::INT32;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnInstanceNormDescriptor, RejectsNonFp32ParametersOrCompute) {
    CudnnInstanceNormDescriptor descriptor = makeDescriptor();
    descriptor.parameterDataType = TensorDescriptor::DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.computeDataType = TensorDescriptor::DataType::FP16;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnInstanceNormDescriptor, RejectsNonPositiveEpsilon) {
    CudnnInstanceNormDescriptor descriptor = makeDescriptor();
    descriptor.epsilon = 0.0f;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.epsilon = -1.0e-5f;
    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}
