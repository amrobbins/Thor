#include "Utilities/TensorOperations/DeepLearning/CudnnRmsNormRhtAbsMax.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

namespace {

CudnnRmsNormRhtAbsMaxDescriptor makeDescriptor() {
    CudnnRmsNormRhtAbsMaxDescriptor descriptor;
    descriptor.outerSize = 8;
    descriptor.normalizedFeatureCount = 2048;
    descriptor.inputDataType = TensorDescriptor::DataType::BF16;
    descriptor.outputDataType = TensorDescriptor::DataType::BF16;
    descriptor.parameterDataType = TensorDescriptor::DataType::BF16;
    descriptor.absMaxDataType = TensorDescriptor::DataType::FP32;
    descriptor.epsilon = 1.0e-5f;
    descriptor.numThreads = 128;
    descriptor.rowsPerCta = 2;
    return descriptor;
}

}  // namespace

TEST(CudnnRmsNormRhtAmaxDescriptor, AcceptsDocumentedBf16Fp32Surface) {
    CudnnRmsNormRhtAbsMaxDescriptor descriptor = makeDescriptor();

    EXPECT_NO_THROW(descriptor.validate());
    EXPECT_EQ(descriptor.resolvedNumThreads(), 128u);
    EXPECT_EQ(descriptor.resolvedRowsPerCta(), 2u);
    EXPECT_EQ(descriptor.absMaxElementCount(), 4u);
}

TEST(CudnnRmsNormRhtAmaxDescriptor, ResolvesDocumentedThreadTable) {
    struct Case {
        uint64_t n;
        uint32_t expected_threads;
    };
    for (const Case c : {Case{2048, 128}, Case{4096, 256}, Case{7168, 128}, Case{8192, 512}, Case{16384, 1024}, Case{32768, 512}}) {
        CudnnRmsNormRhtAbsMaxDescriptor descriptor = makeDescriptor();
        descriptor.normalizedFeatureCount = c.n;
        descriptor.numThreads = 0;
        EXPECT_EQ(descriptor.resolvedNumThreads(), c.expected_threads) << "N=" << c.n;
        EXPECT_NO_THROW(descriptor.validate()) << "N=" << c.n;
    }
}

TEST(CudnnRmsNormRhtAmaxDescriptor, RejectsUnsupportedDtypes) {
    CudnnRmsNormRhtAbsMaxDescriptor descriptor = makeDescriptor();
    descriptor.inputDataType = TensorDescriptor::DataType::FP16;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.parameterDataType = TensorDescriptor::DataType::FP32;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.outputDataType = TensorDescriptor::DataType::FP32;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.absMaxDataType = TensorDescriptor::DataType::BF16;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);
}

TEST(CudnnRmsNormRhtAmaxDescriptor, RejectsInvalidShapeAndLaunchConstraints) {
    CudnnRmsNormRhtAbsMaxDescriptor descriptor = makeDescriptor();
    descriptor.outerSize = 0;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.normalizedFeatureCount = 2050;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.numThreads = 127;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.numThreads = 512;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.rowsPerCta = 3;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);

    descriptor = makeDescriptor();
    descriptor.outerSize = 7;
    descriptor.rowsPerCta = 2;
    EXPECT_THROW(descriptor.validate(), std::invalid_argument);
}

TEST(CudnnRmsNormRhtAmaxDescriptor, ResolvesRowsPerCtaAndAbsoluteMaxElements) {
    CudnnRmsNormRhtAbsMaxDescriptor descriptor = makeDescriptor();
    descriptor.outerSize = 12;
    descriptor.rowsPerCta = 0;
    EXPECT_EQ(descriptor.resolvedRowsPerCta(), 2u);
    EXPECT_EQ(descriptor.absMaxElementCount(), 6u);

    descriptor.outerSize = 9;
    uint32_t unused;
    EXPECT_THROW(unused = descriptor.resolvedRowsPerCta(), std::invalid_argument);
}

TEST(CudnnRmsNormRhtAmaxDescriptor, AmaxIsAbsoluteMaximumScaleMetadata) {
    CudnnRmsNormRhtAbsMaxDescriptor descriptor = makeDescriptor();
    descriptor.outerSize = 16;
    descriptor.rowsPerCta = 4;

    EXPECT_EQ(descriptor.absMaxDataType, TensorDescriptor::DataType::FP32);
    EXPECT_EQ(descriptor.absMaxElementCount(), 4u);
    EXPECT_NO_THROW(descriptor.validate());
}
