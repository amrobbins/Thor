#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

namespace {

CudnnAttentionDescriptor makeDescriptor() {
    CudnnAttentionDescriptor descriptor;
    descriptor.q = AttentionTensorSpec::bhsd(3, 4, 64, 64, TensorDescriptor::DataType::FP16);
    descriptor.k = AttentionTensorSpec::bhsd(3, 4, 80, 64, TensorDescriptor::DataType::FP16);
    descriptor.v = AttentionTensorSpec::bhsd(3, 4, 80, 64, TensorDescriptor::DataType::FP16);
    descriptor.o = AttentionTensorSpec::bhsd(3, 4, 64, 64, TensorDescriptor::DataType::FP16);
    descriptor.computeDataType = TensorDescriptor::DataType::FP32;
    descriptor.intermediateDataType = TensorDescriptor::DataType::FP32;
    return descriptor;
}

CudnnAttentionDescriptor makePackedDescriptor() {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.q = AttentionTensorSpec::bshd(3, 4, 64, 64, TensorDescriptor::DataType::FP16);
    descriptor.k = AttentionTensorSpec::bshd(3, 4, 80, 64, TensorDescriptor::DataType::FP16);
    descriptor.v = AttentionTensorSpec::bshd(3, 4, 80, 64, TensorDescriptor::DataType::FP16);
    descriptor.o = AttentionTensorSpec::bshd(3, 4, 64, 64, TensorDescriptor::DataType::FP16);
    return descriptor;
}

CudnnAttentionDescriptor makePagedDescriptor() {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.k = AttentionTensorSpec::bhsd(6, 4, 64, 64, TensorDescriptor::DataType::FP16);
    descriptor.v = AttentionTensorSpec::bhsd(6, 4, 64, 64, TensorDescriptor::DataType::FP16);
    descriptor.usePaddingMask = true;
    descriptor.usePagedKvCache = true;
    descriptor.pagedKv.maxSequenceLengthKv = 128;
    return descriptor;
}

}  // namespace

TEST(CudnnAttentionDescriptor, AllowsPackedRaggedQOAndKVOffsetPairs) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;

    EXPECT_NO_THROW(descriptor.validateForward());
}


TEST(CudnnAttentionDescriptor, RejectsRaggedOffsetsWithoutBshdPackedLayout) {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedQWithoutOutputOffset) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedOutputWithoutQOffset) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.o.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedKWithoutVOffset) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.k.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedVWithoutKOffset) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.v.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedOffsetsWithoutPaddingMaskSequenceLengthMode) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsRaggedAttentionWithAdditiveBiasUntilPackedBiasLayoutIsDefined) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;
    descriptor.useBias = true;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsCombiningRaggedOffsetsWithPagedKvCache) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;
    descriptor.usePagedKvCache = true;
    descriptor.pagedKv.maxSequenceLengthKv = 128;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, AllowsPagedKvForwardDescriptorWithPositiveMaxSequenceLength) {
    CudnnAttentionDescriptor descriptor = makePagedDescriptor();

    EXPECT_NO_THROW(descriptor.validateForward());
}

TEST(CudnnAttentionDescriptor, RejectsPagedKvForwardDescriptorWithoutPositiveMaxSequenceLength) {
    CudnnAttentionDescriptor descriptor = makePagedDescriptor();
    descriptor.pagedKv.maxSequenceLengthKv = 0;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsPagedKvForwardDescriptorWithoutPaddingMaskSequenceLengths) {
    CudnnAttentionDescriptor descriptor = makePagedDescriptor();
    descriptor.usePaddingMask = false;

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, RejectsPagedKvBackwardUntilTrainingSemanticsAreDefined) {
    CudnnAttentionDescriptor descriptor = makePagedDescriptor();
    descriptor.generateStats = true;

    EXPECT_THROW(descriptor.validateBackward(), std::invalid_argument);
}
