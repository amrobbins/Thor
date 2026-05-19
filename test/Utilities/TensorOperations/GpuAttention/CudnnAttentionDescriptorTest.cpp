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

AttentionTensorSpec scoreBiasSpec(std::vector<int64_t> dims) {
    AttentionTensorSpec spec;
    spec.dimensions = dims;
    spec.strides.resize(spec.dimensions.size(), 1);
    for (int64_t i = static_cast<int64_t>(spec.dimensions.size()) - 2; i >= 0; --i) {
        spec.strides[static_cast<size_t>(i)] = spec.strides[static_cast<size_t>(i + 1)] * spec.dimensions[static_cast<size_t>(i + 1)];
    }
    spec.dataType = TensorDescriptor::DataType::FP32;
    spec.ragged = false;
    return spec;
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

TEST(CudnnAttentionDescriptor, AllowsAdditiveBiasBroadcastSurface) {
    const std::vector<std::vector<int64_t>> allowed{{1, 1, 64, 80}, {1, 4, 64, 80}, {3, 1, 64, 80}, {3, 4, 64, 80}};
    for (const auto& dims : allowed) {
        CudnnAttentionDescriptor descriptor = makeDescriptor();
        descriptor.useBias = true;
        descriptor.bias = scoreBiasSpec(dims);
        EXPECT_NO_THROW(descriptor.validateForward()) << "bias dims " << testing::PrintToString(dims);
    }
}

TEST(CudnnAttentionDescriptor, RejectsInvalidAdditiveBiasBroadcastShape) {
    CudnnAttentionDescriptor descriptor = makeDescriptor();
    descriptor.useBias = true;
    descriptor.bias = scoreBiasSpec({3, 2, 64, 80});

    EXPECT_THROW(descriptor.validateForward(), std::invalid_argument);
}

TEST(CudnnAttentionDescriptor, AllowsRaggedAttentionWithFullDenseAdditiveBias) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;
    descriptor.useBias = true;

    EXPECT_NO_THROW(descriptor.validateForward());
}

TEST(CudnnAttentionDescriptor, RejectsRaggedAttentionBackwardWithFullDenseAdditiveBias) {
    CudnnAttentionDescriptor descriptor = makePackedDescriptor();
    descriptor.q.ragged = true;
    descriptor.o.ragged = true;
    descriptor.k.ragged = true;
    descriptor.v.ragged = true;
    descriptor.usePaddingMask = true;
    descriptor.useBias = true;
    descriptor.generateStats = true;

    EXPECT_THROW(descriptor.validateBackward(), std::invalid_argument);
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
