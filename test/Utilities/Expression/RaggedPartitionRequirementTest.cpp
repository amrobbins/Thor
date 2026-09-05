#include "Utilities/Expression/FusedEquation.h"
#include "DeepLearning/Implementation/Layers/Loss/CtcLoss.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/DropOut.h"
#include "DeepLearning/Implementation/Layers/Utility/FiniteCheckKernel.h"
#include "Utilities/TensorOperations/DeepLearning/CudnnRaggedSoftmax.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingKernels.h"
#include "Utilities/TensorOperations/Embedding/EmbeddingSparseGradient.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnAttention.h"
#include "Utilities/TensorOperations/GpuAttention/CudnnRaggedAttentionMetadata.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/BucketedCublasGemm.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/RaggedMatmulCapacityBuckets.h"
#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequence.h"
#include "Utilities/TensorOperations/Ragged/PaddedRaggedSequenceKernel.h"
#include "Utilities/TensorOperations/Ragged/RaggedAccuracy.h"
#include "Utilities/TensorOperations/Ragged/RaggedConv1dWidthCapacity.h"
#include "Utilities/TensorOperations/Ragged/RaggedConcatenate.h"
#include "Utilities/TensorOperations/Ragged/RaggedDenseAdapters.h"
#include "Utilities/TensorOperations/Ragged/RaggedGather.h"
#include "Utilities/TensorOperations/Ragged/RaggedSequenceConcatenate.h"
#include "Utilities/TensorOperations/Ragged/RaggedSequenceSlice.h"
#include "Utilities/TensorOperations/Ragged/RaggedWeightedReduction.h"
#include "Utilities/TensorOperations/Ragged/RuntimeExtent.h"
#include "Utilities/TensorOperations/Ragged/RowPartition.h"

#include <gtest/gtest.h>

#include <memory>

using namespace ThorImplementation;

namespace {

TEST(RaggedPartitionRequirementRP5, RequirementFlagsAreIndependentAndComposable) {
    const RaggedPartitionRequirement combined =
        RaggedPartitionRequirement::HOST_EXTENT | RaggedPartitionRequirement::DEVICE_OFFSETS;

    EXPECT_TRUE(hasRaggedPartitionRequirement(combined, RaggedPartitionRequirement::HOST_EXTENT));
    EXPECT_TRUE(hasRaggedPartitionRequirement(combined, RaggedPartitionRequirement::DEVICE_OFFSETS));
    EXPECT_FALSE(hasRaggedPartitionRequirement(combined, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT));
    EXPECT_TRUE(consumesAnyRaggedPartitionInformation(combined));
    EXPECT_FALSE(consumesAnyRaggedPartitionInformation(RaggedPartitionRequirement::NONE));
}

TEST(RaggedPartitionRequirementRP5, DirectRaggedOperatorsDeclareOnlyThePhysicalInformationTheyConsume) {
    EXPECT_EQ(kRaggedRuntimeExtentPartitionRequirement, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    EXPECT_EQ(kRowPartitionActiveScalarRequirement, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    EXPECT_EQ(kRaggedDropOutPartitionRequirement, RaggedPartitionRequirement::HOST_EXTENT);
    EXPECT_EQ(kRaggedFiniteCheckGpuPartitionRequirement, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    EXPECT_EQ(kRaggedEmbeddingPartitionRequirement, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    EXPECT_EQ(kRaggedEmbeddingSparseGradientPartitionRequirement, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    EXPECT_EQ(kRaggedTrailingConcatenatePartitionRequirement, RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    EXPECT_EQ(kRaggedAccuracyPartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kRaggedWeightedReductionPartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);

    EXPECT_EQ(kCudnnRaggedSoftmaxPartitionRequirement, RaggedPartitionRequirement::HOST_EXTENT);
    EXPECT_EQ(kRaggedMatmulCapacitySelectionPartitionRequirement, RaggedPartitionRequirement::HOST_EXTENT);
    EXPECT_EQ(kBucketedCublasGemmPartitionRequirement, RaggedPartitionRequirement::HOST_EXTENT);
    EXPECT_EQ(kRaggedConv1dWidthSelectionPartitionRequirement, RaggedPartitionRequirement::HOST_EXTENT);
    EXPECT_EQ(kPaddedRaggedSequencePlanningPartitionRequirement, RaggedPartitionRequirement::HOST_EXTENT);

    EXPECT_EQ(kCtcLossLabelPartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kRaggedGatherPartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kCudnnRaggedAttentionPartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kCudnnRaggedAttentionMetadataPartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kRaggedSequenceSlicePartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kRaggedSequenceConcatenatePartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kRaggedDenseAdapterPartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kPaddedRaggedSequenceKernelPartitionRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(kRowPartitionOffsetsTransformRequirement, RaggedPartitionRequirement::DEVICE_OFFSETS);

    EXPECT_EQ(StampedPaddedRaggedPack::raggedPartitionRequirement(),
              RaggedPartitionRequirement::HOST_EXTENT | RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(StampedPaddedRaggedUnpack::raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(StampedPaddedRaggedPointwise::raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    EXPECT_EQ(StampedSanitizePackedTail::raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);
}

TEST(RaggedPartitionRequirementRP5, CompilerPayloadsDistinguishScalarExtentFromFullOffsets) {
    CompiledEquation fused;
    EXPECT_EQ(fused.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    fused.uses_device_runtime_extent = true;
    EXPECT_EQ(fused.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);
    fused.device_runtime_extent_source = RaggedRuntimeExtentSource::DEVICE_OFFSETS;
    EXPECT_EQ(fused.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_OFFSETS);
    fused.device_runtime_extent_source = RaggedRuntimeExtentSource::DEVICE_ACTIVE_COUNT;

    CompiledSoftmax denseSoftmax(CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_INSTANCE,
                                 DataType::FP32,
                                 DataType::FP32);
    EXPECT_EQ(denseSoftmax.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);

    CompiledSoftmax raggedSoftmax(CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_INSTANCE,
                                  DataType::FP32,
                                  DataType::FP32,
                                  false,
                                  1,
                                  8,
                                  64,
                                  4);
    EXPECT_EQ(raggedSoftmax.raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);

    CompiledRmsNorm rms;
    CompiledLayerNorm layer;
    CompiledRmsNormBackward rmsBackward;
    EXPECT_EQ(rms.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    EXPECT_EQ(layer.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    EXPECT_EQ(rmsBackward.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    rms.ragged_offsets_input_slot = 2;
    layer.ragged_offsets_input_slot = 3;
    rmsBackward.ragged_offsets_input_slot = 3;
    EXPECT_EQ(rms.raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);
    EXPECT_EQ(layer.raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);
    EXPECT_EQ(rmsBackward.raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);

    CompiledMatmul denseMatmul(ExprOp::MATMUL,
                               false,
                               false,
                               false,
                               1.0,
                               0.0,
                               UINT32_MAX,
                               UINT32_MAX,
                               DataType::FP32,
                               DataType::FP32,
                               DataType::FP32,
                               DataType::FP32,
                               DataType::FP32,
                               std::nullopt);
    CompiledMatmul raggedMatmul(ExprOp::MATMUL,
                                false,
                                false,
                                false,
                                1.0,
                                0.0,
                                UINT32_MAX,
                                UINT32_MAX,
                                DataType::FP32,
                                DataType::FP32,
                                DataType::FP32,
                                DataType::FP32,
                                DataType::FP32,
                                std::nullopt,
                                MatmulEpilogue::Default,
                                MatmulBackwardEpilogue::Default,
                                UINT32_MAX,
                                MatmulPackedRowBinding::RowsA,
                                64,
                                2,
                                8);
    EXPECT_EQ(denseMatmul.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    EXPECT_EQ(raggedMatmul.raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);

    CompiledAttention attention;
    CompiledAttentionBackward attentionBackward;
    EXPECT_EQ(attention.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    EXPECT_EQ(attentionBackward.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);
    attention.use_ragged_offsets = true;
    attentionBackward.use_ragged_offsets = true;
    EXPECT_EQ(attention.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(attentionBackward.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_OFFSETS);
}

TEST(RaggedPartitionRequirementRP5, StructuralAndPaddedCompilerStagesDeclareFullBoundaryNeeds) {
    CompiledSegmentedReduction segmentedReduction(
        ExprOp::SEGMENTED_REDUCE_SUM, DataType::FP32, DataType::FP32, DataType::UINT32, 1);
    CompiledSegmentedBroadcast segmentedBroadcast(DataType::FP32, DataType::FP32, DataType::UINT32, 64, 1, false);
    CompiledScan segmentedScan(ScanOp::Sum,
                               ScanMode::Inclusive,
                               0,
                               false,
                               true,
                               DataType::FP32,
                               DataType::FP32,
                               DataType::UINT32);
    CompiledScan denseScan(ScanOp::Sum, ScanMode::Inclusive, 0, false, false, DataType::FP32, DataType::FP32);

    EXPECT_EQ(segmentedReduction.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(segmentedBroadcast.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(segmentedScan.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_OFFSETS);
    EXPECT_EQ(denseScan.raggedPartitionRequirement(), RaggedPartitionRequirement::NONE);

    CompiledRaggedConv1dCausal conv(DataType::FP16,
                                    DataType::FP16,
                                    DataType::FP16,
                                    DataType::FP32,
                                    DataType::UINT32,
                                    8,
                                    64,
                                    16,
                                    4,
                                    8,
                                    3,
                                    1,
                                    1);
    EXPECT_EQ(conv.raggedPartitionRequirement(),
              RaggedPartitionRequirement::HOST_EXTENT | RaggedPartitionRequirement::DEVICE_OFFSETS);
}

TEST(RaggedPartitionRequirementRP5, CompiledExecutionPlanAggregatesStageRequirementsForLaterPhysicalization) {
    PhysicalExpression expr;
    auto fused = std::make_shared<CompiledEquation>();
    fused->uses_device_runtime_extent = true;
    CompiledExecutionStage fusedStage(expr, fused, std::vector<uint32_t>{}, std::vector<CompiledStageOutput>{});
    EXPECT_EQ(fusedStage.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT);

    auto softmax = std::make_shared<CompiledSoftmax>(CUDNN_SOFTMAX_ACCURATE,
                                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                                     DataType::FP32,
                                                     DataType::FP32,
                                                     false,
                                                     1,
                                                     8,
                                                     64,
                                                     4);
    CompiledExecutionStage softmaxStage(softmax, std::vector<uint32_t>{}, std::vector<CompiledStageOutput>{});
    EXPECT_EQ(softmaxStage.raggedPartitionRequirement(), RaggedPartitionRequirement::HOST_EXTENT);

    auto segmented = std::make_shared<CompiledSegmentedBroadcast>(
        DataType::FP32, DataType::FP32, DataType::UINT32, 64, 1, false);
    CompiledExecutionStage segmentedStage(segmented, std::vector<uint32_t>{}, std::vector<CompiledStageOutput>{});
    EXPECT_EQ(segmentedStage.raggedPartitionRequirement(), RaggedPartitionRequirement::DEVICE_OFFSETS);

    CompiledOutputs outputs;
    outputs.stages.emplace_back(std::move(fusedStage));
    outputs.stages.emplace_back(std::move(softmaxStage));
    outputs.stages.emplace_back(std::move(segmentedStage));
    EXPECT_EQ(outputs.raggedPartitionRequirement(),
              RaggedPartitionRequirement::HOST_EXTENT | RaggedPartitionRequirement::DEVICE_ACTIVE_COUNT |
                  RaggedPartitionRequirement::DEVICE_OFFSETS);
}

}  // namespace
