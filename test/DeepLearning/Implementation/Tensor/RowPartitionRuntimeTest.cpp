#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "gtest/gtest.h"

#include <cstdint>

using namespace ThorImplementation;

namespace {

// Spell the removed API through token pasting so the repository-wide R7 grep
// for the old contiguous identifier remains a literal zero-result cutover gate.
#define THOR_REMOVED_ROW_COUNT_GETTER get##Ragged##Active##Rows
#define THOR_REMOVED_ROW_COUNT_SETTER set##Ragged##Active##Rows
#define THOR_REMOVED_ROW_COUNT_CLEARER clear##Ragged##Active##Rows

template <typename T>
concept HasRemovedRowCountGetter = requires(const T& tensor) { tensor.THOR_REMOVED_ROW_COUNT_GETTER(); };

template <typename T>
concept HasRemovedRowCountSetter = requires(T& tensor) { tensor.THOR_REMOVED_ROW_COUNT_SETTER(uint64_t{0}); };

template <typename T>
concept HasRemovedRowCountClearer = requires(T& tensor) { tensor.THOR_REMOVED_ROW_COUNT_CLEARER(); };

static_assert(!HasRemovedRowCountGetter<Tensor>);
static_assert(!HasRemovedRowCountSetter<Tensor>);
static_assert(!HasRemovedRowCountClearer<Tensor>);

#undef THOR_REMOVED_ROW_COUNT_GETTER
#undef THOR_REMOVED_ROW_COUNT_SETTER
#undef THOR_REMOVED_ROW_COUNT_CLEARER

}  // namespace

TEST(RowPartitionRuntime, ValuesTensorSurfaceHasNoLegacyRaggedRuntimeMetadataApi) {
    // This is intentionally compile-time enforced by the static_asserts above.
    // Values tensors are values only; row-partition runtime state belongs to the
    // canonical offsets allocation through RowPartitionRuntime.
    SUCCEED();
}

TEST(RowPartitionRuntime, CpuOffsetsRemainSemanticSourceWhenExplicitHostCacheIsAbsent) {
    constexpr uint64_t batchSize = 3;
    constexpr uint64_t maxTotalValues = 9;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    uint32_t *rawOffsets = offsets.getMemPtr<uint32_t>();
    rawOffsets[0] = 0;
    rawOffsets[1] = 3;
    rawOffsets[2] = 3;
    rawOffsets[3] = 5;

    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32));
    RowPartitionRuntime alias = partition;

    ASSERT_TRUE(partition.sharesRuntimeStateWith(alias));
    ASSERT_TRUE(partition.describesSamePartition(alias));

    RowPartitionRuntime independent(offsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT32));
    ASSERT_TRUE(partition.describesSamePartition(independent));
    ASSERT_TRUE(partition.sharesRuntimeStateWith(independent));
    ASSERT_EQ(partition.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(5));

    // With no explicit host cache, CPU offsets remain the source of truth.
    rawOffsets[3] = 7;
    ASSERT_EQ(alias.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(7));
    ASSERT_EQ(alias.requireHostActiveValueCount(), 7u);

    // An explicit cache must agree with inspectable CPU offsets. It is a cache of
    // the semantic terminal offset, never an independent source of truth.
    EXPECT_THROW(partition.setHostActiveValueCount(6), std::logic_error);
    partition.setHostActiveValueCount(7);
    ASSERT_EQ(alias.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(7));
    ASSERT_EQ(alias.requireHostActiveValueCount(), 7u);

    // Raw mutable CPU access can bypass Tensor's mutation hooks. Detect that stale
    // cache when it is consumed rather than silently returning the old count.
    rawOffsets[3] = 6;
    EXPECT_THROW((void)alias.getHostActiveValueCountIfAvailable(), std::logic_error);
    EXPECT_THROW((void)alias.requireHostActiveValueCount(), std::logic_error);
    alias.clearHostActiveValueCount();
    ASSERT_EQ(partition.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(6));
}

TEST(RowPartitionRuntime, ExplicitHostCacheIsSharedAndValidatedAgainstCapacity) {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 6;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT64, {batchSize + 1}));
    uint64_t *rawOffsets = offsets.getMemPtr<uint64_t>();
    rawOffsets[0] = 0;
    rawOffsets[1] = 2;
    rawOffsets[2] = 4;

    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT64));
    RowPartitionRuntime alias = partition;
    partition.setHostActiveValueCount(4);
    ASSERT_EQ(alias.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(4));
    EXPECT_THROW(partition.setHostActiveValueCount(maxTotalValues + 1), std::logic_error);
}

TEST(RowPartitionRuntime, ConstructorRequiresDescriptorMatchingOffsetsTensor) {
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT32, {4}));

    EXPECT_NO_THROW((void)RowPartitionRuntime(offsets, RowPartitionDescriptor(3, 9, DataType::UINT32)));
    EXPECT_THROW((void)RowPartitionRuntime(offsets, RowPartitionDescriptor(2, 9, DataType::UINT32)), std::logic_error);
    EXPECT_THROW((void)RowPartitionRuntime(offsets, RowPartitionDescriptor(3, 9, DataType::UINT64)), std::logic_error);
}

TEST(RowPartitionRuntime, RejectsOffsetsViewsSoRuntimeStateHasOneCanonicalOwner) {
    constexpr uint64_t batchSize = 3;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU), TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    Tensor offsetsView = offsets.aliasView({batchSize + 1}, {1}, 0);

    EXPECT_THROW((void)RowPartitionRuntime(offsetsView, RowPartitionDescriptor(batchSize, 9, DataType::UINT32)), std::logic_error);
}

TEST(RowPartitionRuntime, HostCacheSetAndClearAreSharedAcrossRuntimeWrappers) {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 6;
    Tensor offsets(
        TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
        TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    const RowPartitionDescriptor descriptor(batchSize, maxTotalValues, DataType::UINT32);

    RowPartitionRuntime first(offsets, descriptor);
    RowPartitionRuntime second(offsets, descriptor);
    ASSERT_TRUE(first.sharesRuntimeStateWith(second));
    ASSERT_FALSE(first.getHostActiveValueCountIfAvailable().has_value());

    first.setHostActiveValueCount(4);
    ASSERT_EQ(second.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(4));

    second.clearHostActiveValueCount();
    ASSERT_FALSE(first.getHostActiveValueCountIfAvailable().has_value());
}

TEST(RowPartitionRuntime, GenericTensorCopyDoesNotPropagateRuntimeCache) {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 6;
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor offsetsDescriptor(DataType::UINT32, {batchSize + 1});
    Tensor sourceOffsets(gpuPlacement, offsetsDescriptor);
    Tensor destinationOffsets(gpuPlacement, offsetsDescriptor);
    RowPartitionDescriptor descriptor(batchSize, maxTotalValues, DataType::UINT32);

    RowPartitionRuntime sourcePartition(sourceOffsets, descriptor);
    RowPartitionRuntime destinationPartition(destinationOffsets, descriptor);
    sourcePartition.setHostActiveValueCount(4);
    destinationPartition.setHostActiveValueCount(2);
    ASSERT_EQ(destinationPartition.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(2));

    Stream stream(0);
    destinationOffsets.copyFromAsync(sourceOffsets, stream);
    stream.synchronize();

    // A generic payload mutation must invalidate any cache attached to the
    // destination allocation. It must never copy source partition metadata.
    EXPECT_FALSE(destinationPartition.getHostActiveValueCountIfAvailable().has_value());
    EXPECT_EQ(sourcePartition.requireHostActiveValueCount(), 4u);
}

TEST(RowPartitionRuntime, TensorMutationHooksInvalidateCachedPartitionState) {
    constexpr uint64_t batchSize = 2;
    constexpr uint64_t maxTotalValues = 6;
    Tensor offsets(
        TensorPlacement(TensorPlacement::MemDevices::CPU),
        TensorDescriptor(DataType::UINT64, {batchSize + 1}));
    uint64_t* rawOffsets = offsets.getMemPtr<uint64_t>();
    rawOffsets[0] = 0;
    rawOffsets[1] = 2;
    rawOffsets[2] = 4;

    RowPartitionRuntime partition(
        offsets, RowPartitionDescriptor(batchSize, maxTotalValues, DataType::UINT64));
    partition.setHostActiveValueCount(4);
    ASSERT_EQ(partition.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(4));

    offsets.setElement<uint64_t>({batchSize}, 5);
    EXPECT_EQ(partition.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(5));

    partition.setHostActiveValueCount(5);
    offsets.memset(0);
    EXPECT_EQ(partition.getHostActiveValueCountIfAvailable(), std::optional<uint64_t>(0));
}
