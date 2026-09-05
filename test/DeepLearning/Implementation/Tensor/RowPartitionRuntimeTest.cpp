#include "DeepLearning/Implementation/Tensor/RowPartitionRuntime.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

using namespace ThorImplementation;

namespace {

#define THOR_REMOVED_ACTIVE_SETTER set##Host##Active##Value##Count
#define THOR_REMOVED_ACTIVE_CLEARER clear##Host##Active##Value##Count
#define THOR_REMOVED_MAX_SETTER set##Host##Max##Active##Row##Length
#define THOR_REMOVED_MAX_CLEARER clear##Host##Max##Active##Row##Length

template <typename T>
concept HasRemovedActiveSetter = requires(T& partition) { partition.THOR_REMOVED_ACTIVE_SETTER(uint64_t{0}); };
template <typename T>
concept HasRemovedActiveClearer = requires(T& partition) { partition.THOR_REMOVED_ACTIVE_CLEARER(); };
template <typename T>
concept HasRemovedMaxSetter = requires(T& partition) { partition.THOR_REMOVED_MAX_SETTER(uint64_t{0}); };
template <typename T>
concept HasRemovedMaxClearer = requires(T& partition) { partition.THOR_REMOVED_MAX_CLEARER(); };

template <typename T>
concept HasPartitionGenerationAccessor = requires(const T& partition) {
    partition.getHostPartitionGenerationIfAvailable();
};

static_assert(!HasRemovedActiveSetter<RowPartitionRuntime>);
static_assert(!HasRemovedActiveClearer<RowPartitionRuntime>);
static_assert(!HasRemovedMaxSetter<RowPartitionRuntime>);
static_assert(!HasRemovedMaxClearer<RowPartitionRuntime>);

static_assert(!HasPartitionGenerationAccessor<RowPartitionRuntime>);

#undef THOR_REMOVED_ACTIVE_SETTER
#undef THOR_REMOVED_ACTIVE_CLEARER
#undef THOR_REMOVED_MAX_SETTER
#undef THOR_REMOVED_MAX_CLEARER

}  // namespace

TEST(RowPartitionRuntime, HostOffsetsAreSingleAuthoritativePublicationAndDeriveScalars) {
    constexpr uint64_t batchSize = 3;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
                   TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    RowPartitionRuntime first(offsets, RowPartitionDescriptor(batchSize, 12, DataType::UINT32, 6));
    RowPartitionRuntime alias(offsets, first.getDescriptor());

    EXPECT_FALSE(first.hasHostOffsets());
    EXPECT_FALSE(first.getHostActiveValueCountIfAvailable().has_value());
    EXPECT_FALSE(first.getHostMaxActiveRowLengthIfAvailable().has_value());

    first.setHostOffsets({0, 2, 2, 7});
    EXPECT_TRUE(alias.hasHostOffsets());
    EXPECT_EQ(alias.requireHostOffsets(), (std::vector<uint64_t>{0, 2, 2, 7}));
    EXPECT_EQ(alias.requireHostActiveValueCount(), 7u);
    EXPECT_EQ(alias.requireHostMaxActiveRowLength(), 5u);

    alias.setHostOffsets({0, 1, 5, 5});
    EXPECT_EQ(first.requireHostActiveValueCount(), 5u);
    EXPECT_EQ(first.requireHostMaxActiveRowLength(), 4u);
}

TEST(RowPartitionRuntime, RebindingValidatesCompletePartition) {
    constexpr uint64_t batchSize = 3;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
                   TensorDescriptor(DataType::UINT64, {batchSize + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batchSize, 9, DataType::UINT64, 4));

    EXPECT_THROW(partition.setHostOffsets({0, 1, 2}), std::logic_error);
    EXPECT_THROW(partition.setHostOffsets({1, 1, 2, 3}), std::logic_error);
    EXPECT_THROW(partition.setHostOffsets({0, 3, 2, 4}), std::logic_error);
    EXPECT_THROW(partition.setHostOffsets({0, 5, 5, 5}), std::logic_error);
    EXPECT_THROW(partition.setHostOffsets({0, 4, 8, 10}), std::logic_error);

    EXPECT_NO_THROW(partition.setHostOffsets({0, 4, 4, 8}));
    EXPECT_EQ(partition.requireHostActiveValueCount(), 8u);
    EXPECT_EQ(partition.requireHostMaxActiveRowLength(), 4u);
}

TEST(RowPartitionRuntime, ShortLongAllEmptyShortRebindingUsesOneSharedState) {
    constexpr uint64_t batchSize = 3;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
                   TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    RowPartitionRuntime first(offsets, RowPartitionDescriptor(batchSize, 16, DataType::UINT32));
    RowPartitionRuntime second(offsets, first.getDescriptor());

    const std::vector<std::vector<uint64_t>> transitions = {
        {0, 1, 1, 3},
        {0, 5, 9, 13},
        {0, 0, 0, 0},
        {0, 2, 2, 4},
    };
    for (const auto& hostOffsets : transitions) {
        first.setHostOffsets(hostOffsets);
        EXPECT_EQ(second.requireHostOffsets(), hostOffsets);
        EXPECT_EQ(second.requireHostActiveValueCount(), hostOffsets.back());
    }
}

TEST(RowPartitionRuntime, GenericOffsetsTensorMutationDoesNotRedefineHostPartition) {
    constexpr uint64_t batchSize = 2;
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::CPU),
                   TensorDescriptor(DataType::UINT32, {batchSize + 1}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(batchSize, 8, DataType::UINT32));
    partition.setHostOffsets({0, 2, 5});

    uint32_t* raw = offsets.getMemPtr<uint32_t>();
    raw[0] = 0;
    raw[1] = 1;
    raw[2] = 1;
    offsets.memset(0);

    EXPECT_EQ(partition.requireHostOffsets(), (std::vector<uint64_t>{0, 2, 5}));
    EXPECT_EQ(partition.requireHostActiveValueCount(), 5u);
    EXPECT_EQ(partition.requireHostMaxActiveRowLength(), 3u);
}

TEST(RowPartitionRuntime, GenericTensorCopyDoesNotCopyOrInvalidateAuthoritativeHostPartition) {
    constexpr uint64_t batchSize = 2;
    TensorPlacement gpu(TensorPlacement::MemDevices::GPU, 0);
    TensorDescriptor descriptor(DataType::UINT32, {batchSize + 1});
    Tensor sourceOffsets(gpu, descriptor);
    Tensor destinationOffsets(gpu, descriptor);
    RowPartitionDescriptor rowDescriptor(batchSize, 8, DataType::UINT32);
    RowPartitionRuntime source(sourceOffsets, rowDescriptor);
    RowPartitionRuntime destination(destinationOffsets, rowDescriptor);
    source.setHostOffsets({0, 1, 6});
    destination.setHostOffsets({0, 2, 4});

    Stream stream(0);
    destinationOffsets.copyFromAsync(sourceOffsets, stream);
    stream.synchronize();

    EXPECT_EQ(source.requireHostOffsets(), (std::vector<uint64_t>{0, 1, 6}));
    EXPECT_EQ(destination.requireHostOffsets(), (std::vector<uint64_t>{0, 2, 4}));
}

TEST(RowPartitionRuntime, FreshRuntimeIsUnboundUntilAuthoritativeHostOffsetsArePublished) {
    Tensor offsets(TensorPlacement(TensorPlacement::MemDevices::GPU, 0),
                   TensorDescriptor(DataType::UINT32, {3}));
    RowPartitionRuntime partition(offsets, RowPartitionDescriptor(2, 8, DataType::UINT32));

    EXPECT_FALSE(partition.hasHostOffsets());
    EXPECT_FALSE(partition.getHostOffsetsIfAvailable().has_value());
    EXPECT_FALSE(partition.getHostActiveValueCountIfAvailable().has_value());
    EXPECT_FALSE(partition.getHostMaxActiveRowLengthIfAvailable().has_value());
    EXPECT_THROW((void)partition.requireHostOffsets(), std::runtime_error);
    EXPECT_THROW((void)partition.requireHostActiveValueCount(), std::runtime_error);
    EXPECT_THROW((void)partition.requireHostMaxActiveRowLength(), std::runtime_error);

    partition.setHostOffsets({0, 1, 3});
    EXPECT_EQ(partition.requireHostOffsets(), (std::vector<uint64_t>{0, 1, 3}));
}

TEST(RowPartitionRuntime, ConstructorRequiresDescriptorMatchingCanonicalOffsetsAllocation) {
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
