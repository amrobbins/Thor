#include "DeepLearning/Implementation/Tensor/TensorSpan.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>

using namespace ThorImplementation;

namespace {

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

TEST(TensorSpan, AddressesARangeOfTheTensorWithoutAView) {
    Tensor buffer(cpuPlacement, TensorDescriptor(DataType::UINT64, {10}));
    uint64_t *words = buffer.getMemPtr<uint64_t>();
    for (uint64_t i = 0; i < 10; ++i) words[i] = 100 + i;

    const TensorSpan span(buffer, 4, 3);
    EXPECT_EQ(span.count, 3u);
    EXPECT_EQ(span.offset, 4u);
    EXPECT_EQ(span.getDataType(), DataType::UINT64);
    EXPECT_EQ(span.bytes(), 3 * sizeof(uint64_t));
    EXPECT_EQ(span.data<uint64_t>(), words + 4);
    EXPECT_EQ(span.address(), static_cast<void *>(words + 4));
    EXPECT_EQ(span.data<uint64_t>()[2], 106u);
    // The span shares the buffer: writes through it land in the tensor.
    span.data<uint64_t>()[0] = 7;
    EXPECT_EQ(words[4], 7u);
    // The tensor itself is untouched: same dimensions, same start.
    EXPECT_EQ(buffer.getDimensions(), (std::vector<uint64_t>{10}));
    EXPECT_EQ(span.tensor.getMemPtr<uint64_t>(), words);
}

TEST(TensorSpan, AWholeTensorConvertsImplicitly) {
    Tensor buffer(cpuPlacement, TensorDescriptor(DataType::UINT8, {2, 6}));
    const TensorSpan span = buffer;
    EXPECT_EQ(span.offset, 0u);
    EXPECT_EQ(span.count, 12u);
    EXPECT_EQ(span.bytes(), 12u);
    EXPECT_EQ(span.address(), buffer.getMemPtr<void>());
    EXPECT_NO_THROW(span.validate("whole"));

    const TensorSpan empty = Tensor();
    EXPECT_FALSE(empty.isInitialized());
    EXPECT_EQ(empty.count, 0u);
    EXPECT_EQ(empty.bytes(), 0u);
    EXPECT_THROW(empty.validate("empty"), std::invalid_argument);
}

TEST(TensorSpan, RejectsRangesOutsideTheTensor) {
    Tensor buffer(cpuPlacement, TensorDescriptor(DataType::UINT64, {8}));
    EXPECT_NO_THROW(TensorSpan(buffer, 0, 8));
    EXPECT_NO_THROW(TensorSpan(buffer, 8, 0));
    EXPECT_NO_THROW(TensorSpan(buffer, 5, 3));
    EXPECT_THROW(TensorSpan(buffer, 5, 4), std::invalid_argument);
    EXPECT_THROW(TensorSpan(buffer, 9, 0), std::invalid_argument);
    EXPECT_THROW(TensorSpan(buffer, 1, UINT64_MAX), std::invalid_argument);
    EXPECT_THROW(TensorSpan(Tensor(), 0, 0), std::invalid_argument);
}

}  // namespace
