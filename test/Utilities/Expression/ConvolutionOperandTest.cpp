#include "Utilities/Expression/ConvolutionOperand.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;

namespace {

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

TEST(ConvolutionOperand, AWholeTensorConvertsWithItsOwnDimensions) {
    Tensor tensor(cpuPlacement, TensorDescriptor(DataType::FP32, {2, 3, 1, 4}));
    const ConvolutionOperand operand = tensor;
    EXPECT_TRUE(operand.storage.tensor == tensor);
    EXPECT_EQ(operand.storage.count, 24u);
    EXPECT_EQ(operand.dims, (std::vector<uint64_t>{2, 3, 1, 4}));
    EXPECT_EQ(operand.descriptor(), tensor.getDescriptor());
    EXPECT_EQ(operand.bytes(), tensor.getArraySizeInBytes());
    EXPECT_EQ(operand.address(), tensor.getMemPtr<void>());
    EXPECT_NO_THROW(requireCudnnContractOperand(operand, "test", "x"));
}

TEST(ConvolutionOperand, APrefixOfLargerStorageIsSeenWithTheGivenDimensions) {
    // A padded ragged value reserved for width 8, convolved at width 5: the first 2*3*5 elements, as [2, 3, 1, 5].
    Tensor storage(cpuPlacement, TensorDescriptor(DataType::FP16, {2 * 3 * 8}));
    const ConvolutionOperand operand(TensorSpan(storage, 0, 30), {2, 3, 1, 5});
    EXPECT_TRUE(operand.storage.tensor == storage);
    EXPECT_EQ(operand.descriptor(), TensorDescriptor(DataType::FP16, {2, 3, 1, 5}));
    EXPECT_EQ(operand.bytes(), 30u * 2u);
    EXPECT_EQ(operand.address(), storage.getMemPtr<void>());
    EXPECT_EQ(storage.getDimensions(), (std::vector<uint64_t>{48}));
    EXPECT_NO_THROW(requireCudnnContractOperand(operand, "test", "x"));

    // The span must hold exactly the dimensions' elements.
    EXPECT_THROW(ConvolutionOperand(TensorSpan(storage, 0, 30), {2, 3, 1, 4}), std::invalid_argument);
    EXPECT_THROW(ConvolutionOperand(TensorSpan(storage, 0, 30), {}), std::invalid_argument);
}

TEST(ConvolutionOperand, TheCudnnContractRejectsAMisalignedStart) {
    Tensor storage(cpuPlacement, TensorDescriptor(DataType::FP32, {64}));
    const ConvolutionOperand misaligned(TensorSpan(storage, 1, 8), {1, 2, 1, 4});
    EXPECT_THROW(requireCudnnContractOperand(misaligned, "test", "x"), std::invalid_argument);
    const ConvolutionOperand aligned(TensorSpan(storage, 4, 8), {1, 2, 1, 4});
    EXPECT_NO_THROW(requireCudnnContractOperand(aligned, "test", "x"));
}

}  // namespace
