#include "Utilities/TensorOperations/DeepLearning/CudnnTensorContract.h"

#include "gtest/gtest.h"

using namespace ThorImplementation;

namespace {

const TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

}  // namespace

// Every Tensor is dense and starts on its allocation's (at least 256-byte) boundary, so every initialized Tensor
// meets cuDNN's contract; only an uninitialized one fails it.
TEST(CudnnTensorContract, AcceptsTensorsAndRejectsUninitialized) {
    Tensor storage(cpuPlacement, TensorDescriptor(DataType::FP32, {64}), 256);
    EXPECT_TRUE(meetsCudnnTensorContract(storage));
    EXPECT_FALSE(meetsCudnnTensorContract(Tensor()));
    EXPECT_NO_THROW(requireCudnnContractTensor(storage, "Op", "x"));
}
