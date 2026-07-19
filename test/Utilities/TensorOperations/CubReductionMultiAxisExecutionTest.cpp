#include "test/Utilities/TensorOperations/CubReductionTestSupport.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ThorImplementation;
using namespace ThorImplementation::CubReductionTestSupport;

TEST(CubReduction, MultiAxisContiguousSuffixUsesFixedSegments) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 2, 3},
                                        stream);

    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{1, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::ContiguousFixedSegment);
    EXPECT_EQ(stamped->getGeometry().reduction_size, 6U);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{2, 1, 1}));

    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {21.0f, 57.0f});
}

TEST(CubReduction, MultiAxisDisjointAndLeadingAxesUseLogicalIndexMapping) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);

    std::shared_ptr<StampedCubReduction> disjoint =
        CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(disjoint->getPath(), CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(disjoint->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 3, 1}));
    disjoint->run();

    std::shared_ptr<StampedCubReduction> leading =
        CubReduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 1}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(leading->getPath(), CubReductionPath::StridedFixedSegment);
    EXPECT_EQ(leading->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 1, 2}));
    leading->run();
    stream.synchronize();

    expectFloatVectorNear(copyGpuTensorAsFloat(disjoint->getOutputTensor(), stream), {18.0f, 26.0f, 34.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(leading->getOutputTensor(), stream), {36.0f, 42.0f});
}

TEST(CubReduction, MultiAxisAllAxesUsesDeviceTransformReduce) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);

    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Mean, std::vector<uint32_t>{0, 1, 2}, DataType::FP32).stamp(input, stream);
    EXPECT_EQ(stamped->getPath(), CubReductionPath::DeviceTransformReduce);
    EXPECT_EQ(stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 1, 1}));
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(stamped->getOutputTensor(), stream), {6.5f});
}

TEST(CubReduction, PreallocatedOutputAcceptsKeepDimensionOrSqueezedShape) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                        {2, 3, 2},
                                        stream);
    CubReduction reduction(CubReductionOp::Sum, std::vector<uint32_t>{0, 2}, DataType::FP32);

    Tensor keep_dimensions(gpuPlacement, TensorDescriptor(DataType::FP32, {1, 3, 1}));
    Tensor squeezed(gpuPlacement, TensorDescriptor(DataType::FP32, {3}));
    Tensor wrong_shape(gpuPlacement, TensorDescriptor(DataType::FP32, {3, 1}));

    std::shared_ptr<StampedCubReduction> keep_stamped = reduction.stamp(input, keep_dimensions, stream);
    std::shared_ptr<StampedCubReduction> squeezed_stamped = reduction.stamp(input, squeezed, stream);
    EXPECT_EQ(keep_stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{1, 3, 1}));
    EXPECT_EQ(squeezed_stamped->getOutputTensor().getDimensions(), (std::vector<uint64_t>{3}));
    EXPECT_THROW(static_cast<void>(reduction.stamp(input, wrong_shape, stream)), std::invalid_argument);

    keep_stamped->run();
    squeezed_stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(keep_dimensions, stream), {18.0f, 26.0f, 34.0f});
    expectFloatVectorNear(copyGpuTensorAsFloat(squeezed, stream), {18.0f, 26.0f, 34.0f});
}

TEST(CubReduction, SqueezedScalarOutputUsesOneElementShape) {
    REQUIRE_CUDA_DEVICE();
    Stream stream(0);
    Tensor input = makeGpuTensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, stream);
    Tensor scalar_output(gpuPlacement, TensorDescriptor(DataType::FP32, {1}));

    std::shared_ptr<StampedCubReduction> stamped =
        CubReduction(CubReductionOp::Product, std::vector<uint32_t>{0, 1}, DataType::FP32)
            .stamp(input, scalar_output, stream);
    stamped->run();
    stream.synchronize();
    expectFloatVectorNear(copyGpuTensorAsFloat(scalar_output, stream), {24.0f});
}
