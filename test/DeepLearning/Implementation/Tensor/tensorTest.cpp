#include "Thor.h"

#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

#pragma GCC diagnostic ignored "-Wsign-compare"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop

using namespace ThorImplementation;

TEST(Tensor, Copies) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

        vector<Tensor> tensors;
        tensors.emplace_back(cpuPlacement, descriptor);
        for (uint32_t i = 0; i < 10; ++i) {
            TensorPlacement placement;
            int num = rand() % 3;
            if (num == 0)
                placement = cpuPlacement;
            else if (num == 1)
                placement = gpu0Placement;
            else
                placement = gpu1Placement;
        }
        tensors.emplace_back(cpuPlacement, descriptor);

        float *inputTensorMem = (float *)tensors[0].getMemPtr();
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            inputTensorMem[i] = ((rand() % 1000) / 10.0f) - 50.0f;
        }

        for (uint32_t i = 1; i < tensors.size(); ++i) {
            tensors[i].copyFromAsync(tensors[i - 1], stream);
        }
        stream.synchronize();

        float *outputTensorMem = (float *)tensors.back().getMemPtr();
        ASSERT_NE(inputTensorMem, outputTensorMem);
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            ASSERT_EQ(inputTensorMem[i], outputTensorMem[i]);
        }
    }
}

// Reshape keeps contents unchanged
TEST(Tensor, Reshapes) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 5; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

        TensorPlacement placement;
        placement = cpuPlacement;

        Tensor tensor(placement, descriptor);
        ASSERT_EQ(tensor.getDescriptor().getDimensions(), dimensions);

        // Write data to the tensor
        float *mem = (float *)tensor.getMemPtr();
        float *expected = new float[dimensions[0] * dimensions[1]];
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            expected[i] = (rand() % 1000) / 10;
            mem[i] = expected[i];
        }

        uint32_t d0;
        uint32_t d1;
        do {
            d0 = (rand() % (dimensions[0] * dimensions[1])) + 1;
            d1 = (dimensions[0] * dimensions[1]) / d0;
        } while (d0 * d1 != dimensions[0] * dimensions[1]);
        dimensions.clear();
        dimensions.push_back(d0);
        dimensions.push_back(d1);
        tensor.reshape(dimensions);
        ASSERT_EQ(tensor.getDescriptor().getDimensions(), dimensions);
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            ASSERT_EQ(mem[i], expected[i]);
        }

        delete[] expected;
    }
}

// Resize destroys contents.
TEST(Tensor, Resizes) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 5; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

        TensorPlacement placement;
        int num = rand() % 2;
        if (num == 0)
            placement = cpuPlacement;
        else
            placement = gpuPlacement;

        Tensor tensor(placement, descriptor);
        ASSERT_EQ(tensor.getDescriptor().getDimensions(), dimensions);

        dimensions.clear();
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        tensor.resize(dimensions);
        ASSERT_EQ(tensor.getDescriptor().getDimensions(), dimensions);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
