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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
