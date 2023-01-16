#include "Thor.h"

#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

#pragma GCC diagnostic ignored "-Wsign-compare"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop

using namespace ThorImplementation;
using namespace std;

TEST(Tensor, Copies) {
    srand(time(nullptr));

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
    srand(time(nullptr));

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
    srand(time(nullptr));

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

TEST(Tensor, AddScalar) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            dataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            dataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            dataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            dataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            dataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            dataType = TensorDescriptor::DataType::INT16;
        else
            dataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(TensorDescriptor::DataType::FP32);
        Tensor source_d = source_h.clone(gpuPlacement);
        Tensor dest_d = source_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 20) / ((rand() % 10) + 1.0);
                else
                    source_float_h_mem[i * dimensions[1] + j] = rand() % 50;
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 50;

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        dest_d.add(source_d, scalar, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                ASSERT_LT(abs((source_float_h_mem[i * dimensions[1] + j] + scalar) - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, SubtractScalarSubtrahend) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            dataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            dataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            dataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            dataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            dataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            dataType = TensorDescriptor::DataType::INT16;
        else
            dataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(TensorDescriptor::DataType::FP32);
        Tensor source_d = source_h.clone(gpuPlacement);
        Tensor dest_d = source_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 20) / ((rand() % 10) + 1.0);
                else if (dataType == TensorDescriptor::DataType::UINT8 || dataType == TensorDescriptor::DataType::UINT16 ||
                         dataType == TensorDescriptor::DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = 19 + rand() % 20;
                else
                    source_float_h_mem[i * dimensions[1] + j] = rand() % 50;
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else if (dataType == TensorDescriptor::DataType::UINT8 || dataType == TensorDescriptor::DataType::UINT16 ||
                 dataType == TensorDescriptor::DataType::UINT32)
            scalar = rand() % 20;
        else
            scalar = rand() % 50;

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        dest_d.subtract(source_d, scalar, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                // printf("%f %f [%ld]\n", (source_float_h_mem[i * dimensions[1] + j] - scalar), dest_gpu_float_h_mem[i * dimensions[1] +
                // j], i * dimensions[1] + j);
                ASSERT_LT(abs((source_float_h_mem[i * dimensions[1] + j] - scalar) - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, SubtractScalarMinuend) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            dataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            dataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            dataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            dataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            dataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            dataType = TensorDescriptor::DataType::INT16;
        else
            dataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(TensorDescriptor::DataType::FP32);
        Tensor source_d = source_h.clone(gpuPlacement);
        Tensor dest_d = source_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 20) / ((rand() % 10) + 1.0);
                else if (dataType == TensorDescriptor::DataType::UINT8 || dataType == TensorDescriptor::DataType::UINT16 ||
                         dataType == TensorDescriptor::DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = rand() % 20;
                else
                    source_float_h_mem[i * dimensions[1] + j] = rand() % 50;
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else if (dataType == TensorDescriptor::DataType::UINT8 || dataType == TensorDescriptor::DataType::UINT16 ||
                 dataType == TensorDescriptor::DataType::UINT32)
            scalar = 19 + rand() % 20;
        else
            scalar = rand() % 50;

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        dest_d.subtract(scalar, source_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                ASSERT_LT(abs((scalar - source_float_h_mem[i * dimensions[1] + j]) - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, MultiplyScalar) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            dataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            dataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            dataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            dataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            dataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            dataType = TensorDescriptor::DataType::INT16;
        else
            dataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(TensorDescriptor::DataType::FP32);
        Tensor source_d = source_h.clone(gpuPlacement);
        Tensor dest_d = source_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                else
                    source_float_h_mem[i * dimensions[1] + j] = rand() % 10;
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10;

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        dest_d.multiply(source_d, scalar, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.035;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = source_float_h_mem[i * dimensions[1] + j] * scalar;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                // source_float_h_mem[i * dimensions[1] + j], scalar, (int)dataType);
                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, DivideScalarDenominator) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            dataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            dataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            dataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            dataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            dataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            dataType = TensorDescriptor::DataType::INT16;
        else
            dataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(TensorDescriptor::DataType::FP32);
        Tensor source_d = source_h.clone(gpuPlacement);
        Tensor dest_d = source_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                else
                    source_float_h_mem[i * dimensions[1] + j] = rand() % 10;
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = ((rand() % 20) + 1) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10 + 1;

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        dest_d.divide(source_d, scalar, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected;
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    expected = source_float_h_mem[i * dimensions[1] + j] / scalar;
                else
                    expected = (int32_t)source_float_h_mem[i * dimensions[1] + j] / (int32_t)scalar;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                // source_float_h_mem[i * dimensions[1] + j], scalar, (int)dataType);
                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, DivideScalarNumerator) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 100));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            dataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            dataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            dataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            dataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            dataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            dataType = TensorDescriptor::DataType::INT16;
        else
            dataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(TensorDescriptor::DataType::FP32);
        Tensor source_d = source_h.clone(gpuPlacement);
        Tensor dest_d = source_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    source_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                else
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10;

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        dest_d.divide(scalar, source_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.035;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected;
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    expected = scalar / source_float_h_mem[i * dimensions[1] + j];
                else
                    expected = (int32_t)scalar / (int32_t)source_float_h_mem[i * dimensions[1] + j];
                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
