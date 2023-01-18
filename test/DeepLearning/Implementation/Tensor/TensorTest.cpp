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
        dimensions.push_back(1 + (rand() % 200));
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
        dimensions.push_back(1 + (rand() % 200));
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
        dimensions.push_back(1 + (rand() % 200));
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
        dimensions.push_back(1 + (rand() % 200));
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
        dimensions.push_back(1 + (rand() % 200));
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
        if (rand() % 2 == 0)
            dest_d.add(source_d, scalar, stream);
        else
            dest_d.add(scalar, source_d, stream);
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

TEST(Tensor, AddTensor) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
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

        Tensor augend_h(cpuPlacement, descriptor);
        Tensor augend_float_h = augend_h.clone(TensorDescriptor::DataType::FP32);
        Tensor augend_d = augend_h.clone(gpuPlacement);
        Tensor addend_h(cpuPlacement, descriptor);
        Tensor addend_float_h = augend_h.clone(TensorDescriptor::DataType::FP32);
        Tensor addend_d = augend_h.clone(gpuPlacement);
        Tensor dest_d = augend_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *augend_float_h_mem = (float *)augend_float_h.getMemPtr();
        float *addend_float_h_mem = (float *)addend_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
                    augend_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                    addend_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                } else {
                    augend_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                    addend_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                    if (dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                        dataType != TensorDescriptor::DataType::UINT32) {
                        if (rand() % 2 == 0)
                            augend_float_h_mem[i * dimensions[1] + j] *= -1;
                        if (rand() % 2 == 0)
                            addend_float_h_mem[i * dimensions[1] + j] *= -1;
                    }
                }
            }
        }

        augend_h.copyFromAsync(augend_float_h, stream);
        augend_d.copyFromAsync(augend_h, stream);
        addend_h.copyFromAsync(addend_float_h, stream);
        addend_d.copyFromAsync(addend_h, stream);
        dest_d.add(augend_d, addend_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.035;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = augend_float_h_mem[i * dimensions[1] + j] + addend_float_h_mem[i * dimensions[1] + j];
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //    augend_float_h_mem[i * dimensions[1] + j], addend_float_h_mem[i * dimensions[1] + j], (int)dataType);
                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
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
        dimensions.push_back(1 + (rand() % 200));
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
                if (dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32) {
                    if (rand() % 2 == 0)
                        source_float_h_mem[i * dimensions[1] + j] *= -1;
                }
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
        if (dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32) {
            if (rand() % 2 == 0)
                scalar *= -1;
        }

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        dest_d.subtract(source_d, scalar, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = source_float_h_mem[i * dimensions[1] + j] - scalar;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       source_float_h_mem[i * dimensions[1] + j], scalar, (int)dataType);
                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
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
        dimensions.push_back(1 + (rand() % 200));
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
                if (dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32) {
                    if (rand() % 2 == 0)
                        source_float_h_mem[i * dimensions[1] + j] *= -1;
                }
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
        if (dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32) {
            if (rand() % 2 == 0)
                scalar *= -1;
        }

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

TEST(Tensor, SubtractTensor) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
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

        Tensor minuend_h(cpuPlacement, descriptor);
        Tensor minuend_float_h = minuend_h.clone(TensorDescriptor::DataType::FP32);
        Tensor minuend_d = minuend_h.clone(gpuPlacement);
        Tensor subtrahend_h(cpuPlacement, descriptor);
        Tensor subtrahend_float_h = minuend_h.clone(TensorDescriptor::DataType::FP32);
        Tensor subtrahend_d = minuend_h.clone(gpuPlacement);
        Tensor dest_d = minuend_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *minuend_float_h_mem = (float *)minuend_float_h.getMemPtr();
        float *subtrahend_float_h_mem = (float *)subtrahend_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
                    minuend_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                    subtrahend_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                } else if (dataType == TensorDescriptor::DataType::UINT8 || dataType == TensorDescriptor::DataType::UINT16 ||
                           dataType == TensorDescriptor::DataType::UINT32) {
                    minuend_float_h_mem[i * dimensions[1] + j] = 9 + rand() % 10;
                    subtrahend_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                } else {
                    minuend_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                    subtrahend_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                }
                if (dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32) {
                    if (rand() % 2 == 0)
                        minuend_float_h_mem[i * dimensions[1] + j] *= -1;
                    if (rand() % 2 == 0)
                        subtrahend_float_h_mem[i * dimensions[1] + j] *= -1;
                }
            }
        }

        minuend_h.copyFromAsync(minuend_float_h, stream);
        minuend_d.copyFromAsync(minuend_h, stream);
        subtrahend_h.copyFromAsync(subtrahend_float_h, stream);
        subtrahend_d.copyFromAsync(subtrahend_h, stream);
        dest_d.subtract(minuend_d, subtrahend_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.035;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = minuend_float_h_mem[i * dimensions[1] + j] - subtrahend_float_h_mem[i * dimensions[1] + j];
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                // minuend_float_h_mem[i * dimensions[1] + j], subtrahend_float_h_mem[i * dimensions[1] + j], (int)dataType);
                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
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
        dimensions.push_back(1 + (rand() % 200));
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
        if (rand() % 2 == 0)
            dest_d.multiply(source_d, scalar, stream);
        else
            dest_d.multiply(scalar, source_d, stream);
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

TEST(Tensor, MultiplyTensor) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
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

        Tensor multiplicand_h(cpuPlacement, descriptor);
        Tensor multiplicand_float_h = multiplicand_h.clone(TensorDescriptor::DataType::FP32);
        Tensor multiplicand_d = multiplicand_h.clone(gpuPlacement);
        Tensor multiplier_h(cpuPlacement, descriptor);
        Tensor multiplier_float_h = multiplicand_h.clone(TensorDescriptor::DataType::FP32);
        Tensor multiplier_d = multiplicand_h.clone(gpuPlacement);
        Tensor dest_d = multiplicand_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *multiplicand_float_h_mem = (float *)multiplicand_float_h.getMemPtr();
        float *multiplier_float_h_mem = (float *)multiplier_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
                    multiplicand_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                    multiplier_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                } else {
                    multiplicand_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                    multiplier_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                }
            }
        }

        multiplicand_h.copyFromAsync(multiplicand_float_h, stream);
        multiplicand_d.copyFromAsync(multiplicand_h, stream);
        multiplier_h.copyFromAsync(multiplier_float_h, stream);
        multiplier_d.copyFromAsync(multiplier_h, stream);
        dest_d.multiply(multiplicand_d, multiplier_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.035;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = multiplicand_float_h_mem[i * dimensions[1] + j] * multiplier_float_h_mem[i * dimensions[1] + j];
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                // multiplicand_float_h_mem[i * dimensions[1] + j], multiplier_float_h_mem[i * dimensions[1] + j], (int)dataType);
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
        dimensions.push_back(1 + (rand() % 200));
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
        float thresh = 0.02;
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
        dimensions.push_back(1 + (rand() % 200));
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

TEST(Tensor, DivideTensor) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
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

        Tensor numerator_h(cpuPlacement, descriptor);
        Tensor numerator_float_h = numerator_h.clone(TensorDescriptor::DataType::FP32);
        Tensor numerator_d = numerator_h.clone(gpuPlacement);
        Tensor denominator_h(cpuPlacement, descriptor);
        Tensor denominator_float_h = numerator_h.clone(TensorDescriptor::DataType::FP32);
        Tensor denominator_d = numerator_h.clone(gpuPlacement);
        Tensor dest_d = numerator_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *numerator_float_h_mem = (float *)numerator_float_h.getMemPtr();
        float *denominator_float_h_mem = (float *)denominator_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
                    numerator_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                    denominator_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                } else {
                    numerator_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                    denominator_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                }
            }
        }

        numerator_h.copyFromAsync(numerator_float_h, stream);
        numerator_d.copyFromAsync(numerator_h, stream);
        denominator_h.copyFromAsync(denominator_float_h, stream);
        denominator_d.copyFromAsync(denominator_h, stream);
        dest_d.divide(numerator_d, denominator_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.035;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected;
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    expected = numerator_float_h_mem[i * dimensions[1] + j] / denominator_float_h_mem[i * dimensions[1] + j];
                else
                    expected =
                        (int32_t)numerator_float_h_mem[i * dimensions[1] + j] / (int32)denominator_float_h_mem[i * dimensions[1] + j];
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                // numerator_float_h_mem[i * dimensions[1] + j], denominator_float_h_mem[i * dimensions[1] + j], (int)dataType);
                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
