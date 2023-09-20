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
        else
            scalar = rand() % 50;
        if (dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32) {
            if (rand() % 2 == 0)
                scalar *= -1;
        }

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

TEST(Tensor, AddScalarWithAlpha) {
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

        float alpha = 200.0f / ((rand() % 100) + 100);
        if (rand() % 2 && dt != 2 && dt != 3 && dt != 4)
            alpha = -alpha;

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 20) / ((rand() % 10) + 1.0);
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
        else
            scalar = rand() % 50;
        if (dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32) {
            if (rand() % 2 == 0)
                scalar *= -1;
        }

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        if (rand() % 2 == 0)
            dest_d.add(source_d, scalar, alpha, stream);
        else
            dest_d.add(scalar, source_d, alpha, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01;
        if (dt == 0)
            thresh = 0.05;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float rawAddend = alpha * source_float_h_mem[i * dimensions[1] + j];
                float addend;
                if (dt == 0)
                    addend = half(rawAddend);
                else if (dt == 1)
                    addend = float(rawAddend);
                else if (dt == 2)
                    addend = uint8_t(rawAddend);
                else if (dt == 3)
                    addend = uint16_t(rawAddend);
                else if (dt == 4)
                    addend = uint32_t(rawAddend);
                else if (dt == 5)
                    addend = int8_t(rawAddend);
                else if (dt == 6)
                    addend = int16_t(rawAddend);
                else
                    addend = int32_t(rawAddend);
                if (dt >= 2 && abs(rawAddend - addend) < 0.07f)
                    thresh = 1.0f;
                float expected = addend + scalar;
                if (dt == 2) {
                    if (expected > 255)
                        expected = 255;
                } else if (dt == 5) {
                    if (expected > 127)
                        expected = 127;
                    else if (expected < -128)
                        expected = -128;
                }

                if (abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]) >= thresh)
                    printf("%f * %f + %f = (%f) %f | %f     dt %d\n",
                           alpha,
                           source_float_h_mem[i * dimensions[1] + j],
                           scalar,
                           rawAddend + scalar,
                           expected,
                           dest_gpu_float_h_mem[i * dimensions[1] + j],
                           dt);

                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, AddTensorWithAlphaBeta) {
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

        float alpha = 200.0f / ((rand() % 100) + 100);
        if (rand() % 2 && dt != 2 && dt != 3 && dt != 4)
            alpha = -alpha;
        float beta = 200.0f / ((rand() % 100) + 100);
        if (rand() % 2 && dt != 2 && dt != 3 && dt != 4)
            beta = -beta;

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
        dest_d.add(augend_d, addend_d, alpha, beta, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.035;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float rawScaledAugend = augend_float_h_mem[i * dimensions[1] + j] * alpha;
                float rawScaledAddend = addend_float_h_mem[i * dimensions[1] + j] * beta;
                float scaledAugend, scaledAddend;
                if (dt == 0) {
                    scaledAugend = half(rawScaledAugend);
                    scaledAddend = half(rawScaledAddend);
                } else if (dt == 1) {
                    scaledAugend = float(rawScaledAugend);
                    scaledAddend = float(rawScaledAddend);
                } else if (dt == 2) {
                    scaledAugend = uint8_t(rawScaledAugend);
                    scaledAddend = uint8_t(rawScaledAddend);
                } else if (dt == 3) {
                    scaledAugend = uint16_t(rawScaledAugend);
                    scaledAddend = uint16_t(rawScaledAddend);
                } else if (dt == 4) {
                    scaledAugend = uint32_t(rawScaledAugend);
                    scaledAddend = uint32_t(rawScaledAddend);
                } else if (dt == 5) {
                    scaledAugend = int8_t(rawScaledAugend);
                    scaledAddend = int8_t(rawScaledAddend);
                } else if (dt == 6) {
                    scaledAugend = int16_t(rawScaledAugend);
                    scaledAddend = int16_t(rawScaledAddend);
                } else if (dt == 7) {
                    scaledAugend = int32_t(rawScaledAugend);
                    scaledAddend = int32_t(rawScaledAddend);
                }
                if (dt >= 2 && abs(rawScaledAugend - scaledAugend) < 0.07f)
                    thresh += 1.0f;
                if (dt >= 2 && abs(rawScaledAddend - scaledAddend) < 0.07f)
                    thresh += 1.0f;
                float expected = scaledAugend + scaledAddend;
                if (dt == 2) {
                    if (expected > 255)
                        expected = 255;
                } else if (dt == 5) {
                    if (expected > 127)
                        expected = 127;
                    else if (expected < -128)
                        expected = -128;
                }
                float actual = dest_gpu_float_h_mem[i * dimensions[1] + j];

                if (abs(expected - actual) >= thresh)
                    printf("%f (%f) + %f (%f) = %f (%f) : %f      | rawAugend %f rawAddend %f alpha %f   beta %f  [%d,%d] dt %d\n",
                           rawScaledAugend,
                           scaledAugend,
                           rawScaledAddend,
                           scaledAddend,
                           rawScaledAugend + rawScaledAddend,
                           expected,
                           actual,
                           augend_float_h_mem[i * dimensions[1] + j],
                           addend_float_h_mem[i * dimensions[1] + j],
                           alpha,
                           beta,
                           i,
                           j,
                           dt);

                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //    augend_float_h_mem[i * dimensions[1] + j], addend_float_h_mem[i * dimensions[1] + j], (int)dataType);
                ASSERT_LT(abs(expected - actual), thresh);
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
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = -source_float_h_mem[i * dimensions[1] + j];
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10;
        if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32)
            scalar = -scalar;

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

TEST(Tensor, MultiplyElementwise) {
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
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    multiplicand_float_h_mem[i * dimensions[1] + j] = -multiplicand_float_h_mem[i * dimensions[1] + j];
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    multiplier_float_h_mem[i * dimensions[1] + j] = -multiplier_float_h_mem[i * dimensions[1] + j];
            }
        }

        multiplicand_h.copyFromAsync(multiplicand_float_h, stream);
        multiplicand_d.copyFromAsync(multiplicand_h, stream);
        multiplier_h.copyFromAsync(multiplier_float_h, stream);
        multiplier_d.copyFromAsync(multiplier_h, stream);
        dest_d.multiplyElementwise(multiplicand_d, multiplier_d, stream);
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

TEST(Tensor, MultiplyScalarTensor) {
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
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = -source_float_h_mem[i * dimensions[1] + j];
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10;
        if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32)
            scalar = -scalar;
        Tensor scalar_d(gpuPlacement, TensorDescriptor(dataType, {1UL}));
        scalar_d.fill(scalar, stream);

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        if (rand() % 2 == 0)
            dest_d.multiplyTensorScalar(source_d, scalar_d, stream);
        else
            dest_d.multiplyScalarTensor(scalar_d, source_d, stream);
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

TEST(Tensor, MultiplyMatrix) {
    // This one needs to handle all 3 interpretations of matrix multiply
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    // FIXME implement
    for (uint32_t t = 0; t < 20; ++t) {
        uint32_t testType = rand() % 3;
        if (testType == 0) {
            // Scalar Tensor
        } else if (testType == 1) {
            // Vector Vector
        } else {
            // Matrix Matrix
        }
    }
}

TEST(Tensor, dotProduct) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        unsigned long numElements = 1 + (rand() % 200);
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else
            dataType = TensorDescriptor::DataType::FP32;

        Tensor A(gpuPlacement, TensorDescriptor(dataType, {1, numElements}));
        Tensor B(gpuPlacement, TensorDescriptor(dataType, {numElements, 1}));
        Tensor C(gpuPlacement, TensorDescriptor(dataType, {1}));
        Tensor A_h = A.clone(cpuPlacement);
        Tensor B_h = B.clone(cpuPlacement);
        Tensor CGpu_h = C.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        A_h.fillRandom(10, -10, stream);
        B_h.fillRandom(10, -10, stream);
        A.copyFromAsync(A_h, stream);
        B.copyFromAsync(B_h, stream);

        C.dotProduct(A, B, stream);
        CGpu_h.copyFromAsync(C, stream);

        Tensor AFp32_h = A_h.clone(TensorDescriptor::DataType::FP32);
        Tensor BFp32_h = A_h.clone(TensorDescriptor::DataType::FP32);
        AFp32_h.copyFromAsync(A_h, stream);
        BFp32_h.copyFromAsync(B_h, stream);

        stream.synchronize();

        float expected = 0.0f;
        float *AMem_h = AFp32_h.getMemPtr<float>();
        float *BMem_h = BFp32_h.getMemPtr<float>();
        for (uint32_t i = 0; i < numElements; ++i) {
            expected += AMem_h[i] * BMem_h[i];
        }

        float *CGpuMem_h = CGpu_h.getMemPtr<float>();
        float actual = CGpuMem_h[0];
        float thresh = max(0.05, abs(0.001 * expected));
        if (abs(expected - actual) >= thresh) {
            printf("actual %f expected %f\n", actual, expected);
        }
        ASSERT_LT(abs(expected - actual), thresh);
    }
}

TEST(Tensor, outerProduct) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        unsigned long numElements = 1 + (rand() % 200);
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else
            dataType = TensorDescriptor::DataType::FP32;

        Tensor A(gpuPlacement, TensorDescriptor(dataType, {1, numElements}));
        Tensor B(gpuPlacement, TensorDescriptor(dataType, {numElements, 1}));
        Tensor C(gpuPlacement, TensorDescriptor(dataType, {numElements, numElements}));
        Tensor A_h = A.clone(cpuPlacement);
        Tensor B_h = B.clone(cpuPlacement);
        Tensor CGpu_h = C.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        A_h.fillRandom(10, -10, stream);
        B_h.fillRandom(10, -10, stream);
        A.copyFromAsync(A_h, stream);
        B.copyFromAsync(B_h, stream);

        C.outerProduct(A, B, stream);
        CGpu_h.copyFromAsync(C, stream);

        Tensor AFp32_h = A_h.clone(TensorDescriptor::DataType::FP32);
        Tensor BFp32_h = A_h.clone(TensorDescriptor::DataType::FP32);
        AFp32_h.copyFromAsync(A_h, stream);
        BFp32_h.copyFromAsync(B_h, stream);

        stream.synchronize();

        float *AMem_h = AFp32_h.getMemPtr<float>();
        float *BMem_h = BFp32_h.getMemPtr<float>();
        float *CMemGpu_h = CGpu_h.getMemPtr<float>();
        for (uint32_t row = 0; row < numElements; ++row) {
            for (uint32_t col = 0; col < numElements; ++col) {
                float expected = AMem_h[row] * BMem_h[col];
                float actual = CMemGpu_h[row * numElements + col];
                float thresh = max(0.05, abs(0.001 * expected));
                if (abs(expected - actual) >= thresh) {
                    printf("[%d,%d] actual %f expected %f\n", row, col, actual, expected);
                }
                ASSERT_LT(abs(expected - actual), thresh);
            }
        }
    }
}

TEST(Tensor, Gemm) {
    // FIXME implement
}

TEST(Tensor, identityMatrixCpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else
            dataType = TensorDescriptor::DataType::FP32;

        uint32_t N = 1 + (rand() % 300);

        Tensor I = Tensor::identityMatrix(N, TensorPlacement::MemDevices::CPU, dataType, stream);
        stream.synchronize();

        if (dt == 0) {
            half *mem = I.getMemPtr<half>();
            for (uint32_t row = 0; row < N; ++row) {
                for (uint32_t col = 0; col < N; ++col) {
                    if (row == col)
                        ASSERT_EQ(mem[row * N + col], half(1.0f));
                    else
                        ASSERT_EQ(mem[row * N + col], half(0.0f));
                }
            }
        } else {
            float *mem = I.getMemPtr<float>();
            for (uint32_t row = 0; row < N; ++row) {
                for (uint32_t col = 0; col < N; ++col) {
                    if (row == col)
                        ASSERT_EQ(mem[row * N + col], 1.0f);
                    else
                        ASSERT_EQ(mem[row * N + col], 0.0f);
                }
            }
        }
    }

    // Ensure that the async host function works properly when the only reference to the tensor is immediately dropped
    Tensor::identityMatrix(300, TensorPlacement::MemDevices::CPU, TensorDescriptor::DataType::FP32, stream);
}

TEST(Tensor, identityMatrixGpu) {}
TEST(Tensor, zerosCpu) {}
TEST(Tensor, zerosGpu) {}
TEST(Tensor, onesCpu) {}
TEST(Tensor, onesGpu) {}
TEST(Tensor, randomsCpu) {}
TEST(Tensor, randomsGpu) {}
TEST(Tensor, valuesCpu) {}
TEST(Tensor, valuesGpu) {}


TEST(Tensor, fillRandom) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            dataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            dataType = TensorDescriptor::DataType::INT8;
        else if (dt == 3)
            dataType = TensorDescriptor::DataType::INT16;
        else if (dt == 4)
            dataType = TensorDescriptor::DataType::INT32;
        else if (dt == 5)
            dataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 6)
            dataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 7)
            dataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 8)
            dataType = TensorDescriptor::DataType::BOOLEAN;
        else if (dt == 9)
            dataType = TensorDescriptor::DataType::PACKED_BOOLEAN;

        uint32_t numDimensions = 1 + (rand() % 5);
        uint32_t maxDimension = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimension = pow(10000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t i = 0; i < numDimensions; ++i) {
            dimensions.push_back(1 + (rand() % maxDimension));
        }
        Tensor tensor(cpuPlacement, TensorDescriptor(dataType, dimensions));
        int16_t minValue;
        int16_t maxValue;
        if (dt < 5) {
            minValue = -100;
            maxValue = 100;
        } else if (dt < 8) {
            minValue = 10;
            maxValue = 200;
        } else if (dt == 8) {
            minValue = false;
            maxValue = true;
        } else {
            minValue = 0;
            maxValue = 255;
        }
        tensor.fillRandom(minValue, maxValue, stream);
        Tensor tensorFp32 = tensor.clone(TensorDescriptor::DataType::FP32);
        tensorFp32.copyFromAsync(tensor, stream);
        stream.synchronize();

        float *mem = tensorFp32.getMemPtr<float>();
        for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
            ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
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
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = -source_float_h_mem[i * dimensions[1] + j];
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = ((rand() % 20) + 1) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10 + 1;
        if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32)
            scalar = -scalar;

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
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = -source_float_h_mem[i * dimensions[1] + j];
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10;
        if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32)
            scalar = -scalar;

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
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    numerator_float_h_mem[i * dimensions[1] + j] = -numerator_float_h_mem[i * dimensions[1] + j];
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    denominator_float_h_mem[i * dimensions[1] + j] = -denominator_float_h_mem[i * dimensions[1] + j];
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
                        (int32_t)numerator_float_h_mem[i * dimensions[1] + j] / (int32_t)denominator_float_h_mem[i * dimensions[1] + j];
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //    numerator_float_h_mem[i * dimensions[1] + j], denominator_float_h_mem[i * dimensions[1] + j], (int)dataType);
                ASSERT_LT(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, PowTensorTensor) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(TensorDescriptor::DataType::FP32, dimensions);

        Tensor base_h(cpuPlacement, termDescriptor);
        Tensor base_float_h = base_h.clone(TensorDescriptor::DataType::FP32);
        Tensor base_d = base_h.clone(gpuPlacement);
        Tensor exponent_h(cpuPlacement, termDescriptor);
        Tensor exponent_float_h = base_h.clone(TensorDescriptor::DataType::FP32);
        Tensor exponent_d = base_h.clone(gpuPlacement);
        Tensor dest_d = base_d.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *base_float_h_mem = (float *)base_float_h.getMemPtr();
        float *exponent_float_h_mem = (float *)exponent_float_h.getMemPtr();
        // In this test I cover 0 base but do not cover negative exponent
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8) {
                    base_float_h_mem[i * dimensions[1] + j] = (rand() % 6) / ((rand() % 6) + 1.0);
                    exponent_float_h_mem[i * dimensions[1] + j] = (rand() % 4) / ((rand() % 4) + 1.0);
                    if (base_float_h_mem[i * dimensions[1] + j] < 0.5)
                        base_float_h_mem[i * dimensions[1] + j] = 0.5;
                } else {
                    base_float_h_mem[i * dimensions[1] + j] = (rand() % 6) / ((rand() % 6) + 1.0);
                    exponent_float_h_mem[i * dimensions[1] + j] = (rand() % 6) / ((rand() % 6) + 1.0);
                }
            }
        }

        base_h.copyFromAsync(base_float_h, stream);
        base_d.copyFromAsync(base_h, stream);
        exponent_h.copyFromAsync(exponent_float_h, stream);
        exponent_d.copyFromAsync(exponent_h, stream);
        dest_d.pow(base_d, exponent_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh;
        if (destDataType == TensorDescriptor::DataType::FP16 || destDataType == TensorDescriptor::DataType::FP32)
            thresh = 0.035f;
        else
            thresh = 1.0f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = (float)powf(base_float_h_mem[i * dimensions[1] + j], exponent_float_h_mem[i * dimensions[1] + j]);
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //     base_float_h_mem[i * dimensions[1] + j], exponent_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                if (expected >= 1000)
                    thresh = 1.0f;
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, PowScalarExponent) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(TensorDescriptor::DataType::FP32, dimensions);

        Tensor base_h(cpuPlacement, termDescriptor);
        Tensor base_float_h = base_h.clone(TensorDescriptor::DataType::FP32);
        Tensor base_d = base_h.clone(gpuPlacement);
        Tensor dest_d = base_d.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *base_float_h_mem = (float *)base_float_h.getMemPtr();
        float exponent;
        // In this test I cover negative exponent, but do not cover 0 base.
        if (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8) {
            exponent = (rand() % 4) / ((rand() % 4) + 1.0);
        } else {
            exponent = (rand() % 6) / ((rand() % 6) + 1.0);
            if (rand() % 2)
                exponent = -exponent;
        }
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                base_float_h_mem[i * dimensions[1] + j] = ((rand() % 5) + 1) / ((rand() % 6) + 1.0);
                if (base_float_h_mem[i * dimensions[1] + j] < 0.5f &&
                    (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8)) {
                    base_float_h_mem[i * dimensions[1] + j] = 0.5f;
                }
            }
        }

        base_h.copyFromAsync(base_float_h, stream);
        base_d.copyFromAsync(base_h, stream);
        dest_d.pow(base_d, exponent, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh;
        if (destDataType == TensorDescriptor::DataType::FP16 || destDataType == TensorDescriptor::DataType::FP32)
            thresh = 0.035f;
        else
            thresh = 1.0f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = (float)powf(base_float_h_mem[i * dimensions[1] + j], exponent);
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //     base_float_h_mem[i * dimensions[1] + j], exponent, (int)destDataType);
                if (expected >= 1000)
                    thresh = 1.0f;
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, PowScalarBase) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(TensorDescriptor::DataType::FP32, dimensions);

        Tensor exponent_h(cpuPlacement, termDescriptor);
        Tensor exponent_float_h = exponent_h.clone(TensorDescriptor::DataType::FP32);
        Tensor exponent_d = exponent_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float base = (rand() % 6) / ((rand() % 6) + 1.0);
        if (base < 0.5f && (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8)) {
            base = 0.5f;
        }
        // Not doing negative base because of complex numbers

        float *exponent_float_h_mem = (float *)exponent_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8) {
                    exponent_float_h_mem[i * dimensions[1] + j] = (rand() % 4) / ((rand() % 4) + 1.0);
                } else {
                    exponent_float_h_mem[i * dimensions[1] + j] = (rand() % 6) / ((rand() % 6) + 1.0);
                }
                if (base != 0.0f && rand() % 2)
                    exponent_float_h_mem[i * dimensions[1] + j] = -exponent_float_h_mem[i * dimensions[1] + j];
            }
        }

        exponent_h.copyFromAsync(exponent_float_h, stream);
        exponent_d.copyFromAsync(exponent_h, stream);
        dest_d.pow(base, exponent_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh;
        if (destDataType == TensorDescriptor::DataType::FP16 || destDataType == TensorDescriptor::DataType::FP32)
            thresh = 0.035f;
        else
            thresh = 1.0f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = (float)powf(base, exponent_float_h_mem[i * dimensions[1] + j]);
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                float unconvertedExpected = expected;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                if (expected >= 1000)
                    thresh = 1.0f;
                if (expected - dest_gpu_float_h_mem[i * dimensions[1] + j] > thresh) {
                    printf("uncoverted expected %f\n", unconvertedExpected);
                    printf("%f %f [%ld]  %f %f %d\n",
                           expected,
                           dest_gpu_float_h_mem[i * dimensions[1] + j],
                           i * dimensions[1] + j,
                           base,
                           exponent_float_h_mem[i * dimensions[1] + j],
                           (int)destDataType);
                }
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Exp) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(TensorDescriptor::DataType::FP32, dimensions);

        Tensor exponent_h(cpuPlacement, termDescriptor);
        Tensor exponent_float_h = exponent_h.clone(TensorDescriptor::DataType::FP32);
        Tensor exponent_d = exponent_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *exponent_float_h_mem = (float *)exponent_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8) {
                    exponent_float_h_mem[i * dimensions[1] + j] = (rand() % 4) / ((rand() % 4) + 1.0);
                } else {
                    exponent_float_h_mem[i * dimensions[1] + j] = (rand() % 6) / ((rand() % 6) + 1.0);
                }
                if (rand() % 2)
                    exponent_float_h_mem[i * dimensions[1] + j] = -exponent_float_h_mem[i * dimensions[1] + j];
            }
        }

        exponent_h.copyFromAsync(exponent_float_h, stream);
        exponent_d.copyFromAsync(exponent_h, stream);
        dest_d.exp(exponent_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh;
        if (destDataType == TensorDescriptor::DataType::FP16 || destDataType == TensorDescriptor::DataType::FP32)
            thresh = 0.05f;
        else
            thresh = 1.0f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = (float)expf(exponent_float_h_mem[i * dimensions[1] + j]);
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //     (float)M_E, exponent_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Log) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (argument_float_h_mem[i * dimensions[1] + j] < 0.2f &&
                    (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8))
                    argument_float_h_mem[i * dimensions[1] + j] = 0.2f;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.log(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh;
        if (destDataType == TensorDescriptor::DataType::FP16 || destDataType == TensorDescriptor::DataType::FP32)
            thresh = 0.15f;
        else
            thresh = 1.0f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = (float)logf(argument_float_h_mem[i * dimensions[1] + j]);
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (abs(expected) >= 200.0f && thresh < 0.5f)
                    thresh = 0.5f;
                if (abs(expected) >= 750.0f && thresh < 2.5f)
                    thresh = 2.5f;
                if (abs(expected) >= 2000.0f && thresh < 5.0f)
                    thresh = 5.0f;
                if (destDataType == TensorDescriptor::DataType::FP16 && abs(expected) >= 50.0f && thresh < 1.0f)
                    thresh = 1.0f;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //     (float)M_E, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, LogBaseX) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (argument_float_h_mem[i * dimensions[1] + j] < 0.2f &&
                    (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8))
                    argument_float_h_mem[i * dimensions[1] + j] = 0.2f;
            }
        }
        float base;
        int r = rand() % 10;
        if (r == 0)
            base = 2.0f;
        else if (r == 1)
            base = 10.0f;
        else
            base = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0f);
        if (base == 1.0f)
            base = 1.5f;
        if (base < 2.0f && (destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::INT8))
            base = 2.0f;
        float conversionFactor = 1.0f / log10f(base);

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.log(argument_d, base, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh;
        if (destDataType == TensorDescriptor::DataType::FP16 || destDataType == TensorDescriptor::DataType::FP32)
            thresh = 0.25f;
        else
            thresh = 1.0f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = (float)log10f(argument_float_h_mem[i * dimensions[1] + j]) * conversionFactor;
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (abs(expected) >= 200.0f && thresh < 0.5f)
                    thresh = 0.7f;
                if (abs(expected) >= 750.0f && thresh < 2.5f)
                    thresh = 2.5f;
                if (abs(expected) >= 2000.0f && thresh < 5.0f)
                    thresh = 5.0f;
                if (destDataType == TensorDescriptor::DataType::FP16 && abs(expected) >= 50.0f && thresh < 1.0f)
                    thresh = 1.0f;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                if (abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]) > thresh) {
                    printf("base %f, log10ConversionFactor %f\n", base, conversionFactor);
                    printf("%f %f [%ld]  %f %d\n",
                           expected,
                           dest_gpu_float_h_mem[i * dimensions[1] + j],
                           i * dimensions[1] + j,
                           argument_float_h_mem[i * dimensions[1] + j],
                           (int)destDataType);
                }
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, MultiplyAccumulate) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor abcDescriptor(sourceDataType, dimensions);

        Tensor a_h(cpuPlacement, abcDescriptor);
        Tensor a_float_h = a_h.clone(TensorDescriptor::DataType::FP32);
        Tensor a_d = a_h.clone(gpuPlacement);
        Tensor b_h = a_h.clone();
        Tensor b_float_h = b_h.clone(TensorDescriptor::DataType::FP32);
        Tensor b_d = b_h.clone(gpuPlacement);
        Tensor c_h = a_h.clone();
        Tensor c_float_h = c_h.clone(TensorDescriptor::DataType::FP32);
        Tensor c_d = c_h.clone(gpuPlacement);
        Tensor dest_d = a_d.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *a_float_h_mem = (float *)a_float_h.getMemPtr();
        float *b_float_h_mem = (float *)b_float_h.getMemPtr();
        float *c_float_h_mem = (float *)c_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                a_float_h_mem[i * dimensions[1] + j] = (rand() % 5) / ((rand() % 5) + 1.0);
                if (rand() % 2)
                    a_float_h_mem[i * dimensions[1] + j] = -a_float_h_mem[i * dimensions[1] + j];
                b_float_h_mem[i * dimensions[1] + j] = (rand() % 5) / ((rand() % 5) + 1.0);
                if (rand() % 2)
                    b_float_h_mem[i * dimensions[1] + j] = -b_float_h_mem[i * dimensions[1] + j];
                c_float_h_mem[i * dimensions[1] + j] = (rand() % 5) / ((rand() % 5) + 1.0);
                if (rand() % 2)
                    c_float_h_mem[i * dimensions[1] + j] = -c_float_h_mem[i * dimensions[1] + j];
            }
        }

        a_h.copyFromAsync(a_float_h, stream);
        a_d.copyFromAsync(a_h, stream);
        b_h.copyFromAsync(b_float_h, stream);
        b_d.copyFromAsync(b_h, stream);
        c_h.copyFromAsync(c_float_h, stream);
        c_d.copyFromAsync(c_h, stream);
        dest_d.multiplyAccumulateElementwise(a_d, b_d, c_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh;
        if (destDataType == TensorDescriptor::DataType::FP16 || destDataType == TensorDescriptor::DataType::FP32)
            thresh = 0.035f;
        else
            thresh = 1.0f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected =
                    a_float_h_mem[i * dimensions[1] + j] * b_float_h_mem[i * dimensions[1] + j] + c_float_h_mem[i * dimensions[1] + j];
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld] %f %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      a_float_h_mem[i * dimensions[1] + j], b_float_h_mem[i * dimensions[1] + j], c_float_h_mem[i * dimensions[1] + j],
                //      (int)destDataType);
                if (expected >= 1000)
                    thresh = 1.0f;
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Ceil) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2 && destDataType != TensorDescriptor::DataType::UINT8 && destDataType != TensorDescriptor::DataType::UINT16 &&
                    destDataType != TensorDescriptor::DataType::UINT32) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.ceil(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.0001f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = ceil(argument_float_h_mem[i * dimensions[1] + j]);
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Floor) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2 && destDataType != TensorDescriptor::DataType::UINT8 && destDataType != TensorDescriptor::DataType::UINT16 &&
                    destDataType != TensorDescriptor::DataType::UINT32) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.floor(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.0001f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = floor(argument_float_h_mem[i * dimensions[1] + j]);
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Round) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2 && destDataType != TensorDescriptor::DataType::UINT8 && destDataType != TensorDescriptor::DataType::UINT16 &&
                    destDataType != TensorDescriptor::DataType::UINT32) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.round(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float thresh = 0.0001f;
                if ((abs(argument_float_h_mem[i * dimensions[1] + j]) - (int32_t)abs(argument_float_h_mem[i * dimensions[1] + j])) - 0.5f <
                    0.0001)
                    thresh = 1.0f;
                float expected = round(argument_float_h_mem[i * dimensions[1] + j]);
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (destDataType == TensorDescriptor::DataType::FP16 && abs(expected) >= 50.0f && thresh < 1.0f)
                    thresh = 1.0f;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType, (int)sourceDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, TruncateFloatingPoint) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 5000) / 100.0f;
                if (rand() % 2 && destDataType != TensorDescriptor::DataType::UINT8 && destDataType != TensorDescriptor::DataType::UINT16 &&
                    destDataType != TensorDescriptor::DataType::UINT32) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.truncateFloatingPoint(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float thresh = 0.0f;
                float expected;
                if (sourceDataType == TensorDescriptor::DataType::FP32)
                    expected = int32_t(argument_float_h_mem[i * dimensions[1] + j]);
                else
                    expected = int32_t((half)(argument_float_h_mem[i * dimensions[1] + j]));
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType, (int)sourceDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Reciprocal) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP16;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2 && destDataType != TensorDescriptor::DataType::UINT8 && destDataType != TensorDescriptor::DataType::UINT16 &&
                    destDataType != TensorDescriptor::DataType::UINT32)
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.reciprocal(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float thresh = 0.01f;
                half expectedHalf = (half)1.0f / (half)argument_float_h_mem[i * dimensions[1] + j];
                float expected = expectedHalf;
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (destDataType == TensorDescriptor::DataType::FP16 && abs(expected) >= 50.0f && thresh < 1.0f)
                    thresh = 1.0f;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType, (int)sourceDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Sqrt) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 100) / ((rand() % 100) + 1.0);
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.sqrt(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float thresh = 0.01f;
                float expected = sqrt(argument_float_h_mem[i * dimensions[1] + j]);
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (destDataType == TensorDescriptor::DataType::FP16 && abs(expected) >= 50.0f && thresh < 1.0f)
                    thresh = 1.0f;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //     expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType, (int)sourceDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, ReciprocalSqrt) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else if (dt == 1)
            destDataType = TensorDescriptor::DataType::FP32;
        else if (dt == 2)
            destDataType = TensorDescriptor::DataType::UINT8;
        else if (dt == 3)
            destDataType = TensorDescriptor::DataType::UINT16;
        else if (dt == 4)
            destDataType = TensorDescriptor::DataType::UINT32;
        else if (dt == 5)
            destDataType = TensorDescriptor::DataType::INT8;
        else if (dt == 6)
            destDataType = TensorDescriptor::DataType::INT16;
        else
            destDataType = TensorDescriptor::DataType::INT32;

        TensorDescriptor::DataType sourceDataType;
        uint32_t st = rand() % 2;
        if (st == 0)
            sourceDataType = TensorDescriptor::DataType::FP16;
        else
            sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.reciprocalSqrt(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float thresh = 0.01f;
                float expected = 1.0f / sqrt(argument_float_h_mem[i * dimensions[1] + j]);
                if ((destDataType == TensorDescriptor::DataType::UINT8 || destDataType == TensorDescriptor::DataType::UINT16 ||
                     destDataType == TensorDescriptor::DataType::UINT32) &&
                    expected <= 0.0f)
                    expected = 0;
                if (destDataType == TensorDescriptor::DataType::FP16 && abs(expected) >= 50.0f && thresh < 1.0f)
                    thresh = 1.0f;
                if (destDataType != TensorDescriptor::DataType::FP16 && destDataType != TensorDescriptor::DataType::FP32)
                    expected = (int32_t)expected;
                // printf("%f %f [%ld]  %f %f %d %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //    expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType, (int)sourceDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Erf) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else
            destDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.erf(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = erf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Erfinv) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else
            destDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 1001) / 1000.0f;
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.erfinv(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                // erf(erfinv(x)) == x
                float expected = erf(dest_gpu_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, argument_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //     expected, dest_gpu_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - argument_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Erfc) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else
            destDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.erfc(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = erfc(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //        expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Erfcinv) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else
            destDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 2001) / 1000.0f;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.erfcinv(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                // erf(erfinv(x)) == x
                float expected = erfc(dest_gpu_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, argument_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //     expected, dest_gpu_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - argument_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

float erfcx(float x) { return exp(pow(x, 2)) * erfc(x); }

TEST(Tensor, Erfcx) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else
            destDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 3) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.erfcx(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = erfcx(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //        expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Tgamma) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else
            destDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 5) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (int32_t(argument_float_h_mem[i * dimensions[1] + j]) == argument_float_h_mem[i * dimensions[1] + j])
                    argument_float_h_mem[i * dimensions[1] + j] = -0.8f;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.tgamma(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = tgamma(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //        expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Lgamma) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 20; ++t) {
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType destDataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            destDataType = TensorDescriptor::DataType::FP16;
        else
            destDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor::DataType sourceDataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor destDescriptor(destDataType, dimensions);
        TensorDescriptor termDescriptor(sourceDataType, dimensions);

        Tensor argument_h(cpuPlacement, termDescriptor);
        Tensor argument_float_h = argument_h.clone(TensorDescriptor::DataType::FP32);
        Tensor argument_d = argument_h.clone(gpuPlacement);
        Tensor dest_d(gpuPlacement, destDescriptor);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 100) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (int32_t(argument_float_h_mem[i * dimensions[1] + j]) == argument_float_h_mem[i * dimensions[1] + j])
                    argument_float_h_mem[i * dimensions[1] + j] = -0.8f;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.lgamma(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = lgamma(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //        expected, argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, MinScalar) {
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

        Tensor mem_h(cpuPlacement, descriptor);
        Tensor mem_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);
        Tensor mem_d(gpuPlacement, descriptor);
        Tensor mem_gpu_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);

        float *mem_float_h_mem = (float *)mem_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    mem_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                else
                    mem_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    mem_float_h_mem[i * dimensions[1] + j] = -mem_float_h_mem[i * dimensions[1] + j];
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10;
        if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32)
            scalar = -scalar;

        mem_h.copyFromAsync(mem_float_h, stream);
        mem_d.copyFromAsync(mem_h, stream);
        mem_d.min(scalar, stream);
        mem_gpu_float_h.copyFromAsync(mem_d, stream);
        stream.synchronize();

        float thresh = 0.003;
        float *mem_gpu_float_h_mem = (float *)mem_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            float original = mem_float_h_mem[i];
            float expected = original > scalar ? scalar : original;
            if (abs(expected - mem_gpu_float_h_mem[i]) >= thresh)
                printf("[%d] expected %f actual %f original %f maxValue %f\n", i, expected, mem_gpu_float_h_mem[i], original, scalar);
            ASSERT_LT(abs(expected - mem_gpu_float_h_mem[i]), thresh);
        }
    }
}

TEST(Tensor, MaxScalar) {
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

        Tensor mem_h(cpuPlacement, descriptor);
        Tensor mem_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);
        Tensor mem_d(gpuPlacement, descriptor);
        Tensor mem_gpu_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);

        float *mem_float_h_mem = (float *)mem_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    mem_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                else
                    mem_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    mem_float_h_mem[i * dimensions[1] + j] = -mem_float_h_mem[i * dimensions[1] + j];
            }
        }
        float scalar;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
            scalar = (rand() % 20) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10;
        if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32)
            scalar = -scalar;

        mem_h.copyFromAsync(mem_float_h, stream);
        mem_d.copyFromAsync(mem_h, stream);
        mem_d.max(scalar, stream);
        mem_gpu_float_h.copyFromAsync(mem_d, stream);
        stream.synchronize();

        float thresh = 0.003;
        float *mem_gpu_float_h_mem = (float *)mem_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            float original = mem_float_h_mem[i];
            float expected = original < scalar ? scalar : original;
            if (abs(expected - mem_gpu_float_h_mem[i]) >= thresh)
                printf("[%d] expected %f actual %f original %f maxValue %f\n", i, expected, mem_gpu_float_h_mem[i], original, scalar);
            ASSERT_LT(abs(expected - mem_gpu_float_h_mem[i]), thresh);
        }
    }
}

TEST(Tensor, Bound) {
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

        Tensor mem_h(cpuPlacement, descriptor);
        Tensor mem_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);
        Tensor mem_d(gpuPlacement, descriptor);
        Tensor mem_gpu_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);

        float *mem_float_h_mem = (float *)mem_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32)
                    mem_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                else
                    mem_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    mem_float_h_mem[i * dimensions[1] + j] = -mem_float_h_mem[i * dimensions[1] + j];
            }
        }
        float minValue, maxValue;
        if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
            minValue = (rand() % 20) / ((rand() % 10) + 1.0);
            maxValue = (rand() % 20) / ((rand() % 10) + 1.0);
        } else {
            minValue = rand() % 10;
            maxValue = rand() % 10;
        }
        if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32)
            minValue = -minValue;
        if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
            dataType != TensorDescriptor::DataType::UINT32)
            maxValue = -maxValue;
        if (maxValue < minValue)
            swap(maxValue, minValue);

        mem_h.copyFromAsync(mem_float_h, stream);
        mem_d.copyFromAsync(mem_h, stream);
        mem_d.bound(minValue, maxValue, stream);
        mem_gpu_float_h.copyFromAsync(mem_d, stream);
        stream.synchronize();

        float thresh = 0.003;
        float *mem_gpu_float_h_mem = (float *)mem_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            float original = mem_float_h_mem[i];
            float expected;
            if (original > maxValue)
                expected = maxValue;
            else if (original > minValue)
                expected = original;
            else  // False fall through case (i.e. NaN) should go to min value.
                expected = minValue;
            if (abs(expected - mem_gpu_float_h_mem[i]) >= thresh)
                printf("[%d] expected %f actual %f original %f minValue %f, maxValue %f\n",
                       i,
                       expected,
                       mem_gpu_float_h_mem[i],
                       original,
                       minValue,
                       maxValue);
            ASSERT_LT(abs(expected - mem_gpu_float_h_mem[i]), thresh);
        }
    }
}

TEST(Tensor, MinTensor) {
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

        Tensor mem_h(cpuPlacement, descriptor);
        Tensor mem_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);
        Tensor other_h(cpuPlacement, descriptor);
        Tensor other_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);
        Tensor mem_d(gpuPlacement, descriptor);
        Tensor other_d(gpuPlacement, descriptor);
        Tensor mem_gpu_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);

        float *mem_float_h_mem = (float *)mem_float_h.getMemPtr();
        float *other_float_h_mem = (float *)other_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
                    mem_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                    other_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                } else {
                    mem_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                    other_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                }
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    mem_float_h_mem[i * dimensions[1] + j] = -mem_float_h_mem[i * dimensions[1] + j];
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    other_float_h_mem[i * dimensions[1] + j] = -other_float_h_mem[i * dimensions[1] + j];
            }
        }

        mem_h.copyFromAsync(mem_float_h, stream);
        mem_d.copyFromAsync(mem_h, stream);
        other_h.copyFromAsync(other_float_h, stream);
        other_d.copyFromAsync(other_h, stream);
        mem_d.min(other_d, stream);
        mem_gpu_float_h.copyFromAsync(mem_d, stream);
        stream.synchronize();

        float thresh = 0.003;
        float *mem_gpu_float_h_mem = (float *)mem_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            float original = mem_float_h_mem[i];
            float expected = original > other_float_h_mem[i] ? other_float_h_mem[i] : original;
            if (abs(expected - mem_gpu_float_h_mem[i]) >= thresh)
                printf("[%d] expected %f actual %f original %f maxValue %f\n",
                       i,
                       expected,
                       mem_gpu_float_h_mem[i],
                       original,
                       other_float_h_mem[i]);
            ASSERT_LT(abs(expected - mem_gpu_float_h_mem[i]), thresh);
        }
    }
}

TEST(Tensor, MaxTensor) {
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

        Tensor mem_h(cpuPlacement, descriptor);
        Tensor mem_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);
        Tensor other_h(cpuPlacement, descriptor);
        Tensor other_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);
        Tensor mem_d(gpuPlacement, descriptor);
        Tensor other_d(gpuPlacement, descriptor);
        Tensor mem_gpu_float_h = mem_h.clone(TensorDescriptor::DataType::FP32);

        float *mem_float_h_mem = (float *)mem_float_h.getMemPtr();
        float *other_float_h_mem = (float *)other_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
                    mem_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                    other_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                } else {
                    mem_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                    other_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                }
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    mem_float_h_mem[i * dimensions[1] + j] = -mem_float_h_mem[i * dimensions[1] + j];
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    other_float_h_mem[i * dimensions[1] + j] = -other_float_h_mem[i * dimensions[1] + j];
            }
        }

        mem_h.copyFromAsync(mem_float_h, stream);
        mem_d.copyFromAsync(mem_h, stream);
        other_h.copyFromAsync(other_float_h, stream);
        other_d.copyFromAsync(other_h, stream);
        mem_d.max(other_d, stream);
        mem_gpu_float_h.copyFromAsync(mem_d, stream);
        stream.synchronize();

        float thresh = 0.003;
        float *mem_gpu_float_h_mem = (float *)mem_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0] * dimensions[1]; ++i) {
            float original = mem_float_h_mem[i];
            float expected = original < other_float_h_mem[i] ? other_float_h_mem[i] : original;
            if (abs(expected - mem_gpu_float_h_mem[i]) >= thresh)
                printf("[%d] expected %f actual %f original %f maxValue %f\n",
                       i,
                       expected,
                       mem_gpu_float_h_mem[i],
                       original,
                       other_float_h_mem[i]);
            ASSERT_LT(abs(expected - mem_gpu_float_h_mem[i]), thresh);
        }
    }
}

TEST(Tensor, Transpose) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else
            dataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(TensorDescriptor::DataType::FP32);
        Tensor source_d(gpuPlacement, descriptor);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
                    source_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                } else {
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                }
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = -source_float_h_mem[i * dimensions[1] + j];
            }
        }

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        Tensor dest_d = source_d.transposeMatrix(stream);
        ASSERT_EQ(dest_d.getDimensions().size(), 2);
        ASSERT_EQ(dest_d.getDimensions()[0], dimensions[1]);
        ASSERT_EQ(dest_d.getDimensions()[1], dimensions[0]);
        Tensor dest_gpu_h = dest_d.clone(cpuPlacement);
        Tensor dest_gpu_float_h = dest_gpu_h.clone(TensorDescriptor::DataType::FP32);
        dest_gpu_h.copyFromAsync(dest_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_gpu_h, stream);
        stream.synchronize();

        float thresh = 0.003;
        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = source_float_h_mem[i * dimensions[1] + j];
                float actual = dest_gpu_float_h_mem[j * dimensions[0] + i];
                if (abs(expected - actual) >= thresh)
                    printf("[%d][%d] expected %f actual %f\n", i, j, expected, actual);
                ASSERT_LT(abs(expected - actual), thresh);
            }
        }
    }
}

TEST(Tensor, TransposeSquareMatrixInPlace) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        uint32_t width = 1 + (rand() % 200);
        dimensions.push_back(width);
        dimensions.push_back(width);
        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            dataType = TensorDescriptor::DataType::FP16;
        else
            dataType = TensorDescriptor::DataType::FP32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(TensorDescriptor::DataType::FP32);
        Tensor source_d(gpuPlacement, descriptor);
        Tensor source_gpu_float_h = source_h.clone(TensorDescriptor::DataType::FP32);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == TensorDescriptor::DataType::FP16 || dataType == TensorDescriptor::DataType::FP32) {
                    source_float_h_mem[i * dimensions[1] + j] = ((rand() % 10) + 1) / ((rand() % 10) + 1.0);
                } else {
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 10) + 1;
                }
                if (rand() % 2 && dataType != TensorDescriptor::DataType::UINT8 && dataType != TensorDescriptor::DataType::UINT16 &&
                    dataType != TensorDescriptor::DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = -source_float_h_mem[i * dimensions[1] + j];
            }
        }

        source_h.copyFromAsync(source_float_h, stream);
        source_d.copyFromAsync(source_h, stream);
        source_d.transposeSquareMatrixInPlace(stream);
        source_h.copyFromAsync(source_d, stream);
        source_gpu_float_h.copyFromAsync(source_h, stream);
        stream.synchronize();

        float thresh = 0.003;
        float *source_gpu_float_h_mem = (float *)source_gpu_float_h.getMemPtr();
        for (uint32_t i = 0; i < width; ++i) {
            for (uint32_t j = 0; j < width; ++j) {
                float expected = source_float_h_mem[i * dimensions[1] + j];
                float actual = source_gpu_float_h_mem[j * dimensions[0] + i];
                if (abs(expected - actual) >= thresh)
                    printf("[%d][%d] expected %f actual %f\n", i, j, expected, actual);
                ASSERT_LT(abs(expected - actual), thresh);
            }
        }
    }
}

TEST(Tensor, FillCpu) {
    srand(time(nullptr));

    Stream stream(0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

    for (uint32_t test = 0; test < 20; ++test) {
        uint32_t numDimensions = 1 + (rand() % 5);
        vector<uint64_t> dimensions;
        uint32_t maxDimensionSize = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimensionSize = pow(10000000.0, 1.0 / numDimensions);
        uint32_t totalNumElements = 1;
        while (dimensions.size() < numDimensions) {
            dimensions.push_back(1 + (rand() % maxDimensionSize));
            totalNumElements *= dimensions.back();
        }

        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 10;
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
        else if (dt == 7)
            dataType = TensorDescriptor::DataType::INT32;
        else if (dt == 8)
            dataType = TensorDescriptor::DataType::BOOLEAN;
        else
            dataType = TensorDescriptor::DataType::PACKED_BOOLEAN;

        TensorDescriptor descriptor(dataType, dimensions);

        float fillValue;
        if (dt < 2) {
            fillValue = (rand() % 100) / (1.0 + (rand() % 50));
            if (rand() % 2)
                fillValue = -fillValue;
        } else if (dt == 2) {
            fillValue = rand() % 100;
        } else if (dt == 3) {
            fillValue = rand() % 1000;
        } else if (dt == 4) {
            fillValue = rand() % 10000;
        } else if (dt == 5) {
            fillValue = rand() % 100;
            if (rand() % 2)
                fillValue = -fillValue;
        } else if (dt == 6) {
            fillValue = rand() % 1000;
            if (rand() % 2)
                fillValue = -fillValue;
        } else if (dt == 7) {
            fillValue = rand() % 10000;
            if (rand() % 2)
                fillValue = -fillValue;
        } else {
            fillValue = rand() % 2 ? true : false;
        }

        Tensor t_h(cpuPlacement, descriptor);
        t_h.fill(fillValue, stream);
        stream.synchronize();

        if (dt == 0) {
            half *mem = (half *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                half value = mem[i];
                ASSERT_EQ((half)fillValue, value);
            }
        } else if (dt == 1) {
            float *mem = (float *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                float value = mem[i];
                ASSERT_EQ((float)fillValue, value);
            }
        } else if (dt == 2) {
            uint8_t *mem = (uint8_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                uint8_t value = mem[i];
                ASSERT_EQ((uint8_t)fillValue, value);
            }
        } else if (dt == 3) {
            uint16_t *mem = (uint16_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                uint16_t value = mem[i];
                ASSERT_EQ((uint16_t)fillValue, value);
            }
        } else if (dt == 4) {
            uint32_t *mem = (uint32_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                uint32_t value = mem[i];
                ASSERT_EQ((uint32_t)fillValue, value);
            }
        } else if (dt == 5) {
            int8_t *mem = (int8_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                int8_t value = mem[i];
                ASSERT_EQ((int8_t)fillValue, value);
            }
        } else if (dt == 6) {
            int16_t *mem = (int16_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                int16_t value = mem[i];
                ASSERT_EQ((int16_t)fillValue, value);
            }
        } else if (dt == 7) {
            int32_t *mem = (int32_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                int32_t value = mem[i];
                ASSERT_EQ((int32_t)fillValue, value);
            }
        } else if (dt == 8) {
            bool *mem = (bool *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                bool value = mem[i];
                ASSERT_EQ((bool)fillValue, value);
            }
        } else {
            uint8_t expected = fillValue ? 0b11111111 : 0b00000000;
            uint8_t *mem = (uint8_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < (totalNumElements + 7) / 8; ++i) {
                uint8_t value = mem[i];
                ASSERT_EQ(expected, value);
            }
        }
    }
}

TEST(Tensor, FillGpu) {
    srand(time(nullptr));

    Stream stream(0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);

    for (uint32_t test = 0; test < 1; ++test) {
        uint32_t numDimensions = 1 + (rand() % 5);
        vector<uint64_t> dimensions;
        uint32_t maxDimensionSize = pow(100000.0, 1.0 / numDimensions);
        uint32_t totalNumElements = 1;
        while (dimensions.size() < numDimensions) {
            dimensions.push_back(1 + (rand() % maxDimensionSize));
            totalNumElements *= dimensions.back();
        }

        TensorDescriptor::DataType dataType;
        uint32_t dt = rand() % 10;
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
        else if (dt == 7)
            dataType = TensorDescriptor::DataType::INT32;
        else if (dt == 8)
            dataType = TensorDescriptor::DataType::BOOLEAN;
        else
            dataType = TensorDescriptor::DataType::PACKED_BOOLEAN;

        TensorDescriptor descriptor(dataType, dimensions);

        float fillValue;
        if (dt < 2) {
            fillValue = (rand() % 100) / (1.0 + (rand() % 50));
            if (rand() % 2)
                fillValue = -fillValue;
        } else if (dt == 2) {
            fillValue = rand() % 100;
        } else if (dt == 3) {
            fillValue = rand() % 1000;
        } else if (dt == 4) {
            fillValue = rand() % 10000;
        } else if (dt == 5) {
            fillValue = rand() % 100;
            if (rand() % 2)
                fillValue = -fillValue;
        } else if (dt == 6) {
            fillValue = rand() % 1000;
            if (rand() % 2)
                fillValue = -fillValue;
        } else if (dt == 7) {
            fillValue = rand() % 10000;
            if (rand() % 2)
                fillValue = -fillValue;
        } else {
            fillValue = rand() % 2 ? true : false;
        }

        Tensor t_h(cpuPlacement, descriptor);
        Tensor t_d = t_h.clone(gpuPlacement);
        t_d.fill(fillValue, stream);
        t_h.copyFromAsync(t_d, stream);
        stream.synchronize();

        if (dt == 0) {
            half *mem = (half *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                half value = mem[i];
                ASSERT_EQ((half)fillValue, value);
            }
        } else if (dt == 1) {
            float *mem = (float *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                float value = mem[i];
                ASSERT_EQ((float)fillValue, value);
            }
        } else if (dt == 2) {
            uint8_t *mem = (uint8_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                uint8_t value = mem[i];
                ASSERT_EQ((uint8_t)fillValue, value);
            }
        } else if (dt == 3) {
            uint16_t *mem = (uint16_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                uint16_t value = mem[i];
                ASSERT_EQ((uint16_t)fillValue, value);
            }
        } else if (dt == 4) {
            uint32_t *mem = (uint32_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                uint32_t value = mem[i];
                ASSERT_EQ((uint32_t)fillValue, value);
            }
        } else if (dt == 5) {
            int8_t *mem = (int8_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                int8_t value = mem[i];
                ASSERT_EQ((int8_t)fillValue, value);
            }
        } else if (dt == 6) {
            int16_t *mem = (int16_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                int16_t value = mem[i];
                ASSERT_EQ((int16_t)fillValue, value);
            }
        } else if (dt == 7) {
            int32_t *mem = (int32_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                int32_t value = mem[i];
                ASSERT_EQ((int32_t)fillValue, value);
            }
        } else if (dt == 8) {
            bool *mem = (bool *)t_h.getMemPtr();
            for (uint32_t i = 0; i < totalNumElements; ++i) {
                bool value = mem[i];
                ASSERT_EQ((bool)fillValue, value);
            }
        } else {
            // Note that this is actually wrong for a packed boolean whose flags are segregated to the proper dimension
            // packed boolean needs to be fixed overall and this condition will change then.
            uint8_t expected = fillValue ? 0b11111111 : 0b00000000;
            uint8_t *mem = (uint8_t *)t_h.getMemPtr();
            for (uint32_t i = 0; i < (totalNumElements + 7) / 8; ++i) {
                uint8_t value = mem[i];
                ASSERT_EQ(expected, value);
            }
        }
    }
}

TEST(Tensor, MemsetCpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

    for (uint32_t test = 0; test < 5; ++test) {
        uint32_t numDimensions = 1 + (rand() % 5);
        vector<uint64_t> dimensions;
        uint32_t maxDimensionSize = pow(100000.0, 1.0 / numDimensions);
        uint32_t totalNumElements = 1;
        while (dimensions.size() < numDimensions) {
            dimensions.push_back(1 + (rand() % maxDimensionSize));
            totalNumElements *= dimensions.back();
        }

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

        int8_t fillValue;
        fillValue = rand() % 20;
        uint64_t numElementToSet = 1 + rand() % totalNumElements;

        Tensor t_h(cpuPlacement, descriptor);
        // First set all elements to a value representing their prior state
        // Then set the desired number of elements, then need to check all elements
        t_h.memset(fillValue + 1);
        t_h.memset(fillValue, numElementToSet);

        int8_t *tMem_h = (int8_t *)t_h.getMemPtr();
        uint64_t totalNumFilledBytes = numElementToSet * (t_h.getArraySizeInBytes() / t_h.getTotalNumElements());
        for (uint64_t i = 0; i < totalNumFilledBytes; ++i) {
            int8_t expected = fillValue;
            if (i >= totalNumFilledBytes)
                expected = fillValue + 1;

            int8_t actual = tMem_h[i];

            if (expected != actual)
                printf("numElementsToSet %ld i %ld fillValue %d, value %d dt %d\n",
                       numElementToSet,
                       i,
                       (int32_t)fillValue,
                       (int32_t)actual,
                       dt);
            ASSERT_EQ((uint32_t)expected, (uint32_t)actual);
        }
    }
}

TEST(Tensor, MemsetAsyncCpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t test = 0; test < 5; ++test) {
        uint32_t numDimensions = 1 + (rand() % 5);
        vector<uint64_t> dimensions;
        uint32_t maxDimensionSize = pow(100000.0, 1.0 / numDimensions);
        uint32_t totalNumElements = 1;
        while (dimensions.size() < numDimensions) {
            dimensions.push_back(1 + (rand() % maxDimensionSize));
            totalNumElements *= dimensions.back();
        }

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

        int8_t fillValue;
        fillValue = rand() % 20;
        uint64_t numElementToSet = 1 + rand() % totalNumElements;

        Tensor t_h(cpuPlacement, descriptor);
        // First set all elements to a value representing their prior state
        // Then set the desired number of elements, then need to check all elements
        t_h.memsetAsync(stream, fillValue + 1);
        t_h.memsetAsync(stream, fillValue, numElementToSet);
        stream.synchronize();

        int8_t *tMem_h = (int8_t *)t_h.getMemPtr();
        uint64_t totalNumFilledBytes = numElementToSet * (t_h.getArraySizeInBytes() / t_h.getTotalNumElements());
        for (uint64_t i = 0; i < totalNumFilledBytes; ++i) {
            int8_t expected = fillValue;
            if (i >= totalNumFilledBytes)
                expected = fillValue + 1;

            int8_t actual = tMem_h[i];

            if (expected != actual)
                printf("numElementsToSet %ld i %ld fillValue %d, value %d dt %d\n",
                       numElementToSet,
                       i,
                       (int32_t)fillValue,
                       (int32_t)actual,
                       dt);
            ASSERT_EQ((uint32_t)expected, (uint32_t)actual);
        }
    }
}

TEST(Tensor, ClearCpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

    for (uint32_t test = 0; test < 5; ++test) {
        uint32_t numDimensions = 1 + (rand() % 5);
        vector<uint64_t> dimensions;
        uint32_t maxDimensionSize = pow(100000.0, 1.0 / numDimensions);
        uint32_t totalNumElements = 1;
        while (dimensions.size() < numDimensions) {
            dimensions.push_back(1 + (rand() % maxDimensionSize));
            totalNumElements *= dimensions.back();
        }

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

        int8_t fillValue;
        fillValue = rand() % 20;
        uint64_t numElementToSet = 1 + rand() % totalNumElements;

        Tensor t_h(cpuPlacement, descriptor);
        // First set all elements to a value representing their prior state
        t_h.memset(12);
        t_h.clear();

        int8_t *tMem_h = (int8_t *)t_h.getMemPtr();
        uint64_t totalNumFilledBytes = t_h.getArraySizeInBytes();
        for (uint64_t i = 0; i < totalNumFilledBytes; ++i) {
            int8_t expected = 0;
            int8_t actual = tMem_h[i];
            if (expected != actual)
                printf("numElementsToSet %ld i %ld fillValue %d, value %d dt %d\n",
                       numElementToSet,
                       i,
                       (int32_t)fillValue,
                       (int32_t)actual,
                       dt);
            ASSERT_EQ((uint32_t)expected, (uint32_t)actual);
        }
    }
}
TEST(Tensor, ClearAsyncCpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t test = 0; test < 5; ++test) {
        uint32_t numDimensions = 1 + (rand() % 5);
        vector<uint64_t> dimensions;
        uint32_t maxDimensionSize = pow(100000.0, 1.0 / numDimensions);
        uint32_t totalNumElements = 1;
        while (dimensions.size() < numDimensions) {
            dimensions.push_back(1 + (rand() % maxDimensionSize));
            totalNumElements *= dimensions.back();
        }

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

        int8_t fillValue;
        fillValue = rand() % 20;
        uint64_t numElementToSet = 1 + rand() % totalNumElements;

        Tensor t_h(cpuPlacement, descriptor);
        // First set all elements to a value representing their prior state
        t_h.memset(12);
        t_h.clearAsync(stream);
        stream.synchronize();

        int8_t *tMem_h = (int8_t *)t_h.getMemPtr();
        uint64_t totalNumFilledBytes = t_h.getArraySizeInBytes();
        for (uint64_t i = 0; i < totalNumFilledBytes; ++i) {
            int8_t expected = 0;
            int8_t actual = tMem_h[i];
            if (expected != actual)
                printf("numElementsToSet %ld i %ld fillValue %d, value %d dt %d\n",
                       numElementToSet,
                       i,
                       (int32_t)fillValue,
                       (int32_t)actual,
                       dt);
            ASSERT_EQ((uint32_t)expected, (uint32_t)actual);
        }
    }
}

TEST(Tensor, MemsetAsyncGpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);

    for (uint32_t test = 0; test < 5; ++test) {
        uint32_t numDimensions = 1 + (rand() % 5);
        vector<uint64_t> dimensions;
        uint32_t maxDimensionSize = pow(100000.0, 1.0 / numDimensions);
        uint32_t totalNumElements = 1;
        while (dimensions.size() < numDimensions) {
            dimensions.push_back(1 + (rand() % maxDimensionSize));
            totalNumElements *= dimensions.back();
        }

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

        int8_t fillValue;
        fillValue = rand() % 20;
        uint64_t numElementToSet = 1 + rand() % totalNumElements;

        Tensor t_h(cpuPlacement, descriptor);
        Tensor t_d = t_h.clone(gpuPlacement);
        // First set all elements to a value representing their prior state
        // Then set the desired number of elements, then need to check all elements
        t_d.memsetAsync(stream, fillValue + 1);
        t_d.memsetAsync(stream, fillValue, numElementToSet);
        t_h.copyFromAsync(t_d, stream);
        stream.synchronize();

        int8_t *tMem_h = (int8_t *)t_h.getMemPtr();
        uint64_t totalNumFilledBytes = numElementToSet * (t_h.getArraySizeInBytes() / t_h.getTotalNumElements());
        for (uint64_t i = 0; i < totalNumFilledBytes; ++i) {
            int8_t expected = fillValue;
            if (i >= totalNumFilledBytes)
                expected = fillValue + 1;

            int8_t actual = tMem_h[i];

            if (expected != actual)
                printf("numElementsToSet %ld i %ld fillValue %d, value %d dt %d\n",
                       numElementToSet,
                       i,
                       (int32_t)fillValue,
                       (int32_t)actual,
                       dt);
            ASSERT_EQ((uint32_t)expected, (uint32_t)actual);
        }
    }
}

TEST(Tensor, ClearAsyncGpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);

    for (uint32_t test = 0; test < 5; ++test) {
        uint32_t numDimensions = 1 + (rand() % 5);
        vector<uint64_t> dimensions;
        uint32_t maxDimensionSize = pow(100000.0, 1.0 / numDimensions);
        uint32_t totalNumElements = 1;
        while (dimensions.size() < numDimensions) {
            dimensions.push_back(1 + (rand() % maxDimensionSize));
            totalNumElements *= dimensions.back();
        }

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

        Tensor t_h(cpuPlacement, descriptor);
        Tensor t_d = t_h.clone(gpuPlacement);
        // First set all elements to a value representing their prior state
        // Then set the desired number of elements, then need to check all elements
        t_d.memsetAsync(stream, 9);
        t_d.clearAsync(stream);
        t_h.copyFromAsync(t_d, stream);
        stream.synchronize();

        int8_t *tMem_h = (int8_t *)t_h.getMemPtr();
        uint64_t totalNumFilledBytes = t_h.getArraySizeInBytes();
        for (uint64_t i = 0; i < totalNumFilledBytes; ++i) {
            int8_t expected = 0;
            int8_t actual = tMem_h[i];
            ASSERT_EQ((uint32_t)expected, (uint32_t)actual);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
