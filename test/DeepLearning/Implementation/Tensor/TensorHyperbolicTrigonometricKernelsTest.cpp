#include "Thor.h"

#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

#pragma GCC diagnostic ignored "-Wsign-compare"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop

using namespace ThorImplementation;
using namespace std;

TEST(Tensor, Sinh) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 1000; ++t) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 2000) / 1000.0f;
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.sinh(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = sinhf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Cosh) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 1000; ++t) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 5000) / 1000.0f;
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.cosh(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = coshf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Tanh) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Stream stream(0);
    for (uint32_t t = 0; t < 1000; ++t) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.tanh(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = tanhf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

float cschf(float r) { return 1.0f / sinhf(r); }

bool isCschAsymptote(float r) {
    r = abs(r);
    if (r < 0.01)
        return true;
    return false;
}

TEST(Tensor, Csch) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (isCschAsymptote(argument_float_h_mem[i * dimensions[1] + j]))
                    argument_float_h_mem[i * dimensions[1] + j] = 1.5;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.csch(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.03f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = cschf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

float sechf(float r) { return 1.0f / coshf(r); }

TEST(Tensor, Sech) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 1000) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.sech(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.05f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = sechf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

float cothf(float r) { return 1.0f / tanhf(r); }

bool isCothAsymptote(float r) { return isCschAsymptote(r); }

TEST(Tensor, Coth) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (isCothAsymptote(argument_float_h_mem[i * dimensions[1] + j]))
                    argument_float_h_mem[i * dimensions[1] + j] = 1.5;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.coth(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = cothf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Asinh) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) / ((rand() % 100) + 1));
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.asinh(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = asinhf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Acosh) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = 1.01f + ((rand() % 1000) / ((rand() % 100) + 1));
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.acosh(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = acoshf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Atanh) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 9500) / 10000.0f;
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.atanh(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = atanhf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Acsch) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = (rand() % 10000) / 1000.0f;
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (isCschAsymptote(argument_float_h_mem[i * dimensions[1] + j]))
                    argument_float_h_mem[i * dimensions[1] + j] = 1.0f;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.acsch(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = asinhf(1.0f / argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Asech) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = 0.025f + ((rand() % 9000) / 10000.0f);
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.asech(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = acoshf(1.0f / argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

bool isAcothUndefined(float r) {
    r = abs(r);
    if (r < 1.05)
        return true;
    return false;
}

TEST(Tensor, Acoth) {
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
        Tensor dest_d = argument_h.clone(destDataType);
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, TensorDescriptor::DataType::FP32);

        float *argument_float_h_mem = (float *)argument_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 10000) + 1) / 1000.0f;
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (isAcothUndefined(argument_float_h_mem[i * dimensions[1] + j]))
                    argument_float_h_mem[i * dimensions[1] + j] = 2.0f;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.acoth(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = atanhf(1.0f / argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
