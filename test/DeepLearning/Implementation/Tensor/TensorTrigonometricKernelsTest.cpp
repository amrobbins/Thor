#include "Thor.h"

#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

#pragma GCC diagnostic ignored "-Wsign-compare"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop

using namespace ThorImplementation;
using namespace std;

TEST(Tensor, Sin) {
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
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 100) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.sin(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = sinf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Cos) {
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
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.cos(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = cosf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

bool isTanAsymptote(float r) {
    r -= M_PI / 2.0;
    while (r < 0.0f) {
        if (r > -0.01f)
            return true;
        r += M_PI;
    }
    while (r > 0.0f) {
        if (r < 0.01f)
            return true;
        r -= M_PI;
    }
    if (r > -0.01f)
        return true;
    return false;
}

TEST(Tensor, Tan) {
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
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (isTanAsymptote(argument_float_h_mem[i * dimensions[1] + j]))
                    argument_float_h_mem[i * dimensions[1] + j] = M_PI;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.tan(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = tanf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

float cscf(float r) { return 1.0f / sinf(r); }

bool isCscAsymptote(float r) {
    r = abs(r);
    uint32_t piMultiples = r / M_PI;
    r -= piMultiples * M_PI;
    if (r < 0.01f)
        return true;
    return false;
}

TEST(Tensor, Csc) {
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
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (isCscAsymptote(argument_float_h_mem[i * dimensions[1] + j]))
                    argument_float_h_mem[i * dimensions[1] + j] = 1.5 * M_PI;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.csc(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = cscf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

float secf(float r) { return 1.0f / cosf(r); }

bool isSecAsymptote(float r) { return isTanAsymptote(r); }

TEST(Tensor, Sec) {
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
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (isSecAsymptote(argument_float_h_mem[i * dimensions[1] + j]))
                    argument_float_h_mem[i * dimensions[1] + j] = M_PI;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.sec(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.05f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = secf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

float cotf(float r) { return 1.0f / tanf(r); }

bool isCotAsymptote(float r) { return isCscAsymptote(r); }

TEST(Tensor, Cot) {
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
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
                if (isCotAsymptote(argument_float_h_mem[i * dimensions[1] + j]))
                    argument_float_h_mem[i * dimensions[1] + j] = 1.5 * M_PI;
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.cot(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = cotf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Asin) {
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
                argument_float_h_mem[i * dimensions[1] + j] = 1.0f / ((rand() % 100) + 1.0f);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.asin(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = asinf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Acos) {
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
                argument_float_h_mem[i * dimensions[1] + j] = 1.0f / ((rand() % 100) + 1.0f);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.acos(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = acosf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Atan) {
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
                argument_float_h_mem[i * dimensions[1] + j] = ((rand() % 1000) + 1) / ((rand() % 1000) + 1.0f);
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.atan(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = atanf(argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Acsc) {
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
                argument_float_h_mem[i * dimensions[1] + j] = 1.05f + (((rand() % 1000) + 1) / ((rand() % 1000) + 1.0f));
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.acsc(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = asinf(1.0f / argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Asec) {
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
                argument_float_h_mem[i * dimensions[1] + j] = 1.05f + (((rand() % 1000) + 1) / ((rand() % 1000) + 1.0f));
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.asec(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = acosf(1.0f / argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //       expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}

TEST(Tensor, Acot) {
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
                argument_float_h_mem[i * dimensions[1] + j] = (((rand() % 1000) + 1) / ((rand() % 1000) + 1.0f));
                if (rand() % 2) {
                    argument_float_h_mem[i * dimensions[1] + j] = -argument_float_h_mem[i * dimensions[1] + j];
                }
            }
        }

        argument_h.copyFromAsync(argument_float_h, stream);
        argument_d.copyFromAsync(argument_h, stream);
        dest_d.acot(argument_d, stream);
        dest_gpu_float_h.copyFromAsync(dest_d, stream);
        stream.synchronize();

        float *dest_gpu_float_h_mem = (float *)dest_gpu_float_h.getMemPtr();
        float thresh = 0.01f;
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                float expected = atanf(1.0f / argument_float_h_mem[i * dimensions[1] + j]);
                if (destDataType == TensorDescriptor::DataType::FP16)
                    expected = (half)expected;
                // printf("%f %f [%ld]  %f %f %d\n", expected, dest_gpu_float_h_mem[i * dimensions[1] + j], i * dimensions[1] + j,
                //      expected, (float)(half)argument_float_h_mem[i * dimensions[1] + j], (int)destDataType);
                ASSERT_LE(abs(expected - dest_gpu_float_h_mem[i * dimensions[1] + j]), thresh);
            }
        }
    }
}
