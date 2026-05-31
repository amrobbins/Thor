#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <fstream>
#include <iostream>
#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

#include "test/Utilities/TensorOperations/GpuMatrixMultiply/MatrixMultiplyTestHelper.h"

#pragma GCC diagnostic ignored "-Wsign-compare"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop

using namespace ThorImplementation;
using namespace std;

static vector<DataType> allWholeElementTensorDataTypes() {
    return {DataType::FP16,
            DataType::BF16,
            DataType::FP8_E4M3,
            DataType::FP8_E5M2,
            DataType::FP32,
            DataType::FP64,
            DataType::INT8,
            DataType::INT16,
            DataType::INT32,
            DataType::INT64,
            DataType::UINT8,
            DataType::UINT16,
            DataType::UINT32,
            DataType::UINT64,
            DataType::BOOLEAN};
}

static vector<DataType> allTensorDataTypes() {
    vector<DataType> dataTypes = allWholeElementTensorDataTypes();
    dataTypes.push_back(DataType::PACKED_BOOLEAN);
    return dataTypes;
}

static Tensor copyToCpuFp32ForVerification(Tensor source, Stream stream) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorDescriptor fp32Descriptor(DataType::FP32, source.getDimensions());
    Tensor result(cpuPlacement, fp32Descriptor);

    // The C++ Tensor copy contract intentionally rejects preserving cross-placement downcasts because they require
    // hidden temporaries.  Verification code still often wants a CPU FP32 view of arbitrary GPU tensors, so spell the
    // temporary explicitly here: downcast on the source GPU first, then copy the already-FP32 value to CPU.
    if (source.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU &&
        source.getDescriptor().getArraySizeInBytes() > fp32Descriptor.getArraySizeInBytes()) {
        Tensor convertedOnSourceGpu(source.getPlacement(), fp32Descriptor);
        convertedOnSourceGpu.copyFromAsync(source, stream);
        result.copyFromAsync(convertedOnSourceGpu, stream);
        stream.synchronize();
    } else {
        result.copyFromAsync(source, stream);
    }

    return result;
}

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
        TensorDescriptor descriptor(DataType::FP32, dimensions);

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

TEST(Tensor, CopiesSeeBackingMemorySwaps) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorDescriptor descriptor(DataType::FP32, {4});

    Tensor active(cpuPlacement, descriptor);
    Tensor prefetch(cpuPlacement, descriptor);
    Tensor stampedCopy = active;

    const uint64_t activeId = active.getTensorId();
    const uint64_t prefetchId = prefetch.getTensorId();

    float *activeMemBefore = active.getMemPtr<float>();
    float *prefetchMemBefore = prefetch.getMemPtr<float>();
    ASSERT_NE(activeMemBefore, prefetchMemBefore);

    for (uint32_t i = 0; i < 4; ++i) {
        activeMemBefore[i] = 10.0f + static_cast<float>(i);
        prefetchMemBefore[i] = 20.0f + static_cast<float>(i);
    }

    active.swapBackingMemoryWith(prefetch);

    ASSERT_EQ(active.getTensorId(), activeId);
    ASSERT_EQ(prefetch.getTensorId(), prefetchId);
    ASSERT_EQ(stampedCopy.getTensorId(), activeId);

    ASSERT_EQ(active.getMemPtr<float>(), prefetchMemBefore);
    ASSERT_EQ(stampedCopy.getMemPtr<float>(), prefetchMemBefore);
    ASSERT_EQ(prefetch.getMemPtr<float>(), activeMemBefore);

    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(stampedCopy.getMemPtr<float>()[i], 20.0f + static_cast<float>(i));
        ASSERT_EQ(prefetch.getMemPtr<float>()[i], 10.0f + static_cast<float>(i));
    }
}

// Reshape keeps contents unchanged
TEST(Tensor, PreservingCpuToCpuDowncastStaysCpuOnly) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    TensorDescriptor sourceDescriptor(DataType::FP64, {6});
    TensorDescriptor destDescriptor(DataType::FP32, {6});
    Tensor source(cpuPlacement, sourceDescriptor);
    Tensor dest(cpuPlacement, destDescriptor);

    double *sourceMem = static_cast<double *>(source.getMemPtr());
    for (uint32_t i = 0; i < 6; ++i) {
        sourceMem[i] = 1.25 + static_cast<double>(i);
    }

    dest.copyFromAsync(source, stream);
    stream.synchronize();

    float *destMem = static_cast<float *>(dest.getMemPtr());
    for (uint32_t i = 0; i < 6; ++i) {
        ASSERT_EQ(destMem[i], static_cast<float>(sourceMem[i]));
        ASSERT_EQ(sourceMem[i], 1.25 + static_cast<double>(i));
    }
}

TEST(Tensor, PreservingCrossPlacementDowncastThrowsWithoutExplicitTemporary) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);

    TensorDescriptor sourceDescriptor(DataType::FP32, {6});
    TensorDescriptor destDescriptor(DataType::FP16, {6});
    Tensor source(cpuPlacement, sourceDescriptor);
    Tensor dest(gpuPlacement, destDescriptor);

    float *sourceMem = static_cast<float *>(source.getMemPtr());
    for (uint32_t i = 0; i < 6; ++i) {
        sourceMem[i] = 1.25f + static_cast<float>(i);
    }

    try {
        dest.copyFromAsync(source, stream);
        FAIL() << "Expected preserving cross-placement downcast to require an explicit temporary on the C++ side";
    } catch (const std::runtime_error &e) {
        ASSERT_NE(std::string(e.what()).find("hidden temporary"), std::string::npos);
    }

    for (uint32_t i = 0; i < 6; ++i) {
        ASSERT_EQ(sourceMem[i], 1.25f + static_cast<float>(i));
    }
}

TEST(Tensor, Reshapes) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 5; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        TensorDescriptor descriptor(DataType::FP32, dimensions);

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
        TensorDescriptor descriptor(DataType::FP32, dimensions);

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

TEST(Tensor, AddTensor) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 20; ++t) {
        Stream stream(0);
        vector<unsigned long> dimensions;
        dimensions.push_back(1 + (rand() % 200));
        dimensions.push_back(1 + (rand() % 200));
        DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else
            dataType = DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor augend_h(cpuPlacement, descriptor);
        Tensor augend_float_h = augend_h.clone(DataType::FP32);
        Tensor augend_d = augend_h.clone(gpuPlacement);
        Tensor addend_h(cpuPlacement, descriptor);
        Tensor addend_float_h = augend_h.clone(DataType::FP32);
        Tensor addend_d = augend_h.clone(gpuPlacement);
        Tensor dest_d = augend_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, DataType::FP32);

        float *augend_float_h_mem = (float *)augend_float_h.getMemPtr();
        float *addend_float_h_mem = (float *)addend_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == DataType::FP16 || dataType == DataType::FP32) {
                    augend_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                    addend_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                } else {
                    augend_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                    addend_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                    if (dataType != DataType::UINT8 && dataType != DataType::UINT16 && dataType != DataType::UINT32) {
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

TEST(Tensor, identityMatrixCpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            dataType = DataType::FP16;
        else
            dataType = DataType::FP32;

        uint32_t N = 1 + (rand() % 300);

        Tensor I = Tensor::identityMatrix(N, cpuPlacement, dataType, stream);
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
    Tensor::identityMatrix(300, cpuPlacement, DataType::FP32, stream);
}

TEST(Tensor, identityMatrixGpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 2;
        if (dt == 0)
            dataType = DataType::FP16;
        else
            dataType = DataType::FP32;

        uint32_t N = 1 + (rand() % 300);

        Tensor I = Tensor::identityMatrix(N, gpuPlacement, dataType, stream);
        Tensor I_h = I.clone(cpuPlacement);
        I_h.copyFromAsync(I, stream);
        stream.synchronize();

        if (dt == 0) {
            half *mem = I_h.getMemPtr<half>();
            for (uint32_t row = 0; row < N; ++row) {
                for (uint32_t col = 0; col < N; ++col) {
                    if (row == col)
                        ASSERT_EQ(mem[row * N + col], half(1.0f));
                    else
                        ASSERT_EQ(mem[row * N + col], half(0.0f));
                }
            }
        } else {
            float *mem = I_h.getMemPtr<float>();
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
    Tensor::identityMatrix(300, cpuPlacement, DataType::FP32, stream);
}

TEST(Tensor, identityMatrixSupportsAllNonPackedDataTypesCpuAndGpu) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);
    const uint32_t N = 11;

    for (DataType dataType : allWholeElementTensorDataTypes()) {
        for (TensorPlacement placement : {cpuPlacement, gpuPlacement}) {
            Tensor identity = Tensor::identityMatrix(N, placement, dataType, stream);
            Tensor identityFp32 = copyToCpuFp32ForVerification(identity, stream);
            stream.synchronize();

            float *mem = identityFp32.getMemPtr<float>();
            for (uint32_t row = 0; row < N; ++row) {
                for (uint32_t col = 0; col < N; ++col) {
                    ASSERT_EQ(mem[row * N + col], row == col ? 1.0f : 0.0f)
                        << "dataType=" << TensorDescriptor::getElementTypeName(dataType) << " row=" << row << " col=" << col;
                }
            }
        }
    }
}

TEST(Tensor, zerosCpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::INT8;
        else if (dt == 3)
            dataType = DataType::INT16;
        else if (dt == 4)
            dataType = DataType::INT32;
        else if (dt == 5)
            dataType = DataType::UINT8;
        else if (dt == 6)
            dataType = DataType::UINT16;
        else if (dt == 7)
            dataType = DataType::UINT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else if (dt == 9)
            dataType = DataType::PACKED_BOOLEAN;

        uint32_t numDimensions = 1 + (rand() % 5);
        uint32_t maxDimension = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimension = pow(10000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t i = 0; i < numDimensions; ++i) {
            dimensions.push_back(1 + (rand() % maxDimension));
        }
        Tensor tensor = Tensor::zeros(cpuPlacement, TensorDescriptor(dataType, dimensions), stream);
        Tensor tensorFp32 = tensor.clone(DataType::FP32);
        tensorFp32.copyFromAsync(tensor, stream);
        stream.synchronize();

        float *mem = tensorFp32.getMemPtr<float>();
        for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
            ASSERT_TRUE(mem[i] == 0.0f);
        }
    }
}

TEST(Tensor, zerosGpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::INT8;
        else if (dt == 3)
            dataType = DataType::INT16;
        else if (dt == 4)
            dataType = DataType::INT32;
        else if (dt == 5)
            dataType = DataType::UINT8;
        else if (dt == 6)
            dataType = DataType::UINT16;
        else if (dt == 7)
            dataType = DataType::UINT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else if (dt == 9)
            dataType = DataType::PACKED_BOOLEAN;

        uint32_t numDimensions = 1 + (rand() % 5);
        uint32_t maxDimension = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimension = pow(10000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t i = 0; i < numDimensions; ++i) {
            dimensions.push_back(1 + (rand() % maxDimension));
        }
        Tensor tensor = Tensor::zeros(gpuPlacement, TensorDescriptor(dataType, dimensions), stream);
        Tensor tensorFp32 = tensor.clone(cpuPlacement, DataType::FP32);
        tensorFp32.copyFromAsync(tensor, stream);
        stream.synchronize();

        float *mem = tensorFp32.getMemPtr<float>();
        for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
            ASSERT_TRUE(mem[i] == 0.0f);
        }
    }
}

TEST(Tensor, randomsCpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::INT8;
        else if (dt == 3)
            dataType = DataType::INT16;
        else if (dt == 4)
            dataType = DataType::INT32;
        else if (dt == 5)
            dataType = DataType::UINT8;
        else if (dt == 6)
            dataType = DataType::UINT16;
        else if (dt == 7)
            dataType = DataType::UINT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else if (dt == 9)
            dataType = DataType::PACKED_BOOLEAN;

        uint32_t numDimensions = 1 + (rand() % 5);
        uint32_t maxDimension = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimension = pow(10000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t i = 0; i < numDimensions; ++i) {
            dimensions.push_back(1 + (rand() % maxDimension));
        }
        int16_t minValue;
        int16_t maxValue;
        if (dt < 5) {
            int r = rand() % 3;
            if (r == 0) {
                minValue = -100;
                maxValue = 100;
            } else if (r == 1) {
                minValue = 10;
                maxValue = 100;
            } else {
                minValue = -100;
                maxValue = -10;
            }
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
        Tensor tensor = Tensor::randoms(cpuPlacement, TensorDescriptor(dataType, dimensions), stream, minValue, maxValue);
        stream.synchronize();

        if (dt == 9) {
            uint8_t *mem = tensor.getMemPtr<uint8_t>();
            uint32_t numElements = (tensor.getTotalNumElements() + 7) / 8;
            for (uint32_t i = 0; i < numElements; ++i) {
                ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
            }
        } else {
            Tensor tensorFp32_h = tensor.clone(DataType::FP32);
            tensorFp32_h.copyFromAsync(tensor, stream);
            stream.synchronize();
            float *mem = tensorFp32_h.getMemPtr<float>();
            for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
                ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
            }
        }
    }
}

TEST(Tensor, randomsGpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::INT8;
        else if (dt == 3)
            dataType = DataType::INT16;
        else if (dt == 4)
            dataType = DataType::INT32;
        else if (dt == 5)
            dataType = DataType::UINT8;
        else if (dt == 6)
            dataType = DataType::UINT16;
        else if (dt == 7)
            dataType = DataType::UINT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else if (dt == 9)
            dataType = DataType::PACKED_BOOLEAN;

        uint32_t numDimensions = 1 + (rand() % 5);
        uint32_t maxDimension = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimension = pow(10000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t i = 0; i < numDimensions; ++i) {
            dimensions.push_back(1 + (rand() % maxDimension));
        }
        int16_t minValue;
        int16_t maxValue;
        if (dt < 5) {
            int r = rand() % 3;
            if (r == 0) {
                minValue = -100;
                maxValue = 100;
            } else if (r == 1) {
                minValue = 10;
                maxValue = 100;
            } else {
                minValue = -100;
                maxValue = -10;
            }
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
        Tensor tensor = Tensor::randoms(gpuPlacement, TensorDescriptor(dataType, dimensions), stream, minValue, maxValue);
        stream.synchronize();

        if (dt == 9) {
            Tensor tensor_h = tensor.clone(cpuPlacement);
            tensor_h.copyFromAsync(tensor, stream);
            stream.synchronize();
            uint8_t *mem = tensor_h.getMemPtr<uint8_t>();
            uint32_t numElements = (tensor.getTotalNumElements() + 7) / 8;
            for (uint32_t i = 0; i < numElements; ++i) {
                ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
            }
        } else {
            Tensor tensorFp32_h = tensor.clone(cpuPlacement, DataType::FP32);
            tensorFp32_h.copyFromAsync(tensor, stream);
            stream.synchronize();
            float *mem = tensorFp32_h.getMemPtr<float>();
            for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
                ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
            }
        }
    }
}

TEST(Tensor, valuesCpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::INT8;
        else if (dt == 3)
            dataType = DataType::INT16;
        else if (dt == 4)
            dataType = DataType::INT32;
        else if (dt == 5)
            dataType = DataType::UINT8;
        else if (dt == 6)
            dataType = DataType::UINT16;
        else if (dt == 7)
            dataType = DataType::UINT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else
            dataType = DataType::PACKED_BOOLEAN;

        uint32_t numDimensions = 1 + (rand() % 5);
        uint32_t maxDimension = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimension = pow(10000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t i = 0; i < numDimensions; ++i) {
            dimensions.push_back(1 + (rand() % maxDimension));
        }
        int16_t value;
        if (dt < 5) {
            value = (rand() % 200) - 100;
        } else if (dt < 8) {
            value = rand() % 200;
        } else if (dt >= 8) {
            value = rand() % 2;
        }
        Tensor tensor = Tensor::values(cpuPlacement, TensorDescriptor(dataType, dimensions), stream, value);
        stream.synchronize();

        if (dt == 9) {
            if (value)
                value = 0b11111111;
            uint8_t *mem = tensor.getMemPtr<uint8_t>();
            uint32_t numElements = (tensor.getTotalNumElements() + 7) / 8;
            for (uint32_t i = 0; i < numElements; ++i) {
                if (mem[i] != value)
                    printf("dt %d i %d actual %d vs expected %d\n", dt, i, (uint32_t)mem[i], value);
                ASSERT_TRUE(mem[i] == value);
            }
        } else {
            Tensor tensorFp32 = tensor.clone(DataType::FP32);
            tensorFp32.copyFromAsync(tensor, stream);
            stream.synchronize();
            float *mem = tensorFp32.getMemPtr<float>();
            for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
                if (mem[i] != value)
                    printf("dt %d i %d actual %f vs expected %d\n", dt, i, mem[i], value);
                ASSERT_TRUE(mem[i] == value);
            }
        }
    }
}

TEST(Tensor, valuesGpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::INT8;
        else if (dt == 3)
            dataType = DataType::INT16;
        else if (dt == 4)
            dataType = DataType::INT32;
        else if (dt == 5)
            dataType = DataType::UINT8;
        else if (dt == 6)
            dataType = DataType::UINT16;
        else if (dt == 7)
            dataType = DataType::UINT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else
            dataType = DataType::PACKED_BOOLEAN;

        uint32_t numDimensions = 1 + (rand() % 5);
        uint32_t maxDimension = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimension = pow(10000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t i = 0; i < numDimensions; ++i) {
            dimensions.push_back(1 + (rand() % maxDimension));
        }
        int16_t value;
        if (dt < 5) {
            value = (rand() % 200) - 100;
        } else if (dt < 8) {
            value = rand() % 200;
        } else if (dt >= 8) {
            value = rand() % 2;
        }
        Tensor tensor = Tensor::values(gpuPlacement, TensorDescriptor(dataType, dimensions), stream, value);
        stream.synchronize();

        if (dt == 9) {
            if (value)
                value = 0b11111111;
            Tensor tensor_h = tensor.clone(cpuPlacement);
            tensor_h.copyFromAsync(tensor, stream);
            stream.synchronize();
            uint8_t *mem = tensor_h.getMemPtr<uint8_t>();
            uint32_t numElements = (tensor.getTotalNumElements() + 7) / 8;
            for (uint32_t i = 0; i < numElements; ++i) {
                if (mem[i] != value)
                    printf("dt %d i %d actual %d vs expected %d\n", dt, i, (uint32_t)mem[i], value);
                ASSERT_TRUE(mem[i] == value);
            }
        } else {
            Tensor tensorFp32 = copyToCpuFp32ForVerification(tensor, stream);
            stream.synchronize();
            float *mem = tensorFp32.getMemPtr<float>();
            for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
                if (mem[i] != value)
                    printf("dt %d i %d actual %f vs expected %d\n", dt, i, mem[i], value);
                ASSERT_TRUE(mem[i] == value);
            }
        }
    }
}

TEST(Tensor, valuesSupportsAllDataTypesCpuAndGpu) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);
    vector<uint64_t> dimensions{37};

    for (DataType dataType : allTensorDataTypes()) {
        for (TensorPlacement placement : {cpuPlacement, gpuPlacement}) {
            Tensor tensor = Tensor::values(placement, TensorDescriptor(dataType, dimensions), stream, 1.0);
            Tensor tensorFp32 = copyToCpuFp32ForVerification(tensor, stream);
            stream.synchronize();

            float *mem = tensorFp32.getMemPtr<float>();
            for (uint64_t i = 0; i < tensor.getTotalNumElements(); ++i) {
                ASSERT_EQ(mem[i], 1.0f) << "dataType=" << TensorDescriptor::getElementTypeName(dataType) << " i=" << i;
            }
        }
    }
}

TEST(Tensor, fillRandomCpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::INT8;
        else if (dt == 3)
            dataType = DataType::INT16;
        else if (dt == 4)
            dataType = DataType::INT32;
        else if (dt == 5)
            dataType = DataType::UINT8;
        else if (dt == 6)
            dataType = DataType::UINT16;
        else if (dt == 7)
            dataType = DataType::UINT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else if (dt == 9)
            dataType = DataType::PACKED_BOOLEAN;

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
            int r = rand() % 3;
            if (r == 0) {
                minValue = -100;
                maxValue = 100;
            } else if (r == 1) {
                minValue = 10;
                maxValue = 100;
            } else {
                minValue = -100;
                maxValue = -10;
            }
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
        stream.synchronize();

        if (dt == 9) {
            uint8_t *mem = tensor.getMemPtr<uint8_t>();
            uint32_t numElements = (tensor.getTotalNumElements() + 7) / 8;
            for (uint32_t i = 0; i < numElements; ++i) {
                ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
            }
        } else {
            Tensor tensorFp32_h = tensor.clone(DataType::FP32);
            tensorFp32_h.copyFromAsync(tensor, stream);
            stream.synchronize();
            float *mem = tensorFp32_h.getMemPtr<float>();
            for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
                if (!(mem[i] <= maxValue && mem[i] >= minValue))
                    printf("[%d] %d <= %f <= %d failed.   dt = %d\n", i, minValue, mem[i], maxValue, dt);
                ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
            }
        }
    }
}

TEST(Tensor, fillRandomGpu) {
    srand(time(nullptr));
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 20; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::INT8;
        else if (dt == 3)
            dataType = DataType::INT16;
        else if (dt == 4)
            dataType = DataType::INT32;
        else if (dt == 5)
            dataType = DataType::UINT8;
        else if (dt == 6)
            dataType = DataType::UINT16;
        else if (dt == 7)
            dataType = DataType::UINT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else if (dt == 9)
            dataType = DataType::PACKED_BOOLEAN;

        uint32_t numDimensions = 1 + (rand() % 5);
        uint32_t maxDimension = pow(100000.0, 1.0 / numDimensions);
        if (rand() % 5 == 0)
            maxDimension = pow(10000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t i = 0; i < numDimensions; ++i) {
            dimensions.push_back(1 + (rand() % maxDimension));
        }
        Tensor tensor(gpuPlacement, TensorDescriptor(dataType, dimensions));
        int16_t minValue;
        int16_t maxValue;
        if (dt < 5) {
            int r = rand() % 3;
            if (r == 0) {
                minValue = -100;
                maxValue = 100;
            } else if (r == 1) {
                minValue = 10;
                maxValue = 100;
            } else {
                minValue = -100;
                maxValue = -10;
            }
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
        stream.synchronize();

        if (dt == 9) {
            Tensor tensor_h = tensor.clone(cpuPlacement);
            tensor_h.copyFromAsync(tensor, stream);
            stream.synchronize();
            uint8_t *mem = tensor_h.getMemPtr<uint8_t>();
            uint32_t numElements = (tensor.getTotalNumElements() + 7) / 8;
            for (uint32_t i = 0; i < numElements; ++i) {
                ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
            }
        } else {
            Tensor tensorFp32_h = tensor.clone(cpuPlacement, DataType::FP32);
            tensorFp32_h.copyFromAsync(tensor, stream);
            stream.synchronize();
            float *mem = tensorFp32_h.getMemPtr<float>();
            for (uint32_t i = 0; i < tensor.getTotalNumElements(); ++i) {
                if (!(mem[i] <= maxValue && mem[i] >= minValue))
                    printf("[%d] %d <= %f <= %d failed.   dt = %d\n", i, minValue, mem[i], maxValue, dt);
                ASSERT_TRUE(mem[i] <= maxValue && mem[i] >= minValue);
            }
        }
    }
}

TEST(Tensor, fillRandomSupportsAllDataTypesCpuAndGpu) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);
    vector<uint64_t> dimensions{257};

    for (DataType dataType : allTensorDataTypes()) {
        double minValue = -1.0;
        double maxValue = 1.0;
        if (dataType == DataType::UINT8 || dataType == DataType::UINT16 || dataType == DataType::UINT32 || dataType == DataType::UINT64 ||
            dataType == DataType::BOOLEAN || dataType == DataType::PACKED_BOOLEAN) {
            minValue = 0.0;
            maxValue = dataType == DataType::BOOLEAN || dataType == DataType::PACKED_BOOLEAN ? 1.0 : 5.0;
        }

        for (TensorPlacement placement : {cpuPlacement, gpuPlacement}) {
            Tensor tensor(placement, TensorDescriptor(dataType, dimensions));
            tensor.fillRandom(minValue, maxValue, stream);
            Tensor tensorFp32 = copyToCpuFp32ForVerification(tensor, stream);
            stream.synchronize();

            float *mem = tensorFp32.getMemPtr<float>();
            for (uint64_t i = 0; i < tensor.getTotalNumElements(); ++i) {
                ASSERT_GE(mem[i], minValue) << "dataType=" << TensorDescriptor::getElementTypeName(dataType) << " i=" << i;
                ASSERT_LE(mem[i], maxValue) << "dataType=" << TensorDescriptor::getElementTypeName(dataType) << " i=" << i;
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
        DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else
            dataType = DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor source_h(cpuPlacement, descriptor);
        Tensor source_float_h = source_h.clone(DataType::FP32);
        Tensor source_d = source_h.clone(gpuPlacement);
        Tensor dest_d = source_d.clone();
        Tensor dest_gpu_float_h = dest_d.clone(cpuPlacement, DataType::FP32);

        float *source_float_h_mem = (float *)source_float_h.getMemPtr();
        for (uint32_t i = 0; i < dimensions[0]; ++i) {
            for (uint32_t j = 0; j < dimensions[1]; ++j) {
                if (dataType == DataType::FP16 || dataType == DataType::FP32)
                    source_float_h_mem[i * dimensions[1] + j] = (rand() % 10) / ((rand() % 10) + 1.0);
                else
                    source_float_h_mem[i * dimensions[1] + j] = rand() % 10;
                if (rand() % 2 && dataType != DataType::UINT8 && dataType != DataType::UINT16 && dataType != DataType::UINT32)
                    source_float_h_mem[i * dimensions[1] + j] = -source_float_h_mem[i * dimensions[1] + j];
            }
        }
        float scalar;
        if (dataType == DataType::FP16 || dataType == DataType::FP32)
            scalar = ((rand() % 20) + 1) / ((rand() % 10) + 1.0);
        else
            scalar = rand() % 10 + 1;
        if (rand() % 2 && dataType != DataType::UINT8 && dataType != DataType::UINT16 && dataType != DataType::UINT32)
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
                if (dataType == DataType::FP16 || dataType == DataType::FP32)
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

        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else if (dt == 7)
            dataType = DataType::INT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else
            dataType = DataType::PACKED_BOOLEAN;

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

        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else if (dt == 7)
            dataType = DataType::INT32;
        else if (dt == 8)
            dataType = DataType::BOOLEAN;
        else
            dataType = DataType::PACKED_BOOLEAN;

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

        DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else
            dataType = DataType::INT32;

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

        DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else
            dataType = DataType::INT32;

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

        DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else
            dataType = DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        int8_t fillValue;
        fillValue = rand() % 20;
        uint64_t numElementToSet = 1 + rand() % totalNumElements;

        Tensor t_h(cpuPlacement, descriptor);
        // First set all elements to a value representing their prior state
        t_h.memset(12);
        t_h.memset(0);

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

        DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else
            dataType = DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        int8_t fillValue;
        fillValue = rand() % 20;
        uint64_t numElementToSet = 1 + rand() % totalNumElements;

        Tensor t_h(cpuPlacement, descriptor);
        // First set all elements to a value representing their prior state
        t_h.memset(12);
        t_h.memsetAsync(stream, 0);
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

        DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else
            dataType = DataType::INT32;

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

        DataType dataType;
        uint32_t dt = rand() % 8;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::INT8;
        else if (dt == 6)
            dataType = DataType::INT16;
        else
            dataType = DataType::INT32;

        TensorDescriptor descriptor(dataType, dimensions);

        Tensor t_h(cpuPlacement, descriptor);
        Tensor t_d = t_h.clone(gpuPlacement);
        // First set all elements to a value representing their prior state
        // Then set the desired number of elements, then need to check all elements
        t_d.memsetAsync(stream, 9);
        t_d.memsetAsync(stream, 0);
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

unsigned long long getFreeMemoryLinux() {
    std::string token;
    unsigned long long memFree = 0;

    std::ifstream file("/proc/meminfo");
    if (file.is_open()) {
        while (file >> token) {
            if (token == "MemFree:") {
                if (file >> memFree) {
                    break;
                }
            }
            // Skip the rest of the line
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        file.close();
    }
    return memFree;  // kB
}

TEST(Tensor, loadFromFileCpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::UINT64;
        else if (dt == 6)
            dataType = DataType::INT8;
        else if (dt == 7)
            dataType = DataType::INT16;
        else if (dt == 8)
            dataType = DataType::INT32;
        else
            dataType = DataType::INT64;

        uint32_t numDimensions = (rand() % 5) + 1;
        uint64_t maxStageSize;
        maxStageSize = pow(1000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t d = 0; d < numDimensions; ++d)
            dimensions.push_back(1 + (rand() % maxStageSize));

        TensorDescriptor descriptor(dataType, dimensions);
        Tensor tensor(cpuPlacement, descriptor);

        // Prepare a test file
        char testFileName[] = "/tmp/tensor_test_data.bin.XXXXXX";  // XXXXXX will be replaced with a unique sequence

        int fileDescriptor = mkstemp(testFileName);
        ASSERT_NE(fileDescriptor, -1);

        Tensor dataTensor;
        if (dataType == DataType::FP16)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -500, 500);
        else if (dataType == DataType::FP32)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -20000, 2000);
        else if (dataType == DataType::UINT8)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 255);
        else if (dataType == DataType::UINT16)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 65535);
        else if (dataType == DataType::UINT32)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 4294967295);
        else if (dataType == DataType::UINT64)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 254294967295);
        else if (dataType == DataType::INT8)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -128, 127);
        else if (dataType == DataType::INT16)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -32768, 32767);
        else if (dataType == DataType::INT32)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -2147483648, 2147483647);
        else  // dataType == DataType::INT64)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -252147483648, 252147483647);
        Tensor suffixDataTensor(cpuPlacement, TensorDescriptor(DataType::UINT8, {99}));
        uint8_t suffixValue = rand() % 256;
        suffixDataTensor.fill(suffixValue, stream);
        stream.synchronize();
        ssize_t bytes_written;
        bytes_written = write(fileDescriptor, dataTensor.getMemPtr(), dataTensor.getArraySizeInBytes());
        assert(bytes_written == static_cast<ssize_t>(dataTensor.getArraySizeInBytes()));
        bytes_written = write(fileDescriptor, suffixDataTensor.getMemPtr(), suffixDataTensor.getArraySizeInBytes());
        assert(bytes_written == static_cast<ssize_t>(suffixDataTensor.getArraySizeInBytes()));
        close(fileDescriptor);

        uint32_t fileOffset = 0;
        if (tensor.getTotalNumElements() > 200)
            fileOffset = rand() % 100;
        Tensor::FileAccess fileAccess = Tensor::FileAccess::READ_ONLY;
        if (rand() % 2)
            fileAccess = Tensor::FileAccess::READ_WRITE;
        // Attach the file to the tensor
        tensor.attachFile(testFileName, fileOffset, fileAccess);

        // Call loadFromFile, then clean up.
        try {
            tensor.loadFromFile(stream);
            stream.synchronize();
        } catch (runtime_error &err) {
            printf("runtime_error: %s\n", err.what());
            ASSERT_TRUE(false);
        }
        tensor.detachFile();
        remove(testFileName);

        // Verify the result. Checking that they are byte for byte the same.
        const uint8_t *tensorMem = (uint8_t *)tensor.getMemPtr();
        const uint8_t *dataTensorMem = (uint8_t *)dataTensor.getMemPtr();
        for (uint64_t i = 0; i < dataTensor.getArraySizeInBytes(); ++i) {
            if (i + fileOffset >= dataTensor.getArraySizeInBytes()) {
                EXPECT_EQ(suffixValue, tensorMem[i]);
            } else {
                ASSERT_EQ(dataTensorMem[i + fileOffset], tensorMem[i]);
            }
        }
    }
}

TEST(Tensor, loadFromFileGpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::UINT64;
        else if (dt == 6)
            dataType = DataType::INT8;
        else if (dt == 7)
            dataType = DataType::INT16;
        else if (dt == 8)
            dataType = DataType::INT32;
        else
            dataType = DataType::INT64;

        uint32_t numDimensions = (rand() % 5) + 1;
        uint64_t maxStageSize;
        maxStageSize = pow(1000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t d = 0; d < numDimensions; ++d)
            dimensions.push_back(1 + (rand() % maxStageSize));

        TensorDescriptor descriptor(dataType, dimensions);
        Tensor tensor(gpuPlacement, descriptor);
        Tensor tensor_h(cpuPlacement, descriptor);

        // Prepare a test file
        char testFileName[] = "/tmp/tensor_test_data.bin.XXXXXX";  // XXXXXX will be replaced with a unique sequence

        int fileDescriptor = mkstemp(testFileName);
        ASSERT_NE(fileDescriptor, -1);

        Tensor dataTensor;
        if (dataType == DataType::FP16)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -500, 500);
        else if (dataType == DataType::FP32)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -20000, 2000);
        else if (dataType == DataType::UINT8)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 255);
        else if (dataType == DataType::UINT16)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 65535);
        else if (dataType == DataType::UINT32)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 4294967295);
        else if (dataType == DataType::UINT64)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 254294967295);
        else if (dataType == DataType::INT8)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -128, 127);
        else if (dataType == DataType::INT16)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -32768, 32767);
        else if (dataType == DataType::INT32)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -2147483648, 2147483647);
        else  // dataType == DataType::INT64)
            dataTensor = Tensor::randoms(cpuPlacement, descriptor, stream, -252147483648, 252147483647);
        Tensor suffixDataTensor(cpuPlacement, TensorDescriptor(DataType::UINT8, {99}));
        uint8_t suffixValue = rand() % 256;
        suffixDataTensor.fill(suffixValue, stream);
        stream.synchronize();
        ssize_t bytes_written;
        bytes_written = write(fileDescriptor, dataTensor.getMemPtr(), dataTensor.getArraySizeInBytes());
        assert(bytes_written == static_cast<ssize_t>(dataTensor.getArraySizeInBytes()));
        bytes_written = write(fileDescriptor, suffixDataTensor.getMemPtr(), suffixDataTensor.getArraySizeInBytes());
        assert(bytes_written == static_cast<ssize_t>(suffixDataTensor.getArraySizeInBytes()));
        close(fileDescriptor);

        uint32_t fileOffset = 0;
        if (tensor.getTotalNumElements() > 200)
            fileOffset = rand() % 100;
        Tensor::FileAccess fileAccess = Tensor::FileAccess::READ_ONLY;
        if (rand() % 2)
            fileAccess = Tensor::FileAccess::READ_WRITE;
        // Attach the file to the tensor
        tensor.attachFile(testFileName, fileOffset, fileAccess);

        // Call loadFromFile, then clean up.
        try {
            tensor.loadFromFile(stream);
            tensor_h.copyFromAsync(tensor, stream);
            stream.synchronize();
        } catch (runtime_error &err) {
            printf("runtime_error: %s\n", err.what());
            ASSERT_TRUE(false);
        }
        tensor.detachFile();
        remove(testFileName);

        // Verify the result. Checking that they are byte for byte the same.
        const uint8_t *tensorMem = (uint8_t *)tensor_h.getMemPtr();
        const uint8_t *dataTensorMem = (uint8_t *)dataTensor.getMemPtr();
        for (uint64_t i = 0; i < dataTensor.getArraySizeInBytes(); ++i) {
            if (i + fileOffset >= dataTensor.getArraySizeInBytes()) {
                EXPECT_EQ(suffixValue, tensorMem[i]);
            } else {
                ASSERT_EQ(dataTensorMem[i + fileOffset], tensorMem[i]);
            }
        }
    }
}

void printFileContents(const std::string &filename) {
    // Open the file in binary mode to read bytes
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Failed to open file: %s\n", filename.c_str());
        return;
    }

    // Read and print each byte as an unsigned integer
    uint8_t byte;
    size_t index = 0;
    while (file.read(reinterpret_cast<char *>(&byte), sizeof(byte))) {
        printf("%zu: %u ", index++, static_cast<unsigned int>(byte));
    }
    printf("\n");

    file.close();  // Close the file after reading
}

void printTensorContents(Tensor tensor) {
    uint8_t *mem = (uint8_t *)tensor.getMemPtr();
    uint64_t numBytes = tensor.getArraySizeInBytes();
    for (uint64_t i = 0; i < numBytes; ++i)
        printf("%zu: %u ", i, mem[i]);
    printf("\n");
}

TEST(Tensor, writeToFileCpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::UINT64;
        else if (dt == 6)
            dataType = DataType::INT8;
        else if (dt == 7)
            dataType = DataType::INT16;
        else if (dt == 8)
            dataType = DataType::INT32;
        else
            dataType = DataType::INT64;

        uint32_t numDimensions = (rand() % 5) + 1;
        uint64_t maxStageSize;
        maxStageSize = pow(1000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t d = 0; d < numDimensions; ++d)
            dimensions.push_back(1 + (rand() % maxStageSize));

        TensorDescriptor descriptor(dataType, dimensions);

        // Prepare a test file
        char testFileName[] = "/tmp/tensor_test_data.bin.XXXXXX";  // XXXXXX will be replaced with a unique sequence

        int fileDescriptor = mkstemp(testFileName);
        ASSERT_NE(fileDescriptor, -1);

        Tensor tensor(cpuPlacement, descriptor);
        if (dataType == DataType::FP16)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, -500, 500);
        else if (dataType == DataType::FP32)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, -20000, 2000);
        else if (dataType == DataType::UINT8)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 255);
        else if (dataType == DataType::UINT16)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 65535);
        else if (dataType == DataType::UINT32)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 4294967295);
        else if (dataType == DataType::UINT64)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, 0, 254294967295);
        else if (dataType == DataType::INT8)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, -128, 127);
        else if (dataType == DataType::INT16)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, -32768, 32767);
        else if (dataType == DataType::INT32)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, -2147483648, 2147483647);
        else  // dataType == DataType::INT64)
            tensor = Tensor::randoms(cpuPlacement, descriptor, stream, -252147483648, 252147483647);

        uint32_t fileOffset = 0;
        fileOffset = rand() % 1000;
        Tensor::FileAccess fileAccess = Tensor::FileAccess::WRITE_ONLY;
        if (rand() % 2)
            fileAccess = Tensor::FileAccess::READ_WRITE;
        // Attach the file to the tensor
        tensor.attachFile(testFileName, fileOffset, fileAccess);

        // Call loadFromFile, then clean up.
        Tensor fileTensor = tensor.clone();
        fileTensor.attachFile(testFileName, fileOffset, rand() % 2 ? Tensor::FileAccess::READ_ONLY : Tensor::FileAccess::READ_WRITE);
        try {
            tensor.dumpToFile(stream);
            fileTensor.loadFromFile(stream);
            stream.synchronize();
        } catch (runtime_error &err) {
            printf("runtime_error: %s\n", err.what());
            ASSERT_TRUE(false);
        }
        tensor.detachFile();
        remove(testFileName);

        // Verify the result. Checking that they are byte for byte the same.
        const uint8_t *tensorMem = (uint8_t *)tensor.getMemPtr();
        const uint8_t *fileTensorMem = (uint8_t *)fileTensor.getMemPtr();
        for (uint64_t i = 0; i < tensor.getArraySizeInBytes(); ++i) {
            if (fileTensorMem[i] != tensorMem[i])
                printf("%ld\n", i);
            ASSERT_EQ(fileTensorMem[i], tensorMem[i]);
        }
    }
}

TEST(Tensor, writeToFileGpu) {
    srand(time(nullptr));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU);
    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        DataType dataType;
        uint32_t dt = rand() % 10;
        if (dt == 0)
            dataType = DataType::FP16;
        else if (dt == 1)
            dataType = DataType::FP32;
        else if (dt == 2)
            dataType = DataType::UINT8;
        else if (dt == 3)
            dataType = DataType::UINT16;
        else if (dt == 4)
            dataType = DataType::UINT32;
        else if (dt == 5)
            dataType = DataType::UINT64;
        else if (dt == 6)
            dataType = DataType::INT8;
        else if (dt == 7)
            dataType = DataType::INT16;
        else if (dt == 8)
            dataType = DataType::INT32;
        else
            dataType = DataType::INT64;

        uint32_t numDimensions = (rand() % 5) + 1;
        uint64_t maxStageSize;
        maxStageSize = pow(1000000.0, 1.0 / numDimensions);
        vector<uint64_t> dimensions;
        for (uint32_t d = 0; d < numDimensions; ++d)
            dimensions.push_back(1 + (rand() % maxStageSize));

        TensorDescriptor descriptor(dataType, dimensions);

        // Prepare a test file
        char testFileName[] = "/tmp/tensor_test_data.bin.XXXXXX";  // XXXXXX will be replaced with a unique sequence

        int fileDescriptor = mkstemp(testFileName);
        ASSERT_NE(fileDescriptor, -1);

        Tensor tensor(gpuPlacement, descriptor);
        if (dataType == DataType::FP16)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, -500, 500);
        else if (dataType == DataType::FP32)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, -20000, 2000);
        else if (dataType == DataType::UINT8)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, 0, 255);
        else if (dataType == DataType::UINT16)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, 0, 65535);
        else if (dataType == DataType::UINT32)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, 0, 4294967295);
        else if (dataType == DataType::UINT64)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, 0, 254294967295);
        else if (dataType == DataType::INT8)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, -128, 127);
        else if (dataType == DataType::INT16)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, -32768, 32767);
        else if (dataType == DataType::INT32)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, -2147483648, 2147483647);
        else  // dataType == DataType::INT64)
            tensor = Tensor::randoms(gpuPlacement, descriptor, stream, -252147483648, 252147483647);

        uint32_t fileOffset = 0;
        fileOffset = rand() % 1000;
        Tensor::FileAccess fileAccess = Tensor::FileAccess::WRITE_ONLY;
        if (rand() % 2)
            fileAccess = Tensor::FileAccess::READ_WRITE;
        // Attach the file to the tensor
        tensor.attachFile(testFileName, fileOffset, fileAccess);

        // Call loadFromFile, then clean up.
        Tensor fileTensor = tensor.clone(cpuPlacement);
        Tensor tensor_h = tensor.clone(cpuPlacement);
        tensor_h.copyFromAsync(tensor, stream);
        fileTensor.attachFile(testFileName, fileOffset, rand() % 2 ? Tensor::FileAccess::READ_ONLY : Tensor::FileAccess::READ_WRITE);
        try {
            tensor.dumpToFile(stream);
            fileTensor.loadFromFile(stream);
            stream.synchronize();
        } catch (runtime_error &err) {
            printf("runtime_error: %s\n", err.what());
            ASSERT_TRUE(false);
        }
        tensor.detachFile();
        remove(testFileName);

        // Verify the result. Checking that they are byte for byte the same.
        const uint8_t *tensorMem = (uint8_t *)tensor_h.getMemPtr();
        const uint8_t *fileTensorMem = (uint8_t *)fileTensor.getMemPtr();
        for (uint64_t i = 0; i < tensor.getArraySizeInBytes(); ++i) {
            if (fileTensorMem[i] != tensorMem[i])
                printf("%ld\n", i);
            ASSERT_EQ(fileTensorMem[i], tensorMem[i]);
        }
    }
}
