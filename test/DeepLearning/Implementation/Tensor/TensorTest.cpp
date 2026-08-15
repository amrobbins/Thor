#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <atomic>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

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

static vector<DataType> allTensorDataTypes() { return allWholeElementTensorDataTypes(); }

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


TEST(Tensor, MemsetHonorsAliasViewStorageOffsetOnCpuAndGpu) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    Stream stream(0);

    Tensor host(cpuPlacement, TensorDescriptor(DataType::FP32, {6, 2}));
    float* hostValues = host.getMemPtr<float>();
    for (uint64_t i = 0; i < 12; ++i)
        hostValues[i] = static_cast<float>(i + 1);

    Tensor hostTail = host.aliasView({4}, {1}, 8);
    hostTail.memsetAsync(stream, 0);
    stream.synchronize();
    for (uint64_t i = 0; i < 8; ++i)
        EXPECT_EQ(hostValues[i], static_cast<float>(i + 1));
    for (uint64_t i = 8; i < 12; ++i)
        EXPECT_EQ(hostValues[i], 0.0f);

    for (uint64_t i = 0; i < 12; ++i)
        hostValues[i] = static_cast<float>(i + 1);
    Tensor device(gpuPlacement, TensorDescriptor(DataType::FP32, {6, 2}));
    device.copyFromAsync(host, stream);
    Tensor deviceTail = device.aliasView({4}, {1}, 8);
    deviceTail.memsetAsync(stream, 0);
    host.copyFromAsync(device, stream);
    stream.synchronize();

    hostValues = host.getMemPtr<float>();
    for (uint64_t i = 0; i < 8; ++i)
        EXPECT_EQ(hostValues[i], static_cast<float>(i + 1));
    for (uint64_t i = 8; i < 12; ++i)
        EXPECT_EQ(hostValues[i], 0.0f);
}

TEST(TensorSharedOwnership, CopiesAndAliasesKeepAllocationAliveAfterOriginalReset) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorDescriptor descriptor(DataType::FP32, {8});

    Tensor original(cpuPlacement, descriptor);
    float *originalMemory = original.getMemPtr<float>();
    for (uint32_t i = 0; i < 8; ++i)
        originalMemory[i] = 10.0f + static_cast<float>(i);

    const uint64_t tensorId = original.getTensorId();
    Tensor copy = original;
    Tensor alias = original.aliasView({4}, {1}, 2);

    original.dropReference();

    ASSERT_FALSE(original.isInitialized());
    ASSERT_EQ(original.getTensorId(), 0U);
    ASSERT_TRUE(copy.isInitialized());
    ASSERT_TRUE(alias.isInitialized());
    ASSERT_EQ(copy.getTensorId(), tensorId);
    ASSERT_EQ(alias.getTensorId(), tensorId);
    ASSERT_EQ(copy.getMemPtr<float>(), originalMemory);
    ASSERT_EQ(alias.getMemPtr<float>(), originalMemory + 2);

    for (uint32_t i = 0; i < 4; ++i)
        ASSERT_EQ(alias.getMemPtr<float>()[i], 12.0f + static_cast<float>(i));
}

TEST(TensorSharedOwnership, MoveTransfersHandleAndMovedFromTensorIsUninitialized) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorDescriptor descriptor(DataType::FP32, {4});

    Tensor original(cpuPlacement, descriptor);
    const uint64_t tensorId = original.getTensorId();
    float *memory = original.getMemPtr<float>();

    Tensor moved = std::move(original);

    ASSERT_FALSE(original.isInitialized());
    ASSERT_EQ(original.getTensorId(), 0U);
    ASSERT_TRUE(moved.isInitialized());
    ASSERT_EQ(moved.getTensorId(), tensorId);
    ASSERT_EQ(moved.getMemPtr<float>(), memory);
}

TEST(TensorSharedOwnership, DistinctHandlesCanCopyMoveAndDestroyConcurrently) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorDescriptor descriptor(DataType::FP32, {16});
    Tensor stableSource(cpuPlacement, descriptor);

    const uint64_t tensorId = stableSource.getTensorId();
    float *memory = stableSource.getMemPtr<float>();
    std::atomic<bool> failed{false};

    constexpr uint32_t numThreads = 8;
    constexpr uint32_t iterationsPerThread = 10000;
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (uint32_t threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
        threads.emplace_back([&]() {
            for (uint32_t iteration = 0; iteration < iterationsPerThread; ++iteration) {
                Tensor copy = stableSource;
                Tensor moved = std::move(copy);
                Tensor assigned;
                assigned = moved;

                if (copy.isInitialized() || copy.getTensorId() != 0U || !moved.isInitialized() || !assigned.isInitialized() ||
                    moved.getTensorId() != tensorId || assigned.getTensorId() != tensorId || moved.getMemPtr<float>() != memory ||
                    assigned.getMemPtr<float>() != memory) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }

    for (std::thread &thread : threads)
        thread.join();

    ASSERT_FALSE(failed.load(std::memory_order_relaxed));
    ASSERT_TRUE(stableSource.isInitialized());
    ASSERT_EQ(stableSource.getTensorId(), tensorId);
    ASSERT_EQ(stableSource.getMemPtr<float>(), memory);
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

TEST(Tensor, identityMatrixSupportsAllDataTypesCpuAndGpu) {
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
        uint32_t dt = rand() % 9;
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
        uint32_t dt = rand() % 9;
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
        uint32_t dt = rand() % 9;
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

        {
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
        uint32_t dt = rand() % 9;
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

        {
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
        uint32_t dt = rand() % 9;
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

        {
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
        uint32_t dt = rand() % 9;
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

        {
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
        uint32_t dt = rand() % 9;
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

        {
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
        uint32_t dt = rand() % 9;
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

        {
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
            dataType == DataType::BOOLEAN) {
            minValue = 0.0;
            maxValue = dataType == DataType::BOOLEAN ? 1.0 : 5.0;
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
        uint32_t dt = rand() % 9;
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
        uint32_t dt = rand() % 9;
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
