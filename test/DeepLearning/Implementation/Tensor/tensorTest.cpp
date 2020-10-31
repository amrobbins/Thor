#include "Thor.h"

#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

#pragma GCC diagnostic ignored "-Wsign-compare"
#include "gtest/gtest.h"
#pragma GCC diagnostic pop

using namespace ThorImplementation;

TEST(DistributedTensor, InstantiatesTensor) {
    vector<unsigned long> dimensions;
    dimensions.push_back(100);
    dimensions.push_back(200);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor tensor(descriptor);
}

TEST(DistributedTensor, InstantiatesTensorInstance) {
    vector<unsigned long> dimensions;
    dimensions.push_back(100);
    dimensions.push_back(200);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement placement(TensorPlacement::MemDevices::CPU);

    Tensor instance(placement, descriptor);
}

// Copy from Instance to Instance

TEST(DistributedTensor, InstanceToInstanceCopyCpuToCpu) {
    int dim0 = 16;
    int dim1 = 1000;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement placement(TensorPlacement::MemDevices::CPU);

    Tensor instance0(placement, descriptor);
    Tensor instance1(placement, descriptor);

    assert(instance0.getDescriptor() == instance1.getDescriptor());
    assert(instance0.getPlacement() == instance1.getPlacement());

    float *mem0 = (float *)instance0.getMemPtr();
    float *mem1 = (float *)instance1.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            mem0[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    instance1.copyFromAsync(instance0, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(mem1[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(mem0[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToInstanceCopyCpuToGpuToCpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - no gpus\n");
        return;
    }

    int dim0 = 16;
    int dim1 = 1000;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Tensor instance0(cpuPlacement, descriptor);
    Tensor instance1(gpuPlacement, descriptor);
    Tensor instance2(cpuPlacement, descriptor);

    assert(instance0.getDescriptor() == instance1.getDescriptor());
    assert(instance1.getDescriptor() == instance2.getDescriptor());

    float *mem0 = (float *)instance0.getMemPtr();
    float *mem2 = (float *)instance2.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            mem0[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    instance1.copyFromAsync(instance0, stream);
    instance2.copyFromAsync(instance1, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(mem2[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(mem0[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToInstanceCopyGpuToSameGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - no gpus\n");
        return;
    }

    int dim0 = 16;
    int dim1 = 1000;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    Tensor instance0(cpuPlacement, descriptor);
    Tensor instance1(gpuPlacement, descriptor);
    Tensor instance2(gpuPlacement, descriptor);
    Tensor instance3(cpuPlacement, descriptor);

    assert(instance0.getDescriptor() == instance1.getDescriptor());
    assert(instance1.getDescriptor() == instance2.getDescriptor());
    assert(instance2.getDescriptor() == instance3.getDescriptor());

    float *mem0 = (float *)instance0.getMemPtr();
    float *mem3 = (float *)instance3.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            mem0[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    instance1.copyFromAsync(instance0, stream);
    instance2.copyFromAsync(instance1, stream);
    instance3.copyFromAsync(instance2, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(mem3[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(mem0[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToInstanceCopyGpuToPeerGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 16;
    int dim1 = 1000;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    Tensor instance0(cpuPlacement, descriptor);
    Tensor instance1(gpu0Placement, descriptor);
    Tensor instance2(gpu1Placement, descriptor);
    Tensor instance3(cpuPlacement, descriptor);

    assert(instance0.getDescriptor() == instance1.getDescriptor());
    assert(instance1.getDescriptor() == instance2.getDescriptor());
    assert(instance2.getDescriptor() == instance3.getDescriptor());

    float *mem0 = (float *)instance0.getMemPtr();
    float *mem3 = (float *)instance3.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            mem0[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    instance1.copyFromAsync(instance0, stream);
    instance2.copyFromAsync(instance1, stream);
    instance3.copyFromAsync(instance2, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(mem3[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(mem0[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToInstanceCopyWithTypeConversion) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 4;
    int dim1 = 300;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptorFp32(TensorDescriptor::DataType::FP32, dimensions);
    TensorDescriptor descriptorFp16(TensorDescriptor::DataType::FP16, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    Stream stream0(0);
    Stream stream1(1);

    vector<Tensor> instances;
    vector<Stream> destStreams;
    instances.emplace_back(cpuPlacement, descriptorFp16);
    destStreams.push_back(stream0);
    instances.emplace_back(gpu0Placement, descriptorFp16);
    destStreams.push_back(stream0);
    // upconvert gpu -> gpu
    instances.emplace_back(gpu1Placement, descriptorFp32);
    destStreams.push_back(stream1);
    instances.emplace_back(cpuPlacement, descriptorFp32);
    destStreams.push_back(stream1);
    instances.emplace_back(cpuPlacement, descriptorFp16);
    destStreams.push_back(stream0);
    // upconvert cpu -> gpu
    instances.emplace_back(gpu1Placement, descriptorFp32);
    destStreams.push_back(stream1);
    instances.emplace_back(gpu1Placement, descriptorFp16);
    destStreams.push_back(stream1);
    // upconvert gpu -> cpu
    instances.emplace_back(cpuPlacement, descriptorFp32);
    destStreams.push_back(stream1);

    half *memIn = (half *)instances.front().getMemPtr();
    float *memOut = (float *)instances.back().getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            memIn[i * dim1 + j] = (half)(float)(i * dim1 + j);
        }
    }

    for (unsigned int i = 1; i < instances.size(); ++i) {
        destStreams[i].waitEvent(destStreams[i - 1].putEvent());
        instances[i].copyFromAsync(instances[i - 1], destStreams[i]);
    }

    destStreams[instances.size() - 1].synchronize();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(memOut[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ((float)(memIn[i * dim1 + j]), i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToInstanceMoveWithTypeConversion) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 4;
    int dim1 = 300;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptorFp32(TensorDescriptor::DataType::FP32, dimensions);
    TensorDescriptor descriptorFp16(TensorDescriptor::DataType::FP16, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    Stream stream0(0);
    Stream stream1(1);

    vector<Tensor> instances;
    vector<Stream> destStreams;

    instances.emplace_back(cpuPlacement, descriptorFp16);
    destStreams.push_back(stream0);
    instances.emplace_back(gpu0Placement, descriptorFp16);
    destStreams.push_back(stream0);
    // upconvert gpu -> gpu
    instances.emplace_back(gpu1Placement, descriptorFp32);
    destStreams.push_back(stream1);
    // downconvert gpu -> gpu
    instances.emplace_back(gpu0Placement, descriptorFp16);
    destStreams.push_back(stream0);
    // upconvert gpu -> cpu
    instances.emplace_back(cpuPlacement, descriptorFp32);
    destStreams.push_back(stream0);
    // downconvert cpu -> gpu
    instances.emplace_back(gpu1Placement, descriptorFp16);
    destStreams.push_back(stream1);
    instances.emplace_back(gpu0Placement, descriptorFp32);
    destStreams.push_back(stream0);
    // downconvert gpu -> cpu
    instances.emplace_back(cpuPlacement, descriptorFp16);
    destStreams.push_back(stream0);
    // upconvert cpu -> gpu
    instances.emplace_back(gpu0Placement, descriptorFp32);
    destStreams.push_back(stream0);
    instances.emplace_back(cpuPlacement, descriptorFp32);
    destStreams.push_back(stream0);

    half *memIn = (half *)instances.front().getMemPtr();
    float *memOut = (float *)instances.back().getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            memIn[i * dim1 + j] = (half)(float)(i * dim1 + j);
        }
    }

    for (unsigned int i = 1; i < instances.size(); ++i) {
        destStreams[i].waitEvent(destStreams[i - 1].putEvent());
        instances[i].moveFromAsync(instances[i - 1], destStreams[i]);
    }

    destStreams[instances.size() - 1].synchronize();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ((float)memOut[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ((float)(memIn[i * dim1 + j]), i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToInstanceManyRandomMoves) {
    srand(time(NULL));

    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 4;
    int dim1 = 300;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptorFp32(TensorDescriptor::DataType::FP32, dimensions);
    TensorDescriptor descriptorFp16(TensorDescriptor::DataType::FP16, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    vector<Tensor> instances;

    vector<TensorPlacement> placements;
    placements.push_back(cpuPlacement);
    placements.push_back(gpu0Placement);
    placements.push_back(gpu1Placement);

    instances.emplace_back(cpuPlacement, descriptorFp32);
    for (int i = 0; i < 1000; ++i)
        instances.emplace_back(placements[rand() % 3], descriptorFp32);
    instances.emplace_back(cpuPlacement, descriptorFp32);

    float *memIn = (float *)instances.front().getMemPtr();
    float *memOut = (float *)instances.back().getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            memIn[i * dim1 + j] = (float)(i * dim1 + j);
        }
    }

    Stream stream(0);

    for (unsigned int i = 1; i < instances.size(); ++i)
        instances[i].moveFromAsync(instances[i - 1], stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ((float)memOut[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ((float)(memIn[i * dim1 + j]), i * dim1 + j);
        }
    }
}

//-----------------------------------------------------------
//
// Copy from Instance to DistributedTensor
//
//-----------------------------------------------------------
TEST(DistributedTensor, InstanceToSingleInstanceTensorCopyCpuToCpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - no gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);

    Tensor sourceInstance(cpuPlacement, descriptor);
    Tensor destCpuInstance(cpuPlacement, descriptor);

    destTensor.addInstance(cpuPlacement);

    float *sourceCpuMem = (float *)sourceInstance.getMemPtr();
    float *destCpuMem = (float *)destCpuInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceCpuMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    destTensor.copyFromAsync(sourceInstance, stream);
    destCpuInstance.copyFromAsync(destTensor.getInstance(cpuPlacement), stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceCpuMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToSingleInstanceTensorCopyCpuToGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - no gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);

    Tensor sourceInstance(cpuPlacement, descriptor);
    Tensor destCpuInstance(cpuPlacement, descriptor);

    destTensor.addInstance(gpu0Placement);

    float *sourceCpuMem = (float *)sourceInstance.getMemPtr();
    float *destCpuMem = (float *)destCpuInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceCpuMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    destTensor.copyFromAsync(sourceInstance, stream);
    destCpuInstance.copyFromAsync(destTensor.getInstance(gpu0Placement), stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceCpuMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToSingleInstanceTensorCopyGpuToSameGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - no gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);

    Tensor sourceCpuInstance(cpuPlacement, descriptor);
    Tensor sourceInstance(gpu0Placement, descriptor);
    Tensor destCpuInstance(cpuPlacement, descriptor);

    destTensor.addInstance(gpu0Placement);

    float *sourceCpuMem = (float *)sourceCpuInstance.getMemPtr();
    float *destCpuMem = (float *)destCpuInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceCpuMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    sourceInstance.copyFromAsync(sourceCpuInstance, stream);
    destTensor.copyFromAsync(sourceInstance, stream);
    destCpuInstance.copyFromAsync(destTensor.getInstance(gpu0Placement), stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceCpuMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToSingleInstanceTensorCopyGpuToPeerGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    Tensor sourceCpuInstance(cpuPlacement, descriptor);
    Tensor destCpuInstance(cpuPlacement, descriptor);

    Tensor sourceInstance(gpu0Placement, descriptor);
    destTensor.addInstance(gpu1Placement);

    float *sourceCpuMem = (float *)sourceCpuInstance.getMemPtr();
    float *destCpuMem = (float *)destCpuInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceCpuMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    sourceInstance.copyFromAsync(sourceCpuInstance, stream);
    destTensor.copyFromAsync(sourceInstance, stream);
    destCpuInstance.copyFromAsync(destTensor.getInstance(gpu1Placement), stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceCpuMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToMultiInstanceTensorCopyCpuSource) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    Tensor sourceInstance(cpuPlacement, descriptor);
    vector<Tensor> destCpuInstances;

    destTensor.addInstance(cpuPlacement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu1Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(cpuPlacement);

    float *sourceMem = (float *)sourceInstance.getMemPtr();
    vector<float *> destCpuMems;

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    destTensor.copyFromAsync(sourceInstance, stream);

    unordered_map<unsigned long, Tensor> destTensorInstances = destTensor.getInstances();
    for (auto entry : destTensorInstances) {
        destCpuInstances.emplace_back(cpuPlacement, descriptor);
        destCpuInstances.back().copyFromAsync(entry.second, stream);
        destCpuMems.push_back((float *)destCpuInstances.back().getMemPtr());
    }

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (float *destCpuMem : destCpuMems) {
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
            }
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToMultiInstanceTensorCopyGpu0Source) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    Tensor sourceCpuInstance(cpuPlacement, descriptor);

    Tensor sourceInstance(gpu0Placement, descriptor);
    vector<Tensor> destCpuInstances;

    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu1Placement);
    destTensor.addInstance(gpu1Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(cpuPlacement);
    destTensor.addInstance(cpuPlacement);

    float *sourceMem = (float *)sourceCpuInstance.getMemPtr();
    vector<float *> destCpuMems;

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceInstance.copyFromAsync(sourceCpuInstance, stream);
    destTensor.copyFromAsync(sourceInstance, stream);

    unordered_map<unsigned long, Tensor> destTensorInstances = destTensor.getInstances();
    for (auto entry : destTensorInstances) {
        destCpuInstances.emplace_back(cpuPlacement, descriptor);
        destCpuInstances.back().copyFromAsync(entry.second, stream);
        destCpuMems.push_back((float *)destCpuInstances.back().getMemPtr());
    }

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (float *destCpuMem : destCpuMems) {
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
            }
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, InstanceToMultiInstanceTensorCopyGpu1Source) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    Tensor sourceCpuInstance(cpuPlacement, descriptor);

    Tensor sourceInstance(gpu1Placement, descriptor);
    vector<Tensor> destCpuInstances;

    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu1Placement);
    destTensor.addInstance(gpu1Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(cpuPlacement);
    destTensor.addInstance(cpuPlacement);

    float *sourceMem = (float *)sourceCpuInstance.getMemPtr();
    vector<float *> destCpuMems;

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceInstance.copyFromAsync(sourceCpuInstance, stream);
    destTensor.copyFromAsync(sourceInstance, stream);

    unordered_map<unsigned long, Tensor> destTensorInstances = destTensor.getInstances();
    for (auto entry : destTensorInstances) {
        destCpuInstances.emplace_back(cpuPlacement, descriptor);
        destCpuInstances.back().copyFromAsync(entry.second, stream);
        destCpuMems.push_back((float *)destCpuInstances.back().getMemPtr());
    }

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (float *destCpuMem : destCpuMems) {
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
            }
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

// Copy from DistributedTensor to Instance

TEST(DistributedTensor, SingleInstanceTensorOnCpuToInstanceOnCpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 80;
    int dim1 = 3612;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);

    sourceTensor.addInstance(cpuPlacement);
    Tensor destInstance(cpuPlacement, descriptor);

    float *sourceMem = (float *)sourceTensor.getInstance(cpuPlacement).getMemPtr();
    float *destMem = (float *)destInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    destInstance.copyFromAsync(sourceTensor, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, SingleInstanceTensorOnGpuToInstanceOnCpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 80;
    int dim1 = 3612;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);

    sourceTensor.addInstance(gpu0Placement);
    Tensor destInstance(cpuPlacement, descriptor);

    DistributedTensor sourceCpuTensor(descriptor);
    Tensor sourceCpuInstance(cpuPlacement, descriptor);

    float *sourceMem = (float *)sourceCpuInstance.getMemPtr();
    float *destMem = (float *)destInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceTensor.copyFromAsync(sourceCpuInstance, stream);
    destInstance.copyFromAsync(sourceTensor, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, SingleInstanceTensorOnCpuToInstanceOnGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 80;
    int dim1 = 3612;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);

    sourceTensor.addInstance(cpuPlacement);
    Tensor destInstance(gpu0Placement, descriptor);

    DistributedTensor destCpuTensor(descriptor);
    Tensor destCpuInstance(cpuPlacement, descriptor);

    float *sourceMem = (float *)sourceTensor.getInstance(cpuPlacement).getMemPtr();
    float *destMem = (float *)destCpuInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    destInstance.copyFromAsync(sourceTensor, stream);
    destCpuInstance.copyFromAsync(destInstance, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, SingleInstanceTensorOnGpuToInstanceOnSameGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 80;
    int dim1 = 3612;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);

    sourceTensor.addInstance(gpu0Placement);
    Tensor destInstance(gpu0Placement, descriptor);

    DistributedTensor sourceCpuTensor(descriptor);
    Tensor sourceCpuInstance(cpuPlacement, descriptor);
    DistributedTensor destCpuTensor(descriptor);
    Tensor destCpuInstance(cpuPlacement, descriptor);

    float *sourceMem = (float *)sourceCpuInstance.getMemPtr();
    float *destMem = (float *)destCpuInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceTensor.copyFromAsync(sourceCpuInstance, stream);
    destInstance.copyFromAsync(sourceTensor, stream);
    destCpuInstance.copyFromAsync(destInstance, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, SingleInstanceTensorOnGpuToInstanceOnPeerGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 80;
    int dim1 = 3612;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);

    sourceTensor.addInstance(gpu0Placement);
    Tensor destInstance(gpu1Placement, descriptor);

    DistributedTensor sourceCpuTensor(descriptor);
    Tensor sourceCpuInstance(cpuPlacement, descriptor);
    DistributedTensor destCpuTensor(descriptor);
    Tensor destCpuInstance(cpuPlacement, descriptor);

    float *sourceMem = (float *)sourceCpuInstance.getMemPtr();
    float *destMem = (float *)destCpuInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceTensor.copyFromAsync(sourceCpuInstance, stream);
    destInstance.copyFromAsync(sourceTensor, stream);
    destCpuInstance.copyFromAsync(destInstance, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, MultiInstanceTensorToInstanceOnCpu) {
    if (MachineEvaluator::instance().getNumGpus() < 1) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 80;
    int dim1 = 101;
    int dim2 = 361;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    dimensions.push_back(dim2);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);

    sourceTensor.addInstance(cpuPlacement);
    sourceTensor.addInstance(gpu0Placement);
    sourceTensor.addInstance(gpu1Placement);
    sourceTensor.addInstance(gpu1Placement);
    sourceTensor.addInstance(gpu0Placement);
    sourceTensor.addInstance(cpuPlacement);
    Tensor destInstance(cpuPlacement, descriptor);

    DistributedTensor sourceCpuTensor(descriptor);
    Tensor sourceCpuInstance(cpuPlacement, descriptor);
    DistributedTensor destCpuTensor(descriptor);
    vector<Tensor> destCpuInstances;

    float *sourceMem = (float *)sourceCpuInstance.getMemPtr();
    float *destMem = (float *)destInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                sourceMem[i * (dim1 + dim2) + j * dim2 + k] = i * (dim1 + dim2) + j * dim2 + k;
            }
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceTensor.copyFromAsync(sourceCpuInstance, stream);
    destInstance.copyFromAsync(sourceTensor, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                ASSERT_EQ(destMem[i * (dim1 + dim2) + j * dim2 + k], i * (dim1 + dim2) + j * dim2 + k);
            }
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                ASSERT_EQ(sourceMem[i * (dim1 + dim2) + j * dim2 + k], i * (dim1 + dim2) + j * dim2 + k);
            }
        }
    }
}

TEST(DistributedTensor, MultiInstanceTensorToInstanceOnGpu) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 80;
    int dim1 = 3612;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);

    sourceTensor.addInstance(cpuPlacement);
    sourceTensor.addInstance(gpu0Placement);
    sourceTensor.addInstance(gpu1Placement);
    sourceTensor.addInstance(gpu1Placement);
    sourceTensor.addInstance(gpu0Placement);
    sourceTensor.addInstance(cpuPlacement);
    Tensor destInstance(gpu0Placement, descriptor);
    Tensor destCpuInstance(cpuPlacement, descriptor);

    DistributedTensor sourceCpuTensor(descriptor);
    Tensor sourceCpuInstance(cpuPlacement, descriptor);
    DistributedTensor destCpuTensor(descriptor);
    vector<Tensor> destCpuInstances;

    float *sourceMem = (float *)sourceCpuInstance.getMemPtr();
    float *destMem = (float *)destCpuInstance.getMemPtr();

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceTensor.copyFromAsync(sourceCpuInstance, stream);
    destInstance.copyFromAsync(sourceTensor, stream);
    destCpuInstance.copyFromAsync(destInstance, stream);

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(destMem[i * dim1 + j], i * dim1 + j);
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

//     2. tensor to tensor
TEST(DistributedTensor, SingleInstanceTensorOnCpuToMultiInstanceTensor) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    vector<Tensor> destCpuInstances;

    sourceTensor.addInstance(cpuPlacement);
    destTensor.addInstance(cpuPlacement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu1Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(cpuPlacement);

    float *sourceMem = (float *)sourceTensor.getInstance(cpuPlacement).getMemPtr();
    vector<float *> destCpuMems;

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    destTensor.copyFromAsync(sourceTensor, stream);

    unordered_map<unsigned long, Tensor> destTensorInstances = destTensor.getInstances();
    for (auto entry : destTensorInstances) {
        destCpuInstances.emplace_back(cpuPlacement, descriptor);
        destCpuInstances.back().copyFromAsync(entry.second, stream);
        destCpuMems.push_back((float *)destCpuInstances.back().getMemPtr());
    }

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (float *destCpuMem : destCpuMems) {
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
            }
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, SingleInstanceTensorOnGpuToMultiInstanceTensor) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    vector<Tensor> destCpuInstances;

    sourceTensor.addInstance(gpu1Placement);
    destTensor.addInstance(cpuPlacement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu1Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(cpuPlacement);

    DistributedTensor sourceCpuTensor(descriptor);
    Tensor sourceCpuInstance(cpuPlacement, descriptor);

    float *sourceMem = (float *)sourceCpuInstance.getMemPtr();
    vector<float *> destCpuMems;

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceTensor.copyFromAsync(sourceCpuInstance, stream);
    destTensor.copyFromAsync(sourceTensor, stream);

    unordered_map<unsigned long, Tensor> destTensorInstances = destTensor.getInstances();
    for (auto entry : destTensorInstances) {
        destCpuInstances.emplace_back(cpuPlacement, descriptor);
        destCpuInstances.back().copyFromAsync(entry.second, stream);
        destCpuMems.push_back((float *)destCpuInstances.back().getMemPtr());
    }

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (float *destCpuMem : destCpuMems) {
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
            }
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

TEST(DistributedTensor, MultiInstanceTensorToMultiInstanceTensor) {
    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 25;
    int dim1 = 2187;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor sourceTensor(descriptor);
    DistributedTensor destTensor(descriptor);
    DistributedTensor destCpuTensor(descriptor);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpu0Placement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement gpu1Placement(TensorPlacement::MemDevices::GPU, 1);

    vector<Tensor> destCpuInstances;

    sourceTensor.addInstance(cpuPlacement);
    sourceTensor.addInstance(cpuPlacement);
    sourceTensor.addInstance(gpu1Placement);
    sourceTensor.addInstance(gpu1Placement);
    destTensor.addInstance(cpuPlacement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu1Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(gpu0Placement);
    destTensor.addInstance(cpuPlacement);

    float *sourceMem = (float *)sourceTensor.getInstance(cpuPlacement).getMemPtr();
    vector<float *> destCpuMems;

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            sourceMem[i * dim1 + j] = i * dim1 + j;
        }
    }

    Stream stream(0);

    // Perform the copy
    sourceTensor.copyFromAsync(sourceTensor.getInstance(cpuPlacement), stream);
    destTensor.copyFromAsync(sourceTensor, stream);

    unordered_map<unsigned long, Tensor> destTensorInstances = destTensor.getInstances();
    for (auto entry : destTensorInstances) {
        destCpuInstances.emplace_back(cpuPlacement, descriptor);
        destCpuInstances.back().copyFromAsync(entry.second, stream);
        destCpuMems.push_back((float *)destCpuInstances.back().getMemPtr());
    }

    // Here I am checking the copy from instance of source tensor to the rest of the source tensor
    unordered_map<unsigned long, Tensor> sourceTensorInstances = sourceTensor.getInstances();
    for (auto entry : sourceTensorInstances) {
        destCpuInstances.emplace_back(cpuPlacement, descriptor);
        destCpuInstances.back().copyFromAsync(entry.second, stream);
        destCpuMems.push_back((float *)destCpuInstances.back().getMemPtr());
    }

    cudaError_t cudaStatus = cudaStreamSynchronize(stream.getStream());
    ASSERT_EQ(cudaStatus, cudaSuccess);

    for (float *destCpuMem : destCpuMems) {
        for (int i = 0; i < dim0; ++i) {
            for (int j = 0; j < dim1; ++j) {
                ASSERT_EQ(destCpuMem[i * dim1 + j], i * dim1 + j);
            }
        }
    }

    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            ASSERT_EQ(sourceMem[i * dim1 + j], i * dim1 + j);
        }
    }
}

void addRemoveInstances(DistributedTensor distributedTensor) {
    vector<TensorPlacement> placements;
    placements.emplace_back(TensorPlacement::MemDevices::CPU);
    placements.emplace_back(TensorPlacement::MemDevices::GPU, 0);
    placements.emplace_back(TensorPlacement::MemDevices::GPU, 1);

    // per placement
    vector<int> addedCount;
    addedCount.push_back(0);
    addedCount.push_back(0);
    addedCount.push_back(0);

    distributedTensor.addInstance(placements[rand() % 3]);

    int numAdded = 0;
    for (int i = 0; i < 10000; ++i) {
        if (numAdded > 0 && (rand() % 2 == 1)) {
            // Remove an instance
            int placementIndex = rand() % 3;
            while (addedCount[placementIndex] == 0) {
                placementIndex = rand() % 3;
            }
            distributedTensor.removeInstance(placements[placementIndex]);
            numAdded -= 1;
            addedCount[placementIndex] -= 1;
        } else {
            // Add an instance
            int placementIndex = rand() % 3;
            distributedTensor.addInstance(placements[placementIndex]);
            numAdded += 1;
            addedCount[placementIndex] += 1;
        }
    }

    for (unsigned int i = 0; i < addedCount.size(); ++i) {
        for (int j = 0; j < addedCount[i]; ++j) {
            distributedTensor.removeInstance(placements[i]);
        }
    }

    distributedTensor.addInstance(placements[rand() % 3]);
}

// multithreaded add/remove instances
TEST(DistributedTensor, MultiThreadedAddRemoveInstancesWorks) {
    srand(time(NULL));

    if (MachineEvaluator::instance().getNumGpus() < 2) {
        printf("skipped - not enough gpus\n");
        return;
    }

    int dim0 = 5;
    int dim1 = 10;

    vector<unsigned long> dimensions;
    dimensions.push_back(dim0);
    dimensions.push_back(dim1);
    TensorDescriptor descriptor(TensorDescriptor::DataType::FP32, dimensions);

    DistributedTensor distributedTensor(descriptor);

    deque<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back(addRemoveInstances, distributedTensor);
    }

    while (!threads.empty()) {
        threads.front().join();
        threads.pop_front();
    }

    // Exactly 20 remain after 10 threads randomly added and removed instances to the tensor all at once
    ASSERT_EQ(distributedTensor.getInstances().size(), 20);

    // And the reference count of the tensor and of each instance is 1
    ASSERT_EQ(distributedTensor.getReferenceCount(), 1);
    while (distributedTensor.getNumInstances() > 0) {
        Tensor instance = distributedTensor.getAnyInstance();
        distributedTensor.removeInstance(instance.getTensorId());
        ASSERT_EQ(instance.getReferenceCount(), 1);
    }
}

// Tensor operations

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
