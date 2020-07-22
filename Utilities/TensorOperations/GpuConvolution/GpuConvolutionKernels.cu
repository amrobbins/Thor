#include "Utilities/TensorOperations/GpuConvolution/GpuConvolution.h"

#include <cuda.h>
#include <cuda_fp16.h>

__global__ void addConvolutionBiasKernel(
    half *data, half *biases, unsigned int batchDimension, unsigned int channelDimension, unsigned int elementDimension) {
    unsigned int elementIndex = blockIdx.x * 2048 + threadIdx.x;
    if (elementIndex >= elementDimension)
        return;
    half bias = biases[blockIdx.y];

    unsigned long batchItemIndex = blockIdx.z * channelDimension * elementDimension;
    unsigned long channelIndex = blockIdx.y * elementDimension;
    unsigned long batchChannelIndex = batchItemIndex + channelIndex;

    data[batchChannelIndex + elementIndex] += bias;

    elementIndex += 256;
    if (elementIndex >= elementDimension)
        return;

    data[batchChannelIndex + elementIndex] += bias;

    elementIndex += 256;
    if (elementIndex >= elementDimension)
        return;

    data[batchChannelIndex + elementIndex] += bias;

    elementIndex += 256;
    if (elementIndex >= elementDimension)
        return;

    data[batchChannelIndex + elementIndex] += bias;

    elementIndex += 256;
    if (elementIndex >= elementDimension)
        return;

    data[batchChannelIndex + elementIndex] += bias;

    elementIndex += 256;
    if (elementIndex >= elementDimension)
        return;

    data[batchChannelIndex + elementIndex] += bias;

    elementIndex += 256;
    if (elementIndex >= elementDimension)
        return;

    data[batchChannelIndex + elementIndex] += bias;

    elementIndex += 256;
    if (elementIndex >= elementDimension)
        return;

    data[batchChannelIndex + elementIndex] += bias;
}

__device__ __forceinline__ float gck_warpReduce32(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0x0000ffff, val, 8);
    val += __shfl_down_sync(0x000000ff, val, 4);
    val += __shfl_down_sync(0x0000000f, val, 2);
    return val + __shfl_down_sync(0x00003, val, 1);
}

__device__ __forceinline__ float gck_warpReduceBottom16(float val) {
    val += __shfl_down_sync(0x0000ffff, val, 8);
    val += __shfl_down_sync(0x000000ff, val, 4);
    val += __shfl_down_sync(0x0000000f, val, 2);
    return val + __shfl_down_sync(0x00003, val, 1);
}

__device__ __forceinline__ float gck_warpReduceBottom4(float val) {
    val += __shfl_down_sync(0x0000000f, val, 2);
    return val + __shfl_down_sync(0x00003, val, 1);
}

// Sum the gradient for all output elements per channel per batch item
__global__ void computeBiasesGradientPerBatchElement(
    half *errorInput, float *workspace, unsigned int batchDimension, unsigned int channelDimension, unsigned int elementDimension) {
    __shared__ float partialSums[16];

    unsigned int elementIndex = threadIdx.x;
    if ((elementIndex & 0xFFFFFFE0) >= elementDimension) {
        if (threadIdx.x % 32 == 0)
            partialSums[threadIdx.x / 32] = 0.0f;
        return;
    }

    unsigned long elementStartIndex = blockIdx.y * channelDimension * elementDimension + blockIdx.x * elementDimension;
    float buff = 0.0f;

    // Compute 512 partial sums and then reduce into a single sum
    while (elementIndex < elementDimension) {
        buff += (float)errorInput[elementStartIndex + elementIndex];
        elementIndex += 512;
    }

    // Reduce to the final single sum and write the result
    buff = gck_warpReduce32(buff);
    if (threadIdx.x % 32 == 0)
        partialSums[threadIdx.x / 32] = buff;
    __syncthreads();
    if (threadIdx.x < 32)
        buff = gck_warpReduceBottom16(partialSums[threadIdx.x]);
    if (threadIdx.x == 0) {
        unsigned int batchElement = blockIdx.y;
        unsigned int channelElement = blockIdx.x;
        workspace[channelElement * batchDimension + batchElement] = buff;
    }
}

// Sum the corresponding channel of each batch item to get the channels bias gradient
__global__ void computeOverallBiasesGradient(float *workspace, half *biasesGradient, unsigned int batchDimension) {
    __shared__ float partialSums[4];

    unsigned int batchIndex = threadIdx.x;
    if ((batchIndex & 0xFFFFFFE0) >= batchDimension) {
        if (threadIdx.x % 32 == 0)
            partialSums[threadIdx.x / 32] = 0.0f;
        return;
    }

    float buff = 0.0f;
    unsigned int channelIndex = blockIdx.x;
    while (batchIndex < batchDimension) {
        buff += workspace[channelIndex * batchDimension + batchIndex];
        batchIndex += 128;
    }

    buff = gck_warpReduce32(buff);
    if (threadIdx.x % 32 == 0)
        partialSums[threadIdx.x / 32] = buff;
    __syncthreads();
    if (threadIdx.x < 32)
        buff = gck_warpReduceBottom4(partialSums[threadIdx.x]);
    if (threadIdx.x == 0)
        biasesGradient[channelIndex] = (half)buff;
}

void GpuConvolution::addConvolutionBias(Tensor dataOutput, Tensor biases, Stream stream) {
    assert(dataOutput.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(dataOutput.getPlacement() == biases.getPlacement());
    int gpuNum = stream.getGpuNum();

    vector<unsigned long> dataDimensions = dataOutput.getDescriptor().getDimensions();
    vector<unsigned long> biasDimensions = biases.getDescriptor().getDimensions();
    assert(dataDimensions.size() == 4);
    assert(biasDimensions.size() == 1);

    unsigned int batchDimension = dataDimensions[0];
    unsigned int channelDimension = dataDimensions[1];
    unsigned int elementDimension = dataDimensions[2] * dataDimensions[3];

    assert(biasDimensions[0] == channelDimension);

    ScopedGpu scopedGpu(gpuNum);

    dim3 blockSize(256);
    dim3 gridSize((elementDimension + 2047) / 2048, channelDimension, batchDimension);
    addConvolutionBiasKernel<<<gridSize, blockSize, 0, stream>>>(
        (half *)dataOutput.getMemPtr(), (half *)biases.getMemPtr(), batchDimension, channelDimension, elementDimension);
}

void GpuConvolution::computeConvolutionBiasesGradient(Tensor errorInput, Tensor biasesGradient, Tensor workspace, Stream stream) {
    assert(errorInput.getPlacement().getMemDevice() == TensorPlacement::MemDevices::GPU);
    assert(errorInput.getPlacement() == biasesGradient.getPlacement());
    assert(errorInput.getPlacement() == workspace.getPlacement());
    int gpuNum = stream.getGpuNum();

    vector<unsigned long> dataDimensions = errorInput.getDescriptor().getDimensions();
    vector<unsigned long> biasDimensions = biasesGradient.getDescriptor().getDimensions();
    vector<unsigned long> workspaceDimensions = workspace.getDescriptor().getDimensions();

    assert(dataDimensions.size() == 4);
    unsigned int batchDimension = dataDimensions[0];
    unsigned int channelDimension = dataDimensions[1];
    unsigned int elementDimension = dataDimensions[2] * dataDimensions[3];

    assert(biasDimensions.size() == 1);
    assert(biasDimensions[0] == channelDimension);
    assert(biasesGradient.getDescriptor().getDataType() == TensorDescriptor::DataType::FP16);

    assert(workspace.getDescriptor().getArraySizeInBytes() == batchDimension * channelDimension * sizeof(float));

    ScopedGpu scopedGpu(gpuNum);

    dim3 blockSize(512);
    dim3 gridSize(channelDimension, batchDimension);
    computeBiasesGradientPerBatchElement<<<gridSize, blockSize, 0, stream>>>(
        (half *)errorInput.getMemPtr(), (float *)workspace.getMemPtr(), batchDimension, channelDimension, elementDimension);

    blockSize = dim3(128);
    gridSize = dim3(channelDimension);
    computeOverallBiasesGradient<<<gridSize, blockSize, 0, stream>>>(
        (float *)workspace.getMemPtr(), (half *)biasesGradient.getMemPtr(), batchDimension);
}
