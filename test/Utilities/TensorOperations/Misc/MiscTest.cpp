#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>

using std::advance;
using std::set;

inline int randomElement(set<int> &filledSet) {
    assert(!filledSet.empty());
    set<int>::iterator it = filledSet.begin();
    advance(it, rand() % filledSet.size());
    int element = *it;
    filledSet.erase(it);
    return element;
}

TEST(MapInt, MapsCorrectly) {
    srand(time(NULL));

    half source[4096];
    unsigned int mapping[4096];
    half dest[4096];

    set<int> destinationIndex;

    cudaError_t cudaStatus;

    half *source_d;
    unsigned int *mapping_d;
    half *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&mapping_d, 4096 * sizeof(int));
    assert(cudaStatus == cudaSuccess);

    for (int i = 0; i < 10; ++i) {
        int numElements = (rand() % 4096) + 1;
        for (int i = 0; i < numElements; ++i)
            destinationIndex.insert(i);
        for (int i = 0; i < numElements; ++i)
            mapping[i] = randomElement(destinationIndex);
        for (int i = 0; i < numElements; ++i)
            source[i] = ((rand() % 100) / 10.0f) - 5.0f;

        cudaStatus = cudaMemcpyAsync(source_d, source, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(mapping_d, mapping, numElements * sizeof(int), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchMap<unsigned int>(dest_d, source_d, mapping_d, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ(dest[i], source[mapping[i]]);
        }
    }

    cudaStatus = cudaFree(source_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(mapping_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(MapLong, MapsCorrectly) {
    srand(time(NULL));

    half source[4096];
    unsigned long mapping[4096];
    half dest[4096];

    set<int> destinationIndex;

    cudaError_t cudaStatus;

    half *source_d;
    unsigned long *mapping_d;
    half *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&mapping_d, 4096 * sizeof(long));
    assert(cudaStatus == cudaSuccess);

    for (int i = 0; i < 10; ++i) {
        int numElements = (rand() % 4096) + 1;
        for (int i = 0; i < numElements; ++i)
            destinationIndex.insert(i);
        for (int i = 0; i < numElements; ++i)
            mapping[i] = randomElement(destinationIndex);
        for (int i = 0; i < numElements; ++i)
            source[i] = ((rand() % 100) / 10.0f) - 5.0f;

        cudaStatus = cudaMemcpyAsync(source_d, source, numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(mapping_d, mapping, numElements * sizeof(long), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchMap<unsigned long>(dest_d, source_d, mapping_d, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ(dest[i], source[mapping[i]]);
        }
    }

    cudaStatus = cudaFree(source_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(mapping_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}

inline void computeIndex(int flatIndex, int index[], int numDimensions, long stridePerDimension[]) {
    for (int i = 0; i < numDimensions; ++i) {
        index[i] = flatIndex / stridePerDimension[i];
        flatIndex -= index[i] * stridePerDimension[i];
    }
    assert(flatIndex == 0);
}

#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
inline int computeFlatIndex(int index[], long stridePerDimension[], int numDimensions) {
    int flatIndex = 0;
    for (int i = 0; i < numDimensions; ++i)
        flatIndex += index[i] * stridePerDimension[i];
    return flatIndex;
}
#pragma GCC diagnostic pop

TEST(Split, SplitsCorrectly) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    long axisElementsPerDestArray[10];
    long stridePerSourceDimension[10];
    long stridePerDestDimension[5 * 10];

    for (int test = 0; test < 50; ++test) {
        vector<Tensor> partsCpu;
        vector<Tensor> partsGpu;
        Tensor wholeCpu;
        Tensor wholeGpu;

        vector<unsigned long> wholeDimensions;
        int numDimensions = (rand() % 6) + 1;
        int axis = rand() % numDimensions;
        for (int d = 0; d < numDimensions; d++) {
            if (d == axis)
                wholeDimensions.push_back(0);
            else
                wholeDimensions.push_back((rand() % 10) + 1);
        }

        int numSplitTensors = (rand() % 5) + 2;
        for (int t = 0; t < numSplitTensors; ++t) {
            vector<unsigned long> splitArrayDimensions = wholeDimensions;
            splitArrayDimensions[axis] = (rand() % 5) + 1;
            axisElementsPerDestArray[t] = splitArrayDimensions[axis];
            wholeDimensions[axis] += splitArrayDimensions[axis];

            TensorDescriptor partDescriptor(TensorDescriptor::DataType::FP16, splitArrayDimensions);
            partsCpu.emplace_back(cpuPlacement, partDescriptor);
            partsGpu.emplace_back(gpuPlacement, partDescriptor);
        }
        TensorDescriptor wholeDescriptor(TensorDescriptor::DataType::FP16, wholeDimensions);
        wholeCpu = Tensor(cpuPlacement, wholeDescriptor);
        wholeGpu = Tensor(gpuPlacement, wholeDescriptor);

        long numElements = wholeCpu.getDescriptor().getTotalNumElements();
        half *mem = (half *)wholeCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            mem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        wholeGpu.copyFromAsync(wholeCpu, stream);

        half *splitTensorMemArray[10];
        half **splitTensorMemArray_d;
        for (int i = 0; i < numSplitTensors; ++i)
            splitTensorMemArray[i] = (half *)partsGpu[i].getMemPtr();
        cudaStatus = cudaMalloc(&splitTensorMemArray_d, numSplitTensors * sizeof(half *));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            splitTensorMemArray_d, splitTensorMemArray, numSplitTensors * sizeof(half *), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        long *axisElementsPerDestArray_d;
        cudaStatus = cudaMalloc(&axisElementsPerDestArray_d, numSplitTensors * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(axisElementsPerDestArray_d,
                                     axisElementsPerDestArray,
                                     numSplitTensors * sizeof(unsigned long),
                                     cudaMemcpyHostToDevice,
                                     stream.getStream());
        assert(cudaStatus == cudaSuccess);

        stridePerSourceDimension[numDimensions - 1] = 1;
        for (int dest = 0; dest < numSplitTensors; dest++)
            stridePerDestDimension[dest * numDimensions + numDimensions - 1] = 1;
        for (int i = numDimensions - 2; i >= 0; --i) {
            stridePerSourceDimension[i] = stridePerSourceDimension[i + 1] * wholeDimensions[i + 1];
            for (int dest = 0; dest < numSplitTensors; ++dest)
                if (i + 1 == axis)
                    stridePerDestDimension[dest * numDimensions + i] =
                        stridePerDestDimension[dest * numDimensions + i + 1] * axisElementsPerDestArray[dest];
                else
                    stridePerDestDimension[dest * numDimensions + i] =
                        stridePerDestDimension[dest * numDimensions + i + 1] * wholeDimensions[i + 1];
        }
        long *stridePerSourceDimension_d;
        cudaStatus = cudaMalloc(&stridePerSourceDimension_d, numDimensions * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            stridePerSourceDimension_d, stridePerSourceDimension, numDimensions * sizeof(long), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        long *stridePerDestDimension_d;
        cudaStatus = cudaMalloc(&stridePerDestDimension_d, numDimensions * numSplitTensors * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(stridePerDestDimension_d,
                                     stridePerDestDimension,
                                     numDimensions * numSplitTensors * sizeof(long),
                                     cudaMemcpyHostToDevice,
                                     stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchSplit(splitTensorMemArray_d,
                    (half *)wholeGpu.getMemPtr(),
                    numElements,
                    numDimensions,
                    numSplitTensors,
                    axis,
                    axisElementsPerDestArray_d,
                    stridePerSourceDimension_d,
                    stridePerDestDimension_d,
                    stream);

        for (int i = 0; i < numSplitTensors; ++i)
            partsCpu[i].copyFromAsync(partsGpu[i], stream);
        stream.synchronize();

        half *wholeMem = (half *)wholeCpu.getMemPtr();
        vector<int> destTensorAxisIndexStart;
        destTensorAxisIndexStart.push_back(0);
        for (int i = 1; i < numSplitTensors; ++i)
            destTensorAxisIndexStart.push_back(destTensorAxisIndexStart.back() + axisElementsPerDestArray[i - 1]);

        int sourceIndex[10];
        int destIndex[10];
        for (int sourceFlatIndex = 0; sourceFlatIndex < numElements; ++sourceFlatIndex) {
            computeIndex(sourceFlatIndex, sourceIndex, numDimensions, stridePerSourceDimension);
            int dest = 0;
            for (; sourceIndex[axis] >= destTensorAxisIndexStart[dest] + axisElementsPerDestArray[dest]; ++dest)
                ;
            for (int i = 0; i < numDimensions; ++i) {
                if (i == axis)
                    destIndex[i] = sourceIndex[i] - destTensorAxisIndexStart[dest];
                else
                    destIndex[i] = sourceIndex[i];
            }
            int destFlatIndex = computeFlatIndex(destIndex, stridePerDestDimension + dest * numDimensions, numDimensions);

            half *partsMem = (half *)partsCpu[dest].getMemPtr();
            ASSERT_EQ((float)wholeMem[sourceFlatIndex], (float)partsMem[destFlatIndex]);
        }

        cudaStatus = cudaFree(stridePerDestDimension_d);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaFree(stridePerSourceDimension_d);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaFree(axisElementsPerDestArray_d);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaFree(splitTensorMemArray_d);
        assert(cudaStatus == cudaSuccess);
    }
}

TEST(Concatenate, ConcatenatesCorrectly) {
    srand(time(NULL));

    cudaError_t cudaStatus;

    Stream stream(0);

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    long axisElementsPerSourceArray[10];
    long stridePerDestDimension[10];
    long stridePerSourceDimension[5 * 10];

    for (int test = 0; test < 50; ++test) {
        vector<Tensor> partsCpu;
        vector<Tensor> partsGpu;
        Tensor wholeCpu;
        Tensor wholeGpu;

        vector<unsigned long> wholeDimensions;
        int numDimensions = (rand() % 6) + 1;
        int axis = rand() % numDimensions;
        for (int d = 0; d < numDimensions; d++) {
            if (d == axis)
                wholeDimensions.push_back(0);
            else
                wholeDimensions.push_back((rand() % 10) + 1);
        }

        int numSplitTensors = (rand() % 5) + 2;
        for (int t = 0; t < numSplitTensors; ++t) {
            vector<unsigned long> splitArrayDimensions = wholeDimensions;
            splitArrayDimensions[axis] = (rand() % 5) + 1;
            axisElementsPerSourceArray[t] = splitArrayDimensions[axis];
            wholeDimensions[axis] += splitArrayDimensions[axis];

            TensorDescriptor partDescriptor(TensorDescriptor::DataType::FP16, splitArrayDimensions);
            partsCpu.emplace_back(cpuPlacement, partDescriptor);
            partsGpu.emplace_back(gpuPlacement, partDescriptor);
        }
        TensorDescriptor wholeDescriptor(TensorDescriptor::DataType::FP16, wholeDimensions);
        wholeCpu = Tensor(cpuPlacement, wholeDescriptor);
        wholeGpu = Tensor(gpuPlacement, wholeDescriptor);

        for (unsigned int i = 0; i < partsCpu.size(); ++i) {
            long numElements = partsCpu[i].getDescriptor().getTotalNumElements();
            half *mem = (half *)partsCpu[i].getMemPtr();
            for (int i = 0; i < numElements; ++i) {
                mem[i] = ((rand() % 100) / 10.0f) - 5.0f;
            }
            partsGpu[i].copyFromAsync(partsCpu[i], stream);
        }

        half *splitTensorMemArray[10];
        half **splitTensorMemArray_d;
        for (int i = 0; i < numSplitTensors; ++i)
            splitTensorMemArray[i] = (half *)partsGpu[i].getMemPtr();
        cudaStatus = cudaMalloc(&splitTensorMemArray_d, numSplitTensors * sizeof(half *));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            splitTensorMemArray_d, splitTensorMemArray, numSplitTensors * sizeof(half *), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        long *axisElementsPerSourceArray_d;
        cudaStatus = cudaMalloc(&axisElementsPerSourceArray_d, numSplitTensors * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(axisElementsPerSourceArray_d,
                                     axisElementsPerSourceArray,
                                     numSplitTensors * sizeof(unsigned long),
                                     cudaMemcpyHostToDevice,
                                     stream.getStream());
        assert(cudaStatus == cudaSuccess);

        stridePerDestDimension[numDimensions - 1] = 1;
        for (int dest = 0; dest < numSplitTensors; dest++)
            stridePerSourceDimension[dest * numDimensions + numDimensions - 1] = 1;
        for (int i = numDimensions - 2; i >= 0; --i) {
            stridePerDestDimension[i] = stridePerDestDimension[i + 1] * wholeDimensions[i + 1];
            for (int dest = 0; dest < numSplitTensors; ++dest)
                if (i + 1 == axis)
                    stridePerSourceDimension[dest * numDimensions + i] =
                        stridePerSourceDimension[dest * numDimensions + i + 1] * axisElementsPerSourceArray[dest];
                else
                    stridePerSourceDimension[dest * numDimensions + i] =
                        stridePerSourceDimension[dest * numDimensions + i + 1] * wholeDimensions[i + 1];
        }
        long *stridePerDestDimension_d;
        cudaStatus = cudaMalloc(&stridePerDestDimension_d, numDimensions * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            stridePerDestDimension_d, stridePerDestDimension, numDimensions * sizeof(long), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        long *stridePerSourceDimension_d;
        cudaStatus = cudaMalloc(&stridePerSourceDimension_d, numDimensions * numSplitTensors * sizeof(long));
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(stridePerSourceDimension_d,
                                     stridePerSourceDimension,
                                     numDimensions * numSplitTensors * sizeof(long),
                                     cudaMemcpyHostToDevice,
                                     stream.getStream());
        assert(cudaStatus == cudaSuccess);

        long numElements = wholeCpu.getDescriptor().getTotalNumElements();
        launchConcatenate((half *)wholeGpu.getMemPtr(),
                          splitTensorMemArray_d,
                          numElements,
                          numDimensions,
                          numSplitTensors,
                          axis,
                          axisElementsPerSourceArray_d,
                          stridePerDestDimension_d,
                          stridePerSourceDimension_d,
                          stream);

        wholeCpu.copyFromAsync(wholeGpu, stream);
        stream.synchronize();

        half *wholeMem = (half *)wholeCpu.getMemPtr();
        vector<int> sourceTensorAxisIndexStart;
        sourceTensorAxisIndexStart.push_back(0);
        for (int i = 1; i < numSplitTensors; ++i)
            sourceTensorAxisIndexStart.push_back(sourceTensorAxisIndexStart.back() + axisElementsPerSourceArray[i - 1]);

        int destIndex[10];
        int sourceIndex[10];
        for (int destFlatIndex = 0; destFlatIndex < numElements; ++destFlatIndex) {
            computeIndex(destFlatIndex, destIndex, numDimensions, stridePerDestDimension);
            int source = 0;
            for (; destIndex[axis] >= sourceTensorAxisIndexStart[source] + axisElementsPerSourceArray[source]; ++source)
                ;
            for (int i = 0; i < numDimensions; ++i) {
                if (i == axis)
                    sourceIndex[i] = destIndex[i] - sourceTensorAxisIndexStart[source];
                else
                    sourceIndex[i] = destIndex[i];
            }
            int sourceFlatIndex = computeFlatIndex(sourceIndex, stridePerSourceDimension + source * numDimensions, numDimensions);

            half *partsMem = (half *)partsCpu[source].getMemPtr();
            // printf("sourceFlatIndex%d %d destFlatIndex %d sourceIndex%d[%d][%d] destIndex[%d][%d] source %f dest %f\n", source,
            // sourceFlatIndex, destFlatIndex, source, sourceIndex[0], sourceIndex[1], destIndex[0], destIndex[1],
            // (float)partsMem[sourceFlatIndex], (float)wholeMem[destFlatIndex]);
            ASSERT_EQ((float)wholeMem[destFlatIndex], (float)partsMem[sourceFlatIndex]);
        }

        cudaStatus = cudaFree(stridePerSourceDimension_d);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaFree(stridePerDestDimension_d);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaFree(axisElementsPerSourceArray_d);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaFree(splitTensorMemArray_d);
        assert(cudaStatus == cudaSuccess);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
