#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/TensorOperations/Misc/Concatenate.h"
#include "Utilities/TensorOperations/Misc/Split.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <set>

using namespace ThorImplementation;
using namespace std;

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

            TensorDescriptor partDescriptor(DataType::FP16, splitArrayDimensions);
            partsCpu.emplace_back(cpuPlacement, partDescriptor);
            partsGpu.emplace_back(gpuPlacement, partDescriptor);
        }
        TensorDescriptor wholeDescriptor(DataType::FP16, wholeDimensions);
        wholeCpu = Tensor(cpuPlacement, wholeDescriptor);
        wholeGpu = Tensor(gpuPlacement, wholeDescriptor);

        long numElements = wholeCpu.getDescriptor().getTotalNumElements();
        half *mem = (half *)wholeCpu.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            mem[i] = ((rand() % 100) / 10.0f) - 5.0f;
        }
        wholeGpu.copyFromAsync(wholeCpu, stream);

        std::vector<void *> splitTensorMemArray(numSplitTensors);
        for (int i = 0; i < numSplitTensors; ++i)
            splitTensorMemArray[i] = partsGpu[i].getMemPtr();

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

        uint64_t innerElements = 1;
        for (int d = axis + 1; d < numDimensions; ++d) innerElements *= wholeDimensions[d];
        uint64_t outerSlices = 1;
        for (int d = 0; d < axis; ++d) outerSlices *= wholeDimensions[d];
        std::vector<uint64_t> spanElements(numSplitTensors);
        for (int i = 0; i < numSplitTensors; ++i) spanElements[i] = axisElementsPerDestArray[i] * innerElements;
        const std::vector<DataType> types(numSplitTensors, DataType::FP16);

        launchSplit(splitTensorMemArray, types, wholeGpu.getMemPtr(), DataType::FP16, outerSlices, spanElements, stream);

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
    }
}

TEST(Concatenate, ConcatenatesCorrectly) {
    srand(time(NULL));

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

            TensorDescriptor partDescriptor(DataType::FP16, splitArrayDimensions);
            partsCpu.emplace_back(cpuPlacement, partDescriptor);
            partsGpu.emplace_back(gpuPlacement, partDescriptor);
        }
        TensorDescriptor wholeDescriptor(DataType::FP16, wholeDimensions);
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

        std::vector<void *> splitTensorMemArray(numSplitTensors);
        for (int i = 0; i < numSplitTensors; ++i)
            splitTensorMemArray[i] = partsGpu[i].getMemPtr();

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

        uint64_t innerElements = 1;
        for (int d = axis + 1; d < numDimensions; ++d) innerElements *= wholeDimensions[d];
        uint64_t outerSlices = 1;
        for (int d = 0; d < axis; ++d) outerSlices *= wholeDimensions[d];
        std::vector<uint64_t> spanElements(numSplitTensors);
        for (int i = 0; i < numSplitTensors; ++i) spanElements[i] = axisElementsPerSourceArray[i] * innerElements;
        const std::vector<DataType> types(numSplitTensors, DataType::FP16);

        long numElements = wholeCpu.getDescriptor().getTotalNumElements();
        launchConcatenate(wholeGpu.getMemPtr(), DataType::FP16, splitTensorMemArray, types, outerSlices, spanElements, stream);

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
    }
}
