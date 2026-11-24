#include "Utilities/TensorOperations/Arithmetic/Sum.h"

#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

TEST(Sum, SumsCorrectly) {
    srand(time(NULL));

    half source[10][4096];
    half dest[4096];
    float dest_cpu[4096];

    cudaError_t cudaStatus;

    half **source_d;
    half *dest_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&dest_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&source_d, 10 * sizeof(half *));
    assert(cudaStatus == cudaSuccess);
    half *sourceArray[10];
    for (int i = 0; i < 10; ++i) {
        cudaStatus = cudaMalloc(&(sourceArray[i]), 4096 * sizeof(half));
        assert(cudaStatus == cudaSuccess);
    }
    cudaStatus = cudaMemcpyAsync(source_d, sourceArray, 10 * sizeof(half *), cudaMemcpyHostToDevice, stream.getStream());
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int numInstances = (rand() % 10) + 1;
        int numElements = (rand() % 4096) + 1;

        for (int i = 0; i < numInstances; ++i) {
            for (int j = 0; j < numElements; ++j) {
                source[i][j] = ((rand() % 100) / 10.0f) - 5.0f;
            }
            cudaStatus = cudaMemcpyAsync(sourceArray[i], source[i], numElements * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
            assert(cudaStatus == cudaSuccess);
        }

        launchSum(dest_d, source_d, numInstances, numElements, stream);

        cudaStatus = cudaMemcpyAsync(dest, dest_d, numElements * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int i = 0; i < numInstances; ++i) {
            for (int j = 0; j < numElements; ++j) {
                if (i == 0)
                    dest_cpu[j] = (float)source[i][j];
                else
                    dest_cpu[j] = dest_cpu[j] + (float)source[i][j];
            }
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (int i = 0; i < numElements; ++i) {
            float diff = abs((float)dest[i] - (float)(half)dest_cpu[i]);
            if (diff >= thresh) {
                printf("numElements %d element %d %f %f", numElements, i, (float)dest[i], (float)(half)dest_cpu[i]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    for (int i = 0; i < 10; ++i) {
        cudaStatus = cudaFree(sourceArray[i]);
        assert(cudaStatus == cudaSuccess);
    }
    cudaStatus = cudaFree(source_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(dest_d);
    assert(cudaStatus == cudaSuccess);
}
