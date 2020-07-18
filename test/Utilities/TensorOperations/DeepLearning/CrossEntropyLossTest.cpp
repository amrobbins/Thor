#include "MLDev.h"

#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

TEST(CrossEntropyLoss, ComputesCorrectAnswer) {
    srand(time(NULL));

    float labels[4096];
    float probabilities[4096];
    float loss[4096];
    float loss_cpu[4096];

    cudaError_t cudaStatus;

    float *labels_d;
    float *probabilities_d;
    float *workspace_d;
    float *loss_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&labels_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&probabilities_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&workspace_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&loss_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        int batchSize = (rand() % 8) + 1;
        int numElementsPerBatch = (rand() % 512) + 1;
        for (int b = 0; b < batchSize; ++b) {
            double totalProb = 0.0;
            while (totalProb < 0.00001) {
                totalProb = 0.0;
                for (int i = 0; i < numElementsPerBatch; ++i) {
                    if (rand() % 2)
                        labels[i + b * numElementsPerBatch] = 0.0f;
                    else
                        labels[i + b * numElementsPerBatch] = ((rand() % 1000) / 999.0f);
                    if (rand() % 50 == 0) {
                        int sel = rand() % 5;
                        if (sel == 0)
                            probabilities[i + b * numElementsPerBatch] = 0.0f;
                        else if (sel == 1)
                            probabilities[i + b * numElementsPerBatch] = -0.0f;
                        else if (sel == 2)
                            probabilities[i + b * numElementsPerBatch] = NAN;
                        else if (sel == 3)
                            probabilities[i + b * numElementsPerBatch] = std::numeric_limits<float>::infinity();
                        else if (sel == 4)
                            probabilities[i + b * numElementsPerBatch] = -std::numeric_limits<float>::infinity();
                    } else
                        probabilities[i + b * numElementsPerBatch] = rand() % 10000;
                    totalProb += probabilities[i + b * numElementsPerBatch];
                }
            }
            for (int i = 0; i < numElementsPerBatch; ++i) {
                probabilities[i + b * numElementsPerBatch] /= totalProb;
            }
        }

        cudaStatus =
            cudaMemcpyAsync(labels_d, labels, numElementsPerBatch * batchSize * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            probabilities_d, probabilities, numElementsPerBatch * batchSize * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchCrossEntropyLoss(labels_d, probabilities_d, workspace_d, loss_d, numElementsPerBatch, batchSize, stream);

        cudaStatus = cudaMemcpyAsync(loss, loss_d, batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (int b = 0; b < batchSize; ++b) {
            loss_cpu[b] = 0.0f;
            for (int i = 0; i < numElementsPerBatch; ++i) {
                float probability = probabilities[i + b * numElementsPerBatch];
                if (probability < 1.0e-15f || !isfinite(probability))
                    probability = 1.0e-15f;
                loss_cpu[b] += -labels[i + b * numElementsPerBatch] * log(probability);
            }
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (int b = 0; b < batchSize; ++b) {
            float diff = abs(loss[b] - loss_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("numElementsPerBatch %d element %d %f %f\n", numElementsPerBatch, b, loss[b], loss_cpu[b]);
            }
            ASSERT_LT(diff, thresh);
        }
    }

    cudaStatus = cudaFree(labels_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(probabilities_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(workspace_d);
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaFree(loss_d);
    assert(cudaStatus == cudaSuccess);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
