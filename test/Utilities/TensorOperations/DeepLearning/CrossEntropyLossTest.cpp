#include "Thor.h"

#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

using namespace ThorImplementation;

TEST(CrossEntropyLoss, ComputesCorrectAnswer_perClassLabels) {
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
                    } else {
                        probabilities[i + b * numElementsPerBatch] = rand() % 10000;
                        totalProb += probabilities[i + b * numElementsPerBatch];
                    }
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

        launchCrossEntropyLoss_perClassLabels(labels_d, probabilities_d, workspace_d, loss_d, numElementsPerBatch, batchSize, stream);

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

TEST(CrossEntropyLoss, ComputesCorrectAnswer_classIndexLabels) {
    srand(time(NULL));

    uint32_t labels[4096];
    float probabilities[4096];
    float loss[4096];
    float loss_cpu[4096];

    cudaError_t cudaStatus;

    uint32_t *labels_d;
    float *probabilities_d;
    float *workspace_d;
    float *loss_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&labels_d, 4096 * sizeof(uint32_t));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&probabilities_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&workspace_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&loss_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    for (int t = 0; t < 50; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;
        uint32_t numClasses = (rand() % 512) + 1;
        for (uint32_t b = 0; b < batchSize; ++b) {
            double totalProb = 0.0;
            while (totalProb < 0.00001) {
                totalProb = 0.0;
                if (rand() % 50 == 0) {
                    labels[b] = 10000 + (rand() % 1000000);
                } else {
                    labels[b] = rand() % numClasses;
                }
                for (uint32_t i = 0; i < numClasses; ++i) {
                    if (rand() % 50 == 0) {
                        int sel = rand() % 5;
                        if (sel == 0)
                            probabilities[i + b * numClasses] = 0.0f;
                        else if (sel == 1)
                            probabilities[i + b * numClasses] = -0.0f;
                        else if (sel == 2)
                            probabilities[i + b * numClasses] = NAN;
                        else if (sel == 3)
                            probabilities[i + b * numClasses] = std::numeric_limits<float>::infinity();
                        else if (sel == 4)
                            probabilities[i + b * numClasses] = -std::numeric_limits<float>::infinity();
                    } else {
                        probabilities[i + b * numClasses] = rand() % 10000;
                        totalProb += probabilities[i + b * numClasses];
                    }
                }
            }
            for (uint32_t i = 0; i < numClasses; ++i) {
                probabilities[i + b * numClasses] /= totalProb;
            }
        }

        cudaStatus = cudaMemcpyAsync(labels_d, labels, batchSize * sizeof(uint32_t), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            probabilities_d, probabilities, numClasses * batchSize * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        launchCrossEntropyLoss_classIndexLabels(labels_d, probabilities_d, workspace_d, loss_d, numClasses, batchSize, stream);

        cudaStatus = cudaMemcpyAsync(loss, loss_d, batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (uint32_t b = 0; b < batchSize; ++b) {
            uint32_t trueClass = labels[b];
            float probability;
            if (labels[b] >= numClasses) {
                probability = 1.0e-15f;
            } else {
                probability = probabilities[trueClass + b * numClasses];
                if (probability < 1.0e-15f || !isfinite(probability))
                    probability = 1.0e-15f;
            }
            loss_cpu[b] = -log(probability);
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(loss[b] - loss_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("numClasses %d element %d %f %f\n", numClasses, b, loss[b], loss_cpu[b]);
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

TEST(LossGradient, ComputesCorrectAnswer_classIndexLabels) {
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 1500) + 1;
        uint32_t numClasses = (rand() % 100) + 1;
        Tensor labels_h(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT32, {batchSize, 1}));
        Tensor probabilities_h(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor lossGradient_h(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        uint32_t *labelsMem = (uint32_t *)labels_h.getMemPtr();
        float *probabilitiesMem = (float *)probabilities_h.getMemPtr();
        float *lossGradientMem = (float *)lossGradient_h.getMemPtr();

        for (uint32_t b = 0; b < batchSize; ++b) {
            labelsMem[b] = rand() % numClasses;
            for (uint32_t c = 0; c < numClasses; ++c) {
                probabilitiesMem[c + b * numClasses] = (rand() % 1000) / 1000.0f;
                lossGradientMem[c + b * numClasses] = probabilitiesMem[c + b * numClasses] - (labelsMem[b] == c ? 1.0f : 0.0f);
            }
        }

        Stream stream(0);
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor probabilities_d = probabilities_h.clone();
        labels_d.copyFromAsync(labels_h, stream);
        probabilities_d.copyFromAsync(probabilities_h, stream);

        Tensor lossGradient_d = lossGradient_h.clone(gpuPlacement);
        Tensor lossGradientGpu_h = lossGradient_h.clone();

        launchLossGradient_classIndexLabels((uint32_t *)labels_d.getMemPtr(),
                                            (float *)probabilities_d.getMemPtr(),
                                            (float *)lossGradient_d.getMemPtr(),
                                            numClasses,
                                            batchSize,
                                            stream);
        lossGradientGpu_h.copyFromAsync(lossGradient_d, stream);
        stream.synchronize();

        float *lossGradientGpuMem = (float *)lossGradientGpu_h.getMemPtr();
        float thresh = 0.001;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                float diff = abs(lossGradientMem[c + b * numClasses] - lossGradientGpuMem[c + b * numClasses]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf("numClasses %d batchItem %d %f %f\n",
                           numClasses,
                           b,
                           lossGradientMem[c + b * numClasses],
                           lossGradientGpuMem[c + b * numClasses]);
                }
                EXPECT_LT(diff, thresh);
            }
        }
    }

    /*
    template <typename LABEL_TYPE, typename PROBABILITY_TYPE, typename LOSS_GRADIENT_TYPE>
    void launchLossGradient_classIndexLabels(LABEL_TYPE *labels_d,
                                             PROBABILITY_TYPE *probabilities_d,
                                             LOSS_GRADIENT_TYPE *lossGradient_d,
                                             uint64_t numClasses,
                                             uint64_t batchSize,
                                             Stream stream)
    */
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
