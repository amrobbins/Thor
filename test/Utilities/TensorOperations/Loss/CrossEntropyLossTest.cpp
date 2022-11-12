#include "Thor.h"

#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

using namespace ThorImplementation;

const float MIN_PROBABILITY_FLOAT = 0.000000000000000000000000000000000001f;
const half MIN_PROBABILITY_HALF = 0.000062f;

TEST(CategoricalCrossEntropyLoss, ComputesCorrectAnswer_categoricalOneHotLabels) {
    srand(time(NULL));

    float labels[4096];
    float probabilities[4096];
    float loss[4096];
    float loss_cpu[4096];
    float gradient[4096];
    float gradient_cpu[4096];

    cudaError_t cudaStatus;

    float *labels_d;
    float *probabilities_d;
    float *workspace_d;
    float *loss_d;
    float *gradient_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&labels_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&probabilities_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&workspace_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&loss_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&gradient_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;
        uint32_t numClasses = (rand() % 512) + 1;
        for (uint32_t b = 0; b < batchSize; ++b) {
            double totalProb = 0.0;
            while (totalProb < 0.00001) {
                totalProb = 0.0;
                for (uint32_t i = 0; i < numClasses; ++i) {
                    if (rand() % 2)
                        labels[i + b * numClasses] = 0.0f;
                    else
                        labels[i + b * numClasses] = ((rand() % 1000) / 999.0f);
                    if (rand() % 50 == 0) {
                        uint32_t sel = rand() % 5;
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

        cudaStatus = cudaMemcpyAsync(labels_d, labels, numClasses * batchSize * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            probabilities_d, probabilities, numClasses * batchSize * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        uint32_t lossScalingFactor = 1 + rand() % 4;
        launchElementWiseCrossEntropyLoss<float, float, float>(labels_d,
                                                               probabilities_d,
                                                               loss_d,
                                                               gradient_d,
                                                               numClasses,
                                                               batchSize,
                                                               true,
                                                               lossScalingFactor,
                                                               CrossEntropyLossType::CATEGORICAL,
                                                               false,
                                                               stream);

        cudaStatus = cudaMemcpyAsync(loss, loss_d, batchSize * numClasses * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus =
            cudaMemcpyAsync(gradient, gradient_d, batchSize * numClasses * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float probability = probabilities[i + b * numClasses];
                float rawProbability = (!isfinite(probability) || isnan(probability)) ? 0.0f : probability;
                if (probability < MIN_PROBABILITY_FLOAT || (!isfinite(probability) || isnan(probability)))
                    probability = MIN_PROBABILITY_FLOAT;
                loss_cpu[b * numClasses + i] = -labels[i + b * numClasses] * logf(probability);
                gradient_cpu[b * numClasses + i] = (rawProbability - labels[i + b * numClasses]) * lossScalingFactor;
            }
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(loss[b * numClasses + i] - loss_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf("loss batchItem %d class %d, %f %f\n", b, i, loss_cpu[b * numClasses + i], loss[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(gradient[b * numClasses + i] - gradient_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf(
                        "gradient batchItem %d class %d,  %f %f\n", b, i, gradient_cpu[b * numClasses + i], gradient[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
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
    cudaStatus = cudaFree(gradient_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(CategoricalCrossEntropyLoss, ComputesCorrectAnswer_categoricalOneHotLabels_halfPrecision) {
    srand(time(NULL));

    half labels[4096];
    half probabilities[4096];
    half loss[4096];
    half loss_cpu[4096];
    half gradient[4096];
    half gradient_cpu[4096];

    cudaError_t cudaStatus;

    half *labels_d;
    half *probabilities_d;
    half *workspace_d;
    half *loss_d;
    half *gradient_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&labels_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&probabilities_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&workspace_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&loss_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&gradient_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;
        uint32_t numClasses = (rand() % 512) + 1;
        for (uint32_t b = 0; b < batchSize; ++b) {
            double totalProb = 0.0;
            while (totalProb < 0.00001) {
                totalProb = 0.0;
                for (uint32_t i = 0; i < numClasses; ++i) {
                    if (rand() % 2)
                        labels[i + b * numClasses] = 0.0f;
                    else
                        labels[i + b * numClasses] = ((rand() % 1000) / 999.0f);
                    if (rand() % 50 == 0) {
                        uint32_t sel = rand() % 5;
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
                        probabilities[i + b * numClasses] = (float)(rand() % 10000);
                        totalProb += probabilities[i + b * numClasses];
                    }
                }
            }
            for (uint32_t i = 0; i < numClasses; ++i) {
                probabilities[i + b * numClasses] = probabilities[i + b * numClasses] / totalProb;
            }
        }

        cudaStatus = cudaMemcpyAsync(labels_d, labels, numClasses * batchSize * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            probabilities_d, probabilities, numClasses * batchSize * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        uint32_t lossScalingFactor = 1 + rand() % 4;
        launchElementWiseCrossEntropyLoss<half, half, half>(labels_d,
                                                            probabilities_d,
                                                            loss_d,
                                                            gradient_d,
                                                            numClasses,
                                                            batchSize,
                                                            true,
                                                            lossScalingFactor,
                                                            CrossEntropyLossType::CATEGORICAL,
                                                            false,
                                                            stream);

        cudaStatus = cudaMemcpyAsync(loss, loss_d, batchSize * numClasses * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus =
            cudaMemcpyAsync(gradient, gradient_d, batchSize * numClasses * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                half probability = probabilities[i + b * numClasses];
                half rawProbability = (!isfinite(probability) || isnan(probability)) ? (half)0.0f : probability;
                if (probability < MIN_PROBABILITY_HALF || (!isfinite(probability) || isnan(probability)))
                    probability = MIN_PROBABILITY_HALF;
                loss_cpu[b * numClasses + i] = -labels[i + b * numClasses] * logf((float)probability);
                gradient_cpu[b * numClasses + i] = (rawProbability - labels[i + b * numClasses]) * lossScalingFactor;
            }
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs((float)loss[b * numClasses + i] - (float)loss_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf(
                        "loss batchItem %d class %d, %f %f\n", b, i, (float)loss_cpu[b * numClasses + i], (float)loss[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(gradient[b * numClasses + i] - gradient_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf("gradient batchItem %d class %d,  %f %f\n",
                           b,
                           i,
                           (float)gradient_cpu[b * numClasses + i],
                           (float)gradient[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
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
    cudaStatus = cudaFree(gradient_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(CategoricalCrossEntropyLoss, ComputesCorrectAnswer_classIndexLabels) {
    srand(time(NULL));

    uint32_t labels[4096];
    float probabilities[4096];
    float loss[4096];
    float loss_cpu[4096];
    float gradient[4096];
    float gradient_cpu[4096];

    cudaError_t cudaStatus;

    uint32_t *labels_d;
    float *probabilities_d;
    float *workspace_d;
    float *loss_d;
    float *gradient_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&labels_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&probabilities_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&workspace_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&loss_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&gradient_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;
        uint32_t numClasses = (rand() % 16) + 1;

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
                        uint32_t sel = rand() % 5;
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

        uint32_t lossScalingFactor = 1 + rand() % 4;
        launchElementWiseCrossEntropyLoss<uint32_t, float, float>(labels_d,
                                                                  probabilities_d,
                                                                  loss_d,
                                                                  gradient_d,
                                                                  numClasses,
                                                                  batchSize,
                                                                  true,
                                                                  lossScalingFactor,
                                                                  CrossEntropyLossType::CATEGORICAL,
                                                                  true,
                                                                  stream);

        cudaStatus = cudaMemcpyAsync(loss, loss_d, batchSize * numClasses * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus =
            cudaMemcpyAsync(gradient, gradient_d, batchSize * numClasses * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float probability = probabilities[i + b * numClasses];
                float rawProbability = (!isfinite(probability) || isnan(probability)) ? 0.0f : probability;
                if (probability < MIN_PROBABILITY_FLOAT || (!isfinite(probability) || isnan(probability)))
                    probability = MIN_PROBABILITY_FLOAT;
                if (labels[b] == i) {
                    loss_cpu[b * numClasses + i] = -logf(probability);
                    gradient_cpu[b * numClasses + i] = (rawProbability - 1) * lossScalingFactor;
                } else {
                    loss_cpu[b * numClasses + i] = 0.0f;
                    gradient_cpu[b * numClasses + i] = rawProbability * lossScalingFactor;
                }
                // printf("batch %d class %d label %i prob %f loss %f gradient %f lossScalingFactor %d\n", b, i, labels[b] == i,
                // probability, loss_cpu[b * numClasses + i], gradient_cpu[b * numClasses + i], lossScalingFactor);
            }
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(loss[b * numClasses + i] - loss_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf("loss batchItem %d class %d, %f %f\n", b, i, loss_cpu[b * numClasses + i], loss[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(gradient[b * numClasses + i] - gradient_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf("gradient batchItem %d class %d, %f %f\n", b, i, gradient_cpu[b * numClasses + i], gradient[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
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
    cudaStatus = cudaFree(gradient_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(CategoricalCrossEntropyLoss, ComputesCorrectAnswer_classIndexLabels_halfPrecision) {
    srand(time(NULL));

    uint16_t labels[4096];
    half probabilities[4096];
    half loss[4096];
    half loss_cpu[4096];
    half gradient[4096];
    half gradient_cpu[4096];

    cudaError_t cudaStatus;

    uint16_t *labels_d;
    half *probabilities_d;
    half *workspace_d;
    half *loss_d;
    half *gradient_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&labels_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&probabilities_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&workspace_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&loss_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&gradient_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;
        uint32_t numClasses = (rand() % 16) + 1;

        for (uint32_t b = 0; b < batchSize; ++b) {
            double totalProb = 0.0;
            while (totalProb < 0.00001) {
                totalProb = 0.0;
                if (rand() % 50 == 0) {
                    labels[b] = 10000 + (rand() % 10000);
                } else {
                    labels[b] = rand() % numClasses;
                }
                for (uint32_t i = 0; i < numClasses; ++i) {
                    if (rand() % 50 == 0) {
                        uint32_t sel = rand() % 5;
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
                        probabilities[i + b * numClasses] = (float)(rand() % 10000);
                        totalProb += probabilities[i + b * numClasses];
                    }
                }
            }
            for (uint32_t i = 0; i < numClasses; ++i) {
                probabilities[i + b * numClasses] = probabilities[i + b * numClasses] / totalProb;
            }
        }

        cudaStatus = cudaMemcpyAsync(labels_d, labels, batchSize * sizeof(uint16_t), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(
            probabilities_d, probabilities, numClasses * batchSize * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        uint32_t lossScalingFactor = 1 + rand() % 4;
        launchElementWiseCrossEntropyLoss<uint16_t, half, half>(labels_d,
                                                                probabilities_d,
                                                                loss_d,
                                                                gradient_d,
                                                                numClasses,
                                                                batchSize,
                                                                true,
                                                                lossScalingFactor,
                                                                CrossEntropyLossType::CATEGORICAL,
                                                                true,
                                                                stream);

        cudaStatus = cudaMemcpyAsync(loss, loss_d, batchSize * numClasses * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus =
            cudaMemcpyAsync(gradient, gradient_d, batchSize * numClasses * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                half probability = probabilities[i + b * numClasses];
                half rawProbability = (!isfinite(probability) || isnan(probability)) ? (half)0.0f : probability;
                if (probability < MIN_PROBABILITY_HALF || (!isfinite(probability) || isnan(probability)))
                    probability = MIN_PROBABILITY_HALF;
                if (labels[b] == i) {
                    loss_cpu[b * numClasses + i] = -logf(probability);
                    gradient_cpu[b * numClasses + i] = (rawProbability - 1) * lossScalingFactor;
                } else {
                    loss_cpu[b * numClasses + i] = 0.0f;
                    gradient_cpu[b * numClasses + i] = rawProbability * lossScalingFactor;
                }
                // printf("batch %d class %d label %i prob %f loss %f gradient %f lossScalingFactor %d\n", b, i, labels[b] == i,
                // (float)probability, (float)loss_cpu[b * numClasses + i], (float)gradient_cpu[b * numClasses + i], lossScalingFactor);
            }
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(loss[b * numClasses + i] - loss_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf(
                        "loss batchItem %d class %d, %f %f\n", b, i, (float)loss_cpu[b * numClasses + i], (float)loss[b * numClasses + i]);
                }
                EXPECT_LT(diff, thresh);
            }
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(gradient[b * numClasses + i] - gradient_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf("gradient batchItem %d class %d, %f %f\n",
                           b,
                           i,
                           (float)gradient_cpu[b * numClasses + i],
                           (float)gradient[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
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
    cudaStatus = cudaFree(gradient_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(BinaryCrossEntropyLoss, ComputesCorrectAnswer) {
    srand(time(NULL));

    float labels[4096];
    float probabilities[4096];
    float loss[4096];
    float loss_cpu[4096];
    float gradient[4096];
    float gradient_cpu[4096];

    cudaError_t cudaStatus;

    float *labels_d;
    float *probabilities_d;
    float *workspace_d;
    float *loss_d;
    float *gradient_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&labels_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&probabilities_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&workspace_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&loss_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&gradient_d, 4096 * sizeof(float));
    assert(cudaStatus == cudaSuccess);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;

        for (uint32_t b = 0; b < batchSize; ++b) {
            labels[b] = rand() % 2;
            if (rand() % 50 == 0) {
                uint32_t sel = rand() % 5;
                if (sel == 0)
                    probabilities[b] = 0.0f;
                else if (sel == 1)
                    probabilities[b] = -0.0f;
                else if (sel == 2)
                    probabilities[b] = NAN;
                else if (sel == 3)
                    probabilities[b] = std::numeric_limits<float>::infinity();
                else if (sel == 4)
                    probabilities[b] = -std::numeric_limits<float>::infinity();
            } else {
                probabilities[b] = (rand() % 10000) / 9999.0f;
            }
        }

        cudaStatus = cudaMemcpyAsync(labels_d, labels, batchSize * sizeof(uint32_t), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(probabilities_d, probabilities, batchSize * sizeof(float), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        uint32_t lossScalingFactor = 1 + rand() % 4;
        launchElementWiseCrossEntropyLoss<uint32_t, float, float>(labels_d,
                                                                  probabilities_d,
                                                                  loss_d,
                                                                  gradient_d,
                                                                  100,  // ignored
                                                                  batchSize,
                                                                  true,
                                                                  lossScalingFactor,
                                                                  CrossEntropyLossType::BINARY,
                                                                  true,
                                                                  stream);

        cudaStatus = cudaMemcpyAsync(loss, loss_d, batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(gradient, gradient_d, batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (uint32_t b = 0; b < batchSize; ++b) {
            float probability = probabilities[b];
            float rawProbability = (!isfinite(probability) || isnan(probability)) ? 0.0f : probability;
            if (probability < MIN_PROBABILITY_FLOAT || (!isfinite(probability) || isnan(probability)))
                probability = MIN_PROBABILITY_FLOAT;
            loss_cpu[b] = -(labels[b] * logf(probability) + (1.0f - labels[b]) * (logf(1.0f - probability)));
            gradient_cpu[b] = (rawProbability - labels[b]) * lossScalingFactor;
            // printf("batch %d label %f prob %f loss %f gradient %f lossScalingFactor %d\n", b, labels[b], probability, loss_cpu[b],
            // gradient_cpu[b], lossScalingFactor);
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(loss[b] - loss_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("loss batchItem %d, %f %f\n", b, loss_cpu[b], loss[b]);
            }
            ASSERT_LT(diff, thresh);
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(gradient[b] - gradient_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("gradient batchItem %d, %f %f\n", b, gradient_cpu[b], gradient[b]);
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
    cudaStatus = cudaFree(gradient_d);
    assert(cudaStatus == cudaSuccess);
}

TEST(BinaryCrossEntropyLoss, ComputesCorrectAnswer_halfPrecision) {
    srand(time(NULL));

    half labels[4096];
    half probabilities[4096];
    half loss[4096];
    half loss_cpu[4096];
    half gradient[4096];
    half gradient_cpu[4096];

    cudaError_t cudaStatus;

    half *labels_d;
    half *probabilities_d;
    half *workspace_d;
    half *loss_d;
    half *gradient_d;

    Stream stream(0);

    cudaStatus = cudaMalloc(&labels_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&probabilities_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&workspace_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&loss_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);
    cudaStatus = cudaMalloc(&gradient_d, 4096 * sizeof(half));
    assert(cudaStatus == cudaSuccess);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;

        for (uint32_t b = 0; b < batchSize; ++b) {
            labels[b] = (half)(float)(rand() % 2);
            if (rand() % 50 == 0) {
                uint32_t sel = rand() % 5;
                if (sel == 0)
                    probabilities[b] = 0.0f;
                else if (sel == 1)
                    probabilities[b] = -0.0f;
                else if (sel == 2)
                    probabilities[b] = NAN;
                else if (sel == 3)
                    probabilities[b] = std::numeric_limits<float>::infinity();
                else if (sel == 4)
                    probabilities[b] = -std::numeric_limits<float>::infinity();
            } else {
                probabilities[b] = (rand() % 10000) / 9999.0f;
            }
        }

        cudaStatus = cudaMemcpyAsync(labels_d, labels, batchSize * sizeof(uint32_t), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(probabilities_d, probabilities, batchSize * sizeof(half), cudaMemcpyHostToDevice, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        uint32_t lossScalingFactor = 1 + rand() % 4;
        launchElementWiseCrossEntropyLoss<half, half, half>(labels_d,
                                                            probabilities_d,
                                                            loss_d,
                                                            gradient_d,
                                                            100,  // ignored
                                                            batchSize,
                                                            true,
                                                            lossScalingFactor,
                                                            CrossEntropyLossType::BINARY,
                                                            true,
                                                            stream);

        cudaStatus = cudaMemcpyAsync(loss, loss_d, batchSize * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpyAsync(gradient, gradient_d, batchSize * sizeof(half), cudaMemcpyDeviceToHost, stream.getStream());
        assert(cudaStatus == cudaSuccess);

        for (uint32_t b = 0; b < batchSize; ++b) {
            half probability = probabilities[b];
            half rawProbability = (!isfinite(probability) || isnan(probability)) ? (half)0.0f : probability;
            if (probability < MIN_PROBABILITY_HALF || (!isfinite(probability) || isnan(probability)))
                probability = MIN_PROBABILITY_HALF;
            loss_cpu[b] = -(labels[b] * logf(probability) + (1.0f - labels[b]) * (logf(1.0f - probability)));
            gradient_cpu[b] = (rawProbability - labels[b]) * lossScalingFactor;
            // printf("batch %d label %f prob %f loss %f gradient %f lossScalingFactor %d\n", b, (float)labels[b], (float)probability,
            // (float)loss_cpu[b], (float)gradient_cpu[b], lossScalingFactor);
        }

        cudaStatus = cudaStreamSynchronize(stream.getStream());
        assert(cudaStatus == cudaSuccess);

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(loss[b] - loss_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("loss batchItem %d, %f %f\n", b, (float)loss_cpu[b], (float)loss[b]);
            }
            ASSERT_LT(diff, thresh);
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(gradient[b] - gradient_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("gradient batchItem %d, %f %f\n", b, (float)gradient_cpu[b], (float)gradient[b]);
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
    cudaStatus = cudaFree(gradient_d);
    assert(cudaStatus == cudaSuccess);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
