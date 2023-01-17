#include "Thor.h"

#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

using namespace ThorImplementation;
using namespace std;

const float MIN_PROBABILITY_FLOAT = 0.000000000000000000000000000000000001f;
const half MIN_PROBABILITY_HALF = 0.000062f;

TEST(CategoricalCrossEntropyLoss, ComputesCorrectAnswer_categoricalOneHotLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    float *labels;
    float *probabilities;
    float *loss;
    float *loss_cpu;
    float *gradient;
    float *gradient_cpu;

    float *labels_d;
    float *probabilities_d;
    float *loss_d;
    float *gradient_d;

    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;
        uint32_t numClasses = (rand() % 512) + 1;

        Tensor labelsT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor labelsT_d = labelsT.clone(gpuPlacement);
        Tensor probabilitiesT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor probabilitiesT_d = probabilitiesT.clone(gpuPlacement);
        Tensor lossT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor lossT_d = lossT.clone(gpuPlacement);
        Tensor lossT_cpu = lossT.clone();
        Tensor gradientT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor gradientT_d = gradientT.clone(gpuPlacement);
        Tensor gradientT_cpu = gradientT.clone();

        labels = (float *)labelsT.getMemPtr();
        labels_d = (float *)labelsT_d.getMemPtr();
        probabilities = (float *)probabilitiesT.getMemPtr();
        probabilities_d = (float *)probabilitiesT_d.getMemPtr();
        loss = (float *)lossT.getMemPtr();
        loss_d = (float *)lossT_d.getMemPtr();
        loss_cpu = (float *)lossT_cpu.getMemPtr();
        gradient = (float *)gradientT.getMemPtr();
        gradient_d = (float *)gradientT_d.getMemPtr();
        gradient_cpu = (float *)gradientT_cpu.getMemPtr();

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

        labelsT_d.copyFromAsync(labelsT, stream);
        probabilitiesT_d.copyFromAsync(probabilitiesT, stream);

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

        lossT.copyFromAsync(lossT_d, stream);
        gradientT.copyFromAsync(gradientT_d, stream);

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float probability = probabilities[i + b * numClasses];
                if (!isfinite(probability) || isnan(probability))
                    probability = 0.0f;
                float rawProbability = probability;
                if (probability < MIN_PROBABILITY_FLOAT)
                    probability = MIN_PROBABILITY_FLOAT;
                loss_cpu[b * numClasses + i] = -labels[i + b * numClasses] * logf(probability);
                gradient_cpu[b * numClasses + i] = (rawProbability - labels[i + b * numClasses]) * lossScalingFactor;
            }
        }

        stream.synchronize();

        float thresh = 0.001;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(loss[b * numClasses + i] - loss_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf("loss batchItem %d class %d, %f %f\n", b, i, loss_cpu[b * numClasses + i], loss[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
        }

        thresh = 0.00001;
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
}

TEST(CategoricalCrossEntropyLoss, ComputesCorrectAnswer_categoricalOneHotLabels_halfPrecision) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    half *labels;
    half *probabilities;
    half *loss;
    half *loss_cpu;
    half *gradient;
    half *gradient_cpu;

    half *labels_d;
    half *probabilities_d;
    half *loss_d;
    half *gradient_d;

    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 8) + 1;
        uint32_t numClasses = (rand() % 512) + 1;

        Tensor labelsT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor labelsT_d = labelsT.clone(gpuPlacement);
        Tensor probabilitiesT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor probabilitiesT_d = probabilitiesT.clone(gpuPlacement);
        Tensor lossT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor lossT_d = lossT.clone(gpuPlacement);
        Tensor lossT_cpu = lossT.clone();
        Tensor gradientT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor gradientT_d = gradientT.clone(gpuPlacement);
        Tensor gradientT_cpu = gradientT.clone();

        labels = (half *)labelsT.getMemPtr();
        labels_d = (half *)labelsT_d.getMemPtr();
        probabilities = (half *)probabilitiesT.getMemPtr();
        probabilities_d = (half *)probabilitiesT_d.getMemPtr();
        loss = (half *)lossT.getMemPtr();
        loss_d = (half *)lossT_d.getMemPtr();
        loss_cpu = (half *)lossT_cpu.getMemPtr();
        gradient = (half *)gradientT.getMemPtr();
        gradient_d = (half *)gradientT_d.getMemPtr();
        gradient_cpu = (half *)gradientT_cpu.getMemPtr();

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
                            probabilities[i + b * numClasses] = std::numeric_limits<half>::infinity();
                        else if (sel == 4)
                            probabilities[i + b * numClasses] = -std::numeric_limits<half>::infinity();
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

        labelsT_d.copyFromAsync(labelsT, stream);
        probabilitiesT_d.copyFromAsync(probabilitiesT, stream);

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

        lossT.copyFromAsync(lossT_d, stream);
        gradientT.copyFromAsync(gradientT_d, stream);

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                half probability = probabilities[i + b * numClasses];
                if (!isfinite(probability) || isnan(probability))
                    probability = 0.0f;
                float rawProbability = probability;
                if (probability < MIN_PROBABILITY_HALF)
                    probability = MIN_PROBABILITY_HALF;
                loss_cpu[b * numClasses + i] = -labels[i + b * numClasses] * logf(probability);
                gradient_cpu[b * numClasses + i] = (rawProbability - labels[i + b * numClasses]) * lossScalingFactor;
            }
        }

        stream.synchronize();

        float thresh = 0.005;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t i = 0; i < numClasses; ++i) {
                float diff = abs(loss[b * numClasses + i] - loss_cpu[b * numClasses + i]);
                if (diff >= thresh || !isfinite(diff)) {
                    printf(
                        "loss batchItem %d class %d, %f %f\n", b, i, (float)loss_cpu[b * numClasses + i], (float)loss[b * numClasses + i]);
                }
                ASSERT_LT(diff, thresh);
            }
        }

        thresh = 0.00001;
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
}

TEST(CategoricalCrossEntropyLoss, ComputesCorrectAnswer_classIndexLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    uint32_t *labels;
    float *probabilities;
    float *loss;
    float *loss_cpu;
    float *gradient;
    float *gradient_cpu;

    uint32_t *labels_d;
    float *probabilities_d;
    float *loss_d;
    float *gradient_d;

    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 400) + 1;
        uint32_t numClasses = (rand() % 1500) + 2;

        Tensor labelsT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT32, {batchSize, 1}));
        Tensor labelsT_d = labelsT.clone(gpuPlacement);
        Tensor probabilitiesT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor probabilitiesT_d = probabilitiesT.clone(gpuPlacement);
        Tensor lossT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor lossT_d = lossT.clone(gpuPlacement);
        Tensor lossT_cpu = lossT.clone();
        Tensor gradientT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor gradientT_d = gradientT.clone(gpuPlacement);
        Tensor gradientT_cpu = gradientT.clone();

        labels = (uint32_t *)labelsT.getMemPtr();
        labels_d = (uint32_t *)labelsT_d.getMemPtr();
        probabilities = (float *)probabilitiesT.getMemPtr();
        probabilities_d = (float *)probabilitiesT_d.getMemPtr();
        loss = (float *)lossT.getMemPtr();
        loss_d = (float *)lossT_d.getMemPtr();
        loss_cpu = (float *)lossT_cpu.getMemPtr();
        gradient = (float *)gradientT.getMemPtr();
        gradient_d = (float *)gradientT_d.getMemPtr();
        gradient_cpu = (float *)gradientT_cpu.getMemPtr();

        for (uint32_t b = 0; b < batchSize; ++b) {
            double totalProb = 0.0;
            while (totalProb < 0.00001) {
                totalProb = 0.0;
                if (rand() % 50 == 0) {
                    labels[b] = 10000 + (rand() % 1000000);
                } else {
                    labels[b] = rand() % numClasses;
                }
                for (uint32_t c = 0; c < numClasses; ++c) {
                    if (rand() % 50 == 0) {
                        uint32_t sel = rand() % 5;
                        if (sel == 0)
                            probabilities[c + b * numClasses] = 0.0f;
                        else if (sel == 1)
                            probabilities[c + b * numClasses] = -0.0f;
                        else if (sel == 2)
                            probabilities[c + b * numClasses] = NAN;
                        else if (sel == 3)
                            probabilities[c + b * numClasses] = std::numeric_limits<float>::infinity();
                        else if (sel == 4)
                            probabilities[c + b * numClasses] = -std::numeric_limits<float>::infinity();
                    } else {
                        probabilities[c + b * numClasses] = rand() % 10000;
                        totalProb += probabilities[c + b * numClasses];
                    }
                }
            }
            for (uint32_t c = 0; c < numClasses; ++c) {
                probabilities[c + b * numClasses] /= totalProb;
            }
        }

        labelsT_d.copyFromAsync(labelsT, stream);
        probabilitiesT_d.copyFromAsync(probabilitiesT, stream);

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

        lossT.copyFromAsync(lossT_d, stream);
        gradientT.copyFromAsync(gradientT_d, stream);

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                float probability = probabilities[c + b * numClasses];
                float clampedProbability = (!isfinite(probability) || isnan(probability)) ? 0.0f : probability;
                if (probability < MIN_PROBABILITY_FLOAT || (!isfinite(probability) || isnan(probability)))
                    probability = MIN_PROBABILITY_FLOAT;
                if (labels[b] == c) {
                    loss_cpu[b * numClasses + c] = -logf(probability);
                    gradient_cpu[b * numClasses + c] = (clampedProbability - 1) * lossScalingFactor;
                } else {
                    loss_cpu[b * numClasses + c] = 0.0f;
                    gradient_cpu[b * numClasses + c] = clampedProbability * lossScalingFactor;
                }
            }
        }

        stream.synchronize();

        float thresh = 0.001;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                float diff = abs(loss[b * numClasses + c] - loss_cpu[b * numClasses + c]);
                if (diff >= thresh || !isfinite(diff)) {
                    uint32_t e = b * numClasses + c;
                    printf("loss batchItem %d class %d label %d, %f %f\n", b, c, labels[b], loss_cpu[e], loss[e]);
                }
                ASSERT_LT(diff, thresh);
            }
        }

        thresh = 0.00001f;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                float diff = abs(gradient[b * numClasses + c] - gradient_cpu[b * numClasses + c]);
                if (diff >= thresh || !isfinite(diff)) {
                    uint32_t e = b * numClasses + c;
                    printf("gradient batchItem %d class %d label %d probability %f scalingFactor %d, %f %f\n",
                           b,
                           c,
                           labels[b],
                           probabilities[e],
                           lossScalingFactor,
                           gradient_cpu[e],
                           gradient[e]);
                }
                ASSERT_LT(diff, thresh);
            }
        }
    }
}

TEST(CategoricalCrossEntropyLoss, ComputesCorrectAnswer_classIndexLabels_halfPrecision) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    uint16_t *labels;
    half *probabilities;
    half *loss;
    half *loss_cpu;
    half *gradient;
    half *gradient_cpu;

    uint16_t *labels_d;
    half *probabilities_d;
    half *loss_d;
    half *gradient_d;

    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 400) + 1;
        uint32_t numClasses = (rand() % 1500) + 2;

        Tensor labelsT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT16, {batchSize, 1}));
        Tensor labelsT_d = labelsT.clone(gpuPlacement);
        Tensor probabilitiesT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numClasses}));
        Tensor probabilitiesT_d = probabilitiesT.clone(gpuPlacement);
        Tensor lossT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numClasses}));
        Tensor lossT_d = lossT.clone(gpuPlacement);
        Tensor lossT_cpu = lossT.clone();
        Tensor gradientT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numClasses}));
        Tensor gradientT_d = gradientT.clone(gpuPlacement);
        Tensor gradientT_cpu = gradientT.clone();

        labels = (uint16_t *)labelsT.getMemPtr();
        labels_d = (uint16_t *)labelsT_d.getMemPtr();
        probabilities = (half *)probabilitiesT.getMemPtr();
        probabilities_d = (half *)probabilitiesT_d.getMemPtr();
        loss = (half *)lossT.getMemPtr();
        loss_d = (half *)lossT_d.getMemPtr();
        loss_cpu = (half *)lossT_cpu.getMemPtr();
        gradient = (half *)gradientT.getMemPtr();
        gradient_d = (half *)gradientT_d.getMemPtr();
        gradient_cpu = (half *)gradientT_cpu.getMemPtr();

        for (uint32_t b = 0; b < batchSize; ++b) {
            double totalProb = 0.0;
            while (totalProb < 0.00001) {
                totalProb = 0.0;
                if (rand() % 50 == 0) {
                    labels[b] = 10000 + (rand() % 1000000);
                } else {
                    labels[b] = rand() % numClasses;
                }
                for (uint32_t c = 0; c < numClasses; ++c) {
                    if (rand() % 50 == 0) {
                        uint32_t sel = rand() % 5;
                        if (sel == 0)
                            probabilities[c + b * numClasses] = 0.0f;
                        else if (sel == 1)
                            probabilities[c + b * numClasses] = -0.0f;
                        else if (sel == 2)
                            probabilities[c + b * numClasses] = NAN;
                        else if (sel == 3)
                            probabilities[c + b * numClasses] = std::numeric_limits<half>::infinity();
                        else if (sel == 4)
                            probabilities[c + b * numClasses] = -std::numeric_limits<half>::infinity();
                    } else {
                        probabilities[c + b * numClasses] = (float)(rand() % 10000);
                        totalProb += probabilities[c + b * numClasses];
                    }
                }
            }
            for (uint32_t c = 0; c < numClasses; ++c) {
                probabilities[c + b * numClasses] = probabilities[c + b * numClasses] / totalProb;
            }
        }

        labelsT_d.copyFromAsync(labelsT, stream);
        probabilitiesT_d.copyFromAsync(probabilitiesT, stream);

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

        lossT.copyFromAsync(lossT_d, stream);
        gradientT.copyFromAsync(gradientT_d, stream);

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                half probability = probabilities[c + b * numClasses];
                half clampedProbability = (!isfinite(probability) || isnan(probability)) ? 0.0f : (float)probability;
                if (probability < MIN_PROBABILITY_HALF || (!isfinite(probability) || isnan(probability)))
                    probability = MIN_PROBABILITY_HALF;
                if (labels[b] == c) {
                    loss_cpu[b * numClasses + c] = -logf(probability);
                    gradient_cpu[b * numClasses + c] = (clampedProbability - 1) * lossScalingFactor;
                } else {
                    loss_cpu[b * numClasses + c] = 0.0f;
                    gradient_cpu[b * numClasses + c] = clampedProbability * lossScalingFactor;
                }
            }
        }

        stream.synchronize();

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                float diff = abs(loss[b * numClasses + c] - loss_cpu[b * numClasses + c]);
                if (diff >= thresh || !isfinite(diff)) {
                    uint32_t e = b * numClasses + c;
                    printf("loss batchItem %d class %d label %d, %f %f\n", b, c, (uint32_t)labels[b], (float)loss_cpu[e], (float)loss[e]);
                }
                ASSERT_LT(diff, thresh);
            }
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                float diff = abs(gradient[b * numClasses + c] - gradient_cpu[b * numClasses + c]);
                thresh = max(0.0001, abs(gradient_cpu[b * numClasses + c] * 0.003));
                if (diff >= thresh || !isfinite(diff)) {
                    uint32_t e = b * numClasses + c;
                    printf("gradient batchItem %d class %d label %d probability %f scalingFactor %d, %f %f\n",
                           b,
                           c,
                           (uint32_t)labels[b],
                           (float)probabilities[e],
                           lossScalingFactor,
                           (float)gradient_cpu[e],
                           (float)gradient[e]);
                }
                ASSERT_LT(diff, thresh);
            }
        }
    }
}

TEST(BinaryCrossEntropyLoss, ComputesCorrectAnswer) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    uint32_t *labels;
    float *probabilities;
    float *loss;
    float *loss_cpu;
    float *gradient;
    float *gradient_cpu;

    uint32_t *labels_d;
    float *probabilities_d;
    float *loss_d;
    float *gradient_d;

    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 5000) + 1;

        Tensor labelsT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT32, {batchSize, 1}));
        Tensor labelsT_d = labelsT.clone(gpuPlacement);
        Tensor probabilitiesT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, 1}));
        Tensor probabilitiesT_d = probabilitiesT.clone(gpuPlacement);
        Tensor lossT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, 1}));
        Tensor lossT_d = lossT.clone(gpuPlacement);
        Tensor lossT_cpu = lossT.clone();
        Tensor gradientT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, 1}));
        Tensor gradientT_d = gradientT.clone(gpuPlacement);
        Tensor gradientT_cpu = gradientT.clone();

        labels = (uint32_t *)labelsT.getMemPtr();
        labels_d = (uint32_t *)labelsT_d.getMemPtr();
        probabilities = (float *)probabilitiesT.getMemPtr();
        probabilities_d = (float *)probabilitiesT_d.getMemPtr();
        loss = (float *)lossT.getMemPtr();
        loss_d = (float *)lossT_d.getMemPtr();
        loss_cpu = (float *)lossT_cpu.getMemPtr();
        gradient = (float *)gradientT.getMemPtr();
        gradient_d = (float *)gradientT_d.getMemPtr();
        gradient_cpu = (float *)gradientT_cpu.getMemPtr();

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
                probabilities[b] = (rand() % 1000) / 999.0f;
            }
        }

        labelsT_d.copyFromAsync(labelsT, stream);
        probabilitiesT_d.copyFromAsync(probabilitiesT, stream);

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

        lossT.copyFromAsync(lossT_d, stream);
        gradientT.copyFromAsync(gradientT_d, stream);

        for (uint32_t b = 0; b < batchSize; ++b) {
            float probability = probabilities[b];
            if (!isfinite(probability) || isnan(probability))
                probability = 0.0f;
            float trueClampedProbability = probability;
            if (probability < MIN_PROBABILITY_FLOAT)
                trueClampedProbability = MIN_PROBABILITY_FLOAT;
            float falseClampedProbability = probability;
            if (1.0 - probability < MIN_PROBABILITY_FLOAT)
                falseClampedProbability = 0.9999999;
            loss_cpu[b] = -(labels[b] * logf(trueClampedProbability) + (1.0f - labels[b]) * (logf(1.0f - falseClampedProbability)));
            if (!isfinite(loss_cpu[b]))
                printf("INF label %d, prob %f\n", labels[b], probability);
            gradient_cpu[b] = (probability - labels[b]) * lossScalingFactor;
        }

        stream.synchronize();

        float thresh = 0.001;
        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(loss[b] - loss_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("loss batchItem %d scalingFactor %d, %f %f\n", b, lossScalingFactor, loss_cpu[b], loss[b]);
            }
            ASSERT_LT(diff, thresh);
        }

        thresh = 0.00001f;
        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(gradient[b] - gradient_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("gradient batchItem %d scalingFactor %d, %f %f\n", b, lossScalingFactor, gradient_cpu[b], gradient[b]);
            }
            ASSERT_LT(diff, thresh);
        }
    }
}

TEST(BinaryCrossEntropyLoss, ComputesCorrectAnswer_halfPrecision) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    uint16_t *labels;
    half *probabilities;
    half *loss;
    half *loss_cpu;
    half *gradient;
    half *gradient_cpu;

    uint16_t *labels_d;
    half *probabilities_d;
    half *loss_d;
    half *gradient_d;

    Stream stream(0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 5000) + 1;

        Tensor labelsT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT16, {batchSize, 1}));
        Tensor labelsT_d = labelsT.clone(gpuPlacement);
        Tensor probabilitiesT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, 1}));
        Tensor probabilitiesT_d = probabilitiesT.clone(gpuPlacement);
        Tensor lossT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, 1}));
        Tensor lossT_d = lossT.clone(gpuPlacement);
        Tensor lossT_cpu = lossT.clone();
        Tensor gradientT(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, 1}));
        Tensor gradientT_d = gradientT.clone(gpuPlacement);
        Tensor gradientT_cpu = gradientT.clone();

        labels = (uint16_t *)labelsT.getMemPtr();
        labels_d = (uint16_t *)labelsT_d.getMemPtr();
        probabilities = (half *)probabilitiesT.getMemPtr();
        probabilities_d = (half *)probabilitiesT_d.getMemPtr();
        loss = (half *)lossT.getMemPtr();
        loss_d = (half *)lossT_d.getMemPtr();
        loss_cpu = (half *)lossT_cpu.getMemPtr();
        gradient = (half *)gradientT.getMemPtr();
        gradient_d = (half *)gradientT_d.getMemPtr();
        gradient_cpu = (half *)gradientT_cpu.getMemPtr();

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
                    probabilities[b] = std::numeric_limits<half>::infinity();
                else if (sel == 4)
                    probabilities[b] = -std::numeric_limits<half>::infinity();
            } else {
                probabilities[b] = (rand() % 1000) / 999.0f;
            }
        }

        labelsT_d.copyFromAsync(labelsT, stream);
        probabilitiesT_d.copyFromAsync(probabilitiesT, stream);

        uint32_t lossScalingFactor = 1 + rand() % 4;
        launchElementWiseCrossEntropyLoss<uint16_t, half, half>(labels_d,
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

        lossT.copyFromAsync(lossT_d, stream);
        gradientT.copyFromAsync(gradientT_d, stream);

        for (uint32_t b = 0; b < batchSize; ++b) {
            half probability = probabilities[b];
            if (!isfinite(probability) || isnan(probability))
                probability = 0.0f;
            half trueClampedProbability = probability;
            if (probability < MIN_PROBABILITY_HALF)
                trueClampedProbability = MIN_PROBABILITY_HALF;
            half falseClampedProbability = probability;
            if (1.0 - probability < MIN_PROBABILITY_HALF)
                falseClampedProbability = 0.9995;
            loss_cpu[b] = -(labels[b] * logf(trueClampedProbability) + (1.0f - labels[b]) * (logf(1.0f - falseClampedProbability)));
            if (!isfinite(loss_cpu[b]))
                printf("INF label %d, prob %f\n", labels[b], (float)probability);
            gradient_cpu[b] = (probability - labels[b]) * lossScalingFactor;
        }

        stream.synchronize();

        float thresh = 0.01;
        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(loss[b] - loss_cpu[b]);
            if (diff >= thresh || !isfinite(diff)) {
                printf("loss batchItem %d scalingFactor %d, %f %f\n", b, lossScalingFactor, (float)loss_cpu[b], (float)loss[b]);
            }
            ASSERT_LT(diff, thresh);
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            float diff = abs(gradient[b] - gradient_cpu[b]);
            thresh = max(0.0001, abs(gradient_cpu[b] * 0.003));
            if (diff >= thresh || !isfinite(diff)) {
                printf("gradient batchItem %d scalingFactor %d, %f %f\n", b, lossScalingFactor, (float)gradient_cpu[b], (float)gradient[b]);
            }
            ASSERT_LT(diff, thresh);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
