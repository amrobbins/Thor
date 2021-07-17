#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>

using namespace std;
using namespace ThorImplementation;

TEST(ComputeCategoricalAccuracy, perClassLabelComputesCorrectly) {
    srand(time(NULL));

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 500) + 20;
        uint32_t numClasses = (rand() % 100) + 2;

        Tensor accuracy_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor accuracy_gpu_h = accuracy_d.clone(cpuPlacement);
        Tensor predictions_h(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor predictions_d = predictions_h.clone(gpuPlacement);
        Tensor labels_h = predictions_h.clone();
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor workspace_d(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));

        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            for (uint32_t curClass = 0; curClass < numClasses; ++curClass) {
                ((float *)predictions_h.getMemPtr())[batch * numClasses + curClass] = (rand() % 100000000) / 100000000.0f;
                ((float *)labels_h.getMemPtr())[batch * numClasses + curClass] = (rand() % 100000000) / 100000000.0f;
            }
        }

        Stream stream(0);
        predictions_d.copyFromAsync(predictions_h, stream);
        labels_d.copyFromAsync(labels_h, stream);

        launchComputeCategoricalAccuracy_perClassLabels((float *)accuracy_d.getMemPtr(),
                                                        (float *)predictions_d.getMemPtr(),
                                                        (float *)labels_d.getMemPtr(),
                                                        (uint8_t *)workspace_d.getMemPtr(),
                                                        numClasses,
                                                        batchSize,
                                                        stream);

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);

        bool duplicateMaxPrediction = false;
        bool duplicateMaxLabel = false;
        uint32_t possibleDuplicates = 0;
        float *predictionsMem = (float *)predictions_h.getMemPtr();
        float *labelsMem = (float *)labels_h.getMemPtr();
        uint32_t numCorrect = 0;
        for (uint32_t batchItem = 0; batchItem < batchSize; ++batchItem) {
            float maxPrediction = predictionsMem[batchItem * numClasses + 0];
            uint32_t maxPredictionClass = 0;
            float maxLabel = labelsMem[batchItem * numClasses + 0];
            uint32_t maxLabelClass = 0;
            for (uint32_t classNum = 1; classNum < numClasses; ++classNum) {
                if (predictionsMem[batchItem * numClasses + classNum] > maxPrediction) {
                    maxPrediction = predictionsMem[batchItem * numClasses + classNum];
                    maxPredictionClass = classNum;
                    duplicateMaxPrediction = false;
                } else if (predictionsMem[batchItem * numClasses + classNum] == maxPrediction) {
                    duplicateMaxPrediction = true;
                }
                if (labelsMem[batchItem * numClasses + classNum] > maxLabel) {
                    maxLabel = labelsMem[batchItem * numClasses + classNum];
                    maxLabelClass = classNum;
                    duplicateMaxLabel = false;
                } else if (labelsMem[batchItem * numClasses + classNum] == maxLabel) {
                    duplicateMaxLabel = true;
                }
            }
            if (maxPredictionClass == maxLabelClass) {
                numCorrect += 1;
            }
            if (duplicateMaxPrediction || duplicateMaxLabel) {
                possibleDuplicates += 1;
            }
        }
        float accuracy = numCorrect / (double)batchSize;

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);
        stream.synchronize();
        float actual = ((float *)accuracy_gpu_h.getMemPtr())[0];

        float thresh = (actual / 100.0f) + (possibleDuplicates / (double)batchSize);
        ASSERT_LE(abs(accuracy - actual), thresh);
    }
}

TEST(ComputeCategoricalAccuracy, classIndexLabelComputesCorrectly) {
    srand(time(NULL));

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 500) + 20;
        uint32_t numClasses = (rand() % 100) + 2;

        Tensor accuracy_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor accuracy_gpu_h = accuracy_d.clone(cpuPlacement);
        Tensor predictions_h(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        Tensor predictions_d = predictions_h.clone(gpuPlacement);
        Tensor labels_h(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT32, {batchSize, 1}));
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor workspace_d(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));

        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            for (uint32_t curClass = 0; curClass < numClasses; ++curClass) {
                ((float *)predictions_h.getMemPtr())[batch * numClasses + curClass] = (rand() % 100000000) / 100000000.0f;
            }
            ((uint32_t *)labels_h.getMemPtr())[batch] = rand() % numClasses;
        }

        Stream stream(0);
        predictions_d.copyFromAsync(predictions_h, stream);
        labels_d.copyFromAsync(labels_h, stream);

        launchComputeCategoricalAccuracy_classIndexLabels((float *)accuracy_d.getMemPtr(),
                                                          (float *)predictions_d.getMemPtr(),
                                                          (uint32_t *)labels_d.getMemPtr(),
                                                          (uint8_t *)workspace_d.getMemPtr(),
                                                          numClasses,
                                                          batchSize,
                                                          stream);

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);

        bool duplicateMaxPrediction = false;
        uint32_t possibleDuplicates = 0;
        float *predictionsMem = (float *)predictions_h.getMemPtr();
        uint32_t numCorrect = 0;
        for (uint32_t batchItem = 0; batchItem < batchSize; ++batchItem) {
            float maxPrediction = predictionsMem[batchItem * numClasses + 0];
            uint32_t maxPredictionClass = 0;
            for (uint32_t classNum = 1; classNum < numClasses; ++classNum) {
                if (predictionsMem[batchItem * numClasses + classNum] > maxPrediction) {
                    maxPrediction = predictionsMem[batchItem * numClasses + classNum];
                    maxPredictionClass = classNum;
                    duplicateMaxPrediction = false;
                } else if (predictionsMem[batchItem * numClasses + classNum] == maxPrediction) {
                    duplicateMaxPrediction = true;
                }
            }
            if (maxPredictionClass == ((uint32_t *)labels_h.getMemPtr())[batchItem]) {
                numCorrect += 1;
            }
            if (duplicateMaxPrediction) {
                possibleDuplicates += 1;
            }
        }
        float accuracy = numCorrect / (double)batchSize;

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);
        stream.synchronize();
        float actual = ((float *)accuracy_gpu_h.getMemPtr())[0];

        float thresh = (actual / 100.0f) + (possibleDuplicates / (double)batchSize);
        ASSERT_LE(abs(accuracy - actual), thresh);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
