#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>

using namespace std;
using namespace ThorImplementation;

TEST(ComputeBinaryAccuracy, computesCorrectly21) {
    srand(time(NULL));

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    TensorDescriptor::DataType predictionsDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType labelsDataType = TensorDescriptor::DataType::UINT8;

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 3500) + 20;

        Tensor accuracy_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor accuracy_gpu_h = accuracy_d.clone(cpuPlacement);
        Tensor predictions_h(cpuPlacement, TensorDescriptor(predictionsDataType, {batchSize, 1}));
        Tensor predictions_d = predictions_h.clone(gpuPlacement);
        Tensor labels_h(cpuPlacement, TensorDescriptor(labelsDataType, {batchSize, 1}));
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor workspace_d(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize}));

        uint32_t numCorrect = 0;
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            float prediction = (rand() % 100000) / 100000.0f;
            uint8_t label = (float)(rand() % 2);
            ((half *)predictions_h.getMemPtr())[batch] = prediction;
            ((uint8_t *)labels_h.getMemPtr())[batch] = label;
            if (label == (prediction >= 0.5f))
                numCorrect += 1;
        }
        float expected = numCorrect / (float)batchSize;

        Stream stream(0);
        predictions_d.copyFromAsync(predictions_h, stream);
        labels_d.copyFromAsync(labels_h, stream);

        shared_ptr<BatchReduce> batchReduce = createBinaryAccuracyBatchReduce(batchSize, stream);

        launchComputeBinaryAccuracy<half, uint8_t>(
            accuracy_d, (half *)predictions_d.getMemPtr(), (uint8_t *)labels_d.getMemPtr(), workspace_d, batchSize, batchReduce, stream);

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);
        stream.synchronize();
        float actual = ((float *)accuracy_gpu_h.getMemPtr())[0];
        float thresh = 0.01;
        if (abs(expected - actual) >= thresh)
            printf("expected %f actual %f\n", expected, actual);
        ASSERT_LT(abs(expected - actual), thresh);
    }
}

TEST(ComputeBinaryAccuracy, computesCorrectly22) {
    srand(time(NULL));

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    TensorDescriptor::DataType predictionsDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType labelsDataType = TensorDescriptor::DataType::FP16;

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 3500) + 20;

        Tensor accuracy_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor accuracy_gpu_h = accuracy_d.clone(cpuPlacement);
        Tensor predictions_h(cpuPlacement, TensorDescriptor(predictionsDataType, {batchSize, 1}));
        Tensor predictions_d = predictions_h.clone(gpuPlacement);
        Tensor labels_h(cpuPlacement, TensorDescriptor(labelsDataType, {batchSize, 1}));
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor workspace_d(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize}));

        uint32_t numCorrect = 0;
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            float prediction = (rand() % 100000) / 100000.0f;
            half label = (float)(rand() % 2);
            ((half *)predictions_h.getMemPtr())[batch] = prediction;
            ((half *)labels_h.getMemPtr())[batch] = label;
            if (label == (prediction >= 0.5f))
                numCorrect += 1;
        }
        float expected = numCorrect / (float)batchSize;

        Stream stream(0);
        predictions_d.copyFromAsync(predictions_h, stream);
        labels_d.copyFromAsync(labels_h, stream);

        shared_ptr<BatchReduce> batchReduce = createBinaryAccuracyBatchReduce(batchSize, stream);

        launchComputeBinaryAccuracy<half, half>(
            accuracy_d, (half *)predictions_d.getMemPtr(), (half *)labels_d.getMemPtr(), workspace_d, batchSize, batchReduce, stream);

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);
        stream.synchronize();
        float actual = ((float *)accuracy_gpu_h.getMemPtr())[0];
        float thresh = 0.004;
        if (abs(expected - actual) >= thresh)
            printf("expected %f actual %f\n", expected, actual);
        ASSERT_LT(abs(expected - actual), thresh);
    }
}

TEST(ComputeBinaryAccuracy, computesCorrectly24) {
    srand(time(NULL));

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    TensorDescriptor::DataType predictionsDataType = TensorDescriptor::DataType::FP16;
    TensorDescriptor::DataType labelsDataType = TensorDescriptor::DataType::FP32;

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 3500) + 20;

        Tensor accuracy_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor accuracy_gpu_h = accuracy_d.clone(cpuPlacement);
        Tensor predictions_h(cpuPlacement, TensorDescriptor(predictionsDataType, {batchSize, 1}));
        Tensor predictions_d = predictions_h.clone(gpuPlacement);
        Tensor labels_h(cpuPlacement, TensorDescriptor(labelsDataType, {batchSize, 1}));
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor workspace_d(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize}));

        uint32_t numCorrect = 0;
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            float prediction = (rand() % 100000) / 100000.0f;
            float label = rand() % 2;
            ((half *)predictions_h.getMemPtr())[batch] = prediction;
            ((float *)labels_h.getMemPtr())[batch] = label;
            if (label == (prediction >= 0.5f))
                numCorrect += 1;
        }
        float expected = numCorrect / (float)batchSize;

        Stream stream(0);
        predictions_d.copyFromAsync(predictions_h, stream);
        labels_d.copyFromAsync(labels_h, stream);

        shared_ptr<BatchReduce> batchReduce = createBinaryAccuracyBatchReduce(batchSize, stream);

        launchComputeBinaryAccuracy<half, float>(
            accuracy_d, (half *)predictions_d.getMemPtr(), (float *)labels_d.getMemPtr(), workspace_d, batchSize, batchReduce, stream);

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);
        stream.synchronize();
        float actual = ((float *)accuracy_gpu_h.getMemPtr())[0];
        float thresh = 0.005;
        if (abs(expected - actual) >= thresh)
            printf("expected %f actual %f\n", expected, actual);
        ASSERT_LT(abs(expected - actual), thresh);
    }
}

TEST(ComputeBinaryAccuracy, computesCorrectly41) {
    srand(time(NULL));

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    TensorDescriptor::DataType predictionsDataType = TensorDescriptor::DataType::FP32;
    TensorDescriptor::DataType labelsDataType = TensorDescriptor::DataType::INT8;

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 3500) + 20;

        Tensor accuracy_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor accuracy_gpu_h = accuracy_d.clone(cpuPlacement);
        Tensor predictions_h(cpuPlacement, TensorDescriptor(predictionsDataType, {batchSize, 1}));
        Tensor predictions_d = predictions_h.clone(gpuPlacement);
        Tensor labels_h(cpuPlacement, TensorDescriptor(labelsDataType, {batchSize, 1}));
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor workspace_d(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize}));

        uint32_t numCorrect = 0;
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            float prediction = (rand() % 100000) / 100000.0f;
            int8_t label = (float)(rand() % 2);
            ((float *)predictions_h.getMemPtr())[batch] = prediction;
            ((int8_t *)labels_h.getMemPtr())[batch] = label;
            if (label == (prediction >= 0.5f))
                numCorrect += 1;
        }
        float expected = numCorrect / (float)batchSize;

        Stream stream(0);
        predictions_d.copyFromAsync(predictions_h, stream);
        labels_d.copyFromAsync(labels_h, stream);

        shared_ptr<BatchReduce> batchReduce = createBinaryAccuracyBatchReduce(batchSize, stream);

        launchComputeBinaryAccuracy<float, int8_t>(
            accuracy_d, (float *)predictions_d.getMemPtr(), (int8_t *)labels_d.getMemPtr(), workspace_d, batchSize, batchReduce, stream);

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);
        stream.synchronize();
        float actual = ((float *)accuracy_gpu_h.getMemPtr())[0];
        float thresh = 0.00001;
        if (abs(expected - actual) >= thresh)
            printf("expected %f actual %f\n", expected, actual);
        ASSERT_LT(abs(expected - actual), thresh);
    }
}

TEST(ComputeBinaryAccuracy, computesCorrectly42) {
    srand(time(NULL));

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    TensorDescriptor::DataType predictionsDataType = TensorDescriptor::DataType::FP32;
    TensorDescriptor::DataType labelsDataType = TensorDescriptor::DataType::FP16;

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 3500) + 20;

        Tensor accuracy_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor accuracy_gpu_h = accuracy_d.clone(cpuPlacement);
        Tensor predictions_h(cpuPlacement, TensorDescriptor(predictionsDataType, {batchSize, 1}));
        Tensor predictions_d = predictions_h.clone(gpuPlacement);
        Tensor labels_h(cpuPlacement, TensorDescriptor(labelsDataType, {batchSize, 1}));
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor workspace_d(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize}));

        uint32_t numCorrect = 0;
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            float prediction = (rand() % 100000) / 100000.0f;
            half label = (float)(rand() % 2);
            ((float *)predictions_h.getMemPtr())[batch] = prediction;
            ((half *)labels_h.getMemPtr())[batch] = label;
            if (label == (prediction >= 0.5f))
                numCorrect += 1;
        }
        float expected = numCorrect / (float)batchSize;

        Stream stream(0);
        predictions_d.copyFromAsync(predictions_h, stream);
        labels_d.copyFromAsync(labels_h, stream);

        shared_ptr<BatchReduce> batchReduce = createBinaryAccuracyBatchReduce(batchSize, stream);

        launchComputeBinaryAccuracy<float, half>(
            accuracy_d, (float *)predictions_d.getMemPtr(), (half *)labels_d.getMemPtr(), workspace_d, batchSize, batchReduce, stream);

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);
        stream.synchronize();
        float actual = ((float *)accuracy_gpu_h.getMemPtr())[0];
        float thresh = 0.00001;
        if (abs(expected - actual) >= thresh)
            printf("expected %f actual %f\n", expected, actual);
        ASSERT_LT(abs(expected - actual), thresh);
    }
}

TEST(ComputeBinaryAccuracy, computesCorrectly44) {
    srand(time(NULL));

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);

    TensorDescriptor::DataType predictionsDataType = TensorDescriptor::DataType::FP32;
    TensorDescriptor::DataType labelsDataType = TensorDescriptor::DataType::UINT32;

    for (uint32_t t = 0; t < 10; ++t) {
        uint32_t batchSize = (rand() % 3500) + 20;

        Tensor accuracy_d = Tensor(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {1}));
        Tensor accuracy_gpu_h = accuracy_d.clone(cpuPlacement);
        Tensor predictions_h(cpuPlacement, TensorDescriptor(predictionsDataType, {batchSize, 1}));
        Tensor predictions_d = predictions_h.clone(gpuPlacement);
        Tensor labels_h(cpuPlacement, TensorDescriptor(labelsDataType, {batchSize, 1}));
        Tensor labels_d = labels_h.clone(gpuPlacement);
        Tensor workspace_d(gpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize}));

        uint32_t numCorrect = 0;
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            float prediction = (rand() % 100000) / 100000.0f;
            uint32_t label = rand() % 2;
            ((float *)predictions_h.getMemPtr())[batch] = prediction;
            ((uint32_t *)labels_h.getMemPtr())[batch] = label;
            if (label == (prediction >= 0.5f))
                numCorrect += 1;
        }
        float expected = numCorrect / (float)batchSize;

        Stream stream(0);
        predictions_d.copyFromAsync(predictions_h, stream);
        labels_d.copyFromAsync(labels_h, stream);

        shared_ptr<BatchReduce> batchReduce = createBinaryAccuracyBatchReduce(batchSize, stream);

        launchComputeBinaryAccuracy<float, uint32_t>(
            accuracy_d, (float *)predictions_d.getMemPtr(), (uint32_t *)labels_d.getMemPtr(), workspace_d, batchSize, batchReduce, stream);

        accuracy_gpu_h.copyFromAsync(accuracy_d, stream);
        stream.synchronize();
        float actual = ((float *)accuracy_gpu_h.getMemPtr())[0];
        float thresh = 0.00001;
        if (abs(expected - actual) >= thresh)
            printf("expected %f actual %f\n", expected, actual);
        ASSERT_LT(abs(expected - actual), thresh);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
