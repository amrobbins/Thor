#include "test/DeepLearning/Implementation/Layers/LayerTestHelper.h"
#include "test/DeepLearning/Implementation/Layers/NoOpLayer.h"

#include "Thor.h"

#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <vector>

using namespace std;

using namespace ThorImplementation;

TEST(CategoricalCrossEntropy, ComputesCorrectElementWiseResult_oneHotLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 20; ++test) {
        vector<unsigned long> dimensions;
        int numElements = 1;
        for (int i = 0; i < 2; ++i) {
            dimensions.push_back((rand() % 300) + 1);
            if (i == 1)
                dimensions.back() += 1;
            numElements *= dimensions.back();
        }
        int batchSize = dimensions[0];
        int numElementsPerBatch = dimensions[1];

        bool inferenceOnly = (rand() % 5) == 0;

        TensorDescriptor elementwiseDescriptorFP32(TensorDescriptor::DataType::FP32, dimensions);
        TensorDescriptor elementwiseDescriptorFP16(TensorDescriptor::DataType::FP16, dimensions);

        Tensor labelsCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor activationsCpu(cpuPlacement, elementwiseDescriptorFP16);
        Tensor labelsGpu(gpuPlacement, elementwiseDescriptorFP32);
        Tensor activationsGpu(gpuPlacement, elementwiseDescriptorFP16);

        vector<unsigned long> batchDimensions;
        batchDimensions.push_back(batchSize);
        TensorDescriptor batchwiseDescriptor(TensorDescriptor::DataType::FP32, batchDimensions);
        Tensor lossCpu(cpuPlacement, batchwiseDescriptor);
        Tensor lossGpu_h(cpuPlacement, batchwiseDescriptor);

        float *labels = (float *)labelsCpu.getMemPtr();
        half *activations = (half *)activationsCpu.getMemPtr();
        double totalActivations = 0.0;
        while (totalActivations < 0.01) {
            totalActivations = 0.0;
            for (int i = 0; i < numElements; ++i) {
                activations[i] = ((rand() % 1000) / 999.0f);
                totalActivations += (double)activations[i];
                labels[i] = ((rand() % 1000) / 999.0f);
                // printf("%d: activation %f label %f\n", i, (float)activations[i], labels[i]);
            }
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> activationsInput = make_shared<NetworkInput>(activationsGpu);
        layers.push_back(activationsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<Softmax> softmax = make_shared<Softmax>(true);
        layers.push_back(softmax);
        shared_ptr<CrossEntropy> crossEntropy =
            make_shared<CrossEntropy>(CrossEntropyLossType::CATEGORICAL, TensorDescriptor::DataType::FP16, false);
        if (inferenceOnly)
            crossEntropy->setConstructForInferenceOnly(true);
        layers.push_back(crossEntropy);
        shared_ptr<LossShaper> lossShaper = make_shared<LossShaper>(LossShaper::OutputLossType::ELEMENTWISE);
        layers.push_back(lossShaper);
        shared_ptr<NetworkOutput> lossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = activationsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, softmax, 0, 0);
        LayerTestHelper::connectTwoLayers(softmax, crossEntropy, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, crossEntropy, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(crossEntropy, lossShaper, 0, 0);
        LayerTestHelper::connectTwoLayers(lossShaper, lossOutput, 0);
        LayerTestHelper::initializeNetwork(layers);

        if (inferenceOnly) {
            ASSERT_TRUE(crossEntropy->getErrorOutput().isEmpty());
        }
        ASSERT_TRUE(crossEntropy->getErrorInput().isEmpty());

        assert(lossOutput->getFeatureInput().isPresent());
        assert(lossOutput->getFeatureOutput().isPresent());
        Optional<Tensor> maybeFO = lossOutput->getFeatureOutput();
        assert(maybeFO.isPresent());
        assert(!maybeFO.isEmpty());
        maybeFO.get();

        Tensor outputGpu = lossOutput->getFeatureOutput();

        // Network is runnable here
        activationsInput->forward(activationsCpu, false);
        labelsInput->forward(labelsCpu, false);

        labelsStream.waitEvent(lossOutput->getOutputReadyEvent());
        lossGpu_h.copyFromAsync(outputGpu, labelsStream);

        Tensor errorOutputGpu;
        Tensor errorOutputCpu;
        Tensor errorOutputGpu_h;
        if (!inferenceOnly) {
            errorOutputGpu = crossEntropy->getErrorOutput();
            errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
            errorOutputGpu_h = errorOutputCpu.clone();
            errorOutputGpu_h.copyFromAsync(errorOutputGpu, labelsStream);
        }

        labelsStream.synchronize();

        // Compute the expected loss
        float *labelsMem = (float *)labelsCpu.getMemPtr();
        half *activationsMem = (half *)activationsCpu.getMemPtr();
        float *lossMem = (float *)lossCpu.getMemPtr();
        Tensor sumOfExponentials(cpuPlacement, batchwiseDescriptor);
        float *sumOfExponentialsMem = (float *)sumOfExponentials.getMemPtr();
        Tensor exponentials(cpuPlacement, elementwiseDescriptorFP32);
        float *exponentialsMem = (float *)exponentials.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            sumOfExponentialsMem[b] = 0.0f;
            for (int i = 0; i < numElementsPerBatch; ++i) {
                exponentialsMem[b * numElementsPerBatch + i] = exp((float)activationsMem[b * numElementsPerBatch + i]);
                sumOfExponentialsMem[b] += exponentialsMem[b * numElementsPerBatch + i];
            }
        }
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numElementsPerBatch; ++i) {
                exponentialsMem[b * numElementsPerBatch + i] /= sumOfExponentialsMem[b];
                if (exponentialsMem[b * numElementsPerBatch + i] < 1.0e-15f || !isfinite(exponentialsMem[b * numElementsPerBatch + i]) ||
                    isnan(exponentialsMem[b * numElementsPerBatch + i]))
                    exponentialsMem[b * numElementsPerBatch + i] = 1.0e-15f;
            }
        }

        for (int b = 0; b < batchSize; ++b) {
            lossMem[b] = 0.0f;
            for (int i = 0; i < numElementsPerBatch; ++i) {
                lossMem[b] -= labelsMem[b * numElementsPerBatch + i] * logf(exponentialsMem[b * numElementsPerBatch + i]);
            }
        }

        // Verify the loss output
        float *lossMemFromGpu = (float *)lossGpu_h.getMemPtr();
        // FIXME not batch size, this is raw loss now!
        for (int b = 0; b < batchSize; ++b) {
            float thresh = std::max(lossMem[b] / 1000.0f, 0.01f);
            if (!(abs(lossMem[b] - lossMemFromGpu[b]) < thresh))
                printf("cpuF %f gpuF %f    %d\n", lossMem[b], lossMemFromGpu[b], b);
            EXPECT_LT(abs(lossMem[b] - lossMemFromGpu[b]), thresh);
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass

        stream.synchronize();

        assert(Loss::getLossScalingFactor() > 0.0f);
        assert(Loss::getLossScalingFactor() < 1000000.0f);

        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < numElementsPerBatch; ++i) {
                errorOutputMem[b * numElementsPerBatch + i] =
                    (Loss::getLossScalingFactor() *
                     (exponentialsMem[b * numElementsPerBatch + i] - labelsMem[b * numElementsPerBatch + i]));
            }
        }

        // Verify the loss gradient
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        float thresh = 0.1f;
        for (int i = 0; i < numElements; ++i) {
            ASSERT_LT(abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]), thresh);
            if (abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]) >= thresh) {
                printf("cpu %f gpu %f\n", (float)errorOutputMem[i], (float)errorOutputFromGpu[i]);
                fflush(stdout);
            }
        }

        // Verify the loss gradient passes through the softmax layer unchanged
        Tensor softmaxErrorOutputFromGpu_h = softmax->getErrorOutput().get().clone(cpuPlacement);
        softmaxErrorOutputFromGpu_h.copyFromAsync(softmax->getErrorOutput().get(), stream);
        stream.synchronize();
        half *softmaxErrorOutputFromGpu = (half *)softmaxErrorOutputFromGpu_h.getMemPtr();
        for (int i = 0; i < numElements; ++i) {
            ASSERT_EQ((float)errorOutputFromGpu[i], (float)softmaxErrorOutputFromGpu[i]);
            // printf("%f %f\n", (float)errorOutputFromGpu[i], (float)softmaxErrorOutputFromGpu[i]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

TEST(CategoricalCrossEntropy, ComputesCorrectElementWiseResult_classIndexLabels) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (uint32_t test = 0; test < 10; ++test) {
        vector<uint64_t> dimensions = {(uint64_t)(rand() % 300) + 1, 1};
        // vector<uint64_t> dimensions = {(uint64_t)1, 1};
        uint32_t batchSize = dimensions[0];
        uint32_t numClasses = (rand() % 500) + 2;

        bool inferenceOnly = (rand() % 5) == 0;

        Tensor labelsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::UINT16, {batchSize, 1}));
        Tensor activationsCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numClasses}));
        Tensor labelsGpu = labelsCpu.clone(gpuPlacement);
        Tensor activationsGpu = activationsCpu.clone(gpuPlacement);
        Tensor lossCpu(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP16, {batchSize, numClasses}));
        Tensor lossGpu_h = lossCpu.clone();

        uint16_t *labels = (uint16_t *)labelsCpu.getMemPtr();
        half *activations = (half *)activationsCpu.getMemPtr();
        double totalActivations = 0.0;
        while (totalActivations < 0.01) {
            totalActivations = 0.0;
            for (uint32_t b = 0; b < batchSize; ++b) {
                labels[b] = rand() % numClasses;
                for (uint32_t c = 0; c < numClasses; ++c) {
                    activations[b * numClasses + c] = ((rand() % 1000) / 999.0f);
                    totalActivations += (double)activations[b * numClasses + c];
                }
            }
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> activationsInput = make_shared<NetworkInput>(activationsGpu);
        layers.push_back(activationsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<Softmax> softmax = make_shared<Softmax>(true);
        layers.push_back(softmax);
        shared_ptr<CrossEntropy> crossEntropy =
            make_shared<CrossEntropy>(CrossEntropyLossType::CATEGORICAL, TensorDescriptor::DataType::FP16, true);
        if (inferenceOnly)
            crossEntropy->setConstructForInferenceOnly(true);
        layers.push_back(crossEntropy);
        shared_ptr<NetworkOutput> lossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = activationsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, softmax, 0, 0);
        LayerTestHelper::connectTwoLayers(softmax, crossEntropy, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, crossEntropy, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(crossEntropy, lossOutput, 0, 0);
        LayerTestHelper::initializeNetwork(layers);

        if (inferenceOnly) {
            ASSERT_TRUE(crossEntropy->getErrorOutput().isEmpty());
        }
        ASSERT_TRUE(crossEntropy->getErrorInput().isEmpty());

        Tensor outputGpu = lossOutput->getFeatureOutput();

        // Network is runnable here
        activationsInput->forward(activationsCpu, false);
        labelsInput->forward(labelsCpu, false);

        labelsStream.waitEvent(lossOutput->getOutputReadyEvent());
        lossGpu_h.copyFromAsync(outputGpu, labelsStream);

        Tensor errorOutputGpu;
        Tensor errorOutputCpu;
        Tensor errorOutputGpu_h;
        if (!inferenceOnly) {
            errorOutputGpu = crossEntropy->getErrorOutput();
            errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
            errorOutputGpu_h = errorOutputCpu.clone();
            errorOutputGpu_h.copyFromAsync(errorOutputGpu, labelsStream);
        }

        labelsStream.synchronize();

        // Compute the expected loss
        half *activationsMem = (half *)activationsCpu.getMemPtr();
        half *lossMem = (half *)lossCpu.getMemPtr();
        Tensor sumOfExponentials(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize}));
        float *sumOfExponentialsMem = (float *)sumOfExponentials.getMemPtr();
        Tensor exponentials(cpuPlacement, TensorDescriptor(TensorDescriptor::DataType::FP32, {batchSize, numClasses}));
        float *exponentialsMem = (float *)exponentials.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            sumOfExponentialsMem[b] = 0.0f;
            for (uint32_t c = 0; c < numClasses; ++c) {
                exponentialsMem[b * numClasses + c] = exp((float)activationsMem[b * numClasses + c]);
                sumOfExponentialsMem[b] += exponentialsMem[b * numClasses + c];
            }
        }
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                exponentialsMem[b * numClasses + c] /= sumOfExponentialsMem[b];
                if (exponentialsMem[b * numClasses + c] < 1.0e-15f || !isfinite(exponentialsMem[b * numClasses + c]))
                    exponentialsMem[b * numClasses + c] = 1.0e-15f;
            }
        }

        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                uint32_t e = b * numClasses + c;
                uint16_t label = labels[b];
                if (c == label)
                    lossMem[e] = -logf(exponentialsMem[e]);
                else
                    lossMem[e] = 0.0f;
            }
        }

        // Verify the loss output
        half *lossMemFromGpu = (half *)lossGpu_h.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                uint32_t e = b * numClasses + c;
                float thresh = std::max((float)lossMem[e] * 0.03f, 0.01f);
                if (abs((float)lossMem[e] - (float)lossMemFromGpu[e]) >= thresh) {
                    printf("cpuF %f gpuF %f  label %d  batchSize %d numClasses %d   batch: %d class %d\n",
                           (float)lossMem[e],
                           (float)lossMemFromGpu[e],
                           labels[b],
                           batchSize,
                           numClasses,
                           b,
                           c);
                }
                ASSERT_LT(abs((float)lossMem[e] - (float)lossMemFromGpu[e]), thresh);
            }
        }

        if (inferenceOnly) {
            LayerTestHelper::tearDownNetwork(layers);
            continue;
        }

        // Backward pass

        stream.synchronize();

        half *errorOutputMem = (half *)errorOutputCpu.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t c = 0; c < numClasses; ++c) {
                errorOutputMem[b * numClasses + c] =
                    (Loss::getLossScalingFactor() * (exponentialsMem[b * numClasses + c] - (labels[b] == c ? 1.0f : 0.0f)));
            }
        }

        // Verify the loss gradient
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        float thresh = 0.1f;
        for (uint32_t b = 0; b < batchSize; ++b) {
            if (abs((float)errorOutputMem[b] - (float)errorOutputFromGpu[b]) >= thresh) {
                printf("cpu %f gpu %f   %d\n", (float)errorOutputMem[b], (float)errorOutputFromGpu[b], b);
                fflush(stdout);
            }
            ASSERT_LT(abs((float)errorOutputMem[b] - (float)errorOutputFromGpu[b]), thresh);
        }

        // Verify the loss gradient passes through the softmax layer unchanged
        Tensor softmaxErrorOutputFromGpu_h = softmax->getErrorOutput().get().clone(cpuPlacement);
        softmaxErrorOutputFromGpu_h.copyFromAsync(softmax->getErrorOutput().get(), stream);
        stream.synchronize();
        half *softmaxErrorOutputFromGpu = (half *)softmaxErrorOutputFromGpu_h.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            ASSERT_EQ((float)errorOutputFromGpu[b], (float)softmaxErrorOutputFromGpu[b]);
            // printf("%f %f\n", (float)errorOutputFromGpu[b], (float)softmaxErrorOutputFromGpu[b]);
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}
