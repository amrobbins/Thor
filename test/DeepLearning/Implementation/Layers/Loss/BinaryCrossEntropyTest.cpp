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

#include <set>
#include <vector>

using namespace std;

using namespace ThorImplementation;

half sigmoidH(half x) {
    half one = 1.0f;
    return one / (half)(one + expf(-x));
}

TEST(BinaryCrossEntropy, ComputesCorrectElementWiseResult) {
    srand(time(NULL));

    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);

    for (int test = 0; test < 100; ++test) {
        uint32_t batchSize = (rand() % 3000) + 1;
        vector<unsigned long> dimensions = {batchSize};

        bool inferenceOnly = (rand() % 5) == 0;

        TensorDescriptor labelDescriptor(TensorDescriptor::DataType::BOOLEAN, dimensions);
        TensorDescriptor elementwiseDescriptorFP32(TensorDescriptor::DataType::FP32, dimensions);
        TensorDescriptor elementwiseDescriptorFP16(TensorDescriptor::DataType::FP16, dimensions);

        Tensor labelsCpu(cpuPlacement, labelDescriptor);
        Tensor activationsCpu(cpuPlacement, elementwiseDescriptorFP16);
        Tensor labelsGpu(gpuPlacement, labelDescriptor);
        Tensor activationsGpu(gpuPlacement, elementwiseDescriptorFP16);

        Tensor lossCpu(cpuPlacement, elementwiseDescriptorFP32);
        Tensor lossGpu_h(cpuPlacement, elementwiseDescriptorFP32);

        bool *labels = (bool *)labelsCpu.getMemPtr();
        half *activations = (half *)activationsCpu.getMemPtr();
        double totalActivations = 0.0;
        while (totalActivations < 0.01) {
            totalActivations = 0.0;
            for (uint32_t i = 0; i < batchSize; ++i) {
                activations[i] = ((rand() % 1000) / 999.0f);
                totalActivations += activations[i];
                labels[i] = rand() % 2;
            }
        }

        vector<shared_ptr<Layer>> layers;
        shared_ptr<NetworkInput> activationsInput = make_shared<NetworkInput>(activationsGpu);
        layers.push_back(activationsInput);
        shared_ptr<NoOpLayer> noOpLayer = make_shared<NoOpLayer>();
        layers.push_back(noOpLayer);
        shared_ptr<NetworkInput> labelsInput = make_shared<NetworkInput>(labelsGpu);
        layers.push_back(labelsInput);
        shared_ptr<Sigmoid> sigmoid = make_shared<Sigmoid>(true);
        layers.push_back(sigmoid);
        shared_ptr<CrossEntropy> crossEntropy =
            make_shared<CrossEntropy>(CrossEntropyLossType::BINARY, TensorDescriptor::DataType::FP16, false);
        if (inferenceOnly)
            crossEntropy->setConstructForInferenceOnly(true);
        layers.push_back(crossEntropy);
        shared_ptr<NetworkOutput> lossOutput = make_shared<NetworkOutput>(gpuPlacement);
        layers.push_back(lossOutput);

        Stream stream = activationsInput->getStream();
        Stream labelsStream = labelsInput->getStream();

        LayerTestHelper::connectTwoLayers(activationsInput, noOpLayer);
        LayerTestHelper::connectTwoLayers(noOpLayer, sigmoid, 0, 0);
        LayerTestHelper::connectTwoLayers(sigmoid, crossEntropy, 0, (int)Loss::ConnectionType::FORWARD_BACKWARD);
        LayerTestHelper::connectTwoLayers(labelsInput, crossEntropy, 0, (int)Loss::ConnectionType::LABELS);
        LayerTestHelper::connectTwoLayers(crossEntropy, lossOutput, 0, 0);
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
            errorOutputGpu = sigmoid->getErrorOutput();
            errorOutputCpu = Tensor(cpuPlacement, errorOutputGpu.getDescriptor());
            errorOutputGpu_h = errorOutputCpu.clone();
            errorOutputGpu_h.copyFromAsync(errorOutputGpu, stream);
        }

        labelsStream.synchronize();

        /**
         * Binary Cross Entropy Loss (i.e. sigmoid then cross entropy loss):
         * loss = -( label * log(probability) + (1 - label) * (log(1 - probability)) )
         * where label is 0 or 1
         * Gradient of Binary Cross Entropy with respect to the predicted probability:
         * gradient = probability - label
         *
         * In the loss function, log(0) is avoided by choosing a minimum value close to the minimum positive value of fp16 or fp32
         * respectively. Note that this minimum does not affect the gradient.
         */
        float *lossMem = (float *)lossCpu.getMemPtr();
        bool *labelMem = (bool *)labelsCpu.getMemPtr();
        half *activationsMem = (half *)activationsCpu.getMemPtr();
        Tensor probabilities = activationsCpu.clone();
        half *probabilityMem = (half *)probabilities.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            probabilityMem[b] = sigmoidH(activationsMem[b]);
            lossMem[b] = -(labelMem[b] * logf(probabilityMem[b]) + (1 - labelMem[b]) * (logf(1.0f - probabilityMem[b])));
        }

        // Verify the loss output
        float *lossMemFromGpu = (float *)lossGpu_h.getMemPtr();
        for (uint32_t b = 0; b < batchSize; ++b) {
            float thresh = std::max(lossMem[b] / 1000.0f, 0.01f);
            if (!(abs(lossMem[b] - lossMemFromGpu[b]) < thresh))
                printf("loss[%d] cpuF %f gpuF %f activation %f probability %f label %i\n",
                       b,
                       lossMem[b],
                       lossMemFromGpu[b],
                       (float)activationsMem[b],
                       (float)probabilityMem[b],
                       labelMem[b]);
            ASSERT_LT(abs(lossMem[b] - lossMemFromGpu[b]), thresh);
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
        for (uint32_t b = 0; b < batchSize; ++b) {
            errorOutputMem[b] = Loss::getLossScalingFactor() * (probabilityMem[b] - labelMem[b]);
        }

        // Verify the loss gradient
        half *errorOutputFromGpu = (half *)errorOutputGpu_h.getMemPtr();
        float thresh = 0.1f;
        for (uint32_t i = 0; i < batchSize; ++i) {
            ASSERT_LT(abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]), thresh);
            if (abs((float)errorOutputMem[i] - (float)errorOutputFromGpu[i]) >= thresh) {
                printf("cpu %f gpu %f\n", (float)errorOutputMem[i], (float)errorOutputFromGpu[i]);
                fflush(stdout);
            }
        }

        LayerTestHelper::tearDownNetwork(layers);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
