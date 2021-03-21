#pragma once

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

// Runs a network and measures performance
class HackathonExecutor {
   public:
    HackathonExecutor(Thor::Network network, uint64_t gpuNum, uint64_t numStamps, uint64_t batchSize, uint64_t numBatches) {
        vector<ThorImplementation::StampedNetwork> stampedNetworks;

        printf("Starting pre-optimize\n");
        network.preOptimize(gpuNum, batchSize);
        printf("Done pre-optimize\n");

        for (uint32_t i = 0; i < numStamps; ++i) {
            stampedNetworks.emplace_back();
            printf("Starting stamp\n");
            Thor::Network::StatusCode statusCode = network.stampNetwork(gpuNum, batchSize, stampedNetworks.back());
            printf("Done stamp\n");
            assert(statusCode == Thor::Network::StatusCode::SUCCESS);
            stampedNetworks.back().initialize();

            for (uint32_t t = 0; t < stampedNetworks[i].trainableLayers.size(); ++t)
                stampedNetworks[i].trainableLayers[t]->setLearningRate(0.001);

            printf("Using %ld bytes to stamp network of batchSize %ld\n",
                   stampedNetworks.back().bytesRequired,
                   stampedNetworks.back().batchSize);
        }

        TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
        ThorImplementation::Tensor inputTensor(
            cpuPlacement,
            ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {batchSize, 3, 224, 224}));
        ThorImplementation::Tensor labelsTensor(
            cpuPlacement, ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::FP16, {batchSize, 1000}));

        printf("Starting training\n");
        fflush(stdout);

        Event startEvent;
        map<string, ThorImplementation::Tensor> batchInput;
        map<string, ThorImplementation::Tensor> batchOutput;
        vector<map<string, Event>> outputReadyEvents(numStamps);
        vector<Event> processingFinishedEvent(numStamps);
        batchInput["images"] = inputTensor;
        batchInput["labels"] = labelsTensor;
        omp_set_num_threads(numStamps);
        startEvent = stampedNetworks[0].outputs[0]->getStream().putEvent(true, true);
#pragma omp parallel for schedule(static, 1)
        for (uint32_t j = 0; j < numStamps; ++j) {
            for (uint32_t i = 0; i < numBatches; ++i) {
                processingFinishedEvent[j] = stampedNetworks[j].sendBatch(batchInput, batchOutput, outputReadyEvents[j]);
            }
        }
        for (uint32_t j = 1; j < numStamps; ++j)
            stampedNetworks[0].inputs[0]->getStream().waitEvent(processingFinishedEvent[j]);
        Event endEvent = stampedNetworks[0].inputs[0]->getStream().putEvent(true, true);
        printf("waiting for end\n");
        double milliseconds = endEvent.synchronizeAndReportElapsedTimeInMilliseconds(startEvent);
        printf("Done training\n");

        printf("%lf batchItems per second\n", (numStamps * batchSize * numBatches) / (milliseconds / 1000.0));
        fflush(stdout);

        for (uint32_t i = 0; i < numStamps; ++i)
            stampedNetworks[i].clear();
    }
};
