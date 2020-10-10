#include "Thor.h"

using namespace Thor;

int main() {
    vector<ThorImplementation::StampedNetwork> stampedNetworks;

    Network alexNet = buildAlexNet();
    int gpuNum = 0;
    uint64_t batchSize = 256;
    int64_t numStamps = 2;

    printf("Starting pre-optimize\n");
    alexNet.preOptimize(gpuNum, batchSize);
    printf("Done pre-optimize\n");

    for (int i = 0; i < numStamps; ++i) {
        stampedNetworks.emplace_back();
        printf("Starting stamp\n");
        Network::StatusCode statusCode = alexNet.stampNetwork(gpuNum, batchSize, stampedNetworks.back());
        printf("Done stamp\n");
        assert(statusCode == Network::StatusCode::SUCCESS);
        stampedNetworks.back().initialize();

        for (uint32_t t = 0; t < stampedNetworks[i].trainableLayers.size(); ++t)
            stampedNetworks[i].trainableLayers[t]->setLearningRate(0.001);

        printf(
            "Using %ld bytes for batchSize %ld AlexNet network\n", stampedNetworks.back().bytesRequired, stampedNetworks.back().batchSize);
    }

    TensorPlacement gpuPlacement(TensorPlacement::MemDevices::GPU, 0);
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU, 0);
    ThorImplementation::Tensor inputTensor(
        cpuPlacement,
        ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::UINT8, {batchSize, 3, 224, 224}));
    ThorImplementation::Tensor labelsTensor(
        cpuPlacement, ThorImplementation::TensorDescriptor(ThorImplementation::TensorDescriptor::DataType::FP16, {batchSize, 1000}));

    printf("Starting training\n");
    fflush(stdout);

    Event startEvent;
    uint64_t numBatches = 100;
    map<string, ThorImplementation::Tensor> batchInput;
    map<string, ThorImplementation::Tensor> batchOutput;
    batchInput["images"] = inputTensor;
    batchInput["labels"] = labelsTensor;
    startEvent = stampedNetworks[0].outputs[0]->getStream().putEvent(true, true);
    for (uint32_t i = 0; i < numBatches; ++i) {
        for (int j = 0; j < numStamps; ++j) {
            stampedNetworks[j].sendBatch(batchInput, batchOutput);
        }
    }
    for (int j = 1; j < numStamps; ++j)
        stampedNetworks[0].inputs[0]->getStream().waitEvent(stampedNetworks[j].inputs[0]->getStream().putEvent());
    Event endEvent = stampedNetworks[0].inputs[0]->getStream().putEvent(true, true);
    printf("waiting for end\n");
    double milliseconds = endEvent.synchronizeAndReportElapsedTimeInMilliseconds(startEvent);
    printf("Done training\n");

    printf("%ld batchItems per hour\n", uint64_t((numStamps * batchSize * numBatches * 3600L) / (milliseconds / 1000.0)));
    fflush(stdout);

    for (int i = 0; i < numStamps; ++i)
        stampedNetworks[i].clear();

    return 0;
}
