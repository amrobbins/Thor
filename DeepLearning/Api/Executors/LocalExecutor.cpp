#include "DeepLearning/Api/Executors/LocalExecutor.h"

using std::map;
using std::string;

using namespace Thor;

LocalExecutor LocalExecutor::Builder::build() {
    assert(_network.isPresent());
    assert(_loader);
    // assert(_hyperparameterController.isPresent());

    if (_visualizers.isEmpty()) {
        _visualizers = vector<Visualizer>();
        _visualizers.get().push_back(ConsoleVisualizer::Builder().build());
    }

    LocalExecutor localExecutor;
    localExecutor.network = _network;
    localExecutor.loader = _loader;
    // localExecutor.hyperparameterController = _hyperparameterController;
    localExecutor.visualizers = _visualizers;

    uint64_t batchSize = localExecutor.loader->getBatchSize();

    // Stamp the network
    // FIXME: save known optimal kernels on disk
    localExecutor.network.preOptimize(0, batchSize);
    // FIXME: stamp N networks per GPU, currently just stamping 1 network on gpu 0.
    localExecutor.stampedNetworks.emplace_back();
    Thor::Network::StatusCode statusCode = localExecutor.network.stampNetwork(0, batchSize, localExecutor.stampedNetworks.back());
    assert(statusCode == Thor::Network::StatusCode::SUCCESS);
    localExecutor.stampedNetworks.back().initialize();

    localExecutor.initialized = true;
    return localExecutor;
}

uint64_t LocalExecutor::trainBatches(uint32_t batches, ExampleType exampleType) {
    assert(batches > 0);

    uint64_t nextStampToProcess =
        0;  // FIXME: this should be based on first expected to be done. Also there should be a GPU side input queue.

    vector<Event> endEvents;

    uint64_t batchNum;
    for (uint64_t i = 0; i < batches; ++i) {
        // FIXME: send state/loss to hyperparameter controller

        map<string, ThorImplementation::Tensor> batchInput;
        map<string, ThorImplementation::Tensor> batchOutput;
        map<std::string, ThorImplementation::Tensor> batchTensorMap = loader->getBatch(exampleType, batchNum);
        batchInput["images"] = batchTensorMap["examples"];
        batchInput["labels"] = batchTensorMap["labels"];

        Event startEvent;
        startEvent = stampedNetworks[0].outputs[0]->getStream().putEvent(true, true);
        for (uint32_t j = 0; j < stampedNetworks.size(); ++j) {
            stampedNetworks[j].sendBatch(batchInput, batchOutput);
        }
        endEvents.push_back(stampedNetworks[nextStampToProcess].inputs[0]->getStream().putEvent(true, true));

        // FIXME TEMP:
        nextStampToProcess += 1;
        nextStampToProcess %= stampedNetworks.size();
    }

    /*
        monitor thread {
            for i:
                double milliseconds = endEvent.synchronizeAndReportElapsedTimeInMilliseconds(startEvent);
                // FIXME: send state/loss to visualizers
                // return batch buffer
        }
    */

    return batchNum;
}

void LocalExecutor::trainTillEpochIsFinished(ExampleType exampleType) {
    uint64_t nextBatchNum = loader->getNextBatchNum(exampleType);
    uint64_t batchesToTrain = loader->getNumBatchesPerEpoch(exampleType) - nextBatchNum;
    trainBatches(batchesToTrain, exampleType);
}
