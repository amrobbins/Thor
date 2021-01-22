#include "DeepLearning/Api/Executors/LocalExecutor.h"

using std::map;
using std::string;

using namespace Thor;

shared_ptr<LocalExecutor> LocalExecutor::Builder::build() {
    assert(_network.isPresent());
    assert(_loader);
    // FIXME: add hyperparameter controller
    // assert(_hyperparameterController.isPresent());

    if (_visualizers.isEmpty()) {
        _visualizers = vector<shared_ptr<Visualizer>>();
        _visualizers.get().push_back(ConsoleVisualizer::Builder().build());
    }

    shared_ptr<LocalExecutor> localExecutor = make_shared<LocalExecutor>();
    localExecutor->network = _network;
    localExecutor->loader = _loader;
    // localExecutor->hyperparameterController = _hyperparameterController;
    localExecutor->visualizers = _visualizers;

    uint64_t batchSize = localExecutor->loader->getBatchSize();

    // Stamp the network
    // FIXME: stamp N networks per GPU, currently just stamping 1 network on gpu 0.
    // FIXME: save known optimal kernels on disk
    localExecutor->network.preOptimize(0, batchSize);
    localExecutor->stampedNetworks.emplace_back();
    Thor::Network::StatusCode statusCode = localExecutor->network.stampNetwork(0, batchSize, localExecutor->stampedNetworks.back());
    assert(statusCode == Thor::Network::StatusCode::SUCCESS);
    localExecutor->stampedNetworks.back().initialize();

    localExecutor->hyperparameterControllerExecutionState.reset(new AsyncQueue<ExecutionState>(32));
    for (uint64_t i = 0; i < localExecutor->visualizers.size(); ++i)
        localExecutor->visualizerExecutionState.emplace_back(new AsyncQueue<ExecutionState>(32));

    localExecutor->loanedBufferQueue.reset(new AsyncQueue<LoanedBufferMetadata>(32));

    localExecutor->initialized = true;
    return localExecutor;
}

uint64_t LocalExecutor::trainBatches(uint32_t batches, ExampleType exampleType) {
    assert(batches > 0);

    // FIXME: this should be based on first expected to be done. Also there should be a GPU side input and output queue.
    uint64_t nextStampToProcess = 0;

    vector<map<string, Event>> outputReadyEvents(stampedNetworks.size());
    vector<Event> processingFinishedEvents(stampedNetworks.size());
    Event endEvent;

    // FIXME:
    uint64_t batchletsPerBatch = 1;

    uint64_t epochBatchNum;
    for (uint64_t batch = 0; batch < batches; ++batch) {
        for (uint64_t batchlet = 0; batchlet < batchletsPerBatch; ++batchlet) {
            map<string, ThorImplementation::Tensor> batchInput;
            map<string, ThorImplementation::Tensor> batchOutput;
            map<std::string, ThorImplementation::Tensor> batchTensorMap = loader->getBatch(exampleType, epochBatchNum);
            batchInput["images"] = batchTensorMap["examples"];
            batchInput["labels"] = batchTensorMap["labels"];

            stampedNetworks[nextStampToProcess].sendBatch(
                batchInput, batchOutput, outputReadyEvents[nextStampToProcess], processingFinishedEvents[nextStampToProcess]);

            // process all outputs
            // FIXME: for now I will only deal with the loss output, the other outputs (predicted labels and whatever else)
            //        are ignored for now. I think that eventually I will need to enqueue all outputs in an async queue,
            //        together with some metadata from their corresponding inputs.

            // need to return the buffers -> create a monitoring thread that gets an event and vector of buffers to return to the loader.
            loanedBufferQueue->push(LoanedBufferMetadata(processingFinishedEvents[nextStampToProcess], exampleType, batchTensorMap));

            nextStampToProcess += 1;
            nextStampToProcess %= stampedNetworks.size();
        }

        // Now the input stream from each stamp waits till all gradient update work is done for the stamp before it will begin processing
        // the next set of inputs passed to it.
    }

    for (uint32_t i = 0; i < processingFinishedEvents.size(); ++i) {
        processingFinishedEvents[i].synchronize();
    }

    return epochBatchNum;
}

void LocalExecutor::trainTillEpochIsFinished(ExampleType exampleType) {
    uint64_t nextBatchNum = loader->getNextBatchNum(exampleType);
    uint64_t batchesToTrain = loader->getNumBatchesPerEpoch(exampleType) - nextBatchNum;
    trainBatches(batchesToTrain, exampleType);
}
