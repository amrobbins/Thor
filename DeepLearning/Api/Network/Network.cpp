#include "DeepLearning/Api/Network/Network.h"

using namespace Thor;
using namespace std;

using ThorImplementation::TensorPlacement;

string Network::statusCodeToString(int statusCode) {
    if ((StatusCode)statusCode == StatusCode::SUCCESS)
        return "SUCCESS";
    else if ((StatusCode)statusCode == StatusCode::FLOATING_INPUT)
        return "FLOATING INPUT";
    else if ((StatusCode)statusCode == StatusCode::DANGLING_OUTPUT)
        return "DANGLING OUTPUT";
    else if ((StatusCode)statusCode == StatusCode::GPU_OUT_OF_MEMORY)
        return "GPU OUT OF MEMORY";
    else if ((StatusCode)statusCode == StatusCode::DUPLICATE_NAMED_NETWORK_INPUT)
        return "DUPLICATE NAMED NETWORK INPUT";
    else if ((StatusCode)statusCode == StatusCode::DUPLICATE_NAMED_NETWORK_OUTPUT)
        return "DUPLICATE NAMED NETWORK OUTPUT";
    else if ((StatusCode)statusCode == StatusCode::DEADLOCK_CYCLE)
        return "DEADLOCK CYCLE";
    assert(false);
}

Network::~Network() {
    for (uint32_t i = 0; i < stampedNetworks.size(); ++i) {
        // Calls parentCleanup then cleanUp then clears all the shared pointers:
        stampedNetworks[i].clear();
    }
    stampedNetworks.clear();
}

// Records the layers in sorted DAG order.
Network::StatusCode Network::createDagAndFreeze() {
    if (!frozen) {
        StatusCode status = evaluateGraph();
        if (status != StatusCode::SUCCESS)
            return status;
        topologicalSort();
        frozen = true;
    }

    return StatusCode::SUCCESS;
}

// Calls preomptimize on each layer, one at a time, in DAG order.
void Network::preOptimize(uint32_t gpuNum, uint32_t batchSize) {
    for (auto it = orderedNetwork.begin(); it != orderedNetwork.end(); ++it) {
        Optional<Tensor> inputTensor = it->first;
        shared_ptr<Layer> layer = it->second;

        if (inputTensor.isPresent()) {
            layer->preOptimize(inputTensor.get(), batchSize, MachineEvaluator::instance().getCopyStreamFromCpu(gpuNum));
        }
    }
}

// Returns 0 on success, returns an error code (i.e. out of memory) on failure
Network::StatusCode Network::stampNetwork(uint32_t gpuNum, std::vector<Event> &initDoneEvents, uint32_t batchSize) {
    ThorImplementation::StampedNetwork stampedNetwork;
    stampedNetwork.gpuNum = gpuNum;

    // FIXME: check for non-first instance to use shared weights
    // FIXME: support other gpus
    firstInstanceBytes = computeFirstInstanceMemRequirements(batchSize, TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum));
    nonFirstInstanceBytes = computeNonFirstInstanceMemRequirements(batchSize, TensorPlacement(TensorPlacement::MemDevices::GPU, gpuNum));
    stampedNetwork.bytesRequired = firstInstanceBytes;
    stampedNetwork.batchSize = batchSize;

    // Leave 100MB of headroom
    // FIXME: need to determine if this is the not the first instance and use shared weights and shared weights mem requirements
    if (MachineEvaluator::instance().getFreeMemBytes(gpuNum) < firstInstanceBytes + 100000000)
        return StatusCode::GPU_OUT_OF_MEMORY;

    // 1. Stamp (i.e. construct) all layers
    // 2. At the moment, I connect the layers upon stamping them, I think I should change that and stamp everything first,
    //    then for the next phase, connect them
    stampedNetwork.clear();
    try {
        // FIXME: need to throw GPU_OUT_OF_MEMORY when stamping and run out of memory

        for (auto it = orderedNetwork.begin(); it != orderedNetwork.end(); ++it) {
            Optional<Tensor> inputTensor = it->first;
            shared_ptr<Layer> layer = it->second;

            const shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
            if (networkInput) {
                stampNetworkInput(networkInput, gpuNum, batchSize, stampedNetwork);
                continue;
            }

            const shared_ptr<NetworkOutput> networkOutput = dynamic_pointer_cast<NetworkOutput>(layer);
            if (networkOutput) {
                stampNetworkOutput(inputTensor, networkOutput, gpuNum, batchSize, stampedNetwork);
                continue;
            }

            const shared_ptr<Stub> stub = dynamic_pointer_cast<Stub>(layer);
            if (stub) {
                // FIXME: Stub should cause all dangling tensors to be optimized away.
                //        currently when forward is called for a layer that is a stub, output tensor will not have been allocated
                //        and can cause memory out of bounds. Since stub is a future feature it is not being fixed yet.
                continue;
            }

            stampLayer(inputTensor, layer, gpuNum, batchSize, stampedNetwork);
        }

        // 3. Now that all layers have been constructed and connected, compile all layers
        for (const std::shared_ptr<Layer> &apiLayer : allLayersInNetworkList) {
            // It is possible for an implementation layer to have no physical layer, for example a stub layer
            if (stampedNetwork.apiLayerToPhysicalLayerShared.count(apiLayer->getId()) == 0)
                continue;

            shared_ptr<ThorImplementation::Layer> implementationLayer = stampedNetwork.apiLayerToPhysicalLayerShared[apiLayer->getId()];
            apiLayer->compile(implementationLayer);
            stampedNetwork.floatingPointOperationsPerExampleForward += implementationLayer->floatingPointOperationsPerExampleForward();
            stampedNetwork.floatingPointOperationsPerExampleBackward += implementationLayer->floatingPointOperationsPerExampleForward();
        }

        // Now that all layers are constructed, connected and compiled, initialize all layers
        // (that in turn initialize their optimizers)
        for (shared_ptr<Layer> layer : allLayersInNetworkList) {
            shared_ptr<ThorImplementation::Layer> implementationLayer = stampedNetwork.apiLayerToPhysicalLayerShared[layer->getId()];
            shared_ptr<TrainableWeightsBiasesLayer> trainableLayer = dynamic_pointer_cast<TrainableWeightsBiasesLayer>(layer);
            if (trainableLayer != nullptr) {
                shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> implementationTrainableLayer =
                    dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(implementationLayer);
                vector<Event> layerEvents =
                    trainableLayer->initialize(implementationTrainableLayer, true, nullptr, Optional<Event>::empty());
                initDoneEvents.insert(initDoneEvents.end(), make_move_iterator(layerEvents.begin()), make_move_iterator(layerEvents.end()));
            } else {
                vector<Event> layerEvents = layer->initialize(implementationLayer);
                initDoneEvents.insert(initDoneEvents.end(), make_move_iterator(layerEvents.begin()), make_move_iterator(layerEvents.end()));
            }
        }

    } catch (GpuOutOfMemoryError ex) {
        stampedNetwork.clear();
        return StatusCode::GPU_OUT_OF_MEMORY;
    }

    stampedNetworks.push_back(stampedNetwork);

    return StatusCode::SUCCESS;
}

Network::StatusCode Network::place(uint32_t batchSize,
                                   std::vector<Event> &initDoneEvents,
                                   bool inferenceOnly,
                                   std::vector<int32_t> forcedDevices,
                                   uint32_t forcedNumStampsPerGpu) {
    // FIXME: multiple stamps, multiple gpus
    // FIXME: smart placement and stamping
    assert(forcedNumStampsPerGpu == 0 || forcedNumStampsPerGpu == 1);
    vector<int32_t> gpu0 = {0};
    assert(forcedDevices == gpu0 || forcedDevices.empty());

    StatusCode statusCode;

    if (optimizer != nullptr) {
        attachOptimizerToLayers(false);
    }

    if (!inferenceOnly) {
        for (auto trainableLayer : allTrainableLayersInNetwork) {
            if (!trainableLayer->hasOptimizer()) {
                string message = "A layer of type ";
                message += trainableLayer->getLayerType();
                message += " does not have an optimizer assigned, but the network is being placed for training.";
                throw std::runtime_error(message);
            }
        }
    }

    vector<int32_t> devices;
    vector<uint32_t> numStampsPerDevice;
    if (forcedDevices.empty())
        devices = {0};
    else
        devices = forcedDevices;
    for (uint32_t i = 0; i < devices.size(); ++i) {
        if (forcedNumStampsPerGpu > 0) {
            numStampsPerDevice.push_back(forcedNumStampsPerGpu);
        } else {
            numStampsPerDevice.push_back(1);
        }
    }

    StatusCode dagStatus = createDagAndFreeze();
    if (dagStatus != StatusCode::SUCCESS) {
        printf("ERROR: evaluateGraph() returned %s\n", statusCodeToString((int)dagStatus).c_str());
        fflush(stdout);
    }
    assert(dagStatus == StatusCode::SUCCESS);

    // FIXME: pull preOptimize into initialize
    for (uint32_t i = 0; i < devices.size(); ++i) {
        preOptimize(devices[i], batchSize);
    }
    for (uint32_t i = 0; i < devices.size(); ++i) {
        for (uint32_t j = 0; j < numStampsPerDevice[i]; ++j) {
            statusCode = stampNetwork(devices[i], initDoneEvents, batchSize);
            if (statusCode != StatusCode::SUCCESS) {
                return statusCode;
            }
        }
    }

    return StatusCode::SUCCESS;
}

void Network::save(string filename, bool keep_optimizer) {
    // First I must synchronize with all devices to make sure the final batch is completely finished updating the weights.
    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    for (uint32_t gpu = 0; gpu < numGpus; ++gpu)
        Stream::deviceSynchronize(gpu);

    // FIXME:
    assert(false);
}

void Network::save_as_keras(string filename, bool keep_optimizer) {
    // First I must synchronize with all devices to make sure the final batch is completely finished updating the weights.
    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    for (uint32_t gpu = 0; gpu < numGpus; ++gpu)
        Stream::deviceSynchronize(gpu);

    // FIXME:
    assert(false);
}

void Network::load(string filename) {
    // First I must synchronize with all devices to make sure the final batch is completely finished updating the weights.
    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    for (uint32_t gpu = 0; gpu < numGpus; ++gpu)
        Stream::deviceSynchronize(gpu);

    // FIXME:
    assert(false);
}

void Network::load_from_keras(string filename) {
    // First I must synchronize with all devices to make sure the final batch is completely finished updating the weights.
    uint32_t numGpus = MachineEvaluator::instance().getNumGpus();
    for (uint32_t gpu = 0; gpu < numGpus; ++gpu)
        Stream::deviceSynchronize(gpu);

    // FIXME:
    assert(false);
}

// Determine the graph structure
// Tensors are the edges that connect the Layers which are nodes.
Network::StatusCode Network::evaluateGraph() {
    allTensors.clear();
    apiTensorToApiLoadingLayers.clear();
    apiTensorToApiDrivingLayer.clear();
    apiLayerToApiInputTensors.clear();
    apiLayerToApiOutputTensors.clear();

    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;

        // Handle each class of layers
        shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        if (networkInput) {
            Tensor outputTensor = networkInput->getFeatureOutput();
            allTensors.insert(outputTensor);
            assert(apiTensorToApiDrivingLayer.count(outputTensor) == 0);
            apiTensorToApiDrivingLayer[outputTensor] = networkInput;
            apiLayerToApiOutputTensors[networkInput].push_back(outputTensor);
            continue;
        }

        shared_ptr<NetworkOutput> networkOutput = dynamic_pointer_cast<NetworkOutput>(layer);
        if (networkOutput) {
            Tensor inputTensor = networkOutput->getFeatureInput();
            allTensors.insert(inputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(networkOutput);
            apiLayerToApiInputTensors[networkOutput].push_back(inputTensor);
            continue;
        }

        shared_ptr<Stub> stub = dynamic_pointer_cast<Stub>(layer);
        if (stub) {
            Tensor inputTensor = stub->getFeatureInput();
            allTensors.insert(inputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(stub);
            apiLayerToApiInputTensors[stub].push_back(inputTensor);
            continue;
        }

        shared_ptr<Loss> loss = dynamic_pointer_cast<Loss>(layer);
        if (loss) {
            // Predictions and Labels in, Loss out
            // Note: getPredictions() does not return the featureInput tensor when there is an initial transformation layer, like sigmoid
            Tensor rawPredictionsTensor = loss->getFeatureInput();
            Tensor labelsTensor = loss->getLabels();
            Tensor lossTensor = loss->getLoss();
            allTensors.insert(rawPredictionsTensor);
            allTensors.insert(labelsTensor);
            allTensors.insert(lossTensor);
            apiTensorToApiLoadingLayers[rawPredictionsTensor].push_back(loss);
            apiTensorToApiLoadingLayers[labelsTensor].push_back(loss);
            apiLayerToApiInputTensors[loss].push_back(rawPredictionsTensor);
            apiLayerToApiInputTensors[loss].push_back(labelsTensor);
            assert(apiTensorToApiDrivingLayer.count(lossTensor) == 0);
            apiTensorToApiDrivingLayer[lossTensor] = loss;
            apiLayerToApiOutputTensors[loss].push_back(lossTensor);
            continue;
        }

        shared_ptr<Metric> metric = dynamic_pointer_cast<Metric>(layer);
        if (metric) {
            Tensor inputTensor = metric->getFeatureInput();
            Tensor labelsTensor = metric->getLabels();
            Tensor outputTensor = metric->getFeatureOutput();
            allTensors.insert(inputTensor);
            allTensors.insert(labelsTensor);
            allTensors.insert(outputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(metric);
            apiTensorToApiLoadingLayers[labelsTensor].push_back(metric);
            apiLayerToApiInputTensors[metric].push_back(inputTensor);
            apiLayerToApiInputTensors[metric].push_back(labelsTensor);
            assert(apiTensorToApiDrivingLayer.count(outputTensor) == 0);
            apiTensorToApiDrivingLayer[outputTensor] = metric;
            apiLayerToApiOutputTensors[metric].push_back(outputTensor);
            continue;
        }

        shared_ptr<MultiConnectionLayer> multiConnectionLayer = dynamic_pointer_cast<MultiConnectionLayer>(layer);
        if (multiConnectionLayer) {
            vector<Tensor> inputTensors = multiConnectionLayer->getFeatureInputs();
            vector<Tensor> outputTensors = multiConnectionLayer->getFeatureOutputs();
            assert(!inputTensors.empty());
            assert(!outputTensors.empty());
            for (uint32_t i = 0; i < inputTensors.size(); ++i) {
                allTensors.insert(inputTensors[i]);
                apiTensorToApiLoadingLayers[inputTensors[i]].push_back(multiConnectionLayer);
                apiLayerToApiInputTensors[multiConnectionLayer].push_back(inputTensors[i]);
            }
            for (uint32_t i = 0; i < outputTensors.size(); ++i) {
                allTensors.insert(outputTensors[i]);
                assert(apiTensorToApiDrivingLayer.count(outputTensors[i]) == 0);
                apiTensorToApiDrivingLayer[outputTensors[i]] = multiConnectionLayer;
                apiLayerToApiOutputTensors[multiConnectionLayer].push_back(outputTensors[i]);
            }
            continue;
        }

        // So it is a base single connection layer
        Tensor inputTensor = layer->getFeatureInput();
        Tensor outputTensor = layer->getFeatureOutput();
        allTensors.insert(inputTensor);
        allTensors.insert(outputTensor);
        assert(apiTensorToApiDrivingLayer.count(outputTensor) == 0);
        apiTensorToApiDrivingLayer[outputTensor] = layer;
        apiTensorToApiLoadingLayers[inputTensor].push_back(layer);
        apiLayerToApiInputTensors[layer].push_back(inputTensor);
        apiLayerToApiOutputTensors[layer].push_back(outputTensor);
    }

    StatusCode status;
    status = checkForDuplicateInOutPortNames();
    if (status != StatusCode::SUCCESS)
        return status;

    status = checkForFloatingInputs();
    if (status != StatusCode::SUCCESS)
        return status;

    status = checkForDanglingOutputs();
    if (status != StatusCode::SUCCESS)
        return status;

    status = checkForDeadlockCycles();
    if (status != StatusCode::SUCCESS)
        return status;

    return StatusCode::SUCCESS;
}

Network::StatusCode Network::checkForDuplicateInOutPortNames() {
    StatusCode status = StatusCode::SUCCESS;

    set<string> inputNames;
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;
        const shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        if (networkInput != nullptr) {
            if (inputNames.count(networkInput->getName()) != 0) {
                printf("Duplicate network input name used: %s\n", networkInput->getName().c_str());
                status = StatusCode::DUPLICATE_NAMED_NETWORK_INPUT;
            }
            inputNames.insert(networkInput->getName());
        }
    }

    set<string> outputNames;
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;
        const shared_ptr<NetworkOutput> networkOutput = dynamic_pointer_cast<NetworkOutput>(layer);
        if (networkOutput != nullptr) {
            if (outputNames.count(networkOutput->getName()) != 0) {
                printf("Duplicate network output name used: %s\n", networkOutput->getName().c_str());
                status = StatusCode::DUPLICATE_NAMED_NETWORK_OUTPUT;
            }
            outputNames.insert(networkOutput->getName());
        }
    }

    return status;
}

/**
 * A tensor has a floating input when nothing is connected to write to it. -> No Driver.
 */
Network::StatusCode Network::checkForFloatingInputs() {
    for (auto it = allTensors.begin(); it != allTensors.end(); ++it) {
        Tensor tensor = *it;
        if (apiTensorToApiDrivingLayer.count(tensor) == 0) {
            printf("Tensor with id = %ld (original id %ld) is not driven.\n", tensor.getId(), tensor.getOriginalId());
            fflush(stdout);
            return StatusCode::FLOATING_INPUT;
        }
    }
    return StatusCode::SUCCESS;
}

/**
 * A tensor has a dangling output when nothing is connected to read from it -> No Loader.
 */
Network::StatusCode Network::checkForDanglingOutputs() {
    for (auto it = allTensors.begin(); it != allTensors.end(); ++it) {
        Tensor tensor = *it;
        if (apiTensorToApiLoadingLayers.count(tensor) == 0) {
            printf("tensor with id = %ld (original id %ld) is not loaded.\n", tensor.getId(), tensor.getOriginalId());
            fflush(stdout);
            return StatusCode::DANGLING_OUTPUT;
        }
    }
    return StatusCode::SUCCESS;
}

/**
 * A deadlock cycle occurs when a layer that requires all of its input to arrive before it drives its output
 * is connected in a way where there is a path from its output to its input.
 */
Network::StatusCode Network::checkForDeadlockCycles() {
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;
        if (layer->mustConnectAllInputsToDriveOutput()) {
            vector<Tensor> outputs = apiLayerToApiOutputTensors[layer];
            for (uint32_t i = 0; i < outputs.size(); ++i) {
                if (terminatesWithoutHitting(outputs[i], layer) == false)
                    return StatusCode::DEADLOCK_CYCLE;
            }
        }
    }
    return StatusCode::SUCCESS;
}

bool Network::terminatesWithoutHitting(Tensor tensor, shared_ptr<Layer> layer) {
    vector<shared_ptr<Layer>> tensorLoadingLayers = apiTensorToApiLoadingLayers[tensor];
    for (uint32_t i = 0; i < tensorLoadingLayers.size(); ++i) {
        shared_ptr<Layer> loadingLayer = tensorLoadingLayers[i];
        if (loadingLayer == layer) {
            return false;
        } else {
            vector<Tensor> layerOutputTensors = apiLayerToApiOutputTensors[loadingLayer];
            for (uint32_t j = 0; j < layerOutputTensors.size(); ++j) {
                Tensor outputTensor = layerOutputTensors[j];
                if (terminatesWithoutHitting(outputTensor, layer) == false)
                    return false;
            }
        }
    }
    return true;
}

void Network::topologicalSort() {
    deque<pair<Optional<Tensor>, shared_ptr<Layer>>> workQueue;

    orderedNetwork.clear();

    // Put all network inputs into the work queue
    for (auto it = network.begin(); it != network.end(); ++it) {
        shared_ptr<Layer> layer = *it;

        const shared_ptr<NetworkInput> networkInput = dynamic_pointer_cast<NetworkInput>(layer);
        if (networkInput) {
            Tensor outputTensor = layer->getFeatureOutput();
            vector<shared_ptr<Layer>> loadingLayers = apiTensorToApiLoadingLayers[outputTensor];
            for (uint32_t i = 0; i < loadingLayers.size(); ++i) {
                workQueue.push_front(make_pair(outputTensor, loadingLayers[i]));
            }

            orderedNetwork.push_back(make_pair(Optional<Tensor>::empty(), layer));
        }
    }

    while (!workQueue.empty()) {
        // Visit a node, connect the output tensor that corresponds to this input tensor by adding the loading layer
        // and its input tensor to orderedNetwork
        // After connecting an output tensor to its loading layer, add that loading layer and its input tensor to the work queue.
        pair<Optional<Tensor>, shared_ptr<Layer>> workNode = workQueue.back();
        workQueue.pop_back();
        Optional<Tensor> inputTensor = workNode.first;
        shared_ptr<Layer> layer = workNode.second;

        // printf("connecting tensor %ld into layer id %ld\n", inputTensor.get().getId(), layer->getId());

        // For layers, such as concatenate, that need all inputs to be connected before creating the output
        layer->informThatInputConnectionMade(inputTensor);

        vector<Tensor> outputTensors = layer->getOutputsFromInput(inputTensor);
        for (uint32_t t = 0; t < outputTensors.size(); ++t) {
            Tensor outputTensor = outputTensors[t];
            vector<shared_ptr<Layer>> loadingLayers = apiTensorToApiLoadingLayers[outputTensor];
            for (uint32_t i = 0; i < loadingLayers.size(); ++i) {
                workQueue.push_front(make_pair(outputTensor, loadingLayers[i]));
            }
        }

        orderedNetwork.push_back(make_pair(inputTensor, layer));
    }
}

// TODO: create a slice of a network that uses at most N bytes, given a specified batch size. return both network slices.
uint64_t Network::computeFirstInstanceMemRequirements(uint32_t batchSize, TensorPlacement tensorPlacement) {
    uint64_t bytes = 0;

    for (auto it = network.begin(); it != network.end(); ++it) {
        const shared_ptr<Layer> layer = *it;
        // It is only valid to get first instance bytes on single layers
        bytes += layer->getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }
    return bytes;
}

uint64_t Network::computeNonFirstInstanceMemRequirements(uint32_t batchSize, TensorPlacement tensorPlacement) {
    uint64_t bytes = 0;

    for (auto it = network.begin(); it != network.end(); ++it) {
        const shared_ptr<Layer> layer = *it;
        bytes += layer->getNonFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }
    return bytes;
}

void Network::createBatchDimensions(vector<uint64_t> &batchDimensions, vector<uint64_t> tensorDimensions, uint32_t batchSize) {
    assert(!tensorDimensions.empty());

    batchDimensions.clear();
    batchDimensions.push_back(batchSize);
    for (uint32_t i = 0; i < tensorDimensions.size(); ++i)
        batchDimensions.push_back(tensorDimensions[i]);
}

// Note that when stamping, a stamped layer does not connect to
// adjacent layers. That is done later.
// A stamped layer may be implemented by serveral actual layers, in that case
// the intermediate layers are connected to form a single logical layer
// that is ready to connect to its inputs and outputs.
void Network::stampNetworkInput(const shared_ptr<Thor::NetworkInput> networkInput,
                                uint32_t gpuNum,
                                uint32_t batchSize,
                                ThorImplementation::StampedNetwork &stampedNetwork) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    shared_ptr<ThorImplementation::Layer> outputLayer;
    Tensor outputTensor = networkInput->getFeatureOutput();

    // Stamp network input
    shared_ptr<ThorImplementation::NetworkInput> implementationNetworkInput = networkInput->stamp(placement, batchSize);
    if (DEBUG_STAMP) {
        printf("stamped network input\n");
        fflush(stdout);
    }
    stampedNetwork.inputsShared.push_back(implementationNetworkInput);
    stampedNetwork.inputs.push_back(implementationNetworkInput.get());
    stampedNetwork.inputNamedShared[implementationNetworkInput->getName()] = implementationNetworkInput;
    stampedNetwork.inputNamed[implementationNetworkInput->getName()] = implementationNetworkInput.get();
    outputLayer = implementationNetworkInput;
    stampedNetwork.apiLayerToPhysicalLayerShared[networkInput->getId()] = implementationNetworkInput;
    stampedNetwork.apiLayerToPhysicalLayer[networkInput->getId()] = implementationNetworkInput.get();
    stampedNetwork.physicalLayerToApiLayerShared[implementationNetworkInput] = networkInput->getId();
    stampedNetwork.physicalLayerToApiLayer[implementationNetworkInput.get()] = networkInput->getId();

    // Map the api tensor to its physical driving layer
    stampedNetwork.apiTensorToPhysicalDrivingLayerShared[outputTensor] = outputLayer;
    stampedNetwork.apiTensorToPhysicalDrivingLayer[outputTensor] = outputLayer.get();
}

void Network::addToNetwork(Layer *layer) {
    frozen = false;

    assert(layer != nullptr);
    addLayerToNetwork(layer);
}

void Network::addLayerToNetwork(const Layer *layer) {
    // Ensure every layer can be added just once
    shared_ptr<Layer> layerClone = layer->clone();
    if (allLayersInNetwork.count(layerClone) == 1)
        return;
    allLayersInNetwork.insert(layerClone);
    allLayersInNetworkList.push_back(layerClone);
    shared_ptr<TrainableWeightsBiasesLayer> trainableWeightsBiasesLayer = dynamic_pointer_cast<TrainableWeightsBiasesLayer>(layerClone);
    if (trainableWeightsBiasesLayer)
        allTrainableLayersInNetwork.push_back(trainableWeightsBiasesLayer);
    network.push_back(layerClone);

    auto networkInput = dynamic_cast<const NetworkInput *>(layer);
    auto networkOutput = dynamic_cast<const NetworkOutput *>(layer);
    auto stub = dynamic_cast<const Stub *>(layer);
    auto loss = dynamic_cast<const Loss *>(layer);
    auto metric = dynamic_cast<const Metric *>(layer);
    auto multiConnectionLayer = dynamic_cast<const MultiConnectionLayer *>(layer);
    if (networkInput) {
        Tensor outputTensor = networkInput->getFeatureOutput();
        apiTensorByOriginalId[outputTensor.getOriginalId()] = outputTensor;
    } else if (networkOutput) {
        Tensor inputTensor = networkOutput->getFeatureInput();
        apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
    } else if (stub) {
        Tensor inputTensor = stub->getFeatureInput();
        apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
    } else if (loss) {
        Tensor rawPredictionsTensor = loss->getFeatureInput();
        Tensor labelsTensor = loss->getLabels();
        Tensor lossTensor = loss->getLoss();
        apiTensorByOriginalId[rawPredictionsTensor.getOriginalId()] = rawPredictionsTensor;
        apiTensorByOriginalId[labelsTensor.getOriginalId()] = labelsTensor;
        apiTensorByOriginalId[lossTensor.getOriginalId()] = lossTensor;
    } else if (metric) {
        Tensor inputTensor = metric->getFeatureInput();
        Tensor labelsTensor = metric->getLabels();
        Tensor outputTensor = metric->getFeatureOutput();
        apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
        apiTensorByOriginalId[labelsTensor.getOriginalId()] = labelsTensor;
        apiTensorByOriginalId[outputTensor.getOriginalId()] = outputTensor;
    } else if (multiConnectionLayer) {
        vector<Tensor> inputTensors = multiConnectionLayer->getFeatureInputs();
        vector<Tensor> outputTensors = multiConnectionLayer->getFeatureOutputs();
        assert(!inputTensors.empty());
        assert(!outputTensors.empty());
        for (uint32_t i = 0; i < inputTensors.size(); ++i) {
            apiTensorByOriginalId[inputTensors[i].getOriginalId()] = inputTensors[i];
        }
        for (uint32_t i = 0; i < outputTensors.size(); ++i) {
            apiTensorByOriginalId[outputTensors[i].getOriginalId()] = outputTensors[i];
        }
    } else {
        // base Layer type
        Tensor inputTensor = layer->getFeatureInput();
        Tensor outputTensor = layer->getFeatureOutput();
        apiTensorByOriginalId[inputTensor.getOriginalId()] = inputTensor;
        apiTensorByOriginalId[outputTensor.getOriginalId()] = outputTensor;
    }
}

// An optimizer is used to optimize all weights and biases in a network
// If a new optimizer is added to the network it will replace the old one.
void Network::addToNetwork(Optimizer *optimizer) { this->optimizer = optimizer->clone(); }

shared_ptr<Optimizer> Network::getOptimizer() { return optimizer; }

// For future multi-gpu support, optimizers for the same layer on different GPU's will need to accumulate into a single weights memory
// and then broadcast the updated weights to the optimizers on the other gpus.
// FIXME: What is the right place for network/optimizer interaction code? Is it here? Is it in Optimizer?
//  I also have Optimizer::updateHyperParameters so currently there is code like that in both places.
//  This pattern needs to be revisited.
void Network::attachOptimizerToLayers(bool replaceIfExisting) {
    // Once the optimizer is distributed to the layers, the layers own the optimizer instances.
    // If additional layers are added later on, attachOptimizerToLayers(false) will need to be called before training more.
    assert(optimizer != nullptr);

    for (std::shared_ptr<TrainableWeightsBiasesLayer> &trainableLayer : allTrainableLayersInNetwork) {
        if (replaceIfExisting or !trainableLayer->hasOptimizer())
            trainableLayer->attachOptimizer(optimizer);
    }
}

void Network::stampLayer(Tensor inputTensor,
                         const shared_ptr<Thor::Layer> layer,
                         uint32_t gpuNum,
                         uint32_t batchSize,
                         ThorImplementation::StampedNetwork &stampedNetwork) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    shared_ptr<ThorImplementation::Layer> physicalDrivingLayer = stampedNetwork.apiTensorToPhysicalDrivingLayerShared[inputTensor];
    shared_ptr<Thor::Layer> apiDrivingLayer =
        apiTensorToApiDrivingLayer.count(inputTensor) == 0 ? nullptr : apiTensorToApiDrivingLayer[inputTensor];

    // If the api tensor has multiple loads and the physical driving layer is not a fanout,
    // then replace the physical driving layer with a newly stamped fanout
    uint32_t numLoadingLayers = apiTensorToApiLoadingLayers[inputTensor].size();
    assert(numLoadingLayers > 0);
    shared_ptr<ThorImplementation::TensorFanout> implementationTensorFanout =
        dynamic_pointer_cast<ThorImplementation::TensorFanout>(physicalDrivingLayer);
    if (apiTensorToApiLoadingLayers[inputTensor].size() != 1) {
        if (implementationTensorFanout == nullptr) {
            implementationTensorFanout = make_shared<ThorImplementation::TensorFanout>();
            Layer::connectTwoLayers(physicalDrivingLayer, implementationTensorFanout, apiDrivingLayer, nullptr, inputTensor);
            physicalDrivingLayer = implementationTensorFanout;

            stampedNetwork.otherLayersShared.push_back(implementationTensorFanout);
            stampedNetwork.otherLayers.push_back(implementationTensorFanout.get());
            apiDrivingLayer = nullptr;
            stampedNetwork.apiTensorToPhysicalDrivingLayerShared[inputTensor] = physicalDrivingLayer;
            stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer.get();

            if (DEBUG_STAMP) {
                printf("stamped tensor fanout - id %ld - driving %s - num output connections afterward %ld\n",
                       physicalDrivingLayer->getId(),
                       layer->getLayerType().c_str(),
                       implementationTensorFanout->getStreams().size());
                fflush(stdout);
            }
        } else {
            if (DEBUG_STAMP) {
                printf("connecting existing tensor fanout - id %ld - driving %s - num output connections before hand %ld\n",
                       physicalDrivingLayer->getId(),
                       layer->getLayerType().c_str(),
                       implementationTensorFanout->getStreams().size());
                fflush(stdout);
            }
        }
    }

    // Stamp the layer
    // Unless it was previously stamped on a prior pass, if so just connect the tensor.
    shared_ptr<ThorImplementation::Layer> implementationLayer = nullptr;
    // vector<Event> initializationReadyEvents;
    bool layerPreviouslyStamped = (stampedNetwork.apiLayerToPhysicalLayer.count(layer->getId()) == 1);
    // In case of a tensor fanout, there is no apiLayer...
    if (layerPreviouslyStamped) {
        implementationLayer = stampedNetwork.apiLayerToPhysicalLayerShared[layer->getId()];

        if (DEBUG_STAMP) {
            printf("connecting to %s\n", layer->getLayerType().c_str());
            fflush(stdout);
        }
    } else {
        implementationLayer = layer->stamp(placement, physicalDrivingLayer, apiDrivingLayer, inputTensor);
        stampedNetwork.apiLayerToPhysicalLayerShared[layer->getId()] = implementationLayer;
        stampedNetwork.apiLayerToPhysicalLayer[layer->getId()] = implementationLayer.get();
        stampedNetwork.physicalLayerToApiLayerShared[implementationLayer] = layer->getId();
        stampedNetwork.physicalLayerToApiLayer[implementationLayer.get()] = layer->getId();

        if (DEBUG_STAMP) {
            printf("stamped %s (physical layer id = %ld, api layer id = %ld) driven by physical layer id = %ld\n",
                   layer->getLayerType().c_str(),
                   implementationLayer->getId(),
                   layer->getId(),
                   physicalDrivingLayer->getId());
            fflush(stdout);
        }
    }
    Layer::connectTwoLayers(physicalDrivingLayer, implementationLayer, apiDrivingLayer, layer, inputTensor);

    vector<Tensor> apiOutputTensors = layer->getAllOutputTensors();
    for (uint32_t i = 0; i < apiOutputTensors.size(); ++i) {
        stampedNetwork.apiTensorToPhysicalDrivingLayerShared[apiOutputTensors[i]] = implementationLayer;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[apiOutputTensors[i]] = implementationLayer.get();
    }

    if (!layerPreviouslyStamped) {
        shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> implementationTrainableLayer =
            dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(implementationLayer);
        if (implementationTrainableLayer != nullptr) {
            stampedNetwork.trainableLayersShared.push_back(implementationTrainableLayer);
            stampedNetwork.trainableLayers.push_back(implementationTrainableLayer.get());
        } else {
            stampedNetwork.otherLayersShared.push_back(implementationLayer);
            stampedNetwork.otherLayers.push_back(implementationLayer.get());
        }
    }

    // eturn initializationReadyEvents;
}

void Network::stampNetworkOutput(Tensor inputTensor,
                                 const shared_ptr<Thor::NetworkOutput> networkOutput,
                                 uint32_t gpuNum,
                                 uint32_t batchSize,
                                 ThorImplementation::StampedNetwork &stampedNetwork) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    shared_ptr<ThorImplementation::Layer> physicalDrivingLayer = stampedNetwork.apiTensorToPhysicalDrivingLayerShared[inputTensor];
    shared_ptr<Thor::Layer> apiDrivingLayer =
        apiTensorToApiDrivingLayer.count(inputTensor) == 0 ? nullptr : apiTensorToApiDrivingLayer[inputTensor];

    // If the api tensor has multiple loads and the physical driving layer is not a fanout,
    // then replace the physical driving layer with a newly stamped fanout
    uint32_t numLoadingLayers = apiTensorToApiLoadingLayers[inputTensor].size();
    shared_ptr<ThorImplementation::TensorFanout> implementationTensorFanout =
        dynamic_pointer_cast<ThorImplementation::TensorFanout>(physicalDrivingLayer);
    assert(numLoadingLayers > 0);
    if (apiTensorToApiLoadingLayers[inputTensor].size() != 1 && implementationTensorFanout == nullptr) {
        implementationTensorFanout = make_shared<ThorImplementation::TensorFanout>();
        Layer::connectTwoLayers(physicalDrivingLayer, implementationTensorFanout, apiDrivingLayer, nullptr, inputTensor);
        physicalDrivingLayer = implementationTensorFanout;

        stampedNetwork.otherLayersShared.push_back(implementationTensorFanout);
        stampedNetwork.otherLayers.push_back(implementationTensorFanout.get());
        apiDrivingLayer = nullptr;
        stampedNetwork.apiTensorToPhysicalDrivingLayerShared[inputTensor] = physicalDrivingLayer;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer.get();
        if (DEBUG_STAMP) {
            printf("stamped tensor fanout - network output\n");
            fflush(stdout);
        }
    }

    // Stamp the network output
    TensorPlacement cpuPlacement(TensorPlacement::MemDevices::CPU);
    shared_ptr<ThorImplementation::Layer> implementationLayer =
        ((Layer *)networkOutput.get())->stamp(cpuPlacement, physicalDrivingLayer, apiDrivingLayer, inputTensor);
    shared_ptr<ThorImplementation::NetworkOutput> implementationNetworkOutput =
        dynamic_pointer_cast<ThorImplementation::NetworkOutput>(implementationLayer);
    Layer::connectTwoLayers(physicalDrivingLayer, implementationNetworkOutput, apiDrivingLayer, networkOutput, inputTensor);
    assert(implementationNetworkOutput != nullptr);
    stampedNetwork.outputsShared.push_back(implementationNetworkOutput);
    stampedNetwork.outputs.push_back(implementationNetworkOutput.get());
    stampedNetwork.outputNamedShared[implementationNetworkOutput->getName()] = implementationNetworkOutput;
    stampedNetwork.outputNamed[implementationNetworkOutput->getName()] = implementationNetworkOutput.get();
    if (DEBUG_STAMP) {
        printf("stamped network output\n");
        fflush(stdout);
    }

    stampedNetwork.apiLayerToPhysicalLayerShared[networkOutput->getId()] = implementationNetworkOutput;
    stampedNetwork.apiLayerToPhysicalLayer[networkOutput->getId()] = implementationNetworkOutput.get();
    stampedNetwork.physicalLayerToApiLayerShared[implementationNetworkOutput] = networkOutput->getId();
    stampedNetwork.physicalLayerToApiLayer[implementationNetworkOutput.get()] = networkOutput->getId();
}

Tensor Network::getApiTensorByOriginalId(uint64_t originalId) {
    if (apiTensorByOriginalId.count(originalId) == 0) {
        printf("Looking for tensor orig id %ld.\n I have these:\n\n", originalId);
        for (auto it = apiTensorByOriginalId.begin(); it != apiTensorByOriginalId.end(); ++it) {
            printf("tensor orig id %ld\n", it->first);
            fflush(stdout);
        }
    }
    assert(apiTensorByOriginalId.count(originalId) != 0);
    return apiTensorByOriginalId[originalId];
}
