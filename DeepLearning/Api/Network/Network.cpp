#include "DeepLearning/Api/Network/Network.h"

using namespace Thor;

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

Network::StatusCode Network::preOptimize(uint32_t gpuNum, uint32_t batchSize) {
    if (!frozen) {
        StatusCode status = evaluateGraph();
        if (status != StatusCode::SUCCESS)
            return status;
        topologicalSort();
        frozen = true;
    }

    for (auto it = orderedNetwork.begin(); it != orderedNetwork.end(); ++it) {
        Optional<Tensor> inputTensor = it->first;
        Layer *layer = it->second;

        if (inputTensor.isPresent()) {
            layer->preOptimize(inputTensor.get(), batchSize, MachineEvaluator::instance().getCopyStreamFromCpu(gpuNum));
        }
    }

    return StatusCode::SUCCESS;
}

// Returns 0 on success, returns an error code (i.e. out of memory) on failure
Network::StatusCode Network::stampNetwork(uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork) {
    stampedNetwork.gpuNum = gpuNum;

    StatusCode preoptimizeStatus = preOptimize(gpuNum, batchSize);
    if (preoptimizeStatus != StatusCode::SUCCESS) {
        printf("ERROR: evaluateGraph() returned %s\n", statusCodeToString((int)preoptimizeStatus).c_str());
        fflush(stdout);
    }
    assert(preoptimizeStatus == StatusCode::SUCCESS);

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

    stampedNetwork.clear();
    try {
        // FIXME: need to throw GPU_OUT_OF_MEMORY when stamping and run out of memory

        for (auto it = orderedNetwork.begin(); it != orderedNetwork.end(); ++it) {
            Optional<Tensor> inputTensor = it->first;
            Layer *layer = it->second;

            const NetworkInput *networkInput = dynamic_cast<const NetworkInput *>(layer);
            if (networkInput) {
                stampNetworkInput(networkInput, gpuNum, batchSize, stampedNetwork);
                continue;
            }

            const NetworkOutput *networkOutput = dynamic_cast<const NetworkOutput *>(layer);
            if (networkOutput) {
                stampNetworkOutput(inputTensor, networkOutput, gpuNum, batchSize, stampedNetwork);
                continue;
            }

            const Stub *stub = dynamic_cast<const Stub *>(layer);
            if (stub) {
                // FIXME: Stub should cause all dangling tensors to be optimized away.
                //        currently when forward is called for a layer that is a stub, output tensor will not have been allocated
                //        and can cause memory out of bounds. Since stub is a future feature it is not being fixed yet.
                continue;
            }

            stampLayer(inputTensor, layer, gpuNum, batchSize, stampedNetwork);
        }

        // reorderStampedNetworkForTestability(stampedNetwork);

        // All layers are connected, so now they can all be compiled
        for (uint32_t i = 0; i < stampedNetwork.trainableLayers.size(); ++i) {
            stampedNetwork.trainableLayers[i]->parentCompile();
            stampedNetwork.trainableLayers[i]->compile();
            stampedNetwork.floatingPointOperationsPerExampleForward +=
                stampedNetwork.trainableLayers[i]->floatingPointOperationsPerExampleForward();
            stampedNetwork.floatingPointOperationsPerExampleBackward +=
                stampedNetwork.trainableLayers[i]->floatingPointOperationsPerExampleForward();
        }
        for (uint32_t i = 0; i < stampedNetwork.inputs.size(); ++i) {
            stampedNetwork.inputs[i]->parentCompile();
            stampedNetwork.inputs[i]->compile();
            stampedNetwork.floatingPointOperationsPerExampleForward += stampedNetwork.inputs[i]->floatingPointOperationsPerExampleForward();
            stampedNetwork.floatingPointOperationsPerExampleBackward +=
                stampedNetwork.inputs[i]->floatingPointOperationsPerExampleForward();
        }
        for (uint32_t i = 0; i < stampedNetwork.outputs.size(); ++i) {
            stampedNetwork.outputs[i]->parentCompile();
            stampedNetwork.outputs[i]->compile();
            stampedNetwork.floatingPointOperationsPerExampleForward +=
                stampedNetwork.outputs[i]->floatingPointOperationsPerExampleForward();
            stampedNetwork.floatingPointOperationsPerExampleBackward +=
                stampedNetwork.outputs[i]->floatingPointOperationsPerExampleForward();
        }
        for (uint32_t i = 0; i < stampedNetwork.otherLayers.size(); ++i) {
            stampedNetwork.otherLayers[i]->parentCompile();
            stampedNetwork.otherLayers[i]->compile();
            stampedNetwork.floatingPointOperationsPerExampleForward +=
                stampedNetwork.otherLayers[i]->floatingPointOperationsPerExampleForward();
            stampedNetwork.floatingPointOperationsPerExampleBackward +=
                stampedNetwork.otherLayers[i]->floatingPointOperationsPerExampleForward();
        }

    } catch (GpuOutOfMemoryError ex) {
        stampedNetwork.clear();
        return StatusCode::GPU_OUT_OF_MEMORY;
    }

    stampedNetworks.push_back(stampedNetwork);

    return StatusCode::SUCCESS;
}

/*
void reorderStampedNetworkForTestability(StampedNetwork &stampedNetwork) {
    StampedNetwork initialStampedNework = stampedNetwork;
    stampedNetwork.inputs.clear();
    stampedNetwork.outputs.clear();
    stampedNetwork.trainableLayers.clear();
    stampedNetwork.otherLayers.clear();

    reorderLayers(initialStampedNetwork, initialStampedNetwork.inputs, stampedNetwork.inputs);
    reorderLayers(initialStampedNetwork, initialStampedNetwork.inputs, stampedNetwork.outputs);
    reorderLayers(initialStampedNetwork, initialStampedNetwork.inputs, stampedNetwork.trainableLayers);
    reorderLayers(initialStampedNetwork, initialStampedNetwork.inputs, stampedNetwork.otherLayers);
}

void reorderLayers(StampedNetwork &stampedNetwork, vector<Layer*> &layersToReoder, vector<Layer*> &destinationStorage) {
    put api layers in one vector, non api layers in another vector
    sort impl layers based on their corresponding api id, will need to use a lambda comparison function

    for(uint32_t i = 0; i < layersToReorder.size(); ++i) {

    }
}
*/

// Determine the graph structure
// Tensors are the edges that connect the Layers which are nodes.
Network::StatusCode Network::evaluateGraph() {
    allTensors.clear();
    apiTensorToApiLoadingLayers.clear();
    apiTensorToApiDrivingLayer.clear();
    apiLayerToApiInputTensors.clear();
    apiLayerToApiOutputTensors.clear();

    for (auto it = network.begin(); it != network.end(); ++it) {
        Layer *layer = it->get();

        // Handle each class of layers
        NetworkInput *networkInput = dynamic_cast<NetworkInput *>(layer);
        if (networkInput) {
            Tensor outputTensor = networkInput->getFeatureOutput();
            allTensors.insert(outputTensor);
            assert(apiTensorToApiDrivingLayer.count(outputTensor) == 0);
            apiTensorToApiDrivingLayer[outputTensor] = networkInput;
            apiLayerToApiOutputTensors[networkInput].push_back(outputTensor);
            continue;
        }

        NetworkOutput *networkOutput = dynamic_cast<NetworkOutput *>(layer);
        if (networkOutput) {
            Tensor inputTensor = networkOutput->getFeatureInput();
            allTensors.insert(inputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(networkOutput);
            apiLayerToApiInputTensors[networkOutput].push_back(inputTensor);
            continue;
        }

        Stub *stub = dynamic_cast<Stub *>(layer);
        if (stub) {
            Tensor inputTensor = stub->getFeatureInput();
            allTensors.insert(inputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(stub);
            apiLayerToApiInputTensors[stub].push_back(inputTensor);
            continue;
        }

        Loss *loss = dynamic_cast<Loss *>(layer);
        if (loss) {
            // Predictions and Labels in, Loss out
            Tensor predictionsTensor = loss->getPredictions();
            Tensor labelsTensor = loss->getLabels();
            Tensor lossTensor = loss->getLoss();
            allTensors.insert(predictionsTensor);
            allTensors.insert(labelsTensor);
            allTensors.insert(lossTensor);
            apiTensorToApiLoadingLayers[predictionsTensor].push_back(loss);
            apiTensorToApiLoadingLayers[labelsTensor].push_back(loss);
            apiLayerToApiInputTensors[loss].push_back(predictionsTensor);
            apiLayerToApiInputTensors[loss].push_back(labelsTensor);
            assert(apiTensorToApiDrivingLayer.count(lossTensor) == 0);
            apiTensorToApiDrivingLayer[lossTensor] = loss;
            apiLayerToApiOutputTensors[loss].push_back(lossTensor);
            continue;
        }

        Metric *metric = dynamic_cast<Metric *>(layer);
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

        MultiConnectionLayer *multiConnectionLayer = dynamic_cast<MultiConnectionLayer *>(layer);
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
        Layer *layer = it->get();
        const NetworkInput *networkInput = dynamic_cast<const NetworkInput *>(layer);
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
        Layer *layer = it->get();
        const NetworkOutput *networkOutput = dynamic_cast<const NetworkOutput *>(layer);
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
        if (apiTensorToApiDrivingLayer.count(tensor) == 0)
            return StatusCode::FLOATING_INPUT;
    }
    return StatusCode::SUCCESS;
}

/**
 * A tensor has a dangling output when nothing is connected to read from it -> No Loader.
 */
Network::StatusCode Network::checkForDanglingOutputs() {
    for (auto it = allTensors.begin(); it != allTensors.end(); ++it) {
        Tensor tensor = *it;
        if (apiTensorToApiLoadingLayers.count(tensor) == 0)
            return StatusCode::DANGLING_OUTPUT;
    }
    return StatusCode::SUCCESS;
}

/**
 * A deadlock cycle occurs when a layer that requires all of its input to arrive before it drives its output
 * is connected in a way where there is a path from its output to its input.
 */
Network::StatusCode Network::checkForDeadlockCycles() {
    for (auto it = network.begin(); it != network.end(); ++it) {
        Layer *layer = it->get();
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

bool Network::terminatesWithoutHitting(Tensor tensor, Layer *layer) {
    vector<Layer *> tensorLoadingLayers = apiTensorToApiLoadingLayers[tensor];
    for (uint32_t i = 0; i < tensorLoadingLayers.size(); ++i) {
        Layer *loadingLayer = tensorLoadingLayers[i];
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
    deque<pair<Optional<Tensor>, Layer *>> workQueue;

    orderedNetwork.clear();

    // Put all network inputs into the work queue
    for (auto it = network.begin(); it != network.end(); ++it) {
        Layer *layer = it->get();

        const NetworkInput *networkInput = dynamic_cast<const NetworkInput *>(layer);
        if (networkInput) {
            Tensor outputTensor = layer->getFeatureOutput();
            vector<Layer *> loadingLayers = apiTensorToApiLoadingLayers[outputTensor];
            for (uint32_t i = 0; i < loadingLayers.size(); ++i) {
                workQueue.push_back(make_pair(outputTensor, loadingLayers[i]));
            }

            orderedNetwork.push_back(make_pair(Optional<Tensor>::empty(), layer));
        }
    }

    while (!workQueue.empty()) {
        // Visit a node, connect the output tensor that corresponds to this input tensor by adding the loading layer and its input tensor to
        // orderedNetwork
        // After connecting an output tensor to its loading layer, add that loading layer and its input tensor to the work queue.
        pair<Optional<Tensor>, Layer *> workNode = workQueue.back();
        workQueue.pop_back();
        Optional<Tensor> inputTensor = workNode.first;
        Layer *layer = workNode.second;

        // For layers, such as concatenate, that need all inputs to be connected before creating the output
        layer->informThatInputConnectionMade(inputTensor);

        vector<Tensor> outputTensors = layer->getOutputsFromInput(inputTensor);
        for (uint32_t t = 0; t < outputTensors.size(); ++t) {
            Tensor outputTensor = outputTensors[t];
            vector<Layer *> loadingLayers = apiTensorToApiLoadingLayers[outputTensor];
            for (uint32_t i = 0; i < loadingLayers.size(); ++i) {
                workQueue.push_back(make_pair(outputTensor, loadingLayers[i]));
            }
        }

        orderedNetwork.push_back(make_pair(inputTensor, layer));
    }
}

// TODO: create a slice of a network that uses at most N bytes, given a specified batch size. return both network slices.
uint64_t Network::computeFirstInstanceMemRequirements(uint32_t batchSize, TensorPlacement tensorPlacement) {
    uint64_t bytes = 0;

    for (auto it = network.begin(); it != network.end(); ++it) {
        const Layer *layer = it->get();
        // It is only valid to get first instance bytes on single layers
        assert(!layer->isMultiLayer());
        bytes += layer->getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }
    return bytes;
}

uint64_t Network::computeNonFirstInstanceMemRequirements(uint32_t batchSize, TensorPlacement tensorPlacement) {
    uint64_t bytes = 0;

    for (auto it = network.begin(); it != network.end(); ++it) {
        const Layer *layer = it->get();
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
void Network::stampNetworkInput(const Thor::NetworkInput *networkInput,
                                uint32_t gpuNum,
                                uint32_t batchSize,
                                ThorImplementation::StampedNetwork &stampedNetwork) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    ThorImplementation::Layer *outputLayer;
    Tensor outputTensor = networkInput->getFeatureOutput();

    // Stamp network input
    ThorImplementation::NetworkInput *implementationNetworkInput = networkInput->stamp(placement, batchSize, stampedNetwork.initializers);
    if (DEBUG_STAMP) {
        printf("stamped network input\n");
        fflush(stdout);
    }
    stampedNetwork.inputs.push_back(implementationNetworkInput);
    stampedNetwork.inputNamed[implementationNetworkInput->getName()] = implementationNetworkInput;
    outputLayer = implementationNetworkInput;
    stampedNetwork.apiLayerToPhysicalLayer[networkInput->getId()] = implementationNetworkInput;
    stampedNetwork.physicalLayerToApiLayer[implementationNetworkInput] = networkInput->getId();

    // Map the api tensor to its physical driving layer
    stampedNetwork.apiTensorToPhysicalDrivingLayer[outputTensor] = outputLayer;
}

void Network::addToNetwork(Layer *layer) {
    frozen = false;

    assert(layer != nullptr);

    if (layer->isMultiLayer()) {
        layer->convertToSingleLayersAndAddToNetwork();
    } else {
        addSingleLayerToNetwork(layer);
    }
}

void Network::stampLayer(
    Tensor inputTensor, const Thor::Layer *layer, uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    ThorImplementation::Layer *physicalDrivingLayer = stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor];
    Thor::Layer *apiDrivingLayer = apiTensorToApiDrivingLayer.count(inputTensor) == 0 ? nullptr : apiTensorToApiDrivingLayer[inputTensor];

    // If the api tensor has multiple loads and the physical driving layer is not a fanout,
    // then replace the physical driving layer with a newly stamped fanout
    uint32_t numLoadingLayers = apiTensorToApiLoadingLayers[inputTensor].size();
    assert(numLoadingLayers > 0);
    ThorImplementation::TensorFanout *implementationTensorFanout = dynamic_cast<ThorImplementation::TensorFanout *>(physicalDrivingLayer);
    if (apiTensorToApiLoadingLayers[inputTensor].size() != 1 && implementationTensorFanout == nullptr) {
        implementationTensorFanout = new ThorImplementation::TensorFanout();
        Layer::connectTwoLayers(physicalDrivingLayer, implementationTensorFanout, apiDrivingLayer, nullptr, inputTensor);
        physicalDrivingLayer = implementationTensorFanout;

        stampedNetwork.otherLayers.push_back(implementationTensorFanout);
        apiDrivingLayer = nullptr;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer;

        if (DEBUG_STAMP) {
            printf("stamped tensor fanout - network input\n");
            fflush(stdout);
        }
    }

    // Stamp the layer
    // Unless it was previously stamped on a prior pass, if so just connect the tensor.
    ThorImplementation::Layer *implementationLayer = nullptr;
    bool layerPreviouslyStamped = false;
    if (stampedNetwork.apiLayerToPhysicalLayer.count(layer->getId()) == 1) {
        layerPreviouslyStamped = true;
        implementationLayer = stampedNetwork.apiLayerToPhysicalLayer[layer->getId()];
        Layer::connectTwoLayers(physicalDrivingLayer, implementationLayer, apiDrivingLayer, layer, inputTensor);
    } else {
        implementationLayer = layer->stamp(placement, physicalDrivingLayer, apiDrivingLayer, inputTensor, stampedNetwork.initializers);
        stampedNetwork.apiLayerToPhysicalLayer[layer->getId()] = implementationLayer;
        stampedNetwork.physicalLayerToApiLayer[implementationLayer] = layer->getId();

        if (DEBUG_STAMP) {
            printf("stamped %s\n", layer->getLayerType().c_str());
            fflush(stdout);
        }
    }

    vector<Tensor> apiOutputTensors = layer->getAllOutputTensors();
    for (uint32_t i = 0; i < apiOutputTensors.size(); ++i)
        stampedNetwork.apiTensorToPhysicalDrivingLayer[apiOutputTensors[i]] = implementationLayer;

    if (!layerPreviouslyStamped) {
        ThorImplementation::TrainableWeightsBiasesLayer *implementationTrainableLayer =
            dynamic_cast<ThorImplementation::TrainableWeightsBiasesLayer *>(implementationLayer);
        if (implementationTrainableLayer != nullptr) {
            stampedNetwork.trainableLayers.push_back(implementationTrainableLayer);
        } else {
            stampedNetwork.otherLayers.push_back(implementationLayer);
        }
    }
}

void Network::stampNetworkOutput(Tensor inputTensor,
                                 const Thor::NetworkOutput *networkOutput,
                                 uint32_t gpuNum,
                                 uint32_t batchSize,
                                 ThorImplementation::StampedNetwork &stampedNetwork) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    ThorImplementation::Layer *physicalDrivingLayer = stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor];
    Thor::Layer *apiDrivingLayer = apiTensorToApiDrivingLayer.count(inputTensor) == 0 ? nullptr : apiTensorToApiDrivingLayer[inputTensor];

    // If the api tensor has multiple loads and the physical driving layer is not a fanout,
    // then replace the physical driving layer with a newly stamped fanout
    uint32_t numLoadingLayers = apiTensorToApiLoadingLayers[inputTensor].size();
    ThorImplementation::TensorFanout *implementationTensorFanout = dynamic_cast<ThorImplementation::TensorFanout *>(physicalDrivingLayer);
    assert(numLoadingLayers > 0);
    if (apiTensorToApiLoadingLayers[inputTensor].size() != 1 && implementationTensorFanout == nullptr) {
        implementationTensorFanout = new ThorImplementation::TensorFanout();
        Layer::connectTwoLayers(physicalDrivingLayer, implementationTensorFanout, apiDrivingLayer, nullptr, inputTensor);
        physicalDrivingLayer = implementationTensorFanout;

        stampedNetwork.otherLayers.push_back(implementationTensorFanout);
        apiDrivingLayer = nullptr;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer;
        if (DEBUG_STAMP) {
            printf("stamped tensor fanout - network output\n");
            fflush(stdout);
        }
    }

    // Stamp the network output
    ThorImplementation::NetworkOutput *implementationNetworkOutput =
        dynamic_cast<ThorImplementation::NetworkOutput *>(((Layer *)networkOutput)
                                                              ->stamp(ThorImplementation::TensorPlacement::MemDevices::CPU,
                                                                      physicalDrivingLayer,
                                                                      apiDrivingLayer,
                                                                      inputTensor,
                                                                      stampedNetwork.initializers));
    assert(implementationNetworkOutput != nullptr);
    stampedNetwork.outputs.push_back(implementationNetworkOutput);
    stampedNetwork.outputNamed[implementationNetworkOutput->getName()] = implementationNetworkOutput;
    if (DEBUG_STAMP) {
        printf("stamped network output\n");
        fflush(stdout);
    }

    stampedNetwork.apiLayerToPhysicalLayer[networkOutput->getId()] = implementationNetworkOutput;
    stampedNetwork.physicalLayerToApiLayer[implementationNetworkOutput] = networkOutput->getId();
}
