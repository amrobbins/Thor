#include "DeepLearning/Api/Network/Network.h"

using namespace Thor;

// Returns 0 on success, returns an error code (i.e. out of memory) on failure
Network::StatusCode Network::stampNetwork(uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork) {
    if (!frozen) {
        StatusCode status = evaluateGraph();
        if (status != StatusCode::SUCCESS)
            return status;
        topologicalSort();
        firstInstanceBytes = computeFirstInstanceMemRequirements(batchSize);
        nonFirstInstanceBytes = computeNonFirstInstanceMemRequirements(batchSize);
        frozen = true;
    }

    // Leave 100MB of headroom
    // FIXME: need to determine if this is the not the first instance and use shared weights and shared weights mem requirements
    printf("Allocating %ld bytes to stamp network   batchSize %d\n", firstInstanceBytes, batchSize);
    fflush(stdout);
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
                continue;
            }

            stampLayer(inputTensor, layer, gpuNum, batchSize, stampedNetwork);
        }

        // All layers are connected, so now they can all be compiled
        for (uint32_t i = 0; i < stampedNetwork.inputs.size(); ++i) {
            printf("A");
            fflush(stdout);
            stampedNetwork.inputs[i]->parentCompile();
            stampedNetwork.inputs[i]->compile();
        }
        for (uint32_t i = 0; i < stampedNetwork.outputs.size(); ++i) {
            printf("B");
            fflush(stdout);
            stampedNetwork.outputs[i]->parentCompile();
            stampedNetwork.outputs[i]->compile();
        }
        for (uint32_t i = 0; i < stampedNetwork.trainableLayers.size(); ++i) {
            printf("C");
            fflush(stdout);
            stampedNetwork.trainableLayers[i]->parentCompile();
            stampedNetwork.trainableLayers[i]->compile();
        }
        for (uint32_t i = 0; i < stampedNetwork.otherLayers.size(); ++i) {
            printf("D");
            fflush(stdout);
            stampedNetwork.otherLayers[i]->parentCompile();
            stampedNetwork.otherLayers[i]->compile();
        }
        printf("E\n");
        fflush(stdout);

    } catch (GpuOutOfMemoryError ex) {
        stampedNetwork.clear();
        return StatusCode::GPU_OUT_OF_MEMORY;
    }

    return StatusCode::SUCCESS;
}

// Determine the graph structure
// Tensors are the edges that connect the Layers which are nodes.
Network::StatusCode Network::evaluateGraph() {
    allTensors.clear();
    apiTensorToApiLoadingLayers.clear();
    apiTensorToApiDrivingLayer.clear();

    for (auto it = network.begin(); it != network.end(); ++it) {
        Layer *layer = it->get();

        // Handle each class of layers
        NetworkInput *networkInput = dynamic_cast<NetworkInput *>(layer);
        if (networkInput) {
            Tensor outputTensor = networkInput->getFeatureOutput();
            allTensors.insert(outputTensor);
            assert(apiTensorToApiDrivingLayer.count(outputTensor) == 0);
            apiTensorToApiDrivingLayer[outputTensor] = networkInput;
            printf("associated tensor %ld with its driving layer %ld\n",
                   outputTensor.getId(),
                   apiTensorToApiDrivingLayer[outputTensor]->getId());
            continue;
        }

        NetworkOutput *networkOutput = dynamic_cast<NetworkOutput *>(layer);
        if (networkOutput) {
            Tensor inputTensor = networkOutput->getFeatureInput();
            allTensors.insert(inputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(networkOutput);
            continue;
        }

        Stub *stub = dynamic_cast<Stub *>(layer);
        if (stub) {
            Tensor inputTensor = stub->getFeatureInput();
            allTensors.insert(inputTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(stub);
            continue;
        }

        Loss *loss = dynamic_cast<Loss *>(layer);
        if (loss) {
            Tensor inputTensor = loss->getFeatureInput();
            Tensor labelsTensor = loss->getLabels();
            Tensor predictionsTensor = loss->getPredictions();
            Tensor lossTensor = loss->getLoss();
            allTensors.insert(inputTensor);
            allTensors.insert(labelsTensor);
            allTensors.insert(predictionsTensor);
            allTensors.insert(lossTensor);
            apiTensorToApiLoadingLayers[inputTensor].push_back(loss);
            apiTensorToApiLoadingLayers[labelsTensor].push_back(loss);
            assert(apiTensorToApiDrivingLayer.count(predictionsTensor) == 0);
            apiTensorToApiDrivingLayer[predictionsTensor] = loss;
            assert(apiTensorToApiDrivingLayer.count(lossTensor) == 0);
            apiTensorToApiDrivingLayer[lossTensor] = loss;
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
            }
            for (uint32_t i = 0; i < outputTensors.size(); ++i) {
                allTensors.insert(outputTensors[i]);
                assert(apiTensorToApiDrivingLayer.count(outputTensors[i]) == 0);
                apiTensorToApiDrivingLayer[outputTensors[i]] = multiConnectionLayer;
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
    }

    StatusCode status;
    status = checkForFloatingInputs();
    if (status != StatusCode::SUCCESS)
        return status;

    status = checkForDanglingOutputs();
    if (status != StatusCode::SUCCESS)
        return status;

    return StatusCode::SUCCESS;
}

Network::StatusCode Network::checkForFloatingInputs() {
    for (auto it = allTensors.begin(); it != allTensors.end(); ++it) {
        Tensor tensor = *it;
        if (apiTensorToApiLoadingLayers.count(tensor) == 0) {
            printf("Dangling tensor %ld\n", tensor.getId());
            // FIXME:
            //            return StatusCode::FLOATING_INPUT;
        } else {
            assert(apiTensorToApiLoadingLayers[tensor].size() == 1);
            printf("layer %ld loads tensor %ld\n", apiTensorToApiLoadingLayers[tensor][0]->getId(), tensor.getId());
        }
    }
    return StatusCode::SUCCESS;
}

Network::StatusCode Network::checkForDanglingOutputs() {
    for (auto it = allTensors.begin(); it != allTensors.end(); ++it) {
        Tensor tensor = *it;
        if (apiTensorToApiDrivingLayer.count(tensor) == 0) {
            printf("Floating tensor %ld\n", tensor.getId());
            // FIXME:
            //            return StatusCode::DANGLING_OUTPUT;
        } else {
            printf("layer %ld drives tensor %ld\n", apiTensorToApiDrivingLayer[tensor]->getId(), tensor.getId());
        }
    }
    return StatusCode::SUCCESS;
}

// FIXME: Support will be needed for layers like concatenate
//        (in that case the output tensor dimensions are only known after all of the input tensors are connected).
void Network::topologicalSort() {
    deque<pair<Optional<Tensor>, Layer *>> workQueue;

    orderedNetwork.clear();

    // Put all network inputs into the work queue
    for (auto it = network.begin(); it != network.end(); ++it) {
        Layer *layer = it->get();

        const NetworkInput *networkInput = dynamic_cast<const NetworkInput *>(layer);
        if (networkInput) {
            workQueue.push_back(make_pair(layer->getFeatureOutput(), layer));
            orderedNetwork.push_back(make_pair(Optional<Tensor>::empty(), layer));
            printf("put input layer %ld in work queue\n", networkInput->getId());
        }
    }

    while (!workQueue.empty()) {
        // Visit a node, connect the output tensor that corresponds to this input tensor by adding the loading layer and its input tensor to
        // orderedNetwork
        // After connecting an output tensor to its loading layer, add that loading layer and its input tensor to the work queue.
        pair<Optional<Tensor>, Layer *> workNode = workQueue.front();
        workQueue.pop_front();
        Optional<Tensor> inputTensor = workNode.first;
        Layer *layer = workNode.second;

        vector<Tensor> outputTensors;

        const Loss *lossLayer = dynamic_cast<const Loss *>(layer);
        const MultiConnectionLayer *multiConnectionLayer = dynamic_cast<const MultiConnectionLayer *>(layer);
        // Input layers are just layers. Output layers and stubs have no output tensors.
        if (lossLayer) {
            if (lossLayer->getConnectionType(inputTensor) == (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD) {
                outputTensors.push_back(lossLayer->getPredictions());
                outputTensors.push_back(lossLayer->getLoss());
            }
        } else if (multiConnectionLayer) {
            assert(inputTensor.isPresent());  // in the future this may not be required
            printf("want output for input tensor %ld of layer %ld\n", inputTensor.get().getId(), multiConnectionLayer->getId());
            outputTensors.push_back(multiConnectionLayer->getFeatureOutput(inputTensor));
        } else {
            outputTensors.push_back(layer->getFeatureOutput());
        }

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
uint64_t Network::computeFirstInstanceMemRequirements(uint32_t batchSize) {
    uint64_t bytes = 0;

    for (auto it = network.begin(); it != network.end(); ++it) {
        const Layer *layer = it->get();
        printf("%ld  layer %ld\n", layer->getFirstInstanceMemRequirementInBytes(batchSize), layer->getId());
        bytes += layer->getFirstInstanceMemRequirementInBytes(batchSize);
    }
    return bytes;
}

uint64_t Network::computeNonFirstInstanceMemRequirements(uint32_t batchSize) {
    uint64_t bytes = 0;

    for (auto it = network.begin(); it != network.end(); ++it) {
        const Layer *layer = it->get();
        bytes += layer->getNonFirstInstanceMemRequirementInBytes(batchSize);
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
    ThorImplementation::NetworkInput *implementationNetworkInput = networkInput->stamp(placement, batchSize);
    stampedNetwork.inputs.push_back(implementationNetworkInput);
    outputLayer = implementationNetworkInput;

    // Stamp type converter if needed
    ThorImplementation::TypeConversion *implementationTypeConversion = nullptr;
    if (outputTensor.getDataType() != Tensor::DataType::FP16) {
        implementationTypeConversion = new ThorImplementation::TypeConversion(ThorImplementation::TensorDescriptor::DataType::FP16);
        implementationNetworkInput->connectToNextLayer(implementationTypeConversion);
        outputLayer = implementationTypeConversion;

        stampedNetwork.otherLayers.push_back(implementationTypeConversion);

        outputTensor.setDataType(Tensor::DataType::FP16);
    }

    // Map the api tensor to its physical driving layer
    stampedNetwork.apiTensorToPhysicalDrivingLayer[outputTensor] = outputLayer;
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
    if (apiTensorToApiLoadingLayers[inputTensor].size() != 1 && implementationTensorFanout != nullptr) {
        implementationTensorFanout = new ThorImplementation::TensorFanout();
        Layer::connectTwoLayers(physicalDrivingLayer, implementationTensorFanout, apiDrivingLayer, nullptr, inputTensor);
        physicalDrivingLayer = implementationTensorFanout;

        stampedNetwork.otherLayers.push_back(implementationTensorFanout);
        apiDrivingLayer = nullptr;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer;
    }

    // Stamp the layer
    // Unless it was previously stamped on a prior pass, if so just connect the tensor.
    ThorImplementation::Layer *implementationLayer = nullptr;
    bool layerPreviouslyStamped = false;
    if (stampedNetwork.apiLayerToPhysicalLayer.count(layer) == 1) {
        layerPreviouslyStamped = true;
        implementationLayer = stampedNetwork.apiLayerToPhysicalLayer[layer];
        Layer::connectTwoLayers(physicalDrivingLayer, implementationLayer, apiDrivingLayer, layer, inputTensor);
    } else {
        implementationLayer = layer->stamp(placement, physicalDrivingLayer, apiDrivingLayer, inputTensor);
        stampedNetwork.apiLayerToPhysicalLayer[layer] = implementationLayer;
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
    printf("apiDrivingLayer 1   %p\n", apiDrivingLayer);

    // If the api tensor has multiple loads and the physical driving layer is not a fanout,
    // then replace the physical driving layer with a newly stamped fanout
    uint32_t numLoadingLayers = apiTensorToApiLoadingLayers[inputTensor].size();
    ThorImplementation::TensorFanout *implementationTensorFanout = dynamic_cast<ThorImplementation::TensorFanout *>(physicalDrivingLayer);
    assert(numLoadingLayers > 0);
    if (apiTensorToApiLoadingLayers[inputTensor].size() != 1 && implementationTensorFanout != nullptr) {
        implementationTensorFanout = new ThorImplementation::TensorFanout();
        Layer::connectTwoLayers(physicalDrivingLayer, implementationTensorFanout, apiDrivingLayer, nullptr, inputTensor);
        physicalDrivingLayer = implementationTensorFanout;

        stampedNetwork.otherLayers.push_back(implementationTensorFanout);
        apiDrivingLayer = nullptr;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer;
        printf("stamped fanout\n");
    }

    // Stamp type converter if needed
    if (networkOutput->getDataType() != inputTensor.getDataType()) {
        ThorImplementation::TypeConversion *implementationTypeConversion =
            new ThorImplementation::TypeConversion(Tensor::convertToImplementationDataType(networkOutput->getDataType()));
        Thor::Layer::connectTwoLayers(physicalDrivingLayer, implementationTypeConversion, apiDrivingLayer, nullptr, inputTensor);
        physicalDrivingLayer = implementationTypeConversion;
        inputTensor.setDataType(networkOutput->getDataType());

        stampedNetwork.otherLayers.push_back(implementationTypeConversion);
        apiDrivingLayer = nullptr;
        stampedNetwork.apiTensorToPhysicalDrivingLayer[inputTensor] = physicalDrivingLayer;
    }

    // Stamp the network output
    ThorImplementation::NetworkOutput *implementationNetworkOutput = dynamic_cast<ThorImplementation::NetworkOutput *>(
        ((Layer *)networkOutput)->stamp(placement, physicalDrivingLayer, apiDrivingLayer, inputTensor));
    assert(implementationNetworkOutput != nullptr);
    stampedNetwork.outputs.push_back(implementationNetworkOutput);
}
