#include "DeepLearning/Api/Network/Network.h"

using namespace Thor;

Network::StatusCode Network::create() {
    StatusCode status = evaluateGraph();
    if (status != StatusCode::SUCCESS)
        return status;

    if (status != StatusCode::SUCCESS)
        return status;

    return StatusCode::SUCCESS;
}

void computeMemRequirements(uint64_t &fixedBytes, uint64_t &perBatchItemBytes) {}

// Returns 0 on success, returns an error code (i.e. out of memory) on failure
Network::StatusCode Network::stampNetwork(uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork) {
    stampedNetwork.clear();

    // Instantiate all layers
    for (auto it = network.begin(); it != network.end(); ++it) {
        try {
            const Layer *layer = it->get();

            const NetworkInput *networkInput = dynamic_cast<const NetworkInput *>(layer);
            if (networkInput) {
                stampNetworkInput(networkInput, gpuNum, batchSize, stampedNetwork);
                continue;
            }

            const NetworkOutput *networkOutput = dynamic_cast<const NetworkOutput *>(layer);
            if (networkOutput) {
                stampNetworkOutput(networkOutput, gpuNum, batchSize, stampedNetwork);
                continue;
            }

            const Loss *loss = dynamic_cast<const Loss *>(layer);
            if (loss) {
                stampLoss(loss, gpuNum, batchSize, stampedNetwork);
                continue;
            }

            const MultiConnectionLayer *multiConnectionLayer = dynamic_cast<const MultiConnectionLayer *>(layer);
            if (multiConnectionLayer) {
                stampMultiConnectionLayer(multiConnectionLayer, gpuNum, batchSize, stampedNetwork);
            }

            // So it is a base single connection layer
            stampBaseLayer(layer, gpuNum, batchSize, stampedNetwork);

        } catch (GpuOutOfMemoryError ex) {
            // Since the network was never compiled, can just clear it and the reference counted resources will be released.
            stampedNetwork.clear();
            return StatusCode::GPU_OUT_OF_MEMORY;
        }
    }

    // Now that all implementation layers are present, connect all implementation layers
    // It is done in this order because that is much easier than recursively traversing the network from input to output.
    for (auto it = network.begin(); it != network.end(); ++it) {
        // outputLayer[layer->getId()] is the layer that makes the output connection
        // inputLayer[layer->getId()] is the layer on which to connect the input tensor
    }

    return StatusCode::SUCCESS;
}

// Determine the graph structure
// Tensors are the edges that connect the Layers which are nodes.

Network::StatusCode Network::evaluateGraph() {
    for (auto it = network.begin(); it != network.end(); ++it) {
        Layer *layer = it->get();

        // Handle each class of layers
        const NetworkInput *networkInput = dynamic_cast<const NetworkInput *>(layer);
        if (networkInput) {
            uint32_t layerId = networkInput->getId();
            Tensor outputTensor = networkInput->getFeatureOutput();
            allTensors.insert(outputTensor.getId());
            assert(tensorToDrivingLayer.count(outputTensor.getId()) == 0);
            tensorToDrivingLayer[outputTensor.getId()] = layerId;
            continue;
        }

        const NetworkOutput *networkOutput = dynamic_cast<const NetworkOutput *>(layer);
        if (networkOutput) {
            uint32_t layerId = networkOutput->getId();
            Tensor inputTensor = networkOutput->getFeatureInput();
            allTensors.insert(inputTensor.getId());
            tensorToLoadingLayers[inputTensor.getId()].push_back(layerId);
            continue;
        }

        const Loss *loss = dynamic_cast<const Loss *>(layer);
        if (loss) {
            uint32_t layerId = loss->getId();
            Tensor inputTensor = loss->getFeatureInput();
            Tensor outputTensor = loss->getPredictions();
            Tensor lossTensor = loss->getLoss();
            allTensors.insert(inputTensor.getId());
            allTensors.insert(outputTensor.getId());
            // Do not insert lossTensor into allTensors because it does not need to be loaded, that is optional.
            assert(tensorToDrivingLayer.count(outputTensor.getId()) == 0);
            tensorToDrivingLayer[outputTensor.getId()] = layerId;
            tensorToLoadingLayers[inputTensor.getId()].push_back(layerId);
            tensorToLoadingLayers[lossTensor.getId()].push_back(layerId);
            continue;
        }

        const MultiConnectionLayer *multiConnectionLayer = dynamic_cast<const MultiConnectionLayer *>(layer);
        if (multiConnectionLayer) {
            uint32_t layerId = layer->getId();
            vector<Tensor> inputTensors = multiConnectionLayer->getFeatureInputs();
            vector<Tensor> outputTensors = multiConnectionLayer->getFeatureOutputs();
            assert(!inputTensors.empty());
            assert(!outputTensors.empty());
            for (uint32_t i = 0; i < inputTensors.size(); ++i) {
                allTensors.insert(inputTensors[i].getId());
                tensorToLoadingLayers[inputTensors[i].getId()].push_back(layerId);
            }
            for (uint32_t i = 0; i < outputTensors.size(); ++i) {
                allTensors.insert(outputTensors[i].getId());
                assert(tensorToDrivingLayer.count(outputTensors[i].getId()) == 0);
                tensorToDrivingLayer[outputTensors[i].getId()] = layerId;
            }
            continue;
        }

        // So it is a base single connection layer
        uint32_t layerId = layer->getId();
        Tensor inputTensor = layer->getFeatureInput();
        Tensor outputTensor = layer->getFeatureOutput();
        allTensors.insert(inputTensor.getId());
        allTensors.insert(outputTensor.getId());
        assert(tensorToDrivingLayer.count(outputTensor.getId()) == 0);
        tensorToDrivingLayer[outputTensor.getId()] = layerId;
        tensorToLoadingLayers[inputTensor.getId()].push_back(layerId);
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
        uint32_t tensorId = *it;
        if (tensorToLoadingLayers.count(tensorId) == 0)
            return StatusCode::FLOATING_INPUT;
    }
    return StatusCode::SUCCESS;
}

Network::StatusCode Network::checkForDanglingOutputs() {
    for (auto it = allTensors.begin(); it != allTensors.end(); ++it) {
        uint32_t tensorId = *it;
        if (tensorToDrivingLayer.count(tensorId) == 0)
            return StatusCode::DANGLING_OUTPUT;
    }
    return StatusCode::SUCCESS;
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

    // Stamp network intput
    ThorImplementation::NetworkInput *implementationNetworkInput =
        dynamic_cast<ThorImplementation::NetworkInput *>(((Layer *)networkInput)->stamp(placement, batchSize));
    assert(implementationNetworkInput != nullptr);
    stampedNetwork.inputs.push_back(implementationNetworkInput);

    // Stamp type converter if needed
    ThorImplementation::TypeConversion *implementationTypeConversion = nullptr;
    if (networkInput->getDataType() == Tensor::DataType::FP32) {
        implementationTypeConversion = new ThorImplementation::TypeConversion(ThorImplementation::TensorDescriptor::DataType::FP16);
        stampedNetwork.otherLayers.push_back(implementationTypeConversion);
        implementationNetworkInput->connectToNextLayer(implementationTypeConversion);
    }

    // Stamp FanOut if needed
    uint32_t numLoadingLayers = tensorToLoadingLayers[networkInput->getFeatureOutput().get().getId()].size();
    ThorImplementation::TensorFanout *implementationTensorFanout = nullptr;
    if (numLoadingLayers != 1) {
        assert(numLoadingLayers > 1);
        implementationTensorFanout = new ThorImplementation::TensorFanout();
        stampedNetwork.otherLayers.push_back(implementationTensorFanout);
        if (implementationTypeConversion != nullptr) {
            implementationTypeConversion->connectToNextLayer(implementationTensorFanout);
        } else {
            implementationNetworkInput->connectToNextLayer(implementationTensorFanout);
        }
    }

    // Record the implementation output layer
    if (implementationTensorFanout != nullptr) {
        outputLayer[networkInput->getId()] = implementationTensorFanout;
    } else if (implementationTypeConversion != nullptr) {
        outputLayer[networkInput->getId()] = implementationTypeConversion;
    } else {
        assert(implementationNetworkInput != nullptr);
        outputLayer[networkInput->getId()] = implementationNetworkInput;
    }
}

void Network::stampNetworkOutput(const Thor::NetworkOutput *networkOutput,
                                 uint32_t gpuNum,
                                 uint32_t batchSize,
                                 ThorImplementation::StampedNetwork &stampedNetwork) {
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);

    // Stamp the network output
    ThorImplementation::NetworkOutput *implementationNetworkOutput =
        dynamic_cast<ThorImplementation::NetworkOutput *>(((Layer *)networkOutput)->stamp(placement, batchSize));
    assert(implementationNetworkOutput != nullptr);
    stampedNetwork.outputs.push_back(implementationNetworkOutput);

    // Stamp type converter if needed
    Tensor networkOutputTensor = networkOutput->getFeatureOutput();
    ThorImplementation::TypeConversion *implementationTypeConversion = nullptr;
    if (networkOutputTensor.getDataType() == Tensor::DataType::FP32) {
        implementationTypeConversion = new ThorImplementation::TypeConversion(ThorImplementation::TensorDescriptor::DataType::FP32);
        stampedNetwork.otherLayers.push_back(implementationTypeConversion);
        implementationTypeConversion->connectToNextLayer(implementationNetworkOutput);
    }

    // Record the implementation input layer
    if (implementationTypeConversion != nullptr)
        inputLayer[networkOutput->getId()] = implementationTypeConversion;
    else
        inputLayer[networkOutput->getId()] = implementationNetworkOutput;
}

// Loss is different from a base layer because it has two outputs: 1. predictions 2. loss
void Network::stampLoss(const Thor::Loss *loss, uint32_t gpuNum, uint32_t batchSize, ThorImplementation::StampedNetwork &stampedNetwork) {
    // Stamp the layer
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    ThorImplementation::Layer *implementationLayer = loss->stamp(placement, batchSize);
    stampedNetwork.otherLayers.push_back(implementationLayer);

    for (int i = 0; i < 2; ++i) {
        Tensor outputTensor;
        if (i == 0)
            outputTensor = loss->getPredictions();
        else
            outputTensor = loss->getLoss();

        // Stamp FanOut if needed
        uint32_t numLoadingLayers = tensorToLoadingLayers[outputTensor.getId()].size();
        ThorImplementation::TensorFanout *implementationTensorFanout = nullptr;
        if (numLoadingLayers != 1) {
            assert(numLoadingLayers > 1);
            implementationTensorFanout = new ThorImplementation::TensorFanout();
            stampedNetwork.otherLayers.push_back(implementationTensorFanout);
            implementationLayer->connectToNextLayer(implementationTensorFanout);
        }

        // Record the implementation output layers
        map<uint32_t, ThorImplementation::Layer *> &outputMap = i == 0 ? outputLayer : outputLossLayer;

        if (implementationTensorFanout != nullptr)
            outputMap[loss->getId()] = implementationTensorFanout;
        else
            outputMap[loss->getId()] = implementationLayer;
    }
    // Record the implementation input layer
    inputLayer[loss->getId()] = implementationLayer;
}

void Network::stampMultiConnectionLayer(const Thor::MultiConnectionLayer *multiConnectionLayer,
                                        uint32_t gpuNum,
                                        uint32_t batchSize,
                                        ThorImplementation::StampedNetwork &stampedNetwork) {
    // Stamp the layer
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    ThorImplementation::Layer *implementationLayer = multiConnectionLayer->stamp(placement, batchSize);
    ThorImplementation::TrainableWeightsBiasesLayer *trainableLayer =
        dynamic_cast<ThorImplementation::TrainableWeightsBiasesLayer *>(implementationLayer);
    if (trainableLayer != nullptr)
        stampedNetwork.trainableLayers.push_back(trainableLayer);
    else
        stampedNetwork.otherLayers.push_back(implementationLayer);

    vector<Tensor> featureOutputs = multiConnectionLayer->getFeatureOutputs();
    for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
        Tensor featureOutput = featureOutputs[i];

        // Stamp FanOut if needed
        uint32_t numLoadingLayers = tensorToLoadingLayers[featureOutput.getId()].size();
        ThorImplementation::TensorFanout *implementationTensorFanout = nullptr;
        if (numLoadingLayers != 1) {
            assert(numLoadingLayers > 1);
            implementationTensorFanout = new ThorImplementation::TensorFanout();
            stampedNetwork.otherLayers.push_back(implementationTensorFanout);
            implementationLayer->connectToNextLayer(implementationTensorFanout);
        }

        // Record the implementation output layer(s)
        if (implementationTensorFanout != nullptr)
            outputLayer[multiConnectionLayer->getId()] = implementationTensorFanout;
        else
            outputLayer[multiConnectionLayer->getId()] = implementationLayer;
    }
    // Record the implementation input layer
    inputLayer[multiConnectionLayer->getId()] = implementationLayer;
}

void Network::stampBaseLayer(const Thor::Layer *layer,
                             uint32_t gpuNum,
                             uint32_t batchSize,
                             ThorImplementation::StampedNetwork &stampedNetwork) {
    // Stamp the layer
    ThorImplementation::TensorPlacement placement(TensorPlacement::MemDevices::GPU, gpuNum);
    ThorImplementation::Layer *implementationLayer = layer->stamp(placement, batchSize);
    stampedNetwork.otherLayers.push_back(implementationLayer);

    // Stamp FanOut if needed
    uint32_t numLoadingLayers = tensorToLoadingLayers[layer->getFeatureOutput().get().getId()].size();
    ThorImplementation::TensorFanout *implementationTensorFanout = nullptr;
    if (numLoadingLayers != 1) {
        assert(numLoadingLayers > 1);
        implementationTensorFanout = new ThorImplementation::TensorFanout();
        stampedNetwork.otherLayers.push_back(implementationTensorFanout);
        implementationLayer->connectToNextLayer(implementationTensorFanout);
    }

    // Record the implementation input and output layers
    inputLayer[layer->getId()] = implementationLayer;
    if (implementationTensorFanout != nullptr)
        outputLayer[layer->getId()] = implementationTensorFanout;
    else
        outputLayer[layer->getId()] = implementationLayer;
}
