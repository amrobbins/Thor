#include "DeepLearning/Implementation/ThorError.h"
#include "DeepLearning/Api/Network/StampedNetwork.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"

#include <limits>
#include <optional>

namespace ThorImplementation {

void StampedNetwork::initialize(bool initializeWeights, bool copyWeightsFromOtherStamp, StampedNetwork *otherStamp) {
    // First, ensure the shared pointers and raw pointers match
    for (auto it = inputsShared.begin(); it != inputsShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(inputs, it->get()) == 1);
    for (auto it = outputsShared.begin(); it != outputsShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(outputs, it->get()) == 1);
    for (auto it = trainableLayersShared.begin(); it != trainableLayersShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(trainableLayers, it->get()) == 1);
    for (auto it = otherLayersShared.begin(); it != otherLayersShared.end(); ++it)
        THOR_THROW_IF_FALSE(count(otherLayers, it->get()) == 1);
    for (auto it = apiTensorToPhysicalDrivingLayerShared.begin(); it != apiTensorToPhysicalDrivingLayerShared.end(); ++it) {
        THOR_THROW_IF_FALSE(apiTensorToPhysicalDrivingLayer.count(it->first) == 1);
        THOR_THROW_IF_FALSE(apiTensorToPhysicalDrivingLayer[it->first] == it->second.get());
    }
    for (auto it = apiLayerToPhysicalLayerShared.begin(); it != apiLayerToPhysicalLayerShared.end(); ++it) {
        THOR_THROW_IF_FALSE(apiLayerToPhysicalLayer.count(it->first) == 1);
        THOR_THROW_IF_FALSE(apiLayerToPhysicalLayer[it->first] == it->second.get());
    }
    for (auto it = physicalLayerToApiLayerShared.begin(); it != physicalLayerToApiLayerShared.end(); ++it) {
        THOR_THROW_IF_FALSE(physicalLayerToApiLayer.count(it->first.get()) == 1);
        THOR_THROW_IF_FALSE(physicalLayerToApiLayer[it->first.get()] == it->second);
    }
    for (auto it = apiTensorToApiDrivingLayerShared.begin(); it != apiTensorToApiDrivingLayerShared.end(); ++it) {
        THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer.count(it->first) == 1);
        THOR_THROW_IF_FALSE(apiTensorToApiDrivingLayer[it->first] == it->second.get());
    }
    for (auto it = inputNamedShared.begin(); it != inputNamedShared.end(); ++it) {
        THOR_THROW_IF_FALSE(inputNamed.count(it->first) == 1);
        THOR_THROW_IF_FALSE(inputNamed[it->first] == it->second.get());
    }
    for (auto it = outputNamedShared.begin(); it != outputNamedShared.end(); ++it) {
        THOR_THROW_IF_FALSE(outputNamed.count(it->first) == 1);
        THOR_THROW_IF_FALSE(outputNamed[it->first] == it->second.get());
    }

    // // FIXME: This overlaps + fights with newer deserialization/initialization logic
    // // Now that checks have been run, initialize the stamp
    // THOR_THROW_IF_FALSE(!(initializeWeights && copyWeightsFromOtherStamp));
    // if (initializeWeights) {
    //     // Weights are shared by all stamps so weights are only initialized once
    //     for (uint32_t i = 0; i < initializers.size(); ++i)
    //         initializers[i]->initialize();
    // } else if (copyWeightsFromOtherStamp) {
    //     // Every GPU needs its a copy of the weights, if they have already been initialized in a weights memory, then copy that memory
    //     // to the target GPU.
    //     THOR_THROW_IF_FALSE(otherStamp != nullptr);
    //     // FIXME use trainable layer stamped ids to copy weights and when present biases from other stamp to this stamp
    //     std::unordered_map<uint64_t, ThorImplementation::TrainableLayer *> trainableLayerMap;
    //     for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
    //         trainableLayerMap[trainableLayers[i]->getStampedId()] = trainableLayers[i];
    //     }
    //     std::vector<Stream> streams;
    //     Stream stream;
    //     for (uint32_t i = 0; i < otherStamp->trainableLayers.size(); ++i) {
    //         uint32_t stampedId = otherStamp->trainableLayers[i]->getStampedId();
    //         if (i == 0) {
    //             streams.push_back(trainableLayerMap[stampedId]->getStreams()[0]);
    //         }
    //         Tensor uninitializedWeights = trainableLayerMap[stampedId]->getWeights();
    //         std::optional<Tensor> uninitializedBiases = trainableLayerMap[stampedId]->getBiases();
    //         ThorImplementation::TrainableLayer *initializedLayer = otherStamp->trainableLayers[i];
    //         Tensor initializedWeights = initializedLayer->getWeights();
    //         std::optional<Tensor> initializedBiases = initializedLayer->getBiases();
    //         uninitializedWeights.copyFromAsync(initializedWeights, streams.back());
    //         if (initializedBiases.has_value()) {
    //             THOR_THROW_IF_FALSE(uninitializedBiases.has_value());
    //             uninitializedBiases.value().copyFromAsync(initializedBiases.value(), stream);
    //         }
    //     }
    //     for (uint32_t i = 0; i < streams.size(); ++i) {
    //         streams[i].synchronize();
    //     }
    // }

    // // FIXME: get rid of implementation layer initialize, that is owned by API layer. Implementation layer has compile.
    // // so implementationLayer.compile then apiLayer.initialize()
    // for (uint32_t i = 0; i < inputs.size(); ++i) {
    //     inputs[i]->parentInitialize();
    //     inputs[i]->initialize();
    // }
    // for (uint32_t i = 0; i < outputs.size(); ++i) {
    //     outputs[i]->parentInitialize();
    //     outputs[i]->initialize();
    // }
    // for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
    //     trainableLayers[i]->parentInitialize();
    //     trainableLayers[i]->initialize();
    // }
    // for (uint32_t i = 0; i < otherLayers.size(); ++i) {
    //     otherLayers[i]->parentInitialize();
    //     otherLayers[i]->initialize();
    // }
}

// Note that all processing is finished at the end of any input stream of the stamp.
// Note *input* stream - this is not the case for the loader streams
Event StampedNetwork::sendBatch(std::map<std::string, Tensor> batchInputs,
                                std::map<std::string, Tensor> &batchOutputs,
                                std::map<std::string, Event> &outputReadyEvents,
                                bool isInferenceOnly) {
    THOR_THROW_IF_FALSE(batchInputs.size() == inputs.size());

    std::optional<uint32_t> batchSize;
    for (const auto &[inputName, inputTensor] : batchInputs) {
        (void)inputName;
        const std::vector<uint64_t> dimensions = inputTensor.getDescriptor().getDimensions();
        THOR_THROW_IF_FALSE(!dimensions.empty());
        THOR_THROW_IF_FALSE(dimensions[0] <= std::numeric_limits<uint32_t>::max());
        if (!batchSize.has_value()) {
            batchSize = static_cast<uint32_t>(dimensions[0]);
        } else {
            THOR_THROW_IF_FALSE(batchSize.value() == dimensions[0]);
        }
    }
    THOR_THROW_IF_FALSE(batchSize.has_value());

    for (uint32_t i = 0; i < inputs.size(); ++i) {
        auto it = batchInputs.find(inputs[i]->getName());
        THOR_THROW_IF_FALSE(it != batchInputs.end());
        Tensor inputTensor = it->second;
        inputs[i]->forward(inputTensor, isInferenceOnly, batchSize.value());
    }

    // The stream from input 0 waits for all outputs to be ready.
    //
    // NetworkOutput may offload its value through a dedicated download stream when
    // the requested output placement differs from the producing layer placement
    // (for example GPU loss -> CPU stats tensor).  In that case getStream() is the
    // producing/compute stream, not the stream that owns the final D2H copy.  Waiting
    // on a fresh event from getStream() can therefore release host callbacks before
    // the CPU output buffer contains the current batch.  The NetworkOutput-owned
    // outputReadyEvent is recorded after the final copy to the public output tensor,
    // so downstream consumers must wait on that event instead.
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        batchOutputs[outputs[i]->getName()] = outputs[i]->getFeatureOutput().value();
        Event outputReadyEvent = outputs[i]->getOutputReadyEvent();
        outputReadyEvents[outputs[i]->getName()] = outputReadyEvent;
        inputs[0]->getStream().waitEvent(outputReadyEvent);
    }

    // Processing is finished when the stream from input 0 is ready
    Event processingFinishedEvent = inputs[0]->getStream().putEvent(true, true);

    // The streams from all other inputs wait for the stream from input 0 to be ready
    for (uint i = 1; i < inputs.size(); ++i) {
        inputs[i]->getStream().waitEvent(processingFinishedEvent);
    }

    return processingFinishedEvent;
}

void StampedNetwork::clear() {
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        inputs[i]->cleanup();
    }
    inputs.clear();

    for (uint32_t i = 0; i < outputs.size(); ++i) {
        outputs[i]->cleanup();
    }
    outputs.clear();

    for (uint32_t i = 0; i < trainableLayers.size(); ++i) {
        trainableLayers[i]->cleanup();
    }
    trainableLayers.clear();

    for (uint32_t i = 0; i < otherLayers.size(); ++i) {
        otherLayers[i]->cleanup();
    }
    otherLayers.clear();

    apiTensorToPhysicalDrivingLayer.clear();
    apiLayerToPhysicalLayer.clear();
    physicalLayerToApiLayer.clear();
    apiTensorToApiDrivingLayer.clear();
    inputNamed.clear();
    outputNamed.clear();

    inputsShared.clear();
    outputsShared.clear();
    trainableLayersShared.clear();
    otherLayersShared.clear();
    apiTensorToPhysicalDrivingLayerShared.clear();
    apiLayerToPhysicalLayerShared.clear();
    physicalLayerToApiLayerShared.clear();
    apiTensorToApiDrivingLayerShared.clear();
    inputNamedShared.clear();
    outputNamedShared.clear();
}

}  // namespace ThorImplementation
