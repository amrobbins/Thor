#include "DeepLearning/Api/Training/PhaseGraphConnector.h"

#include "DeepLearning/Api/Layers/Utility/NetworkInput.h"
#include "DeepLearning/Api/Layers/Utility/NetworkOutput.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

using namespace std;

namespace Thor {
namespace {

string phaseContext(const PhaseGraphNetworkSpec& spec) {
    if (!spec.phaseName.empty()) {
        return "phase '" + spec.phaseName + "'";
    }
    if (spec.network != nullptr) {
        return "network '" + spec.network->getNetworkName() + "'";
    }
    return "unnamed phase";
}

vector<shared_ptr<NetworkInput>> apiNetworkInputs(const Network& networkConst, bool includePassThroughInputs) {
    Network& network = const_cast<Network&>(networkConst);
    vector<shared_ptr<NetworkInput>> inputs;
    const uint32_t numLayers = network.getNumLayers();
    inputs.reserve(numLayers);
    for (uint32_t layerIndex = 0; layerIndex < numLayers; ++layerIndex) {
        shared_ptr<NetworkInput> input = dynamic_pointer_cast<NetworkInput>(network.getLayer(layerIndex));
        if (input == nullptr) {
            continue;
        }
        if (!includePassThroughInputs && input->hasPassThroughSource()) {
            continue;
        }
        inputs.push_back(input);
    }
    return inputs;
}

vector<shared_ptr<NetworkOutput>> apiNetworkOutputs(const Network& networkConst) {
    Network& network = const_cast<Network&>(networkConst);
    vector<shared_ptr<NetworkOutput>> outputs;
    const uint32_t numLayers = network.getNumLayers();
    outputs.reserve(numLayers);
    for (uint32_t layerIndex = 0; layerIndex < numLayers; ++layerIndex) {
        shared_ptr<NetworkOutput> output = dynamic_pointer_cast<NetworkOutput>(network.getLayer(layerIndex));
        if (output == nullptr) {
            continue;
        }
        outputs.push_back(output);
    }
    return outputs;
}

struct ActivePhaseRecord {
    PhaseGraphNetworkSpec spec;
    vector<shared_ptr<NetworkInput>> inputs;
    vector<shared_ptr<NetworkOutput>> outputs;
    size_t activeIndex = 0;
};

struct OutputProducerRecord {
    size_t activePhaseIndex = 0;
    shared_ptr<NetworkOutput> output;
};

Tensor outputProducerTensor(const NetworkOutput& output) {
    THOR_THROW_IF_FALSE(output.getFeatureInput().has_value());
    return output.getFeatureInput().value();
}

void validateProducerConsumerDescriptors(const string& tensorName,
                                         const NetworkOutput& producerOutput,
                                         const NetworkInput& consumerInput,
                                         const string& producerPhaseName,
                                         const string& consumerPhaseName) {
    Tensor producerTensor = outputProducerTensor(producerOutput);
    if (producerTensor.getDimensions() != consumerInput.getDimensions() || producerTensor.getDataType() != consumerInput.getDataType()) {
        throw runtime_error("Phase graph input '" + tensorName + "' in phase '" + consumerPhaseName +
                            "' is incompatible with output '" + tensorName + "' from phase '" + producerPhaseName + "'.");
    }
}

Tensor ensureExternalInput(ComposedPhaseGraph& graph, const NetworkInput& sourceInput) {
    const string& inputName = sourceInput.getName();
    auto existingIt = graph.externalInputTensorsByName.find(inputName);
    if (existingIt != graph.externalInputTensorsByName.end()) {
        const Tensor& existing = existingIt->second;
        if (existing.getDimensions() != sourceInput.getDimensions() || existing.getDataType() != sourceInput.getDataType()) {
            throw runtime_error("Phase graph external input '" + inputName + "' has incompatible descriptors across active phases.");
        }
        return existing;
    }

    NetworkInput destinationInput = NetworkInput::Builder()
                                        .network(*graph.network)
                                        .name(inputName)
                                        .dimensions(sourceInput.getDimensions())
                                        .dataType(sourceInput.getDataType())
                                        .dimensionsIncludeBatch(sourceInput.dimensionsIncludeBatch())
                                        .external(true)
                                        .build();
    Tensor destinationTensor = destinationInput.getFeatureOutput().value();
    graph.externalInputTensorsByName[inputName] = destinationTensor;
    return destinationTensor;
}

vector<size_t> topologicalPhaseOrder(const vector<ActivePhaseRecord>& phases,
                                     const map<string, OutputProducerRecord>& outputProducerByName) {
    const size_t n = phases.size();
    vector<set<size_t>> dependents(n);
    vector<size_t> indegree(n, 0);

    for (size_t consumerIndex = 0; consumerIndex < n; ++consumerIndex) {
        const ActivePhaseRecord& consumer = phases[consumerIndex];
        for (const shared_ptr<NetworkInput>& input : consumer.inputs) {
            const string inputName = input->getName();
            auto producerIt = outputProducerByName.find(inputName);
            if (producerIt == outputProducerByName.end()) {
                if (!input->isExternal()) {
                    throw runtime_error("Phase graph non-external input '" + inputName + "' in phase '" + consumer.spec.phaseName +
                                        "' is not satisfied by any active phase output.");
                }
                continue;
            }

            const size_t producerIndex = producerIt->second.activePhaseIndex;
            if (producerIndex == consumerIndex) {
                if (!input->isExternal()) {
                    throw runtime_error("Phase graph non-external input '" + inputName + "' in phase '" + consumer.spec.phaseName +
                                        "' is only exported by the same phase. Phase graph links require another active phase producer.");
                }
                continue;
            }

            validateProducerConsumerDescriptors(inputName,
                                                *producerIt->second.output,
                                                *input,
                                                phases[producerIndex].spec.phaseName,
                                                consumer.spec.phaseName);
            if (dependents[producerIndex].insert(consumerIndex).second) {
                ++indegree[consumerIndex];
            }
        }
    }

    priority_queue<size_t, vector<size_t>, greater<size_t>> ready;
    for (size_t i = 0; i < n; ++i) {
        if (indegree[i] == 0) {
            ready.push(i);
        }
    }

    vector<size_t> order;
    order.reserve(n);
    while (!ready.empty()) {
        size_t current = ready.top();
        ready.pop();
        order.push_back(current);
        for (size_t dependent : dependents[current]) {
            THOR_THROW_IF_FALSE(indegree[dependent] > 0);
            --indegree[dependent];
            if (indegree[dependent] == 0) {
                ready.push(dependent);
            }
        }
    }

    if (order.size() != n) {
        throw runtime_error("Phase graph active phases contain a cycle through matching NetworkOutput/NetworkInput names.");
    }
    return order;
}

}  // namespace

ComposedPhaseGraph buildComposedPhaseGraphByName(const vector<PhaseGraphNetworkSpec>& phaseSpecs,
                                                 const PhaseGraphComposeOptions& options) {
    if (phaseSpecs.empty()) {
        throw runtime_error("Phase graph composition requires at least one phase spec.");
    }
    if (options.networkName.empty()) {
        throw runtime_error("Phase graph composition requires a non-empty destination network name.");
    }

    vector<ActivePhaseRecord> activePhases;
    activePhases.reserve(phaseSpecs.size());
    for (const PhaseGraphNetworkSpec& phaseSpec : phaseSpecs) {
        if (!phaseSpec.active) {
            continue;
        }
        if (phaseSpec.phaseName.empty()) {
            throw runtime_error("Phase graph composition requires every active phase to have a non-empty phaseName.");
        }
        if (phaseSpec.network == nullptr) {
            throw runtime_error("Phase graph composition received a null network for " + phaseContext(phaseSpec) + ".");
        }

        ActivePhaseRecord record;
        record.spec = phaseSpec;
        record.inputs = apiNetworkInputs(*phaseSpec.network, /*includePassThroughInputs=*/false);
        record.outputs = apiNetworkOutputs(*phaseSpec.network);
        record.activeIndex = activePhases.size();
        if (record.outputs.empty()) {
            throw runtime_error("Phase graph " + phaseContext(phaseSpec) + " has no NetworkOutput layers to export.");
        }

        set<string> inputNamesWithinPhase;
        for (const shared_ptr<NetworkInput>& input : record.inputs) {
            const string inputName = input->getName();
            if (inputName.empty()) {
                throw runtime_error("Phase graph " + phaseContext(phaseSpec) + " has a NetworkInput with an empty name.");
            }
            if (!inputNamesWithinPhase.insert(inputName).second) {
                throw runtime_error("Phase graph " + phaseContext(phaseSpec) + " contains duplicate NetworkInput name '" + inputName + "'.");
            }
        }

        set<string> outputNamesWithinPhase;
        for (const shared_ptr<NetworkOutput>& output : record.outputs) {
            const string outputName = output->getName();
            if (outputName.empty()) {
                throw runtime_error("Phase graph " + phaseContext(phaseSpec) + " has a NetworkOutput with an empty name.");
            }
            if (!outputNamesWithinPhase.insert(outputName).second) {
                throw runtime_error("Phase graph " + phaseContext(phaseSpec) + " exports duplicate output name '" + outputName + "'.");
            }
        }

        activePhases.push_back(record);
    }

    if (activePhases.empty()) {
        throw runtime_error("Phase graph composition requires at least one active phase.");
    }

    map<string, OutputProducerRecord> outputProducerByName;
    for (size_t phaseIndex = 0; phaseIndex < activePhases.size(); ++phaseIndex) {
        const ActivePhaseRecord& phase = activePhases[phaseIndex];
        for (const shared_ptr<NetworkOutput>& output : phase.outputs) {
            const string outputName = output->getName();
            if (outputProducerByName.count(outputName) != 0) {
                throw runtime_error("Phase graph output name collision: active phase '" + phase.spec.phaseName +
                                    "' exports '" + outputName + "', which was already exported by another active phase.");
            }
            outputProducerByName[outputName] = OutputProducerRecord{phaseIndex, output};
        }
    }

    const vector<size_t> phaseOrder = topologicalPhaseOrder(activePhases, outputProducerByName);

    ComposedPhaseGraph graph;
    graph.network = make_shared<Network>(options.networkName);

    map<string, Tensor> producedOutputTensorsByName;
    vector<tuple<string, Tensor, bool>> outputsInCreationOrder;

    for (size_t phaseIndex : phaseOrder) {
        const ActivePhaseRecord& phase = activePhases[phaseIndex];
        graph.activePhaseNames.push_back(phase.spec.phaseName);

        ApiTensorRemap remap;
        for (const shared_ptr<NetworkInput>& input : phase.inputs) {
            const string inputName = input->getName();
            auto producerIt = outputProducerByName.find(inputName);
            Tensor destinationTensor;
            if (producerIt != outputProducerByName.end() && producerIt->second.activePhaseIndex != phaseIndex) {
                auto producedIt = producedOutputTensorsByName.find(inputName);
                if (producedIt == producedOutputTensorsByName.end()) {
                    throw runtime_error("Phase graph internal error: producer output '" + inputName + "' was not cloned before consumer phase '" +
                                        phase.spec.phaseName + "'.");
                }
                destinationTensor = producedIt->second;
            } else {
                if (!input->isExternal()) {
                    throw runtime_error("Phase graph non-external input '" + inputName + "' in phase '" + phase.spec.phaseName +
                                        "' is not satisfied by another active phase output.");
                }
                destinationTensor = ensureExternalInput(graph, *input);
            }
            remap.map(input->getFeatureOutput().value(), destinationTensor);
        }

        vector<string> phaseOutputNames;
        phaseOutputNames.reserve(phase.outputs.size());
        for (const shared_ptr<NetworkOutput>& output : phase.outputs) {
            phaseOutputNames.push_back(output->getName());
        }

        ApiSubgraphCloneOptions cloneOptions;
        cloneOptions.namePrefix = phase.spec.phaseName + "/";
        cloneOptions.inferenceOnly = options.inferenceOnly;
        cloneOptions.cloneTrainableParameters = true;
        ApiSubgraphCloneResult cloneResult = graph.network->cloneInferenceSubgraphInto(*phase.spec.network, phaseOutputNames, remap, cloneOptions);

        for (const shared_ptr<NetworkOutput>& output : phase.outputs) {
            const string outputName = output->getName();
            auto outputIt = cloneResult.outputTensorsByName.find(outputName);
            if (outputIt == cloneResult.outputTensorsByName.end()) {
                throw runtime_error("Phase graph " + phaseContext(phase.spec) + " did not produce cloned output '" + outputName + "'.");
            }
            producedOutputTensorsByName[outputName] = outputIt->second;
            graph.outputTensorsByName[outputName] = outputIt->second;
            outputsInCreationOrder.emplace_back(outputName, outputIt->second, output->isExternal());
        }
    }

    if (options.exposePhaseOutputsAsNetworkOutputs) {
        for (const auto& [outputName, outputTensor, externalOutput] : outputsInCreationOrder) {
            if (!externalOutput) {
                continue;
            }
            NetworkOutput::Builder().network(*graph.network).name(outputName).inputTensor(outputTensor).external(true).build();
        }
    }

    return graph;
}

}  // namespace Thor
