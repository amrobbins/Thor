#pragma once

#include "DeepLearning/Api/Network/Network.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Thor {

struct PhaseGraphNetworkSpec {
    std::string phaseName;
    std::shared_ptr<Network> network;
    bool active = true;
};

struct PhaseGraphComposeOptions {
    std::string networkName = "composed_phase_graph";
    bool inferenceOnly = false;
    bool exposePhaseOutputsAsNetworkOutputs = true;
};

struct ComposedPhaseGraph {
    std::shared_ptr<Network> network;
    std::map<std::string, Tensor> externalInputTensorsByName;
    std::map<std::string, Tensor> outputTensorsByName;
    std::vector<std::string> activePhaseNames;
};

ComposedPhaseGraph buildComposedPhaseGraphByName(const std::vector<PhaseGraphNetworkSpec>& phaseSpecs,
                                                 const PhaseGraphComposeOptions& options = PhaseGraphComposeOptions{});

}  // namespace Thor
