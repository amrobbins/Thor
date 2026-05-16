#include "DeepLearning/Api/Network/PlacedNetwork.h"

#include "Utilities/Common/Event.h"

#include <utility>

using namespace std;
using json = nlohmann::json;

namespace Thor {

PlacedNetwork::~PlacedNetwork() {
    for (uint32_t i = 0; i < stampedNetworks.size(); ++i) {
        // Calls parentCleanup then cleanUp then clears all the shared pointers:
        stampedNetworks[i].clear();
    }
    stampedNetworks.clear();
}

void PlacedNetwork::save(const std::string &directory, bool overwrite, bool saveOptimizerState) {
    network.save(stampedNetworks, directory, overwrite, saveOptimizerState);
}

std::map<std::string, ThorImplementation::Tensor> PlacedNetwork::infer(std::map<std::string, ThorImplementation::Tensor> batchInputs,
                                                                       uint64_t stampIndex) {
    THOR_THROW_IF_FALSE(stampIndex < stampedNetworks.size());

    std::map<std::string, ThorImplementation::Tensor> batchOutputs;
    std::map<std::string, Event> outputReadyEvents;
    Event done = stampedNetworks[stampIndex].sendBatch(std::move(batchInputs), batchOutputs, outputReadyEvents, true);
    done.synchronize();
    return batchOutputs;
}

}  // namespace Thor
