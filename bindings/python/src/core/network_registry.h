#pragma once

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/Tensor.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

namespace Thor::PythonBindings {

inline std::vector<std::weak_ptr<Network>>& pythonNetworkRegistry() {
    static std::vector<std::weak_ptr<Network>> registry;
    return registry;
}

inline std::mutex& pythonNetworkRegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

inline void registerPythonNetwork(const std::shared_ptr<Network>& network) {
    if (network == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(pythonNetworkRegistryMutex());
    auto& registry = pythonNetworkRegistry();
    registry.erase(std::remove_if(registry.begin(), registry.end(), [](const std::weak_ptr<Network>& entry) { return entry.expired(); }),
                   registry.end());

    for (const std::weak_ptr<Network>& entry : registry) {
        if (std::shared_ptr<Network> existing = entry.lock(); existing != nullptr && existing.get() == network.get()) {
            return;
        }
    }
    registry.push_back(network);
}

inline std::vector<std::shared_ptr<Network>> pythonNetworksContainingAllTensors(const std::vector<Tensor>& tensors) {
    std::vector<std::shared_ptr<Network>> matches;
    if (tensors.empty()) {
        return matches;
    }

    std::lock_guard<std::mutex> lock(pythonNetworkRegistryMutex());
    auto& registry = pythonNetworkRegistry();
    registry.erase(std::remove_if(registry.begin(), registry.end(), [](const std::weak_ptr<Network>& entry) { return entry.expired(); }),
                   registry.end());

    for (const std::weak_ptr<Network>& entry : registry) {
        std::shared_ptr<Network> network = entry.lock();
        if (network == nullptr) {
            continue;
        }

        bool containsAll = true;
        for (const Tensor& tensor : tensors) {
            if (!tensor.isInitialized() || !network->hasApiTensorByOriginalId(tensor.getOriginalId())) {
                containsAll = false;
                break;
            }
        }
        if (containsAll) {
            matches.push_back(std::move(network));
        }
    }
    return matches;
}

}  // namespace Thor::PythonBindings
