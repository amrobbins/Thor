#pragma once

#include "DeepLearning/Api/Network/Network.h"
#include "Optimizer.h"

#include <cmath>
#include <string>
#include <unordered_map>

class Sgd : public Optimizer {
   public:
    class Builder;

    Sgd();
    virtual ~Sgd();

    virtual void setNetwork(Thor::Network *network);

    // returns a map of updated parameters
    std::unordered_map<std::string, float> updateParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);
    std::unordered_map<std::string, float> initializeStampedNetworkParameters(ThorImplementation::StampedNetwork &stampedNetwork,
                                                                              uint64_t epoch,
                                                                              uint64_t batch,
                                                                              uint64_t batchesPerEpoch);
    std::unordered_map<std::string, float> getAllParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch);

   private:
    float initialLearningRate;
    float decay;
    float momentum;
    bool useNesterov;
    uint64_t currentEpoch;
    bool parametersInitialized;

    Thor::Network *network;
};

class Sgd::Builder {
   public:
    virtual std::shared_ptr<Sgd> build() {
        if (_initialLearningRate.isEmpty())
            _initialLearningRate = 0.01f;
        if (_decay.isEmpty())
            _decay = 0.1f;
        if (_momentum.isEmpty())
            _momentum = 0.0f;
        if (_useNesterov.isEmpty())
            _useNesterov = true;

        assert(_initialLearningRate > 0.0f);
        assert(_decay < 1.0f);
        assert(_decay >= 0.0f);
        assert(_momentum >= 0.0f);

        std::shared_ptr<Sgd> sgd = std::make_shared<Sgd>();
        sgd->initialLearningRate = _initialLearningRate;
        sgd->decay = _decay;
        sgd->momentum = _momentum;
        sgd->useNesterov = _useNesterov;
        sgd->currentEpoch = 0;
        sgd->parametersInitialized = false;
        sgd->network = nullptr;

        return sgd;
    }

    Sgd::Builder initialLearningRate(float _initialLearningRate) {
        assert(!this->_initialLearningRate.isPresent());
        this->_initialLearningRate = _initialLearningRate;
        return *this;
    }

    Sgd::Builder decay(float _decay) {
        assert(!this->_decay.isPresent());
        this->_decay = _decay;
        return *this;
    }

    Sgd::Builder momentum(float _momentum) {
        assert(!this->_momentum.isPresent());
        this->_momentum = _momentum;
        return *this;
    }

    Sgd::Builder useNesterov(bool _useNesterov) {
        assert(!this->_useNesterov.isPresent());
        this->_useNesterov = _useNesterov;
        return *this;
    }

   private:
    Optional<float> _initialLearningRate;
    Optional<float> _decay;
    Optional<float> _momentum;
    Optional<bool> _useNesterov;
};
