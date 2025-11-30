#pragma once

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

namespace Thor {

class Sgd : public Optimizer {
   public:
    class Builder;

    virtual ~Sgd();

    virtual void setConstantLearningRate(float newCurrentLearningRate);
    virtual void setInitialLearningRate(float newInitialLearningRate);
    virtual float getInitialLearningRate();
    virtual void setDecay(float newDecay);
    virtual float getDecay();
    virtual void setMomentum(float newMomentum);
    virtual float getMomentum();
    virtual void setUseNesterovMomentum(bool newUseNesterovMomentum);
    virtual bool getUseNesterovMomentum();

    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                          Optional<Event> sisterOptimizerLoadedEvent) {
        return {};
    }

   protected:
    virtual std::shared_ptr<ThorImplementation::Optimizer> stamp(
        std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer);

    virtual std::shared_ptr<Optimizer> clone() const;

    void updateParameters();

   private:
    float initialLearningRate;
    float decay;
    float momentum;
    bool useNesterovMomentum;
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
        if (_useNesterovMomentum.isEmpty())
            _useNesterovMomentum = true;

        assert(_initialLearningRate > 0.0f);
        assert(_decay < 1.0f);
        assert(_decay >= 0.0f);
        assert(_momentum >= 0.0f);

        Sgd sgd;
        sgd.initialLearningRate = _initialLearningRate;
        sgd.decay = _decay;
        sgd.momentum = _momentum;
        sgd.useNesterovMomentum = _useNesterovMomentum;

        assert(_network.isPresent());
        assert(_network.get() != nullptr);
        sgd.addToNetwork(_network);

        // Network clones the Optimizer when it is added to the network
        assert(std::dynamic_pointer_cast<Sgd>(_network.get()->getOptimizer()) != nullptr);
        return std::dynamic_pointer_cast<Sgd>(_network.get()->getOptimizer());
    }

    virtual Sgd::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Sgd::Builder initialLearningRate(float _initialLearningRate) {
        assert(!this->_initialLearningRate.isPresent());
        this->_initialLearningRate = _initialLearningRate;
        return *this;
    }

    virtual Sgd::Builder decay(float _decay) {
        assert(!this->_decay.isPresent());
        this->_decay = _decay;
        return *this;
    }

    virtual Sgd::Builder momentum(float _momentum) {
        assert(!this->_momentum.isPresent());
        this->_momentum = _momentum;
        return *this;
    }

    virtual Sgd::Builder useNesterovMomentum(bool _useNesterovMomentum) {
        assert(!this->_useNesterovMomentum.isPresent());
        this->_useNesterovMomentum = _useNesterovMomentum;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<float> _initialLearningRate;
    Optional<float> _decay;
    Optional<float> _momentum;
    Optional<bool> _useNesterovMomentum;
};

}  // namespace Thor