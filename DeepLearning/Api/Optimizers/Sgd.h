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

    Sgd();
    Sgd(uint64_t originalId);

    virtual ~Sgd() = default;

    virtual std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer);

    virtual void setConstantLearningRate(float newCurrentLearningRate, PlacedNetwork *placedNetwork);
    virtual void setInitialLearningRate(float newInitialLearningRate, PlacedNetwork *placedNetwork);
    virtual float getInitialLearningRate();
    virtual void setDecay(float newDecay, PlacedNetwork *placedNetwork);
    virtual float getDecay();
    virtual void setMomentum(float newMomentum, PlacedNetwork *placedNetwork);
    virtual float getMomentum();
    virtual void setUseNesterovMomentum(bool newUseNesterovMomentum, PlacedNetwork *placedNetwork);
    virtual bool getUseNesterovMomentum();
    virtual uint64_t getEpoch();

    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                     std::string filenamePrefix,
                                     bool saveOptimizerState) const;
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader,
                                                  const nlohmann::json &j,
                                                  Network *network);

    virtual nlohmann::json architectureJson() const;

    virtual std::string getType() const { return "SGD"; }

   protected:
    virtual std::shared_ptr<Optimizer> clone() const;

    void updateParameters(PlacedNetwork *placedNetwork);

   private:
    float initialLearningRate;
    float decay;
    float momentum;
    bool useNesterovMomentum;
    uint64_t startResumeEpoch = 0;
};

class Sgd::Builder {
   public:
    virtual std::shared_ptr<Sgd> build() {
        if (_initialLearningRate.isEmpty())
            _initialLearningRate = 0.01f;
        if (_decay.isEmpty())
            _decay = 0.0f;
        if (_momentum.isEmpty())
            _momentum = 0.0f;
        if (_useNesterovMomentum.isEmpty())
            _useNesterovMomentum = false;

        assert(_initialLearningRate > 0.0f);
        assert(_decay < 1.0f);
        assert(_decay >= 0.0f);
        assert(_momentum >= 0.0f);

        Sgd sgd;
        sgd.initialLearningRate = _initialLearningRate;
        sgd.decay = _decay;
        sgd.momentum = _momentum;
        sgd.useNesterovMomentum = _useNesterovMomentum;

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.isPresent() && _network.get() != nullptr) {
            sgd.addToNetwork(_network);
            assert(std::dynamic_pointer_cast<Sgd>(_network.get()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<Sgd>(_network.get()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<Sgd>(sgd.clone());
        }
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
