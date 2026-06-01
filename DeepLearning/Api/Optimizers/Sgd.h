#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Sgd.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace Thor {

class Sgd : public Optimizer {
   public:
    class Builder;

    Sgd();
    Sgd(uint64_t originalId);

    ~Sgd() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;

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

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                             std::string filenamePrefix,
                             bool saveOptimizerState) const override;
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader,
                                                  const nlohmann::json &j,
                                                  Network *network);

    nlohmann::json architectureJson() const override;

    std::string getType() const override { return "SGD"; }

   protected:
    std::shared_ptr<Optimizer> clone() const override;

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
        if (!_initialLearningRate.has_value())
            _initialLearningRate = 0.01f;
        if (!_decay.has_value())
            _decay = 0.0f;
        if (!_momentum.has_value())
            _momentum = 0.0f;
        if (!_useNesterovMomentum.has_value())
            _useNesterovMomentum = false;

        THOR_THROW_IF_FALSE(_initialLearningRate.value() > 0.0f);
        THOR_THROW_IF_FALSE(_decay.value() < 1.0f);
        THOR_THROW_IF_FALSE(_decay.value() >= 0.0f);
        THOR_THROW_IF_FALSE(_momentum.value() >= 0.0f);

        Sgd sgd;
        sgd.initialLearningRate = _initialLearningRate.value();
        sgd.decay = _decay.value();
        sgd.momentum = _momentum.value();
        sgd.useNesterovMomentum = _useNesterovMomentum.value();

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.has_value() && _network.value() != nullptr) {
            sgd.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<Sgd>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<Sgd>(_network.value()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<Sgd>(sgd.clone());
        }
    }

    virtual Sgd::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual Sgd::Builder &initialLearningRate(float _initialLearningRate) {
        THOR_THROW_IF_FALSE(!this->_initialLearningRate.has_value());
        this->_initialLearningRate = _initialLearningRate;
        return *this;
    }

    virtual Sgd::Builder &decay(float _decay) {
        THOR_THROW_IF_FALSE(!this->_decay.has_value());
        this->_decay = _decay;
        return *this;
    }

    virtual Sgd::Builder &momentum(float _momentum) {
        THOR_THROW_IF_FALSE(!this->_momentum.has_value());
        this->_momentum = _momentum;
        return *this;
    }

    virtual Sgd::Builder &useNesterovMomentum(bool _useNesterovMomentum) {
        THOR_THROW_IF_FALSE(!this->_useNesterovMomentum.has_value());
        this->_useNesterovMomentum = _useNesterovMomentum;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<float> _initialLearningRate;
    std::optional<float> _decay;
    std::optional<float> _momentum;
    std::optional<bool> _useNesterovMomentum;
};

}  // namespace Thor
