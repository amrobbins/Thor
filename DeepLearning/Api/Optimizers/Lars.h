#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Lars.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class Lars : public Optimizer {
   public:
    class Builder;
    Lars();
    Lars(uint64_t originalId);
    ~Lars() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;
    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                  std::optional<Event> sisterOptimizerLoadedEvent) override;

    virtual void setAlpha(float newAlpha, PlacedNetwork* placedNetwork);
    virtual float getAlpha();
    virtual void setMomentum(float newMomentum, PlacedNetwork* placedNetwork);
    virtual float getMomentum();
    virtual void setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork);
    virtual float getWeightDecay();
    virtual void setTrustCoefficient(float newTrustCoefficient, PlacedNetwork* placedNetwork);
    virtual float getTrustCoefficient();
    virtual void setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork);
    virtual float getEpsilon();
    virtual void setUseNesterovMomentum(bool newUseNesterovMomentum, PlacedNetwork* placedNetwork);
    virtual bool getUseNesterovMomentum();

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                             std::string filenamePrefix,
                             bool saveOptimizerState) const override;
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader,
                                                  const nlohmann::json& j,
                                                  Network* network);

    nlohmann::json architectureJson() const override;

    std::string getType() const override { return "Lars"; }

   protected:
    void updateParameters(PlacedNetwork* placedNetwork);

    std::shared_ptr<Optimizer> clone() const override;

   private:
    float alpha;
    float momentum;
    float weightDecay;
    float trustCoefficient;
    float epsilon;
    bool useNesterovMomentum;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    std::optional<std::string> velocityFile;
};

class Lars::Builder {
   public:
    virtual std::shared_ptr<Lars> build() {
        if (!_alpha.has_value())
            _alpha = 0.01f;
        if (!_momentum.has_value())
            _momentum = 0.9f;
        if (!_weightDecay.has_value())
            _weightDecay = 0.0f;
        if (!_trustCoefficient.has_value())
            _trustCoefficient = 0.001f;
        if (!_epsilon.has_value())
            _epsilon = 1e-8f;
        if (!_useNesterovMomentum.has_value())
            _useNesterovMomentum = false;

        Lars lars;
        lars.alpha = _alpha.value();
        lars.momentum = _momentum.value();
        lars.weightDecay = _weightDecay.value();
        lars.trustCoefficient = _trustCoefficient.value();
        lars.epsilon = _epsilon.value();
        lars.useNesterovMomentum = _useNesterovMomentum.value();

        THOR_THROW_IF_FALSE(lars.alpha > 0.0f);
        THOR_THROW_IF_FALSE(lars.momentum >= 0.0f);
        THOR_THROW_IF_FALSE(lars.weightDecay >= 0.0f);
        THOR_THROW_IF_FALSE(lars.trustCoefficient > 0.0f);
        THOR_THROW_IF_FALSE(lars.epsilon > 0.0f);

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.has_value() && _network.value() != nullptr) {
            lars.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<Lars>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<Lars>(_network.value()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<Lars>(lars.clone());
        }
    }
    virtual Lars::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual Lars::Builder& alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        this->_alpha = _alpha;
        return *this;
    }
    virtual Lars::Builder& momentum(float _momentum) {
        THOR_THROW_IF_FALSE(!this->_momentum.has_value());
        this->_momentum = _momentum;
        return *this;
    }
    virtual Lars::Builder& weightDecay(float _weightDecay) {
        THOR_THROW_IF_FALSE(!this->_weightDecay.has_value());
        this->_weightDecay = _weightDecay;
        return *this;
    }
    virtual Lars::Builder& trustCoefficient(float _trustCoefficient) {
        THOR_THROW_IF_FALSE(!this->_trustCoefficient.has_value());
        this->_trustCoefficient = _trustCoefficient;
        return *this;
    }
    virtual Lars::Builder& epsilon(float _epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = _epsilon;
        return *this;
    }
    virtual Lars::Builder& useNesterovMomentum(bool _useNesterovMomentum) {
        THOR_THROW_IF_FALSE(!this->_useNesterovMomentum.has_value());
        this->_useNesterovMomentum = _useNesterovMomentum;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<float> _alpha;
    std::optional<float> _momentum;
    std::optional<float> _weightDecay;
    std::optional<float> _trustCoefficient;
    std::optional<float> _epsilon;
    std::optional<bool> _useNesterovMomentum;
};

}  // namespace Thor
