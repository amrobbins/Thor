#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adafactor.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class Adafactor : public Optimizer {
   public:
    class Builder;
    Adafactor();
    Adafactor(uint64_t originalId);
    ~Adafactor() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;
    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                  std::optional<Event> sisterOptimizerLoadedEvent) override;

    virtual void setAlpha(float newAlpha, PlacedNetwork* placedNetwork);
    virtual float getAlpha();
    virtual void setBeta2(float newBeta2, PlacedNetwork* placedNetwork);
    virtual float getBeta2();
    virtual void setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork);
    virtual float getEpsilon();
    virtual void setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork);
    virtual float getWeightDecay();
    virtual void setFactorSecondMoment(bool newFactorSecondMoment, PlacedNetwork* placedNetwork);
    virtual bool getFactorSecondMoment();

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                             std::string filenamePrefix,
                             bool saveOptimizerState) const override;
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader,
                                                  const nlohmann::json& j,
                                                  Network* network);

    nlohmann::json architectureJson() const override;

    std::string getType() const override { return "Adafactor"; }

   protected:
    void updateParameters(PlacedNetwork* placedNetwork);

    std::shared_ptr<Optimizer> clone() const override;

   private:
    float alpha;
    float beta2;
    float epsilon;
    float weightDecay;
    bool factorSecondMoment;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    std::optional<std::string> selectedOptimizerStateKind;
    std::optional<std::string> secondMomentFile;
    std::optional<std::string> rowSecondMomentFile;
    std::optional<std::string> columnSecondMomentFile;
};

class Adafactor::Builder {
   public:
    virtual std::shared_ptr<Adafactor> build() {
        if (!_alpha.has_value())
            _alpha = 0.001f;
        if (!_beta2.has_value())
            _beta2 = 0.999f;
        if (!_epsilon.has_value())
            _epsilon = 1e-30f;
        if (!_weightDecay.has_value())
            _weightDecay = 0.0f;
        if (!_factorSecondMoment.has_value())
            _factorSecondMoment = true;

        Adafactor adafactor;
        adafactor.alpha = _alpha.value();
        adafactor.beta2 = _beta2.value();
        adafactor.epsilon = _epsilon.value();
        adafactor.weightDecay = _weightDecay.value();
        adafactor.factorSecondMoment = _factorSecondMoment.value();

        THOR_THROW_IF_FALSE(adafactor.alpha > 0.0f);
        THOR_THROW_IF_FALSE(adafactor.beta2 >= 0.0f && adafactor.beta2 < 1.0f);
        THOR_THROW_IF_FALSE(adafactor.epsilon > 0.0f);
        THOR_THROW_IF_FALSE(adafactor.weightDecay >= 0.0f);

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.has_value() && _network.value() != nullptr) {
            adafactor.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<Adafactor>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<Adafactor>(_network.value()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<Adafactor>(adafactor.clone());
        }
    }
    virtual Adafactor::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual Adafactor::Builder& alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        this->_alpha = _alpha;
        return *this;
    }
    virtual Adafactor::Builder& beta2(float _beta2) {
        THOR_THROW_IF_FALSE(!this->_beta2.has_value());
        this->_beta2 = _beta2;
        return *this;
    }
    virtual Adafactor::Builder& epsilon(float _epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = _epsilon;
        return *this;
    }
    virtual Adafactor::Builder& weightDecay(float _weightDecay) {
        THOR_THROW_IF_FALSE(!this->_weightDecay.has_value());
        this->_weightDecay = _weightDecay;
        return *this;
    }
    virtual Adafactor::Builder& factorSecondMoment(bool _factorSecondMoment) {
        THOR_THROW_IF_FALSE(!this->_factorSecondMoment.has_value());
        this->_factorSecondMoment = _factorSecondMoment;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<float> _alpha;
    std::optional<float> _beta2;
    std::optional<float> _epsilon;
    std::optional<float> _weightDecay;
    std::optional<bool> _factorSecondMoment;
};

}  // namespace Thor
