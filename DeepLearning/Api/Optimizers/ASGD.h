#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/ASGD.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class ASGD : public Optimizer {
   public:
    class Builder;
    ASGD();
    ASGD(uint64_t originalId);
    ~ASGD() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;
    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                  std::optional<Event> sisterOptimizerLoadedEvent) override;

    virtual void setAlpha(float newAlpha, PlacedNetwork* placedNetwork);
    virtual float getAlpha();
    virtual void setLambd(float newLambd, PlacedNetwork* placedNetwork);
    virtual float getLambd();
    virtual void setPower(float newPower, PlacedNetwork* placedNetwork);
    virtual float getPower();
    virtual void setT0(float newT0, PlacedNetwork* placedNetwork);
    virtual float getT0();
    virtual void setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork);
    virtual float getWeightDecay();
    virtual float getT();

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                             std::string filenamePrefix,
                             bool saveOptimizerState) const override;
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader,
                                                  const nlohmann::json& j,
                                                  Network* network);

    nlohmann::json architectureJson() const override;

    std::string getType() const override { return "ASGD"; }

   protected:
    void updateParameters(PlacedNetwork* placedNetwork);

    std::shared_ptr<Optimizer> clone() const override;

   private:
    float t = 0.0f;
    float alpha;
    float lambd;
    float power;
    float t0;
    float weightDecay;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    std::optional<std::string> averagedWeightsFile;
};

class ASGD::Builder {
   public:
    virtual std::shared_ptr<ASGD> build() {
        if (!_alpha.has_value())
            _alpha = 0.01f;
        if (!_lambd.has_value())
            _lambd = 1e-4f;
        if (!_power.has_value())
            _power = 0.75f;
        if (!_t0.has_value())
            _t0 = 1e6f;
        if (!_weightDecay.has_value())
            _weightDecay = 0.0f;

        ASGD asgd;
        asgd.alpha = _alpha.value();
        asgd.lambd = _lambd.value();
        asgd.power = _power.value();
        asgd.t0 = _t0.value();
        asgd.weightDecay = _weightDecay.value();

        THOR_THROW_IF_FALSE(asgd.alpha > 0.0f);
        THOR_THROW_IF_FALSE(asgd.lambd >= 0.0f);
        THOR_THROW_IF_FALSE(asgd.power >= 0.0f);
        THOR_THROW_IF_FALSE(asgd.t0 >= 1.0f);
        THOR_THROW_IF_FALSE(asgd.weightDecay >= 0.0f);

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.has_value() && _network.value() != nullptr) {
            asgd.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<ASGD>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<ASGD>(_network.value()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<ASGD>(asgd.clone());
        }
    }
    virtual ASGD::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual ASGD::Builder& alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        this->_alpha = _alpha;
        return *this;
    }
    virtual ASGD::Builder& lambd(float _lambd) {
        THOR_THROW_IF_FALSE(!this->_lambd.has_value());
        this->_lambd = _lambd;
        return *this;
    }
    virtual ASGD::Builder& power(float _power) {
        THOR_THROW_IF_FALSE(!this->_power.has_value());
        this->_power = _power;
        return *this;
    }
    virtual ASGD::Builder& t0(float _t0) {
        THOR_THROW_IF_FALSE(!this->_t0.has_value());
        this->_t0 = _t0;
        return *this;
    }
    virtual ASGD::Builder& weightDecay(float _weightDecay) {
        THOR_THROW_IF_FALSE(!this->_weightDecay.has_value());
        this->_weightDecay = _weightDecay;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<float> _alpha;
    std::optional<float> _lambd;
    std::optional<float> _power;
    std::optional<float> _t0;
    std::optional<float> _weightDecay;
};

}  // namespace Thor
