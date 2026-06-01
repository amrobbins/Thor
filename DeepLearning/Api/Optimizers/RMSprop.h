#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/RMSprop.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class RMSprop : public Optimizer {
   public:
    class Builder;
    RMSprop();
    RMSprop(uint64_t originalId);
    ~RMSprop() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;
    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                  std::optional<Event> sisterOptimizerLoadedEvent) override;

    virtual void setAlpha(float newAlpha, PlacedNetwork* placedNetwork);
    virtual float getAlpha();
    virtual void setRho(float newRho, PlacedNetwork* placedNetwork);
    virtual float getRho();
    virtual void setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork);
    virtual float getEpsilon();

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                             std::string filenamePrefix,
                             bool saveOptimizerState) const override;
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader,
                                                  const nlohmann::json& j,
                                                  Network* network);

    nlohmann::json architectureJson() const override;

    std::string getType() const override { return "RMSprop"; }

   protected:
    void updateParameters(PlacedNetwork* placedNetwork);

    std::shared_ptr<Optimizer> clone() const override;

   private:
    float alpha;
    float rho;
    float epsilon;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    std::optional<std::string> squareAverageFile;
};

class RMSprop::Builder {
   public:
    virtual std::shared_ptr<RMSprop> build() {
        if (!_alpha.has_value())
            _alpha = 0.001f;
        if (!_rho.has_value())
            _rho = 0.9f;
        if (!_epsilon.has_value())
            _epsilon = 1e-7f;

        RMSprop rmsprop;
        rmsprop.alpha = _alpha.value();
        rmsprop.rho = _rho.value();
        rmsprop.epsilon = _epsilon.value();

        THOR_THROW_IF_FALSE(rmsprop.alpha > 0.0f);
        THOR_THROW_IF_FALSE(rmsprop.rho >= 0.0f && rmsprop.rho < 1.0f);
        THOR_THROW_IF_FALSE(rmsprop.epsilon > 0.0f);

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.has_value() && _network.value() != nullptr) {
            rmsprop.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<RMSprop>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<RMSprop>(_network.value()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<RMSprop>(rmsprop.clone());
        }
    }
    virtual RMSprop::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual RMSprop::Builder& alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        this->_alpha = _alpha;
        return *this;
    }
    virtual RMSprop::Builder& rho(float _rho) {
        THOR_THROW_IF_FALSE(!this->_rho.has_value());
        this->_rho = _rho;
        return *this;
    }
    virtual RMSprop::Builder& epsilon(float _epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = _epsilon;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<float> _alpha;
    std::optional<float> _rho;
    std::optional<float> _epsilon;
};

}  // namespace Thor
