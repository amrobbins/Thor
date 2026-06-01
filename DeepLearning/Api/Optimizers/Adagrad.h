#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adagrad.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class Adagrad : public Optimizer {
   public:
    class Builder;
    Adagrad();
    Adagrad(uint64_t originalId);
    ~Adagrad() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;
    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                  std::optional<Event> sisterOptimizerLoadedEvent) override;

    virtual void setAlpha(float newAlpha, PlacedNetwork* placedNetwork);
    virtual float getAlpha();
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

    std::string getType() const override { return "Adagrad"; }

   protected:
    void updateParameters(PlacedNetwork* placedNetwork);

    std::shared_ptr<Optimizer> clone() const override;

   private:
    float alpha;
    float epsilon;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    std::optional<std::string> accumulatorFile;
};

class Adagrad::Builder {
   public:
    virtual std::shared_ptr<Adagrad> build() {
        if (!_alpha.has_value())
            _alpha = 0.01f;
        if (!_epsilon.has_value())
            _epsilon = 1e-7f;

        Adagrad adagrad;
        adagrad.alpha = _alpha.value();
        adagrad.epsilon = _epsilon.value();

        THOR_THROW_IF_FALSE(adagrad.alpha > 0.0f);
        THOR_THROW_IF_FALSE(adagrad.epsilon > 0.0f);

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.has_value() && _network.value() != nullptr) {
            adagrad.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<Adagrad>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<Adagrad>(_network.value()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<Adagrad>(adagrad.clone());
        }
    }
    virtual Adagrad::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual Adagrad::Builder& alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        this->_alpha = _alpha;
        return *this;
    }
    virtual Adagrad::Builder& epsilon(float _epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = _epsilon;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<float> _alpha;
    std::optional<float> _epsilon;
};

}  // namespace Thor
