#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adadelta.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Thor {

class Adadelta : public Optimizer {
   public:
    class Builder;
    Adadelta();
    Adadelta(uint64_t originalId);
    ~Adadelta() override = default;

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

    std::string getType() const override { return "Adadelta"; }

   protected:
    void updateParameters(PlacedNetwork* placedNetwork);

    std::shared_ptr<Optimizer> clone() const override;

   private:
    float alpha;
    float rho;
    float epsilon;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    std::optional<std::string> gradientSquareAverageFile;
    std::optional<std::string> updateSquareAverageFile;
};

class Adadelta::Builder {
   public:
    virtual std::shared_ptr<Adadelta> build() {
        if (!_alpha.has_value())
            _alpha = 1.0f;
        if (!_rho.has_value())
            _rho = 0.95f;
        if (!_epsilon.has_value())
            _epsilon = 1e-7f;

        Adadelta adadelta;
        adadelta.alpha = _alpha.value();
        adadelta.rho = _rho.value();
        adadelta.epsilon = _epsilon.value();

        THOR_THROW_IF_FALSE(adadelta.alpha > 0.0f);
        THOR_THROW_IF_FALSE(adadelta.rho >= 0.0f && adadelta.rho < 1.0f);
        THOR_THROW_IF_FALSE(adadelta.epsilon > 0.0f);

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.has_value() && _network.value() != nullptr) {
            adadelta.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<Adadelta>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<Adadelta>(_network.value()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<Adadelta>(adadelta.clone());
        }
    }
    virtual Adadelta::Builder& network(Network& _network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual Adadelta::Builder& alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        this->_alpha = _alpha;
        return *this;
    }
    virtual Adadelta::Builder& rho(float _rho) {
        THOR_THROW_IF_FALSE(!this->_rho.has_value());
        this->_rho = _rho;
        return *this;
    }
    virtual Adadelta::Builder& epsilon(float _epsilon) {
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
