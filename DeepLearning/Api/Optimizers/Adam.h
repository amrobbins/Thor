#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <optional>

namespace Thor {

class Adam : public Optimizer {
   public:
    class Builder;
    Adam();
    Adam(uint64_t originalId);
    ~Adam() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;
    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                          std::optional<Event> sisterOptimizerLoadedEvent) override;
    virtual void setAlpha(float newAlpha, PlacedNetwork *placedNetwork);
    virtual float getAlpha();
    virtual void setBeta1(float newBeta1, PlacedNetwork *placedNetwork);
    virtual float getBeta1();
    virtual void setBeta2(float newBeta2, PlacedNetwork *placedNetwork);
    virtual float getBeta2();
    virtual void setEpsilon(float newEpsilon, PlacedNetwork *placedNetwork);
    virtual float getEpsilon();

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                     std::string filenamePrefix,
                                     bool saveOptimizerState) const override;
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader,
                                                  const nlohmann::json &j,
                                                  Network *network);

    nlohmann::json architectureJson() const override;

    std::string getType() const override { return "Adam"; }

   protected:
    void updateParameters(PlacedNetwork *placedNetwork);

    std::shared_ptr<Optimizer> clone() const override;

   private:
    float t = 0.0f;
    float alpha;
    float beta1;
    float beta2;
    float epsilon;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    std::optional<std::string> mFile;
    std::optional<std::string> vFile;
    std::optional<std::string> mBiasFile;
    std::optional<std::string> vBiasFile;
};

class Adam::Builder {
   public:
    virtual std::shared_ptr<Adam> build() {
        if (!_alpha.has_value())
            _alpha = 0.001f;
        if (!_beta1.has_value())
            _beta1 = 0.9f;
        if (!_beta2.has_value())
            _beta2 = 0.999f;
        if (!_epsilon.has_value())
            _epsilon = 1e-7f;
        Adam adam;
        adam.alpha = _alpha.value();
        adam.beta1 = _beta1.value();
        adam.beta2 = _beta2.value();
        adam.epsilon = _epsilon.value();

        THOR_THROW_IF_FALSE(adam.alpha > 0.0f);
        THOR_THROW_IF_FALSE(adam.beta1 >= 0.0f && adam.beta1 < 1.0f);
        THOR_THROW_IF_FALSE(adam.beta2 >= 0.0f && adam.beta2 < 1.0f);
        THOR_THROW_IF_FALSE(adam.epsilon > 0);

        // When network is passed to the builder, this optimizer becomes the network default optimizer:
        if (_network.has_value() && _network.value() != nullptr) {
            adam.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<Adam>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<Adam>(_network.value()->getDefaultOptimizer());
        } else {
            return std::dynamic_pointer_cast<Adam>(adam.clone());
        }
    }
    virtual Adam::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }
    virtual Adam::Builder &alpha(float _alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        this->_alpha = _alpha;
        return *this;
    }
    virtual Adam::Builder &beta1(float _beta1) {
        THOR_THROW_IF_FALSE(!this->_beta1.has_value());
        this->_beta1 = _beta1;
        return *this;
    }
    virtual Adam::Builder &beta2(float _beta2) {
        THOR_THROW_IF_FALSE(!this->_beta2.has_value());
        this->_beta2 = _beta2;
        return *this;
    }
    virtual Adam::Builder &epsilon(float _epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = _epsilon;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<float> _alpha;
    std::optional<float> _beta1;
    std::optional<float> _beta2;
    std::optional<float> _epsilon;
};

}  // namespace Thor
