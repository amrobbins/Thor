#pragma once

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Adam.h"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

namespace Thor {

class Adam : public Optimizer {
   public:
    class Builder;
    virtual ~Adam() = default;

    virtual std::shared_ptr<ThorImplementation::Optimizer> stamp(
        std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> trainableLayer);
    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                          Optional<Event> sisterOptimizerLoadedEvent);
    virtual void setAlpha(float newAlpha);
    virtual float getAlpha();
    virtual void setBeta1(float newBeta1);
    virtual float getBeta1();
    virtual void setBeta2(float newBeta2);
    virtual float getBeta2();
    virtual void setEpsilon(float newEpsilon);
    virtual float getEpsilon();

    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     TrainableWeightsBiasesLayer const *owningLayer,
                                     std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalOwningLayer,
                                     bool saveOptimizerState) const;
    static std::shared_ptr<Optimizer> deserialize(thor_file::TarReader &archiveReader, const nlohmann::json &j);

   protected:
    void updateParameters();

    virtual std::shared_ptr<Optimizer> clone() const;

   private:
    float t = 0.0f;
    float alpha;
    float beta1;
    float beta2;
    float epsilon;

    thor_file::TarReader *archiveReader = nullptr;
    Optional<std::string> mFile;
    Optional<std::string> vFile;
    Optional<std::string> mBiasFile;
    Optional<std::string> vBiasFile;
};

class Adam::Builder {
   public:
    virtual std::shared_ptr<Adam> build() {
        if (_alpha.isEmpty())
            _alpha = 0.001f;
        if (_beta1.isEmpty())
            _beta1 = 0.9f;
        if (_beta2.isEmpty())
            _beta2 = 0.999f;
        if (_epsilon.isEmpty())
            _epsilon = 1e-7f;
        Adam adam;
        adam.alpha = _alpha;
        adam.beta1 = _beta1;
        adam.beta2 = _beta2;
        adam.epsilon = _epsilon;
        assert(_network.isPresent());
        assert(_network.get() != nullptr);
        adam.addToNetwork(_network);
        assert(std::dynamic_pointer_cast<Adam>(_network.get()->getOptimizer()) != nullptr);
        return std::dynamic_pointer_cast<Adam>(_network.get()->getOptimizer());
    }
    virtual Adam::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }
    virtual Adam::Builder alpha(float _alpha) {
        assert(!this->_alpha.isPresent());
        this->_alpha = _alpha;
        return *this;
    }
    virtual Adam::Builder beta1(float _beta1) {
        assert(!this->_beta1.isPresent());
        this->_beta1 = _beta1;
        return *this;
    }
    virtual Adam::Builder beta2(float _beta2) {
        assert(!this->_beta2.isPresent());
        this->_beta2 = _beta2;
        return *this;
    }
    virtual Adam::Builder epsilon(float _epsilon) {
        assert(!this->_epsilon.isPresent());
        this->_epsilon = _epsilon;
        return *this;
    }

   private:
    Optional<Network *> _network;
    Optional<float> _alpha;
    Optional<float> _beta1;
    Optional<float> _beta2;
    Optional<float> _epsilon;
};

}  // namespace Thor
