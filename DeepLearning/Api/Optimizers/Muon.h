#pragma once

#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/AdamW.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Muon.h"
#include "Utilities/Expression/NewtonSchulzOrthogonalization.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>

namespace Thor {

class Muon : public Optimizer {
   public:
    class Builder;
    Muon();
    Muon(uint64_t originalId);
    ~Muon() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;
    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::Optimizer> physicalSisterOptimizer,
                                  std::optional<Event> sisterOptimizerLoadedEvent) override;

    void setAlpha(float newAlpha, PlacedNetwork* placedNetwork);
    float getAlpha() const;
    void setBeta(float newBeta, PlacedNetwork* placedNetwork);
    float getBeta() const;
    void setEpsilon(float newEpsilon, PlacedNetwork* placedNetwork);
    float getEpsilon() const;
    void setWeightDecay(float newWeightDecay, PlacedNetwork* placedNetwork);
    float getWeightDecay() const;
    void setNesterov(bool newNesterov, PlacedNetwork* placedNetwork);
    bool getNesterov() const;
    uint32_t getNumIterations() const;
    float getCoefficientA() const;
    float getCoefficientB() const;
    float getCoefficientC() const;
    bool getTransposeTallMatrices() const;
    std::shared_ptr<Optimizer> getFallbackOptimizer() const;

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer,
                             std::string filenamePrefix,
                             bool saveOptimizerState) const override;
    static std::shared_ptr<Optimizer> deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader,
                                                  const nlohmann::json& j,
                                                  Network* network);

    nlohmann::json architectureJson() const override;

    std::string getType() const override { return "Muon"; }

   protected:
    void updateParameters(PlacedNetwork* placedNetwork);

    std::shared_ptr<Optimizer> clone() const override;

   private:
    static std::shared_ptr<Optimizer> makeDefaultFallbackOptimizer();

    float alpha;
    float beta;
    float epsilon;
    float weightDecay;
    bool nesterov;
    ThorImplementation::NewtonSchulzOrthogonalizationOptions orthogonalizationOptions;
    std::shared_ptr<Optimizer> fallbackOptimizer;

    std::shared_ptr<thor_file::TarReader> archiveReader = nullptr;
    std::optional<std::string> momentumFile;
};

class Muon::Builder {
   public:
    virtual std::shared_ptr<Muon> build() {
        if (!_alpha.has_value())
            _alpha = 0.02f;
        if (!_beta.has_value())
            _beta = 0.95f;
        if (!_epsilon.has_value())
            _epsilon = 1.0e-8f;
        if (!_weightDecay.has_value())
            _weightDecay = 0.0f;
        if (!_nesterov.has_value())
            _nesterov = true;
        if (!_numIterations.has_value())
            _numIterations = 5;
        if (!_coefficientA.has_value())
            _coefficientA = 3.4445f;
        if (!_coefficientB.has_value())
            _coefficientB = -4.775f;
        if (!_coefficientC.has_value())
            _coefficientC = 2.0315f;
        if (!_transposeTallMatrices.has_value())
            _transposeTallMatrices = true;
        if (_fallbackOptimizer == nullptr)
            _fallbackOptimizer = Muon::makeDefaultFallbackOptimizer();

        Muon muon;
        muon.alpha = _alpha.value();
        muon.beta = _beta.value();
        muon.epsilon = _epsilon.value();
        muon.weightDecay = _weightDecay.value();
        muon.nesterov = _nesterov.value();
        muon.orthogonalizationOptions.numIterations = _numIterations.value();
        muon.orthogonalizationOptions.coefficientA = _coefficientA.value();
        muon.orthogonalizationOptions.coefficientB = _coefficientB.value();
        muon.orthogonalizationOptions.coefficientC = _coefficientC.value();
        muon.orthogonalizationOptions.epsilon = _epsilon.value();
        muon.orthogonalizationOptions.transposeTallMatrices = _transposeTallMatrices.value();
        muon.orthogonalizationOptions.computeDType = ThorImplementation::DataType::FP32;
        muon.orthogonalizationOptions.outputDType = ThorImplementation::DataType::FP32;
        muon.fallbackOptimizer = _fallbackOptimizer->clone();

        THOR_THROW_IF_FALSE(muon.alpha > 0.0f);
        THOR_THROW_IF_FALSE(muon.beta >= 0.0f && muon.beta < 1.0f);
        THOR_THROW_IF_FALSE(muon.epsilon > 0.0f);
        THOR_THROW_IF_FALSE(muon.weightDecay >= 0.0f);
        THOR_THROW_IF_FALSE(muon.orthogonalizationOptions.numIterations > 0);
        THOR_THROW_IF_FALSE(std::isfinite(muon.orthogonalizationOptions.coefficientA));
        THOR_THROW_IF_FALSE(std::isfinite(muon.orthogonalizationOptions.coefficientB));
        THOR_THROW_IF_FALSE(std::isfinite(muon.orthogonalizationOptions.coefficientC));
        THOR_THROW_IF_FALSE(muon.fallbackOptimizer != nullptr);

        if (_network.has_value() && _network.value() != nullptr) {
            muon.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<Muon>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<Muon>(_network.value()->getDefaultOptimizer());
        }
        return std::dynamic_pointer_cast<Muon>(muon.clone());
    }

    virtual Muon::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }
    virtual Muon::Builder& alpha(float alpha) {
        THOR_THROW_IF_FALSE(!this->_alpha.has_value());
        this->_alpha = alpha;
        return *this;
    }
    virtual Muon::Builder& beta(float beta) {
        THOR_THROW_IF_FALSE(!this->_beta.has_value());
        this->_beta = beta;
        return *this;
    }
    virtual Muon::Builder& epsilon(float epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = epsilon;
        return *this;
    }
    virtual Muon::Builder& weightDecay(float weightDecay) {
        THOR_THROW_IF_FALSE(!this->_weightDecay.has_value());
        this->_weightDecay = weightDecay;
        return *this;
    }
    virtual Muon::Builder& nesterov(bool nesterov) {
        THOR_THROW_IF_FALSE(!this->_nesterov.has_value());
        this->_nesterov = nesterov;
        return *this;
    }
    virtual Muon::Builder& numIterations(uint32_t numIterations) {
        THOR_THROW_IF_FALSE(!this->_numIterations.has_value());
        this->_numIterations = numIterations;
        return *this;
    }
    virtual Muon::Builder& coefficientA(float coefficientA) {
        THOR_THROW_IF_FALSE(!this->_coefficientA.has_value());
        this->_coefficientA = coefficientA;
        return *this;
    }
    virtual Muon::Builder& coefficientB(float coefficientB) {
        THOR_THROW_IF_FALSE(!this->_coefficientB.has_value());
        this->_coefficientB = coefficientB;
        return *this;
    }
    virtual Muon::Builder& coefficientC(float coefficientC) {
        THOR_THROW_IF_FALSE(!this->_coefficientC.has_value());
        this->_coefficientC = coefficientC;
        return *this;
    }
    virtual Muon::Builder& transposeTallMatrices(bool transposeTallMatrices) {
        THOR_THROW_IF_FALSE(!this->_transposeTallMatrices.has_value());
        this->_transposeTallMatrices = transposeTallMatrices;
        return *this;
    }
    virtual Muon::Builder& fallbackOptimizer(std::shared_ptr<Optimizer> fallbackOptimizer) {
        THOR_THROW_IF_FALSE(this->_fallbackOptimizer == nullptr);
        THOR_THROW_IF_FALSE(fallbackOptimizer != nullptr);
        this->_fallbackOptimizer = fallbackOptimizer->clone();
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<float> _alpha;
    std::optional<float> _beta;
    std::optional<float> _epsilon;
    std::optional<float> _weightDecay;
    std::optional<bool> _nesterov;
    std::optional<uint32_t> _numIterations;
    std::optional<float> _coefficientA;
    std::optional<float> _coefficientB;
    std::optional<float> _coefficientC;
    std::optional<bool> _transposeTallMatrices;
    std::shared_ptr<Optimizer> _fallbackOptimizer;
};

}  // namespace Thor
