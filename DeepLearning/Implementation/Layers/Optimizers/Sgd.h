#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"

namespace ThorImplementation {

class Sgd final : public Optimizer {
   public:
    Sgd(uint64_t id, float initialLearningRate, float decay, float momentum, bool useNesterovMomentum, uint64_t startResumeEpoch = 0);

    void compile(const Tensor &weights, Stream &gradientUpdateStream) override;
    void updateWeights(uint32_t batchSize) override;

    std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) override;
    std::unordered_map<std::string, float> getAllHyperParameters() override;
    void setInitialLearningRate(float initaialLearningRate);
    void setDecay(float decay);
    void setMomentum(float momentum);
    void setUseNesterovMomentum(bool useNesterovMomentum);

    float getInitialLearningRate() const;
    float getDecay() const;
    float getMomentum() const;
    bool getUseNesterovMomentum() const;
    uint64_t getEpoch() const;

    std::shared_ptr<Optimizer> clone() const override {
        return std::make_shared<Sgd>(getId(), initialLearningRate, decay, momentum, useNesterovMomentum, 0);
    }

   protected:
    float initialLearningRate;
    float decay;
    float momentum;
    bool useNesterovMomentum;

    uint64_t currentEpoch = UINT64_MAX;
    uint64_t currentBatch;
    float currentLearningRate;
};

}  // namespace ThorImplementation
