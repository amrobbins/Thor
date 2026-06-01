#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"

#include <memory>

namespace ThorImplementation {

class Sgd final : public CustomOptimizer {
   public:
    Sgd(uint64_t id, float initialLearningRate, float decay, float momentum, bool useNesterovMomentum, uint64_t startResumeEpoch = 0);

    void setInitialLearningRate(float initialLearningRate);
    void setDecay(float decay);
    void setMomentum(float momentum);
    void setUseNesterovMomentum(bool useNesterovMomentum);

    float getInitialLearningRate() const;
    float getDecay() const;
    float getMomentum() const;
    bool getUseNesterovMomentum() const;
    uint64_t getEpoch() const;
    float getCurrentLearningRate() const;

    std::shared_ptr<Optimizer> clone() const override;

    struct RuntimeState;

   private:
    Sgd(uint64_t id, std::shared_ptr<RuntimeState> runtimeState);

    std::shared_ptr<RuntimeState> runtimeState;
};

}  // namespace ThorImplementation
