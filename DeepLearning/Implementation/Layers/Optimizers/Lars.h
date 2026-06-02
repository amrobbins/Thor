#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"

#include <memory>

namespace ThorImplementation {

class Lars final : public CustomOptimizer {
   public:
    Lars(uint64_t id,
         float alpha,
         float momentum,
         float weightDecay,
         float trustCoefficient,
         float epsilon,
         bool useNesterovMomentum);

    float getAlpha() const;
    float getMomentum() const;
    float getWeightDecay() const;
    float getTrustCoefficient() const;
    float getEpsilon() const;
    bool getUseNesterovMomentum() const;

    void setAlpha(float alpha);
    void setMomentum(float momentum);
    void setWeightDecay(float weightDecay);
    void setTrustCoefficient(float trustCoefficient);
    void setEpsilon(float epsilon);
    void setUseNesterovMomentum(bool useNesterovMomentum);

    std::shared_ptr<Optimizer> clone() const override;

    struct RuntimeState;

   private:
    Lars(uint64_t id, std::shared_ptr<RuntimeState> runtimeState);

    std::shared_ptr<RuntimeState> runtimeState;
};

}  // namespace ThorImplementation
