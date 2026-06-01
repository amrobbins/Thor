#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"

#include <memory>

namespace ThorImplementation {

class AdamW final : public CustomOptimizer {
   public:
    AdamW(uint64_t id, float alpha, float beta1, float beta2, float epsilon, float weightDecay);

    float getT() const;
    float getAlpha() const;
    float getBeta1() const;
    float getBeta2() const;
    float getEpsilon() const;
    float getWeightDecay() const;

    void setT(float t);
    void setAlpha(float alpha);
    void setBeta1(float beta1);
    void setBeta2(float beta2);
    void setEpsilon(float epsilon);
    void setWeightDecay(float weightDecay);

    std::shared_ptr<Optimizer> clone() const override;

    struct RuntimeState;

   private:
    AdamW(uint64_t id, std::shared_ptr<RuntimeState> runtimeState);

    std::shared_ptr<RuntimeState> runtimeState;
};

}  // namespace ThorImplementation
