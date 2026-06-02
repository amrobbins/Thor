#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"

#include <memory>

namespace ThorImplementation {

class ASGD final : public CustomOptimizer {
   public:
    ASGD(uint64_t id, float alpha, float lambd, float power, float t0, float weightDecay);

    void setAlpha(float alpha);
    void setLambd(float lambd);
    void setPower(float power);
    void setT0(float t0);
    void setWeightDecay(float weightDecay);
    void setT(float t);

    float getAlpha() const;
    float getLambd() const;
    float getPower() const;
    float getT0() const;
    float getWeightDecay() const;
    float getT() const;

    std::shared_ptr<Optimizer> clone() const override;

    struct RuntimeState;

   private:
    ASGD(uint64_t id, std::shared_ptr<RuntimeState> runtimeState);

    std::shared_ptr<RuntimeState> runtimeState;
};

}  // namespace ThorImplementation
