#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"

#include <memory>

namespace ThorImplementation {

class RMSprop final : public CustomOptimizer {
   public:
    RMSprop(uint64_t id, float alpha, float rho, float epsilon);

    float getAlpha() const;
    float getRho() const;
    float getEpsilon() const;

    void setAlpha(float alpha);
    void setRho(float rho);
    void setEpsilon(float epsilon);

    std::shared_ptr<Optimizer> clone() const override;

    struct RuntimeState;

   private:
    RMSprop(uint64_t id, std::shared_ptr<RuntimeState> runtimeState);

    std::shared_ptr<RuntimeState> runtimeState;
};

}  // namespace ThorImplementation
