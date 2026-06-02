#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"

#include <memory>

namespace ThorImplementation {

class RAdam final : public CustomOptimizer {
   public:
    RAdam(uint64_t id, float alpha, float beta1, float beta2, float epsilon);

    float getT() const;
    float getAlpha() const;
    float getBeta1() const;
    float getBeta2() const;
    float getEpsilon() const;

    void setT(float t);
    void setAlpha(float alpha);
    void setBeta1(float beta1);
    void setBeta2(float beta2);
    void setEpsilon(float epsilon);

    std::shared_ptr<Optimizer> clone() const override;

    struct RuntimeState;

   private:
    RAdam(uint64_t id, std::shared_ptr<RuntimeState> runtimeState);

    std::shared_ptr<RuntimeState> runtimeState;
};

}  // namespace ThorImplementation
