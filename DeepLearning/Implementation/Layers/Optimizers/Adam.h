#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/TrainableWeightsBiasesLayer.h"
#include "Utilities/TensorOperations/Optimizers/Adam.h"

#include <unordered_map>

namespace ThorImplementation {

class Adam final : public Optimizer {
   public:
    Adam(uint64_t id, float alpha, float beta1, float beta2, float epsilon);

    void compile(const Tensor &weights, Stream &gradientUpdateStream) override;
    void updateWeights(uint32_t batchSize) override;

    std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) override;
    std::unordered_map<std::string, float> getAllHyperParameters() override;

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

    static constexpr float MIN_FP16_EPSILON = 1.0e-4f;

   protected:
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
    float t;

    Optional<Tensor> mBuffer;
    Optional<Tensor> vBuffer;

    Optional<std::string> mFile;
    Optional<std::string> vFile;
    Optional<std::string> mBiasFile;
    Optional<std::string> vBiasFile;

    DynamicExpression buildExpression() override { assert(false); /*using compile not dynamic expression, fixme*/ };
};

}  // namespace ThorImplementation
