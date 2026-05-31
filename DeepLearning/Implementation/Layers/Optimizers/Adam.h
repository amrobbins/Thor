#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "Utilities/TensorOperations/Optimizers/Adam.h"

#include <unordered_map>

#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"

namespace ThorImplementation {

class Adam final : public Optimizer {
   public:
    Adam(uint64_t id, float alpha, float beta1, float beta2, float epsilon);

    void compile(const Tensor &weights, Stream &gradientUpdateStream, bool materializeDenseGradient = true) override;
    SparseRowGradient compileSparseRows(const Tensor &weights, uint64_t maxSparseRows, Stream &gradientUpdateStream) override;
    [[nodiscard]] SparseRowOptimizerExpression toSparseRowUpdateExpression(const Tensor &weights, SparseRowGradient &sparseRowGradient) override;
    [[nodiscard]] bool supportsSparseRowGradients() const override { return true; }
    [[nodiscard]] bool supportsSparseRowUpdateFusion() const override { return true; }
    [[nodiscard]] bool supportsDenseUpdateFusion() const override { return true; }
    [[nodiscard]] DenseOptimizerExpression toDenseUpdateExpression(const Tensor &weights,
                                                                   const Expression &gradient,
                                                                   const std::string &namePrefix) override;
    [[nodiscard]] std::unordered_map<std::string, float> denseUpdateRuntimeScalars(uint32_t batchSize,
                                                                                   const std::string &namePrefix) override;
    [[nodiscard]] std::unordered_map<std::string, float> sparseRowUpdateRuntimeScalars(uint32_t batchSize) override;

    void updateWeights(uint32_t batchSize) override;
    void updateSparseRows(uint32_t batchSize) override;

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

    std::shared_ptr<Optimizer> clone() const override { return std::make_shared<Adam>(getId(), alpha, beta1, beta2, epsilon); }

   protected:
    float alpha;
    float beta1;
    float beta2;
    float epsilon;
    float t;
};

}  // namespace ThorImplementation
