#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace ThorImplementation {

class Adafactor final : public Optimizer {
   public:
    Adafactor(uint64_t id, float alpha, float beta2, float epsilon, float weightDecay, bool factorSecondMoment);

    void compile(const Tensor& weights, Stream& gradientUpdateStream, bool materializeDenseGradient = true) override;
    SparseRowGradient compileSparseRows(const Tensor& weights, uint64_t maxSparseRows, Stream& gradientUpdateStream) override;

    [[nodiscard]] bool supportsSparseRowGradients() const override { return true; }
    [[nodiscard]] bool supportsSparseRowUpdateFusion() const override { return true; }
    [[nodiscard]] bool supportsDenseUpdateFusion() const override { return true; }

    [[nodiscard]] DenseOptimizerExpression toDenseUpdateExpression(const Tensor& weights,
                                                                   const Expression& gradient,
                                                                   const std::string& namePrefix) override;
    [[nodiscard]] SparseRowOptimizerExpression toSparseRowUpdateExpression(const Tensor& weights,
                                                                           SparseRowGradient& sparseRowGradient) override;

    [[nodiscard]] std::unordered_map<std::string, float> denseUpdateRuntimeScalars(uint32_t batchSize,
                                                                                   const std::string& namePrefix) override;
    [[nodiscard]] std::unordered_map<std::string, float> sparseRowUpdateRuntimeScalars(uint32_t batchSize) override;

    void updateWeights(uint32_t batchSize) override;
    void updateSparseRows(uint32_t batchSize) override;

    std::unordered_map<std::string, float> updateHyperParameters(uint64_t epoch, uint64_t batch, uint64_t batchesPerEpoch) override;
    std::unordered_map<std::string, float> getAllHyperParameters() override;

    float getAlpha() const;
    float getBeta2() const;
    float getEpsilon() const;
    float getWeightDecay() const;
    bool getFactorSecondMoment() const;

    void setAlpha(float alpha);
    void setBeta2(float beta2);
    void setEpsilon(float epsilon);
    void setWeightDecay(float weightDecay);
    void setFactorSecondMoment(bool factorSecondMoment);

    [[nodiscard]] bool isUsingFactoredPath() const { return selectedOptimizer_ != nullptr && usingFactoredPath_; }
    [[nodiscard]] bool isUsingUnfactoredPath() const { return selectedOptimizer_ != nullptr && !usingFactoredPath_; }
    [[nodiscard]] std::shared_ptr<Optimizer> getSelectedOptimizer() const { return selectedOptimizer_; }

    std::shared_ptr<Optimizer> clone() const override;

    struct RuntimeState;

   private:
    Adafactor(uint64_t id, std::shared_ptr<RuntimeState> runtimeState);

    [[nodiscard]] bool shouldUseFactoredPath(const Tensor& weights) const;
    [[nodiscard]] std::shared_ptr<Optimizer> makeFactoredOptimizer(const std::vector<uint64_t>& weightDims) const;
    [[nodiscard]] std::shared_ptr<Optimizer> makeUnfactoredOptimizer() const;
    void selectOptimizerForDenseWeights(const Tensor& weights);
    void selectUnfactoredOptimizer();
    void mirrorSelectedOptimizerState();

    std::shared_ptr<RuntimeState> runtimeState_;
    std::shared_ptr<Optimizer> selectedOptimizer_;
    bool usingFactoredPath_ = false;
};

}  // namespace ThorImplementation
