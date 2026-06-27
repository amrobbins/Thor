#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "Utilities/Expression/NewtonSchulzOrthogonalization.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace ThorImplementation {

class Muon final : public Optimizer {
   public:
    Muon(uint64_t id,
         float alpha,
         float beta,
         float epsilon,
         float weightDecay,
         bool nesterov,
         NewtonSchulzOrthogonalizationOptions orthogonalizationOptions,
         std::shared_ptr<Optimizer> fallbackOptimizer);

    void compile(const Tensor& weights, Stream& gradientUpdateStream, bool materializeDenseGradient = true) override;
    SparseRowGradient compileSparseRows(const Tensor& weights, uint64_t maxSparseRows, Stream& gradientUpdateStream) override;

    [[nodiscard]] bool supportsSparseRowGradients() const override;
    [[nodiscard]] bool supportsSparseRowUpdateFusion() const override;
    [[nodiscard]] bool supportsDenseUpdateFusion() const override;

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
    void restoreHyperParameters(const std::unordered_map<std::string, float>& hyperParameters) override;

    float getAlpha() const;
    float getBeta() const;
    float getEpsilon() const;
    float getWeightDecay() const;
    bool getNesterov() const;
    const NewtonSchulzOrthogonalizationOptions& getOrthogonalizationOptions() const;

    void setAlpha(float alpha);
    void setBeta(float beta);
    void setEpsilon(float epsilon);
    void setWeightDecay(float weightDecay);
    void setNesterov(bool nesterov);

    [[nodiscard]] bool isUsingMuonMatrixPath() const { return usingMuonMatrixPath_; }
    [[nodiscard]] bool isUsingFallbackPath() const { return selectedOptimizer_ != nullptr && !usingMuonMatrixPath_; }
    [[nodiscard]] std::shared_ptr<Optimizer> getSelectedOptimizer() const { return selectedOptimizer_; }
    [[nodiscard]] std::shared_ptr<Optimizer> getFallbackOptimizer() const { return fallbackOptimizer_; }

    std::shared_ptr<Optimizer> clone() const override;

    struct RuntimeState;

   private:
    Muon(uint64_t id,
         std::shared_ptr<RuntimeState> runtimeState,
         NewtonSchulzOrthogonalizationOptions orthogonalizationOptions,
         std::shared_ptr<Optimizer> fallbackOptimizer);

    [[nodiscard]] bool shouldUseMuonMatrixPath(const Tensor& weights) const;
    [[nodiscard]] std::shared_ptr<Optimizer> makeMuonMatrixOptimizer() const;
    void selectOptimizerForDenseWeights(const Tensor& weights);
    void selectFallbackOptimizer();
    void mirrorSelectedOptimizerState();

    std::shared_ptr<RuntimeState> runtimeState_;
    NewtonSchulzOrthogonalizationOptions orthogonalizationOptions_;
    std::shared_ptr<Optimizer> fallbackOptimizer_;
    std::shared_ptr<Optimizer> selectedOptimizer_;
    bool usingMuonMatrixPath_ = false;
};

}  // namespace ThorImplementation
