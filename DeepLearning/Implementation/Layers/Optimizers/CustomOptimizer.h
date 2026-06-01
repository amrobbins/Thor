#pragma once

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "Utilities/Expression/Expression.h"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ThorImplementation {

struct CustomOptimizerStateSpec {
    std::string name;
    DataType dtype = DataType::FP32;
    std::optional<std::vector<uint64_t>> shape = std::nullopt;

    static CustomOptimizerStateSpec sameShapeAsWeights(std::string name, DataType dtype = DataType::FP32) {
        return CustomOptimizerStateSpec{std::move(name), dtype, std::nullopt};
    }
};

struct CustomOptimizerUpdateExpression {
    std::vector<std::pair<std::string, Expression>> outputs;
};

class CustomOptimizerUpdateContext {
   public:
    enum class Mode { Dense, SparseRows };

    CustomOptimizerUpdateContext(const Tensor& weights, Expression gradient, std::string namePrefix, Mode mode);

    [[nodiscard]] const Tensor& weightsTensor() const { return weights_; }
    [[nodiscard]] const std::string& namePrefix() const { return namePrefix_; }
    [[nodiscard]] Mode mode() const { return mode_; }
    [[nodiscard]] bool isDense() const { return mode_ == Mode::Dense; }
    [[nodiscard]] bool isSparseRows() const { return mode_ == Mode::SparseRows; }

    [[nodiscard]] Expression weights(DataType outputDType = DataType::FP32, DataType computeDType = DataType::FP32) const;
    [[nodiscard]] Expression gradient() const { return gradient_; }
    [[nodiscard]] Expression state(const std::string& name,
                                   DataType outputDType = DataType::FP32,
                                   DataType computeDType = DataType::FP32) const;
    [[nodiscard]] Expression runtimeScalar(const std::string& name,
                                           DataType outputDType = DataType::FP32,
                                           DataType computeDType = DataType::FP32) const;

    [[nodiscard]] std::string weightsInputName() const;
    [[nodiscard]] std::string stateInputName(const std::string& name) const;
    [[nodiscard]] std::string runtimeScalarName(const std::string& name) const;

   private:
    const Tensor& weights_;
    Expression gradient_;
    std::string namePrefix_;
    Mode mode_;
};

class CustomOptimizer : public Optimizer {
   public:
    using UpdateExpressionBuilder = std::function<CustomOptimizerUpdateExpression(const CustomOptimizerUpdateContext&)>;
    using RuntimeScalarBuilder = std::function<std::unordered_map<std::string, float>(uint32_t batchSize,
                                                                                      const std::string& namePrefix)>;

    CustomOptimizer(uint64_t id,
                    std::vector<CustomOptimizerStateSpec> stateSpecs,
                    UpdateExpressionBuilder updateExpressionBuilder,
                    RuntimeScalarBuilder runtimeScalarBuilder = {},
                    bool supportsSparseRowGradients = false);

    void compile(const Tensor& weights, Stream& gradientUpdateStream, bool materializeDenseGradient = true) override;
    SparseRowGradient compileSparseRows(const Tensor& weights, uint64_t maxSparseRows, Stream& gradientUpdateStream) override;

    [[nodiscard]] bool supportsSparseRowGradients() const override { return supportsSparseRowGradients_; }
    [[nodiscard]] bool supportsSparseRowUpdateFusion() const override { return supportsSparseRowGradients_; }
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

    const std::vector<CustomOptimizerStateSpec>& getStateSpecs() const { return stateSpecs_; }

    std::shared_ptr<Optimizer> clone() const override {
        return std::make_shared<CustomOptimizer>(getId(), stateSpecs_, updateExpressionBuilder_, runtimeScalarBuilder_, supportsSparseRowGradients_);
    }

   private:
    void validateReadyToBuildExpression(const Tensor& weights) const;
    void ensureStateParameters(const Tensor& weights);
    CustomOptimizerUpdateExpression buildAndValidateUpdateExpression(const CustomOptimizerUpdateContext& context) const;
    std::unordered_map<std::string, Tensor> stateInputTensors(const std::string& namePrefix);
    std::unordered_map<std::string, Tensor> preallocatedOutputTensors(const CustomOptimizerUpdateExpression& updateExpression,
                                                                      const Tensor& weights);

    std::vector<CustomOptimizerStateSpec> stateSpecs_;
    UpdateExpressionBuilder updateExpressionBuilder_;
    RuntimeScalarBuilder runtimeScalarBuilder_;
    bool supportsSparseRowGradients_ = false;
};

}  // namespace ThorImplementation
