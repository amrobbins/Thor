#pragma once

#include <memory>
#include <optional>

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/Expression/FusedEquation.h"
#include "Utilities/Expression/StampedEquation.h"

namespace ThorImplementation {

class BinaryCrossEntropy : public Loss {
   public:
    explicit BinaryCrossEntropy(DataType lossDataType = DataType::FP32);
    ~BinaryCrossEntropy() override = default;

    void compileImpl() override;
    void cleanup() override;

    void infer(std::optional<Tensor> predictions, std::optional<Tensor> loss, Stream stream) override;
    void backProp(std::optional<Tensor> labels, std::optional<Tensor> predictions, std::optional<Tensor> lossGradient, Stream stream) override;

    std::string getType() override { return "BinaryCrossEntropy"; }

   private:
    static void validateLabelsDType(DataType dtype);
    static Outputs makeForwardOutputs(DataType lossDataType);
    static Outputs makeGradientOutputs(DataType predictionDataType);

    std::unique_ptr<StampedExecutionPlan> forwardPlan;
    std::unique_ptr<StampedExecutionPlan> gradientPlan;
};

}  // namespace ThorImplementation
