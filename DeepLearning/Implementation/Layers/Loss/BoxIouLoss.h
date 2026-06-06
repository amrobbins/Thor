#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"

namespace ThorImplementation {

class BoxIouLoss : public Loss {
   public:
    enum class Kind { IOU = 0, GIOU = 1, DIOU = 2, CIOU = 3 };

    BoxIouLoss(Kind kind, DataType lossDataType = DataType::FP32, float eps = 1.0e-7f);
    ~BoxIouLoss() override = default;

    std::optional<Tensor> createFeatureOutputTensor() override;
    void compileImpl() override;

    std::string getType() override { return "BoxIouLoss"; }
    std::optional<Tensor> getErrorOutput() const { return errorOutput; }

   private:
    void infer(std::optional<Tensor> predictions, std::optional<Tensor> loss, Stream stream) override;
    void backProp(std::optional<Tensor> labels, std::optional<Tensor> predictions, std::optional<Tensor> lossGradient, Stream stream) override;

    void launchKernel(bool computeGradient, Stream stream);

    Kind kind;
    float eps;
    uint32_t batchSize = 0;
    uint32_t boxesPerBatchElement = 0;
    uint32_t totalBoxes = 0;
};

}  // namespace ThorImplementation
