#pragma once

#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"

namespace ThorImplementation {

class BinaryCrossEntropy : public CustomLoss {
   public:
    explicit BinaryCrossEntropy(DataType lossDataType = DataType::FP32);
    ~BinaryCrossEntropy() override = default;

    void compileImpl() override;

    std::string getType() override { return "BinaryCrossEntropy"; }

   private:
    static DynamicExpression makeForwardExpression(DataType lossDataType);
    static DynamicExpression makeGradientExpression();
};

}  // namespace ThorImplementation
