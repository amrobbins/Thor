#pragma once

#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"

namespace ThorImplementation {

class MeanSquaredError : public CustomLoss {
   public:
    explicit MeanSquaredError(DataType lossDataType = DataType::FP32);
    ~MeanSquaredError() override = default;

    void compileImpl() override;

    std::string getType() override { return "MeanSquaredError"; }

   private:
    static DynamicExpression makeForwardExpression(DataType lossDataType);
    static DynamicExpression makeGradientExpression();
};

}  // namespace ThorImplementation
