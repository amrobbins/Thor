#pragma once

#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"

namespace ThorImplementation {

class MeanAbsoluteError : public CustomLoss {
   public:
    explicit MeanAbsoluteError(DataType lossDataType = DataType::FP32);
    ~MeanAbsoluteError() override = default;

    void compileImpl() override;

    std::string getType() override { return "MeanAbsoluteError"; }

   private:
    static DynamicExpression makeForwardExpression(DataType lossDataType);
    static DynamicExpression makeGradientExpression();
};

}  // namespace ThorImplementation
