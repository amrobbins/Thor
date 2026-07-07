#pragma once

#include "DeepLearning/Implementation/Layers/Loss/CustomLoss.h"

namespace ThorImplementation {

class MeanPowerError : public CustomLoss {
   public:
    explicit MeanPowerError(DataType lossDataType = DataType::FP32, float exponent = 1.5f);
    ~MeanPowerError() override = default;

    void compileImpl() override;

    std::string getType() override { return "MeanPowerError"; }

    float getExponent() const { return exponent; }

   private:
    static DynamicExpression makeForwardExpression(float exponent, DataType lossDataType);
    static DynamicExpression makeGradientExpression(float exponent);
    static void validateExponent(float exponent);

    float exponent = 1.5f;
};

}  // namespace ThorImplementation
