#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Parameter/PhysicalParameter.h"
#include "Utilities/Expression/Expression.h"
#include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <functional>
#include <memory>

namespace ThorImplementation {

class FullyConnected : public CustomLayer {
   public:
    using ExpressionTransform = std::function<Expression(const Expression&)>;

    ~FullyConnected() override = default;

    FullyConnected(const uint32_t numOutputFeatures,
                   const bool hasBias,
                   Optional<TensorDescriptor::DataType> weightsDataType,
                   const TensorPlacement& placement,
                   bool inferenceOnly,
                   ExpressionTransform activation = nullptr,
                   int64_t stampedId = -1);

    std::string getLayerType() override { return "FullyConnected"; }

   private:
    static DynamicExpression buildExpression(bool hasBias, TensorPlacement placement, ExpressionTransform activation);
    std::vector<std::shared_ptr<PhysicalParameter>> defineParameters(uint32_t numOutputFeatures,
                                                                     bool hasBias,
                                                                     Optional<TensorDescriptor::DataType> weightsDataType);
};

}  // namespace ThorImplementation
