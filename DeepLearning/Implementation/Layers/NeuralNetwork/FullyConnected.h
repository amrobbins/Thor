#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <memory>

namespace ThorImplementation {

class FullyConnected : public CustomLayer {
   public:
    ~FullyConnected() override = default;

    FullyConnected(const uint32_t numOutputFeatures,
                   const bool hasBias,
                   Optional<TensorDescriptor::DataType> weightsDataType,
                   const TensorPlacement& placement,
                   bool inferenceOnly,
                   int64_t stampedId = -1);

    std::string getLayerType() override { return "FullyConnected"; }

   private:
    static DynamicExpression buildExpression(bool hasBias, TensorPlacement placement);
    std::vector<std::shared_ptr<Parameter>> defineParameters(uint32_t numOutputFeatures,
                                                             bool hasBias,
                                                             Optional<TensorDescriptor::DataType> weightsDataType);
};

}  // namespace ThorImplementation
