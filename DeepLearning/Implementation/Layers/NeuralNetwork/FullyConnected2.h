#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <memory>

namespace ThorImplementation {

class FullyConnected2 : public CustomLayer {
   public:
    ~FullyConnected2() override = default;

    FullyConnected2(const uint32_t numOutputFeatures,
                    const bool hasBias,
                    Optional<DataType> weightsDataType,
                    const TensorPlacement& placement,
                    bool inferenceOnly,
                    int64_t stampedId = -1);

    std::string getLayerType() override { return "FullyConnected"; }

   private:
    static DynamicExpression buildExpression(bool hasBias, TensorPlacement placement);

    std::vector<std::shared_ptr<Parameter>> defineParameters(uint32_t numOutputFeatures, bool hasBias);
};

}  // namespace ThorImplementation
