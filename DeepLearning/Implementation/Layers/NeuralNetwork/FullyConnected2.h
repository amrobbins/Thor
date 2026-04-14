#pragma once

#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "DeepLearning/Implementation/Parameter/Parameter.h"
#include "Utilities/TensorOperations/DeepLearning/Add1dBias.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <memory>

namespace ThorImplementation {

class FullyConnected2 : public CustomLayer {
   public:
    virtual ~FullyConnected2() = default;

    FullyConnected2(const uint32_t numOutputFeatures,
                    const bool hasBias,
                    Optional<DataType> weightsDataType,
                    const TensorPlacement& placement,
                    bool inferenceOnly,
                    int64_t stampedId = -1);

    // void compileImpl() override;

    std::string getLayerType() override { return "FullyConnected"; }

   private:
    static DynamicExpression buildExpression(bool hasBias, TensorPlacement placement);

    std::vector<std::shared_ptr<Parameter>> defineParameters(uint32_t numOutputFeatures, bool hasBias);

    // const uint32_t numOutputFeatures;
    // const DataType weightsDataType;
    // const bool hasBias;

    // uint32_t numInputFeatures = 0;
    // uint32_t batchSize = 0;
};

}  // namespace ThorImplementation
