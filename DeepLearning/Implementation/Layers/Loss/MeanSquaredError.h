#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Loss/MeanSquaredError.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <chrono>
#include <thread>

namespace ThorImplementation {

class MeanSquaredError : public Loss {
   public:
    virtual ~MeanSquaredError();

    MeanSquaredError();

    virtual void compile();

    virtual void cleanup() {}

    // Loss and gradient computation are all handled in infer. Gradient is not computed when in inference only mode.
    virtual void computeElementwiseLoss(Tensor labels, Tensor predictions, Tensor loss, Stream stream) {}
    virtual void computeLossGradient(Tensor labels, Tensor normalizedPredictions, Tensor lossGradient, Stream stream) {}

    virtual void infer(Optional<Tensor> predictions, Optional<Tensor> loss, Stream stream);

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream);

   private:
    TensorDescriptor::DataType labelsDataType;
    unsigned int batchSize;
    BatchReduce *batchReduce;
};

}  // namespace ThorImplementation
