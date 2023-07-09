#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Loss/MeanAbsoluteError.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <cudnn_ops_infer.h>

#include <chrono>
#include <thread>

namespace ThorImplementation {

class MeanAbsoluteError : public Loss {
   public:
    virtual ~MeanAbsoluteError();

    MeanAbsoluteError(TensorDescriptor::DataType lossDataType = TensorDescriptor::DataType::FP32);

    virtual void compile();

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> predictions, Optional<Tensor> loss, Stream stream);

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream);

   private:
    void launchMeanAbsoluteErrorWithFP16Predictions();
    void launchMeanAbsoluteErrorWithFP16PredictionsAndFP16Loss();
    void launchMeanAbsoluteErrorWithFP16PredictionsAndFP32Loss();

    void launchMeanAbsoluteErrorWithFP32Predictions();
    void launchMeanAbsoluteErrorWithFP32PredictionsAndFP16Loss();
    void launchMeanAbsoluteErrorWithFP32PredictionsAndFP32Loss();

    unsigned int batchSize;
    cudnnTensorDescriptor_t errorOutputCudnnTensorDescriptor;
};

}  // namespace ThorImplementation
