#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Loss/MeanAbsolutePercentageError.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <cudnn_ops.h>

#include <chrono>
#include <thread>

namespace ThorImplementation {

class MeanAbsolutePercentageError : public Loss {
   public:
    virtual ~MeanAbsolutePercentageError();

    MeanAbsolutePercentageError(TensorDescriptor::DataType lossDataType = TensorDescriptor::DataType::FP32,
                                float epsilon = 0.0001,
                                float maxMagnitude = 1000.0f);

    virtual void compile();

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> predictions, Optional<Tensor> loss, Stream stream);

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream);

   private:
    void launchMeanAbsolutePercentageErrorWithFP16Predictions();
    void launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP16Loss();
    void launchMeanAbsolutePercentageErrorWithFP16PredictionsAndFP32Loss();

    void launchMeanAbsolutePercentageErrorWithFP32Predictions();
    void launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP16Loss();
    void launchMeanAbsolutePercentageErrorWithFP32PredictionsAndFP32Loss();

    unsigned int batchSize;
    cudnnTensorDescriptor_t errorOutputCudnnTensorDescriptor;

    float epsilon;
    float maxMagnitude;
};

}  // namespace ThorImplementation
