#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Loss/MeanSquaredError.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <cudnn_ops.h>

#include <chrono>
#include <thread>

namespace ThorImplementation {

class MeanSquaredError : public Loss {
   public:
    ~MeanSquaredError() override;

    MeanSquaredError(TensorDescriptor::DataType lossDataType = TensorDescriptor::DataType::FP32);

    void compileImpl() override;

    void cleanup() override {}

    void infer(Optional<Tensor> predictions, Optional<Tensor> loss, Stream stream) override;

    void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream) override;

   private:
    void launchMeanSquaredErrorWithFP16Predictions();
    void launchMeanSquaredErrorWithFP16PredictionsAndFP16Loss();
    void launchMeanSquaredErrorWithFP16PredictionsAndFP32Loss();

    void launchMeanSquaredErrorWithFP32Predictions();
    void launchMeanSquaredErrorWithFP32PredictionsAndFP16Loss();
    void launchMeanSquaredErrorWithFP32PredictionsAndFP32Loss();

    unsigned int batchSize;
    cudnnTensorDescriptor_t errorOutputCudnnTensorDescriptor;
};

}  // namespace ThorImplementation
