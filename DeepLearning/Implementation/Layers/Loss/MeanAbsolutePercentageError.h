#pragma once

#include <optional>
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
    ~MeanAbsolutePercentageError() override;

    MeanAbsolutePercentageError(DataType lossDataType = DataType::FP32,
                                float epsilon = 0.0001,
                                float maxMagnitude = 1000.0f);

    void compileImpl() override;

    void cleanup() override {}

    void infer(std::optional<Tensor> predictions, std::optional<Tensor> loss, Stream stream) override;

    void backProp(std::optional<Tensor> labels, std::optional<Tensor> normalizedPredictions, std::optional<Tensor> lossGradient, Stream stream) override;

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
