#pragma once

#include <optional>
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <cudnn_ops.h>
#include <memory>

#include <chrono>
#include <thread>

namespace ThorImplementation {

class CrossEntropy : public Loss {
   public:
    CrossEntropy();
    CrossEntropy(CrossEntropyLossType crossEntropyLossType, DataType lossDataType, bool indexLabels = false);

    ~CrossEntropy() override;

    CrossEntropy(bool indexLabels);

    void compileImpl() override;

    void cleanup() override {}

    void infer(std::optional<Tensor> predictions, std::optional<Tensor> loss, Stream stream) override;

    void backProp(std::optional<Tensor> labels, std::optional<Tensor> normalizedPredictions, std::optional<Tensor> lossGradient, Stream stream) override;

    std::optional<Tensor> createErrorOutputTensor(bool backPropagateError) override;

    std::string getType() override;

   private:
    void launchCrossEntropyWithFP16Predictions();
    void launchCrossEntropyWithFP16PredictionsAndFP16Loss();
    void launchCrossEntropyWithFP16PredictionsAndFP32Loss();

    void launchCrossEntropyWithFP32Predictions();
    void launchCrossEntropyWithFP32PredictionsAndFP16Loss();
    void launchCrossEntropyWithFP32PredictionsAndFP32Loss();

    unsigned int batchSize;

    // Either takes an integer index (one per batch item) or a vector of floats (one per class per batch item)
    // to indicate an example's true class
    bool indexLabels;

    CrossEntropyLossType crossEntropyLossType;

    uint32_t numClasses;
};

}  // namespace ThorImplementation
