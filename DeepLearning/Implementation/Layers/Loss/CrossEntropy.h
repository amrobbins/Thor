#pragma once

#include "DeepLearning/Implementation/Layers/Loss.h"
#include "Utilities/TensorOperations/Loss/CrossEntropyLoss.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

#include <cudnn_ops_infer.h>

#include <assert.h>
#include <chrono>
#include <thread>

namespace ThorImplementation {

class CrossEntropy : public Loss {
   public:
    CrossEntropy();
    CrossEntropy(CrossEntropyLossType crossEntropyLossType, bool indexLabels = false);

    virtual ~CrossEntropy();

    CrossEntropy(bool indexLabels);

    virtual void compile();

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> predictions, Optional<Tensor> loss, Stream stream);

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream);

   private:
    void launchCrossEntropyWithFP16Predictions();
    void launchCrossEntropyWithFP16PredictionsAndFP16Loss();
    void launchCrossEntropyWithFP16PredictionsAndFP32Loss();

    void launchCrossEntropyWithFP32Predictions();
    void launchCrossEntropyWithFP32PredictionsAndFP16Loss();
    void launchCrossEntropyWithFP32PredictionsAndFP32Loss();

    unsigned int batchSize;
    BatchReduce *batchReduce;

    // Either takes an integer index (one per batch item) or a vector of floats (one per class per batch item)
    // to indicate an example's true class
    bool indexLabels;

    CrossEntropyLossType crossEntropyLossType;
};

}  // namespace ThorImplementation
