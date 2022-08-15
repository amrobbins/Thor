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
    virtual ~CrossEntropy();

    CrossEntropy(bool indexLabels);

    virtual void compile();

    virtual void cleanup() {}

    virtual void infer(Optional<Tensor> predictions, Optional<Tensor> loss, Stream stream);

    virtual void backProp(Optional<Tensor> labels, Optional<Tensor> normalizedPredictions, Optional<Tensor> lossGradient, Stream stream);

   private:
    void launchCrossEntropyForIndexLabelsWithFP16Predictions();
    void launchCrossEntropyForIndexLabelsWithFP16PredictionsAndFP16Loss();
    void launchCrossEntropyForIndexLabelsWithFP16PredictionsAndFP32Loss();

    void launchCrossEntropyForIndexLabelsWithFP32Predictions();
    void launchCrossEntropyForIndexLabelsWithFP32PredictionsAndFP16Loss();
    void launchCrossEntropyForIndexLabelsWithFP32PredictionsAndFP32Loss();

    void launchCrossEntropyForPerClassLabelsWithFP16Predictions();
    void launchCrossEntropyForPerClassLabelsWithFP16PredictionsAndFP16Loss();
    void launchCrossEntropyForPerClassLabelsWithFP16PredictionsAndFP32Loss();

    void launchCrossEntropyForPerClassLabelsWithFP32Predictions();
    void launchCrossEntropyForPerClassLabelsWithFP32PredictionsAndFP16Loss();
    void launchCrossEntropyForPerClassLabelsWithFP32PredictionsAndFP32Loss();

    unsigned int batchSize;
    BatchReduce *batchReduce;

    // Either takes an integer index (one per batch item) or a vector of floats (one per class per batch item)
    // to indicate an example's true class
    bool indexLabels;
};

}  // namespace ThorImplementation
