#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "Utilities/TensorOperations/Misc/BatchReduce.h"

namespace ThorImplementation {

class LossShaper : public Layer {
   public:
    // Numerical losses output elementwise loss
    // Categorical losses output per element classwise loss
    // Loss shaper can reshape numerical elementwise loss into batch loss
    // Loss shaper can reshape per element classwise loss into either batch loss or classwise loss
    enum class InputLossType { NUMERICAL_LOSS = 5, CATEGORICAL_LOSS };
    enum class OutputLossType { BATCH_LOSS = 11, CLASSWISE_LOSS };

    LossShaper(InputLossType inputLossType, OutputLossType outputLossType);
    virtual ~LossShaper();

    virtual Optional<Tensor> createFeatureOutputTensor();
    virtual void compile();
    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream);
    virtual void backward(Optional<Tensor> errorInput);
    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream);

   private:
    bool uninitialized;

    InputLossType inputLossType;
    OutputLossType outputLossType;
    BatchReduce *batchReduce;
};

}  // namespace ThorImplementation
