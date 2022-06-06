#pragma once

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Softmax : public Activation {
   public:
    virtual ~Softmax() {}

    virtual Optional<Tensor> createFeatureOutputTensor();

    virtual void compile();

    void cleanup();

    virtual void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream);

    virtual void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream);

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;

    cudnnTensorDescriptor_t cudnnTensorDescriptor;
};

}  // namespace ThorImplementation
