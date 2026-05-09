#pragma once

#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Softmax : public Activation {
   public:
    Softmax();
    Softmax(bool backwardComputedExternally);

    ~Softmax() override {}

    Optional<Tensor> createFeatureOutputTensor() override;

    void postCompile() override;

    void cleanup();

    void infer(Optional<Tensor> inputTensor, Optional<Tensor> outputTensor, Stream stream) override;

    void backProp(Optional<Tensor> dataIn, Optional<Tensor> errorIn, Optional<Tensor> errorOut, Stream stream) override;

    std::string getType() override;

    virtual bool isBackwardComputedExternally();

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;

    cudnnTensorDescriptor_t cudnnTensorDescriptor;

    bool backwardComputedExternally;
};

}  // namespace ThorImplementation
