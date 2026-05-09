#pragma once

#include <optional>
#include "DeepLearning/Implementation/Layers/Activation/Activation.h"
#include "DeepLearning/Implementation/Layers/Layer.h"

namespace ThorImplementation {

class Softmax : public Activation {
   public:
    Softmax();
    Softmax(bool backwardComputedExternally);

    ~Softmax() override {}

    std::optional<Tensor> createFeatureOutputTensor() override;

    void postCompile() override;

    void cleanup();

    void infer(std::optional<Tensor> inputTensor, std::optional<Tensor> outputTensor, Stream stream) override;

    void backProp(std::optional<Tensor> dataIn, std::optional<Tensor> errorIn, std::optional<Tensor> errorOut, Stream stream) override;

    std::string getType() override;

    virtual bool isBackwardComputedExternally();

   private:
    static const float ALPHA_NO_SCALE;
    static const float BETA_CLEAR;

    cudnnTensorDescriptor_t cudnnTensorDescriptor;

    bool backwardComputedExternally;
};

}  // namespace ThorImplementation
