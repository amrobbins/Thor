#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"

#include <map>
#include <vector>
#include <optional>

namespace Thor {

class MultiConnectionLayer : public Layer {
   public:
    ~MultiConnectionLayer() override {}

    // When there is only one connection, you may use the following version:
    std::optional<Tensor> getFeatureOutput() const override {
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    std::optional<Tensor> getFeatureInput() const override {
        THOR_THROW_IF_FALSE(featureInputs.size() == 1);
        return featureInputs[0];
    }

    // When there is more than one connection, you must use the following version:
    virtual Tensor getFeatureOutput(Tensor inputTensor) const {
        std::map<Tensor, Tensor>::const_iterator it = outputTensorFromInputTensor.find(inputTensor);
        THOR_THROW_IF_FALSE(it != outputTensorFromInputTensor.end());
        return it->second;
    }

    virtual Tensor getFeatureInput(Tensor outputTensor) const {
        std::map<Tensor, Tensor>::const_iterator it = inputTensorFromOutputTensor.find(outputTensor);
        THOR_THROW_IF_FALSE(it != inputTensorFromOutputTensor.end());
        return it->second;
    }

    int getConnectionType(Tensor connectingTensor) const override {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i])
                return 0;
        }
        for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
            if (connectingTensor == featureOutputs[i])
                return 0;
        }
        THOR_UNREACHABLE();
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override { return {getFeatureOutput(inputTensor)}; }

    std::vector<Tensor> getAllOutputTensors() const override { return featureOutputs; }

    // Inputs and outputs are stored in the vector in the same order as they are added to the builder.
    virtual std::vector<Tensor> getFeatureOutputs() const { return featureOutputs; }
    virtual std::vector<Tensor> getFeatureInputs() const { return featureInputs; }

    uint64_t getOutputTensorBytes(uint32_t batchSize) const override {
        return featureOutputs.size() * featureOutputs[0].getTotalSizeInBytes() * batchSize;
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        return getOutputTensorBytes(batchSize);
    }

    uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                      ThorImplementation::TensorPlacement tensorPlacement) const override {
        return getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }

   protected:
    std::vector<Tensor> featureInputs;
    std::vector<Tensor> featureOutputs;

    std::map<Tensor, Tensor> outputTensorFromInputTensor;
    std::map<Tensor, Tensor> inputTensorFromOutputTensor;

   private:
    using Layer::featureInput;
    using Layer::featureOutput;
};

}  // namespace Thor
