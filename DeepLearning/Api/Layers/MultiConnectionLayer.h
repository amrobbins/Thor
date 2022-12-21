#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include <map>
#include <vector>

namespace Thor {

class MultiConnectionLayer : public Layer {
   public:
    virtual ~MultiConnectionLayer() {}

    // When there is only one connection, you may use the following version:
    virtual Optional<Tensor> getFeatureOutput() const {
        assert(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    virtual Optional<Tensor> getFeatureInput() const {
        assert(featureInputs.size() == 1);
        return featureInputs[0];
    }

    // When there is more than one connection, you must use the following version:
    virtual Tensor getFeatureOutput(Tensor inputTensor) const {
        std::map<Tensor, Tensor>::const_iterator it = outputTensorFromInputTensor.find(inputTensor);
        assert(it != outputTensorFromInputTensor.end());
        return it->second;
    }

    virtual Tensor getFeatureInput(Tensor outputTensor) const {
        std::map<Tensor, Tensor>::const_iterator it = inputTensorFromOutputTensor.find(outputTensor);
        assert(it != inputTensorFromOutputTensor.end());
        return it->second;
    }

    virtual int getConnectionType(Tensor connectingTensor) const {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i])
                return 0;
        }
        for (uint32_t i = 0; i < featureOutputs.size(); ++i) {
            if (connectingTensor == featureOutputs[i])
                return 0;
        }
        assert(false);
    }

    virtual std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) { return {getFeatureOutput(inputTensor)}; }

    virtual std::vector<Tensor> getAllOutputTensors() const { return featureOutputs; }

    // Inputs and outputs are stored in the vector in the same order as they are added to the builder.
    virtual std::vector<Tensor> getFeatureOutputs() const { return featureOutputs; }
    virtual std::vector<Tensor> getFeatureInputs() const { return featureInputs; }

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
