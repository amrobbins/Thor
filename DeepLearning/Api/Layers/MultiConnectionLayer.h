#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include <map>
#include <vector>

using std::map;
using std::vector;

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
        printf("outputTensorFromInputTensor size %ld\n", outputTensorFromInputTensor.size());
        for (map<Tensor, Tensor>::const_iterator it = outputTensorFromInputTensor.begin(); it != outputTensorFromInputTensor.end(); ++it) {
            printf("input %ld -> output %ld\n", it->first.getId(), it->second.getId());
        }
        fflush(stdout);

        map<Tensor, Tensor>::const_iterator it = outputTensorFromInputTensor.find(inputTensor);
        assert(it != outputTensorFromInputTensor.end());
        return it->second;
    }

    virtual Tensor getFeatureInput(Tensor outputTensor) const {
        map<Tensor, Tensor>::const_iterator it = inputTensorFromOutputTensor.find(outputTensor);
        assert(it != inputTensorFromOutputTensor.end());
        return it->second;
    }

    // Inputs and outputs are stored in the vector in the same order as they are added to the builder.
    virtual vector<Tensor> getFeatureOutputs() const { return featureOutputs; }
    virtual vector<Tensor> getFeatureInputs() const { return featureInputs; }

   protected:
    vector<Tensor> featureInputs;
    vector<Tensor> featureOutputs;

    map<Tensor, Tensor> outputTensorFromInputTensor;
    map<Tensor, Tensor> inputTensorFromOutputTensor;

   private:
    Tensor featureInput;
    Tensor featureOutput;
};

}  // namespace Thor
