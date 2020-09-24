#pragma once

#include "DeepLearning/Api/Layers/Layer.h"

#include <vector>

using std::vector;

namespace Thor {

class MultiConnectionLayer : public Layer {
   public:
    virtual ~MultiConnectionLayer() {}

    vector<Tensor> getFeatureInputs() const { return featureInputs; }
    vector<Tensor> getFeatureOutputs() const { return featureOutputs; }

   protected:
    vector<Tensor> featureInputs;
    vector<Tensor> featureOutputs;
};

}  // namespace Thor
