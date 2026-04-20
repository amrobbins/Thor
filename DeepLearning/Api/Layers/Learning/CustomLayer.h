#pragma once

#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Parameter/Parameter.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"

namespace Thor {
class CustomLayer : public TrainableLayer, public Parameterizable {
   public:
    virtual ~CustomLayer() = default;

    // CustomLayer(Expression expr, std::unordered_map<string, Parameter> parameters) {
    //
    // }
};

}  // namespace Thor
