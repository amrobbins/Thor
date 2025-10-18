#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "omp.h"

#include <memory>

namespace ThorImplementation {

class Initializer {
   public:
    virtual ~Initializer();

    virtual void initialize(Layer *layer, Tensor tensorToInitialize);

    virtual std::shared_ptr<Initializer> clone();

   protected:
    virtual void performCopy(Tensor buffer, Tensor tensorToInitialize, std::vector<Stream> streams);

    virtual Event initialize(Layer *layer, Tensor tensorToInitialize, std::vector<Stream> streams) = 0;
};

}  // namespace ThorImplementation
