#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "omp.h"

#include <memory>

using std::make_shared;
using std::shared_ptr;

namespace ThorImplementation {

class Initializer {
   public:
    virtual ~Initializer();

    virtual void initialize(Layer *layer, Tensor tensorToInitialize);

    virtual shared_ptr<Initializer> clone();

   protected:
    virtual void performCopy(Tensor buffer, Tensor tensorToInitialize, vector<Stream> streams);

    virtual void initialize(Layer *layer, Tensor tensorToInitialize, vector<Stream> streams) = 0;
};

}  // namespace ThorImplementation
