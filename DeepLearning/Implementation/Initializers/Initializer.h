#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include "omp.h"

#include <memory>

namespace ThorImplementation {

class Initializer {
   public:
    virtual ~Initializer() = default;

    virtual void compile(const Tensor &weights, const Optional<Stream> &stream, const uint64_t layerFanIn, const uint64_t layerFanOut) {
        this->weights = weights;
        this->stream = stream;

        // They are needed for Glorot. Glorot is important.
        // Seems special case it here or change the shape, so just adding Glorot special case support.
        this->layerFanIn = layerFanIn;
        this->layerFanOut = layerFanOut;
        assert(this->layerFanIn > 0);
        assert(this->layerFanOut > 0);
    }

    virtual Event initialize() = 0;
    virtual std::shared_ptr<Initializer> clone();

   protected:
    // virtual Event performCopy(Tensor buffer, Tensor tensorToInitialize, std::vector<Stream> streams);

    Tensor weights;
    Optional<Stream> stream;
    uint64_t layerFanIn = 0;
    uint64_t layerFanOut = 0;
};

}  // namespace ThorImplementation
