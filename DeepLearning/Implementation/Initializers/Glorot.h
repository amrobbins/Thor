#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"

#include <math.h>

namespace ThorImplementation {

class Glorot : public Initializer {
   public:
    enum class Mode { UNIFORM = 7, NORMAL };

    Glorot(Mode mode = Mode::UNIFORM);

    virtual void initialize(Layer *layer, Tensor tensorToInitialize);

    virtual std::shared_ptr<Initializer> clone();

   protected:
    virtual void initialize(Layer *layer, Tensor tensorToInitialize, std::vector<Stream> streams);

    virtual void initializeUniform(uint64_t fanIn, uint64_t fanOut, Tensor tensorToInitialize, std::vector<Stream> streams);
    virtual void initializeNormal(uint64_t fanIn, uint64_t fanOut, Tensor tensorToInitialize, std::vector<Stream> streams);

    const Mode mode;
};

}  // namespace ThorImplementation
