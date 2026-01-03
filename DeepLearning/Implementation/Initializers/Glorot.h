#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"

#include <math.h>

#include <nlohmann/json.hpp>

namespace ThorImplementation {

class Glorot : public Initializer {
   public:
    enum class Mode { UNIFORM = 7, NORMAL };

    Glorot(Mode mode = Mode::UNIFORM);

    virtual Event initialize(Layer *layer, Tensor tensorToInitialize);

    virtual std::shared_ptr<Initializer> clone();

   protected:
    virtual Event initialize(Layer *layer, Tensor tensorToInitialize, std::vector<Stream> streams);

    virtual Event initializeUniform(uint64_t fanIn, uint64_t fanOut, Tensor tensorToInitialize, std::vector<Stream> streams);
    virtual Event initializeNormal(uint64_t fanIn, uint64_t fanOut, Tensor tensorToInitialize, std::vector<Stream> streams);

    const Mode mode;
};

NLOHMANN_JSON_SERIALIZE_ENUM(Glorot::Mode,
                             {
                                 {Glorot::Mode::UNIFORM, "uniform"},
                                 {Glorot::Mode::NORMAL, "normal"},
                             })

}  // namespace ThorImplementation
