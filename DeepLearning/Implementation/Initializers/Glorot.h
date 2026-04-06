#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"

#include <math.h>

#include <nlohmann/json.hpp>

namespace ThorImplementation {

class Glorot : public Initializer {
   public:
    enum class Mode { UNIFORM = 7, NORMAL };

    Glorot(Mode mode = Mode::UNIFORM);

    Event initialize() override;

    std::shared_ptr<Initializer> clone() override;

   protected:
    virtual Event initializeUniform();
    virtual Event initializeNormal();

    const Mode mode;
};

NLOHMANN_JSON_SERIALIZE_ENUM(Glorot::Mode,
                             {
                                 {Glorot::Mode::UNIFORM, "uniform"},
                                 {Glorot::Mode::NORMAL, "normal"},
                             })

}  // namespace ThorImplementation
