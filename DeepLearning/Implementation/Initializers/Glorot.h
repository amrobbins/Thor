#pragma once

#include "DeepLearning/Implementation/Initializers/Initializer.h"

#include <math.h>

#include <nlohmann/json.hpp>

namespace ThorImplementation {

class Glorot : public Initializer {
   public:
    enum class Mode { UNIFORM = 7, NORMAL };

    Glorot(Mode mode = Mode::UNIFORM);

    void initialize(Stream initStream) override;

    std::shared_ptr<Initializer> clone() override;

   protected:
    virtual void initializeUniform(Stream initStream);
    virtual void initializeNormal(Stream initStream);

    const Mode mode;
};

NLOHMANN_JSON_SERIALIZE_ENUM(Glorot::Mode,
                             {
                                 {Glorot::Mode::UNIFORM, "uniform"},
                                 {Glorot::Mode::NORMAL, "normal"},
                             })

}  // namespace ThorImplementation
