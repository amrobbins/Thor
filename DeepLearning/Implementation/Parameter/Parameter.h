#pragma once

#include <string>

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

class Parameter {
   public:
    virtual ~Parameter() = default;

    Parameter(std::string name, Tensor storage, bool trainable, bool trainingEnabled);

    // Parameters are not responsible for computing output gradient, expressions compute the gradients.
    void applyGradient(Tensor gradient, Stream gradientReadyStream);

    virtual bool hasOptimizer();
    virtual void setOptimizer(Optional<std::shared_ptr<Optimizer>> newOptimizer);
    virtual std::shared_ptr<Optimizer> getOptimizer();
    virtual void clearOptimizer();

    virtual std::string getName();
    virtual Tensor getStorage();

    bool isTrainable() const;
    bool isTrainingEnabled() const;
    void setTrainingEnabled(bool enabled);

   protected:
    const std::string name;
    Tensor storage;
    const bool trainable;
    bool trainingEnabled;

    std::shared_ptr<Optimizer> optimizer;
};

}  // namespace ThorImplementation
