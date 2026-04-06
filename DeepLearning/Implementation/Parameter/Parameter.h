#pragma once

#include <string>

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

class Parameter {
   public:
    virtual ~Parameter() = default;

    Parameter(std::string name, bool trainable, bool trainingEnabled);  // Later constraint

    // Remember this is called by API layer so that will hand over the optimizer
    // 1. Create storage given featureInput 2. Compile the optimizer
    virtual void compile(const Tensor &featureInput,
                         const Optional<Stream> &gradientUpdateStream,
                         bool inferenceOnly,
                         const uint64_t layerFanIn,
                         const uint64_t layerFanOut);

    virtual void compile(std::unordered_map<std::string, Tensor> featureInput,
                         const Optional<Stream> &gradientUpdateStream,
                         bool inferenceOnly) {
        assert(false);
    }

    virtual void createStorage(Tensor featureInput, uint32_t gpuId) = 0;
    virtual void createStorage(std::unordered_map<std::string, Tensor> featureInput, uint32_t gpuId);
    void clearStorage();

    Event initialize();

    // Parameters are not responsible for computing output gradient, expressions compute the gradients.
    void applyGradient(uint32_t batchSize);

    bool hasOptimizer();
    void setOptimizer(Optional<std::shared_ptr<Optimizer>> newOptimizer);
    std::shared_ptr<Optimizer> getOptimizer();
    void clearOptimizer();

    bool hasInitializer();
    void setInitializer(Optional<std::shared_ptr<Initializer>> newInitializer);
    std::shared_ptr<Initializer> getInitializer();
    void clearInitializer();

    std::string getName();
    Tensor getStorage();

    [[nodiscard]] bool isTrainable() const;
    [[nodiscard]] bool isTrainingEnabled() const;
    void setTrainingEnabled(bool enabled);

   protected:
    const std::string name;
    Optional<Tensor> storage;
    const bool trainable;
    bool trainingEnabled;

    std::shared_ptr<Optimizer> optimizer;
    std::shared_ptr<Initializer> initializer;
};

}  // namespace ThorImplementation
