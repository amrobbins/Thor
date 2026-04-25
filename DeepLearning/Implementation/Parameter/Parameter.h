#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

class Parameter {
   public:
    struct StorageContext {
        Tensor primaryInput;
        std::vector<Tensor> featureInputs;
        std::unordered_map<std::string, Tensor> namedInputs;

        StorageContext(const Tensor &primaryInput) : primaryInput(primaryInput), featureInputs{primaryInput} {}
        StorageContext(const Tensor &primaryInput,
                       std::vector<Tensor> featureInputs,
                       std::unordered_map<std::string, Tensor> namedInputs = {})
            : primaryInput(primaryInput), featureInputs(std::move(featureInputs)), namedInputs(std::move(namedInputs)) {}
    };

    virtual ~Parameter() = default;

    Parameter(std::string name, bool trainable);  // Later constraint

    // Remember this is called by API layer so that will hand over the optimizer
    // 1. Create storage given featureInput(s) 2. Compile the optimizer
    virtual void compileStorageAndOptimizer(const StorageContext &context,
                                            const Optional<Stream> &gradientUpdateStream,
                                            bool inferenceOnly);
    virtual void compileStorageAndOptimizer(const Tensor &inputTensor, const Optional<Stream> &gradientUpdateStream, bool inferenceOnly);

    void compileInitializer(uint64_t fanIn = 0, uint64_t fanOut = 0);

    virtual void createStorage(const StorageContext &context) { createStorage(context.primaryInput); }
    virtual void createStorage(const Tensor &inputTensor) = 0;
    void clearStorage();

    Event initialize();

    // Parameters are not responsible for computing output gradient, expressions compute the gradients.
    bool applyGradient(uint32_t batchSize);

    bool hasOptimizer();
    void setOptimizer(Optional<std::shared_ptr<Optimizer>> newOptimizer);
    std::shared_ptr<Optimizer> getOptimizer();
    void clearOptimizer();

    bool hasInitializer();
    void setInitializer(Optional<std::shared_ptr<Initializer>> newInitializer);
    std::shared_ptr<Initializer> getInitializer();
    void clearInitializer();

    std::string getName() const;
    Optional<Tensor> getStorage();

    [[nodiscard]] bool isTrainable() const;
    [[nodiscard]] bool isTrainingEnabled() const;
    void setTrainingEnabled(bool enabled);

    bool isStorageInitialized() const;

   protected:
    const std::string name;
    Optional<Tensor> storage;
    const bool trainable;
    bool trainingEnabled;

    std::shared_ptr<Optimizer> optimizer;
    std::shared_ptr<Initializer> initializer;

    Optional<Stream> gradientUpdateStream;

    bool storageInitialized = false;
};

}  // namespace ThorImplementation
