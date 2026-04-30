#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "DeepLearning/Implementation/Layers/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

namespace ThorImplementation {

class PhysicalParameter {
   public:
    struct StorageContext {
        std::unordered_map<std::string, Tensor> namedInputs;

        StorageContext() = default;

        StorageContext(std::unordered_map<std::string, Tensor> namedInputs) : namedInputs(std::move(namedInputs)) {}
        StorageContext(Tensor featureInput) : namedInputs({{"feature_input", featureInput}}) {}

        bool hasInput(const std::string &name) const { return namedInputs.contains(name); }

        std::vector<std::string> getInputNames() const {
            std::vector<std::string> names;
            names.reserve(namedInputs.size());

            for (const auto &[name, _] : namedInputs)
                names.emplace_back(name);

            std::sort(names.begin(), names.end());
            return names;
        }

        std::string getInputNamesString() const {
            if (namedInputs.size() == 0)
                return "<None>";

            std::ostringstream ss;
            ss << "[";

            const std::vector<std::string> names = getInputNames();
            for (uint64_t i = 0; i < names.size(); ++i) {
                if (i > 0)
                    ss << ", ";
                ss << "\"" << names[i] << "\"";
            }

            ss << "]";
            return ss.str();
        }

        const Tensor &getInput(const std::string &name) const {
            auto it = namedInputs.find(name);
            if (it == namedInputs.end()) {
                throw std::runtime_error("Parameter::StorageContext: No input named \"" + name +
                                         "\". Available inputs: " + getInputNamesString());
            }

            return it->second;
        }

        const Tensor &getFeatureInput() const {
            if (namedInputs.size() != 1) {
                throw std::runtime_error(
                    "Parameter::StorageContext::getFeatureInput: There is not exactly 1 input available; "
                    "use Parameter::StorageContext::getInput(name) instead. Available inputs: " +
                    getInputNamesString());
            }

            return namedInputs.begin()->second;
        }
    };

    virtual ~PhysicalParameter() = default;

    PhysicalParameter(std::string name, bool trainable);  // Later add constraint here
    PhysicalParameter(std::string name,
                      bool trainable,
                      const std::vector<uint64_t> &shape,
                      const TensorDescriptor::DataType dtype);  // Later add constraint here

    virtual void compileStorage(const StorageContext &context);
    virtual void compileStorage(const Tensor &inputTensor);
    void compileInitializer(uint64_t fanIn, uint64_t fanOut);
    void compileInitializer();
    virtual void compileOptimizer(const Optional<Stream> &gradientUpdateStream, bool inferenceOnly);

    virtual void createStorage(const StorageContext &context);
    static Tensor allocateStorage(const TensorPlacement placement,
                                  const std::vector<uint64_t> &shape,
                                  const TensorDescriptor::DataType dtype);
    void clearStorage();

    void initialize(Stream initStream);

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

    void informExpressionBased() {
        expressionBased = true;
        needExpressionRecompile = true;
    }
    bool needsExpressionRecompile() const { return needExpressionRecompile; }
    void informExpressionRecompiled() { needExpressionRecompile = false; }

   protected:
    const std::string name;
    Optional<Tensor> storage;
    const bool trainable;
    bool trainingEnabled;
    bool inferenceOnly = false;
    bool expressionBased = false;
    bool needExpressionRecompile = false;

    std::shared_ptr<Optimizer> optimizer;
    std::shared_ptr<Initializer> initializer;

    Optional<Stream> gradientUpdateStream;

    bool storageInitialized = false;

    Optional<std::vector<uint64_t>> shape = Optional<std::vector<uint64_t>>::empty();
    Optional<TensorDescriptor::DataType> dtype;
};

}  // namespace ThorImplementation
