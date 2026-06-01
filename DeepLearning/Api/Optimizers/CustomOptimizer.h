#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Implementation/Layers/Optimizers/CustomOptimizer.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {

class CustomOptimizer : public Optimizer {
   public:
    using StateSpec = ThorImplementation::CustomOptimizerStateSpec;
    using UpdateContext = ThorImplementation::CustomOptimizerUpdateContext;
    using UpdateExpression = ThorImplementation::CustomOptimizerUpdateExpression;
    using UpdateExpressionBuilder = ThorImplementation::CustomOptimizer::UpdateExpressionBuilder;
    using RuntimeScalarBuilder = ThorImplementation::CustomOptimizer::RuntimeScalarBuilder;

    class Builder;

    CustomOptimizer();
    CustomOptimizer(uint64_t originalId);
    CustomOptimizer(std::vector<StateSpec> stateSpecs,
                    UpdateExpressionBuilder updateExpressionBuilder,
                    RuntimeScalarBuilder runtimeScalarBuilder = {},
                    bool supportsSparseRowGradients = false);

    ~CustomOptimizer() override = default;

    std::shared_ptr<ThorImplementation::Optimizer> stamp(std::shared_ptr<ThorImplementation::TrainableLayer> trainableLayer) override;

    nlohmann::json architectureJson() const override;

    std::string getType() const override { return "CustomOptimizer"; }

    const std::vector<StateSpec>& getStateSpecs() const { return stateSpecs; }
    bool getSupportsSparseRowGradients() const { return supportsSparseRowGradients; }

   protected:
    std::shared_ptr<Optimizer> clone() const override;

   private:
    std::vector<StateSpec> stateSpecs;
    UpdateExpressionBuilder updateExpressionBuilder;
    RuntimeScalarBuilder runtimeScalarBuilder;
    bool supportsSparseRowGradients = false;
};

class CustomOptimizer::Builder {
   public:
    virtual ~Builder() = default;

    virtual std::shared_ptr<CustomOptimizer> build() {
        THOR_THROW_IF_FALSE(static_cast<bool>(_updateExpressionBuilder));

        CustomOptimizer optimizer(_stateSpecs,
                                  _updateExpressionBuilder,
                                  _runtimeScalarBuilder,
                                  _supportsSparseRowGradients.value_or(false));

        if (_network.has_value() && _network.value() != nullptr) {
            optimizer.addToNetwork(_network.value());
            THOR_THROW_IF_FALSE(std::dynamic_pointer_cast<CustomOptimizer>(_network.value()->getDefaultOptimizer()) != nullptr);
            return std::dynamic_pointer_cast<CustomOptimizer>(_network.value()->getDefaultOptimizer());
        }

        return std::dynamic_pointer_cast<CustomOptimizer>(optimizer.clone());
    }

    virtual CustomOptimizer::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual CustomOptimizer::Builder& state(std::string name, DataType dtype = DataType::FP32) {
        _stateSpecs.push_back(StateSpec::sameShapeAsWeights(std::move(name), dtype));
        return *this;
    }

    virtual CustomOptimizer::Builder& state(std::string name, std::vector<uint64_t> shape, DataType dtype = DataType::FP32) {
        _stateSpecs.push_back(StateSpec{std::move(name), dtype, std::move(shape)});
        return *this;
    }

    virtual CustomOptimizer::Builder& updateExpression(UpdateExpressionBuilder builder) {
        THOR_THROW_IF_FALSE(!static_cast<bool>(_updateExpressionBuilder));
        _updateExpressionBuilder = std::move(builder);
        return *this;
    }

    virtual CustomOptimizer::Builder& runtimeScalars(RuntimeScalarBuilder builder) {
        THOR_THROW_IF_FALSE(!static_cast<bool>(_runtimeScalarBuilder));
        _runtimeScalarBuilder = std::move(builder);
        return *this;
    }

    virtual CustomOptimizer::Builder& supportsSparseRowGradients(bool enabled) {
        THOR_THROW_IF_FALSE(!_supportsSparseRowGradients.has_value());
        _supportsSparseRowGradients = enabled;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::vector<StateSpec> _stateSpecs;
    UpdateExpressionBuilder _updateExpressionBuilder = {};
    RuntimeScalarBuilder _runtimeScalarBuilder = {};
    std::optional<bool> _supportsSparseRowGradients;
};

}  // namespace Thor
