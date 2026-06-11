#pragma once

#include "DeepLearning/Api/DataType.h"
#include "DeepLearning/Api/Layers/Loss/Loss.h"
#include "DeepLearning/Api/Layers/Loss/LossShaper.h"
#include "DeepLearning/Implementation/Layers/Loss/MultiInputCustomLoss.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {

class MultiInputCustomLoss : public Loss {
   public:
    struct InputSpec {
        std::string name;
        Tensor tensor;
        std::optional<std::string> gradientName;

        bool isDifferentiable() const { return gradientName.has_value(); }
    };

    class Builder;

    MultiInputCustomLoss(ThorImplementation::DynamicExpression lossExpression,
                         ThorImplementation::DynamicExpression gradientExpression,
                         std::vector<InputSpec> inputs,
                         std::string lossName = "loss",
                         std::optional<Tensor> lossTensor = std::nullopt,
                         std::optional<DataType> lossDataType = std::nullopt,
                         std::optional<float> lossWeight = std::nullopt);

    ~MultiInputCustomLoss() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<MultiInputCustomLoss>(*this); }
    std::string getLayerType() const override { return "MultiInputCustomLoss"; }

    const std::vector<InputSpec>& getInputs() const { return inputs; }
    const std::string& getLossName() const { return lossName; }
    const ThorImplementation::DynamicExpression& getLossExpression() const { return lossExpression; }
    const ThorImplementation::DynamicExpression& getGradientExpression() const { return gradientExpression; }

    std::vector<Tensor> getLossInputTensors() const override;
    Tensor getPredictions() const override;
    Tensor getLabels() const override;
    std::optional<Tensor> getFeatureInput() const override;
    int getConnectionType(Tensor connectingTensor) const override;

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    virtual bool isMultiLayer() const { return lossShape != LossShape::RAW; }
    virtual void buildSupportLayersAndAddToNetwork();

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    using PhysicalTensor = ThorImplementation::Tensor;
    using PhysicalTensorMap = std::unordered_map<std::string, PhysicalTensor>;

    static void validateName(const std::string& name, const std::string& what);
    static std::set<std::string> toNameSet(const std::vector<std::string>& names);
    static std::set<std::string> gradientNameSet(const std::vector<InputSpec>& inputs);
    static std::string joinNames(const std::set<std::string>& names);
    static PhysicalTensor makeFakePlacedTensor(const Tensor& apiTensor);
    static Tensor logicalLossTensorFromFakeOutput(const std::vector<uint64_t>& fakeOutputDims, DataType dtype);
    static DataType findOutputDType(const std::shared_ptr<ThorImplementation::CompiledOutputs>& compiledOutputs,
                                    const std::string& outputName);

    void validateInputSpecs() const;
    void validateExpressionNames(const ThorImplementation::DynamicExpression& expression,
                                 const std::set<std::string>& outputNames,
                                 const std::string& what) const;
    Tensor inferExpressionTensor(const ThorImplementation::DynamicExpression& expression,
                                 const std::string& outputName,
                                 const std::string& what) const;
    Tensor inferLossTensor() const;
    void validateGradientTensors() const;

    ThorImplementation::DynamicExpression lossExpression;
    ThorImplementation::DynamicExpression gradientExpression;
    std::vector<InputSpec> inputs;
    std::string lossName = "loss";
};

class MultiInputCustomLoss::Builder {
   public:
    virtual ~Builder() = default;

    virtual MultiInputCustomLoss build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_lossExpression != nullptr);
        THOR_THROW_IF_FALSE(_gradientExpression != nullptr);
        THOR_THROW_IF_FALSE(!_inputs.empty());

        LossShape lossShape = _lossShape.value_or(LossShape::BATCH);
        MultiInputCustomLoss customLoss(*_lossExpression,
                                        *_gradientExpression,
                                        _inputs,
                                        _lossName.value_or("loss"),
                                        _lossTensor,
                                        _lossDataType,
                                        ThorImplementation::normalizeLossWeight(_lossWeight));
        customLoss.lossShape = lossShape;
        customLoss.lossWeight = ThorImplementation::normalizeLossWeight(_lossWeight);
        customLoss.network = _network.value();

        if (customLoss.isMultiLayer()) {
            customLoss.buildSupportLayersAndAddToNetwork();
        } else {
            customLoss.lossShaperInput = customLoss.lossTensor;
            customLoss.addToNetwork(_network.value());
        }

        return customLoss;
    }

    virtual MultiInputCustomLoss::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& lossExpression(ThorImplementation::DynamicExpression expression) {
        THOR_THROW_IF_FALSE(this->_lossExpression == nullptr);
        this->_lossExpression = std::make_shared<ThorImplementation::DynamicExpression>(std::move(expression));
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& gradientExpression(ThorImplementation::DynamicExpression expression) {
        THOR_THROW_IF_FALSE(this->_gradientExpression == nullptr);
        this->_gradientExpression = std::make_shared<ThorImplementation::DynamicExpression>(std::move(expression));
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& input(std::string name, Tensor tensor, std::optional<std::string> gradientName = std::nullopt) {
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        std::string effectiveGradientName = gradientName.value_or(name + "_grad");
        _inputs.push_back(InputSpec{std::move(name), std::move(tensor), std::move(effectiveGradientName)});
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& auxiliaryInput(std::string name, Tensor tensor) {
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        _inputs.push_back(InputSpec{std::move(name), std::move(tensor), std::nullopt});
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& lossName(std::string name) {
        THOR_THROW_IF_FALSE(!this->_lossName.has_value());
        this->_lossName = std::move(name);
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& lossTensor(Tensor tensor) {
        THOR_THROW_IF_FALSE(!this->_lossTensor.has_value());
        THOR_THROW_IF_FALSE(tensor.isInitialized());
        this->_lossTensor = std::move(tensor);
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& lossWeight(float lossWeight) {
        THOR_THROW_IF_FALSE(!this->_lossWeight.has_value());
        ThorImplementation::validateLossWeight(lossWeight);
        this->_lossWeight = ThorImplementation::normalizeLossWeight(lossWeight);
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& lossDataType(DataType lossDataType) {
        THOR_THROW_IF_FALSE(!this->_lossDataType.has_value());
        THOR_THROW_IF_FALSE(lossDataType == DataType::FP32 || lossDataType == DataType::FP16);
        this->_lossDataType = lossDataType;
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& reportsBatchLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::BATCH;
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& reportsElementwiseLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::ELEMENTWISE;
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& reportsPerOutputLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::CLASSWISE;
        return *this;
    }

    virtual MultiInputCustomLoss::Builder& reportsRawLoss() {
        THOR_THROW_IF_FALSE(!this->_lossShape.has_value());
        _lossShape = LossShape::RAW;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::shared_ptr<ThorImplementation::DynamicExpression> _lossExpression;
    std::shared_ptr<ThorImplementation::DynamicExpression> _gradientExpression;
    std::vector<InputSpec> _inputs;
    std::optional<std::string> _lossName;
    std::optional<Tensor> _lossTensor;
    std::optional<DataType> _lossDataType;
    std::optional<float> _lossWeight;
    std::optional<LossShape> _lossShape;
};

}  // namespace Thor
