#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Implementation/Layers/Utility/ScaleGradient.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <cmath>
#include <optional>

namespace Thor {

class ScaleGradient : public Layer {
   public:
    class Builder;
    ScaleGradient();
    ~ScaleGradient() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<ScaleGradient>(*this); }

    std::string getLayerType() const override { return "ScaleGradient"; }
    float getScale() const {
        THOR_THROW_IF_FALSE(scale.has_value());
        return scale.value();
    }

    [[nodiscard]] bool getUseRagged() const { return raggedFeatureInput.has_value(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureInput() const { return raggedFeatureInput; }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureOutput() const { return raggedFeatureOutput; }

    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const override {
        THOR_THROW_IF_FALSE(featureOutput.has_value());
        THOR_THROW_IF_FALSE(outputTensor == featureOutput.value());
        return raggedFeatureInput.has_value();
    }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)placement;
        (void)drivingLayer;
        (void)drivingApiLayer;
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(scale.has_value());
        THOR_THROW_IF_FALSE(connectingApiTensor == getFeatureInput().value());

        return std::make_shared<ThorImplementation::ScaleGradient>(scale.value());
    }

    // Forward aliases the input. Training requires a backward error tensor, which is
    // accounted for by the normal backward workspace planning rather than API output storage.
    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        (void)batchSize;
        (void)tensorPlacement;
        return 0;
    }

   private:
    std::optional<float> scale;
    std::optional<RaggedTensor> raggedFeatureInput;
    std::optional<RaggedTensor> raggedFeatureOutput;

    friend class Builder;
};

class ScaleGradient::Builder {
   public:
    virtual ScaleGradient build() {
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(_featureInput.has_value());
        THOR_THROW_IF_FALSE(_scale.has_value());
        THOR_THROW_IF_FALSE(std::isfinite(_scale.value()));

        ScaleGradient scaleGradient;
        scaleGradient.featureInput = _featureInput;
        scaleGradient.featureOutput = _featureInput.value().clone();
        if (_raggedFeatureInput.has_value()) {
            scaleGradient.raggedFeatureInput = _raggedFeatureInput;
            scaleGradient.raggedFeatureOutput = _raggedFeatureInput->withValues(scaleGradient.featureOutput.value());
        }
        scaleGradient.scale = _scale;
        scaleGradient.initialized = true;
        scaleGradient.addToNetwork(_network.value());
        return scaleGradient;
    }

    virtual ScaleGradient::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &_network;
        return *this;
    }

    virtual ScaleGradient::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        switch (_featureInput.getDataType()) {
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
            case DataType::FP16:
            case DataType::BF16:
            case DataType::FP32:
            case DataType::FP64:
                break;
            default:
                THOR_THROW_LOGIC_ERROR("ScaleGradient requires a floating-point tensor storage type.");
        }
        this->_featureInput = _featureInput;
        return *this;
    }

    virtual ScaleGradient::Builder &featureInput(RaggedTensor _featureInput) {
        THOR_THROW_IF_FALSE(!this->_featureInput.has_value());
        THOR_THROW_IF_FALSE(!this->_raggedFeatureInput.has_value());
        THOR_THROW_IF_FALSE(_featureInput.isInitialized());
        switch (_featureInput.getValuesDataType()) {
            case DataType::FP8_E4M3:
            case DataType::FP8_E5M2:
            case DataType::FP16:
            case DataType::BF16:
            case DataType::FP32:
            case DataType::FP64:
                break;
            default:
                THOR_THROW_LOGIC_ERROR("ScaleGradient requires a floating-point ragged values storage type.");
        }
        this->_raggedFeatureInput = _featureInput;
        this->_featureInput = _featureInput.getValues();
        return *this;
    }

    virtual ScaleGradient::Builder &scale(float _scale) {
        THOR_THROW_IF_FALSE(!this->_scale.has_value());
        THOR_THROW_IF_FALSE(std::isfinite(_scale));
        this->_scale = _scale;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::optional<RaggedTensor> _raggedFeatureInput;
    std::optional<float> _scale;
};

}  // namespace Thor
