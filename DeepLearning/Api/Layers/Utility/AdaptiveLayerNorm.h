#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/AdaptiveLayerNorm.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

class AdaptiveLayerNorm : public MultiConnectionLayer {
   public:
    class Builder;

    enum InputPort : uint32_t { DATA = 0, SCALE = 1, BIAS = 2, NUM_INPUT_PORTS = 3 };

    AdaptiveLayerNorm() = default;
    ~AdaptiveLayerNorm() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<AdaptiveLayerNorm>(*this); }

    std::string getLayerType() const override { return "AdaptiveLayerNorm"; }

    std::optional<Tensor> getFeatureOutput() const override {
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    Tensor getFeatureOutput(Tensor inputTensor) const override {
        (void)inputTensor;
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    Tensor getDataInput() const { return featureInputs.at(DATA); }
    Tensor getScaleInput() const { return featureInputs.at(SCALE); }
    Tensor getBiasInput() const { return featureInputs.at(BIAS); }

    const std::vector<uint64_t>& getNormalizedShape() const { return normalizedShape; }
    double getEpsilon() const { return epsilon; }
    DataType getScaleBiasDataType() const { return scaleBiasDataType; }

    int getConnectionType(Tensor connectingTensor) const override;
    bool mustConnectAllInputsToDriveOutput() override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;

    static void deserialize(const nlohmann::json& j, Network* network);
    nlohmann::json architectureJson() const override;

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

   private:
    static bool isAdaptiveLayerNormInputDataType(DataType dataType);
    static uint64_t checkedFeatureCount(const std::vector<uint64_t>& shape, const std::string& what);
    static void validateNormalizedShapeForInput(const std::vector<uint64_t>& inputDims, const std::vector<uint64_t>& normalizedShape);
    static void validateCudnnFrontendContract(uint64_t normalizedFeatureCount, DataType inputDataType);
    static const char* portName(uint32_t port);

    void resetInputConnectionTracking();

    std::vector<uint64_t> normalizedShape;
    double epsilon = 1.0e-5;
    DataType scaleBiasDataType = DataType::FP32;
    std::set<uint64_t> connectedInputOriginalIds;

    friend class Network;
    friend class Builder;
};

class AdaptiveLayerNorm::Builder {
   public:
    virtual ~Builder() = default;

    virtual AdaptiveLayerNorm build();

    virtual AdaptiveLayerNorm::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual AdaptiveLayerNorm::Builder& featureInput(Tensor featureInput) {
        THOR_THROW_IF_FALSE(featureInput.isInitialized());
        this->_featureInput = featureInput;
        return *this;
    }

    virtual AdaptiveLayerNorm::Builder& scaleInput(Tensor scaleInput) {
        THOR_THROW_IF_FALSE(scaleInput.isInitialized());
        this->_scaleInput = scaleInput;
        return *this;
    }

    virtual AdaptiveLayerNorm::Builder& biasInput(Tensor biasInput) {
        THOR_THROW_IF_FALSE(biasInput.isInitialized());
        this->_biasInput = biasInput;
        return *this;
    }

    virtual AdaptiveLayerNorm::Builder& normalizedShape(const std::vector<uint64_t>& shape) {
        if (!this->_normalizedShape.empty()) {
            throw std::invalid_argument("AdaptiveLayerNorm normalizedShape may only be set once.");
        }
        AdaptiveLayerNorm::checkedFeatureCount(shape, "normalizedShape");
        this->_normalizedShape = shape;
        return *this;
    }

    virtual AdaptiveLayerNorm::Builder& epsilon(double epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = epsilon;
        return *this;
    }

    virtual AdaptiveLayerNorm::Builder& scaleBiasDataType(DataType dtype) {
        THOR_THROW_IF_FALSE(!this->_scaleBiasDataType.has_value());
        this->_scaleBiasDataType = dtype;
        return *this;
    }

   private:
    void verifyConfig() const;

    std::optional<Network*> _network;
    std::optional<Tensor> _featureInput;
    std::optional<Tensor> _scaleInput;
    std::optional<Tensor> _biasInput;
    std::vector<uint64_t> _normalizedShape;
    std::optional<double> _epsilon;
    std::optional<DataType> _scaleBiasDataType;
};

}  // namespace Thor
