#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Tensor/RaggedTensor.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/LayerNorm.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>

namespace Thor {

class LayerNorm : public TrainableLayer {
   public:
    class Builder;

    LayerNorm() = default;
    ~LayerNorm() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<LayerNorm>(*this); }

    std::string getLayerType() const override { return "LayerNorm"; }

    const std::vector<uint64_t>& getNormalizedShape() const { return normalizedShape; }
    double getEpsilon() const { return epsilon; }
    DataType getParameterDataType() const { return parameterDataType; }
    [[nodiscard]] bool getUseRagged() const { return !raggedFeatureInputs.empty(); }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureInput(uint32_t index = 0) const {
        if (index >= raggedFeatureInputs.size()) return std::nullopt;
        return raggedFeatureInputs[index];
    }
    [[nodiscard]] std::optional<RaggedTensor> getRaggedFeatureOutput(uint32_t index = 0) const {
        if (index >= raggedFeatureOutputs.size()) return std::nullopt;
        return raggedFeatureOutputs[index];
    }

    using MultiConnectionLayer::getFeatureOutput;
    int getConnectionType(Tensor connectingTensor) const override;
    std::vector<Tensor> getFeatureInputs() const override;
    Tensor getFeatureOutput(Tensor inputTensor) const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    void informThatInputConnectionMade(Tensor inputTensor) override;
    void resetGraphTraversalState() override;
    bool mustConnectAllInputsToDriveOutput() const override { return !raggedFeatureInputs.empty(); }

    uint64_t getOutputTensorBytes(uint32_t batchSize) const override {
        if (raggedFeatureOutputs.empty()) return MultiConnectionLayer::getOutputTensorBytes(batchSize);
        THOR_THROW_IF_FALSE(!featureOutputs.empty());
        return featureOutputs.size() * featureOutputs[0].getTotalSizeInBytes();
    }
    uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                      ThorImplementation::TensorPlacement tensorPlacement) const override {
        if (raggedFeatureOutputs.empty())
            return TrainableLayer::getNonFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
        (void)batchSize;
        (void)tensorPlacement;
        THOR_THROW_IF_FALSE(!featureOutputs.empty());
        return featureOutputs.size() * featureOutputs[0].getTotalSizeInBytes();
    }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const nlohmann::json& j, Network* network);
    nlohmann::json architectureJson() const override;

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

   private:
    static bool isLayerNormInputDataType(DataType dataType);
    static uint64_t checkedFeatureCount(const std::vector<uint64_t>& shape, const std::string& what);
    static void validateNormalizedShapeForInput(const std::vector<uint64_t>& inputDims, const std::vector<uint64_t>& normalizedShape);

    std::vector<uint64_t> normalizedShape;
    std::vector<RaggedTensor> raggedFeatureInputs;
    std::vector<RaggedTensor> raggedFeatureOutputs;
    std::vector<uint32_t> inputPortIndicesForTensor(Tensor tensor) const;
    std::set<uint32_t> connectedInputPortIndices;
    std::set<uint32_t> emittedRaggedOutputApplications;
    mutable std::unordered_map<uint64_t, uint32_t> nextInputConnectionCursorByTensorOriginalId;
    std::unordered_map<uint64_t, uint32_t> nextTraversalInputCursorByTensorOriginalId;
    double epsilon = 1.0e-5;
    DataType parameterDataType = DataType::FP32;

    friend class Network;
    friend class Builder;
};

class LayerNorm::Builder {
   public:
    virtual ~Builder() = default;

    virtual LayerNorm build();

    virtual LayerNorm::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual LayerNorm::Builder& featureInput(Tensor featureInput) {
        THOR_THROW_IF_FALSE(featureInput.isInitialized());
        THOR_THROW_IF_FALSE(this->_raggedFeatureInputs.empty());
        this->_featureInputs.push_back(featureInput);
        if (_featureInputs.size() > 1) {
            THOR_THROW_IF_FALSE(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            THOR_THROW_IF_FALSE(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual LayerNorm::Builder& featureInput(RaggedTensor featureInput) {
        THOR_THROW_IF_FALSE(featureInput.isInitialized());
        THOR_THROW_IF_FALSE(this->_featureInputs.empty() || !this->_raggedFeatureInputs.empty());
        Tensor values = featureInput.getValues();
        this->_raggedFeatureInputs.push_back(featureInput);
        this->_featureInputs.push_back(values);
        if (_featureInputs.size() > 1) {
            THOR_THROW_IF_FALSE(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            THOR_THROW_IF_FALSE(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
            THOR_THROW_IF_FALSE(_raggedFeatureInputs.back().getBatchSize() == _raggedFeatureInputs.front().getBatchSize());
            THOR_THROW_IF_FALSE(_raggedFeatureInputs.back().getMaxTotalValues() == _raggedFeatureInputs.front().getMaxTotalValues());
            THOR_THROW_IF_FALSE(_raggedFeatureInputs.back().getOffsetsDataType() == _raggedFeatureInputs.front().getOffsetsDataType());
        }
        return *this;
    }

    virtual LayerNorm::Builder& normalizedShape(const std::vector<uint64_t>& shape) {
        if (!this->_normalizedShape.empty()) {
            throw std::invalid_argument("LayerNorm normalizedShape may only be set once.");
        }
        LayerNorm::checkedFeatureCount(shape, "normalizedShape");
        this->_normalizedShape = shape;
        return *this;
    }

    virtual LayerNorm::Builder& epsilon(double epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = epsilon;
        return *this;
    }

    virtual LayerNorm::Builder& parameterDataType(DataType dtype) {
        THOR_THROW_IF_FALSE(!this->_parameterDataType.has_value());
        this->_parameterDataType = dtype;
        return *this;
    }

    virtual LayerNorm::Builder& weightsInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = initializer;
        return *this;
    }

    virtual LayerNorm::Builder& biasesInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_biasesInitializer == nullptr);
        this->_biasesInitializer = initializer;
        return *this;
    }

    virtual LayerNorm::Builder& weightsOptimizer(std::shared_ptr<Optimizer> optimizer) {
        THOR_THROW_IF_FALSE(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = optimizer;
        return *this;
    }

    virtual LayerNorm::Builder& biasesOptimizer(std::shared_ptr<Optimizer> optimizer) {
        THOR_THROW_IF_FALSE(this->_biasesOptimizer == nullptr);
        this->_biasesOptimizer = optimizer;
        return *this;
    }

   private:
    void verifyConfig() const;

    std::optional<Network*> _network;
    std::vector<Tensor> _featureInputs;
    std::vector<RaggedTensor> _raggedFeatureInputs;
    std::vector<uint64_t> _normalizedShape;
    std::optional<double> _epsilon;
    std::optional<DataType> _parameterDataType;
    std::shared_ptr<Initializer> _weightsInitializer = nullptr;
    std::shared_ptr<Initializer> _biasesInitializer = nullptr;
    std::shared_ptr<Optimizer> _weightsOptimizer = nullptr;
    std::shared_ptr<Optimizer> _biasesOptimizer = nullptr;
};

}  // namespace Thor
