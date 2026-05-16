#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/InstanceNorm.h"
#include "DeepLearning/Implementation/ThorError.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

class InstanceNorm : public TrainableLayer {
   public:
    class Builder;

    InstanceNorm() = default;
    ~InstanceNorm() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<InstanceNorm>(*this); }

    std::string getLayerType() const override { return "InstanceNorm"; }

    uint64_t getChannelCount() const { return channelCount; }
    double getEpsilon() const { return epsilon; }
    Tensor::DataType getParameterDataType() const { return parameterDataType; }

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
    static bool isInstanceNormInputDataType(Tensor::DataType dataType);
    static uint64_t checkedChannelCount(uint64_t channelCount);
    static uint64_t channelCountFromInputDims(const std::vector<uint64_t>& inputDims);
    static void validateInputShape(const std::vector<uint64_t>& inputDims);
    static void validateCudnnFrontendContract(uint64_t channelCount, Tensor::DataType inputDataType);

    uint64_t channelCount = 0;
    double epsilon = 1.0e-5;
    Tensor::DataType parameterDataType = Tensor::DataType::FP32;

    friend class Network;
    friend class Builder;
};

class InstanceNorm::Builder {
   public:
    virtual ~Builder() = default;

    virtual InstanceNorm build();

    virtual InstanceNorm::Builder& network(Network& network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
        this->_network = &network;
        return *this;
    }

    virtual InstanceNorm::Builder& featureInput(Tensor featureInput) {
        THOR_THROW_IF_FALSE(featureInput.isInitialized());
        this->_featureInputs.push_back(featureInput);
        if (_featureInputs.size() > 1) {
            THOR_THROW_IF_FALSE(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            THOR_THROW_IF_FALSE(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual InstanceNorm::Builder& epsilon(double epsilon) {
        THOR_THROW_IF_FALSE(!this->_epsilon.has_value());
        this->_epsilon = epsilon;
        return *this;
    }

    virtual InstanceNorm::Builder& parameterDataType(Tensor::DataType dtype) {
        THOR_THROW_IF_FALSE(!this->_parameterDataType.has_value());
        this->_parameterDataType = dtype;
        return *this;
    }

    virtual InstanceNorm::Builder& weightsInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = initializer;
        return *this;
    }

    virtual InstanceNorm::Builder& biasesInitializer(std::shared_ptr<Initializer> initializer) {
        THOR_THROW_IF_FALSE(this->_biasesInitializer == nullptr);
        this->_biasesInitializer = initializer;
        return *this;
    }

    virtual InstanceNorm::Builder& weightsOptimizer(std::shared_ptr<Optimizer> optimizer) {
        THOR_THROW_IF_FALSE(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = optimizer;
        return *this;
    }

    virtual InstanceNorm::Builder& biasesOptimizer(std::shared_ptr<Optimizer> optimizer) {
        THOR_THROW_IF_FALSE(this->_biasesOptimizer == nullptr);
        this->_biasesOptimizer = optimizer;
        return *this;
    }

   private:
    void verifyConfig() const;

    std::optional<Network*> _network;
    std::vector<Tensor> _featureInputs;
    std::optional<double> _epsilon;
    std::optional<Tensor::DataType> _parameterDataType;
    std::shared_ptr<Initializer> _weightsInitializer = nullptr;
    std::shared_ptr<Initializer> _biasesInitializer = nullptr;
    std::shared_ptr<Optimizer> _weightsOptimizer = nullptr;
    std::shared_ptr<Optimizer> _biasesOptimizer = nullptr;
};

}  // namespace Thor
