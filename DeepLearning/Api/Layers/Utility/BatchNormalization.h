#pragma once

#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"

namespace Thor {

class BatchNormalization : public TrainableLayer {
   public:
    class Builder;
    BatchNormalization() {}

    virtual ~BatchNormalization() {}

    std::shared_ptr<Layer> clone() const override { return std::make_shared<BatchNormalization>(*this); }

    virtual Optional<double> getExponentialRunningAverageFactor() { return exponentialRunningAverageFactor; }
    virtual Optional<double> getEpsilon() { return epsilon; }

    std::string getLayerType() const override { return "BatchNormalization"; }

    virtual bool isMultiLayer() const { return false; }

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);
    nlohmann::json architectureJson() const override;

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        assert(initialized);

        std::shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNormalization =
            std::make_shared<ThorImplementation::BatchNormalization>(
                placement, inferenceOnly, numItemsObserved, exponentialRunningAverageFactor, epsilon, Tensor::DataType::FP32, getId());

        return physicalBatchNormalization;
    }

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> physicalLayer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterPhysicalLayer,
                                  Optional<Event> sisterPhysicalLayerLoadedEvent);

   private:
    double exponentialRunningAverageFactor;
    double epsilon;

    Optional<std::string> runningMeansFile;
    Optional<std::string> runningVariancesFile;
    uint64_t numItemsObserved = 0;
    std::shared_ptr<Optimizer> optimizer;
};

class BatchNormalization::Builder {
   public:
    virtual ~Builder() = default;

    virtual BatchNormalization build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());

        BatchNormalization batchNormalization;
        batchNormalization.featureInputs = _featureInputs;
        batchNormalization.exponentialRunningAverageFactor = 0.05;
        if (_exponentialRunningAverageFactor.isPresent())
            batchNormalization.exponentialRunningAverageFactor = _exponentialRunningAverageFactor.get();
        batchNormalization.epsilon = 0.0001;
        if (_epsilon.isPresent())
            batchNormalization.epsilon = _epsilon.get();

        // When this layer gets a specific optimizer, set it now, otherwise network will attach the network default optimizer to it.
        batchNormalization.optimizer = _layerOptimizer;

        batchNormalization.initialized = true;

        for (uint32_t i = 0; i < batchNormalization.featureInputs.size(); ++i) {
            batchNormalization.featureOutputs.push_back(batchNormalization.featureInputs[i].clone());
            batchNormalization.outputTensorFromInputTensor[batchNormalization.featureInputs[i]] = batchNormalization.featureOutputs[i];
            batchNormalization.inputTensorFromOutputTensor[batchNormalization.featureOutputs[i]] = batchNormalization.featureInputs[i];
        }
        batchNormalization.addToNetwork(_network.get());

        return batchNormalization;
    }

    virtual BatchNormalization::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual BatchNormalization::Builder &featureInput(Tensor _featureInput) {
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            assert(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual BatchNormalization::Builder &exponentialRunningAverageFactor(double exponentialRunningAverageFactor) {
        assert(!_exponentialRunningAverageFactor.isPresent());
        assert(exponentialRunningAverageFactor > 0.0);
        assert(exponentialRunningAverageFactor <= 1.0);
        this->_exponentialRunningAverageFactor = exponentialRunningAverageFactor;
        return *this;
    }

    virtual BatchNormalization::Builder &epsilon(double epsilon) {
        assert(!_epsilon.isPresent());
        assert(epsilon > 0.0);
        this->_epsilon = epsilon;
        return *this;
    }

    virtual BatchNormalization::Builder &optimizer(std::shared_ptr<Optimizer> _layerOptimizer) {
        assert(this->_layerOptimizer == nullptr);
        this->_layerOptimizer = _layerOptimizer;
        return *this;
    }

   private:
    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<double> _exponentialRunningAverageFactor;
    Optional<double> _epsilon;
    std::shared_ptr<Optimizer> _layerOptimizer;
};

}  // namespace Thor
