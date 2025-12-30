#pragma once

#include "DeepLearning/Api/Initializers/UniformRandom.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/BatchNormalization.h"

namespace Thor {

class BatchNormalization : public TrainableWeightsBiasesLayer {
   public:
    class Builder;
    BatchNormalization() {}

    virtual ~BatchNormalization() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<BatchNormalization>(*this); }

    virtual Optional<double> getExponentialRunningAverageFactor() { return exponentialRunningAverageFactor; }
    virtual Optional<double> getEpsilon() { return epsilon; }

    virtual std::string getLayerType() const { return "BatchNormalization"; }

    virtual bool isMultiLayer() const { return false; }

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);

        std::shared_ptr<ThorImplementation::BatchNormalization> physicalBatchNormalization =
            std::make_shared<ThorImplementation::BatchNormalization>(true, getId(), exponentialRunningAverageFactor, epsilon);
        stampOptimizer(physicalBatchNormalization);

        return physicalBatchNormalization;
    }

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalLayer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterPhysicalLayer,
                                  Optional<Event> sisterPhysicalLayerLoadedEvent);

    // mem requirements are the weights
    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        uint64_t numChannels = featureInputs[0].getDimensions()[0];
        uint64_t perInstanceWeights = (4 + featureInputs.size()) * numChannels * 2;  // FP16 FIXME not anymore
        uint64_t perInputState = (2 + featureInputs.size()) * numChannels * 2;       // FP16

        uint64_t featureOutputSize = featureOutputs.size() * featureOutputs[0].getTotalSizeInBytes();
        uint64_t errorOutputSize = featureInputs.size() * featureInputs[0].getTotalSizeInBytes();

        return perInstanceWeights + perInputState + batchSize * (featureOutputSize + errorOutputSize);
    }

    virtual uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                              ThorImplementation::TensorPlacement tensorPlacement) const {
        uint64_t numChannels = featureInputs[0].getDimensions()[0];
        uint64_t perInputState = (2 + featureInputs.size()) * numChannels * 2;  // FP16

        uint64_t featureOutputSize = featureOutputs.size() * featureOutputs[0].getTotalSizeInBytes();
        uint64_t errorOutputSize = featureInputs.size() * featureInputs[0].getTotalSizeInBytes();

        return perInputState + batchSize * (featureOutputSize + errorOutputSize);
    }

   private:
    double exponentialRunningAverageFactor;
    double epsilon;

    Optional<std::string> runningMeansFile;
    Optional<std::string> runningVariancesFile;

    Network *network;
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
        batchNormalization.network = _network.get();
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

   private:
    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<double> _exponentialRunningAverageFactor;
    Optional<double> _epsilon;
};

}  // namespace Thor
