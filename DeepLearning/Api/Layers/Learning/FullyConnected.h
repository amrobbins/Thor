#pragma once

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

//#ifdef THOR_TESTING
//#include <gtest/gtest_prod.h>
//#endif

#include <assert.h>

namespace Thor {

class FullyConnected : public TrainableWeightsBiasesLayer {
   public:
    class Builder;

    FullyConnected() {}

    virtual ~FullyConnected() {}

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<FullyConnected>(*this); }

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    virtual bool isMultiLayer() const {
        assert(featureInputs.size() > 0);
        return useBatchNormalization || dropProportion > 0.0f || activationBuilder || featureInputs.front().getDimensions().size() > 1 ||
               featureInputs.front().getDataType() != Tensor::DataType::FP16;
    }

    virtual void buildSupportLayersAndAddToNetwork();

    virtual void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) {
        std::vector<uint64_t> inputDimensions = inputTensor.getDimensions();
        int gpuNum = stream.getGpuNum();
        assert(!inputDimensions.empty());

        // No matter the incoming shape, the tensor is treated as a one dimensional tensor for fully connected purposes
        // It becomes a matrix when the batch dimension is included.
        assert(inputDimensions.size() >= 1);
        uint64_t numInputFeatures = 1;
        for (uint32_t i = 0; i < inputDimensions.size(); ++i)
            numInputFeatures *= inputDimensions[i];

        ThorImplementation::CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
            gpuNum,
            batchSize,
            numInputFeatures,
            numInputFeatures,
            numOutputFeatures,
            false,
            false,
            ThorImplementation::TensorDescriptor::DataType::FP16);
        ThorImplementation::CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
            gpuNum,
            batchSize,
            numOutputFeatures,
            numInputFeatures,
            numOutputFeatures,
            false,
            true,
            ThorImplementation::TensorDescriptor::DataType::FP16);
        ThorImplementation::CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
            gpuNum,
            batchSize,
            numInputFeatures,
            batchSize,
            numOutputFeatures,
            true,
            false,
            ThorImplementation::TensorDescriptor::DataType::FP16);
    }

    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor) const {
        assert(initialized);
        assert(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

        // Note: Network notes when a layer has already been stamped and only adds a connection, does not re-stamp the layer
        std::shared_ptr<ThorImplementation::FullyConnected> fullyConnected =
            std::make_shared<ThorImplementation::FullyConnected>(numOutputFeatures, hasBias, getId());
        return fullyConnected;
    }

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> layer,
                                                               bool isFirstStamp,
                                                               std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterLayer,
                                                               Optional<Event> sisterLayerLoadedEvent,
                                                               std::vector<std::shared_ptr<Initializer>> &initializers);

    // mem requirements are the weights
    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // FIXME: workspace? Or do I assume no workspace at first and can add one later if have extra mem?
        assert(featureInputs.size() > 0);
        assert(featureInputs[0].getDimensions().size() > 0);
        uint64_t numInputFeatures = 1;
        for (uint32_t i = 0; i < featureInputs[0].getDimensions().size(); ++i)
            numInputFeatures *= featureInputs[0].getDimensions()[i];
        uint64_t numWeights = numInputFeatures * numOutputFeatures;
        uint64_t numBiases = numOutputFeatures;
        // have weights and gradient accumulators, as FP16 elements
        uint64_t fixedMem = 2 * (numWeights + numBiases) * 2;
        uint64_t batchSizeDependentMem =
            2 * featureInputs.size() * (featureInputs[0].getTotalSizeInBytes() + featureOutputs[0].getTotalSizeInBytes()) * batchSize;

        return fixedMem + batchSizeDependentMem;
    }

    virtual uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                              ThorImplementation::TensorPlacement tensorPlacement) const {
        uint64_t batchSizeDependentMem =
            2 * featureInputs.size() * (featureInputs[0].getTotalSizeInBytes() + featureOutputs[0].getTotalSizeInBytes()) * batchSize;
        return batchSizeDependentMem;
    }

    virtual std::string getLayerType() const { return "FullyConnected"; }

   private:
    uint32_t numOutputFeatures;
    bool hasBias;
    std::shared_ptr<Initializer::Builder> weightsInitializerBuilder;
    std::shared_ptr<Initializer::Builder> biasInitializerBuilder;
    std::shared_ptr<Activation::Builder> activationBuilder;

    std::shared_ptr<Layer> activation;
    DropOut dropOut;
    BatchNormalization batchNormalization;

    float dropProportion;

    Network *network;
    bool useBatchNormalization;
    Optional<double> batchNormExponentialRunningAverageFactor;
    Optional<double> batchNormEpsilon;

    Optional<std::string> weightsFile;
    Optional<std::string> biasesFile;

    friend class Network;

    //#ifdef THOR_TESTING
    //    FRIEND_TEST(FullyConnectedTest, SerializeProducesExpectedJson);
    //#endif
};

// featureInput and numOutputFeatures are required, all other parameters are optional.
class FullyConnected::Builder {
   public:
    Builder() { _activationExplicitlyRemoved = false; }

    virtual FullyConnected build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());
        assert(_numOutputFeatures.isPresent());
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializerBuilder == nullptr)
            _weightsInitializerBuilder = std::make_shared<Glorot::Builder>(Glorot::Builder());
        if (_biasInitializerBuilder == nullptr)
            _biasInitializerBuilder = std::make_shared<Glorot::Builder>(Glorot::Builder());
        if (!_activationBuilder && !_activationExplicitlyRemoved)
            _activationBuilder = std::make_shared<Relu::Builder>(Relu::Builder());
        if (_dropProportion.isEmpty())
            _dropProportion = 0.0f;
        if (_useBatchNormalization.isEmpty()) {
            _useBatchNormalization = false;
        }

        FullyConnected fullyConnected;

        fullyConnected.network = _network;
        fullyConnected.featureInputs = _featureInputs;
        fullyConnected.numOutputFeatures = _numOutputFeatures;

        fullyConnected.hasBias = _hasBias;
        fullyConnected.weightsInitializerBuilder = _weightsInitializerBuilder->clone();
        fullyConnected.biasInitializerBuilder = _biasInitializerBuilder->clone();
        if (_activationBuilder != nullptr)
            fullyConnected.activationBuilder = _activationBuilder->clone();
        fullyConnected.dropProportion = _dropProportion;
        fullyConnected.useBatchNormalization = _useBatchNormalization;
        fullyConnected.batchNormExponentialRunningAverageFactor = _batchNormExponentialRunningAverageFactor;
        fullyConnected.batchNormEpsilon = _batchNormEpsilon;
        fullyConnected.initialized = true;

        // When the config requires supporting layers then this layer is not actually added to the network but a subnetwork of layers
        // is added to support the config. It is important that after this happens then the getFeatureOutput() function, called on this
        // stand-in pseudo layer returns the actual featureOut of the real subnetwork.
        if (fullyConnected.isMultiLayer()) {
            fullyConnected.buildSupportLayersAndAddToNetwork();
        } else {
            for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i) {
                fullyConnected.featureOutputs.push_back(Tensor(Tensor::DataType::FP16, {fullyConnected.numOutputFeatures}));
                fullyConnected.outputTensorFromInputTensor[fullyConnected.featureInputs[i]] = fullyConnected.featureOutputs.back();
                fullyConnected.inputTensorFromOutputTensor[fullyConnected.featureOutputs.back()] = fullyConnected.featureInputs[i];
            }
            fullyConnected.addToNetwork(_network.get());
        }

        return fullyConnected;
    }

    virtual FullyConnected::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual FullyConnected::Builder &featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1) {
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
            assert(_featureInputs.back().getDimensions() == _featureInputs.front().getDimensions());
        }
        return *this;
    }

    virtual FullyConnected::Builder &numOutputFeatures(uint32_t _numOutputFeatures) {
        assert(!this->_numOutputFeatures.isPresent());
        this->_numOutputFeatures = _numOutputFeatures;
        return *this;
    }

    virtual FullyConnected::Builder &hasBias(bool _hasBias) {
        assert(!this->_hasBias.isPresent());
        this->_hasBias = _hasBias;
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializerBuilder(Initializer::Builder &_weightsInitializerBuilder) {
        assert(this->_weightsInitializerBuilder == nullptr);
        this->_weightsInitializerBuilder = _weightsInitializerBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializerBuilder(Initializer::Builder &&_weightsInitializerBuilder) {
        assert(this->_weightsInitializerBuilder == nullptr);
        this->_weightsInitializerBuilder = _weightsInitializerBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializerBuilder(Initializer::Builder &_biasInitializerBuilder) {
        assert(this->_biasInitializerBuilder == nullptr);
        this->_biasInitializerBuilder = _biasInitializerBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializerBuilder(Initializer::Builder &&_biasInitializerBuilder) {
        assert(this->_biasInitializerBuilder == nullptr);
        this->_biasInitializerBuilder = _biasInitializerBuilder.clone();
        return *this;
    }

    // Adds an activation layer after this FullyConnected layer
    virtual FullyConnected::Builder &activationBuilder(Activation::Builder &_activationBuilder) {
        assert(this->_activationBuilder == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activationBuilder = _activationBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &activationBuilder(Activation::Builder &&_activationBuilder) {
        assert(this->_activationBuilder == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activationBuilder = _activationBuilder.clone();
        return *this;
    }

    virtual FullyConnected::Builder &noActivation() {
        assert(!this->_activationBuilder);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    // FIXME: batchNormalization and dropOut should be passed as builders. To support this Layer::Builder will need to be created with
    // virtual std::shared_ptr<Layer::Builder> clone.

    // Adds a BatchNormalization layer before this FullyConnected layer and before the DropOut layer when that is also present
    // exponentialRunningAverageFactor and epsilon will be set to good default values when not specified.
    virtual FullyConnected::Builder &batchNormalization(Optional<double> exponentialRunningAverageFactor = Optional<double>::empty(),
                                                        Optional<double> epsilon = Optional<double>::empty()) {
        assert(!_useBatchNormalization.isPresent());
        this->_useBatchNormalization = true;
        this->_batchNormExponentialRunningAverageFactor = exponentialRunningAverageFactor;
        this->_batchNormEpsilon = epsilon;
        return *this;
    }

    // Adds a DropOut layer before this FullyConnected layer, but after the BatchNormalization layer when that is also present.
    virtual FullyConnected::Builder &dropOut(float _dropProportion) {
        assert(!this->_dropProportion.isPresent());
        this->_dropProportion = _dropProportion;
        return *this;
    }

   private:
    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<uint32_t> _numOutputFeatures;
    Optional<bool> _hasBias;
    std::shared_ptr<Initializer::Builder> _weightsInitializerBuilder;
    std::shared_ptr<Initializer::Builder> _biasInitializerBuilder;
    std::shared_ptr<Activation::Builder> _activationBuilder;
    bool _activationExplicitlyRemoved;

    Optional<float> _dropProportion;

    Optional<bool> _useBatchNormalization;
    Optional<double> _batchNormExponentialRunningAverageFactor;
    Optional<double> _batchNormEpsilon;
};

}  // namespace Thor
