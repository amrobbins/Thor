#pragma once

#include "DeepLearning/Api/Initializers/Glorot.h"
#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Api/Layers/Utility/Reshape.h"
#include "DeepLearning/Api/Layers/Utility/TypeConverter.h"
#include "DeepLearning/Implementation/Layers/NeuralNetwork/FullyConnected.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "DeepLearning/Implementation/Tensor/TensorPlacement.h"
#include "Utilities/Common/Stream.h"
#include "Utilities/Exceptions.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

// #ifdef THOR_TESTING
// #include <gtest/gtest_prod.h>
// #endif

#include <assert.h>

#include <functional>
#include <utility>

#include "DeepLearning/Api/Layers/Activations/SoftPlus.h"
#include "TrainableLayer.h"

namespace Thor {

class FullyConnected : public TrainableLayer {
   public:
    using ExpressionTransform = std::function<ThorImplementation::Expression(const ThorImplementation::Expression&)>;

    class Builder;

    FullyConnected() {}

    virtual ~FullyConnected() = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<FullyConnected>(*this); }

    nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const override;

    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);

    nlohmann::json architectureJson() const override;

   protected:
    void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) override {
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
            gpuNum, batchSize, numInputFeatures, numInputFeatures, numOutputFeatures, false, false, weightsDataType);
        ThorImplementation::CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
            gpuNum, batchSize, numOutputFeatures, numInputFeatures, numOutputFeatures, false, true, weightsDataType);
        ThorImplementation::CublasMatrixMultiply::instance().chooseOptimalMatrixMultiplyKernel(
            gpuNum, batchSize, numInputFeatures, batchSize, numOutputFeatures, true, false, weightsDataType);
    }

    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  Optional<Event> sisterLayerLoadedEvent);

    std::string getLayerType() const override { return "FullyConnected"; }

   private:
    uint32_t numOutputFeatures;
    bool hasBias;
    std::shared_ptr<Activation> activation;
    Tensor::DataType weightsDataType;
    Tensor::DataType computeDataType;
    Tensor::DataType outputDataType;

    // FIXME: These should not be part of Thor::FullyConnected, the builder yes, but the builder should
    //        associate these with the parameters
    std::shared_ptr<Initializer> weightsInitializer;
    std::shared_ptr<Initializer> biasesInitializer;
    std::shared_ptr<Optimizer> weightsOptimizer;
    std::shared_ptr<Optimizer> biasesOptimizer;

    Optional<ExpressionTransform> prologue;
    Optional<ExpressionTransform> epilogue;

    friend class Network;

    // #ifdef THOR_TESTING
    //     FRIEND_TEST(FullyConnectedTest, SerializeProducesExpectedJson);
    // #endif
};

// featureInput and numOutputFeatures are required, all other parameters are optional.
class FullyConnected::Builder {
   public:
    virtual ~Builder() = default;

    Builder() { _activationExplicitlyRemoved = false; }

    virtual FullyConnected build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());
        assert(_numOutputFeatures.isPresent());
        if (_hasBias.isEmpty())
            _hasBias = false;
        if (_weightsInitializer == nullptr)
            _weightsInitializer = Glorot::Builder().build();
        if (_biasInitializer == nullptr)
            _biasInitializer = Glorot::Builder().build();
        if (!_activation && !_activationExplicitlyRemoved)
            _activation = SoftPlus::Builder().build();
        if (_weightsDataType.isEmpty())
            _weightsDataType = _featureInputs[0].getDataType();
        if (_computeDataType.isEmpty())
            _computeDataType = _featureInputs[0].getDataType();
        if (_outputDataType.isEmpty())
            _outputDataType = _featureInputs[0].getDataType();

        FullyConnected fullyConnected;

        fullyConnected.featureInputs = _featureInputs;
        fullyConnected.numOutputFeatures = _numOutputFeatures;

        fullyConnected.hasBias = _hasBias;
        fullyConnected.weightsInitializer = _weightsInitializer->clone();
        fullyConnected.biasesInitializer = _biasInitializer->clone();
        if (_activation != nullptr)
            fullyConnected.activation = _activation;
        fullyConnected.weightsDataType = _weightsDataType;
        fullyConnected.computeDataType = _computeDataType;
        fullyConnected.outputDataType = _outputDataType;

        // When this layer gets a specific optimizer, set it now, otherwise network will attach the network default optimizer to it.
        fullyConnected.weightsOptimizer = _weightsOptimizer;
        fullyConnected.biasesOptimizer = _biasesOptimizer;

        fullyConnected.prologue = _prologue;
        fullyConnected.epilogue = _epilogue;
        fullyConnected.initialized = true;

        for (uint32_t i = 0; i < fullyConnected.featureInputs.size(); ++i)
            fullyConnected.featureOutputs.push_back(Tensor(fullyConnected.outputDataType, {fullyConnected.numOutputFeatures}));
        fullyConnected.addToNetwork(_network.get());

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

    virtual FullyConnected::Builder &weightsInitializer(std::shared_ptr<Initializer> &_weightsInitializer) {
        assert(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &weightsInitializer(std::shared_ptr<Initializer> &&_weightsInitializer) {
        assert(this->_weightsInitializer == nullptr);
        this->_weightsInitializer = _weightsInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(std::shared_ptr<Initializer> &_biasInitializer) {
        assert(this->_biasInitializer == nullptr);
        this->_biasInitializer = _biasInitializer->clone();
        return *this;
    }

    virtual FullyConnected::Builder &biasInitializer(std::shared_ptr<Initializer> &&_biasInitializer) {
        assert(this->_biasInitializer == nullptr);
        this->_biasInitializer = _biasInitializer->clone();
        return *this;
    }

    // Adds an activation layer after this FullyConnected layer
    virtual FullyConnected::Builder &activation(std::shared_ptr<Activation> &_activation) {
        assert(this->_activation == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual FullyConnected::Builder &activation(std::shared_ptr<Activation> &&_activation) {
        assert(this->_activation == nullptr);
        assert(!_activationExplicitlyRemoved);
        this->_activation = _activation;
        return *this;
    }

    virtual FullyConnected::Builder &weightsDataType(Tensor::DataType _weightsDataType) {
        assert(this->_weightsDataType.isEmpty());
        this->_weightsDataType = _weightsDataType;
        return *this;
    }

    virtual FullyConnected::Builder &computeDataType(Tensor::DataType _computeDataType) {
        assert(this->_computeDataType.isEmpty());
        this->_computeDataType = _computeDataType;
        return *this;
    }

    virtual FullyConnected::Builder &outputDataType(Tensor::DataType _outputDataType) {
        assert(this->_outputDataType.isEmpty());
        this->_outputDataType = _outputDataType;
        return *this;
    }

    virtual FullyConnected::Builder &noActivation() {
        assert(!this->_activation);

        _activationExplicitlyRemoved = true;
        return *this;
    }

    virtual FullyConnected::Builder &prologue(ExpressionTransform transform) {
        assert(this->_prologue.isEmpty());
        assert(transform);

        _prologue = std::move(transform);
        return *this;
    }

    virtual FullyConnected::Builder &epilogue(ExpressionTransform transform) {
        assert(this->_epilogue.isEmpty());
        assert(transform);

        _epilogue = std::move(transform);
        return *this;
    }

    virtual FullyConnected::Builder &weightsOptimizer(std::shared_ptr<Optimizer> _weightsOptimizer) {
        assert(this->_weightsOptimizer == nullptr);
        this->_weightsOptimizer = _weightsOptimizer;
        return *this;
    }

    virtual FullyConnected::Builder &biasesOptimizer(std::shared_ptr<Optimizer> _biasesOptimizer) {
        assert(this->_biasesOptimizer == nullptr);
        this->_biasesOptimizer = _biasesOptimizer;
        return *this;
    }

   private:
    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<uint32_t> _numOutputFeatures;
    Optional<bool> _hasBias;
    std::shared_ptr<Activation> _activation;
    Optional<Tensor::DataType> _weightsDataType;
    Optional<Tensor::DataType> _computeDataType;
    Optional<Tensor::DataType> _outputDataType;

    std::shared_ptr<Initializer> _weightsInitializer;
    std::shared_ptr<Initializer> _biasInitializer;
    std::shared_ptr<Optimizer> _weightsOptimizer;
    std::shared_ptr<Optimizer> _biasesOptimizer;
    bool _activationExplicitlyRemoved;

    // FIXME: Future optimization, automatically fuse adjacent prologue and epilogue expressions from adjacent layers.
    Optional<ExpressionTransform> _prologue;
    Optional<ExpressionTransform> _epilogue;
};

}  // namespace Thor
