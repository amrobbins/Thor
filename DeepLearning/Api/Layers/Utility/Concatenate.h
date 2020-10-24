#pragma once

#include "DeepLearning/Api/Initializers/Initializer.h"
#include "DeepLearning/Api/Initializers/UniformRandomInitializer.h"
#include "DeepLearning/Api/Layers/Activations/Activation.h"
#include "DeepLearning/Api/Layers/Activations/Relu.h"
#include "DeepLearning/Api/Layers/Activations/Tanh.h"
#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Learning/TrainableWeightsBiasesLayer.h"
#include "DeepLearning/Api/Layers/Utility/BatchNormalization.h"
#include "DeepLearning/Api/Layers/Utility/DropOut.h"
#include "DeepLearning/Api/Layers/Utility/Flatten.h"
#include "DeepLearning/Implementation/Layers/Utility/Concatenate.h"
#include "DeepLearning/Implementation/Layers/Utility/Flatten.h"
#include "DeepLearning/Implementation/Tensor/TensorDescriptor.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasMatrixMultiply.h"

#include <assert.h>

namespace Thor {

class Concatenate : public TrainableWeightsBiasesLayer {
   public:
    class Builder;

    Concatenate() {}

    virtual ~Concatenate() {}

    virtual shared_ptr<Layer> clone() const { return make_shared<Concatenate>(*this); }

    virtual Tensor getFeatureOutput(Tensor inputTensor) const {
        map<Tensor, Tensor>::const_iterator it = outputTensorFromInputTensor.find(inputTensor);
        assert(it != outputTensorFromInputTensor.end());
        return it->second;
    }

    virtual Optional<Tensor> getFeatureOutput() const {
        assert(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    virtual Tensor getFeatureInput(Tensor outputTensor) const {
        // Can't identify a particular input from the concatenated output.
        assert(false);
    }

    virtual vector<Tensor> getOutputsFromInput(Tensor inputTensor) {
        if (numInputConnectionsMade == featureInputs.size()) {
            return {featureOutputs[0]};
        } else {
            return vector<Tensor>();
        }
    }

    virtual bool mustConnectAllInputsToDriveOutput() { return true; }
    virtual void informThatInputConnectionMade(Tensor inputTensor) {
        numInputConnectionsMade += 1;
        assert(numInputConnectionsMade <= featureInputs.size());
    }

   protected:
    virtual ThorImplementation::Layer *stamp(ThorImplementation::TensorPlacement placement,
                                             ThorImplementation::Layer *drivingLayer,
                                             Thor::Layer *drivingApiLayer,
                                             Thor::Tensor connectingApiTensor,
                                             vector<shared_ptr<Initializer>> &initializers,
                                             StreamPackage gradientUpdateStreamPackage = StreamPackage()) const {
        assert(initialized);
        assert(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

        // Add 1 to concatenation axis since API does not consider batch size (the first dimension)
        ThorImplementation::Concatenate *concatenate = new ThorImplementation::Concatenate(concatenationAxis + 1);
        Thor::Layer::connectTwoLayers(drivingLayer, concatenate, drivingApiLayer, this, connectingApiTensor);

        return concatenate;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, TensorPlacement tensorPlacement) const {
        // featureOutput and errorInput
        return (featureInputs[0].getTotalSizeInBytes() + featureOutputs[0].getTotalSizeInBytes()) * batchSize;
    }

   private:
    uint32_t concatenationAxis;
    uint32_t numInputConnectionsMade;

    friend class Network;
};

// featureInput and numOutputFeatures are required, all other parameters are optional.
class Concatenate::Builder {
   public:
    Builder() {}

    virtual Concatenate build() {
        assert(_network.isPresent());
        assert(!_featureInputs.empty());
        assert(!_concatenationAxis.isEmpty());
        assert(_concatenationAxis.get() < _featureInputs[0].getDimensions().size());
        set<Tensor> uniqueFeatureInputs(_featureInputs.begin(), _featureInputs.end());
        assert(uniqueFeatureInputs.size() == _featureInputs.size());  // No duplicate inputs

        Concatenate concatenate;
        concatenate.featureInputs = _featureInputs;
        concatenate.concatenationAxis = _concatenationAxis;
        concatenate.numInputConnectionsMade = 0;

        vector<uint64_t> outputDimensions = concatenate.featureInputs[0].getDimensions();
        outputDimensions[concatenate.concatenationAxis] = 0;
        for (uint32_t i = 0; i < concatenate.featureInputs.size(); ++i) {
            outputDimensions[concatenate.concatenationAxis] += concatenate.featureInputs[i].getDimensions()[concatenate.concatenationAxis];
        }
        concatenate.featureOutputs.push_back(Tensor(concatenate.featureInputs[0].getDataType(), outputDimensions));

        for (uint32_t i = 0; i < concatenate.featureInputs.size(); ++i) {
            concatenate.outputTensorFromInputTensor[concatenate.featureInputs[i]] = concatenate.featureOutputs[0];
        }

        concatenate.initialized = true;
        concatenate.addToNetwork(_network);

        return concatenate;
    }

    virtual Concatenate::Builder &network(Network &_network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Concatenate::Builder &featureInput(Tensor _featureInput) {
        assert(!_featureInput.getDimensions().empty());
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1)
            assert(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
        return *this;
    }

    virtual Concatenate::Builder &concatenationAxis(uint32_t _concatenationAxis) {
        assert(!this->_concatenationAxis.isPresent());
        this->_concatenationAxis = _concatenationAxis;
        return *this;
    }

   private:
    Optional<Network *> _network;
    vector<Tensor> _featureInputs;
    Optional<uint32_t> _concatenationAxis;
};

}  // namespace Thor
