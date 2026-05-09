#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/Concatenate.h"


namespace Thor {

class Concatenate : public MultiConnectionLayer {
   public:
    class Builder;

    Concatenate();
    virtual ~Concatenate();

    virtual std::shared_ptr<Layer> clone() const { return std::make_shared<Concatenate>(*this); }

    virtual Tensor getFeatureOutput(Tensor inputTensor) const {
        std::map<Tensor, Tensor>::const_iterator it = outputTensorFromInputTensor.find(inputTensor);
        THOR_THROW_IF_FALSE(it != outputTensorFromInputTensor.end());
        return it->second;
    }

    virtual Optional<Tensor> getFeatureOutput() const {
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    virtual Tensor getFeatureInput(Tensor outputTensor) const {
        // Can't identify a particular input from the concatenated output.
        THOR_UNREACHABLE();
    }

    virtual std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) {
        if (numInputConnectionsMade == featureInputs.size()) {
            return {featureOutputs[0]};
        } else {
            return std::vector<Tensor>();
        }
    }

    virtual bool mustConnectAllInputsToDriveOutput() { return true; }
    virtual void informThatInputConnectionMade(Tensor inputTensor) {
        numInputConnectionsMade += 1;
        THOR_THROW_IF_FALSE(numInputConnectionsMade <= featureInputs.size());
    }

    virtual std::string getLayerType() const { return "Concatenate"; }

    virtual nlohmann::json architectureJson() const;
    static void deserialize(const nlohmann::json &j, Network *network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override {
        (void)inferenceOnly;
        THOR_THROW_IF_FALSE(initialized);
        THOR_THROW_IF_FALSE(outputTensorFromInputTensor.find(connectingApiTensor) != outputTensorFromInputTensor.end());

        // Add 1 to concatenation axis since API does not consider batch size (the first dimension)
        std::shared_ptr<ThorImplementation::Concatenate> concatenate =
            std::make_shared<ThorImplementation::Concatenate>(concatenationAxis + 1);
        return concatenate;
    }

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
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
        THOR_THROW_IF_FALSE(_network.isPresent());
        THOR_THROW_IF_FALSE(!_featureInputs.empty());
        THOR_THROW_IF_FALSE(!_concatenationAxis.isEmpty());
        THOR_THROW_IF_FALSE(_concatenationAxis.get() < _featureInputs[0].getDimensions().size());
        std::set<Tensor> uniqueFeatureInputs(_featureInputs.begin(), _featureInputs.end());
        THOR_THROW_IF_FALSE(uniqueFeatureInputs.size() == _featureInputs.size());  // No duplicate inputs

        Concatenate concatenate;
        concatenate.featureInputs = _featureInputs;
        concatenate.concatenationAxis = _concatenationAxis;
        concatenate.numInputConnectionsMade = 0;

        std::vector<uint64_t> outputDimensions = concatenate.featureInputs[0].getDimensions();
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
        THOR_THROW_IF_FALSE(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual Concatenate::Builder &featureInput(Tensor _featureInput) {
        THOR_THROW_IF_FALSE(!_featureInput.getDimensions().empty());
        this->_featureInputs.push_back(_featureInput);
        if (_featureInputs.size() > 1)
            THOR_THROW_IF_FALSE(_featureInputs.back().getDataType() == _featureInputs.front().getDataType());
        return *this;
    }

    virtual Concatenate::Builder &concatenationAxis(uint32_t _concatenationAxis) {
        THOR_THROW_IF_FALSE(!this->_concatenationAxis.isPresent());
        this->_concatenationAxis = _concatenationAxis;
        return *this;
    }

   private:
    Optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    Optional<uint32_t> _concatenationAxis;
};

}  // namespace Thor
