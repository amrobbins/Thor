#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/Concatenate.h"
#include <optional>


namespace Thor {

class Concatenate : public MultiConnectionLayer {
   public:
    class Builder;

    Concatenate();
    ~Concatenate() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Concatenate>(*this); }

    Tensor getFeatureOutput(Tensor inputTensor) const override {
        std::map<Tensor, Tensor>::const_iterator it = outputTensorFromInputTensor.find(inputTensor);
        THOR_THROW_IF_FALSE(it != outputTensorFromInputTensor.end());
        return it->second;
    }

    std::optional<Tensor> getFeatureOutput() const override {
        THOR_THROW_IF_FALSE(featureOutputs.size() == 1);
        return featureOutputs[0];
    }

    Tensor getFeatureInput(Tensor outputTensor) const override {
        // Can't identify a particular input from the concatenated output.
        THOR_UNREACHABLE();
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override {
        if (numInputConnectionsMade == featureInputs.size()) {
            return {featureOutputs[0]};
        } else {
            return std::vector<Tensor>();
        }
    }

    bool mustConnectAllInputsToDriveOutput() const override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override {
        numInputConnectionsMade += 1;
        THOR_THROW_IF_FALSE(numInputConnectionsMade <= featureInputs.size());
    }

    std::string getLayerType() const override { return "Concatenate"; }

    nlohmann::json architectureJson() const override;
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

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
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
        THOR_THROW_IF_FALSE(_network.has_value());
        THOR_THROW_IF_FALSE(!_featureInputs.empty());
        THOR_THROW_IF_FALSE(_concatenationAxis.has_value());
        THOR_THROW_IF_FALSE(_concatenationAxis.value() < _featureInputs[0].getDimensions().size());
        std::set<Tensor> uniqueFeatureInputs(_featureInputs.begin(), _featureInputs.end());
        THOR_THROW_IF_FALSE(uniqueFeatureInputs.size() == _featureInputs.size());  // No duplicate inputs

        Concatenate concatenate;
        concatenate.featureInputs = _featureInputs;
        concatenate.concatenationAxis = _concatenationAxis.value();
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
        concatenate.addToNetwork(_network.value());

        return concatenate;
    }

    virtual Concatenate::Builder &network(Network &_network) {
        THOR_THROW_IF_FALSE(!this->_network.has_value());
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
        THOR_THROW_IF_FALSE(!this->_concatenationAxis.has_value());
        this->_concatenationAxis = _concatenationAxis;
        return *this;
    }

   private:
    std::optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    std::optional<uint32_t> _concatenationAxis;
};

}  // namespace Thor
