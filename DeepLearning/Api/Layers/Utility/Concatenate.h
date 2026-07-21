#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Implementation/Layers/Utility/Concatenate.h"
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>


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
        (void)inputTensor;
        numInputConnectionsMade += 1;
        THOR_THROW_IF_FALSE(numInputConnectionsMade <= featureInputs.size());
    }
    void resetGraphTraversalState() override { numInputConnectionsMade = 0; }

    std::string getLayerType() const override { return "Concatenate"; }

    int getConnectionType(Tensor connectingTensor) const override {
        for (uint32_t i = 0; i < featureInputs.size(); ++i) {
            if (connectingTensor == featureInputs[i])
                return static_cast<int>(i);
        }
        if (featureOutputs.size() == 1 && connectingTensor == featureOutputs[0])
            return 0;
        throw std::runtime_error("Tensor is not connected to this Concatenate layer.");
    }

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
            std::make_shared<ThorImplementation::Concatenate>(concatenationAxis + 1, featureInputs.size());
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

        const uint32_t concatenationAxis = _concatenationAxis.value();
        const std::vector<uint64_t> &referenceDimensions = _featureInputs[0].getDimensions();
        if (concatenationAxis >= referenceDimensions.size()) {
            THOR_THROW_LOGIC_ERROR("Concatenate API concatenation axis " + std::to_string(concatenationAxis) +
                                   " is out of range for input rank " + std::to_string(referenceDimensions.size()) +
                                   ". input_shapes=" + inputShapesToString() +
                                   ". API tensor dimensions exclude the batch dimension, so valid axes are 0 through rank - 1.");
        }

        for (uint32_t i = 1; i < _featureInputs.size(); ++i) {
            const std::vector<uint64_t> &dimensions = _featureInputs[i].getDimensions();
            if (dimensions.size() != referenceDimensions.size()) {
                THOR_THROW_LOGIC_ERROR("Concatenate API rank mismatch at input[" + std::to_string(i) +
                                       "]. concatenation_axis=" + std::to_string(concatenationAxis) + ", expected_rank=" +
                                       std::to_string(referenceDimensions.size()) + ", actual_rank=" +
                                       std::to_string(dimensions.size()) + ", input_shapes=" + inputShapesToString() +
                                       ". All inputs must have the same rank and identical logical dimensions except on the concatenation axis.");
            }
            if (_featureInputs[i].getDataType() != _featureInputs[0].getDataType()) {
                THOR_THROW_LOGIC_ERROR("Concatenate API data type mismatch between input[0] and input[" +
                                       std::to_string(i) + "]. input_shapes=" + inputShapesToString() +
                                       ". Convert inputs to the same storage data type before concatenating them.");
            }
            for (uint32_t j = 0; j < referenceDimensions.size(); ++j) {
                if (j == concatenationAxis)
                    continue;
                if (dimensions[j] != referenceDimensions[j]) {
                    THOR_THROW_LOGIC_ERROR("Concatenate API input shape mismatch at input[" + std::to_string(i) +
                                           "], logical dimension " + std::to_string(j) +
                                           ". concatenation_axis=" + std::to_string(concatenationAxis) +
                                           ", expected_dimension=" + std::to_string(referenceDimensions[j]) +
                                           ", actual_dimension=" + std::to_string(dimensions[j]) +
                                           ", input_shapes=" + inputShapesToString() +
                                           ". All inputs must have identical logical dimensions except on concatenation axis " +
                                           std::to_string(concatenationAxis) +
                                           ". Check sequence/window lengths, preserved prefix dimensions, and whether the intended axis was selected.");
                }
            }
        }

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
    static std::string dimensionsToString(const std::vector<uint64_t> &dimensions) {
        std::ostringstream out;
        out << '[';
        for (std::size_t i = 0; i < dimensions.size(); ++i) {
            if (i != 0)
                out << ',';
            out << dimensions[i];
        }
        out << ']';
        return out.str();
    }

    std::string inputShapesToString() const {
        std::ostringstream out;
        out << '{';
        for (std::size_t i = 0; i < _featureInputs.size(); ++i) {
            if (i != 0)
                out << ", ";
            out << "input[" << i << "]=" << dimensionsToString(_featureInputs[i].getDimensions());
        }
        out << '}';
        return out.str();
    }

    std::optional<Network *> _network;
    std::vector<Tensor> _featureInputs;
    std::optional<uint32_t> _concatenationAxis;
};

}  // namespace Thor
