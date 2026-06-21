#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/LossWeight.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <utility>
#include <optional>
#include <vector>

namespace Thor {

class Loss : public Layer {
   public:
    enum class LabelType { SPARSE = 5, DENSE, INDEX = SPARSE, ONE_HOT = DENSE };
    enum class LossShape { BATCH = 9, ELEMENTWISE, CLASSWISE, RAW };

    Loss() { numInputConnectionsMade = 0; }
    ~Loss() override {}

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    bool mustConnectAllInputsToDriveOutput() const override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override {
        (void)inputTensor;
        numInputConnectionsMade += 1;
        THOR_THROW_IF_FALSE(numInputConnectionsMade <= getLossInputTensors().size());
    }

    virtual Tensor getPredictions() const { return predictionsTensor; }
    virtual Tensor getLabels() const { return labelsTensor; }
    virtual Tensor getLoss() const { return lossTensor; }
    virtual std::optional<Tensor> getExampleWeights() const { return exampleWeightsTensor; }
    std::optional<float> getLossWeight() const { return lossWeight; }
    virtual std::vector<Tensor> getLossInputTensors() const {
        std::vector<Tensor> inputs{predictionsTensor, labelsTensor};
        if (exampleWeightsTensor.has_value())
            inputs.push_back(exampleWeightsTensor.value());
        return inputs;
    }

    // getPredictions() ia a synonym for getFeatureInput().value() and in losses BY DEFAULT ONLY.
    // If the raw predictions are transformed. i.e. by softmax before becoming predictions
    // then featureInput will be a different tensor than predictions,
    // i.e. featureInput will be the input to softmax and predictions will be the output of softmax
    std::optional<Tensor> getFeatureInput() const override { return predictionsTensor; }
    std::optional<Tensor> getFeatureOutput() const override { return lossTensor; }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == labelsTensor) {
            return (int)ThorImplementation::Loss::ConnectionType::LABELS;
        } else if (connectingTensor == predictionsTensor) {
            return (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        } else if (exampleWeightsTensor.has_value() && connectingTensor == exampleWeightsTensor.value()) {
            return (int)ThorImplementation::Loss::ConnectionType::LABELS;
        } else if (connectingTensor == lossTensor) {
            return 0;
        } else {
            return 0;
        }
        THOR_UNREACHABLE();
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override {
        (void)inputTensor;
        if (numInputConnectionsMade == getLossInputTensors().size())
            return {lossTensor};
        else
            return std::vector<Tensor>();
    }

    std::vector<Tensor> getAllOutputTensors() const override { return {getLoss()}; }

   protected:
    Tensor labelsTensor;
    Tensor predictionsTensor;
    Tensor lossTensor;
    std::optional<Tensor> exampleWeightsTensor;

    DataType lossDataType;
    std::optional<float> lossWeight;

    Network *network;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint32_t fixedMem = 4;  // loss scaling factor, FP32

        // Labels
        uint64_t labelsBytes = labelsTensor.getTotalSizeInBytes();

        // Error Output
        uint64_t errorOutputBytes = predictionsTensor.getTotalSizeInBytes();  // FIXME this is not present for inference only

        // Predictions
        uint64_t predictionsOutputBytes = predictionsTensor.getTotalSizeInBytes();

        // Example weights
        uint64_t exampleWeightsBytes = exampleWeightsTensor.has_value() ? exampleWeightsTensor.value().getTotalSizeInBytes() : 0;

        // Loss
        uint64_t lossBytes = lossTensor.getTotalSizeInBytes();

        return fixedMem + batchSize * (predictionsOutputBytes + labelsBytes + exampleWeightsBytes + errorOutputBytes + lossBytes);
    }

    LossShape lossShape;
    Tensor lossShaperInput;

   private:
    uint32_t numInputConnectionsMade = 0;
};

NLOHMANN_JSON_SERIALIZE_ENUM(Loss::LossShape,
                             {
                                 {Loss::LossShape::BATCH, "batch"},
                                 {Loss::LossShape::ELEMENTWISE, "elementwise"},
                                 {Loss::LossShape::CLASSWISE, "classwise"},
                                 {Loss::LossShape::RAW, "raw"},
                             })

}  // namespace Thor
