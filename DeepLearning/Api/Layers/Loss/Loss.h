#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"
#include "DeepLearning/Implementation/Layers/Loss/LossWeight.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <utility>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace Thor {

class Loss : public Layer {
   public:
    enum class LabelType { SPARSE = 5, DENSE, INDEX = SPARSE, ONE_HOT = DENSE };

    /**
     * Reporting shape for a raw loss tensor with implementation dimensions [batch][d0]...[dn].
     *
     * NONE: do not expose a reportable loss tensor. The raw loss remains the training root.
     * BATCH: one scalar, summing all non-batch values and averaging those sums over the batch.
     * PER_EXAMPLE: one scalar per example, summing all non-batch values.
     * PER_OUTPUT: average over the batch while preserving [d0]...[dn].
     * RAW: preserve [batch][d0]...[dn].
     */
    enum class LossShape { NONE, BATCH, PER_EXAMPLE, PER_OUTPUT, RAW };

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
    void resetGraphTraversalState() override { numInputConnectionsMade = 0; }

    virtual Tensor getPredictions() const { return predictionsTensor; }
    virtual Tensor getLabels() const { return labelsTensor; }
    virtual Tensor getLoss() const {
        if (lossShape == LossShape::NONE)
            throw std::runtime_error(getLayerType() + " was configured with LossShape::NONE and does not expose a reported loss tensor.");
        THOR_THROW_IF_FALSE(lossTensor.isInitialized());
        return lossTensor;
    }
    virtual Tensor getRawLoss() const {
        THOR_THROW_IF_FALSE(lossShaperInput.isInitialized());
        return lossShaperInput;
    }
    bool reportsLoss() const { return lossShape != LossShape::NONE; }
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
    std::optional<Tensor> getFeatureOutput() const override { return getRawLoss(); }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == labelsTensor) {
            return (int)ThorImplementation::Loss::ConnectionType::LABELS;
        } else if (connectingTensor == predictionsTensor) {
            return (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        } else if (exampleWeightsTensor.has_value() && connectingTensor == exampleWeightsTensor.value()) {
            return (int)ThorImplementation::Loss::ConnectionType::LABELS;
        } else if (connectingTensor == getRawLoss()) {
            return 0;
        } else {
            return 0;
        }
        THOR_UNREACHABLE();
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override {
        (void)inputTensor;
        if (numInputConnectionsMade == getLossInputTensors().size())
            return {getRawLoss()};
        else
            return std::vector<Tensor>();
    }

    std::vector<Tensor> getAllOutputTensors() const override { return {getRawLoss()}; }

    [[nodiscard]] std::optional<std::string> getInputPortName(const Tensor& inputTensor) const override {
        if (predictionsTensor.isInitialized() && inputTensor == predictionsTensor) {
            return "predictions";
        }
        if (labelsTensor.isInitialized() && inputTensor == labelsTensor) {
            return "labels";
        }
        if (exampleWeightsTensor.has_value() && inputTensor == exampleWeightsTensor.value()) {
            return "example_weights";
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::string> getOutputPortName(const Tensor& outputTensor) const override {
        if (lossShaperInput.isInitialized() && outputTensor == lossShaperInput) {
            return "raw_loss";
        }
        return std::nullopt;
    }

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
        uint64_t lossBytes = getRawLoss().getTotalSizeInBytes();

        return fixedMem + batchSize * (predictionsOutputBytes + labelsBytes + exampleWeightsBytes + errorOutputBytes + lossBytes);
    }

    LossShape lossShape;
    Tensor lossShaperInput;

    void finalizeLossReporting();

   private:
    uint32_t numInputConnectionsMade = 0;
};

inline void to_json(nlohmann::json &j, const Loss::LossShape &lossShape) {
    switch (lossShape) {
        case Loss::LossShape::NONE:
            j = "none";
            return;
        case Loss::LossShape::BATCH:
            j = "batch";
            return;
        case Loss::LossShape::PER_EXAMPLE:
            j = "per_example";
            return;
        case Loss::LossShape::PER_OUTPUT:
            j = "per_output";
            return;
        case Loss::LossShape::RAW:
            j = "raw";
            return;
    }
    throw std::invalid_argument("Unsupported LossShape enum value.");
}

inline void from_json(const nlohmann::json &j, Loss::LossShape &lossShape) {
    const std::string serializedLossShape = j.get<std::string>();
    if (serializedLossShape == "none") {
        lossShape = Loss::LossShape::NONE;
    } else if (serializedLossShape == "batch") {
        lossShape = Loss::LossShape::BATCH;
    } else if (serializedLossShape == "per_example") {
        lossShape = Loss::LossShape::PER_EXAMPLE;
    } else if (serializedLossShape == "per_output") {
        lossShape = Loss::LossShape::PER_OUTPUT;
    } else if (serializedLossShape == "raw") {
        lossShape = Loss::LossShape::RAW;
    } else {
        throw std::invalid_argument("Unsupported loss shape '" + serializedLossShape +
                                    "'. Expected none, batch, per_example, per_output, or raw.");
    }
}

}  // namespace Thor
