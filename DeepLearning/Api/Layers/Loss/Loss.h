#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Loss.h"

#include <nlohmann/json.hpp>

#include <assert.h>
#include <atomic>
#include <utility>

namespace Thor {

class Loss : public Layer {
   public:
    enum class InputLossType { NUMERICAL_LOSS = 5, CATEGORICAL_LOSS };
    enum class OutputLossType { BATCH_LOSS = 11, CLASSWISE_LOSS };

    Loss() { numInputConnectionsMade = 0; }
    virtual ~Loss() {}

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const { return nlohmann::json{}; }
    static void deserialize(const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    virtual bool mustConnectAllInputsToDriveOutput() { return true; }
    virtual void informThatInputConnectionMade(Tensor inputTensor) {
        numInputConnectionsMade += 1;
        // Only one type of loss is supported at a time.
        assert(numInputConnectionsMade < 3);
    }

    virtual Tensor getPredictions() const { return predictionsTensor; }
    virtual Tensor getLabels() const { return labelsTensor; }
    virtual Tensor getLoss() const { return lossTensor; }

    // getPredictions() ia a synonym for getFeatureInput() and in losses BY DEFAULT ONLY.
    // If the raw predictions are transformed. i.e. by softmax before becoming predictions
    // then featureInput will be a different tensor than predictions,
    // i.e. featureInput will be the input to softmax and predictions will be the output of softmax
    virtual Optional<Tensor> getFeatureInput() const { return predictionsTensor; }
    virtual Optional<Tensor> getFeatureOutput() const { return lossTensor; }

    virtual int getConnectionType(Tensor connectingTensor) const {
        if (connectingTensor == labelsTensor) {
            return (int)ThorImplementation::Loss::ConnectionType::LABELS;
        } else if (connectingTensor == predictionsTensor) {
            return (int)ThorImplementation::Loss::ConnectionType::FORWARD_BACKWARD;
        } else {
            return 0;
        }
        assert(false);
    }

    virtual std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) {
        if (numInputConnectionsMade == 2)
            return {lossTensor};
        else
            return std::vector<Tensor>();
    }

    virtual std::vector<Tensor> getAllOutputTensors() const { return {getLoss()}; }

   protected:
    Tensor labelsTensor;
    Tensor predictionsTensor;
    Tensor lossTensor;

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        uint32_t fixedMem = 4;  // loss scaling factor, FP32

        // Labels
        uint64_t labelsBytes = labelsTensor.getTotalSizeInBytes();

        // Error Output
        uint64_t errorOutputBytes = predictionsTensor.getTotalSizeInBytes();  // FIXME this is not present for inference only

        // Predictions
        uint64_t predictionsOutputBytes = predictionsTensor.getTotalSizeInBytes();

        // Loss
        uint64_t lossBytes = lossTensor.getTotalSizeInBytes();

        return fixedMem + batchSize * (predictionsOutputBytes + labelsBytes + errorOutputBytes + lossBytes);
    }

    // Loss type must be set by deriving class
    ThorImplementation::Loss::LossType lossType;

   private:
    uint32_t numInputConnectionsMade = 0;
};

}  // namespace Thor
