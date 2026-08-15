#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Layers/Metrics/MetricAggregation.h"
#include "DeepLearning/Implementation/Layers/Metric.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <utility>
#include <optional>

namespace Thor {

class Metric : public Layer {
   public:
    Metric() { numInputConnectionsMade = 0; }
    ~Metric() override {}

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    // Note:  Fully connected layers may have multiple independent inputs, each of which is all that
    //        is required to drive an output. This is not the case for layers that require all inputs
    //        to arrive before they can compute the output.
    bool mustConnectAllInputsToDriveOutput() const override { return true; }

    void informThatInputConnectionMade(Tensor inputTensor) override {
        (void)inputTensor;
        numInputConnectionsMade += 1;
        THOR_THROW_IF_FALSE(numInputConnectionsMade <= requiredInputConnectionCount());
    }
    void resetGraphTraversalState() override { numInputConnectionsMade = 0; }

    /** Returns the epoch-level aggregation contract for this metric. */
    virtual MetricAggregation getAggregation() const = 0;
    virtual bool requiresLabels() const { return true; }
    virtual Tensor getPredictions() const { return getFeatureInput().value(); }
    virtual Tensor getLabels() const {
        THOR_THROW_IF_FALSE(requiresLabels());
        return labelsTensor;
    }
    virtual Tensor getMetric() const { return metricTensor; }
    std::optional<Tensor> getFeatureOutput() const override { return getMetric(); }
    std::vector<Tensor> getAllOutputTensors() const override { return {getFeatureOutput().value()}; }

    [[nodiscard]] bool outputTensorDimensionsIncludeBatch(const Tensor& outputTensor) const override {
        THOR_THROW_IF_FALSE(outputTensor == getMetric());
        // Metric outputs are batch reductions. Their API shape is already the complete
        // physical output shape and placement must not prepend a batch-capacity axis.
        return true;
    }

    int getConnectionType(Tensor connectingTensor) const override {
        if (connectingTensor == getFeatureInput().value())
            return (int)ThorImplementation::Metric::ConnectionType::FORWARD;
        else if (requiresLabels() && connectingTensor == labelsTensor)
            return (int)ThorImplementation::Metric::ConnectionType::LABELS;
        else if (connectingTensor == getMetric())
            return (int)ThorImplementation::Metric::ConnectionType::METRIC;
        THOR_UNREACHABLE();
    }

    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override {
        THOR_THROW_IF_FALSE(inputTensor == getFeatureInput().value() || (requiresLabels() && inputTensor == labelsTensor));
        if (numInputConnectionsMade == requiredInputConnectionCount())
            return {metricTensor};
        else
            return std::vector<Tensor>();
    }

   protected:
    Tensor labelsTensor;
    Tensor metricTensor;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // Built-in metrics carry one FP32 validity value per physical batch row.
        uint64_t metricBytes = metricTensor.getTotalSizeInBytes();
        uint64_t batchValidityMaskBytes = static_cast<uint64_t>(batchSize) * sizeof(float);
        return metricBytes + batchValidityMaskBytes;
    }

   private:
    uint32_t requiredInputConnectionCount() const { return requiresLabels() ? 2 : 1; }

    uint32_t numInputConnectionsMade = 0;
};

}  // namespace Thor
