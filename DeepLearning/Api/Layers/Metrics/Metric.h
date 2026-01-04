#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Implementation/Layers/Metric.h"

#include <nlohmann/json.hpp>

#include <assert.h>
#include <atomic>
#include <utility>

namespace Thor {

class Metric : public Layer {
   public:
    Metric() { numInputConnectionsMade = 0; }
    virtual ~Metric() {}

    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter, Stream stream) const;
    static void deserialize(const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    // Note:  Fully connected layers may have multiple independent inputs, each of which is all that
    //        is required to drive an output. This is not the case for layers that require all inputs
    //        to arrive before they can compute the output.
    virtual bool mustConnectAllInputsToDriveOutput() { return true; }

    virtual void informThatInputConnectionMade(Tensor inputTensor) {
        numInputConnectionsMade += 1;
        assert(numInputConnectionsMade < 3);
    }

    virtual Tensor getPredictions() const { return getFeatureInput(); }
    virtual Tensor getLabels() const { return labelsTensor; }
    virtual Tensor getMetric() const { return metricTensor; }
    virtual Optional<Tensor> getFeatureOutput() const { return getMetric(); }
    virtual std::vector<Tensor> getAllOutputTensors() const { return {getFeatureOutput()}; }

    virtual int getConnectionType(Tensor connectingTensor) const {
        if (connectingTensor == getFeatureInput())
            return (int)ThorImplementation::Metric::ConnectionType::FORWARD;
        else if (connectingTensor == getLabels())
            return (int)ThorImplementation::Metric::ConnectionType::LABELS;
        else if (connectingTensor == getMetric())
            return (int)ThorImplementation::Metric::ConnectionType::METRIC;
        assert(false);
    }

    virtual std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) {
        assert(inputTensor == getFeatureInput() || inputTensor == getLabels());
        if (numInputConnectionsMade == 2)
            return {metricTensor};
        else
            return std::vector<Tensor>();
    }

   protected:
    Tensor labelsTensor;
    Tensor metricTensor;

    virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const {
        // Metric
        uint64_t metricBytes = metricTensor.getTotalSizeInBytes();
        return metricBytes;
    }

   private:
    uint32_t numInputConnectionsMade = 0;
};

}  // namespace Thor
