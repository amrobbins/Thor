#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/Utility/FiniteCheck.h"

#include <cstdint>
#include <optional>
#include <string>

namespace Thor {

class FiniteCheck : public Layer {
   public:
    class Builder;

    FiniteCheck();
    ~FiniteCheck() override;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<FiniteCheck>(*this); }
    std::string getLayerType() const override { return "FiniteCheck"; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json &j, Network *network);

    const std::string &getTensorLabel() const { return tensorLabel; }
    bool getCheckForward() const { return checkForward; }
    bool getCheckBackward() const { return checkBackward; }
    bool getFailOnNonFinite() const { return failOnNonFinite; }
    uint32_t getMaxReportedIndices() const { return maxReportedIndices; }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                   ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    std::string tensorLabel;
    bool checkForward = true;
    bool checkBackward = true;
    bool failOnNonFinite = true;
    uint32_t maxReportedIndices = 8;
};

class FiniteCheck::Builder {
   public:
    FiniteCheck build();

    Builder &network(Network &network);
    Builder &featureInput(Tensor featureInput);
    Builder &tensorLabel(std::string tensorLabel);
    Builder &checkForward(bool checkForward);
    Builder &checkBackward(bool checkBackward);
    Builder &failOnNonFinite(bool failOnNonFinite);
    Builder &maxReportedIndices(uint32_t maxReportedIndices);

   private:
    std::optional<Network *> _network;
    std::optional<Tensor> _featureInput;
    std::string _tensorLabel;
    bool _checkForward = true;
    bool _checkBackward = true;
    bool _failOnNonFinite = true;
    uint32_t _maxReportedIndices = 8;
};

}  // namespace Thor
