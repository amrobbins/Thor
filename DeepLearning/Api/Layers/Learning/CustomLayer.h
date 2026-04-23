#pragma once

#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Parameter/Parameter.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {
class CustomLayer : public TrainableLayer, public Parameterizable {
   public:
    using NamedTensor = std::pair<std::string, Tensor>;

    virtual ~CustomLayer() = default;

    CustomLayer(ThorImplementation::DynamicExpression expr,
                const std::vector<NamedTensor>& namedInputs,
                const std::vector<NamedTensor>& namedOutputs,
                bool inferenceOnly = false,
                bool useFastMath = false);

    std::shared_ptr<Layer> clone() const override { return std::make_shared<CustomLayer>(*this); }

    int getConnectionType(Tensor connectingTensor) const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override;

    std::string getLayerType() const override { return "CustomLayer"; }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    nlohmann::json architectureJson() const override;

    const std::vector<std::string>& getInputNames() const { return inputNames; }
    const std::vector<std::string>& getOutputNames() const { return outputNames; }
    const ThorImplementation::DynamicExpression& getExpression() const { return expr; }

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor) const override;

    void compile(std::shared_ptr<ThorImplementation::Layer> physicalLayer) override { physicalLayer->compile(); }

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  Optional<Event> sisterLayerLoadedEvent) override {
        return Layer::initialize(layer);
    }

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    void assignNamedInputs(const std::vector<NamedTensor>& namedInputs);
    void assignNamedOutputs(const std::vector<NamedTensor>& namedOutputs);
    static void validateNamedTensorList(const std::vector<NamedTensor>& namedTensors, const std::string& what);

    ThorImplementation::DynamicExpression expr;
    bool inferenceOnly = false;
    bool useFastMath = false;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, uint32_t> inputPortByName;
    std::unordered_map<std::string, uint32_t> outputPortByName;
    std::set<uint64_t> connectedInputTensorOriginalIds;
};

}  // namespace Thor
