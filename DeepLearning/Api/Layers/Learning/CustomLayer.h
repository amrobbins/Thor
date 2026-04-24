#pragma once

#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Parameter/Parameter.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"

#include <assert.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Thor {
class CustomLayer : public TrainableLayer, public Parameterizable {
   public:
    using TensorMap = std::unordered_map<std::string, Tensor>;

    class Builder;

    virtual ~CustomLayer() = default;

    CustomLayer(ThorImplementation::DynamicExpression expr,
                const std::vector<TensorMap>& inputInterfaces,
                bool inferenceOnly = false,
                bool useFastMath = false);

    const std::vector<std::string>& getInputNames() const { return inputNames; }
    const std::vector<std::string>& getOutputNames() const { return outputNames; }
    TensorMap getOutputInterface(const TensorMap& inputInterface) const;
    const ThorImplementation::DynamicExpression& getExpression() const { return expr; }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    nlohmann::json architectureJson() const override;
    // FIXME: Will need deserialize but first need to support serialization and deserialization of expressions.

    // Graph bookkeeping
    std::shared_ptr<Layer> clone() const override { return std::make_shared<CustomLayer>(*this); }
    int getConnectionType(Tensor connectingTensor) const override;
    std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) override;
    bool mustConnectAllInputsToDriveOutput() override { return true; }
    void informThatInputConnectionMade(Tensor inputTensor) override;
    std::string getLayerType() const override { return "CustomLayer"; }

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
    struct InputBinding {
        uint32_t interfaceIndex;
        uint32_t inputPortIndex;
        std::string inputName;
    };

    void assignInputInterfaces(const std::vector<TensorMap>& inputInterfaces);
    void assignOutputInterfaces(const std::vector<TensorMap>& outputInterfaces);
    void validateInputInterfacesMatchExpression() const;
    void validateOutputInterfacesMatchExpression() const;
    static void validateTensorInterface(const TensorMap& tensorInterface, const std::string& what);
    static void validateInterfaceNames(const TensorMap& tensorInterface,
                                       const std::vector<std::string>& expectedNames,
                                       const std::string& what);
    static std::string joinNames(const std::set<std::string>& names);
    static bool interfaceMatches(const TensorMap& subset, const TensorMap& superset);
    void materializeOutputInterfacesFromInputInterfaces();
    Tensor defaultOutputTensorForInterface(const TensorMap& inputInterface, const std::string& outputName) const;
    uint32_t encodeInputConnection(uint32_t interfaceIndex, uint32_t inputPortIndex) const;
    uint32_t encodeOutputConnection(uint32_t interfaceIndex, uint32_t outputPortIndex) const;

    ThorImplementation::DynamicExpression expr;
    bool inferenceOnly = false;
    bool useFastMath = false;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

    std::vector<TensorMap> inputInterfaces;
    std::vector<TensorMap> outputInterfaces;

    // Per-interface readiness is tracked by logical input port, not by tensor, because the same tensor may satisfy
    // several named inputs and/or participate in several input interfaces.
    std::vector<std::set<uint32_t>> connectedInputPortIndicesByInterface;
    std::vector<bool> emittedOutputInterface;

    std::unordered_map<uint64_t, std::vector<InputBinding>> inputBindingsByTensorOriginalId;

    // getConnectionType() is asked once per physical graph connection. If the same tensor drives multiple logical
    // bindings, repeated calls rotate through those bindings so each physical (interface, port) is connected exactly once.
    mutable std::unordered_map<uint64_t, uint32_t> nextInputBindingConnectionCursorByTensorOriginalId;
};

// expr, network, and at least one complete named input interface are required.
class CustomLayer::Builder {
   public:
    virtual ~Builder() = default;

    virtual CustomLayer build() {
        assert(_network.isPresent());
        assert(_expr != nullptr);
        assert(!_inputInterfaces.empty());
        assert(!_expr->getExpectedInputNames().empty());
        assert(!_expr->getExpectedOutputNames().empty());

        CustomLayer customLayer(*_expr, _inputInterfaces, _inferenceOnly, _useFastMath);

        if (_layerOptimizer != nullptr)
            customLayer.optimizer = _layerOptimizer;

        customLayer.addToNetwork(_network.get());
        return customLayer;
    }

    virtual CustomLayer::Builder& network(Network& _network) {
        assert(!this->_network.isPresent());
        this->_network = &_network;
        return *this;
    }

    virtual CustomLayer::Builder& expression(ThorImplementation::DynamicExpression _expr) {
        assert(this->_expr == nullptr);
        this->_expr = std::make_shared<ThorImplementation::DynamicExpression>(std::move(_expr));
        return *this;
    }

    virtual CustomLayer::Builder& inputInterface(const TensorMap& inputInterface) {
        this->_inputInterfaces.push_back(inputInterface);
        return *this;
    }

    // In the vast majority of pointwise and broadcast fused kernels, the kernel is memory bandwidth bound, not compute bound.
    // So the recommendation here is usually to not use Nvidia --use_fast_math compiler flag, which this option enables.
    // It is potentially less accurate often without a performance gain, so only use if you really know that it is something that you want.
    virtual CustomLayer::Builder& useFastMath(bool useFastMath = true) {
        this->_useFastMath = useFastMath;
        return *this;
    }

    virtual CustomLayer::Builder& optimizer(std::shared_ptr<Optimizer> _layerOptimizer) {
        assert(this->_layerOptimizer == nullptr);
        this->_layerOptimizer = std::move(_layerOptimizer);
        return *this;
    }

   private:
    Optional<Network*> _network;
    std::shared_ptr<ThorImplementation::DynamicExpression> _expr;
    std::vector<TensorMap> _inputInterfaces;
    bool _inferenceOnly = false;
    bool _useFastMath = false;
    std::shared_ptr<Optimizer> _layerOptimizer;
};

}  // namespace Thor
