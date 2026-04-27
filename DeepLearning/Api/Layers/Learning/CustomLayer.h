#pragma once

#include "DeepLearning/Api/Layers/Learning/TrainableLayer.h"
#include "DeepLearning/Api/Parameter/BoundParameter.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
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
                std::vector<std::shared_ptr<ParameterSpecification>> parameters = {},
                bool useFastMath = false);

    CustomLayer(ThorImplementation::DynamicExpression expr,
                std::vector<std::string> inputNames,
                std::vector<std::string> outputNames,
                const std::vector<TensorMap>& inputInterfaces,
                const std::vector<TensorMap>& outputInterfaces,
                std::vector<std::shared_ptr<ParameterSpecification>> parameters = {},
                bool useFastMath = false);

    const std::vector<std::string>& getInputNames() const { return inputNames; }
    const std::vector<std::string>& getOutputNames() const { return outputNames; }
    TensorMap getInputInterface(uint32_t interfaceIndex = 0) const;
    TensorMap getOutputInterface(const TensorMap& inputInterface) const;
    TensorMap getOutputInterfaceByIndex(uint32_t interfaceIndex = 0) const;
    Tensor getOutput(const std::string& outputName, uint32_t interfaceIndex = 0) const;
    const ThorImplementation::DynamicExpression& getExpression() const { return expr; }
    const std::vector<std::shared_ptr<ParameterSpecification>>& getParameters() const override { return parameters; }
    uint64_t getParameterizableId() const override { return getId(); }

    nlohmann::json serialize(thor_file::TarWriter& archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork& stampedNetwork) const override;
    nlohmann::json architectureJson() const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader>& archiveReader, const nlohmann::json& j, Network* network);

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
                                                     Thor::Tensor connectingApiTensor,
                                                     const bool inferenceOnly) const override;

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
    void validateParameterNames() const;
    void validateExpressionCanBindFeatureAndParameterInputs() const;
    static void validateTensorInterface(const TensorMap& tensorInterface, const std::string& what);
    static void validateInterfaceNames(const TensorMap& tensorInterface,
                                       const std::vector<std::string>& expectedNames,
                                       const std::string& what);
    static std::string joinNames(const std::set<std::string>& names);
    static bool interfaceMatches(const TensorMap& subset, const TensorMap& superset);
    void materializeOutputInterfacesFromInputInterfaces();
    TensorMap inferOutputInterfaceFromInputInterface(const TensorMap& inputInterface) const;
    uint32_t encodeInputConnection(uint32_t interfaceIndex, uint32_t inputPortIndex) const;
    uint32_t encodeOutputConnection(uint32_t interfaceIndex, uint32_t outputPortIndex) const;

    ThorImplementation::DynamicExpression expr;
    bool useFastMath = false;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;

    std::vector<TensorMap> inputInterfaces;
    std::vector<TensorMap> outputInterfaces;

    std::vector<std::shared_ptr<ParameterSpecification>> parameters;

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
        if (_inputNames.empty())
            _inputNames = _expr->getExpectedInputNames();
        if (_outputNames.empty())
            _outputNames = _expr->getExpectedOutputNames();
        assert(!_inputNames.empty());
        assert(!_outputNames.empty());

        CustomLayer customLayer(*_expr, _inputNames, _outputNames, _inputInterfaces, _outputInterfaces, _parameters, _useFastMath);

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

    virtual CustomLayer::Builder& inputNames(std::vector<std::string> inputNames) {
        assert(this->_inputNames.empty());
        this->_inputNames = std::move(inputNames);
        return *this;
    }

    virtual CustomLayer::Builder& outputNames(std::vector<std::string> outputNames) {
        assert(this->_outputNames.empty());
        this->_outputNames = std::move(outputNames);
        return *this;
    }

    virtual CustomLayer::Builder& inputInterface(const TensorMap& inputInterface) {
        this->_inputInterfaces.push_back(inputInterface);
        return *this;
    }

    virtual CustomLayer::Builder& outputInterface(const TensorMap& outputInterface) {
        this->_outputInterfaces.push_back(outputInterface);
        return *this;
    }

    // Note: parameters are copied so that later mutations to the parameter that was passed in do not
    //       affect the parameter as it was passed to the builder.
    //       Once a parameter is bound to a layer, the user needs to interact with a BoundParameter instance instead.
    virtual CustomLayer::Builder& parameter(std::shared_ptr<ParameterSpecification> parameter) {
        assert(parameter != nullptr);
        this->_parameters.push_back(std::make_shared<ParameterSpecification>(*parameter));
        return *this;
    }

    virtual CustomLayer::Builder& parameters(std::vector<std::shared_ptr<ParameterSpecification>> parameters) {
        assert(this->_parameters.empty());
        this->_parameters.clear();
        this->_parameters.reserve(parameters.size());

        for (const auto& parameter : parameters) {
            if (parameter == nullptr) {
                throw std::runtime_error("CustomLayer::Builder received a null ParameterSpecification.");
            }

            this->_parameters.push_back(std::make_shared<ParameterSpecification>(*parameter));
        }

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
    std::vector<std::string> _inputNames;
    std::vector<std::string> _outputNames;
    std::vector<TensorMap> _inputInterfaces;
    std::vector<TensorMap> _outputInterfaces;
    std::vector<std::shared_ptr<ParameterSpecification>> _parameters;
    bool _useFastMath = false;
    std::shared_ptr<Optimizer> _layerOptimizer;
};

}  // namespace Thor
