#pragma once

#include "DeepLearning/Api/Layers/Layer.h"
#include "DeepLearning/Api/Network/Network.h"
#include "DeepLearning/Implementation/Layers/CustomLayer.h"
#include "Utilities/Expression/DynamicExpression.h"
#include "Utilities/Expression/Expression.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Thor {

class Slice : public Layer {
   public:
    class Builder;

    Slice() = default;
    ~Slice() override = default;

    std::shared_ptr<Layer> clone() const override { return std::make_shared<Slice>(*this); }
    std::string getLayerType() const override { return "Slice"; }

    uint64_t getAxis() const { return axis; }
    int64_t getStart() const { return start; }
    uint64_t getLength() const { return length; }

    nlohmann::json architectureJson() const override;
    static void deserialize(const nlohmann::json& j, Network* network);

   protected:
    std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                     std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                     std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                     Thor::Tensor connectingApiTensor,
                                                     bool inferenceOnly) const override;

    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                   ThorImplementation::TensorPlacement tensorPlacement) const override;

   private:
    static uint64_t normalizeStart(int64_t start, uint64_t axisLength);

    uint64_t axis = 0;
    int64_t start = 0;
    uint64_t length = 0;

    friend class Builder;
};

class Slice::Builder {
   public:
    virtual ~Builder() = default;

    virtual Slice build();

    virtual Builder& network(Network& network) {
        if (_network.has_value())
            throw std::runtime_error("Slice network may only be set once.");
        _network = &network;
        return *this;
    }

    virtual Builder& featureInput(Tensor featureInput) {
        if (_featureInput.has_value())
            throw std::runtime_error("Slice feature input may only be set once.");
        _featureInput = std::move(featureInput);
        return *this;
    }

    virtual Builder& axis(uint64_t axis) {
        if (_axis.has_value())
            throw std::runtime_error("Slice axis may only be set once.");
        _axis = axis;
        return *this;
    }

    virtual Builder& start(int64_t start) {
        if (_start.has_value())
            throw std::runtime_error("Slice start may only be set once.");
        _start = start;
        return *this;
    }

    virtual Builder& length(uint64_t length) {
        if (_length.has_value())
            throw std::runtime_error("Slice length may only be set once.");
        _length = length;
        return *this;
    }

   private:
    std::optional<Network*> _network;
    std::optional<Tensor> _featureInput;
    std::optional<uint64_t> _axis;
    std::optional<int64_t> _start;
    std::optional<uint64_t> _length;
};

}  // namespace Thor
