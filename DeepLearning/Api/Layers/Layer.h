#pragma once
#include "DeepLearning/Implementation/ThorError.h"

#include "DeepLearning/Api/Tensor/Tensor.h"
#include <optional>
#include "Utilities/TarFile/TarReader.h"
#include "Utilities/TarFile/TarWriter.h"

#include "DeepLearning/Implementation/Layers/Layer.h"
#include "DeepLearning/Implementation/Tensor/Tensor.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ThorImplementation {
class StampedNetwork;
}
namespace Thor {

class Network;
class Initializer;

class Layer {
   public:
    Layer() : initialized(false), id(getUnusedId()) {}
    virtual ~Layer() {}

    uint64_t getId() const { return id; }
    virtual std::string getLayerVersion() const { return "1.0.0"; }

    virtual std::optional<Tensor> getFeatureOutput() const { return featureOutput; }
    virtual std::optional<Tensor> getFeatureInput() const { return featureInput; }

    virtual std::vector<Tensor> getOutputsFromInput(Tensor inputTensor) {
        THOR_THROW_IF_FALSE(getFeatureInput().has_value());
        THOR_THROW_IF_FALSE(getFeatureOutput().has_value());
        THOR_THROW_IF_FALSE(inputTensor == getFeatureInput().value());
        return {getFeatureOutput().value()};
    }

    virtual bool mustConnectAllInputsToDriveOutput() const { return false; }
    virtual void informThatInputConnectionMade(Tensor inputTensor) {}

    // Some API layers keep temporary bookkeeping while Network::topologicalSort() waits for
    // every logical input to arrive before emitting an output tensor.  The same Network can be
    // sorted, placed, or re-placed more than once, so that traversal/stamping state must be reset
    // before each graph walk or physical stamping pass.
    virtual void resetGraphTraversalState() {}

    bool isInitialized() { return initialized; }

    virtual void preOptimize(Tensor inputTensor, uint64_t batchSize, Stream stream) {}

    [[nodiscard]] virtual uint64_t getOutputTensorBytes(uint32_t batchSize) const {
        if (!featureOutput.has_value())
            return 0UL;
        return featureOutput.value().getTotalSizeInBytes() * batchSize;
    }
    [[nodiscard]] virtual uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                                         ThorImplementation::TensorPlacement tensorPlacement) const {
        return getOutputTensorBytes(batchSize);
    }

    [[nodiscard]] virtual uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                                            ThorImplementation::TensorPlacement tensorPlacement) const {
        return getFirstInstanceMemRequirementInBytes(batchSize, tensorPlacement);
    }

    bool operator==(const Layer &other) const { return id == other.id; }
    bool operator!=(const Layer &other) const { return id != other.id; }
    bool operator<(const Layer &other) const { return id < other.id; }
    bool operator>(const Layer &other) const { return id > other.id; }

    virtual int getConnectionType(Tensor connectingTensor) const {
        THOR_THROW_IF_FALSE(connectingTensor == getFeatureInput().value() || connectingTensor == getFeatureOutput().value());
        return 0;
    }

    virtual std::vector<Tensor> getAllOutputTensors() const { return {getFeatureOutput().value()}; }

    virtual std::shared_ptr<Layer> clone() const = 0;

    static uint64_t getUnusedId() { return nextId.fetch_add(1); }

    virtual std::string getLayerType() const = 0;

    // Serialize for the case that there is no state to save
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter, Stream stream) const { return architectureJson(); }
    // Serialize for the case that there is state to save, defaults to no-state version unless overridden.
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     bool saveOptimizerState,
                                     ThorImplementation::StampedNetwork &stampedNetwork) const {
        return serialize(archiveWriter, stream);
    }
    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    virtual nlohmann::json architectureJson() const { return nlohmann::json{}; };

    class Factory {
       public:
        static const Factory Activation;
        static const Factory Layer;
        static const Factory Learning;
        static const Factory Loss;
        static const Factory Metric;

        const std::string &value() const { return v; }

        operator const std::string &() const { return v; }

        bool operator==(const Factory &other) const { return v == other.v; }
        bool operator!=(const Factory &other) const { return !(*this == other); }
        bool operator==(const std::string &other) const { return v == other; }
        bool operator!=(const std::string &other) const { return !(*this == other); }

       private:
        explicit Factory(std::string s) : v(std::move(s)) {}
        std::string v;
    };

   protected:
    std::optional<Tensor> featureInput;
    std::optional<Tensor> featureOutput;

    // stamp (constructor) -> connect (by Network) -> compile (allocate) -> initialize (set state async)
    virtual std::shared_ptr<ThorImplementation::Layer> stamp(ThorImplementation::TensorPlacement placement,
                                                             std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                                             std::shared_ptr<Thor::Layer> drivingApiLayer,
                                                             Thor::Tensor connectingApiTensor,
                                                             const bool inferenceOnly) const = 0;

    virtual void compile(std::shared_ptr<ThorImplementation::Layer> physicalLayer) { physicalLayer->compile(); }

    // initialize() is called for a layer after it has been stamped, connected and then compiled.
    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::Layer> physicalLayer) {
        if (physicalLayer == nullptr)
            return {};
        physicalLayer->initialize();
        return {};
    }

    virtual void addToNetwork(Network *network);

    // virtual void addTemplateLayerToNetwork(Network* network);

    static void connectTwoLayers(std::shared_ptr<ThorImplementation::Layer> drivingLayer,
                                 std::shared_ptr<ThorImplementation::Layer> loadingLayer,
                                 const std::shared_ptr<Thor::Layer> drivingApiLayer = nullptr,
                                 const std::shared_ptr<Thor::Layer> loadingApiLayer = nullptr,
                                 const Thor::Tensor connectingApiTensor = Thor::Tensor());

    bool initialized;

    uint64_t id;

    static std::string to_snake_case(const std::string &input) {
        std::string out;
        out.reserve(input.size() * 2);

        for (size_t i = 0; i < input.size(); ++i) {
            char c = input[i];
            if (std::isupper(c)) {
                if (i > 0)
                    out.push_back('_');
                out.push_back(std::tolower(c));
            } else {
                out.push_back(c);
            }
        }
        return out;
    }

   private:
    static std::atomic<int64_t> nextId;

    friend class Network;
};

inline bool operator==(const std::string &lhs, const Layer::Factory &rhs) { return lhs == rhs.value(); }

inline bool operator!=(const std::string &lhs, const Layer::Factory &rhs) { return !(lhs == rhs); }

}  // namespace Thor
