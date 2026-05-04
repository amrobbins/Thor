#pragma once

#include <boost/interprocess/offset_ptr.hpp>

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

namespace Thor {

class TrainableLayer : public MultiConnectionLayer, public Parameterizable {
   public:
    using Layer::initialize;  // So that compiler doesn't complain about override below.

    TrainableLayer(std::vector<std::shared_ptr<ParameterSpecification>> parameters = {});

    virtual ~TrainableLayer() = default;

    void attachDefaultOptimizer(std::shared_ptr<Optimizer> optimizer);
    std::shared_ptr<Optimizer> getOptimizer() const;
    void removeOptimizer();

    virtual nlohmann::json architectureJson() const override = 0;
    // Trainable layers cannot use serialize(thor_file::TarWriter &archiveWriter, Stream stream), they use the custom signature below.
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter, Stream stream) const final { assert(false); }
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     bool saveOptimizerState,
                                     ThorImplementation::StampedNetwork &stampedNetwork) const override;
    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  Optional<Event> sisterLayerLoadedEvent);

    virtual uint64_t getParameterBytes() const;

    // mem requirements are the weights
    uint64_t getFirstInstanceMemRequirementInBytes(uint32_t batchSize, ThorImplementation::TensorPlacement tensorPlacement) const override {
        // FIXME: workspace? Or do I assume no workspace at first and can add one later if have extra mem?
        return getOutputTensorBytes(batchSize) + getParameterBytes();
    }

    uint64_t getNonFirstInstanceMemRequirementInBytes(uint32_t batchSize,
                                                      ThorImplementation::TensorPlacement tensorPlacement) const override {
        uint64_t batchSizeDependentMem = featureOutputs.size() * featureOutputs[0].getTotalSizeInBytes() * batchSize;
        return batchSizeDependentMem;
    }
    bool hasOptimizer() const;

   protected:
    void compile(std::shared_ptr<ThorImplementation::Layer> physicalLayer) override;

    // FIXME: Delete
    void deserializeParameterArchitectureJson(const nlohmann::json &j, std::shared_ptr<thor_file::TarReader> &archiveReader);

    void serializeParameters(nlohmann::json &j,
                             thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const;

    void deserializeParameters(const nlohmann::json &j, std::shared_ptr<thor_file::TarReader> &archiveReader);

    std::shared_ptr<thor_file::TarReader> archiveReader;
};

}  // namespace Thor
