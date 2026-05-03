#pragma once

#include <boost/interprocess/offset_ptr.hpp>

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Api/Parameter/ParameterSpecification.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

namespace Thor {

class TrainableLayer : public MultiConnectionLayer, public Parameterizable {
   public:
    using Layer::initialize;  // So that compiler doesn't complain about override below.

    virtual ~TrainableLayer() = default;

    uint64_t getParameterizableId() const override { return getId(); }

    void attachOptimizer(std::shared_ptr<Optimizer> optimizer);
    bool hasOptimizer() const;
    std::shared_ptr<Optimizer> getOptimizer() const;
    void removeOptimizer();

    virtual nlohmann::json architectureJson() const = 0;
    // Trainable layers cannot use serialize(thor_file::TarWriter &archiveWriter, Stream stream), they use the custom signature below.
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter, Stream stream) const final { assert(false); }
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     bool saveOptimizerState,
                                     ThorImplementation::StampedNetwork &stampedNetwork) const = 0;
    static void deserialize(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(std::shared_ptr<thor_file::TarReader> &archiveReader, const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                  bool isFirstStamp,
                                  std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                  Optional<Event> sisterLayerLoadedEvent) override;

   protected:
    void compile(std::shared_ptr<ThorImplementation::Layer> physicalLayer) override;

    void stampOptimizer(const std::shared_ptr<ThorImplementation::TrainableLayer> &physicalTrainableLayer) const;
    std::shared_ptr<Optimizer> optimizerForParameter(const std::shared_ptr<ParameterSpecification> &parameter) const;

    void addParameterArchitectureJson(nlohmann::json &j) const;
    void deserializeParameterArchitectureJson(const nlohmann::json &j, std::shared_ptr<thor_file::TarReader> &archiveReader);

    void serializeParameters(nlohmann::json &j,
                             thor_file::TarWriter &archiveWriter,
                             Stream stream,
                             bool saveOptimizerState,
                             ThorImplementation::StampedNetwork &stampedNetwork) const;

    void deserializeParameters(const nlohmann::json &j, std::shared_ptr<thor_file::TarReader> &archiveReader);

    std::shared_ptr<Optimizer> optimizer;
};

}  // namespace Thor
