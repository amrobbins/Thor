#pragma once

#include <boost/interprocess/offset_ptr.hpp>

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"
#include "DeepLearning/Api/Parameter/Parameterizable.h"
#include "DeepLearning/Implementation/Layers/TrainableLayer.h"

#include <nlohmann/json.hpp>

#include "DeepLearning/Api/Parameter/ParameterSpecification.h"

namespace Thor {

class TrainableLayer : public MultiConnectionLayer, public Parameterizable {
   public:
    using Layer::initialize;  // So that compiler doesn't complain about override below.

    virtual ~TrainableLayer() {}

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

    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableLayer> layer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::TrainableLayer> sisterLayer,
                                          Optional<Event> sisterLayerLoadedEvent) {
        return MultiConnectionLayer::initialize(layer);
        // FIXME: What do I need to do here?
    }

   protected:
    virtual void compile(std::shared_ptr<ThorImplementation::Layer> physicalLayer) {
        MultiConnectionLayer::compile(physicalLayer);
        // FIXME: What do I need to do here on the API side?
    }

    virtual void serializeParameters(thor_file::TarWriter &archiveWriter,
                                     Stream stream,
                                     bool saveOptimizerState,
                                     ThorImplementation::StampedNetwork &stampedNetwork);

    virtual void deserializeParameters(thor_file::TarReader &archiveReader,
                                 Stream stream,
                                 bool loadOptimizerState,
                                 ThorImplementation::StampedNetwork &stampedNetwork);
};

}  // namespace Thor
