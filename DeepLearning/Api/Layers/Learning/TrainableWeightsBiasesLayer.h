#pragma once

#include <boost/interprocess/offset_ptr.hpp>

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

#include <nlohmann/json.hpp>

namespace Thor {

class TrainableWeightsBiasesLayer : public MultiConnectionLayer {
   public:
    using Layer::initialize;  // So that compiler doesn't complain about override below.

    virtual ~TrainableWeightsBiasesLayer() {}

    Tensor getWeights() const { return weights; }
    Optional<Tensor> getBiases() const { return biases; }
    Optional<Tensor> getWeightsGradient() const { return weightsGradient; }
    Optional<Tensor> getBiasesGradient() const { return biasesGradient; }

    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter, Stream stream) const final { assert(false); }
    virtual nlohmann::json serialize(thor_file::TarWriter &archiveWriter, Stream stream, bool saveOptimizerState) const = 0;
    static void deserialize(thor_file::TarReader &archiveReader, const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(thor_file::TarReader &, const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> layer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterLayer,
                                          Optional<Event> sisterLayerLoadedEvent) {
        return MultiConnectionLayer::initialize(layer);
    }

    void attachOptimizer(std::shared_ptr<Optimizer> optimizer) { this->optimizer = optimizer; }
    bool hasOptimizer() const { return optimizer != nullptr; }
    std::shared_ptr<Optimizer> getOptimizer() { return optimizer; }

    void removeOptimizer();

   protected:
    // Helper function to call stamp() on the optimizer and then associate it with the physical layer
    virtual void stampOptimizer(std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalTrainableLayer) const {
        // FIXME: when mutiple stamps are supported, optimizer and layer will need to know if there is already one stamped
        //        for that layer on that GPU, to share things like weights, gradientUpdateStream's.
        if (!physicalTrainableLayer->isInferenceOnly())
            assert(hasOptimizer());
        if (hasOptimizer()) {
            std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = optimizer->stamp(physicalTrainableLayer);
            physicalTrainableLayer->setOptimizer(physicalOptimizer);
        }
    }

    virtual void compile(std::shared_ptr<ThorImplementation::Layer> physicalLayer) {
        if (!physicalLayer->isInferenceOnly())
            assert(hasOptimizer());
        if (hasOptimizer()) {
            assert(physicalLayer != nullptr);
            std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalTrainableLayer =
                dynamic_pointer_cast<ThorImplementation::TrainableWeightsBiasesLayer>(physicalLayer);
            assert(physicalTrainableLayer != nullptr);
            std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = physicalTrainableLayer->getOptimizer();
            assert(physicalOptimizer != nullptr);
            optimizer->compile(physicalOptimizer);
        }

        MultiConnectionLayer::compile(physicalLayer);
    }

    Tensor weights;
    Optional<Tensor> biases;
    Optional<Tensor> weightsGradient;
    Optional<Tensor> biasesGradient;
    std::shared_ptr<Optimizer> optimizer;

    thor_file::TarReader *archiveReader = nullptr;
    Optional<std::string> weightsFile;
    Optional<std::string> biasesFile;
};

}  // namespace Thor
