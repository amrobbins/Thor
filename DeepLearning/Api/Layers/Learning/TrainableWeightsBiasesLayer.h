#pragma once

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

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const { return nlohmann::json{}; }
    static void deserialize(const nlohmann::json &j, Network *network);
    using Deserializer = std::function<void(const nlohmann::json &, Network *)>;
    static std::unordered_map<std::string, Deserializer> &get_registry();
    static void register_layer(std::string name, Deserializer fn);

    virtual void stampOptimizer(std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> physicalLayer) const {
        // FIXME: when mutiple stamps are supported, optimizer and layer will need to know if there is already one stamped
        //        for that layer on that GPU, to share things like weights, gradientUpdateStream's.
        assert(physicalLayer->isCompiled());
        if (hasOptimizer()) {
            std::shared_ptr<ThorImplementation::Optimizer> physicalOptimizer = optimizer->stamp(physicalLayer);
            physicalLayer->setOptimizer(physicalOptimizer);
        }
    }

    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> layer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterLayer,
                                          Optional<Event> sisterLayerLoadedEvent) {
        return {};
    }

    void attachOptimizer(std::shared_ptr<Optimizer> optimizer) { this->optimizer = optimizer; }
    bool hasOptimizer() const { return optimizer != nullptr; }
    std::shared_ptr<Optimizer> getOptimizer() { return optimizer; }

    void removeOptimizer();

   protected:
    Tensor weights;
    Optional<Tensor> biases;
    Optional<Tensor> weightsGradient;
    Optional<Tensor> biasesGradient;
    std::shared_ptr<Optimizer> optimizer;

    Optional<std::string> weightsFile;
    Optional<std::string> biasesFile;
};

}  // namespace Thor
