#pragma once

#include "DeepLearning/Api/Layers/MultiConnectionLayer.h"
#include "DeepLearning/Api/Optimizers/Optimizer.h"

#include <nlohmann/json.hpp>

namespace Thor {

class TrainableWeightsBiasesLayer : public MultiConnectionLayer {
   public:
    virtual ~TrainableWeightsBiasesLayer() {}

    Tensor getWeights() const { return weights; }
    Optional<Tensor> getBiases() const { return biases; }
    Optional<Tensor> getWeightsGradient() const { return weightsGradient; }
    Optional<Tensor> getBiasesGradient() const { return biasesGradient; }

    virtual nlohmann::json serialize(const std::string &storageDir, Stream stream) const { return nlohmann::json{}; }
    static void deserialize(const nlohmann::json &j, Network *network);
    static std::unordered_map<std::string, std::function<void(const nlohmann::json&, Network*)>> registry;

    virtual std::vector<Event> initialize(std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> layer,
                                          bool isFirstStamp,
                                          std::shared_ptr<ThorImplementation::TrainableWeightsBiasesLayer> sisterLayer,
                                          Optional<Event> sisterLayerLoadedEvent,
                                          std::vector<std::shared_ptr<Initializer>> &initializers) {
        return {};
    }

    // Override the network default optimizer with a different one
    void setLayerSpecificOptimizer(Optimizer *layerSpecificOptimizer) {
        this->layerSpecificOptimizer = layerSpecificOptimizer;
        /* FIXME: need to save network
         *        get all stamps and replace the implementation optimizer with layerSpecificOptimizer->stamp()
         *          note will need to explicitly clear the existing one first since optimizer is sticky
         *        also during stamp() trainableWeightsBiasesLayers must check if there is a layerSpecificOptimizer and attach it if so.
         */
    }

    // Revert layer to the network default optimizer
    void clearLayerSpecificOptimizer() {
        this->layerSpecificOptimizer.clear();
        /* FIXME: need to save network
         *        get all stamps and replace the implementation optimizer with network->getOptimizer()
         *          note will need to explicitly clear the existing one first since optimizer is sticky
         *        also during stamp() trainableWeightsBiasesLayers must check if there is a layerSpecificOptimizer and attach it if so.
         */
    }

   protected:
    Tensor weights;
    Optional<Tensor> biases;
    Optional<Tensor> weightsGradient;
    Optional<Tensor> biasesGradient;
    Optional<Optimizer *> layerSpecificOptimizer;

    Optional<std::string> weightsFile;
    Optional<std::string> biasesFile;
};

}  // namespace Thor
