#pragma once

#include "DeepLearning/Implementation/Layers/Layer.h"

#include <vector>

class LayerTestHelper {
   public:
    static void connectNetwork(std::vector<std::shared_ptr<ThorImplementation::Layer>> &layers) {
        for (unsigned int i = 0; i < layers.size() - 1; ++i) {
            layers[i]->connectToNextLayer(layers[i + 1].get());
        }
    }

    static void initializeNetwork(std::vector<std::shared_ptr<ThorImplementation::Layer>> &layers) {
        for (unsigned int i = 0; i < layers.size(); ++i) {
            layers[i]->parentCompile();
            layers[i]->compile();
        }
        for (unsigned int i = 0; i < layers.size(); ++i) {
            layers[i]->initialize();
            layers[i]->parentInitialize();
        }
    }

    static void connectAndInitializeNetwork(std::vector<std::shared_ptr<ThorImplementation::Layer>> &layers) {
        connectNetwork(layers);
        initializeNetwork(layers);
    }

    static void tearDownNetwork(std::vector<std::shared_ptr<ThorImplementation::Layer>> &layers) {
        for (unsigned int i = 0; i < layers.size(); ++i) {
            layers[i]->cleanup();
            layers[i]->parentCleanup();
        }
        layers.clear();
    }

    static void connectTwoLayers(std::shared_ptr<ThorImplementation::Layer> firstLayer,
                                 std::shared_ptr<ThorImplementation::Layer> secondLayer,
                                 int driverConnectionType = 0,
                                 int loaderConnectionType = 0) {
        firstLayer->connectToNextLayer(secondLayer.get(), driverConnectionType, loaderConnectionType);
    }

    static void initializeLayer(ThorImplementation::Layer *layer) {
        layer->parentCompile();
        layer->compile();
        layer->parentInitialize();
        layer->initialize();
    }
};
