#pragma once

#include "Thor.h"

#include <vector>

using std::vector;

class LayerTestHelper {
   public:
    static void connectNetwork(vector<Layer *> &layers) {
        for (unsigned int i = 0; i < layers.size() - 1; ++i) {
            layers[i]->connectToNextLayer(layers[i + 1]);
            layers[i]->postConnectToNextLayer();
        }
    }

    static void initializeNetwork(vector<Layer *> &layers) {
        for (unsigned int i = 0; i < layers.size(); ++i) {
            layers[i]->parentCompile();
            layers[i]->compile();
        }
        for (unsigned int i = 0; i < layers.size(); ++i) {
            layers[i]->parentInitialize();
            layers[i]->initialize();
        }
    }

    static void connectAndInitializeNetwork(vector<Layer *> &layers) {
        connectNetwork(layers);
        initializeNetwork(layers);
    }

    static void tearDownNetwork(vector<Layer *> &layers) {
        for (unsigned int i = 0; i < layers.size(); ++i) {
            layers[i]->cleanup();
            layers[i]->parentCleanup();
            delete layers[i];
        }
        layers.clear();
    }

    static void connectTwoLayers(Layer *firstLayer, Layer *secondLayer, int connectionType = 0) {
        firstLayer->connectToNextLayer(secondLayer, connectionType);
        firstLayer->postConnectToNextLayer();
    }

    static void initializeLayer(Layer *layer) {
        layer->parentCompile();
        layer->compile();
        layer->parentInitialize();
        layer->initialize();
    }
};
