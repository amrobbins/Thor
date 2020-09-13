#include "DeepLearning/Api/Layers/Layer.h"

#include <assert.h>

using namespace Thor;

Layer::Layer(LayerBase *layerBase) { layer = shared_ptr<LayerBase>(layerBase); }

bool Layer::operator==(const Layer &other) const { return *layer == *(other.layer); }

bool Layer::operator!=(const Layer &other) const { return *layer != *(other.layer); }

bool Layer::operator<(const Layer &other) const { return *layer < *(other.layer); }

bool Layer::operator>(const Layer &other) const { return *layer > *(other.layer); }
