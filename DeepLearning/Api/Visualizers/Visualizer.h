#pragma once

#include "DeepLearning/Api/Visualizers/VisualizerBase.h"

#include <assert.h>
#include <memory>

namespace Thor {

using std::shared_ptr;

class Visualizer {
   public:
    Visualizer() {}
    Visualizer(VisualizerBase *visualizerBase);

    virtual ~Visualizer() {}

    Visualizer *getVisualizer();

   private:
    shared_ptr<VisualizerBase> visualizer;
};

}  // namespace Thor
