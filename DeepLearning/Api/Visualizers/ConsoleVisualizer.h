#pragma once

#include "DeepLearning/Api/Visualizers/VisualizerBase.h"

namespace Thor {

class Visualizer;

class ConsoleVisualizer : public VisualizerBase {
   public:
    class Builder;
    ConsoleVisualizer() : initialized(false) {}

    virtual ~ConsoleVisualizer();

   private:
    bool initialized;
};

class ConsoleVisualizer::Builder {
   public:
    virtual Visualizer build();
};

}  // namespace Thor
