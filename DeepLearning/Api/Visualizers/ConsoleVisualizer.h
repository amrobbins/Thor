#pragma once

#include "DeepLearning/Api/Visualizers/Visualizer.h"

namespace Thor {

class Visualizer;

class ConsoleVisualizer : public Visualizer {
   public:
    class Builder;
    ConsoleVisualizer() : initialized(false) {}

    virtual ~ConsoleVisualizer() {}

   private:
    bool initialized;
};

class ConsoleVisualizer::Builder {
   public:
    virtual Visualizer build();
};

}  // namespace Thor
