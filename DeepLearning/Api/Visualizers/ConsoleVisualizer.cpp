#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"

using namespace Thor;

Visualizer ConsoleVisualizer::Builder::build() {
    ConsoleVisualizer consoleVisualizer;
    consoleVisualizer.initialized = true;
    return consoleVisualizer;
}
