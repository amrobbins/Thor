#include "DeepLearning/Api/Visualizers/ConsoleVisualizer.h"
#include "DeepLearning/Api/Visualizers/Visualizer.h"

using namespace Thor;

Visualizer ConsoleVisualizer::Builder::build() {
    ConsoleVisualizer *consoleVisualizer = new ConsoleVisualizer();
    consoleVisualizer->initialized = true;
    return Visualizer(consoleVisualizer);
}
