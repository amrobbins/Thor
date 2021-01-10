#include "DeepLearning/Api/Executors/LocalExecutor.h"
#include "DeepLearning/Api/Executors/Executor.h"

using namespace Thor;

LocalExecutor LocalExecutor::Builder::build() {
    assert(_network.isPresent());
    assert(_loader);
    // assert(_hyperparameterController.isPresent());

    if (_visualizers.isEmpty()) {
        _visualizers = vector<Visualizer>();
        _visualizers.get().push_back(ConsoleVisualizer::Builder().build());
    }

    LocalExecutor localExecutor;
    localExecutor.network = _network;
    localExecutor.loader = _loader;
    // localExecutor.hyperparameterController = _hyperparameterController;
    localExecutor.visualizers = _visualizers;
    localExecutor.initialized = true;
    return localExecutor;
}
