#include "DeepLearning/Api/Executors/LocalExecutor.h"
#include "DeepLearning/Api/Executors/Executor.h"

using namespace Thor;

Executor LocalExecutor::Builder::build() {
    assert(_network.isPresent());
    assert(_loader.isPresent());
    assert(_hyperparameterController.isPresent());

    if (_visualizers.isEmpty()) {
        _visualizers = vector<Visualizer>();
        _visualizers.get().push_back(ConsoleVisualizer::Builder().build());
    }

    LocalExecutor *localExecutor = new LocalExecutor();
    localExecutor->network = _network;
    localExecutor->loader = _loader;
    localExecutor->hyperparameterController = _hyperparameterController;
    localExecutor->visualizers = _visualizers;
    localExecutor->initialized = true;
    return Executor(localExecutor);
}
